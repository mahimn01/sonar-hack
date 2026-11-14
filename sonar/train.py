"""train.py — train the gesture model, measure it honestly, export it for the laptop.

Plain words, the steps this script runs:

  1. load the examples (recorded sessions and/or ready-made .npz files, e.g.
     synthetic data from datagen/),
  2. split them BY SESSION: whole sessions are held out for validation, so the
     score answers "does it work on a recording it has never seen?" (with
     --val-person: "... on a person it has never seen?"). Synthetic examples
     are only ever used for training,
  3. train the small CNN with augmentation (class-balanced batches),
  4. print accuracy, per-class precision/recall and the confusion matrix on the
     held-out sessions, both raw and with the engine's confidence threshold,
  5. calibrate the confidence (temperature scaling) so "0.9" means roughly 90 %,
  6. fit the two weights of the physics classifier from the recorded swipes
     (which way round the speakers / microphones are on this laptop),
  7. export `gesture.onnx` + `gesture.json` and check the ONNX file gives the
     same answers as PyTorch.

How to run:
    python -m sonar.train --data data/ --out models/
    python -m sonar.train --data data/ --dataset datagen/out/synth.npz --val-person alice
Runs on CUDA (RTX 4090), Apple MPS or CPU, chosen automatically.

What it changes: writes gesture.onnx, gesture.json, gesture.pt, metrics.json and
confusion.txt into --out. Reads the data only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import GESTURE_LABELS, SonarConfig
from .dataset import Augmenter, build_dataset, center_crop, merge
from .gestures import heuristic_classify


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def remap_labels(data: dict[str, np.ndarray], keep: list[str]) -> dict[str, np.ndarray]:
    """Keep only some gesture classes; examples of the others become "none" (the model learns to ignore them)."""
    old = [str(x) for x in data["labels"]]
    new = ["none"] + [name for name in old if name != "none" and name in keep]
    lut = np.array([new.index(name) if name in new else 0 for name in old])
    out = dict(data)
    out["y"] = lut[data["y"]]
    out["labels"] = np.array(new)
    return out


def split_by_group(
    data: dict[str, np.ndarray], val_person: str | None, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, str]:
    """(train indices, validation indices, description). Never splits inside a session if it can avoid it."""
    n = len(data["y"])
    real = data["source"] != "synthetic"
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    if val_person:
        val = real & (data["person"] == val_person)
        if not val.any():
            raise ValueError(f"no examples of person {val_person!r}; people in the data: {sorted(set(data['person']))}")
        return idx[~val], idx[val], f"held-out person {val_person!r}"
    pool = real if real.any() else np.ones(n, dtype=bool)          # only synthetic data: validate on synthetic groups
    groups = sorted(set(data["group"][pool]))
    if len(groups) >= 3:
        order = list(rng.permutation(groups))
        target = val_fraction * pool.sum()
        chosen: list[str] = []
        for g in order[:-1]:                                       # always leave at least one group for training
            if np.isin(data["group"], chosen).sum() >= target:
                break
            chosen.append(g)
        val = pool & np.isin(data["group"], chosen)
        return idx[~val], idx[val], f"held-out sessions {chosen}"
    # one or two sessions: we have to split inside them. Neighbouring trials of one session look
    # alike, so this score is too optimistic.
    val = np.zeros(n, dtype=bool)
    for c in np.unique(data["y"]):
        members = idx[pool & (data["y"] == c)]
        take = rng.permutation(members)[:max(1, int(round(val_fraction * len(members))))] if len(members) > 1 else []
        val[take] = True
    return idx[~val], idx[val], "random split INSIDE sessions (optimistic: record at least 3 sessions)"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def confusion_matrix(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=int)
    for a, b in zip(y, pred):
        m[a, b] += 1
    return m


def report(y: np.ndarray, pred: np.ndarray, labels: list[str]) -> tuple[dict[str, Any], str]:
    """Accuracy, per-class precision/recall/F1 and a printable confusion matrix (rows = truth)."""
    k = len(labels)
    m = confusion_matrix(y, pred, k)
    per = {}
    for i, name in enumerate(labels):
        tp = m[i, i]
        prec = tp / max(m[:, i].sum(), 1)
        rec = tp / max(m[i].sum(), 1)
        per[name] = {"precision": round(float(prec), 3), "recall": round(float(rec), 3),
                     "f1": round(float(2 * prec * rec / max(prec + rec, 1e-9)), 3), "n": int(m[i].sum())}
    present = [name for name in labels if per[name]["n"] > 0]
    metrics = {
        "accuracy": round(float((y == pred).mean()), 4) if len(y) else 0.0,
        "macro_f1": round(float(np.mean([per[name]["f1"] for name in present])), 4) if present else 0.0,
        "per_class": per,
        "confusion": m.tolist(),
    }
    w = max(len(s) for s in labels) + 2
    lines = [" " * w + "".join(f"{s[:10]:>12}" for s in labels) + "   <- predicted"]
    for i, name in enumerate(labels):
        lines.append(f"{name:<{w}}" + "".join(f"{v:>12}" for v in m[i]))
    return metrics, "\n".join(lines)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature that minimises the validation negative log-likelihood (makes confidences honest)."""
    if len(y) < 20:
        return 1.0
    best_t, best_nll = 1.0, np.inf
    # Only ever soften (T >= 1): on a small, perfectly classified validation set the best-fitting
    # temperature would sharpen every answer to 100 %, and the confidence threshold would stop working.
    for t in np.exp(np.linspace(0.0, np.log(5.0), 60)):
        p = softmax(logits / t)
        nll = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def fit_heuristic(x: np.ndarray, y: np.ndarray, labels: list[str], cfg: SonarConfig) -> dict[str, float]:
    """Sign and scale of the two direction cues of the physics classifier, from recorded swipes.

    For each cue: difference of its mean over left swipes and right swipes. If the
    difference is clear (more than 2 standard errors) the weight is set so that an
    average swipe scores +-1; if not, the cue carries no information on this
    laptop (e.g. both tones come out of the same driver) and gets weight 0.
    """
    out = {"w_speakers": cfg.heur_w_speakers, "w_mics": cfg.heur_w_mics, "n_swipes": 0}
    if "swipe_left" not in labels or "swipe_right" not in labels:
        return out
    li, ri = labels.index("swipe_left"), labels.index("swipe_right")
    feats = {li: [], ri: []}
    for xi, yi in zip(x, y):
        if int(yi) in feats:
            r = heuristic_classify(xi, cfg)
            feats[int(yi)].append((r.lateral_hz, r.lateral_mic_hz))
    if min(len(feats[li]), len(feats[ri])) < 4:
        return out
    left, right = np.array(feats[li]), np.array(feats[ri])
    out["n_swipes"] = len(left) + len(right)
    for j, (key, typical) in enumerate((("w_speakers", 15.0), ("w_mics", 5.0))):
        diff = left[:, j].mean() - right[:, j].mean()
        se = np.sqrt(left[:, j].var() / len(left) + right[:, j].var() / len(right)) + 1e-9
        out[key] = float(np.clip(2.0 * typical / diff, -4.0, 4.0)) if abs(diff) > 2.0 * se and abs(diff) > 1e-3 else 0.0
        out[key.replace("w_", "t_")] = float(diff / se)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    data: dict[str, np.ndarray],
    cfg: SonarConfig,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    epochs: int = 60,
    batch: int = 64,
    lr: float = 2e-3,
    device_name: str = "auto",
    seed: int = 0,
    width: int = 24,
    init: str | None = None,
    verbose: bool = True,
) -> tuple[Any, np.ndarray, list[dict[str, float]]]:
    """Returns (best model on CPU, validation logits, per-epoch history)."""
    import torch
    from torch import nn

    from .model import GestureNet, count_parameters, pick_device

    torch.manual_seed(seed)
    labels = [str(s) for s in data["labels"]]
    pad = int(data["pad"])
    device = pick_device(device_name)
    x_all = data["X"].astype(np.float32)
    y_all = data["y"]
    model = GestureNet(x_all.shape[1], len(labels), width=width).to(device)
    if init:
        state = torch.load(init, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
    aug = Augmenter(cfg, labels, pad=pad, seed=seed)
    rng = np.random.default_rng(seed)

    # Class-balanced sampling: every class is drawn equally often.
    counts = np.bincount(y_all[train_idx], minlength=len(labels)).astype(float)
    weights = 1.0 / np.maximum(counts[y_all[train_idx]], 1.0)
    weights /= weights.sum()
    steps = max(20, int(np.ceil(len(train_idx) / batch)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    x_val = torch.from_numpy(np.ascontiguousarray(center_crop(x_all[val_idx], cfg, pad))) if len(val_idx) else None
    y_val = y_all[val_idx]

    if verbose:
        print(f"device {device}, {count_parameters(model)} weights, {len(train_idx)} train / {len(val_idx)} validation examples")
        print("train class counts:", {name: int(c) for name, c in zip(labels, counts)})
    history: list[dict[str, float]] = []
    best = (-1.0, None, None)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for _ in range(steps):
            pick = rng.choice(train_idx, size=batch, p=weights)
            xb, yb = zip(*(aug(x_all[i], int(y_all[i])) for i in pick))
            xb_t = torch.from_numpy(np.stack(xb)).to(device)
            yb_t = torch.tensor(yb, dtype=torch.long, device=device)
            opt.zero_grad()
            loss = loss_fn(model(xb_t), yb_t)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        entry = {"epoch": epoch + 1, "loss": total / steps}
        if x_val is not None:
            model.eval()
            with torch.no_grad():
                logits = torch.cat([model(x_val[i:i + 256].to(device)).cpu() for i in range(0, len(x_val), 256)]).numpy()
            m, _ = report(y_val, logits.argmax(axis=1), labels)
            entry.update({"val_acc": m["accuracy"], "val_f1": m["macro_f1"]})
            score = m["macro_f1"] + 1e-3 * m["accuracy"] + 1e-6 * epoch          # later epochs win ties
            if score >= best[0]:
                best = (score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, logits)
        history.append(entry)
        if verbose and (epoch % 5 == 4 or epoch == 0 or epoch == epochs - 1):
            extra = f"  val acc {entry['val_acc']:.3f}  macro-F1 {entry['val_f1']:.3f}" if "val_acc" in entry else ""
            print(f"epoch {epoch + 1:3d}/{epochs}  loss {entry['loss']:.3f}{extra}  ({time.time() - t0:.0f} s)")
    if best[1] is not None:
        model.load_state_dict(best[1])
    model = model.cpu().eval()
    return model, (best[2] if best[2] is not None else np.zeros((0, len(labels)))), history


def export_onnx(model: Any, path: Path, in_shape: tuple[int, int, int]) -> None:
    """Write the model as ONNX with a free batch dimension (input "input", output "logits")."""
    import torch

    dummy = torch.zeros((1, *in_shape), dtype=torch.float32)
    kwargs: dict[str, Any] = dict(
        input_names=["input"], output_names=["logits"], opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)      # classic exporter: no extra packages
    except TypeError:
        torch.onnx.export(model, (dummy,), str(path), **kwargs)


def check_onnx(path: Path, model: Any, x: np.ndarray) -> tuple[float, float]:
    """(largest logit difference between ONNX and PyTorch, milliseconds per single inference on CPU)."""
    import onnxruntime as ort
    import torch

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"input": x})[0]
    one = x[:1]
    t0 = time.perf_counter()
    for _ in range(200):
        sess.run(None, {"input": one})
    return float(np.max(np.abs(ref - got))), (time.perf_counter() - t0) / 200 * 1e3


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sonar.train", description="Train and export the gesture model")
    ap.add_argument("--data", nargs="*", default=[], help="folders with recorded sessions")
    ap.add_argument("--dataset", nargs="*", default=[], help="ready-made .npz datasets (e.g. synthetic data)")
    ap.add_argument("--out", default="models", help="output folder")
    ap.add_argument("--name", default="gesture", help="file name stem of the exported model")
    ap.add_argument("--labels", default=",".join(g for g in GESTURE_LABELS if g != "none"),
                    help="gesture classes to recognise; others are trained as 'none'")
    ap.add_argument("--val-person", help="hold out every session of this person for validation")
    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="start from these weights (gesture.pt of an earlier run, e.g. synthetic pre-training)")
    ap.add_argument("--aux-none", action="store_true", help="use wind-up / hand-return bursts of gesture trials as 'none'")
    ap.add_argument("--min-confidence", type=float, default=0.6, help="the engine's threshold, for the 'as deployed' score")
    args = ap.parse_args(argv)

    try:
        parts = []
        if args.data:
            parts.append(build_dataset(args.data, aux_as_none=args.aux_none))
        parts += [load_npz(p) for p in args.dataset]
        if not parts:
            ap.error("give --data and/or --dataset")
        data = remap_labels(merge(parts) if len(parts) > 1 else parts[0], args.labels.split(","))
        cfg = SonarConfig.from_dict(json.loads(str(data["config"])))
        labels = [str(s) for s in data["labels"]]
        train_idx, val_idx, how = split_by_group(data, args.val_person, args.val_fraction, args.seed)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"{len(data['y'])} examples, classes {labels}")
    print(f"validation: {how}")
    if "INSIDE" in how:
        print("WARNING: fewer than 3 sessions, so validation examples come from the same sessions as training.")

    model, val_logits, history = train_model(
        data, cfg, train_idx, val_idx, epochs=args.epochs, batch=args.batch, lr=args.lr,
        device_name=args.device, seed=args.seed, width=args.width, init=args.init,
    )

    pad = int(data["pad"])
    y_val = data["y"][val_idx]
    metrics: dict[str, Any] = {"split": how, "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "history": history[-1]}
    temperature = 1.0
    if len(val_idx):
        m, table = report(y_val, val_logits.argmax(axis=1), labels)
        temperature = fit_temperature(val_logits, y_val)
        prob = softmax(val_logits / temperature)
        pred = np.where(prob.max(axis=1) >= args.min_confidence, prob.argmax(axis=1), 0)     # unsure -> "none"
        m_dep, table_dep = report(y_val, pred, labels)
        gestures = y_val != 0
        metrics.update({
            "validation": m,
            "as_deployed": m_dep,
            "gesture_recall_deployed": round(float((pred[gestures] == y_val[gestures]).mean()), 4) if gestures.any() else None,
            "false_triggers_deployed": round(float((pred[~gestures] != 0).mean()), 4) if (~gestures).any() else None,
        })
        print(f"\nvalidation accuracy {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f} (rows = truth)\n{table}")
        print(f"\nas deployed (confidence >= {args.min_confidence}, temperature {temperature:.2f}): "
              f"gestures recognised {metrics['gesture_recall_deployed']}, "
              f"non-gestures that would trigger {metrics['false_triggers_deployed']}\n{table_dep}")
        for name, v in m["per_class"].items():
            print(f"  {name:<12} precision {v['precision']:.2f}  recall {v['recall']:.2f}  n={v['n']}")
    else:
        table = table_dep = "(no validation data)"

    heur = fit_heuristic(center_crop(data["X"][train_idx].astype(np.float32), cfg, pad), data["y"][train_idx], labels, cfg)
    print(f"\nphysics-classifier weights from {heur['n_swipes']} recorded swipes: "
          f"speakers {heur['w_speakers']:+.2f}, microphones {heur['w_mics']:+.2f}  (0 = that cue is useless on this device)")

    # How strong were the real gestures? The engine ignores bursts far weaker than that.
    real_gestures = (data["y"] != 0) & (data["source"] != "synthetic")
    gesture_p10 = round(float(np.percentile(data["peak_db"][real_gestures], 10)), 1) if real_gestures.any() else None
    if gesture_p10 is not None:
        print(f"gesture strength: 10 % of the recorded gestures were below {gesture_p10} dB (engine will ignore bursts 6 dB weaker)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / f"{args.name}.onnx"
    import torch

    torch.save({"model": model.state_dict(), "labels": labels, "width": args.width}, out / f"{args.name}.pt")
    in_shape = (cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames)
    export_onnx(model, onnx_path, in_shape)
    probe_idx = val_idx if len(val_idx) else train_idx
    probe = np.ascontiguousarray(center_crop(data["X"][probe_idx[:64]].astype(np.float32), cfg, pad))
    diff, ms = check_onnx(onnx_path, model, probe)
    metrics.update({"onnx_max_logit_diff": diff, "onnx_ms_per_inference": round(ms, 3)})
    meta = {
        "labels": labels,
        "temperature": temperature,
        "input_shape": list(in_shape),
        "config": cfg.to_dict(),
        "heuristic": heur,
        **({"gesture_peak_db_p10": gesture_p10} if gesture_p10 is not None else {}),
        "metrics": {k: v for k, v in metrics.items() if k != "history"},
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
    }
    (out / f"{args.name}.json").write_text(json.dumps(meta, indent=1))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out / "confusion.txt").write_text(f"split: {how}\n\nraw:\n{table}\n\nas deployed:\n{table_dep}\n")
    print(f"\nwrote {onnx_path} (+ .json, .pt), ONNX vs PyTorch max logit difference {diff:.2e}, {ms:.2f} ms per inference on CPU")
    if diff > 1e-3:
        print("WARNING: the exported ONNX model disagrees with PyTorch; do not deploy it.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

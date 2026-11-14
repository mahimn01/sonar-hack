"""train_check.py — does a model trained ONLY on synthetic data learn something that transfers?

What it does
------------
1. Loads a synthetic set made by `generate.py` (dataset.npz + manifest.jsonl).
2. Trains a small CNN on it, holding out whole clips (= unseen virtual laptops
   and unseen virtual people) for validation.
   It also runs the core's training-free physics rules on those same held-out
   clips, so rules and model can be compared like for like.
3. Optionally (--core-sim N) tests the same model on clips made by the core
   package's OWN, independently written simulator (`sonar/sim.py`). The model has
   never seen that simulator, so this is the closest thing to a "real laptop"
   test we can run before real recordings exist. It is still a simulation:
   treat the number as an upper bound, not as the demo accuracy.
4. Optionally (--real folder) tests on real examples in the same dataset.npz
   format (what `python -m sonar.dataset` writes), once they exist.
5. Saves the weights (pretrained.pt) so they can be fine-tuned on real clips.

Why
---
To check, tonight, that the synthetic data is learnable and not full of
simulator artefacts, and to hand the team a pre-trained starting point.

How to run (from the repo root)
-------------------------------
    cd ~/Documents/Dev/sonar-hack
    datagen/.venv/bin/python -m datagen.train_check --data datagen/out/synth_v5 --core-sim 40

What it changes: writes pretrained.pt and train_check.json into the --data
folder. Nothing else. No audio devices.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import adapter as ad
from .generate import GESTURE4, LEAD_IN, STORY, label_segments


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_set(folder: str | Path) -> tuple:
    """Returns (X float32 [N,C,F,T] in -1..1, y [N], clip id per example [N], label names, per-example info)."""
    folder = Path(folder)
    z = np.load(folder / "dataset.npz", allow_pickle=False)  # the core's dataset format (sonar/dataset.py)
    pad = int(z["pad"])
    x = z["X"].astype(np.float32)
    x = x[..., pad : x.shape[-1] - pad] if pad else x  # the un-shifted window = exactly what the engine classifies
    y, labels = z["y"], [str(s) for s in z["labels"]]
    groups = np.arange(len(y))
    info = [{} for _ in range(len(y))]
    manifest = folder / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            if row.get("index") is not None:
                groups[row["index"]] = row["seed"]
                info[row["index"]] = {"snr": row.get("echo_snr_db"), "speed": row.get("scene_meta", {}).get("peak_speed"),
                                      "is_stroke": row.get("label") != "none"}
    return x, y, groups, labels, info


def recall_breakdown(model: nn.Module, x: np.ndarray, y: np.ndarray, info: list, device: torch.device, none_idx: int) -> dict:
    """How often is a true gesture recognised, split by how loud and how fast it was?"""
    model.eval()
    with torch.no_grad():
        pred = np.concatenate([model(torch.from_numpy(x[i : i + 512]).to(device).float()).argmax(1).cpu().numpy()
                               for i in range(0, len(x), 512)])
    out = {}
    is_g = y != none_idx
    snr = np.array([(d.get("snr") if d.get("snr") is not None else np.nan) for d in info], dtype=float)
    speed = np.array([(d.get("speed") if d.get("speed") is not None else np.nan) for d in info], dtype=float)
    print("\nRecall of true gestures on held-out synthetic clips, by echo loudness and hand speed")
    for name, val, edges in (("echo SNR (dB)", snr, [-99, 12, 20, 30, 99]), ("peak hand speed (m/s)", speed, [0, 0.6, 1.0, 1.6, 9])):
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = is_g & (val >= lo) & (val < hi)
            if sel.sum() >= 5:
                r = float((pred[sel] == y[sel]).mean())
                out[f"{name} {lo}..{hi}"] = round(r, 3)
                print(f"  {name:22s} {lo:>5} .. {hi:<5}  n={int(sel.sum()):4d}  recognised {100 * r:5.1f}%")
    return out


def mirror(x: torch.Tensor, y: torch.Tensor, labels: list, n_tones: int = 2, n_mics: int = 2) -> tuple:
    """Left-right mirror image of a clip: swap the two speaker tones AND the two mics, swap the swipe labels.

    Physically this is the same scene seen in a mirror, so it doubles the swipe data for free
    and stops the model from preferring one direction. Channel layout (core): power channels
    c = tone * n_mics + mic, then (optionally) one signed "lateral" channel, which changes sign.
    """
    n_power = n_tones * n_mics
    c = torch.arange(n_power).reshape(n_tones, n_mics).flip(0).flip(1).reshape(-1)
    xm = x.clone()
    xm[:, :n_power] = x[:, c]
    if x.shape[1] > n_power:
        xm[:, n_power:] = -x[:, n_power:]
    left, right = labels.index("swipe_left"), labels.index("swipe_right")
    y2 = y.clone()
    y2[y == left], y2[y == right] = right, left
    return xm, y2


def augment(x: torch.Tensor, y: torch.Tensor, labels: list, gen: torch.Generator) -> tuple:
    """Cheap label-safe changes: mirror, small time shift, brightness, a little noise."""
    flip = torch.rand(len(x), generator=gen, device="cpu").to(x.device) < 0.5
    xm, ym = mirror(x, y, labels)
    x = torch.where(flip[:, None, None, None], xm, x)
    y = torch.where(flip, ym, y)
    shift = int(torch.randint(-5, 6, (1,), generator=gen))  # +-0.1 s: the segmenter never cuts identically
    if shift:
        x = torch.roll(x, shift, dims=3)
        if shift > 0:
            x[..., :shift] = 0
        else:
            x[..., shift:] = 0
    gain = 0.8 + 0.4 * torch.rand(len(x), 1, 1, 1, generator=gen, device="cpu").to(x.device)
    n_power = min(x.shape[1], 4)
    x = x.clone()
    x[:, :n_power] = (x[:, :n_power] * gain + 0.03 * torch.randn(x[:, :n_power].shape, generator=gen, device="cpu").to(x.device)).clamp(0, 1)
    return x, y


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """Four conv blocks, then a head that keeps the ORDER of events in time.

    The window always ends just after the burst being judged. A stroke at the
    very end is a gesture; the same stroke followed by the hand dropping back is
    the "hand coming back" burst and must be ignored. A head that averages over
    time cannot tell those apart, so we only average over Doppler frequency and
    keep the 10 time steps. ~90 k weights, far below 1 ms per window on a laptop CPU.
    """

    def __init__(self, n_channels: int = 4, n_classes: int = 5, width: int = 24, n_frames: int = 80) -> None:
        super().__init__()
        layers, c_in = [], n_channels
        for k, c_out in enumerate([width, width * 2, width * 2, width * 4]):
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            layers += [nn.MaxPool2d(2)] if k < 3 else []
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.25), nn.Linear(c_in * (n_frames // 8), n_classes))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Feature vector of a window (used for prototype / few-shot calibration)."""
        return self.features(x).mean(dim=2).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device, n_classes: int) -> tuple:
    """Returns (accuracy, confusion matrix rows=true cols=predicted)."""
    model.eval()
    conf = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(0, len(x), 512):
        xb = torch.from_numpy(x[i : i + 512]).to(device).float()
        pred = model(xb).argmax(1).cpu().numpy()
        for t, p in zip(y[i : i + 512], pred):
            conf[t, p] += 1
    return float(np.trace(conf) / max(1, conf.sum())), conf


def show(conf: np.ndarray, labels: list, title: str) -> dict:
    print(f"\n{title}   (rows = truth, columns = prediction)")
    print(" " * 13 + "".join(f"{name[:10]:>11s}" for name in labels))
    per_class = {}
    for i, name in enumerate(labels):
        total = conf[i].sum()
        per_class[name] = None if total == 0 else round(float(conf[i, i] / total), 3)
        print(f"{name:13s}" + "".join(f"{v:11d}" for v in conf[i]) + (f"   {100 * conf[i, i] / total:5.1f}%" if total else ""))
    return per_class


# --------------------------------------------------------------------------
# test clips from the core package's own simulator (never seen in training)
# --------------------------------------------------------------------------
def core_sim_clip(job: tuple) -> list:
    """One scripted gesture rendered by sonar/sim.py -> labelled model inputs."""
    label, seed = job
    core = ad.load_core()
    sim = core.module("sim")
    if sim is None or not core.available:
        return []
    rng = np.random.default_rng(seed)
    try:  # the core simulator is still changing: a failure here must not stop the run
        speed = rng.uniform(0.7, 1.5)
        t0 = LEAD_IN + rng.uniform(0.1, 0.5)
        way = sim.gesture_waypoints(label, t0, speed)
        path = sim.WaypointPath([w[0] for w in way], [w[1] for w in way])
        scene = sim.hand_reflector(path, echo_db=rng.uniform(-36.0, -22.0))
        scene += sim.person_reflectors(distance_m=rng.uniform(0.5, 0.9), seed=int(seed))
        es = sim.EchoSimulator(core.cfg, seed=int(seed), clock_ppm=float(rng.uniform(-20, 20)),
                               ambient_dbfs=None if rng.random() < 0.5 else -40.0,
                               direct_gain_db=float(rng.uniform(-73.0, -60.0)))
        a, b = sim.gesture_stroke_window(label, t0, speed)
        rx = es.render(int(core.fs * (max(b, LEAD_IN + STORY) + 1.0)), scene)
        stream = ad.stream_features(core, rx)
    except Exception as exc:
        return [("error", f"{type(exc).__name__}: {exc}")]
    return [(e["label"], e["window"]) for e in label_segments(stream, [{"label": label, "t_start": a, "t_end": b}])]


def core_sim_set(n_per_class: int, labels: list, workers: int) -> tuple:
    jobs = [(g, 900_000 + 1000 * i + k) for i, g in enumerate(GESTURE4) for k in range(n_per_class)]
    xs, ys, errors = [], [], []
    with ProcessPoolExecutor(workers) as pool:
        for rows in pool.map(core_sim_clip, jobs, chunksize=2):
            for lab, win in rows:
                if lab == "error":
                    errors.append(win)
                    continue
                xs.append(np.asarray(win, dtype=np.float32))
                ys.append(labels.index(lab))
    if errors:
        print(f"core simulator raised on {len(errors)} of {len(jobs)} clips, e.g. {errors[0]}")
    if not xs:
        return None, None
    return np.stack(xs), np.array(ys)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Train on synthetic data and check that it transfers (see the file header).")
    ap.add_argument("--data", required=True, help="folder made by datagen.generate (several: comma-separated)")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--core-sim", type=int, default=0, help="test on N clips per gesture from sonar/sim.py")
    ap.add_argument("--real", default="", help="folder holding a dataset.npz of REAL examples (as written by python -m sonar.dataset), to test on")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    folders = [f for f in args.data.split(",") if f]
    parts = [load_set(f) for f in folders]
    labels = parts[0][3]
    info = [d for p in parts for d in p[4]]
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    groups = np.concatenate([p[2] + 10_000_000_000 * i for i, p in enumerate(parts)])

    # hold out 20% of the CLIPS (not of the examples): validation = virtual laptops/people never trained on
    uniq = np.unique(groups)
    rng = np.random.default_rng(args.seed)
    val_groups = set(rng.choice(uniq, size=max(1, len(uniq) // 5), replace=False).tolist())
    is_val = np.array([g in val_groups for g in groups])
    x_tr, y_tr, x_va, y_va = x[~is_val], y[~is_val], x[is_val], y[is_val]
    print(f"{len(x_tr)} training / {len(x_va)} validation examples, classes: {labels}")
    print("training class counts:", {labels[i]: int((y_tr == i).sum()) for i in range(len(labels))})

    device = pick_device()
    model = SmallCNN(x.shape[1], len(labels)).to(device)
    counts = np.bincount(y_tr, minlength=len(labels)).astype(np.float64)
    weight = torch.tensor((counts.sum() / (len(labels) * np.maximum(counts, 1))) ** 0.5, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    steps = args.epochs * int(np.ceil(len(x_tr) / 128))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-3, total_steps=steps)
    gen = torch.Generator().manual_seed(args.seed)
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(xt), generator=gen)
        total = 0.0
        for i in range(0, len(order), 128):
            idx = order[i : i + 128]
            xb, yb = augment(xt[idx].to(device).float(), yt[idx].to(device), labels, gen)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(idx)
        acc, _ = evaluate(model, x_va, y_va, device, len(labels))
        print(f"epoch {epoch + 1:2d}  loss {total / len(xt):.3f}  held-out synthetic accuracy {100 * acc:.1f}%")
    print(f"trained in {time.time() - t0:.0f} s on {device}")

    report = {"labels": labels, "n_train": int(len(x_tr)), "n_val": int(len(x_va))}
    acc, conf = evaluate(model, x_va, y_va, device, len(labels))
    report["synthetic_holdout"] = {"accuracy": round(acc, 3), "per_class": show(conf, labels, f"Held-out synthetic clips: {100 * acc:.1f}%")}
    info_va = [d for d, v in zip(info, is_val) if v]
    report["gesture_recall_by_condition"] = recall_breakdown(model, x_va, y_va, info_va, device, labels.index("none"))

    # the same held-out windows through the core's training-free physics rules, for comparison
    core = ad.load_core()
    g = core.module("gestures") if core.available else None
    if g is not None and hasattr(g, "heuristic_classify"):
        try:
            conf = np.zeros((len(labels), len(labels)), dtype=int)
            for win, truth in zip(x_va, y_va):
                conf[truth, labels.index(g.heuristic_classify(win, core.cfg).label)] += 1
            acc = float(np.trace(conf) / max(1, conf.sum()))
            report["core_rules_on_synthetic_holdout"] = {
                "accuracy": round(acc, 3),
                "per_class": show(conf, labels, f"Core physics rules on the same held-out synthetic clips: {100 * acc:.1f}%")}
        except Exception as exc:
            print("core physics rules could not be run:", type(exc).__name__, exc)

    if args.core_sim > 0:
        x_cs, y_cs = core_sim_set(args.core_sim, labels, args.workers)
        if x_cs is None:
            print("\ncore simulator not importable: skipped the cross-simulator test")
        else:
            acc, conf = evaluate(model, x_cs, y_cs, device, len(labels))
            report["core_simulator"] = {"accuracy": round(acc, 3), "n": int(len(y_cs)),
                                        "per_class": show(conf, labels, f"Clips from the core's own simulator (never trained on): {100 * acc:.1f}%")}
    if args.real:
        x_re, y_re, _, _, _ = load_set(args.real)
        acc, conf = evaluate(model, x_re, y_re, device, len(labels))
        report["real"] = {"accuracy": round(acc, 3), "n": int(len(y_re)),
                          "per_class": show(conf, labels, f"REAL examples: {100 * acc:.1f}%")}

    out = Path(folders[0])
    torch.save({"state_dict": model.state_dict(), "labels": labels, "n_channels": int(x.shape[1])}, out / "pretrained.pt")
    (out / "train_check.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out / 'pretrained.pt'} and {out / 'train_check.json'}")


if __name__ == "__main__":
    main()

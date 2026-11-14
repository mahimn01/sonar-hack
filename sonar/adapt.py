"""adapt.py — tune a trained gesture model to one room / table / person from a short calibration.

Plain words: a model trained in one place meets slightly different echoes in another (another
table, another distance, another person's hand). DATA_STRATEGY.md 3.5 planned a 30-second
"do each gesture 3 times" calibration at the demo table. This file does that calibration, in
two parts that can be switched on and off separately:

  1. Normalisation statistics ("BatchNorm"). Inside the network every layer rescales its
     numbers with an average and a spread that were measured on the TRAINING pictures. Here
     they are measured again on the room's own pictures and blended with the trained ones
     (the fewer room pictures, the more the trained values count). No weights are changed, no
     labels are strictly needed, so any recording from the room helps. This is the usual
     "test-time adaptation" of BatchNorm statistics.
  2. Prototypes. The three examples of each gesture are averaged into one "prototype" in the
     network's last-layer space, and a burst that looks like a prototype gets a bonus for that
     gesture on top of the trained answer. (Off unless asked for: see the measured numbers.)

It also lowers the engine's "too faint to be a gesture" gate if the room's gestures are fainter
than the training ones. The result is written as a normal model folder (gesture.onnx +
gesture.json + gesture.pt), so the engine needs no change: start it with
`--model <that folder>/gesture.onnx` to use the adapted model, or without it to use the original.

`eval` measures the effect offline, by replaying recorded sessions: for every person (or session)
the model was NOT trained on, the first few gestures play the role of the calibration and the
rest of that person's pictures are scored with and without the adaptation.

How to run:
    python -m sonar.camrecord --person me --script calibrate        (or sonar.record ... --reps 3)
    python -m sonar.adapt fit --model models/gesture.onnx --calib data/<that session> --out models/room
    python -m sonar.engine --model models/room/gesture.onnx
    python -m sonar.adapt eval --model train_out/runs/<run>/sweep/w24_s0/gesture.onnx --data data --people guest3,guest7
Needs PyTorch and the model's gesture.pt (training machine or MacBook; on the HP: pip install torch).
Everything is computed in batches on --device (CUDA if there is one): the pictures are moved
there once, and there is no per-picture Python.

What it changes: `fit` writes the model folder you name with --out. `eval` writes only the
--json report if asked. Sessions and the original model are only read.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .batch_judge import exact_gpu_maths, pick_device
from .config import SonarConfig
from .dataset import build_dataset, center_crop, merge

VARIANTS = ("base", "bn", "proto", "bn+proto")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_model(model: str | Path, device: Any) -> tuple[Any, dict[str, Any], SonarConfig, list[str]]:
    """(network on `device`, gesture.json contents, the signal config it was trained with, class names)."""
    import torch

    from .model import GestureNet

    path = Path(model)
    meta_path, pt_path = path.with_suffix(".json"), path.with_suffix(".pt")
    if not meta_path.exists() or not pt_path.exists():
        raise ValueError(f"{path}: adaptation needs {meta_path.name} and {pt_path.name} next to the model "
                         "(they are written by sonar.train and copied to models-release/)")
    meta = json.loads(meta_path.read_text())
    cfg = SonarConfig.from_dict(meta["config"])
    labels = [str(s) for s in meta["labels"]]
    state = torch.load(str(pt_path), map_location="cpu")
    width = int(state.get("width", 24)) if isinstance(state, dict) and "model" in state else 24
    net = GestureNet(int(meta["input_shape"][0]), len(labels), width=width)
    net.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    return net.to(device).eval(), meta, cfg, labels


def load_pictures(data: list[str], datasets: list[str], labels: list[str], cfg: SonarConfig) -> dict[str, np.ndarray]:
    """Sessions and/or ready-made .npz datasets as one set of arrays, relabelled to the model's classes."""
    from .train import load_npz, remap_labels

    parts = [build_dataset(data, verbose=False)] if data else []
    parts += [load_npz(p) for p in datasets]
    if not parts:
        raise ValueError("give --data and/or --dataset")
    d = remap_labels(merge(parts) if len(parts) > 1 else parts[0], [name for name in labels if name != "none"])
    if [str(s) for s in d["labels"]] != labels:
        raise ValueError(f"the data has the classes {[str(s) for s in d['labels']]}, the model {labels}")
    want = (cfg.n_feat_channels, cfg.doppler_bins)
    if tuple(d["X"].shape[1:3]) != want:
        raise ValueError(f"the pictures have shape {tuple(d['X'].shape[1:3])}, the model expects {want} "
                         "(another signal profile or microphone count)")
    return d


# ---------------------------------------------------------------------------
# The two adaptations (all tensors stay on the model's device)
# ---------------------------------------------------------------------------
def balanced_weights(y: Any, n_classes: int) -> Any:
    """Per-picture weights that let every class present count the same (training drew its batches that way)."""
    import torch

    counts = torch.bincount(y, minlength=n_classes).clamp_min(1).to(torch.float64)
    w = 1.0 / counts[y]
    return w / w.sum()


def adapt_batchnorm(net: Any, x: Any, w: Any | None = None, prior: float = 16.0) -> list[dict[str, float]]:
    """Blend the room's own statistics into every BatchNorm layer of `net`, in place.

    x [N, C, F, T] room pictures on the model's device; w [N] weights (None: all equal).
    prior: how many room pictures the trained statistics are worth. With N room pictures the
    room's share is N_eff / (N_eff + prior), so three pictures barely move anything and a few
    hundred take over almost completely.
    Layer by layer in one pass: each layer is measured behind the layers already adapted.
    """
    import torch

    n = x.shape[0]
    if n == 0:
        return []
    w = torch.full((n,), 1.0 / n, device=x.device, dtype=torch.float64) if w is None else (w / w.sum()).to(torch.float64)
    share = float(1.0 / (w * w).sum())                             # effective number of pictures
    share = share / (share + prior)
    report = []
    with torch.no_grad(), exact_gpu_maths():
        h = x.float()
        for block in net.features:
            conv, bn, act, pool = block[0], block[1], block[2], block[3]
            z = conv(h)
            per = z.double().mean(dim=(2, 3))                                           # [N, channels]
            mean = (w[:, None] * per).sum(dim=0)
            var = (w[:, None] * ((z.double() - mean[None, :, None, None]) ** 2).mean(dim=(2, 3))).sum(dim=0)
            old_mean, old_var = bn.running_mean.double(), bn.running_var.double()
            shift = float(((mean - old_mean).abs() / old_var.sqrt().clamp_min(1e-6)).mean())
            bn.running_mean.copy_(((1.0 - share) * old_mean + share * mean).to(bn.running_mean.dtype))
            bn.running_var.copy_(((1.0 - share) * old_var + share * var).to(bn.running_var.dtype))
            report.append({"mean_shift_in_spreads": round(shift, 3), "spread_ratio": round(float((var / old_var.clamp_min(1e-12)).sqrt().mean()), 3)})
            h = pool(act(bn(z)))
    for row in report:
        row["room_share"] = round(share, 3)
    return report


def embed(net: Any, x: Any, chunk: int = 4096) -> Any:
    """The network's last-layer vectors [N, D] for pictures x [N, C, F, T]."""
    import torch

    with torch.no_grad(), exact_gpu_maths():
        return torch.cat([net.head[1](net.head[0](net.features(x[i:i + chunk].float()))) for i in range(0, len(x), chunk)])


def make_prototypes(net: Any, x: Any, y: Any, n_classes: int) -> tuple[Any, Any]:
    """(prototypes [P, D] of unit length, map [P, K] saying which class each one speaks for)."""
    import torch

    e = torch.nn.functional.normalize(embed(net, x).double(), dim=1)
    protos, rows = [], []
    for k in range(n_classes):
        sel = y == k
        if int(sel.sum()) == 0:
            continue
        protos.append(torch.nn.functional.normalize(e[sel].mean(dim=0), dim=0))
        row = torch.zeros(n_classes, dtype=torch.float64, device=x.device)
        row[k] = 1.0
        rows.append(row)
    if not protos:
        return torch.zeros((0, e.shape[1]), device=x.device), torch.zeros((0, n_classes), device=x.device)
    return torch.stack(protos).float(), torch.stack(rows).float()


def adapted_module(net: Any, protos: Any | None = None, proto_map: Any | None = None, beta: float = 0.0) -> Any:
    """The network with the prototype bonus built in: logits + beta * (likeness to each prototype). Exports to ONNX."""
    import torch
    from torch import nn

    class Adapted(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = net
            self.register_buffer("protos", protos if protos is not None else torch.zeros((0, 1)))
            self.register_buffer("proto_map", proto_map if proto_map is not None else torch.zeros((0, 1)))
            self.beta = float(beta)
            self.bonus = bool(beta) and protos is not None and protos.shape[0] > 0      # decided once, not inside the exported graph

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            e = self.net.head[1](self.net.head[0](self.net.features(x)))
            logits = self.net.head[3](self.net.head[2](e))
            if self.bonus:
                logits = logits + self.beta * (torch.nn.functional.normalize(e, dim=1) @ self.protos.t()) @ self.proto_map
            return logits

    return Adapted().eval()


# ---------------------------------------------------------------------------
# Scoring (the same numbers train.py prints)
# ---------------------------------------------------------------------------
def score(module: Any, x: Any, y: Any, temperature: float, min_confidence: float = 0.6, chunk: int = 4096) -> dict[str, Any]:
    """Accuracy, and "as deployed" at the engine's confidence threshold: gestures recognised / non-gestures that trigger."""
    import torch

    with torch.no_grad(), exact_gpu_maths():
        logits = torch.cat([module(x[i:i + chunk].float()) for i in range(0, len(x), chunk)])
        probs = torch.softmax(logits.double() / temperature, dim=1)
        conf, pred = probs.max(dim=1)
        deployed = torch.where(conf >= min_confidence, pred, torch.zeros_like(pred))
        gestures = y != 0
        out = {
            "n": int(len(y)), "n_gestures": int(gestures.sum()), "n_none": int((~gestures).sum()),
            "accuracy": float((pred == y).double().mean()) if len(y) else float("nan"),
            "recall": float((deployed[gestures] == y[gestures]).double().mean()) if bool(gestures.any()) else float("nan"),
            "false_triggers": float((deployed[~gestures] != 0).double().mean()) if bool((~gestures).any()) else float("nan"),
        }
    return out


def split_calibration(y: np.ndarray, shots: int, none_shots: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(calibration positions, test positions) of one person's pictures, in recording order.

    The first `shots` examples of every gesture and the first few "none" pictures are the calibration,
    as if the person had just done "each gesture 3 times" at the table; everything after is the test.
    """
    y = np.asarray(y)
    classes = [int(c) for c in np.unique(y) if c != 0]
    take: list[int] = []
    for c in classes:
        take += list(np.flatnonzero(y == c)[:shots])
    nones = np.flatnonzero(y == 0)
    m = min(len(nones) // 2, shots * max(len(classes), 1)) if none_shots is None else min(none_shots, len(nones))
    take += list(nones[:m])
    calib = np.array(sorted(take), dtype=np.int64)
    test = np.setdiff1d(np.arange(len(y)), calib)
    return calib, test


def evaluate(net: Any, x_all: Any, y_all: Any, units: np.ndarray, temperature: float, shots: int = 3, prior: float = 16.0,
             beta: float = 4.0, min_confidence: float = 0.6, min_test_gestures: int = 4, transductive: bool = False,
             ) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Score every variant on every unit (person or session). Returns (one row per unit, the averages).

    transductive: BatchNorm statistics from ALL of the unit's pictures (no labels used): what the engine
    could reach after running in the room for a while, instead of from the calibration alone.
    """
    import torch

    n_classes = int(y_all.max().item()) + 1 if len(y_all) else 1
    n_out = net.head[3].out_features
    rows: list[dict[str, Any]] = []
    for unit in sorted(set(units.tolist())):
        where = np.flatnonzero(units == unit)
        y_np = y_all[torch.from_numpy(where).to(y_all.device)].cpu().numpy()
        calib, test = split_calibration(y_np, shots)
        per_class = np.bincount(y_np[calib], minlength=max(n_classes, n_out))
        gesture_classes = [c for c in np.unique(y_np) if c != 0]
        if not gesture_classes or any(per_class[c] < shots for c in gesture_classes) or int((y_np[test] != 0).sum()) < min_test_gestures:
            rows.append({"unit": str(unit), "skipped": "too few gestures to calibrate with and still test on"})
            continue
        ci = torch.from_numpy(where[calib]).to(x_all.device)
        ti = torch.from_numpy(where[test]).to(x_all.device)
        xc, yc, xt, yt = x_all[ci], y_all[ci], x_all[ti], y_all[ti]
        row: dict[str, Any] = {"unit": str(unit), "calibration": int(len(calib)), "test": int(len(test))}
        tuned = copy.deepcopy(net)
        if transductive:
            adapt_batchnorm(tuned, x_all[torch.from_numpy(where).to(x_all.device)], None, prior)
        else:
            adapt_batchnorm(tuned, xc, balanced_weights(yc, n_out), prior)
        for name in VARIANTS:
            body = tuned if "bn" in name else net
            protos = pmap = None
            if "proto" in name:
                protos, pmap = make_prototypes(body, xc, yc, n_out)
            row[name] = score(adapted_module(body, protos, pmap, beta if "proto" in name else 0.0), xt, yt, temperature, min_confidence)
        rows.append(row)
    used = [r for r in rows if "skipped" not in r]
    mean: dict[str, dict[str, float]] = {}
    for name in VARIANTS:
        mean[name] = {k: float(np.nanmean([r[name][k] for r in used])) if used else float("nan") for k in ("accuracy", "recall", "false_triggers")}
        if used and name != "base":
            mean[name]["units_better"] = float(sum(r[name]["accuracy"] > r["base"]["accuracy"] + 1e-9 for r in used))
            mean[name]["units_worse"] = float(sum(r[name]["accuracy"] < r["base"]["accuracy"] - 1e-9 for r in used))
    return rows, mean


def print_table(rows: list[dict[str, Any]], mean: dict[str, dict[str, float]], by: str) -> None:
    used = [r for r in rows if "skipped" not in r]
    print(f"\n{'':<22}" + "".join(f"{name:>26}" for name in VARIANTS))
    print(f"{by:<22}" + "".join(f"{'acc  recall  false':>26}" for _ in VARIANTS))
    for r in rows:
        if "skipped" in r:
            print(f"{r['unit'][:21]:<22} skipped: {r['skipped']}")
            continue
        print(f"{r['unit'][:21]:<22}" + "".join(
            f"{r[n]['accuracy']:>12.3f} {r[n]['recall']:>6.3f} {r[n]['false_triggers']:>6.3f}" for n in VARIANTS))
    if used:
        print(f"{'MEAN of ' + str(len(used)):<22}" + "".join(
            f"{mean[n]['accuracy']:>12.3f} {mean[n]['recall']:>6.3f} {mean[n]['false_triggers']:>6.3f}" for n in VARIANTS))
        for n in VARIANTS[1:]:
            print(f"  {n:<9} accuracy better for {int(mean[n]['units_better'])}, worse for {int(mean[n]['units_worse'])} of {len(used)}")
    print("acc = share of the test pictures named correctly; recall = gestures recognised at the engine's 0.6 threshold; "
          "false = non-gestures that would trigger there")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def cmd_fit(args: argparse.Namespace) -> int:
    import torch

    from .train import check_onnx, export_onnx

    device = pick_device(args.device)
    net, meta, cfg, labels = load_model(args.model, device)
    calib = load_pictures(args.calib, args.calib_dataset, labels, cfg)
    pad = int(calib["pad"])
    x = torch.from_numpy(np.ascontiguousarray(center_crop(calib["X"], cfg, pad))).to(device)
    y = torch.from_numpy(np.asarray(calib["y"], dtype=np.int64)).to(device)
    counts = {name: int((calib["y"] == i).sum()) for i, name in enumerate(labels)}
    print(f"calibration pictures: {counts} from {len(set(calib['group']))} session(s), device {device}")
    gestures = [name for name in labels if name != "none"]
    if any(counts[g] == 0 for g in gestures):
        print(f"WARNING: no example of {[g for g in gestures if counts[g] == 0]} in the calibration; those keep the trained behaviour")
    temperature = float(meta.get("temperature", 1.0))
    before = score(adapted_module(net), x, y, temperature, args.min_confidence)
    room = None
    if args.room or args.room_dataset:                         # extra recordings from the room: statistics only, labels ignored
        extra = load_pictures(args.room, args.room_dataset, labels, cfg)
        room = torch.from_numpy(np.ascontiguousarray(center_crop(extra["X"], cfg, int(extra["pad"])))).to(device)
    tuned = copy.deepcopy(net)
    info: dict[str, Any] = {"from_model": str(args.model), "sessions": sorted({str(g) for g in calib["group"]}), "pictures": counts,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if not args.no_bn:
        xs = x if room is None else torch.cat([x, room])
        ws = balanced_weights(y, len(labels)) if room is None else None
        info["batchnorm"] = {"prior": args.bn_prior, "pictures": int(len(xs)), "layers": adapt_batchnorm(tuned, xs, ws, args.bn_prior)}
        print(f"BatchNorm statistics: {len(xs)} room pictures, room share {info['batchnorm']['layers'][0]['room_share']:.2f}; "
              f"first layer moved by {info['batchnorm']['layers'][0]['mean_shift_in_spreads']:.2f} spreads")
    protos = pmap = None
    if args.prototypes:
        protos, pmap = make_prototypes(tuned, x, y, len(labels))
        info["prototypes"] = {"classes": [labels[int(i)] for i in pmap.argmax(dim=1).tolist()], "beta": args.beta}
        print(f"prototypes for {info['prototypes']['classes']}, bonus weight {args.beta}")
    module = adapted_module(tuned, protos, pmap, args.beta if args.prototypes else 0.0).cpu()
    after = score(module.to(device), x, y, temperature, args.min_confidence)
    module = module.cpu()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    in_shape = (cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames)
    export_onnx(module, out / "gesture.onnx", in_shape)
    probe = np.ascontiguousarray(center_crop(calib["X"][:64].astype(np.float32), cfg, pad))
    diff, ms = check_onnx(out / "gesture.onnx", module, probe)
    torch.save({"model": tuned.cpu().state_dict(), "labels": labels, "width": int(tuned.head[3].in_features // 4)}, out / "gesture.pt")
    real = calib["y"] != 0
    p10 = meta.get("gesture_peak_db_p10")
    if real.sum() >= 3:
        room_p10 = round(float(np.percentile(calib["peak_db"][real], 10)), 1)
        info["gesture_peak_db_p10"] = {"trained": p10, "room": room_p10}
        if p10 is None or room_p10 < p10:                      # fainter gestures here: do not let the engine's gate drop them
            p10 = room_p10
    info["on_calibration"] = {"before": before, "after": after}
    new_meta = {**meta, **({"gesture_peak_db_p10": p10} if p10 is not None else {}), "adapted": info}
    (out / "gesture.json").write_text(json.dumps(new_meta, indent=1))
    print(f"on the calibration pictures themselves (flattering, they were used to adapt): accuracy {before['accuracy']:.2f} -> {after['accuracy']:.2f}, "
          f"gestures recognised {before['recall']:.2f} -> {after['recall']:.2f}, non-gestures triggering {before['false_triggers']:.2f} -> {after['false_triggers']:.2f}")
    print(f"wrote {out / 'gesture.onnx'} (+ .json, .pt); ONNX vs PyTorch max logit difference {diff:.2e}, {ms:.2f} ms per inference on CPU")
    print(f"use it:   python -m sonar.engine --model {out / 'gesture.onnx'}        (leave --model out to go back to the original)")
    if diff > 1e-3:
        print("WARNING: the exported ONNX model disagrees with PyTorch; do not use it.", file=sys.stderr)
        return 1
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import torch

    device = pick_device(args.device)
    net, meta, cfg, labels = load_model(args.model, device)
    data = load_pictures(args.data, args.dataset, labels, cfg)
    real = ~np.isin(data["source"], ("synthetic", "pseudo", "rescued", "aux", "hardneg", "retimed"))
    people = [p for p in args.people.split(",") if p]
    keep = real & (np.isin(data["person"], people) if people else np.ones(len(real), dtype=bool))
    if not keep.any():
        print(f"ERROR: no recorded pictures of {people or 'anybody'}; people in the data: {sorted(set(data['person'][real]))}", file=sys.stderr)
        return 2
    pad = int(data["pad"])
    t0 = time.perf_counter()
    x = torch.from_numpy(np.ascontiguousarray(center_crop(data["X"][keep], cfg, pad))).to(device)       # once; stays on the device
    y = torch.from_numpy(np.asarray(data["y"][keep], dtype=np.int64)).to(device)
    units = data["person" if args.by == "person" else "group"][keep]
    t1 = time.perf_counter()
    rows, mean = evaluate(net, x, y, units, float(meta.get("temperature", 1.0)), args.shots, args.bn_prior, args.beta,
                          args.min_confidence, transductive=args.transductive)
    seconds = time.perf_counter() - t1
    if not people:
        print("NOTE: no --people given: every person in the data is scored. For people the model was TRAINED on the gain is "
              "understated; score the held-out people of a sweep run (TRAINING_RUN.json lists them) for the honest number.")
    stats = "all of the unit's pictures, no labels" if args.transductive else f"the first {args.shots} of each gesture + a few non-gestures"
    print(f"{int(keep.sum())} pictures of {len(set(units.tolist()))} {args.by}(s) on {device}; calibration = {stats}; "
          f"BatchNorm prior {args.bn_prior}, prototype bonus {args.beta}")
    print_table(rows, mean, args.by)
    print(f"time: pictures to {device} {t1 - t0:.2f} s, all variants for all {args.by}s {seconds:.2f} s")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"model": str(args.model), "by": args.by, "shots": args.shots, "bn_prior": args.bn_prior,
                                               "beta": args.beta, "transductive": args.transductive, "rows": rows, "mean": mean}, indent=1))
    return 0 if any("skipped" not in r for r in rows) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sonar.adapt", description="Adapt a trained gesture model to a room, and measure the effect")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="models/gesture.onnx", help="trained model (its .json and .pt must sit next to it)")
    common.add_argument("--device", default="auto", help="auto (CUDA if there is one, else CPU) | cuda | cpu | mps")
    common.add_argument("--bn-prior", type=float, default=16.0, help="how many room pictures the trained statistics are worth")
    common.add_argument("--beta", type=float, default=4.0, help="weight of the prototype bonus")
    common.add_argument("--min-confidence", type=float, default=0.6, help="the engine's confidence threshold")
    fit = sub.add_parser("fit", parents=[common], help="adapt the model to calibration sessions and write a new model folder")
    fit.add_argument("--calib", nargs="*", default=[], help="calibration session folder(s): each gesture a few times")
    fit.add_argument("--calib-dataset", nargs="*", default=[], help="the same as ready-made .npz files")
    fit.add_argument("--room", nargs="*", default=[], help="more recordings from the room, used for the statistics only (labels ignored)")
    fit.add_argument("--room-dataset", nargs="*", default=[], help="the same as ready-made .npz files")
    fit.add_argument("--out", default="models/room", help="folder for the adapted model")
    fit.add_argument("--no-bn", action="store_true", help="leave the BatchNorm statistics as trained")
    fit.add_argument("--prototypes", action="store_true", help="also add the prototype bonus (see DATA_ENGINE.md for when it helps)")
    ev = sub.add_parser("eval", parents=[common], help="measure the effect on recorded sessions, person by person")
    ev.add_argument("--data", nargs="*", default=[], help="folders with recorded sessions")
    ev.add_argument("--dataset", nargs="*", default=[], help="ready-made .npz datasets (e.g. train_out/sessions/*.npz)")
    ev.add_argument("--people", default="", help="only these people (the ones the model was NOT trained on), e.g. guest3,guest7")
    ev.add_argument("--by", choices=["person", "session"], default="person", help="what counts as one room / one calibration")
    ev.add_argument("--shots", type=int, default=3, help="examples of each gesture used as the calibration")
    ev.add_argument("--transductive", action="store_true", help="BatchNorm statistics from all of the person's pictures, without labels")
    ev.add_argument("--json", help="write the table to this file")
    args = ap.parse_args(argv)
    if pick_device("cpu") is None:
        print("ERROR: sonar.adapt needs PyTorch (pip install torch; the CPU build is enough)", file=sys.stderr)
        return 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        return cmd_fit(args) if args.cmd == "fit" else cmd_eval(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

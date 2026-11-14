"""package.py — turn one of the suite's exported models into the files the engine loads, with its operating point.

What it does:
  1. takes the recipe's final models from <run>/jobs/<arm>__final__s<seed>.pt (trained on everyone on the 4090):
     all seeds by default, because the honest score is the score of their AVERAGE (the suite's report averages
     the seeds' probabilities), so the same average is what gets shipped; --single ships seed 0 alone,
  2. applies the operating point chosen WITHOUT the test person. The engine computes softmax(logits / temperature)
     and fires at confidence >= 0.6. The average of several networks cannot be written as one such formula, so the
     exported graph does it inside: per network (logits - handicap on "none") / temperature, softmax, average, log.
     gesture.json then says temperature 1.0 and the engine needs no change,
  3. sets `gesture_peak_db_p10` (the engine ignores bursts 6 dB weaker than that),
  4. re-exports gesture.onnx, checks it against PyTorch, loads it exactly as the engine does and runs the
     simulated demo visitor through it, then replays every recorded session through the engine's rules.

How to run:
    .venv/bin/python -m recall.package <run folder> <arm> --out models-release-mac-recall [--temperature T --handicap B --peak-p10 P]
Without --temperature / --handicap the point is the median of the per-fold nested choices in analysis.json.
What it changes: writes gesture.onnx / gesture.json / gesture.pt / metrics.json into --out. Nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .common import REPO, STREAMS, fixed_meta

sys.path.insert(0, str(REPO))
LABELS = ("none", "swipe_left", "swipe_right", "push")


def main() -> int:
    import torch

    from sonar.config import SonarConfig
    from sonar.model import GestureNet
    from sonar.train import check_onnx, export_onnx

    from .engine_replay import load_trace, onnx_probs, replay, summarise, trial_table

    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("arm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--handicap", type=float)
    ap.add_argument("--peak-p10", type=float, help="value stored as gesture_peak_db_p10 (default: keep the exported one)")
    ap.add_argument("--no-range-gate", action="store_true", help="replay with the range gate off (the MacBook setting)")
    ap.add_argument("--entering-gate-s", type=float, help="replay with the walking-in gate capped at this many seconds")
    ap.add_argument("--single", action="store_true", help="ship seed 0 alone instead of the average of all seeds")
    args = ap.parse_args()
    run = Path(args.run)
    src = run / "final" / args.arm
    meta = json.loads((src / "gesture.json").read_text())
    t, b = args.temperature, args.handicap
    analysis = run / "analysis.json"
    if (t is None or b is None) and analysis.exists():
        folds = json.loads(analysis.read_text()).get(args.arm, {}).get("points", {}).get("folds", {})
        if folds:
            t = float(np.median([f["t"] for f in folds.values()])) if t is None else t
            b = float(np.median([f["b"] for f in folds.values()])) if b is None else b
    t = 1.0 if t is None else t
    b = 0.0 if b is None else b

    man = json.loads((run / "manifest.json").read_text())
    seeds = man["seeds"][:1] if args.single else man["seeds"]
    states = [torch.load(run / "jobs" / f"{args.arm}__final__s{k}.pt", map_location="cpu") for k in seeds
              if (run / "jobs" / f"{args.arm}__final__s{k}.pt").exists()]
    if not states:
        print(f"ERROR: no final models of {args.arm} in {run / 'jobs'}")
        return 1
    width = int(states[0].get("width", 24))
    cfg = SonarConfig(n_mics=1)

    class Averaged(torch.nn.Module):
        """Several GestureNets and the operating point in one graph: input picture -> log of the averaged probabilities."""

        def __init__(self, members: list, temperature: float, handicap: float) -> None:
            super().__init__()
            self.members = torch.nn.ModuleList(members)
            shift = torch.zeros(len(LABELS))
            shift[0] = handicap
            self.register_buffer("shift", shift)
            self.temperature = float(temperature)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            probs = [torch.softmax((m(x) - self.shift) / self.temperature, dim=1) for m in self.members]
            return torch.log(torch.stack(probs).mean(dim=0) + 1e-12)

    members = []
    for st in states:
        net = GestureNet(cfg.n_feat_channels, len(LABELS), width=width)
        net.load_state_dict(st["model"])
        members.append(net.eval())
    model = Averaged(members, t, b).eval()
    out = (REPO / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # gesture.pt keeps the format `--init` reads (one GestureNet under "model"); the other seeds ride along
    torch.save({"model": states[0]["model"], "labels": list(LABELS), "width": width, "members": [st["model"] for st in states],
                "operating_point": {"temperature": float(t), "none_handicap": float(b)}}, out / "gesture.pt")
    in_shape = (cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames)
    export_onnx(model, out / "gesture.onnx", in_shape)
    with np.load(run / "table.npz", allow_pickle=False) as z:
        probe = np.ascontiguousarray(z["X"][:64].astype(np.float32)[..., 8:88])
    diff, ms = check_onnx(out / "gesture.onnx", model, probe)
    meta["temperature"] = 1.0                 # the operating point is inside the graph (see the header)
    if args.peak_p10 is not None:
        meta["gesture_peak_db_p10"] = float(args.peak_p10)
    meta["metrics"] = {**meta.get("metrics", {}), "networks_averaged": len(states),
                       "operating_point": {"temperature": float(t), "none_handicap": float(b), "inside_the_graph": True,
                                           "chosen_on": "training humans only (nested), see analysis.md"},
                       "onnx_max_logit_diff": diff, "onnx_ms_per_inference": round(ms, 3)}
    meta["trained"] = meta.get("trained", time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "gesture.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}: {len(states)} network(s) averaged, temperature {t:.2f}, none handicap {b:+.2f}, gesture_peak_db_p10 {meta.get('gesture_peak_db_p10')}, "
          f"ONNX vs PyTorch max logit difference {diff:.2e}, {ms:.2f} ms per inference")
    if diff > 1e-3:
        print("ERROR: the exported ONNX model disagrees with PyTorch; do not deploy it.")
        return 1

    # the engine loads it and runs the simulated visitor (same check as scripts/train_4090.py, step 7)
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_4090", REPO / "scripts" / "train_4090.py")
    t4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t4)
    check = t4.verify_engine(out / "gesture.onnx", "default", t4.Log(out / "engine_check.txt"))

    # every recorded session through the engine's own gesture rules with this model (it was TRAINED on these people:
    # this shows the demo behaviour for the people it knows, not the unseen-person score)
    names = sorted(p.stem for p in STREAMS.glob("*.npz"))
    tabs = {}
    for n in names:
        trace = load_trace(n)
        probs, labels, m = onnx_probs(out / "gesture.onnx", trace["windows"])
        gate = max(cfg.seg_min_peak_db, float(m.get("gesture_peak_db_p10", 9.0)) - 6.0)
        fired, _ = replay(trace, probs, labels, cfg, min_peak_db=gate, gate_range=not args.no_range_gate, entering_cap_s=args.entering_gate_s)
        tabs[n] = trial_table(fixed_meta(json.loads((STREAMS / f"{n}.json").read_text())["meta"], n), fired, labels, len(trace["state"]))
    lines = []
    for person in sorted({n.split("_")[1] for n in names}):
        s = summarise([v for n, v in tabs.items() if n.split("_")[1] == person], LABELS)
        lines.append(f"{person:<8} right gesture first on {s['correct']}/{s['n_prompts']} prompts ({s['recall']:.2f}), clean {s['clean_rate']:.2f}, "
                     f"wrong first {s['wrong']}, false events {s['outside_fired']} in {s['outside_minutes']:.1f} min")
    print("engine replay on the people it was trained on (range gate " + ("off" if args.no_range_gate else "on") + "):\n  " + "\n  ".join(lines))
    (out / "metrics.json").write_text(json.dumps({"operating_point": {"temperature": t, "none_handicap": b}, "engine_check": check,
                                                  "in_sample_engine_replay": lines, "source": str(src)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

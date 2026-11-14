"""diag_trials.py — per prompt, what did the engine see and decide? (diagnosis stages c and d)

For one session and one model it prints every prompt with all its bursts: when each started after "GO", how
long and how strong it was, what the engine's gates said (weak / entering / far), what the model answered and
how sure it was, and which gestures finally fired.
How to run:   .venv/bin/python -m recall.diag_trials <session-part> [--model path.onnx] [--only-bad]     Read-only.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .common import REPO, STREAMS, fixed_meta
from .engine_replay import load_trace, onnx_probs, replay, trial_table
from .trainlib import CFG


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--model", default="train_out/recall/baseline/models/gesture.onnx")
    ap.add_argument("--only-bad", action="store_true")
    args = ap.parse_args()
    name = [p.stem for p in sorted(STREAMS.glob("*.npz")) if args.session in p.stem][0]
    trace = load_trace(name)
    probs, labels, m = onnx_probs((REPO / args.model).resolve(), trace["windows"])
    min_peak = max(CFG.seg_min_peak_db, float(m.get("gesture_peak_db_p10", 9.0)) - 6.0)
    fired, verdict = replay(trace, probs, labels, CFG, min_peak_db=min_peak)
    meta = fixed_meta(json.loads((STREAMS / f"{name}.json").read_text())["meta"], name)
    table = trial_table(meta, fired, labels, len(trace["state"]))
    evs = [e for e in meta["events"] if e.get("kind") == "gesture"]
    segs = trace["segs"]
    print(f"{name}   model {args.model}   min peak {min_peak:.1f} dB, temperature {m.get('temperature')}")
    for n, (e, row) in enumerate(zip(evs, table["trials"])):
        if args.only_bad and row["outcome"] == "correct" and row["extra"] == 0:
            continue
        f0 = int(e["start"] // 960) + 1
        f_next = int(evs[n + 1]["start"] // 960) + 1 if n + 1 < len(evs) else f0 + 250
        print(f"#{row['rep']:>2} {row['prompt']:<12} -> {row['outcome']:<8} fired {row['fired']}")
        for k in range(len(segs)):
            if not (f0 - 10 <= segs[k][2] < f_next - 10):
                continue
            p = probs[k]
            order = np.argsort(-p)
            print(f"      burst at {(segs[k][1] - f0) / 50:+5.2f} s  {(segs[k][2] - segs[k][1] + 1) / 50:4.2f} s long  peak {segs[k][4]:4.1f} dB{' CUT' if segs[k][6] else '    '}  "
                  f"gate/answer: {verdict[k]:<8}  model: {labels[order[0]]} {p[order[0]]:.2f}, {labels[order[1]]} {p[order[1]]:.2f}")


if __name__ == "__main__":
    main()

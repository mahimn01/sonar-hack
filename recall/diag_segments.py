"""diag_segments.py — stage (a)/(b) of the diagnosis: does every prompted gesture produce a burst, and when?

For every gesture session it prints, per gesture class: how many prompts got no burst at all, how many
bursts fall inside a prompt window, when the strongest burst starts relative to "GO", how long and how
strong it is, and how often it was cut by the 1.4 s length limit.
How to run:   .venv/bin/python -m recall.diag_segments          Read-only.
"""

from __future__ import annotations

import numpy as np

from .common import GESTURES, load_sessions, trials


def main() -> None:
    for s in load_sessions("gestures"):
        tr = trials(s)
        print(f"\n=== {s.name}  ({len(tr)} prompts, {len(s.segs)} bursts in the whole session)")
        print(f"  {'class':<12} {'n':>3} {'missed':>6} {'bursts/trial':>12} {'best start after GO (s) p10/50/90':>34} "
              f"{'best dur (s) p10/50/90':>24} {'peak dB p10/50/90':>20} {'forced':>6}")
        for g in GESTURES:
            rows = [t for t in tr if t["label"] == g]
            if not rows:
                continue
            missed = sum(1 for t in rows if not t["inside"])
            nb, st, du, pk, fo = [], [], [], [], 0
            for t in rows:
                if not t["inside"]:
                    continue
                nb.append(len(t["inside"]))
                k = max(t["inside"], key=lambda k: s.segs[k, 3])
                st.append((s.segs[k, 0] - t["f0"]) / 50.0)
                du.append((s.segs[k, 1] - s.segs[k, 0] + 1) / 50.0)
                pk.append(s.segs[k, 3])
                fo += int(s.segs[k, 5])
            q = lambda v: "/".join(f"{x:.2f}" for x in np.percentile(v, [10, 50, 90])) if v else "-"
            print(f"  {g:<12} {len(rows):3d} {missed:6d} {np.mean(nb) if nb else 0:12.2f} {q(st):>34} {q(du):>24} "
                  f"{'/'.join(f'{x:.0f}' for x in np.percentile(pk, [10, 50, 90])) if pk else '-':>20} {fo:6d}")
        # which burst of the trial is the strongest: first, second ...?
        order = [sorted(t["inside"], key=lambda k: s.segs[k, 0]).index(max(t["inside"], key=lambda k: s.segs[k, 3]))
                 for t in tr if t["inside"]]
        print("  position of the strongest burst among the trial's bursts (0 = first):", np.bincount(order).tolist())
        n_in = sum(len(t["inside"]) for t in tr)
        print(f"  bursts inside prompt windows: {n_in}; outside: {len(s.segs) - n_in}")


if __name__ == "__main__":
    main()

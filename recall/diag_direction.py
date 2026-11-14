"""diag_direction.py — stage (f) of the diagnosis: what tells LEFT from RIGHT on this one-microphone MacBook?

For each prompted swipe/push it takes the strongest 0.3 s of the labelled burst and prints, per person and
class, the average Doppler shift of the left-speaker tone (L), of the right-speaker tone (R), and L minus R.
If the microphone sits next to the LEFT speaker, a hand moving left approaches both (L strongly positive)
and a hand moving right leaves both (L strongly negative); L-R positive = moving left is the textbook cue.
How to run:   .venv/bin/python -m recall.diag_direction        Read-only.
"""

from __future__ import annotations

import numpy as np

from .common import load_sessions, trials

F = (np.arange(64) - 32) * (2000 / 256)


def cues(img: np.ndarray) -> tuple[float, float, float, float]:
    """img [2, 64, T] -> (centroid L, centroid R, L-R, share of L energy) over the strongest 15 frames."""
    use = np.abs(F) >= 15
    w = np.maximum(img - 0.3, 0) * use[None, :, None]
    e = w.sum(axis=1)                                   # [2, T]
    tot = np.convolve(e.sum(axis=0), np.ones(15), mode="same")
    c = int(np.argmax(tot))
    sl = slice(max(0, c - 7), c + 8)
    ws = w[:, :, sl]
    cen = (ws * F[None, :, None]).sum(axis=(1, 2)) / (ws.sum(axis=(1, 2)) + 1e-9)
    return float(cen[0]), float(cen[1]), float(cen[0] - cen[1]), float(ws[0].sum() / (ws.sum() + 1e-9))


def main() -> None:
    print(f"{'person':<8} {'class':<12} {'n':>3} {'L Hz mean±sd':>16} {'R Hz mean±sd':>16} {'L-R mean±sd':>16} {'L-R >0':>7} {'L>0':>5}")
    for s in load_sessions("gestures"):
        for g in ("swipe_left", "swipe_right", "push", "pull"):
            rows = []
            for t in trials(s):
                if t["label"] != g or not t["inside"]:
                    continue
                k = max(t["inside"], key=lambda k: s.segs[k, 3])
                st, en = int(s.segs[k, 0]), int(s.segs[k, 1])
                img = s.feat[st - 1:en].transpose(1, 2, 0)
                rows.append(cues(img))
            if not rows:
                continue
            r = np.array(rows)
            print(f"{s.person:<8} {g:<12} {len(r):3d} {r[:, 0].mean():8.1f}±{r[:, 0].std():5.1f} {r[:, 1].mean():8.1f}±{r[:, 1].std():5.1f} "
                  f"{r[:, 2].mean():8.1f}±{r[:, 2].std():5.1f} {np.mean(r[:, 2] > 0):7.2f} {np.mean(r[:, 0] > 0):5.2f}")


if __name__ == "__main__":
    main()

"""diag_crossing.py — is the swipe direction still in the signal for very fast swipes (lance)?

Physics on this MacBook (microphone next to the left speaker): during any swipe each tone's echo first rises in
pitch (hand approaching) and then drops (hand leaving). The moment of the switch is when the hand passes
  * the left speaker/microphone, for the left-speaker tone (L),
  * the middle of the laptop, for the right-speaker tone (R).
A hand moving LEFT passes the middle first, so R switches before L; moving RIGHT, L switches before R. The gap is
14 cm / hand speed: 140 ms at 1 m/s, about 30 ms (1.5 frames) at lance's speed.
This script measures that gap per prompted swipe: for each tone, "approach minus leave" energy within +-125 Hz
of the tone, frame by frame, around the strongest moment of the burst; the gap is the shift (in frames, with
sub-frame interpolation) that best lines the two curves up.
How to run:   .venv/bin/python -m recall.diag_crossing        Read-only.
"""

from __future__ import annotations

import numpy as np

from .common import load_sessions, trials

F = (np.arange(64) - 32) * (2000 / 256)
NEAR = (np.abs(F) >= 12) & (np.abs(F) <= 125)


def balance(img: np.ndarray) -> np.ndarray:
    """img [64, T] -> approach-minus-leave energy share per frame, in -1..1 (0 where there is no energy)."""
    w = np.maximum(img - 0.25, 0.0) * NEAR[:, None]
    pos, neg = w[F > 0].sum(axis=0), w[F < 0].sum(axis=0)
    return (pos - neg) / (pos + neg + 1e-6) * np.minimum(1.0, (pos + neg) / 2.0)


def lag_frames(a: np.ndarray, b: np.ndarray, max_lag: int = 8) -> float:
    """Shift of `a` relative to `b` (frames; positive = a happens LATER than b) by cross-correlation with a parabola fit."""
    lags = np.arange(-max_lag, max_lag + 1)
    c = np.array([np.sum(a[max(0, k):len(a) + min(0, k)] * b[max(0, -k):len(b) + min(0, -k)]) for k in lags])
    i = int(np.argmax(c))
    if 0 < i < len(c) - 1 and (c[i - 1] - 2 * c[i] + c[i + 1]) < 0:
        return float(lags[i] + 0.5 * (c[i - 1] - c[i + 1]) / (c[i - 1] - 2 * c[i] + c[i + 1]))
    return float(lags[i])


def main() -> None:
    print("gap = how much LATER the left-speaker tone switches from 'approaching' to 'leaving' than the right-speaker tone")
    print(f"{'session':<34} {'prompt':<12} {'n':>3} {'gap, frames (20 ms) mean±sd':>28} {'share with gap > 0':>19}")
    for s in load_sessions("gestures"):
        for g in ("swipe_left", "swipe_right"):
            gaps = []
            for t in trials(s):
                if t["label"] != g or not t["inside"]:
                    continue
                k = max(t["inside"], key=lambda k: s.segs[k, 3])
                st, en = int(s.segs[k, 0]), int(s.segs[k, 1])
                seg = s.feat[max(st - 6, 0):en + 5].transpose(1, 2, 0)                  # [2, 64, frames]
                e = np.convolve(np.maximum(seg - 0.25, 0).sum(axis=(0, 1)), np.ones(9), mode="same")
                c = int(np.argmax(e))
                sl = slice(max(0, c - 14), c + 15)
                gaps.append(lag_frames(balance(seg[0][:, sl]), balance(seg[1][:, sl])))
            g_ = np.array(gaps)
            print(f"{s.name:<34} {g:<12} {len(g_):3d} {g_.mean():+17.2f} ± {g_.std():4.2f} {np.mean(g_ > 0):19.2f}")


if __name__ == "__main__":
    main()

"""diag_physics.py — how far do training-free physics rules get on these recordings? (a ceiling / sanity check)

Two rule sets are scored on every person, same definition as the model's score (gestures recognised with
confidence >= 0.6, non-gestures that trigger):
  1. the engine's own physics classifier (sonar.gestures.heuristic_classify) with the speaker weight the
     training script fits (it has no person-specific parts beyond that one number);
  2. a 3-number rule made for this MacBook from the diagnosis: over the strongest half second, D = Doppler of
     the left-speaker tone minus the right-speaker tone, S = their mean; |D| large -> swipe (sign = direction),
     else approach-then-retreat -> push. Thresholds are fixed by hand here, NOT fitted per person.
How to run:   .venv/bin/python -m recall.diag_physics        Read-only.
"""

from __future__ import annotations

import sys

import numpy as np

from .common import REPO
from .data import LABELS, all_official
from .trainlib import CFG, score

sys.path.insert(0, str(REPO))
from sonar.config import GESTURE_LABELS  # noqa: E402
from sonar.gestures import heuristic_classify  # noqa: E402

F = (np.arange(64) - 32) * (2000 / 256)


def rule(img: np.ndarray, d_thr: float = 45.0) -> np.ndarray:
    """img [2, 64, 80] -> pseudo-probabilities over LABELS (none, swipe_left, swipe_right, push)."""
    use = np.abs(F) >= 15
    peak = float(img[:, use, :].max())
    if peak < 0.5:
        return np.array([1.0, 0, 0, 0])
    w = np.maximum(img - 0.3, 0) * use[None, :, None]
    e = w.sum(axis=1)
    tot = np.convolve(e.sum(axis=0), np.ones(25), mode="same")
    c = int(np.argmax(tot))
    sl = slice(max(0, c - 12), c + 13)
    ws = w[:, :, sl]
    cen = (ws * F[None, :, None]).sum(axis=(1, 2)) / (ws.sum(axis=(1, 2)) + 1e-9)
    d = cen[0] - cen[1]
    # push: both tones first positive (approach) somewhere in the picture
    pos = (w[:, F > 0, :].sum(axis=(0, 1)))
    neg = (w[:, F < 0, :].sum(axis=(0, 1)))
    t = np.arange(img.shape[-1])
    t_pos = (pos * t).sum() / (pos.sum() + 1e-9)
    t_neg = (neg * t).sum() / (neg.sum() + 1e-9)
    p = np.full(4, 0.02)
    if abs(d) >= d_thr:
        p[1 if d > 0 else 2] = 1.0
    elif pos.sum() > 0.25 * neg.sum() and (t_neg - t_pos > 2 or neg.sum() < 0.3 * pos.sum()):
        p[3] = 1.0
    else:
        p[0] = 1.0
    return p / p.sum()


def main() -> None:
    t = all_official()
    y = t.y()
    x = t.X.astype(np.float32)[..., 8:88]
    cfg = CFG.with_(heur_w_speakers=0.29)             # what sonar.train.fit_heuristic gives on these 12 sessions
    lut = [GESTURE_LABELS.index(n) for n in LABELS]
    heur = np.stack([heuristic_classify(xi, cfg).probs[lut] for xi in x])
    heur = heur / heur.sum(axis=1, keepdims=True)
    mine = np.stack([rule(xi) for xi in x])
    for name, p in (("engine physics rules", heur), ("3-number MacBook rule", mine)):
        print(name)
        for h in ("lance", "aarsh", "mahimn"):
            m = t.human == h
            s = score(np.log(p[m] + 1e-9), y[m], LABELS)
            print(f"  {h:<7} recall {s['recall']:.3f} ({s['hits']}/{s['n_gesture']})  false {s['false']:.3f} ({s['false_n']}/{s['n_none']})  "
                  f"{s['per_class']}  wrong-gesture {s['wrong_gesture']}")


if __name__ == "__main__":
    main()

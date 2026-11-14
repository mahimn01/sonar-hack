"""diag_probe.py — a quick read on the two input changes with a tiny linear model (seconds, no GPU, no CNN).

Question: do the wider Doppler window (+-750 Hz) and the peak-relative burst cutting make the recorded gestures
easier to tell apart ACROSS people? The real answer comes from the CNN runs on the 4090. This is a cheap proxy:
each picture is shrunk (4x4 averaging), and a multinomial logistic regression is fitted on two humans and tested
on the third, for the four combinations of (window, cutting). Same rows for both windows of a cutting.
Reported per combination: share of the held-out human's gestures named correctly (no confidence threshold), and
the share of their non-gestures called a gesture.
A linear model on shrunken pictures is much weaker than the CNN, so only the ORDER of the four rows means anything.

How to run:   .venv/bin/python -m recall.diag_probe <run folder with table.npz and table_rel12.npz>     Read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

LABELS = ("none", "swipe_left", "swipe_right", "push")
HUMANS = ("lance", "aarsh", "mahimn")


def shrink(x: np.ndarray) -> np.ndarray:
    """[N, C, F, 96] -> un-shifted 80 frames, averaged over 4 x 4 blocks, flattened."""
    x = x[..., 8:88].astype(np.float32)
    n, c, f, t = x.shape
    return x.reshape(n, c, f // 4, 4, t // 4, 4).mean(axis=(3, 5)).reshape(n, -1)


def fit_lr(x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float, k: int = 4) -> np.ndarray:
    """Weighted multinomial logistic regression with an L2 penalty; returns [d + 1, k] (last row = bias)."""
    n, d = x.shape
    xb = np.hstack([x, np.ones((n, 1), np.float32)]).astype(np.float64)
    onehot = np.eye(k)[y]

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        th = theta.reshape(d + 1, k)
        z = xb @ th
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        nll = -(w * np.log(p[np.arange(n), y] + 1e-12)).sum()
        grad = xb.T @ ((p - onehot) * w[:, None])
        grad[:-1] += l2 * th[:-1]
        return nll + 0.5 * l2 * float((th[:-1] ** 2).sum()), grad.ravel()

    res = minimize(loss, np.zeros((d + 1) * k), jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return res.x.reshape(d + 1, k)


def run(table: dict[str, np.ndarray], key: str, l2: float) -> dict[str, tuple[int, int, int, int]]:
    lut = {n: i for i, n in enumerate(LABELS)}
    y = np.array([lut.get(str(v), 0) for v in table["label"]])
    x = shrink(table[key])
    scored = np.isin(table["source"], ["gesture", "burst", "idle"])
    train_rows = scored | (table["source"] == "aux")                       # wind-up / return bursts as "none", like the aux recipes
    out = {}
    for h in HUMANS:
        tr = train_rows & (table["human"] != h)
        te = scored & (table["human"] == h)
        mu, sd = x[tr].mean(axis=0), x[tr].std(axis=0) + 0.05
        counts = np.bincount(y[tr], minlength=4).astype(float)
        w = (1.0 / counts)[y[tr]]
        th = fit_lr((x[tr] - mu) / sd, y[tr], w / w.sum() * tr.sum(), l2)
        z = np.hstack([(x[te] - mu) / sd, np.ones((te.sum(), 1))]) @ th
        pred = z.argmax(axis=1)
        g = y[te] != 0
        out[h] = (int((pred[g] == y[te][g]).sum()), int(g.sum()), int((pred[~g] != 0).sum()), int((~g).sum()))
    return out


def main() -> None:
    run_dir = Path(sys.argv[1])
    tables = {}
    for name, f in (("engine cutting", "table.npz"), ("ends 12 dB below its peak", "table_rel12.npz")):
        with np.load(run_dir / f) as z:
            tables[name] = {k: z[k] for k in ("X", "X_wide", "label", "source", "human") if k in z.files}
    for l2 in (30.0, 100.0, 300.0):
        print(f"\nL2 penalty {l2:g}   (gestures named correctly on the held-out human / their non-gestures called a gesture)")
        print(f"  {'burst cutting':<28} {'Doppler window':<14} " + " ".join(f"{h:>18}" for h in HUMANS) + f" {'all three':>20}")
        for cut, t in tables.items():
            for key, win in (("X", "+-250 Hz"), ("X_wide", "+-750 Hz")):
                if key not in t:
                    continue
                r = run(t, key, l2)
                tot = [sum(v[i] for v in r.values()) for i in range(4)]
                print(f"  {cut:<28} {win:<14} " + " ".join(f"{r[h][0]:3d}/{r[h][1]:<3d} {r[h][2]:3d}/{r[h][3]:<4d}".rjust(18) for h in HUMANS) +
                      f" {tot[0]:4d}/{tot[1]} = {tot[0] / tot[1]:.2f}  {tot[2] / tot[3]:.2f}")


if __name__ == "__main__":
    main()

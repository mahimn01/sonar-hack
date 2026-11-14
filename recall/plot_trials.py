"""plot_trials.py — draw what really happened in each prompted trial (for looking, not for training).

For one gesture session it draws a sheet: one row per trial, both pilot-tone Doppler pictures from
0.6 s before "GO" to 4.2 s after, with every burst the segmenter cut marked (green = the burst the
dataset builder labels as the gesture, white = other bursts, red tick = cut by the 1.4 s limit) and
the movement-energy trace underneath.
How to run:   .venv/bin/python -m recall.plot_trials <session-name-part> [class]      Writes PNGs to train_out/recall/plots/.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import REPO, load_sessions, trials

OUT = REPO / "train_out" / "recall" / "plots"


def sheet(s, rows, title, path, pre=30, post=210):
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, 1.55 * n), gridspec_kw={"width_ratios": [1, 1, 0.8]})
    axes = np.atleast_2d(axes)
    for r, t in enumerate(rows):
        a, b = t["f0"] - pre, t["f0"] + post
        lo, hi = max(a, 1), min(b, len(s.feat))
        img = np.zeros((2, 64, b - a + 1), np.float32)
        img[:, :, lo - a:hi - a + 1] = s.feat[lo - 1:hi].transpose(1, 2, 0)
        best = max(t["inside"], key=lambda k: s.segs[k, 3]) if t["inside"] else -1
        for c in range(2):
            ax = axes[r, c]
            ax.imshow(img[c], origin="lower", aspect="auto", vmin=0, vmax=1, cmap="magma",
                      extent=[-pre / 50, post / 50, -250, 250])
            ax.axvline(0, color="cyan", lw=0.8)
            ax.axvline((t["f1"] - t["f0"]) / 50, color="cyan", lw=0.8, ls=":")
            ax.axhline(0, color="gray", lw=0.4)
            for k in range(len(s.segs)):
                st, en, dec, pk, _, forced = s.segs[k]
                if en < a or st > b:
                    continue
                col = "lime" if k == best else "white"
                ax.plot([(st - t["f0"]) / 50, (en - t["f0"]) / 50], [235, 235], color=col, lw=3)
                if forced:
                    ax.plot([(dec - t["f0"]) / 50] * 2, [200, 250], color="red", lw=2)
            ax.set_yticks([-200, 0, 200])
            ax.tick_params(labelsize=6)
            if c == 0:
                ax.set_ylabel(f"{t['label']}\n#{t['rep']}", fontsize=7)
        ax = axes[r, 2]
        tt = (np.arange(lo, hi + 1) - t["f0"]) / 50
        ax.plot(tt, s.act[lo - 1:hi], lw=0.8)
        ax.axhline(1.5, color="r", lw=0.5)
        ax.axvline(0, color="c", lw=0.8)
        ax.set_xlim(-pre / 50, post / 50)
        ax.set_ylim(-3, 55)
        ax.tick_params(labelsize=6)
    axes[0, 0].set_title("tone L 18.5 kHz (left speaker)", fontsize=8)
    axes[0, 1].set_title("tone R 19.0 kHz (right speaker)", fontsize=8)
    axes[0, 2].set_title("activity dB", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    only = sys.argv[2] if len(sys.argv) > 2 else None
    for s in load_sessions("gestures"):
        if want not in s.name:
            continue
        tr = trials(s)
        for g in ("swipe_left", "swipe_right", "push", "pull"):
            if only and g != only:
                continue
            rows = [t for t in tr if t["label"] == g][:10]
            if rows:
                p = OUT / f"trials_{s.person}_{g}.png"
                sheet(s, rows, f"{s.name}  {g}  (cyan = GO .. end of prompt; green = burst labelled as the gesture; red = cut at 1.4 s)", p)
                print(p)


if __name__ == "__main__":
    main()

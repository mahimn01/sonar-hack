"""plot_raw.py — look at the RAW microphone spectrum around the pilot tones during a few trials (no model features).

Why: lance's gestures show up as bright streaks on BOTH sides of each tone at once. A moving hand
(Doppler) puts energy on ONE side; a hand covering the microphone/speaker (shadowing) dims the tone
itself and puts energy on both sides. This plot shows which it is, and the tone levels over time.
How to run:   .venv/bin/python -m recall.plot_raw <session-part> <class> [n_trials]     Read-only; writes a PNG.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal as sps

from .common import REPO, load_sessions, trials

OUT = REPO / "train_out" / "recall" / "plots"


def main() -> None:
    want, cls = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    s = [x for x in load_sessions("gestures") if want in x.name][0]
    audio, fs = sf.read(str(REPO / "mac" / "data" / s.name / "audio.wav"), dtype="float32", always_2d=True)
    rows = [t for t in trials(s) if t["label"] == cls][:n]
    fig, axes = plt.subplots(len(rows), 2, figsize=(15, 3.0 * len(rows)), gridspec_kw={"width_ratios": [2, 1]})
    axes = np.atleast_2d(axes)
    for r, t in enumerate(rows):
        a = max(0, (t["f0"] - 30) * 960)
        b = min(len(audio), (t["f0"] + 200) * 960)
        x = audio[a:b, 0].astype(np.float64)
        f, tt, z = sps.stft(x, fs=fs, window="hann", nperseg=4096, noverlap=4096 - 480, nfft=8192, boundary=None)
        p = 20 * np.log10(np.abs(z) * 2 + 1e-12)
        sel = (f >= 17800) & (f <= 19800)
        ax = axes[r, 0]
        im = ax.imshow(p[sel], origin="lower", aspect="auto", extent=[tt[0] - 0.6, tt[-1] - 0.6, f[sel][0], f[sel][-1]],
                       vmin=-125, vmax=-45, cmap="magma")
        ax.set_ylabel(f"{cls} #{t['rep']}\nHz")
        ax.axvline(0, color="c", lw=0.8)
        plt.colorbar(im, ax=ax, pad=0.01)
        ax2 = axes[r, 1]
        for tone, col in ((18500, "C0"), (19000, "C1")):
            k = int(np.argmin(np.abs(f - tone)))
            ax2.plot(tt - 0.6, p[k], col, label=f"{tone} Hz carrier level")
        ax2.plot(tt - 0.6, 20 * np.log10(np.abs(z[(f >= 19500) & (f <= 21500)]).mean(axis=0) * 2 + 1e-12), "C2", label="chirp band mean")
        ax2.set_ylim(-80, -15)
        ax2.axvline(0, color="c", lw=0.8)
        ax2.legend(fontsize=6)
        ax2.set_ylabel("dBFS")
    fig.suptitle(f"{s.name}: raw spectrum (85 ms window), dBFS")
    fig.tight_layout()
    p = OUT / f"raw_{s.person}_{cls}.png"
    fig.savefig(p, dpi=70)
    print(p)


if __name__ == "__main__":
    main()

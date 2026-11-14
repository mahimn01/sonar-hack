"""wide_probe.py — would a wider Doppler window show the fast gestures properly? (diagnostic only, changes nothing)

The engine keeps +-250 Hz of Doppler around each pilot tone (64 bins). Lance's swipes and Mahimn's fast pushes
shift the tones by up to +-700 Hz, so most of their trace falls outside the picture or lands in the other
tone's picture. This script re-runs the engine's own receive chain on a recording with a wider window
(+-750 Hz: 2x the baseband rate, 192 bins of the same 7.8 Hz) and draws the same trials both ways.
It does NOT touch sonar/: the config check that forbids overlapping tone windows is bypassed only inside this
process.
How to run:   .venv/bin/python -m recall.wide_probe <session-part> <class> [n]      Writes a PNG. Read-only otherwise.
"""

from __future__ import annotations

import dataclasses
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import REPO, load_sessions, trials

sys.path.insert(0, str(REPO))
from sonar.config import SonarConfig  # noqa: E402
from sonar.dataset import load_session  # noqa: E402
from sonar.frontend import Frontend  # noqa: E402
from sonar.gestures import DopplerNormalizer  # noqa: E402

OUT = REPO / "train_out" / "recall" / "plots"


def wide_config(cfg: SonarConfig) -> SonarConfig:
    """The same signal, received with a +-750 Hz Doppler window per tone (bins stay 7.8 Hz wide)."""
    return dataclasses.replace(cfg, decim=12, stft_win=256, stft_fft=512, doppler_bins=192, demod_pass_hz=800.0)


def wide_features(folder, cfg_fn=wide_config) -> tuple[np.ndarray, SonarConfig]:
    """Feature columns [T, C, F] (0..1 above the noise floor) computed by the engine's Frontend + normaliser."""
    meta, audio, cfg = load_session(folder)
    wcfg = cfg_fn(cfg)
    real_validate = SonarConfig.validate
    SonarConfig.validate = lambda self: None          # only here: allow overlapping tone windows
    try:
        fe = Frontend(wcfg, n_mics=audio.shape[1])
        norm = DopplerNormalizer(wcfg)
        cols = []
        for i in range(0, len(audio), 9600):
            for fr in fe.process(audio[i:i + 9600]):
                f = norm.normalize(fr.doppler_db)
                cols.append(np.zeros((wcfg.n_power_channels, wcfg.doppler_bins), np.float32) if f is None else f)
                # the floor is frozen after calibration here (no tracking): fine for a look at a few minutes
    finally:
        SonarConfig.validate = real_validate
    return np.stack(cols), wcfg


def main() -> None:
    want, cls = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    s = [x for x in load_sessions("gestures") if want in x.name][0]
    folder = REPO / "mac" / "data" / s.name
    if not folder.exists():
        folder = REPO / "train_out" / "recall" / "excluded" / s.name
    feat, wcfg = wide_features(folder)
    rows = [t for t in trials(s) if t["label"] == cls][:n]
    fig, axes = plt.subplots(len(rows), 4, figsize=(18, 2.4 * len(rows)))
    axes = np.atleast_2d(axes)
    half = wcfg.doppler_bins // 2 * (wcfg.bb_fs / wcfg.stft_fft)
    for r, t in enumerate(rows):
        a, b = t["f0"] - 30, t["f0"] + 180
        for c in range(2):
            ax = axes[r, c]                                              # the engine's picture today
            ax.imshow(s.feat[a - 1:b, c].T, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="magma", extent=[-0.6, 3.6, -250, 250])
            ax.set_ylabel(f"{cls} #{t['rep']}", fontsize=7)
            ax = axes[r, 2 + c]                                          # the wide picture
            ax.imshow(feat[a - 1:b, c].T, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="magma", extent=[-0.6, 3.6, -half, half])
            for y in (-250, 250):
                ax.axhline(y, color="cyan", lw=0.4)
            ax.axhline(500 if c == 0 else -500, color="white", lw=0.4, ls=":")   # where the other tone's carrier sits
    for c, name in enumerate(("today: tone L +-250 Hz", "today: tone R +-250 Hz", "wide: tone L +-750 Hz", "wide: tone R +-750 Hz")):
        axes[0, c].set_title(name, fontsize=8)
    fig.suptitle(f"{s.name} {cls}: today's picture (left) and a +-750 Hz window (right). Cyan = today's limits; dotted = the other tone")
    fig.tight_layout()
    p = OUT / f"wide_{s.person}_{s.name[9:15]}_{cls}.png"
    fig.savefig(p, dpi=70)
    print(p)


if __name__ == "__main__":
    main()

"""plot.py — draw one recorded session as a picture, to eyeball what the sonar saw.

Plain words: for a session folder this draws, over time, the Doppler pictures
(one row per tone/microphone plus the left/right "lateral" picture), the
movement level, the bursts the segmenter cut out (grey bands), and the prompts
the person was given (coloured bars with the gesture name). If a swipe does not
look different from a push here, no model will tell them apart either.

How to run:
    python -m sonar.plot data/20261114-101500_alice_gestures            (writes session.png into the folder)
    python -m sonar.plot data/<session> --start 30 --seconds 20 --out zoom.png
Needs matplotlib (requirements-train.txt). Read-only apart from the PNG it writes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .dataset import load_session
from .features import stream_features


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sonar.plot", description="Plot a recorded session")
    ap.add_argument("session", help="session folder (contains session.json and audio.wav)")
    ap.add_argument("--start", type=float, default=0.0, help="first second to draw")
    ap.add_argument("--seconds", type=float, default=60.0, help="how many seconds to draw")
    ap.add_argument("--out", help="output PNG (default: <session>/session.png)")
    args = ap.parse_args(argv)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: pip install matplotlib", file=sys.stderr)
        return 2

    meta, audio, cfg = load_session(args.session)
    stream = stream_features(audio, cfg)
    rate = cfg.frame_rate
    a = int(args.start * rate)
    b = min(stream.n_frames, a + int(args.seconds * rate))
    t = np.arange(a, b) / rate
    names = [f"tone {int(f)} Hz, mic {m}" for f in cfg.tone_hz for m in range(cfg.n_mics)]
    if cfg.has_lateral:
        names.append("lateral (red = moving left, blue = moving right)")
    n = len(names)
    fig, axes = plt.subplots(n + 1, 1, figsize=(16, 2.0 * (n + 1)), sharex=True)
    extent = [t[0], t[-1], cfg.doppler_hz[0], cfg.doppler_hz[-1]]
    for c in range(n):
        lateral = cfg.has_lateral and c == n - 1
        axes[c].imshow(
            stream.feat[a:b, c].T, origin="lower", aspect="auto", extent=extent,
            cmap="RdBu_r" if lateral else "magma", vmin=-1 if lateral else 0, vmax=1,
        )
        axes[c].set_ylabel("Doppler Hz")
        axes[c].set_title(names[c], fontsize=9, loc="left")
    ax = axes[-1]
    ax.plot(t, stream.activity_db[a:b], color="black", linewidth=0.8)
    ax.axhline(cfg.seg_min_peak_db, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylabel("movement dB")
    ax.set_xlabel("seconds")
    for seg in stream.segments:
        if seg.end / rate >= t[0] and seg.start / rate <= t[-1]:
            ax.axvspan(seg.start / rate, seg.end / rate, color="gray", alpha=0.3)
    ax.set_xlim(t[0], t[-1])
    colors = {"swipe_left": "tab:red", "swipe_right": "tab:blue", "push": "tab:green", "pull": "tab:orange"}
    top = max(10.0, float(np.max(stream.activity_db[a:b])))
    for ev in meta.get("events", []):
        t0, t1 = ev["start"] / cfg.fs, ev["end"] / cfg.fs
        if t1 < t[0] or t0 > t[-1] or ev["kind"] == "calibration":
            continue
        ax.plot([t0, t1], [top, top], color=colors.get(ev["label"], "tab:purple"), linewidth=4, solid_capstyle="butt")
        ax.text(t0, top * 1.03, ev["label"], fontsize=7)
    fig.suptitle(f"{meta.get('session_id', Path(args.session).name)}   profile {meta.get('profile')}   person {meta.get('person')}")
    fig.tight_layout()
    out = Path(args.out) if args.out else Path(args.session) / "session.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}  ({len(stream.segments)} bursts, grey bands; prompts are the coloured bars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

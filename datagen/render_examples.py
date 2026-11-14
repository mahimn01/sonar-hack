"""render_examples.py — draw example pictures of every class so a human can eyeball them.

What it does
------------
Makes a few random synthetic clips per class and saves one PNG per class plus
an overview PNG. Each panel is the Doppler picture of one clip:
    x = time (s), y = pitch shift of the echo (Hz; up = moving towards the laptop)
    top half    = the LEFT speaker's tone, bottom half = the RIGHT speaker's tone
    cyan lines  = where the scripted gesture stroke really is
    white boxes = the movement bursts the core segmenter cut out, with the label
                  each burst would get in the training set
If the core `sonar/` package is importable the pictures are the core's own
model features (0..1 above the noise floor); otherwise a built-in spectrogram.

Why
---
Before training on synthetic data, look at it. A push should be a blob above
the centre line (then usually one below as the hand comes back); a swipe should
look different in the two halves; "none" should be mostly quiet or slow.

How to run (from the repo root)
-------------------------------
    cd ~/Documents/Dev/sonar-hack
    datagen/.venv/bin/python -m datagen.render_examples --out datagen/examples

Options: --per-class 6, --seed 0, --classes swipe_left,swipe_right,push,pull,none,
--clean (no noise or distractors: shows the pure physics).

What it changes: writes PNG files into --out. Nothing else. Read-only otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import adapter as ad
from . import quicklook as ql
from .generate import LEAD_IN, STORY, label_segments
from .hardware import Profile, clean_hardware, sample_hardware
from .render import render
from .scenes import sample_scene

DEFAULT_CLASSES = "swipe_left,swipe_right,push,pull,none,wave,present,entering"


def one_clip(core: ad.Core, tx_pack: tuple, label: str, seed: int, clean: bool) -> dict:
    """Render one clip and compute the picture to draw."""
    tx, pre, n = tx_pack
    rng = np.random.default_rng(seed)
    if clean:
        hw = clean_hardware(core.fs, "stereo" if core.n_speakers >= 2 else "both")
        hw.direct_gain_db, hw.noise_dbfs = -60.0, -110.0
    else:
        hw = sample_hardware(rng, Profile(band_hz=core.band), core.fs)
    centre = STORY / 2 if label in ("swipe_left", "swipe_right", "push", "pull", "wave") else None
    scene = sample_scene(rng, label, STORY, event_center=centre, lead_in=LEAD_IN, distractors=not clean)
    out = render(tx, hw, scene, n, rng, pre=pre, band=core.band)
    clip = {"scene": scene, "hw": hw, "snr": out.echo_snr_db, "segments": []}
    if core.available and (core.module("pipeline") is not None or core.module("gestures") is not None):
        st = ad.stream_features(core, out.rx)
        n_mic, n_tone = out.rx.shape[1], len(core.tones)
        power = st.feat[:, : n_tone * n_mic].reshape(len(st.feat), n_tone, n_mic, -1).mean(axis=2)  # [T, tones, F]
        clip["image"] = [power[:, k, :].T for k in range(n_tone)]
        clip["rows"] = [f"tone {int(f)} Hz (speaker {s})" for f, s in zip(core.tones, core.tone_speakers)]
        clip["cmaps"] = ["magma"] * n_tone
        if st.feat.shape[1] > n_tone * n_mic:  # the core's two-microphone "which way sideways" channel, -1..1
            clip["image"].append(st.feat[:, -1, :].T)
            clip["rows"].append("mic-pair lateral (red = towards user's left)")
            clip["cmaps"].append("coolwarm")
        clip["extent"] = [0.0, len(st.feat) / st.frame_rate, float(core.cfg.doppler_hz[0]), float(core.cfg.doppler_hz[-1])]
        clip["vmax"] = 1.0
        clip["segments"] = label_segments(st, scene.events)  # labels only, no training pictures needed
    else:
        power, freqs, times, _ = ql.doppler_spectrogram(out.rx, core.fs, core.tones, span_hz=300.0)
        db = 10 * np.log10(power + 1e-30)
        db -= np.median(db)
        clip["image"] = [db[:, :, m] for m in range(db.shape[2])]
        clip["extent"] = [times[0], times[-1], freqs[0], freqs[-1]]
        clip["vmax"] = 40.0
        clip["rows"] = [f"mic {m}" for m in range(db.shape[2])]
    return clip


def draw(clips: list, title: str, path: Path) -> None:
    """One column per clip, one row per tone (or mic)."""
    n_rows = len(clips[0]["image"])
    fig, axes = plt.subplots(n_rows, len(clips), figsize=(3.3 * len(clips), 2.1 * n_rows + 0.6), squeeze=False)
    for j, clip in enumerate(clips):
        sc = clip["scene"]
        for r in range(n_rows):
            ax = axes[r, j]
            cmap = clip.get("cmaps", ["magma"] * n_rows)[r]
            ax.imshow(clip["image"][r], origin="lower", aspect="auto", extent=clip["extent"],
                      vmin=-clip["vmax"] if cmap == "coolwarm" else 0.0, vmax=clip["vmax"], cmap=cmap)
            ax.axhline(0.0, color="w", lw=0.3, alpha=0.4)
            for ev in sc.events:
                ax.axvline(ev["t_start"], color="c", lw=0.8)
                ax.axvline(ev["t_end"], color="c", lw=0.8)
            for seg in clip["segments"]:
                ax.axvspan(seg["t_start"], seg["t_end"], ymin=0.02, ymax=0.98, fill=False, ec="w", ls="--", lw=0.8)
                if r == 0:
                    ax.text(seg["t_start"], clip["extent"][3] * 0.82, seg["label"], color="w", fontsize=6)
            ax.set_xlim(clip["extent"][0], clip["extent"][1])
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(clip["rows"][r] + "\nDoppler (Hz)", fontsize=6)
            if r == n_rows - 1:
                ax.set_xlabel("time (s)", fontsize=6)
        snr = clip["snr"]
        extra = f"SNR {snr:.0f} dB" if np.isfinite(snr) else ""
        speed = sc.meta.get("peak_speed")
        axes[0, j].set_title(
            f"{sc.label} / {sc.meta.get('variant', '')}  {extra}" + (f"  {speed:.1f} m/s" if speed else ""), fontsize=7
        )
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render example spectrograms per class (see the file header).")
    ap.add_argument("--out", default="datagen/examples")
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--classes", default=DEFAULT_CLASSES)
    ap.add_argument("--clean", action="store_true", help="no noise, no distractors: the pure physics")
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    core = ad.load_core(args.profile)
    print(core.describe())
    n = int(core.fs * (LEAD_IN + STORY))
    pre = ad.pre_roll(core)
    tx_pack = (core.tx(pre, n), pre, n)
    tag = "clean" if args.clean else "random"
    overview = []
    for ci, label in enumerate(args.classes.split(",")):
        clips = [one_clip(core, tx_pack, label, args.seed * 7919 + ci * 101 + k, args.clean) for k in range(args.per_class)]
        path = out_dir / f"{label}_{tag}.png"
        draw(clips, f"{label}  ({tag} virtual laptops)   cyan = true stroke, white box = burst the core cut out", path)
        overview.append(clips[0])
        print("wrote", path)
    path = out_dir / f"overview_{tag}.png"
    draw(overview, f"one example per class ({tag})", path)
    print("wrote", path)


if __name__ == "__main__":
    main()

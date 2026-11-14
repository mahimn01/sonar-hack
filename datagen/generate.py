"""generate.py — make a synthetic training set for the gesture model.

What it does
------------
For each example it invents a random virtual laptop + room and a random little
story ("a standing person swipes left in 0.3 s while someone walks past"),
computes what the two microphones would record, and pushes that audio through
the core package's OWN realtime chain (filters -> Doppler picture -> noise-floor
normaliser -> movement segmenter). Every movement burst the core would hand to
the model becomes one training example:
    * the burst that contains the scripted stroke gets the gesture label,
    * every other burst (hand raise, hand dropping back, fidgeting, a bystander,
      a click) is labelled "none". Those negatives are what stop false triggers.
    * now and then a picture of nothing at all is added as "none" too.

Why
---
A model pre-trained on thousands of these needs only a few dozen real clips to
adapt to the real laptop, instead of hours of hand recording.

How to run (from the repo root)
-------------------------------
    cd ~/Documents/Dev/sonar-hack
    datagen/.venv/bin/python -m datagen.generate --out datagen/out/synth_v5 --per-class 650

Useful options: --workers N (default: every CPU core), --save-audio 20 (keep the
first 20 clips per class as FLAC so you can listen / replay them), --min-snr 6
(drop gestures too faint to see), --seed 0, --profile default|low|mono|split (core
signal profile), --swap-mics / --swap-speakers (the real laptop's left/right
channels are the other way round; train.py and scripts/train_4090.py measure this
from recorded swipes).

Best realism, once the demo laptop has recorded a few minutes of "nobody
gesturing" with the probe signal playing (people may sit, type, walk past):
    datagen/.venv/bin/python -m datagen.generate --out datagen/out/synth_real_bg \
        --per-class 650 --background data/<scene session>/audio.wav
Then the laptop, room and noise in every example are REAL and only the hand is
synthetic (see background.py). --background-share 0.7 mixes both kinds.
Each background is converted once to a float32 .npy in <out>/backgrounds/ and
memory-mapped, so many workers share one copy instead of loading one each.

What it changes
---------------
Only writes inside the --out folder:
    dataset.npz     the training pictures in exactly the format of sonar/dataset.py
                    (X [N, C, F, T + 2*pad] float16, y, labels, group, person, source="synthetic",
                    peak_db, pad, config), so `python -m sonar.train --dataset <out>/dataset.npz` works.
                    It is a CACHE of the core feature code: re-run after the core changes.
    manifest.jsonl  one line per example: label, timing, virtual hardware, scene details
    audio/*.flac    optional raw 2-mic audio of the first clips of each class
    config.json     the probe-signal settings and generator arguments used
It never touches audio devices or anything else in the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import adapter as ad
from . import background as bg
from .hardware import Profile, sample_hardware
from .render import render
from .scenes import sample_scene

GESTURE4 = ("swipe_left", "swipe_right", "push", "pull")
# clip types that only ever produce negatives for the gesture model, and how often each is drawn
NEGATIVE_MIX = {"none": 0.6, "wave": 0.1, "entering": 0.1, "leaving": 0.1, "present": 0.1}
LEAD_IN = 1.7  # quiet seconds at the start of every clip: the core learns its noise floor here
STORY = 3.0  # seconds of scripted action after the lead-in

_CORE = None
_TX = None
_BACKGROUNDS: dict = {}  # path -> (audio, measurements), loaded once per worker


def _setup(profile: str | None) -> tuple:
    """Load the probe signal once per worker process."""
    global _CORE, _TX
    if _CORE is None:
        _CORE = ad.load_core(profile)
        n = int(_CORE.fs * (LEAD_IN + STORY))
        pre = ad.pre_roll(_CORE)
        _TX = (_CORE.tx(pre, n), pre, n)
    return _CORE, _TX


def label_segments(stream: ad.Stream, events: list, core: ad.Core | None = None) -> list:
    """Give every movement burst a label.

    stroke : the burst that covers at least half of the scripted stroke -> the gesture label
    aux    : other bursts within 1.2 s of the stroke (hand raise before, hand dropping back after) -> "none"
    burst  : everything else that moved (fidgeting, bystanders, clicks) -> "none"
    With `core` given, each entry also carries the padded training picture in the core's dataset format.
    """
    out = []
    for k, (seg, win) in enumerate(zip(stream.segments, stream.windows)):
        if win is None:
            continue
        t0, t1 = seg.start / stream.frame_rate, (seg.end + 1) / stream.frame_rate
        label, kind = "none", "burst"
        for ev in events:
            if ev["label"] not in GESTURE4:
                continue
            overlap = min(t1, ev["t_end"]) - max(t0, ev["t_start"])
            if overlap >= 0.5 * (ev["t_end"] - ev["t_start"]):
                label, kind = ev["label"], "stroke"
            elif kind != "stroke" and t1 > ev["t_start"] - 1.2 and t0 < ev["t_end"] + 1.2:
                kind = "aux"
        item = {"label": label, "window": win, "t_start": t0, "t_end": t1, "is_stroke": kind == "stroke", "kind": kind,
                "peak_db": float(getattr(seg, "peak_db", 0.0))}
        if core is not None:
            item["padded"] = ad.padded_window(core, stream, k=k)
        out.append(item)
    return out


def idle_example(rng: np.random.Generator, stream: ad.Stream, core: ad.Core, events: list) -> dict | None:
    """One picture of "nothing happening" (the core's dataset builder adds these too)."""
    st = stream.core
    if st is None:
        return None
    pad = ad.train_pad(core)
    quiet = np.nonzero(st.ready & (st.activity_db < 1.0))[0] + 1
    quiet = quiet[(quiet > core.cfg.gesture_frames // 2) & (quiet < st.n_frames - pad)]
    for ev in events:  # stay clear of scripted movement
        quiet = quiet[(quiet / stream.frame_rate < ev["t_start"] - 1.0) | (quiet / stream.frame_rate > ev["t_end"] + 2.0)]
    if len(quiet) == 0:
        return None
    q = int(rng.choice(quiet))
    return {"label": "none", "window": ad.window_at(core, stream, q), "padded": ad.padded_window(core, stream, end_frame=q),
            "t_start": q / stream.frame_rate, "t_end": q / stream.frame_rate, "is_stroke": False, "kind": "idle", "peak_db": 0.0}


def _background(path: str, core: ad.Core) -> tuple:
    if path not in _BACKGROUNDS:
        audio = bg.load_audio(path)
        _BACKGROUNDS[path] = (audio, bg.measure(audio, core))
    return _BACKGROUNDS[path]


def make_clip(job: tuple) -> dict:
    """Worker: one random clip -> labelled model inputs (+ optional raw audio)."""
    seed, scene_label, profile, min_snr, keep_audio = job[:5]
    backgrounds, bg_share = (job[5], job[6]) if len(job) > 5 else ((), 0.0)
    swap_mics, swap_speakers = (job[7], job[8]) if len(job) > 8 else (False, False)
    mics = job[9] if len(job) > 9 else None
    core, (tx, pre, n) = _setup(profile)
    hw_profile = Profile(band_hz=core.band, swap_mics=swap_mics, swap_speakers=swap_speakers)
    rng = np.random.default_rng(seed)
    use_real = bool(backgrounds) and rng.random() < bg_share
    # a gesture that is physically inaudible should not be labelled as a gesture: redraw the room
    for attempt in range(4):
        centre = STORY / 2 + rng.uniform(-0.35, 0.35) if scene_label in GESTURE4 + ("wave",) else None
        scene = sample_scene(rng, scene_label, STORY, event_center=centre, lead_in=LEAD_IN, distractors=not use_real)
        if use_real:  # real laptop + room + noise, synthetic moving scene on top
            audio, measured = _background(str(rng.choice(list(backgrounds))), core)
            rx, info = bg.inject(bg.random_excerpt(rng, audio, n, core.period), core, scene, rng, measured=measured,
                                 profile=hw_profile)
            hw, snr = info["hardware"], info["echo_snr_db"]
        else:
            hw = sample_hardware(rng, hw_profile, core.fs)
            if core.n_speakers >= 2:
                hw.tx_route = "stereo"
            out = render(tx, hw, scene, n, rng, pre=pre, band=core.band)
            rx, snr = out.rx, out.echo_snr_db
        if mics:  # the real laptop has fewer microphones (a MacBook records one channel): keep the left one
            rx = rx[:, :mics]
        if scene_label not in GESTURE4 or not np.isfinite(snr) or snr >= min_snr:
            break
    meta = {
        "seed": int(seed),
        "scene": scene_label,
        "events": scene.events,
        "echo_snr_db": None if not np.isfinite(snr) else round(float(snr), 1),
        "real_background": use_real,
        "n_mics": int(rx.shape[1]),
        "hardware": hw.summary(),
        "scene_meta": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in scene.meta.items()
                       if not isinstance(v, np.ndarray)},
    }
    result = {"meta": meta, "examples": [], "audio": rx if keep_audio else None, "error": None}
    if core.available and (core.module("pipeline") is not None or core.module("gestures") is not None):
        try:
            stream = ad.stream_features(core, rx)
            result["examples"] = label_segments(stream, scene.events, core)
            if rng.random() < 0.2:
                idle = idle_example(rng, stream, core, scene.events)
                if idle is not None:
                    result["examples"].append(idle)
        except Exception as exc:  # the core API is still changing: report it, keep the audio usable
            result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def default_workers() -> int:
    """Every CPU core (Windows process pools allow at most 61 workers)."""
    n = os.cpu_count() or 1
    return min(n, 61) if sys.platform == "win32" else n


def cache_backgrounds(paths: tuple, out_dir: Path) -> tuple:
    """Convert each background recording once to float32 .npy (memory-mapped by every worker)."""
    folder = out_dir / "backgrounds"
    folder.mkdir(parents=True, exist_ok=True)
    cached = []
    for i, path in enumerate(paths):
        target = folder / f"{i:03d}_{Path(path).parent.name or Path(path).stem}.npy"
        np.save(target, np.ascontiguousarray(bg.load_audio(path), dtype=np.float32))
        cached.append(str(target))
    return tuple(cached)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic gesture training data (see the file header).")
    ap.add_argument("--out", required=True, help="output folder (created if missing)")
    ap.add_argument("--per-class", type=int, default=300, help="clips per gesture class")
    ap.add_argument("--negatives", type=float, default=2.0, help="negative-only clips, as a multiple of --per-class")
    ap.add_argument("--workers", type=int, default=default_workers(), help="parallel processes (default: every CPU core)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-snr", type=float, default=6.0, help="redraw gesture clips whose echo is fainter than this (dB in 25 Hz)")
    ap.add_argument("--save-audio", type=int, default=0, help="keep raw audio of the first K clips of each class")
    ap.add_argument("--profile", default=None, help="core signal profile (default / low / mono / split)")
    ap.add_argument("--background", default="", help="comma-separated REAL no-gesture recordings (.wav/.flac/.npy) to build on")
    ap.add_argument("--background-share", type=float, default=1.0, help="share of clips built on a real background (when given)")
    ap.add_argument("--swap-mics", action="store_true", help="the real laptop's microphone channels are swapped (left = channel 1)")
    ap.add_argument("--swap-speakers", action="store_true", help="the real laptop's speaker channels are swapped")
    ap.add_argument("--mics", type=int, default=None, help="microphone channels of the real laptop (default: the simulator's 2)")
    args = ap.parse_args()
    backgrounds = tuple(p for p in args.background.split(",") if p)

    out_dir = Path(args.out)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    if backgrounds:
        backgrounds = cache_backgrounds(backgrounds, out_dir)
        print(f"{len(backgrounds)} background recording(s), memory-mapped from {out_dir / 'backgrounds'}")
    core, _ = _setup(args.profile)
    print(core.describe())
    labels = list(core.gesture_labels)

    # the list of clips to make: each gesture class, plus negative-only clips in a fixed mix
    rng = np.random.default_rng(args.seed)
    kinds = [g for g in GESTURE4 for _ in range(args.per_class)]
    neg_names, neg_p = list(NEGATIVE_MIX), np.array(list(NEGATIVE_MIX.values()))
    kinds += list(rng.choice(neg_names, size=int(args.per_class * args.negatives), p=neg_p / neg_p.sum()))
    seen: dict = {}
    jobs = []
    for i, kind in enumerate(kinds):
        seen[kind] = seen.get(kind, 0) + 1
        jobs.append((args.seed * 1_000_003 + i, str(kind), args.profile, args.min_snr, seen[kind] <= args.save_audio,
                     backgrounds, args.background_share, args.swap_mics, args.swap_speakers, args.mics))

    t0 = time.time()
    print(f"{len(jobs)} clips on {args.workers} worker processes")
    xs, ys, groups, peaks, n_clips, n_missed, errors = [], [], [], [], 0, 0, 0
    mic_counts: set = set()
    with ProcessPoolExecutor(args.workers) as pool, open(out_dir / "manifest.jsonl", "w") as manifest:
        for res in pool.map(make_clip, jobs, chunksize=4):
            n_clips += 1
            meta = res["meta"]
            mic_counts.add(meta["n_mics"])
            if res["error"]:
                errors += 1
                if errors == 1:
                    print("WARNING: core feature chain failed:", res["error"])
            if res["audio"] is not None:
                import soundfile as sf

                name = f"audio/{meta['scene']}_{meta['seed']}.flac"
                sf.write(out_dir / name, res["audio"], core.fs, subtype="PCM_24")
                meta["audio"] = name
            found = False
            for ex in res["examples"]:
                xs.append(np.asarray(ex.get("padded", ex["window"]), dtype=np.float16))
                ys.append(labels.index(ad.GESTURE_MAP.get(ex["label"], "none")))
                groups.append(f"synth-{meta['seed']}")
                peaks.append(ex["peak_db"])
                found |= ex["is_stroke"]
                manifest.write(json.dumps({"index": len(xs) - 1, "label": ex["label"], "kind": ex["kind"],
                                           "t_start": round(ex["t_start"], 3), "t_end": round(ex["t_end"], 3), **meta}) + "\n")
            if meta["scene"] in GESTURE4 and not found:
                n_missed += 1
                manifest.write(json.dumps({"index": None, "label": meta["scene"], "missed": True, **meta}) + "\n")
            if n_clips % 200 == 0:
                print(f"  {n_clips}/{len(jobs)} clips, {len(xs)} examples, {time.time() - t0:.0f} s")

    y = np.array(ys, dtype=np.int64)
    cfg = core.cfg.to_dict() if core.cfg is not None and hasattr(core.cfg, "to_dict") else {"tones": core.tones}
    if len(mic_counts) > 1:
        raise SystemExit(f"clips came out with different microphone counts {sorted(mic_counts)}: give --mics")
    if mic_counts and "n_mics" in cfg:
        cfg["n_mics"] = int(mic_counts.pop())  # the pictures were made from this many channels; train.py reads it here
    if xs:
        # exactly the file layout of sonar/dataset.py, so `python -m sonar.train --dataset <this file>` works.
        # Every clip is its own "session" and "person"; source = "synthetic" keeps it out of validation there.
        np.savez_compressed(
            out_dir / "dataset.npz",
            X=np.stack(xs),
            y=y,
            labels=np.array(labels),
            group=np.array(groups),
            person=np.array(groups),
            source=np.array(["synthetic"] * len(xs)),
            peak_db=np.array(peaks, dtype=np.float32),
            pad=np.array(ad.train_pad(core)),
            config=np.array(json.dumps(cfg, sort_keys=True)),
        )
    (out_dir / "config.json").write_text(json.dumps({"sonar_config": cfg, "args": vars(args), "lead_in_s": LEAD_IN,
                                                     "story_s": STORY, "labels": labels}, indent=2))
    n_gesture_clips = 4 * args.per_class
    print(f"done in {time.time() - t0:.0f} s: {n_clips} clips -> {len(xs)} examples in {out_dir}")
    for i, name in enumerate(labels):
        print(f"  {name:12s} {int((y == i).sum())}")
    print(f"  gesture strokes the core segmenter did not pick up: {n_missed} of {n_gesture_clips}")
    if errors:
        print(f"  clips where the core feature chain raised an error: {errors} (audio is still valid)")


if __name__ == "__main__":
    main()

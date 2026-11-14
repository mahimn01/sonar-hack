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


def label_segments(stream: ad.Stream, events: list, core: ad.Core | None = None, drop_partial: bool = False) -> list:
    """Give every movement burst a label.

    stroke : the burst that covers at least half of the scripted stroke -> the gesture label
    aux    : other bursts within 1.2 s of the stroke (hand raise before, hand dropping back after) -> "none"
    burst  : everything else that moved (fidgeting, bystanders, clicks) -> "none"
    With `core` given, each entry also carries the padded training picture in the core's dataset format.
    `drop_partial`: a burst that holds a PIECE of the stroke, but less than half of it, is left out instead of
    being called "none". (The engine cuts a burst after 1.4 s; with a fitted laptop the hand's way to the start is
    loud enough to open the burst early, so that cut can fall inside the stroke. Half a gesture is not "none".)
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
            elif drop_partial and overlap > 0.0 and kind != "stroke":
                kind = "cut"
            elif kind not in ("stroke", "cut") and t1 > ev["t_start"] - 1.2 and t0 < ev["t_end"] + 1.2:
                kind = "aux"
        if kind == "cut":
            continue
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


MAX_ATTEMPTS = 4  # a gesture too faint to see is redrawn (new room, new person) up to this many times


def unpack_job(job: tuple) -> dict:
    """Name the fields of one job tuple (shorter tuples from older callers still work)."""
    backgrounds, bg_share = (job[5], job[6]) if len(job) > 5 else ((), 0.0)
    swap_mics, swap_speakers = (job[7], job[8]) if len(job) > 8 else (False, False)
    return {"seed": job[0], "scene": job[1], "profile": job[2], "min_snr": job[3], "keep_audio": job[4],
            "backgrounds": backgrounds, "bg_share": bg_share, "swap_mics": swap_mics, "swap_speakers": swap_speakers,
            "mics": job[9] if len(job) > 9 else None,
            "tweak": job[10] if len(job) > 10 else None,  # fixed overrides, only used by the parity fixture
            "motion": job[11] if len(job) > 11 else None,  # file with real tracked hand paths (motion.py), or None
            "motion_share": job[12] if len(job) > 12 else 0.0,
            "calibration": job[13] if len(job) > 13 else None}  # file with a laptop fitted to real recordings, or None


def draw_world(rng: np.random.Generator, core: ad.Core, j: dict, hw_profile: Profile, use_real: bool, n: int) -> tuple:
    """One attempt at a clip: a random story and a random laptop (+ a piece of a real recording).

    Returns (scene, hardware, real excerpt or None, measurements of the real recording or None).
    The CPU and the GPU path both call this, so both make the same draws in the same order.
    """
    from . import calibration

    calibration.activate(j["calibration"])  # None = the generic laptop of hardware.py (the default)
    centre = STORY / 2 + rng.uniform(-0.35, 0.35) if j["scene"] in GESTURE4 + ("wave",) else None
    motion = None
    if j["motion"] and j["motion_share"] > 0:  # real hand paths for a share of the gestures (off: formulas only)
        from .motion import load_library

        motion = (load_library(j["motion"]), j["motion_share"])
    scene = sample_scene(rng, j["scene"], STORY, event_center=centre, lead_in=LEAD_IN, distractors=not use_real, motion=motion)
    excerpt = measured = None
    if use_real:  # real laptop + room + noise, synthetic moving scene on top
        audio, measured = _background(str(rng.choice(list(j["backgrounds"]))), core)
        excerpt = bg.random_excerpt(rng, audio, n, core.period)
        hw = bg.matched_hardware(rng, core, measured, hw_profile)
    else:
        hw = sample_hardware(rng, hw_profile, core.fs)
        if core.n_speakers >= 2:
            hw.tx_route = "stereo"
    for name, value in ((j["tweak"] or {}).get("hw") or {}).items():
        setattr(hw, name, value)
    return scene, hw, excerpt, measured


def accept(scene_label: str, snr: float, min_snr: float) -> bool:
    """A gesture that is physically inaudible should not be labelled as a gesture: False = redraw the room."""
    return scene_label not in GESTURE4 or not np.isfinite(snr) or snr >= min_snr


def clip_meta(j: dict, scene, hw, snr: float, use_real: bool, n_mics: int) -> dict:
    return {
        "seed": int(j["seed"]),
        "scene": j["scene"],
        "events": scene.events,
        "echo_snr_db": None if not np.isfinite(snr) else round(float(snr), 1),
        "real_background": use_real,
        "n_mics": int(n_mics),
        "hardware": hw.summary(),
        "scene_meta": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in scene.meta.items()
                       if not isinstance(v, np.ndarray)},
    }


def make_clip(job: tuple, capture: dict | None = None) -> dict:
    """Worker: one random clip -> labelled model inputs (+ optional raw audio). The CPU reference path.

    `capture`, if given, receives the intermediate results (audio, its parts, the feature stream):
    scripts/check_gpu_parity.py compares them with the GPU path.
    """
    j = unpack_job(job)
    core, (tx, pre, n) = _setup(j["profile"])
    hw_profile = Profile(band_hz=core.band, swap_mics=j["swap_mics"], swap_speakers=j["swap_speakers"])
    rng = np.random.default_rng(j["seed"])
    use_real = bool(j["backgrounds"]) and rng.random() < j["bg_share"]
    for attempt in range(MAX_ATTEMPTS):
        scene, hw, excerpt, measured = draw_world(rng, core, j, hw_profile, use_real, n)
        parts = None
        if use_real:
            rx, info = bg.inject(excerpt, core, scene, rng, measured=measured, hw=hw)
            snr = info["echo_snr_db"]
        else:
            out = render(tx, hw, scene, n, rng, pre=pre, band=core.band)
            rx, snr, parts = out.rx, out.echo_snr_db, out.parts
        if j["mics"]:  # the real laptop has fewer microphones (a MacBook records one channel): keep the left one
            rx = rx[:, :j["mics"]]
        if accept(j["scene"], snr, j["min_snr"]):
            break
    meta = clip_meta(j, scene, hw, snr, use_real, rx.shape[1])
    result = {"meta": meta, "examples": [], "audio": rx if j["keep_audio"] else None, "error": None}
    stream = None
    if core.available and (core.module("pipeline") is not None or core.module("gestures") is not None):
        try:
            stream = ad.stream_features(core, rx)
            result["examples"] = label_segments(stream, scene.events, core, drop_partial=bool(j["calibration"]))
            if rng.random() < 0.2:
                idle = idle_example(rng, stream, core, scene.events)
                if idle is not None:
                    result["examples"].append(idle)
        except Exception as exc:  # the core API is still changing: report it, keep the audio usable
            result["error"] = f"{type(exc).__name__}: {exc}"
    if capture is not None:
        capture.update(rx=rx, parts=parts, snr=snr, attempts=attempt + 1, stream=stream, hw=hw, scene=scene)
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
    ap.add_argument("--calibration", default=None, help="laptop fitted to real recordings (.json from datagen.calibration); "
                                                        "default: none, the generic laptop of hardware.py")
    ap.add_argument("--motion", default=None, help="library of real tracked hand paths (.npz from datagen.mocap_extract); "
                                                   "default: none, every gesture comes from the formulas in kinematics.py")
    ap.add_argument("--motion-share", type=float, default=0.5, help="share of gesture clips that replay a real hand path (with --motion)")
    ap.add_argument("--gen-device", default="cpu", choices=["cpu", "cuda", "mps", "torch-cpu"],
                    help="where the clips are computed: cpu = the numpy reference on every core (default); cuda / mps = "
                         "the same clips on the GPU (gpu_pipeline.py); torch-cpu = the GPU code on the CPU (tests)")
    ap.add_argument("--gen-precision", default="auto", choices=["auto", "fast", "ref", "single"],
                    help="GPU number format: ref = float64 where the CPU code uses it (default), fast = float32 FFTs, "
                         "single = all float32 (always on MPS)")
    ap.add_argument("--gen-batch", type=int, default=0, help="clips per GPU batch (default: fills the GPU's memory)")
    ap.add_argument("--gen-gpus", type=int, default=0, help="with --gen-device cuda: how many GPUs to use (default: all of them)")
    ap.add_argument("--gen-kernel", default="auto", choices=["auto", "on", "off"],
                    help="echo loop on CUDA: auto = the fused kernel (echo_paths.c) if its self-check passes on this GPU, "
                         "else PyTorch operations; off = PyTorch operations")
    ap.add_argument("--gen-features", default="conv", choices=["conv", "fft"], help="long filter in front of the Doppler pictures: conv = a strided convolution (default); fft = the same sum through FFTs (much less GPU arithmetic, same numbers to 1e-15)")
    ap.add_argument("--gen-noise", default="cpu", choices=["cpu", "gpu"],
                    help="hall noise and gain wander: cpu = shaped in the worker processes, as render.py does (default); gpu = the workers only draw the random numbers and the GPU shapes them (about half the CPU work per clip; same audio to 1e-12)")
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
                     backgrounds, args.background_share, args.swap_mics, args.swap_speakers, args.mics)
                    + ((None, str(Path(args.motion).resolve()) if args.motion else None, args.motion_share if args.motion else 0.0,
                         str(Path(args.calibration).resolve()) if args.calibration else None)
                        if (args.motion or args.calibration) else ()))

    t0 = time.time()
    xs, ys, groups, peaks, n_clips, n_missed, errors = [], [], [], [], 0, 0, 0
    mic_counts: set = set()
    gpu_stats: dict = {}
    pool = None
    if args.gen_device == "cpu":
        print(f"{len(jobs)} clips on {args.workers} worker processes")
        pool = ProcessPoolExecutor(args.workers)
        results = pool.map(make_clip, jobs, chunksize=4)
    else:  # the same clips, computed on the GPU (see gpu_pipeline.py)
        from .gpu_pipeline import generate_gpu

        device = "cpu" if args.gen_device == "torch-cpu" else args.gen_device
        print(f"{len(jobs)} clips on {device} (PyTorch), {args.workers} worker processes for the random draws and the labels")
        results = generate_gpu(jobs, args.profile, device, args.workers, precision=args.gen_precision, batch=args.gen_batch,
                               stats=gpu_stats, gpus=args.gen_gpus, kernel=args.gen_kernel, noise=args.gen_noise,
                               features=args.gen_features)
    with open(out_dir / "manifest.jsonl", "w") as manifest:
        for res in results:
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

    if pool is not None:
        pool.shutdown()
    if gpu_stats:
        print(f"  echo loop: {gpu_stats['echo_loop']}" + (f" ({gpu_stats['kernel_note']})" if gpu_stats.get("kernel_note") else "")
              + f"; plans from the workers: {gpu_stats.get('plan_transport')}; hall noise shaped {gpu_stats.get('noise_shaping')}; "
              f"feature filter: {gpu_stats.get('feature_filter')}")
        print(f"  GPU path on {gpu_stats['device']}: batch {gpu_stats['batch']}, precision {gpu_stats['precision']}, {gpu_stats['batches']} batches, "
              f"render {gpu_stats['render_s']:.1f} s, features {gpu_stats['features_s']:.1f} s, {gpu_stats['redraws']} redraws"
              + (f"; {gpu_stats['oom_splits']} batch(es) did not fit in GPU memory and were split: use --gen-batch "
                 f"{max(8, gpu_stats['batch'] // 2)}" if gpu_stats.get("oom_splits") else ""))
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

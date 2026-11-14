"""calibration.py — make the simulator sound like ONE real laptop (fitted to its recordings).

Plain words
-----------
hardware.py draws a different imaginary laptop for every clip, around a guess of a 15-inch
HP. The recordings of the demo MacBook show that guess is wrong for it in ways that matter:
its one microphone sits on the deck next to the left speaker (not on top of the lid), the
speaker tones arrive 25-60 dB louder, hands pass 10-25 cm above the deck, and people swipe
up to five times faster than the formulas assume. A model trained on the wrong laptop
recognised only 0.30 of a stranger's real gestures.

This file
  1. measures what can be read straight off the recordings (`measure_real`): tone levels,
     noise floor, and, for the labelled gestures, summary numbers of the pictures the engine
     makes of them (how strong, how fast, which tone carries the direction, how long);
  2. computes the same numbers for simulated clips (`sim_stats`) and searches for the
     simulator settings that bring them together (`fit`), one setting at a time;
  3. writes the result to a JSON file and applies it (`activate`): hardware.py and scenes.py
     ask this module for the fitted ranges when one is active, and draw around them. Clips whose
     movement is laid over a real recording (background.py) get the "overlay" part of the file.

It fits by comparing SUMMARY NUMBERS of many clips, not by gradient descent through the
renderer on single clips: for a real gesture the true path of the hand is unknown, so there
is nothing to line a single simulated clip up with.

When the recordings include sessions with nobody gesturing (scene, breathing), the trial clips
are laid over those recordings, as most training clips are. Reason, measured: next to the tones
the real hall is 10-25 dB louder at low Doppler (people moving about) than anything the pure
simulation has, and the engine scales every picture by that floor. Settings fitted on pure
simulation therefore come out wrong for the clips that matter. Only the echo level of clips
WITHOUT a recording underneath is then fitted on pure simulation, as a second step.

Two kinds of facts are fitted, and they are kept apart on purpose:
  - DEVICE facts (where the microphone and speakers sit, how loud tones, echoes and noise arrive):
    the same whoever sits in front of the laptop, so they are fitted on every recording.
  - PERSON facts (how high people hold the hand, how fast they move): these differ from human to
    human. A score that tests the model on one held-out human is only honest if the synthetic
    training clips were not drawn around THAT human's habits. `--folds` therefore writes one
    extra file per human, `<name>_no_<human>.json`, whose person facts were chosen by looking
    only at the other humans (lance and lance2 count as one human). The plain file (everyone)
    is for the final model that ships.

How to run (reads the recordings; writes only the JSON files and scratch files in a `work` folder next to them):
    python -m datagen.calibration --real ~/Documents/Dev/sonar-hack-recall/mac/data --out calibration/macbook_v1.json
    python -m datagen.calibration --real data --folds calibration/macbook_v1.json --device cuda --clips 100
`--device cuda` renders the trial clips on the GPU (gpu_pipeline.py); the default is the numpy
reference on the CPU cores. Both are heavy (thousands of trial clips): run them on the GPU box.
Use:  python -m datagen.generate --calibration calibration/macbook_v1.json --mics 1 ...
Without --calibration nothing in the generator changes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ACTIVE: dict | None = None        # the calibration in force in this process (None = the generic laptop)
_LOADED: dict = {}

# What the search starts from: physics plus the recall session's measurements on the MacBook.
START = {
    "hardware": {
        "mic_pos": [-0.13, 0.08, 0.004],     # on the deck, by the left speaker grille (m; x right, y towards the user, z up)
        "mic_jitter": [0.02, 0.03, 0.002],   # how far the draw strays from it (standard deviation, m)
        "spk_x": [0.12, 0.15], "spk_y": [0.06, 0.14],        # MacBook speakers: beside the keyboard, pointing up
        "spk_tilt_deg": [0.0, 25.0],
        "left_minus_right_db": 6.0,          # the left tone arrives this much louder at the microphone
        "direct_gain_db": [-20.0, -12.0],    # tone level at the microphone / tone level played
        "echo_ref_db": [-26.0, -10.0],       # echo of a palm at 30 cm relative to that
        "noise_dbfs": [-92.0, -84.0],
    },
    "scene": {
        "hand_height": [0.10, 0.25],         # the hand's height above the deck during a gesture (m)
        "peak_speed": [0.9, 5.5],            # fastest moment of a stroke (m/s), drawn evenly on a log scale
    },
}


# --------------------------------------------------------------------------
# Applying a calibration (called by hardware.py and scenes.py; no-ops when none is active)
# --------------------------------------------------------------------------
def activate(path: str | None) -> None:
    """Make the calibration in `path` the one this process draws around (None: the generic laptop)."""
    global ACTIVE
    if not path:
        ACTIVE = None
        return
    if path not in _LOADED:
        _LOADED[path] = json.loads(Path(path).read_text())
    ACTIVE = _LOADED[path]


def adjust_hardware(rng: np.random.Generator, hw) -> None:
    """Move a freshly drawn virtual laptop onto the fitted one (geometry and levels), with a little spread."""
    if ACTIVE is None:
        return
    h = ACTIVE["hardware"]
    c = hw.c
    sx, sy = rng.uniform(*h["spk_x"]), rng.uniform(*h["spk_y"])
    hw.speakers = np.array([[-sx, sy, 0.004], [sx, sy + rng.normal(0, 0.004), 0.004]])
    tilt = np.deg2rad(rng.uniform(*h["spk_tilt_deg"]))
    hw.spk_axis = np.array([[0.0, np.sin(tilt), np.cos(tilt)]] * 2)
    old = [[np.linalg.norm(hw.speakers[s] - hw.mics[m]) / c for s in range(2)] for m in range(hw.n_mic)]
    pos = np.asarray(h["mic_pos"]) + rng.normal(0.0, 1.0, 3) * np.asarray(h["mic_jitter"])
    pos[2] = max(pos[2], 0.002)
    hw.mics = np.array([pos + [-0.004, 0, 0], pos + [0.004, 0, 0]])       # one microphone: both "channels" at the same spot
    hw.mic_axis = np.array([0.0, 0.0, 1.0])                               # a hole in the deck: it listens upwards
    hw.mic_n, hw.mic_floor = rng.uniform(0.0, 1.0), rng.uniform(0.3, 0.7)
    hw.table_refl = 0.0                                                   # the microphone is IN the desk plane: no second path
    lr = h.get("left_minus_right_db", 0.0)
    for m in range(hw.n_mic):
        for s in range(2):
            d_new = np.linalg.norm(hw.speakers[s] - hw.mics[m]) / c
            hw.static_delays[m][s] = np.asarray(hw.static_delays[m][s]) + (d_new - hw.static_delays[m][s][0])
            hw.static_amps[m][s] = np.asarray(hw.static_amps[m][s], float).copy()
            hw.static_amps[m][s][0] *= 10.0 ** ((0.5 if s == 0 else -0.5) * lr / 20.0)
    del old
    hw.direct_gain_db = float(rng.uniform(*h["direct_gain_db"]))
    hw.echo_ref_db = float(rng.uniform(*h["echo_ref_db"]))
    hw.noise_dbfs = float(rng.uniform(*h["noise_dbfs"]))


def adjust_overlay(rng: np.random.Generator, hw, speaker_levels: list) -> None:
    """A virtual laptop that was matched to a REAL recording (background.matched_hardware): the fitted corrections for it.

    Used for clips whose movement is laid over a real recording. `speaker_levels` = how loud each speaker's tone
    arrives at the microphone in that recording (dB), which matched_hardware took as the level of that speaker's ECHOES.
    """
    o = (ACTIVE or {}).get("overlay")
    if not o:
        return
    if o.get("equal_speakers") and len(speaker_levels) >= 2:
        # On this laptop the left tone is louder at the microphone only because the microphone sits next to the
        # left speaker. A hand above the keys is about as far from one speaker as from the other, so both echoes
        # get the same level (the average), instead of the left one being louder by that difference.
        mean = float(np.mean(speaker_levels))
        for s, level in enumerate(speaker_levels):
            hw.spk_resp[s].gain_db += mean - level
    if o.get("echo_ref_db"):
        # the real recording brings its real noise floor, which is higher than the simulated one the plain
        # echo level was fitted against: the echo level for these clips has its own range
        hw.echo_ref_db = float(rng.uniform(*o["echo_ref_db"]))


def hand_height(rng: np.random.Generator, default: float) -> float:
    """Height of the gesture above the deck: the fitted range if a calibration is active."""
    return default if ACTIVE is None else float(rng.uniform(*ACTIVE["scene"]["hand_height"]))


def min_stroke_seconds(rng: np.random.Generator, travel: float) -> float:
    """The shortest time a hand needs for a stroke of `travel` metres: muscles limit how hard it can be accelerated.

    A smooth (minimum-jerk) stroke of length D in time T peaks at an acceleration of 5.77 x D / T^2. People reach
    about 30-150 m/s^2 with the hand (3-15 g, the top end only in a flick; lance's recorded swipes show Doppler
    300-900 Hz off the tone, i.e. 3 m/s and more), so a short stroke cannot be as fast as a long sweep:
    20 cm takes at least 0.09-0.20 s (2-4.3 m/s at the peak), 50 cm at least 0.14-0.31 s (3-6.8 m/s).
    """
    a_max = float(np.exp(rng.uniform(np.log(30.0), np.log(150.0))))
    return float(max(0.08, np.sqrt(5.77 * travel / a_max)))


def stroke_seconds(rng: np.random.Generator, travel: float, default: float) -> float:
    """How long a stroke of `travel` metres takes: from the fitted peak-speed range if a calibration is active."""
    if ACTIVE is None:
        return default
    lo, hi = ACTIVE["scene"]["peak_speed"]
    speed = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    # minimum-jerk: peak = 1.875 x average speed; never faster than a hand can be accelerated
    return float(np.clip(1.875 * travel / speed, min_stroke_seconds(rng, travel), 1.4))


def lead_in(prep: float, gap: float) -> tuple:
    """How long the hand takes to get to the start of the stroke, and how long it waits there (seconds).

    In the recordings people were prompted: the hand goes to the start and strokes almost in one movement. The
    whole thing must also fit the engine's longest burst (1.4 s): a slow wind-up plus a long wait would have the
    burst cut off before the stroke. No calibration active: unchanged.
    """
    return (prep, gap) if ACTIVE is None else (min(prep, 0.7), min(gap, 0.15))


def speed_range(default: tuple) -> tuple:
    return default if ACTIVE is None else tuple(ACTIVE["scene"]["peak_speed"])


# --------------------------------------------------------------------------
# Summary numbers of gesture pictures (the same code for real and simulated ones)
# --------------------------------------------------------------------------
KEYS = ("peak_db", "extent_hz", "saturated", "filled", "left_tone_hz", "right_tone_hz")
SCALE = {"peak_db": 6.0, "extent_hz": 60.0, "saturated": 0.05, "filled": 0.15, "left_tone_hz": 40.0, "right_tone_hz": 40.0}


def picture_stats(x: np.ndarray, peak_db: float, hz: np.ndarray) -> dict:
    """x [2, F, T] one-microphone picture (tone left speaker, tone right speaker), values 0..1."""
    x = np.asarray(x, np.float32)
    energy = x.sum(axis=(0, 1))
    strong = np.argsort(energy)[-15:]                                    # the strongest 0.3 s
    out = {"peak_db": float(peak_db), "saturated": float((x >= 0.99).mean()), "filled": float((x.max(axis=(0, 1)) > 0.3).mean())}
    on = np.abs(hz)[None, :, None] * (x > 0.5)
    out["extent_hz"] = float(np.percentile(on[on > 0], 95)) if (on > 0).any() else 0.0
    for name, tone in (("left_tone_hz", 0), ("right_tone_hz", 1)):
        w = np.maximum(x[tone][:, strong] - 0.3 * x[tone].max(), 0.0) * (np.abs(hz) >= 15.0)[:, None]
        out[name] = float((w * hz[:, None]).sum() / (w.sum() + 1e-9))
    return out


def summarise(rows: list) -> dict:
    """rows: (label, stats). Per class the mean of every number; swipes share speed/strength, differ in the tone signs."""
    out = {}
    for label in ("swipe_left", "swipe_right", "push"):
        sel = [s for lab, s in rows if lab == label]
        if sel:
            out[label] = {k: float(np.mean([s[k] for s in sel])) for k in KEYS}
            out[label]["n"] = len(sel)
    return out


def gap(sim: dict, real: dict) -> float:
    """One number for how far simulated pictures are from real ones (0 = same means; 1 = one 'scale' apart on average)."""
    terms = [((sim[c][k] - real[c][k]) / SCALE[k]) ** 2 for c in real if c in sim for k in KEYS]
    return float(np.sqrt(np.mean(terms))) if terms else float("inf")


def usable_sessions(root: Path) -> list:
    """The session folders under `root`, without those somebody set aside in a folder named 'excluded'
    (e.g. mac/excluded/<session>: a recording whose left and right turned out flipped)."""
    from sonar.dataset import find_sessions

    root = Path(root)
    return [f for f in find_sessions(root) if "excluded" not in {part.lower() for part in f.relative_to(root).parts}]


def real_stats(root: Path, log=print) -> tuple[dict, dict, dict]:
    """Cut the real gesture sessions with the engine's own code and summarise their gesture pictures.

    Returns (summary of everyone, summary per person, the single gestures per person as (label, numbers)).
    """
    from sonar.dataset import center_crop, load_session, session_examples

    rows, per_person = [], {}
    for folder in usable_sessions(root):
        meta, _, cfg = load_session(folder)
        if meta.get("script") != "gestures":
            continue
        examples, rep = session_examples(folder)
        for e in examples:
            if e.source == "gesture" and e.label in ("swipe_left", "swipe_right", "push"):
                st = picture_stats(center_crop(e.x, cfg), e.peak_db, cfg.doppler_hz)
                rows.append((e.label, st))
                per_person.setdefault(str(meta.get("person")), []).append((e.label, st))
        log(f"  {folder.name}: {rep['gestures']} gestures found, {rep['missed']} prompts without movement")
    return summarise(rows), {p: summarise(r) for p, r in per_person.items()}, per_person


def _clip_rows(res: dict) -> list:
    """(label, picture numbers) of the gesture strokes one generated clip produced."""
    from sonar.config import default_config
    from sonar.dataset import center_crop

    cfg = default_config().with_(n_mics=1)
    return [(ex["label"], picture_stats(center_crop(ex["padded"], cfg), ex["peak_db"], cfg.doppler_hz))
            for ex in res["examples"] if ex["is_stroke"]]


def _sim_clip(job: tuple) -> list:
    from . import generate as g

    return _clip_rows(g.make_clip(job))


class Sim:
    """Where the trial clips are made: the numpy reference on the CPU cores, or the GPU path (the same clips)."""

    def __init__(self, device: str = "cpu", workers: int = 8) -> None:
        self.device, self.workers = device, max(1, workers)
        self.pool = ProcessPoolExecutor(self.workers)
        self.fleet = None
        if device != "cpu":                       # "torch-cpu" = the GPU code on the CPU (tests), as in datagen.generate
            from .gpu_pipeline import GpuRenderer, gpu_devices

            self.torch_device = "cpu" if device == "torch-cpu" else device
            self.fleet = [GpuRenderer(None, d, "auto") for d in gpu_devices(self.torch_device, 0)]

    def clips(self, jobs: list) -> list:
        """The gesture strokes of all `jobs`, as (label, picture numbers)."""
        if self.fleet is None:
            return [r for part in self.pool.map(_sim_clip, jobs, chunksize=2) for r in part]
        from .gpu_pipeline import generate_gpu

        return [r for res in generate_gpu(jobs, None, self.torch_device, self.workers, gpu=self.fleet, pool=self.pool)
                for r in _clip_rows(res)]

    def close(self) -> None:
        self.pool.shutdown()

    def __enter__(self) -> "Sim":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def sim_stats(cal_path: str | None, n_per_class: int, sim: Sim, motion: str | None = None, seed: int = 0,
              backgrounds: tuple = ()) -> dict:
    """The same summary for simulated one-microphone clips drawn with a calibration file (None = the generic laptop).

    `backgrounds`: real "nobody gesturing" recordings; if given, every trial clip is laid over one of them.
    """
    jobs = []
    for kind in ("swipe_left", "swipe_right", "push"):
        for i in range(n_per_class):
            jobs.append((8_000_000 + seed * 10_007 + len(jobs), kind, None, 6.0, False, tuple(backgrounds), 1.0 if backgrounds else 0.0,
                         False, False, 1, None, motion, 0.5 if motion else 0.0, cal_path))
    rows = sim.clips(jobs)
    out = summarise(rows)
    out["strokes_found"] = len(rows) / max(len(jobs), 1)
    return out


def background_files(root: Path, work: Path, log=print) -> tuple:
    """The recordings with nobody gesturing (scene, breathing) under `root`, as float32 .npy copies in `work`
    that every worker process memory-maps (as datagen.generate does). Empty if there are none.

    Returns ((file, person), ...): the person matters for the per-human fits (a fold's trial clips are not laid
    over the held-out human's own recordings).
    """
    from sonar.dataset import SESSION_FILE

    from . import generate as g

    found = []
    for folder in usable_sessions(root):
        meta = json.loads((folder / SESSION_FILE).read_text())
        if meta.get("script") in ("scene", "breathing") and not meta.get("synthetic"):
            found.append((str(folder / meta.get("audio", "audio.wav")), str(meta.get("person", "unknown"))))
    if not found:
        return ()
    found.sort()
    log(f"  trial clips are laid over {len(found)} recording(s) with nobody gesturing")
    cached = g.cache_backgrounds(tuple(p for p, _ in found), work)
    return tuple((c, person) for c, (_, person) in zip(cached, found))


def tone_band_floor(x: np.ndarray, fs: int, tones: tuple, line_hz: float) -> float:
    """The noise floor around the speaker tones of a quiet recording, in dBFS per Hz (a full-scale sine = -3 dBFS).

    How: the spectrum of 2-second pieces (Hann window), the MEDIAN over the pieces (short events do not count),
    read 150-250 Hz away from every tone. Closer to a tone the hall's own movement shows (10-25 dB more at
    15-60 Hz); that far out it no longer does. The probe's own lines (every `line_hz`) are left out.
    """
    n = 2 * fs
    if len(x) < 3 * n:
        return float("nan")
    pieces = np.asarray(x[: len(x) // n * n], np.float64).reshape(-1, n)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(pieces * win, axis=1)) ** 2 / (win ** 2).sum() / fs * 2.0      # power per Hz, one-sided
    f = np.fft.rfftfreq(n, 1.0 / fs)
    med = np.median(spec, axis=0) / 0.69                                                    # median -> mean of noise power
    sel = np.zeros(len(f), bool)
    for t in tones:
        sel |= (np.abs(f - t) >= 150.0) & (np.abs(f - t) < 250.0)
    for t in tones:
        sel &= np.abs(f - t) >= 150.0                                                       # ... and not next to another tone
    sel &= np.abs(((f + line_hz / 2.0) % line_hz) - line_hz / 2.0) > 3.0
    return float(10.0 * np.log10(np.median(med[sel]) + 1e-30)) if sel.any() else float("nan")


def measure_levels(root: Path) -> dict:
    """Read straight off the 'nobody gesturing' recordings: tone levels at the microphone, in-band noise, clock."""
    from scipy import fft as sfft

    from sonar.dataset import load_session

    from . import adapter as ad
    from . import background as bg

    core = ad.load_core(None)
    out = []
    for folder in usable_sessions(root):
        meta, audio, cfg = load_session(folder)
        if meta.get("script") not in ("scene", "breathing"):
            continue
        m = bg.measure(audio, core)
        seg = audio[: int(20 * cfg.fs) // core.period * core.period, 0].astype(np.float64)
        spec = np.abs(sfft.rfft(seg)) ** 2
        f = sfft.rfftfreq(len(seg), 1.0 / cfg.fs)
        quiet = (f > core.band[0] - 400) & (f < core.band[1] + 400)
        for line in np.arange(0.0, cfg.fs / 2, cfg.fs / core.period):
            quiet &= np.abs(f - line) > 6.0
        density = 2.0 * float(np.median(spec[quiet])) / (len(seg) * cfg.fs) / 0.69             # power per Hz, full scale = 1
        floor = tone_band_floor(audio[:, 0], cfg.fs, tuple(core.tones), cfg.fs / core.period)
        out.append({"session": folder.name, "tone_db": [float(v) for v in m.tone_mic_db[:, 0]], "direct_gain_db": float(m.direct_gain_db),
                    "floor_dbfs_per_hz": floor, "floor_dbfs_if_white": floor + 10.0 * float(np.log10(cfg.fs / 2.0)),
                    "clock_ppm": float(m.clock_ppm), "noise_dbfs_per_hz": float(10 * np.log10(density + 1e-30)),
                    "noise_dbfs_if_white": float(10 * np.log10(density * cfg.fs / 2 + 1e-30)),
                    "peak": float(np.abs(audio).max())})
    floors = [s["floor_dbfs_if_white"] for s in out if np.isfinite(s["floor_dbfs_if_white"])]
    return {"sessions": out,
            "floor_dbfs_if_white_median": float(np.median(floors)) if floors else None,
            "tone_db_median": [float(np.median([s["tone_db"][k] for s in out])) for k in range(len(out[0]["tone_db"]))] if out else [],
            "between_line_power_dbfs_median": float(np.median([s["noise_dbfs_if_white"] for s in out])) if out else None}


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------
ECHO_CENTRES = (-22.0, -14.0, -10.0, -6.0, -2.0, 2.0, 6.0, 10.0)        # echo level ranges tried: centre +- 8 dB


def fit(real: dict, levels: dict, work: Path, n_per_class: int, workers: int, log=print, device: str = "cpu",
        start: dict | None = None, backgrounds: tuple = ()) -> tuple[dict, dict, dict, dict]:
    """Search the settings that bring simulated picture numbers to the real ones.

    Returns (calibration, generic laptop's numbers, fitted numbers, extra). `start` = an earlier calibration to start
    from (its hardware / scene / overlay parts); a setting is only changed when that brings the pictures closer, so
    the result is never further from the recordings than the start. `backgrounds` = real no-gesture recordings: the
    trial clips are then laid over them, and the echo level of clips without a recording is fitted afterwards
    (extra["pure"]).
    """
    cal = json.loads(json.dumps({k: (start or START)[k] for k in ("hardware", "scene", "overlay") if k in (start or START)}))
    backgrounds = tuple(b if isinstance(b, str) else b[0] for b in backgrounds)     # (file, person) pairs or plain files
    over = bool(backgrounds)
    if over:                                                             # clips over a recording have their own echo level
        cal.setdefault("overlay", {})
        cal["overlay"]["equal_speakers"] = True
        cal["overlay"].setdefault("echo_ref_db", list(cal["hardware"]["echo_ref_db"]))
        cal["overlay"].pop("note", None)
    if levels.get("tone_db_median"):                                     # levels are measured, not searched
        t = levels["tone_db_median"]
        mid = float(np.mean(t))
        cal["hardware"]["direct_gain_db"] = [round(mid - 3.0, 1), round(mid + 3.0, 1)]
        if len(t) >= 2:
            cal["hardware"]["left_minus_right_db"] = round(float(t[0] - t[1]), 1)
    if levels.get("floor_dbfs_if_white_median") is not None:             # the floor around the tones, as white noise over the whole band
        mid = levels["floor_dbfs_if_white_median"]
        cal["hardware"]["noise_dbfs"] = [round(mid - 3.0, 1), round(mid + 3.0, 1)]
    work.mkdir(parents=True, exist_ok=True)
    tried = [0]

    def score(c: dict, pool, bgs: tuple) -> tuple[float, dict]:
        tried[0] += 1
        trial = work / f"trial_{int(time.time())}_{tried[0]:03d}.json"   # a new file every time: worker processes remember files by name
        trial.write_text(json.dumps(c))
        s = sim_stats(str(trial.resolve()), n_per_class, pool, backgrounds=bgs)
        return gap(s, real), s

    extra: dict = {"conditions": "trial clips laid over the real no-gesture recordings" if over else "pure simulation"}
    with Sim(device, workers) as pool:
        before = sim_stats(None, n_per_class, pool, backgrounds=backgrounds)
        log(f"  generic laptop: gap {gap(before, real):.2f}   ({extra['conditions']})")
        best, stats = score(cal, pool, backgrounds)
        extra["gap_start"] = best
        log(f"  starting point: gap {best:.2f}")
        echo = ("overlay" if over else "hardware", "echo_ref_db", [[c - 8.0, c + 8.0] for c in ECHO_CENTRES])
        grid = [
            ("hardware", "mic_pos", [[x, y, 0.004] for x in (-0.16, -0.13, -0.10, -0.05) for y in (0.04, 0.10)]),
            echo,
            ("scene", "peak_speed", [[lo, hi] for lo, hi in ((0.6, 2.5), (0.9, 4.0), (0.9, 5.5), (1.4, 6.5), (2.0, 7.0))]),
            ("scene", "hand_height", [[lo, hi] for lo, hi in ((0.06, 0.18), (0.10, 0.25), (0.15, 0.35))]),
            ("hardware", "mic_pos", [[x, y, 0.004] for x in (-0.16, -0.13, -0.10, -0.05) for y in (0.04, 0.10)]),
            echo,
        ]
        for group, name, values in grid:
            for v in values:
                if v == cal[group][name]:
                    continue
                cand = json.loads(json.dumps(cal))
                cand[group][name] = v
                g_, s_ = score(cand, pool, backgrounds)
                log(f"    {group}.{name} = {v}: gap {g_:.2f}" + ("  <- better" if g_ < best else ""))
                if g_ < best:
                    best, stats, cal = g_, s_, cand
        if over:
            # second step: clips WITHOUT a recording underneath have a quieter floor, so they need their own echo level
            log("  clips without a recording underneath (pure simulation): their echo level")
            best_p, stats_p = score(cal, pool, ())
            log(f"    hardware.echo_ref_db = {cal['hardware']['echo_ref_db']}: gap {best_p:.2f}")
            for v in [[c - 8.0, c + 8.0] for c in ECHO_CENTRES]:
                if v == cal["hardware"]["echo_ref_db"]:
                    continue
                cand = json.loads(json.dumps(cal))
                cand["hardware"]["echo_ref_db"] = v
                g_, s_ = score(cand, pool, ())
                log(f"    hardware.echo_ref_db = {v}: gap {g_:.2f}" + ("  <- better" if g_ < best_p else ""))
                if g_ < best_p:
                    best_p, stats_p, cal = g_, s_, cand
            extra["pure"] = {"gap": best_p, "sim": stats_p}
    return cal, before, stats, extra


# --------------------------------------------------------------------------
# One file per held-out human (person facts fitted without that human)
# --------------------------------------------------------------------------
PERSON_GRID = {   # the person facts that are tried, every combination (the same ranges fit() walks through)
    "peak_speed": [[0.6, 2.5], [0.9, 4.0], [0.9, 5.5], [1.4, 6.5], [2.0, 7.0]],
    "hand_height": [[0.06, 0.18], [0.10, 0.25], [0.15, 0.35]],
}


def newest(folder: Path, name: str = "macbook") -> Path | None:
    """The everyone file with the highest version number in `folder` (macbook_v2.json before macbook_v1.json), or None."""
    found = []
    for p in Path(folder).glob(f"{name}_v*.json"):
        m = re.fullmatch(rf"{name}_v(\d+)", p.stem)                      # not the per-fold files (..._no_<human>) or reports
        if m:
            found.append((int(m.group(1)), p))
    return max(found)[1] if found else None


def human_of(person: str) -> str:
    """Which human a recorded 'person' name belongs to: a second session of the same human carries a number (lance2 -> lance)."""
    name = str(person).strip().lower()
    return re.sub(r"[\d_\- ]+$", "", name) or name


def fold_path(base: Path, human: str) -> Path:
    """Where the file for the fold that holds out `human` lives: macbook_v1.json -> macbook_v1_no_lance.json."""
    return base.with_name(f"{base.stem}_no_{human}.json")


def fit_folds(base: dict, rows_by_person: dict, work: Path, n_per_class: int, workers: int, device: str = "cpu",
              log=print, backgrounds: tuple = ()) -> tuple[dict, dict]:
    """For every human in the recordings: the calibration to use when THAT human is held out for scoring.

    The device facts are copied from `base` (they do not depend on who sits there). The person facts are chosen from
    PERSON_GRID by the same picture numbers as fit(), compared with the gestures of the OTHER humans only.
    The trial clips of a combination are made once and serve every fold. Returns ({human: calibration}, report).
    """
    work.mkdir(parents=True, exist_ok=True)
    combos = [{"peak_speed": sp, "hand_height": hh} for sp in PERSON_GRID["peak_speed"] for hh in PERSON_GRID["hand_height"]]
    humans = sorted({human_of(p) for p in rows_by_person})
    backgrounds = tuple((b, "") if isinstance(b, str) else b for b in backgrounds)
    stamp = int(time.time())

    def trial_clips(sim: Sim, tag: str, bgs: tuple) -> list:
        """The picture numbers of every combination, trial clips laid over `bgs` (none: pure simulation)."""
        got = []
        for k, scene in enumerate(combos):
            trial = work / f"fold_trial_{stamp}_{tag}_{k:02d}.json"            # a new name every run (workers remember files by name)
            trial.write_text(json.dumps({"hardware": base["hardware"], "scene": scene,
                                         **({"overlay": base["overlay"]} if base.get("overlay") else {})}))
            got.append(sim_stats(str(trial.resolve()), n_per_class, sim, backgrounds=bgs))
        return got

    # Without recordings to lay the clips over, one set of trial clips serves every fold. With them, every fold gets
    # its own: its clips are never laid over the held-out human's own recordings.
    sims_of: dict = {}
    with Sim(device, workers) as sim:
        if not backgrounds:
            shared = trial_clips(sim, "all", ())
            sims_of = {human: shared for human in humans + [None]}
            log(f"  {len(combos)} combinations of trial clips made (pure simulation)")
        else:
            for human in humans:
                bgs = tuple(b for b, person in backgrounds if human_of(person) != human)
                if not bgs:
                    log(f"  {human}: no recording left to lay the trial clips over, no file written")
                    continue
                sims_of[human] = trial_clips(sim, human, bgs)
                log(f"  without {human}: {len(combos)} combinations of trial clips laid over {len(bgs)} recording(s) of the others")
    out, table = {}, []
    everyone = summarise([r for rows in rows_by_person.values() for r in rows])
    for human in humans + [None]:                                 # None = nobody held out (for the report only)
        if human not in sims_of:
            continue
        sims = sims_of[human]
        used = sorted(p for p in rows_by_person if human is None or human_of(p) != human)
        real = summarise([r for p in used for r in rows_by_person[p]])
        gaps = [gap(s, real) for s in sims]
        best = int(np.argmin(gaps))
        table.append({"held_out": human, "people_used": used, "gaps": [round(g_, 4) for g_ in gaps], "chosen": combos[best]})
        if human is None:
            log(f"  nobody held out (for comparison): would choose {combos[best]} (gap {gaps[best]:.2f}); "
                f"the everyone file has {base['scene']}")
            continue
        if not used:
            log(f"  {human}: nobody else has recorded gestures, no file written")
            continue
        held = sorted(p for p in rows_by_person if human_of(p) == human)
        log(f"  without {human} ({', '.join(held)}): fitted on {', '.join(used)} -> peak speed {combos[best]['peak_speed']} m/s, "
            f"hand height {combos[best]['hand_height']} m (gap {gaps[best]:.2f})")
        out[human] = {"hardware": base["hardware"], "scene": combos[best], **({"overlay": base["overlay"]} if base.get("overlay") else {}),
                      "version": base.get("version", 1),
                      "device": base.get("device"), "made": time.strftime("%Y-%m-%d %H:%M"),
                      "fold": {"held_out": human, "held_out_people": held, "people_used": used,
                               "note": "person facts (scene) chosen without the held-out human" + (
                                   ", on trial clips never laid over that human's own recordings" if backgrounds else "")
                               + "; device facts (hardware) are the everyone fit, they do not depend on who sits in front of the laptop"},
                      "measured": base.get("measured"), "real": real, "real_everyone": everyone, "sim_after": sims[best],
                      "gap": {"after": gaps[best], "against_everyone": gap(sims[best], everyone), "scales": SCALE}}
    return out, {"combinations": combos, "folds": table, "clips_per_gesture": n_per_class, "device": device,
                 "sim": {str(h): s for h, s in sims_of.items()} if backgrounds else sims_of.get(None),
                 "conditions": ("trial clips laid over the real no-gesture recordings of the fold's own humans"
                                if backgrounds else "pure simulation")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit the simulator to one laptop's recordings (see the file header)")
    ap.add_argument("--real", required=True, help="folder with the recorded sessions (read only)")
    ap.add_argument("--out", help="fit everything and write this calibration file (.json)")
    ap.add_argument("--folds", metavar="FILE", help="instead: take the device facts from this calibration file and write one "
                                                    "file per human next to it (FILE_no_<human>.json), person facts fitted without that human")
    ap.add_argument("--start", metavar="FILE", help="with --out: start the search from this earlier calibration instead of from scratch")
    ap.add_argument("--clips", type=int, default=20, help="simulated clips per gesture for every setting tried (default 20)")
    ap.add_argument("--workers", type=int, default=0, help="CPU processes (default: every core)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "torch-cpu"],
                    help="where the trial clips are made: cpu = the numpy reference on the CPU cores (default), cuda / mps = the GPU "
                         "path (the same clips, much faster on a CUDA card), torch-cpu = the GPU code on the CPU (tests)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):          # a fit runs for minutes: show every line at once, also through a pipe or into a file
        sys.stdout.reconfigure(line_buffering=True)
    if bool(args.out) == bool(args.folds):
        ap.error("give --out FILE (fit everything) or --folds FILE (one file per held-out human)")
    if not args.workers:
        import os

        args.workers = min(os.cpu_count() or 1, 61 if sys.platform == "win32" else 10**6)
    t0 = time.time()
    root = Path(args.real).expanduser()
    print(f"real recordings: {root}")
    real, per_person, rows_by_person = real_stats(root)
    if args.folds:
        base_path = Path(args.folds)
        base = json.loads(base_path.read_text())
        print(f"device facts from {base_path}; person facts: {len(PERSON_GRID['peak_speed']) * len(PERSON_GRID['hand_height'])} combinations, "
              f"{3 * args.clips} trial clips each, on {args.device}")
        cals, report = fit_folds(base, rows_by_person, base_path.parent / "work", args.clips, args.workers, args.device,
                                 backgrounds=background_files(root, base_path.parent / "work"))
        for human, cal in cals.items():
            fold_path(base_path, human).write_text(json.dumps(cal, indent=1))
            print(f"wrote {fold_path(base_path, human)}")
        report.update(base=str(base_path), made=time.strftime("%Y-%m-%d %H:%M"), source=str(root))
        base_path.with_name(f"{base_path.stem}_folds.report.json").write_text(json.dumps(report, indent=1))
        print(f"wrote {base_path.with_name(base_path.stem + '_folds.report.json')} (every combination's gap per fold) in {time.time() - t0:.0f} s")
        return 0
    levels = measure_levels(root)
    print(f"measured: tone levels at the microphone {[round(v, 1) for v in levels['tone_db_median']]} dB (received / played)"
          + (f"; noise floor around the tones {levels['floor_dbfs_if_white_median']:.1f} dBFS as white noise over the whole band"
             if levels.get("floor_dbfs_if_white_median") is not None else ""))
    out = Path(args.out)
    start = json.loads(Path(args.start).read_text()) if args.start else None
    cal, before, after, extra = fit(real, levels, out.parent / "work", args.clips, args.workers, device=args.device, start=start,
                                    backgrounds=background_files(root, out.parent / "work"))
    m = re.search(r"_v(\d+)$", out.stem)
    cal.update({"version": int(m.group(1)) if m else 1, "device": "MacBook Pro, one microphone", "made": time.strftime("%Y-%m-%d %H:%M"),
                "started_from": str(args.start) if args.start else None, "clips_per_gesture": args.clips,
                "source": str(root), "measured": levels, "real": real, "real_per_person": per_person,
                "sim_before": before, "sim_after": after, "fit": extra,
                "gap": {"before": gap(before, real), "start": extra.get("gap_start"), "after": gap(after, real), "scales": SCALE}})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal, indent=1))
    print(f"\n{'':<14}{'':<12}" + "".join(f"{k:>15}" for k in KEYS))
    for c in real:
        for name, d in (("real", real), ("sim before", before), ("sim after", after)):
            if c in d:
                print(f"{c:<14}{name:<12}" + "".join(f"{d[c][k]:15.2f}" for k in KEYS))
    print(f"\ngap (0 = same means, 1 = one scale apart on average; {extra['conditions']}): generic laptop "
          f"{cal['gap']['before']:.2f}, starting point {extra['gap_start']:.2f}, fitted {cal['gap']['after']:.2f}"
          + (f"; pure simulation with its own echo level: {extra['pure']['gap']:.2f}" if "pure" in extra else ""))
    print(f"wrote {out} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

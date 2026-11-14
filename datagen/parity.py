"""parity.py — does the GPU path make the same training pictures as the CPU path?

Plain words
-----------
The GPU code (render_torch.py, sonar/features_torch.py, gpu_pipeline.py) is a second
implementation of the clip maker. A second implementation can drift from the first
without anybody noticing, and then the model would be trained on pictures that differ
from what the live engine computes. This module is the guard against that:

  1. it builds a fixed list of clips (the fixture): every gesture, every kind of
     non-gesture, clips laid over a "real" background, one-microphone clips, swapped
     channels, and hard cases (fastest strokes, clipping, large clock mismatch ...);
  2. it makes every clip twice, with the CPU reference (generate.make_clip) and with
     the GPU path (gpu_pipeline.generate_gpu), from the same seeds;
  3. it compares, clip by clip: the audio before rounding, the final audio, the
     feature stream, where the movement bursts are, the labels and the pictures;
  4. it compares the CPU results with a small file stored in the repository
     (tests/fixtures/gpu_parity_reference.npz), so a later change to EITHER path shows.

How close is close enough (LIMITS below; measured values are in TRAINING.md)
----------------------------------------------------------------------------
* Before the converter the two paths agree to a few parts in ten million; the limits
  sit 50-100x above what is measured and 100x or more below what any real mistake
  (wrong delay, wrong channel, wrong filter) produces.
* The converter rounds to 16 or 24 bits. A value that sits almost exactly between two
  steps can round the other way, so some samples land ONE step apart. How many follows
  exactly from the difference right before the converter (both paths hand that signal
  out): a sample flips when a step boundary lies between the two values, so the share
  of flipped samples is the average difference divided by the step. A quiet 16-bit clip:
  a few in a million. A loud or clipped clip, or a 24-bit one (steps 256x finer): up to
  tens of percent, each still one step = far below the simulated noise. The check allows
  1.5x that share (+ 2 in 100,000) and a largest difference of one step plus the largest
  difference before the converter. Anything beyond that is not rounding.
* Right before the converter the paths differ more than at the earlier stages (a few
  parts in a million): the float32 rounding noise of the echo sum, which the CPU code has
  too, is amplified by the simulated microphone, whose response rises above 20 kHz.
* One such sample is a tiny click. It can move a handful of noise-floor pixels of a
  picture by up to ~0.02 (of a 0..1 range) and leaves the rest untouched. So pictures
  are held to a small AVERAGE difference and a high correlation, with a loose cap on
  the largest single pixel.
* The feature code is also checked on IDENTICAL audio, where no rounding luck is
  involved: there it must agree to 1e-4 and find the same bursts.
* A burst boundary is a threshold decision; a click can move one by a frame in rare
  clips. A small share of clips may differ there (counted and listed), no more.

Read-only helper module apart from write_reference(), which writes the one fixture file.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import generate as g

REFERENCE_FILE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "gpu_parity_reference.npz"
SEED0 = 9_000_000                 # fixture seeds, far away from the seeds training data uses
FIXTURE_VERSION = 1

# Largest allowed differences, per GPU number format (see the file header for the reasoning).
LIMITS: dict[str, dict[str, float]] = {
    # measured on the 1,000-clip fixture, CPU reference against the GPU code on a CPU ("ref"; worst clip in brackets)
    "ref": {"dyn_rel": 2e-5,            # moving echoes before rounding            [6.6e-7]
            "static_rel": 1e-8,         # direct sound + room before rounding      [4.0e-10]
            "snr_db": 1e-3,             # echo loudness figure, dB                 [3.9e-6]
            "conv_rel": 1e-4,           # audio right before the converter         [see TRAINING.md]
            "overlay_abs": 2e-6,        # overlay clips, no rounding               [3.0e-8]
            "same_audio_feat": 1e-4,    # feature code on identical audio          [2.4e-7]
            "same_audio_seg": 0.0,      # ... share of clips with other bursts     [0]
            "stream_mean": 3e-3,        # feature stream, average difference       [8.4e-4]
            "pic_mean": 4e-3,           # pictures, average difference             [1.0e-3]
            "pic_max": 0.12,            # pictures, largest single pixel           [0.026]
            "pic_corr": 0.99,           # pictures, correlation                    [0.998]
            "seg_differs": 0.02},       # share of clips with a burst moved        [0.002]
    "fast": {"dyn_rel": 2e-5, "static_rel": 1e-5, "snr_db": 1e-3, "conv_rel": 1e-4, "overlay_abs": 2e-6, "same_audio_feat": 1e-4,
             "same_audio_seg": 0.0, "stream_mean": 6e-3, "pic_mean": 8e-3, "pic_max": 0.2, "pic_corr": 0.98, "seg_differs": 0.1},
    "single": {"dyn_rel": 5e-3, "static_rel": 1e-3, "snr_db": 2e-2, "conv_rel": 2e-2, "overlay_abs": 2e-4, "same_audio_feat": 0.05,
               "same_audio_seg": 0.02, "stream_mean": 2e-2, "pic_mean": 2e-2, "pic_max": 0.6, "pic_corr": 0.95, "seg_differs": 0.15},
}
# The stored CPU results against today's CPU results (another computer rounds a little differently).
DRIFT_LIMITS = {"thumb_mean": 0.6, "act_median_db": 0.05, "seg_differs": 0.03, "rms_rel": 1e-3}

_HARD = [   # (name, hardware overrides) - cases the random draw rarely or never produces
    ("clock +30 ppm", {"clock_ppm": 30.0}), ("clock -30 ppm", {"clock_ppm": -30.0}),
    ("clock +120 ppm", {"clock_ppm": 120.0}), ("clock -120 ppm", {"clock_ppm": -120.0}),
    ("clipping hard, 16 bit", {"direct_gain_db": 12.0, "adc_bits": 16}), ("clipping hard, 24 bit", {"direct_gain_db": 12.0, "adc_bits": 24}),
    ("clipping mild", {"direct_gain_db": 3.0, "adc_bits": 16}), ("no rounding", {"adc_bits": 0}),
    ("latency 0", {"latency_samples": 0}), ("latency 1900", {"latency_samples": 1900}),
    ("no desk bounce", {"table_refl": 0.0}), ("strong desk bounce", {"table_refl": 0.85}),
    ("hum 50 Hz", {"hum_hz": 50.0, "hum_db": -24.0}), ("hum 120 Hz", {"hum_hz": 120.0, "hum_db": -24.0}),
    ("gain step +6 dB", {"gain_step_db": 6.0}), ("gain step -6 dB", {"gain_step_db": -6.0}),
    ("quietest room, 24 bit", {"noise_dbfs": -100.0, "babble_dbfs": -75.0, "hf_burst_rate": 0.0, "adc_bits": 24}),
    ("loud hall with clicks", {"babble_dbfs": -30.0, "hf_burst_rate": 2.0, "hf_burst_db": 0.0}),
    ("interfering tone", {"tones": [(19200.0, -15.0, 2.0)]}),
    ("strongest echo", {"echo_ref_db": 6.0}), ("weakest echo", {"echo_ref_db": -30.0}),
    ("loudest speaker, 16 bit", {"direct_gain_db": -40.0, "adc_bits": 16}), ("quietest speaker, 16 bit", {"direct_gain_db": -78.0, "adc_bits": 16}),
]
# seeds whose scenes have the fastest strokes, limb jumps (lowest decimation) and the most reflectors
_FAST = [(9100191, "swipe_left"), (9100195, "swipe_left"), (9100114, "swipe_right"), (9100259, "swipe_right"),
         (9100200, "push"), (9100044, "push"), (9100038, "pull"), (9100093, "pull"), (9100170, "entering"),
         (9100117, "entering"), (9100258, "leaving"), (9100010, "none"), (9100013, "none"), (9100133, "none")]


# --------------------------------------------------------------------------
# The fixture
# --------------------------------------------------------------------------
def fixture_jobs(n_clips: int = 1000, background: str | None = None, profile: str | None = None) -> list[tuple]:
    """(name, job) pairs. `background` is a no-gesture recording for the overlay clips (None: no overlay clips).

    The list is always built at full size (1,000) and thinned evenly, so a small fixture
    still has every kind of clip.
    """
    jobs: list[tuple] = []
    seed = [SEED0]

    def add(name: str, kind: str, n: int, seeds: list | None = None, bgs: tuple = (), swap: tuple = (False, False),
            mics: int | None = None, tweak: dict | None = None) -> None:
        for k in range(n):
            s = seeds[k] if seeds else seed[0]
            seed[0] += 0 if seeds else 1
            jobs.append((name, (s, kind, profile, 6.0, False, bgs, 1.0 if bgs else 0.0, swap[0], swap[1], mics, tweak)))

    for kind in g.GESTURE4:                                    # 4 x 100 plain gesture clips
        add(kind, kind, 100)
    add("none", "none", 140)                                   # every "none" variant many times over
    for kind, n in (("wave", 30), ("entering", 30), ("leaving", 30), ("present", 20), ("empty", 10)):
        add(kind, kind, n)
    for kind in g.GESTURE4:                                    # one microphone (MacBook), swapped channels
        add(f"{kind}, one microphone", kind, 10, mics=1)
        add(f"{kind}, swapped microphones", kind, 5, swap=(True, False))
        add(f"{kind}, swapped speakers", kind, 5, swap=(False, True))
    add("none, one microphone", "none", 10, mics=1)
    add("none, both swapped", "none", 10, swap=(True, True))
    for s, kind in _FAST:                                      # fastest movement
        add(f"{kind}, fastest / most reflectors", kind, 1, seeds=[s])
    for name, hw in _HARD:                                     # hard cases: 4 gestures + 1 none each
        for kind in (*g.GESTURE4, "none"):
            add(f"{kind}, {name}", kind, 1, tweak={"hw": hw})
    if background:                                             # synthetic movement over a recording
        for kind in g.GESTURE4:
            add(f"{kind}, on a recording", kind, 22, bgs=(background,))
        add("none, on a recording", "none", 20, bgs=(background,))
        add("wave, on a recording", "wave", 6, bgs=(background,))
        add("push, on a recording, one microphone", "push", 4, bgs=(background,), mics=1)
        add("swipe_left, on a recording, swapped", "swipe_left", 4, bgs=(background,), swap=(True, True))
        add("pull, on a recording, +30 ppm", "pull", 3, bgs=(background,), tweak={"hw": {"clock_ppm": 30.0}})
    while len(jobs) < 1000:                                    # top up with plain clips of every kind
        kinds = (*g.GESTURE4, "none", "wave", "entering", "leaving")
        add(kinds[len(jobs) % len(kinds)], kinds[len(jobs) % len(kinds)], 1)
    jobs = jobs[:1000]
    if n_clips >= len(jobs):
        return jobs
    keep = np.unique(np.linspace(0, len(jobs) - 1, n_clips).round().astype(int))
    return [jobs[i] for i in keep]


def build_background(folder: Path, profile: str | None = None) -> str:
    """A stand-in for a real "nobody gesturing" recording: 4 rendered no-gesture clips of one virtual laptop.

    Made with the CPU reference renderer from fixed seeds, saved as float32 .npy in `folder`.
    """
    from .hardware import Profile, sample_hardware
    from .render import render
    from .scenes import sample_scene

    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "fixture_background.npy"
    if target.exists():
        return str(target)
    core, (tx, pre, n) = g._setup(profile)
    rng = np.random.default_rng(SEED0 - 1)
    hw = sample_hardware(rng, Profile(band_hz=core.band), core.fs)
    hw.tx_route = "stereo" if core.n_speakers >= 2 else hw.tx_route
    hw.adc_bits, hw.clock_ppm, hw.latency_samples = 24, 4.0, 311
    pieces = []
    for kind in ("none", "present", "empty", "none"):
        scene = sample_scene(rng, kind, g.STORY, lead_in=g.LEAD_IN)
        pieces.append(render(tx, hw, scene, n, rng, pre=pre, band=core.band).rx)
    np.save(target, np.concatenate(pieces).astype(np.float32))
    return str(target)


# --------------------------------------------------------------------------
# One clip through each path
# --------------------------------------------------------------------------
def _stream_dict(st) -> dict:
    return {"feat": st.feat, "ready": st.ready, "activity": np.asarray(st.activity_db, np.float32),
            "segments": [(int(s.start), int(s.end), int(s.decided), float(s.peak_db)) for s in st.segments],
            "not_before": list(st.not_before)}


def _examples(result: dict) -> list:
    return [(ex["label"], ex["kind"], round(float(ex["t_start"]), 3), np.asarray(ex.get("padded"), np.float32))
            for ex in result["examples"]]


def reference_clip(job: tuple) -> dict:
    """Worker: one clip with the CPU reference, with everything the comparison needs."""
    cap: dict = {}
    res = g.make_clip(job, capture=cap)
    if res["error"]:
        raise RuntimeError(f"CPU reference failed on seed {job[0]}: {res['error']}")
    parts = cap["parts"]
    return {"rx": cap["rx"], "snr": float(cap["snr"]), "attempts": int(cap["attempts"]), "adc_bits": int(cap["hw"].adc_bits or 0),
            "scale": float(10.0 ** (cap["hw"].direct_gain_db / 20.0)),
            "dynamic": None if parts is None else parts["dynamic"].astype(np.float32),
            "static": None if parts is None else parts["static"].astype(np.float32),
            "converter_in": None if parts is None else parts["converter_in"],
            "examples": _examples(res), **_stream_dict(cap["stream"].core)}


def _err(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(largest difference, RMS difference / RMS of b, correlation) of a against the reference b."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    d = a - b
    rms = float(np.sqrt(np.mean(b * b)))
    sa, sb = float(a.std()), float(b.std())
    corr = float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)) if sa > 0 and sb > 0 else (1.0 if sa == sb else 0.0)
    return float(np.abs(d).max()) if d.size else 0.0, (float(np.sqrt(np.mean(d * d))) / rms if rms > 0 else float(np.abs(d).max(initial=0.0))), corr


def compare_clip(name: str, job: tuple, ref: dict, got: dict, gres: dict, same_audio) -> dict:
    """All the numbers for one clip. `got` = the GPU path's captured audio, `gres` its result, `same_audio` the
    GPU feature code run on the CPU's audio."""
    r: dict = {"name": name, "seed": int(job[0]), "kind": job[1], "attempts_same": ref["attempts"] == got["attempts"]}
    r["snr_db"] = abs(ref["snr"] - got["snr"]) if np.isfinite(ref["snr"]) and np.isfinite(got["snr"]) else \
        (0.0 if np.isnan(ref["snr"]) and np.isnan(got["snr"]) else float("inf"))
    overlay = ref["dynamic"] is None
    a, b = got["rx"], ref["rx"]
    r["rx_max"], r["rx_rel"], r["rx_corr"] = _err(a, b)
    r["flips"] = float(np.mean(a != b))
    bits = ref["adc_bits"]
    r["bits"] = 0 if overlay else bits
    if not overlay and r["attempts_same"]:
        r["dyn_rel"] = _err(got["dynamic"], ref["dynamic"])[1]
        r["static_rel"] = _err(got["static"], ref["static"])[1]
        d = np.abs(got["converter_in"] - ref["converter_in"])           # the difference right before the converter
        r["conv_rel"] = float(np.sqrt(np.mean(d * d) / max(float(np.mean(ref["converter_in"] ** 2)), 1e-300)))
        if bits:
            step = 2.0 ** -(bits - 1)
            r["steps_over"] = (r["rx_max"] / step) / (1.0 + float(d.max()) / step)          # <= 1: one step + that difference
            r["flips_over"] = r["flips"] / (1.5 * float(d.mean()) / step + 2e-5)            # <= 1: flips as rounding predicts
        else:                                      # no rounding: only float32 storage between the two
            r["unrounded_over"] = r["rx_max"] / (float(d.max()) + 1.2e-7 * max(float(np.abs(b).max()), 1e-6) + 1e-12)
    # the feature code on identical audio
    r["same_audio_feat"] = float(np.abs(same_audio.feat - ref["feat"]).max())
    r["same_audio_seg"] = [(s.start, s.end, s.decided) for s in same_audio.segments] == [s[:3] for s in ref["segments"]]
    # end to end
    gst = _stream_dict(gres["stream"])
    n_pow = ref["feat"].shape[1] - (ref["feat"].shape[1] % 2)          # an odd channel count = the last one is left/right
    r["stream_max"], _, r["stream_corr"] = _err(gst["feat"][:, :n_pow], ref["feat"][:, :n_pow])
    r["stream_mean"] = float(np.abs(gst["feat"] - ref["feat"]).mean())
    r["seg_same"] = [s[:3] for s in gst["segments"]] == [s[:3] for s in ref["segments"]] and gst["not_before"] == ref["not_before"]
    ex_c, ex_g = ref["examples"], _examples(gres)
    r["labels_same"] = [e[:3] for e in ex_c] == [e[:3] for e in ex_g]
    r["n_pictures"] = len(ex_c)
    if r["seg_same"] and r["labels_same"] and ex_c:
        pc, pg = np.stack([e[3] for e in ex_c]), np.stack([e[3] for e in ex_g])
        r["pic_max"], _, r["pic_corr"] = _err(pg[:, :n_pow], pc[:, :n_pow])
        r["pic_mean"] = float(np.abs(pg - pc).mean())
        if pc.shape[1] > n_pow:
            r["lat_max"], _, r["lat_corr"] = _err(pg[:, n_pow:], pc[:, n_pow:])
            r["lat_mean"] = float(np.abs(pg[:, n_pow:] - pc[:, n_pow:]).mean())
    return r


# --------------------------------------------------------------------------
# The small stored record of the CPU results
# --------------------------------------------------------------------------
def signature(ref: dict) -> dict:
    """A few hundred numbers that pin down one CPU clip: coarse feature picture, activity trace, bursts, level."""
    feat = ref["feat"]
    t, c, f = feat.shape
    tt, ff = t // 5, f // 8
    block = feat[:tt * 5].reshape(tt, 5, c, ff, 8).mean(axis=(1, 4))                  # [T/5, C, F/8]
    thumb = np.zeros((tt, 5, ff), np.uint8)
    thumb[:, :c] = np.clip(np.round(127.5 * (block + 1.0)), 0, 255).astype(np.uint8)   # -1..1 -> 0..255
    segs = np.full((6, 3), -1, np.int16)
    for k, s in enumerate(ref["segments"][:6]):
        segs[k] = s[:3]
    rx = ref["rx"].astype(np.float64)
    rms = np.zeros(2, np.float32)
    rms[:rx.shape[1]] = np.sqrt(np.mean(rx * rx, axis=0))
    return {"thumb": thumb, "act": ref["activity"].astype(np.float16), "segs": segs, "rms": rms,
            "n_pictures": np.uint8(len(ref["examples"])), "attempts": np.uint8(ref["attempts"])}


def clip_key(name: str, job: tuple) -> int:
    """A stable number for one fixture clip (its place in the stored file)."""
    return int.from_bytes(hashlib.sha256(f"{name}|{job[0]}|{job[1]}".encode()).digest()[:7], "big")


def pack_signatures(sigs: list, seeds: list) -> dict:
    out = {k: np.stack([s[k] for s in sigs]) for k in sigs[0]}
    out["seeds"] = np.asarray(seeds, np.int64)
    h = hashlib.sha256()
    for k in sorted(out):
        h.update(k.encode())
        h.update(np.ascontiguousarray(out[k]).tobytes())
    out["sha256"] = np.array(h.hexdigest())
    out["version"] = np.array(FIXTURE_VERSION)
    return out


def write_reference(packed: dict, path: Path = REFERENCE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **packed)


def drift(packed: dict, path: Path = REFERENCE_FILE) -> dict:
    """Today's CPU results against the stored ones. status: same | close | changed | missing | other-fixture."""
    if not path.exists():
        return {"status": "missing", "detail": f"{path} does not exist (run scripts/check_gpu_parity.py --write-reference)"}
    with np.load(path, allow_pickle=False) as z:
        old = {k: z[k] for k in z.files}
    pos = {int(s): i for i, s in enumerate(old["seeds"])}
    rows = [(i, pos[int(s)]) for i, s in enumerate(packed["seeds"]) if int(s) in pos]
    if int(old.get("version", 0)) != FIXTURE_VERSION or len(rows) < 0.9 * len(packed["seeds"]):
        return {"status": "other-fixture", "detail": "the stored file was made for a different fixture"}
    a, b = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    if len(rows) == len(old["seeds"]) == len(packed["seeds"]) and str(old["sha256"]) == str(packed["sha256"]):
        return {"status": "same", "detail": f"bit-identical to the stored CPU results (sha256 {str(old['sha256'])[:16]})", "clips": len(rows)}
    thumb = np.abs(packed["thumb"][a].astype(np.float32) - old["thumb"][b].astype(np.float32)).mean(axis=(1, 2, 3))
    act = np.median(np.abs(packed["act"][a].astype(np.float32) - old["act"][b].astype(np.float32)), axis=1)
    seg = np.any(packed["segs"][a] != old["segs"][b], axis=(1, 2))
    rms = np.abs(packed["rms"][a] - old["rms"][b]).max(axis=1) / np.maximum(old["rms"][b].max(axis=1), 1e-12)
    worst = {"thumb_mean": float(thumb.max()), "act_median_db": float(act.max()), "seg_differs": float(seg.mean()), "rms_rel": float(rms.max())}
    bad = [k for k, v in worst.items() if v > DRIFT_LIMITS[k]]
    return {"status": "changed" if bad else "close", "worst": worst, "limits": DRIFT_LIMITS, "over": bad, "clips": len(rows),
            "detail": ("CPU results differ from the stored ones beyond rounding: " + ", ".join(bad)) if bad else
                      "not bit-identical to the stored CPU results but within rounding (another computer or library version)"}


# --------------------------------------------------------------------------
# The whole check
# --------------------------------------------------------------------------
def run(n_clips: int, device: str, precision: str = "auto", workers: int = 4, batch: int = 0, profile: str | None = None,
        work: Path | None = None, data: Path | None = None, log=None, chunk: int = 0, pick=None, kernel: str = "auto",
        motion: str | None = None, calibration: str | None = None, noise: str = "cpu", features: str = "conv") -> dict:
    """Make the fixture with both paths and compare. Returns the report (report["passed"] is the verdict).

    `pick(name, job) -> bool`, if given, keeps only some fixture clips (to look at a few closely).
    """
    import torch
    from concurrent.futures import ProcessPoolExecutor

    from sonar.features import stream_from_columns

    from .gpu_pipeline import GpuRenderer, default_batch, generate_gpu

    log = log or (lambda text: print(text, flush=True))
    t_start = time.time()
    work = Path(work or (Path(__file__).resolve().parent / "out" / "parity"))
    background = build_background(work, profile)
    jobs = fixture_jobs(n_clips, background, profile)
    if pick is not None:
        jobs = [(name, job) for name, job in jobs if pick(name, job)]
    n_stored = len(jobs)                                     # clips that belong to the stored reference
    if motion:                                               # extra clips that replay real tracked hand paths (motion.py)
        for k in range(max(8, min(40, n_clips // 25))):
            kind = (*g.GESTURE4, "none")[k % 5]
            jobs.append((f"{kind}, real hand path", (SEED0 + 500_000 + k, kind, profile, 6.0, False, (), 0.0, False, False, None, None,
                                                    str(Path(motion).resolve()), 1.0)))
    if calibration:                                          # extra clips drawn around a laptop fitted to real recordings
        cal = str(Path(calibration).resolve())               # (calibration.py): one microphone, loud tones, fast strokes;
        for k in range(min(40, n_clips // 25)):              # every 2nd replays a real hand path, every 4th lies over a recording
            kind = (*g.GESTURE4, "none")[k % 5]
            lib = str(Path(motion).resolve()) if motion and k % 2 else None
            bgs = (background,) if k % 4 == 3 else ()
            jobs.append((f"{kind}, fitted laptop", (SEED0 + 600_000 + k, kind, profile, 6.0, False, bgs, 1.0 if bgs else 0.0, False, False, 1,
                                                    None, lib, 1.0 if lib else 0.0, cal)))
    gpu = GpuRenderer(profile, device, precision, kernel=kernel, noise=noise, features=features)
    limits = LIMITS[gpu.precision]
    batch = batch or default_batch(device, gpu.precision)
    chunk = chunk or max(batch, 4 * workers)
    core = gpu.core
    log(f"parity check: {len(jobs)} clips, CPU reference (numpy) against {gpu.device} (PyTorch {torch.__version__}, "
        f"number format '{gpu.precision}'), {workers} worker processes")
    log(f"echo loop on the GPU side: {gpu.echo_impl}" + (f" ({gpu.kernel_note})" if gpu.kernel_note else "")
        + ("; hall noise and gain wander shaped on the device" if gpu.shape_on_device else "")
        + ("; feature filter through FFTs" if gpu.fir == "fft" else ""))
    rows: list[dict] = []
    sigs: list[dict] = []
    seconds = {"cpu": 0.0, "gpu": 0.0}
    with ProcessPoolExecutor(max(1, workers)) as pool:
        for a in range(0, len(jobs), chunk):
            part = jobs[a:a + chunk]
            only = [j for _, j in part]
            t0 = time.time()
            refs = list(pool.map(reference_clip, only, chunksize=1))
            seconds["cpu"] += time.time() - t0
            caps: dict = {}
            t0 = time.time()
            gres = list(generate_gpu(only, profile, device, workers, precision=gpu.precision, batch=batch,
                                     capture=lambda i, d: caps.__setitem__(i, d), gpu=gpu, pool=pool))
            seconds["gpu"] += time.time() - t0
            # the GPU feature code on the CPU's own audio (no rounding luck involved)
            same: dict = {}
            by_mics: dict = {}
            for i, r in enumerate(refs):
                by_mics.setdefault(r["rx"].shape[1], []).append(i)
            for mics, idx in by_mics.items():
                db, lat = gpu.columns(torch.from_numpy(np.stack([refs[i]["rx"] for i in idx])).to(gpu.device))
                cfg = core.cfg if mics == core.cfg.n_mics else core.cfg.with_(n_mics=mics)
                for row, i in enumerate(idx):
                    same[i] = stream_from_columns(cfg, db[row], None if lat is None else lat[row])
            for i, (name, job) in enumerate(part):
                rows.append(compare_clip(name, job, refs[i], caps[i], gres[i], same[i]))
                if a + i < n_stored:
                    sigs.append(signature(refs[i]))
            log(f"  {min(a + chunk, len(jobs))}/{len(jobs)} clips ({time.time() - t_start:.0f} s)")
    report = judge(rows, limits)
    report.update({"clips": len(jobs), "device": str(gpu.device), "precision": gpu.precision, "torch": torch.__version__,
                   "echo_loop": gpu.echo_impl, "kernel_note": gpu.kernel_note,
                   "noise_shaping": "on the device" if gpu.shape_on_device else "in the CPU workers", "feature_filter": gpu.fir,
                   "seconds": {k: round(v, 1) for k, v in seconds.items()}, "limits": limits,
                   "per_clip_ms": {k: round(1e3 * v / len(jobs), 1) for k, v in seconds.items()}})
    report["packed"] = pack_signatures(sigs, [clip_key(name, job) for name, job in jobs[:n_stored]])
    if profile not in (None, "default"):                     # the stored file holds the default profile's CPU results
        report["drift"] = {"status": "skipped", "detail": f"signal profile '{profile}': the stored file is for the default profile"}
    elif getattr(core.cfg, "seg_rel_off_db", 0.0) > 0:       # ... cut the engine's usual way (SONAR_SEG_REL_OFF_DB not set)
        report["drift"] = {"status": "skipped", "detail": "peak-relative burst cutting is switched on (SONAR_SEG_REL_OFF_DB): "
                                                          "the stored file is for the usual cutting"}
    elif n_stored >= 1000:
        report["drift"] = drift(report["packed"])
    else:
        report["drift"] = {"status": "skipped", "detail": "small fixture: not compared with the stored file"}
    if report["drift"]["status"] == "changed":
        report["passed"] = False
        report["failures"].append("the CPU reference no longer matches tests/fixtures/gpu_parity_reference.npz: " + report["drift"]["detail"])
    if data is not None:
        report["sessions"] = sessions(Path(data), gpu, log)
        bad = [s for s in report["sessions"] if s["feat_max"] > limits["same_audio_feat"] or not s["segments_same"]]
        if bad:
            report["passed"] = False
            report["failures"].append(f"real recordings: {len(bad)} session(s) give different features on the GPU")
    report["total_seconds"] = round(time.time() - t_start, 1)
    return report


def sessions(root: Path, gpu, log=print, max_sessions: int = 6, seconds: float = 120.0) -> list[dict]:
    """Real recordings under `root`: the engine's CPU feature code against the GPU feature code on the same audio."""
    import torch

    from sonar.dataset import find_sessions, load_session
    from sonar.features import stream_features, stream_from_columns
    from sonar.features_torch import TorchFrontend

    out = []
    for folder in find_sessions(root)[:max_sessions] if root.exists() else []:
        try:
            _, audio, cfg = load_session(folder)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            log(f"  session {folder.name}: skipped ({type(exc).__name__})")
            continue
        audio = audio[:int(seconds * cfg.fs)]
        ref = stream_features(audio, cfg)
        fe = TorchFrontend(cfg, audio.shape[1], gpu.device, precision="single" if gpu.precision == "single" else "auto")
        db, lat = fe.columns(torch.from_numpy(np.ascontiguousarray(audio))[None].to(gpu.device))
        got = stream_from_columns(fe.cfg, db[0].cpu().numpy(), None if lat is None else lat[0].cpu().numpy())
        same = [(s.start, s.end, s.decided) for s in got.segments] == [(s.start, s.end, s.decided) for s in ref.segments]
        out.append({"session": folder.name, "seconds": round(len(audio) / cfg.fs, 1), "bursts": len(ref.segments),
                    "feat_max": float(np.abs(got.feat - ref.feat).max()), "segments_same": bool(same)})
        log(f"  session {folder.name}: {out[-1]['seconds']} s, {out[-1]['bursts']} bursts, largest feature difference "
            f"{out[-1]['feat_max']:.1e}, bursts {'identical' if same else 'DIFFERENT'}")
    return out


def judge(rows: list[dict], limits: dict) -> dict:
    """Turn the per-clip numbers into checks with a verdict each."""
    def vals(key: str, where=lambda r: True) -> list:
        return [(r[key], r) for r in rows if r.get(key) is not None and where(r)]

    def worst(pairs: list, largest: bool = True, n: int = 5) -> list:
        order = sorted(pairs, key=lambda p: -p[0] if largest else p[0])[:n]
        return [{"value": float(v), "clip": r["name"], "seed": r["seed"]} for v, r in order]

    checks = []

    def check(name: str, pairs: list, limit: float, largest: bool = True, what: str = "") -> None:
        if not pairs:
            return
        v = [p[0] for p in pairs]
        value = max(v) if largest else min(v)
        ok = value <= limit if largest else value >= limit
        checks.append({"check": name, "worst": float(value), "median": float(np.median(v)), "limit": limit, "ok": bool(ok),
                       "n": len(v), "what": what, "worst_clips": worst(pairs, largest)})

    def share(name: str, flags: list, limit: float, what: str) -> None:
        bad = [r for ok, r in flags if not ok]
        frac = len(bad) / max(len(flags), 1)
        checks.append({"check": name, "worst": frac, "median": frac, "limit": limit, "ok": frac <= limit, "n": len(flags), "what": what,
                       "worst_clips": [{"value": 1.0, "clip": r["name"], "seed": r["seed"]} for r in bad[:8]]})

    share("same number of redraws", [(r["attempts_same"], r) for r in rows], 0.0, "share of clips redrawn a different number of times")
    check("echo loudness (dB)", vals("snr_db"), limits["snr_db"], what="difference of the 'how audible is the gesture' figure")
    check("moving echoes, before rounding", vals("dyn_rel"), limits["dyn_rel"], what="RMS difference / RMS")
    check("direct sound + room, before rounding", vals("static_rel"), limits["static_rel"], what="RMS difference / RMS")
    check("audio right before the converter", vals("conv_rel"), limits["conv_rel"], what="RMS difference / RMS")
    check("final audio: largest difference / what rounding allows", vals("steps_over"), 1.0001,
          what="<= 1: one converter step plus the largest difference before the converter")
    check("final audio: samples one step apart / what rounding predicts", vals("flips_over"), 1.0,
          what="<= 1: share of flipped samples against 1.5 x (average difference before the converter / step)")
    check("  (for information) 16 bit: share of samples one step apart", vals("flips", lambda r: r["bits"] == 16), 1.0)
    check("  (for information) 24 bit: share of samples one step apart", vals("flips", lambda r: r["bits"] == 24), 1.0)
    check("final audio, pure clip without rounding", vals("unrounded_over"), 1.0, what="<= 1")
    check("final audio, overlays (no rounding): largest difference", vals("rx_max", lambda r: r["bits"] == 0 and "dyn_rel" not in r),
          limits["overlay_abs"], what="full scale = 1")
    check("final audio: correlation", vals("rx_corr"), 0.99999, largest=False)
    check("features from IDENTICAL audio: largest difference", vals("same_audio_feat"), limits["same_audio_feat"], what="pictures are 0..1")
    share("features from IDENTICAL audio: same bursts", [(r["same_audio_seg"], r) for r in rows], limits["same_audio_seg"], "share of clips with different bursts")
    check("feature stream: average difference", vals("stream_mean"), limits["stream_mean"], what="pictures are 0..1")
    share("same bursts and labels", [(r["seg_same"] and r["labels_same"], r) for r in rows], limits["seg_differs"],
          "share of clips where a burst boundary or a label differs")
    check("pictures: average difference", vals("pic_mean"), limits["pic_mean"], what="the model's input, 0..1")
    check("pictures: largest single pixel (Doppler pictures)", vals("pic_max"), limits["pic_max"], what="the model's input, 0..1")
    check("pictures: correlation (Doppler pictures)", vals("pic_corr"), limits["pic_corr"], largest=False)
    check("left/right picture: average difference", vals("lat_mean"), limits["pic_mean"], what="-1..1; an angle, single pixels can wrap")
    check("  (for information) left/right picture: largest single pixel", vals("lat_max"), 2.0, what="-1..1")
    failures = [f"{c['check']}: {c['worst']:.3g} against a limit of {c['limit']:.3g}" for c in checks if not c["ok"]]
    return {"passed": not failures, "failures": failures, "checks": checks,
            "pictures_compared": int(sum(r["n_pictures"] for r in rows if "pic_mean" in r)),
            "pictures_total": int(sum(r["n_pictures"] for r in rows))}


def print_report(report: dict, log=print) -> None:
    log("")
    log(f"{'check':<66} {'worst':>10} {'median':>10} {'limit':>10}  result")
    for c in report["checks"]:
        log(f"{c['check'][:66]:<66} {c['worst']:>10.3g} {c['median']:>10.3g} {c['limit']:>10.3g}  {'ok' if c['ok'] else 'FAIL'}")
    log("")
    log("worst clips per check:")
    for c in report["checks"]:
        if c["worst_clips"] and (not c["ok"] or c["check"].startswith(("pictures", "moving", "final audio, 16", "same bursts"))):
            log(f"  {c['check']}: " + "; ".join(f"{w['clip']} (seed {w['seed']}) {w['value']:.3g}" for w in c["worst_clips"][:5]))
    log("")
    log(f"pictures compared pixel by pixel: {report['pictures_compared']} of {report['pictures_total']} "
        f"(the rest belong to clips whose bursts differ, counted above)")
    log(f"time per clip: CPU reference {report['per_clip_ms']['cpu']:.0f} ms x cores, GPU path {report['per_clip_ms']['gpu']:.0f} ms "
        f"(wall, on {report['device']})")
    d = report["drift"]
    log(f"stored CPU results: {d['status']} - {d['detail']}")
    for s in report.get("sessions", []):
        log(f"real recording {s['session']}: largest feature difference {s['feat_max']:.1e}, bursts {'identical' if s['segments_same'] else 'DIFFERENT'}")
    log("")
    log(f"echo loop: {report.get('echo_loop', 'PyTorch operations')}; hall noise shaped {report.get('noise_shaping', 'in the CPU workers')}; "
        f"feature filter: {report.get('feature_filter', 'conv')}")
    if report["passed"]:
        log(f"PARITY OK: the GPU path makes the same training pictures as the CPU path ({report['clips']} clips, {report['total_seconds']:.0f} s)")
    else:
        log("PARITY FAILED:")
        for f in report["failures"]:
            log(f"  - {f}")


def to_json(report: dict) -> str:
    slim = {k: v for k, v in report.items() if k != "packed"}
    slim["reference_sha256"] = str(report["packed"]["sha256"])
    return json.dumps(slim, indent=1, default=float)

"""replay.py — run a recording (or a simulated scene) through the pipeline and score it.

Plain words: this is the tuning bench for presence and breathing. It feeds a
recorded session, or the simulator's "busy hall" scene, through exactly the
same `Pipeline` the live engine uses and prints

  * one line per second: what presence and breathing said, plus the raw
    detector levels behind those decisions;
  * a score per labelled part of the session: how often "moving" fired while
    the person was supposed to be still, how often the breathing estimator had
    to start over, and the measured breathing rate against the paced rate that
    `sonar.record --script breathing` wrote into session.json.

How to run (from the repo root):
    python -m sonar.replay data_tune/<session folder>          one recorded session
    python -m sonar.replay data_tune                            every session under a folder
    python -m sonar.replay --sim hall --seconds 60 --profile clean     the simulated busy hall
    python -m sonar.replay data_tune/<session> --quiet          scores only
    python -m sonar.replay data_tune/<session> --taps           also where (which distance) things move
    python -m sonar.replay data_tune/<session> --noise          also how the room's noise is spread beside our band

A recording is replayed with its own SIGNAL settings (tones, chirp, channels)
but with the CURRENT processing settings, so a re-run shows the effect of a
code or threshold change.

What it changes: nothing. Read-only; prints to the terminal (and writes the
JSON report only if --json is given).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import SonarConfig, get_profile, profile_names
from .pipeline import Pipeline, PipelineFrame

# What was played and how it was recorded. Everything else in a stored config is a
# processing setting and is taken from the current code instead.
SIGNAL_KEYS = (
    "fs", "frame_len", "n_out", "tone_hz", "tone_channels", "tone_amp", "chirp_lo_hz", "chirp_hi_hz",
    "chirp_amp", "chirp_channel", "n_mics",
)
STILL_LABELS = ("present_still", "breathing")      # parts of a session where the person was asked to sit still


# ---------------------------------------------------------------------------
# One pass through the pipeline, keeping what the scores need
# ---------------------------------------------------------------------------
@dataclass
class Trace:
    """Per-frame read-outs of one replay (arrays are indexed by frame)."""

    cfg: SonarConfig
    t: np.ndarray                     # [T] stream time (s)
    state: np.ndarray                 # [T] presence state ("" while calibrating)
    moving: np.ndarray                # [T] bool
    conf: np.ndarray                  # [T] presence confidence
    range_m: np.ndarray               # [T] presence range (nan when empty)
    motionless_s: np.ndarray          # [T]
    fast_db: np.ndarray               # [T] gross-movement level, whole room
    slow_db: np.ndarray               # [T] micro-movement level
    near_db: np.ndarray               # [T]
    far_db: np.ndarray                # [T]
    activity_db: np.ndarray           # [T] pilot-tone (Doppler) movement energy
    breathing: list[tuple[float, Any]] = field(default_factory=list)   # (t, BreathingResult)
    resyncs: int = 0
    snr_db: float = 0.0

    def between(self, t0: float, t1: float) -> np.ndarray:
        return (self.t >= t0) & (self.t < t1)


def trace_frames(cfg: SonarConfig, frames: list[PipelineFrame], pipe: Pipeline | None = None) -> Trace:
    """Collect the per-frame read-outs of an already processed list of frames."""
    def col(get: Any, default: float) -> np.ndarray:
        return np.array([default if fr.presence is None else get(fr.presence) for fr in frames], dtype=np.float64)

    return Trace(
        cfg=cfg,
        t=np.array([fr.t for fr in frames]),
        state=np.array(["" if fr.presence is None else fr.presence.state for fr in frames]),
        moving=np.array([bool(fr.presence is not None and fr.presence.moving) for fr in frames]),
        conf=col(lambda p: p.confidence, 0.0),
        range_m=col(lambda p: np.nan if p.range_m is None else p.range_m, np.nan),
        motionless_s=col(lambda p: p.motionless_s, 0.0),
        fast_db=col(lambda p: p.fast_db, 0.0),
        slow_db=col(lambda p: p.slow_db, 0.0),
        near_db=col(lambda p: p.near_db, 0.0),
        far_db=col(lambda p: p.far_db, 0.0),
        activity_db=np.array([fr.activity_db for fr in frames]),
        breathing=[(fr.t, fr.breathing) for fr in frames if fr.breathing is not None],
        resyncs=0 if pipe is None else pipe.frontend.n_resync,
        snr_db=0.0 if pipe is None else pipe.snr_db(),
    )


def run_blocks(cfg: SonarConfig, blocks: Iterator[np.ndarray], n_mics: int | None = None, keep_cir: bool = False) -> tuple[Trace, list[PipelineFrame]]:
    """Push audio blocks through a fresh pipeline. Echo profiles are dropped unless `keep_cir`."""
    pipe = Pipeline(cfg, n_mics=n_mics)
    frames: list[PipelineFrame] = []
    for rx in blocks:
        for fr in pipe.process(rx):
            fr.window = None                       # the gesture pictures are not needed here
            fr.feat = None
            if not keep_cir:
                fr.cir = None
            frames.append(fr)
    return trace_frames(pipe.cfg, frames, pipe), frames


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
@dataclass
class Score:
    """How the detectors did over one stretch of time."""

    label: str
    t0: float
    t1: float
    paced_bpm: float | None = None
    present_share: float = 0.0        # share of the time presence said "present"
    empty_share: float = 0.0
    entering_share: float = 0.0
    moving_share: float = 0.0         # share of the time `moving` was true
    moving_onsets: int = 0            # times `moving` switched on
    state_changes: int = 0            # presence state changes
    conf_median: float = 0.0
    conf_low_share: float = 0.0       # share of the time presence confidence was under 0.3
    range_median: float = float("nan")
    range_spread: float = float("nan")   # 10th..90th percentile width of the reported range (m)
    breath_updates: int = 0
    tracking_share: float = 0.0       # share of breathing updates that carried a rate
    restarts: int = 0                 # breathing updates in state "moving" (the measurement started over)
    bpm_median: float | None = None
    bpm_err_median: float | None = None   # median |bpm - paced|
    bpm_err_max: float | None = None
    bpm_within_1: float | None = None     # share of reported rates within 1 breath/min of the paced rate
    first_bpm_s: float | None = None      # seconds from t0 to the first reported rate
    chest_range_median: float = float("nan")

    def line(self) -> str:
        paced = "" if self.paced_bpm is None else f" paced {self.paced_bpm:.0f}"
        out = (
            f"{self.label:<14}{self.t0:6.1f}-{self.t1:6.1f} s{paced:<9} | present {self.present_share:4.0%} empty {self.empty_share:4.0%}"
            f" entering {self.entering_share:4.0%} | moving {self.moving_share:5.1%} ({self.moving_onsets} onsets)"
            f" | conf med {self.conf_median:.2f} low {self.conf_low_share:4.0%} | range {self.range_median:.2f} m +-{self.range_spread / 2:.2f}"
        )
        bpm = "--" if self.bpm_median is None else f"{self.bpm_median:.1f}"
        out += f" | breathing {self.tracking_share:4.0%} tracked, {self.restarts} restarts, median {bpm}"
        if self.bpm_err_median is not None:
            out += f" (err med {self.bpm_err_median:.1f} max {self.bpm_err_max:.1f}, within 1: {self.bpm_within_1:.0%})"
        if self.first_bpm_s is not None:
            out += f", first rate after {self.first_bpm_s:.0f} s"
        return out


def score(trace: Trace, t0: float, t1: float, label: str = "", paced_bpm: float | None = None) -> Score:
    """Score the stretch t0..t1 (seconds of stream time)."""
    s = Score(label=label, t0=t0, t1=t1, paced_bpm=paced_bpm)
    m = trace.between(t0, t1) & (trace.state != "")
    if not np.any(m):
        return s
    state, moving = trace.state[m], trace.moving[m]
    s.present_share = float(np.mean(state == "present"))
    s.empty_share = float(np.mean(state == "empty"))
    s.entering_share = float(np.mean(state == "entering"))
    s.moving_share = float(np.mean(moving))
    s.moving_onsets = int(np.sum(moving[1:] & ~moving[:-1]))
    s.state_changes = int(np.sum(state[1:] != state[:-1]))
    s.conf_median = float(np.median(trace.conf[m]))
    s.conf_low_share = float(np.mean(trace.conf[m] < 0.3))
    rng = trace.range_m[m]
    rng = rng[np.isfinite(rng)]
    if len(rng):
        s.range_median = float(np.median(rng))
        s.range_spread = float(np.percentile(rng, 90) - np.percentile(rng, 10))
    upd = [(t, b) for t, b in trace.breathing if t0 <= t < t1]
    s.breath_updates = len(upd)
    if upd:
        rates = [(t, b.bpm) for t, b in upd if b.bpm is not None]
        s.tracking_share = len(rates) / len(upd)
        s.restarts = sum(b.state == "moving" for _, b in upd)
        chest = [b.range_m for _, b in upd if b.range_m is not None]
        if chest:
            s.chest_range_median = float(np.median(chest))
        if rates:
            vals = np.array([r for _, r in rates])
            s.bpm_median = float(np.median(vals))
            s.first_bpm_s = float(rates[0][0] - t0)
            if paced_bpm is not None:
                err = np.abs(vals - paced_bpm)
                s.bpm_err_median, s.bpm_err_max = float(np.median(err)), float(np.max(err))
                s.bpm_within_1 = float(np.mean(err <= 1.0))
    return s


# ---------------------------------------------------------------------------
# Sources: a recorded session, or the simulator
# ---------------------------------------------------------------------------
def session_config(meta: dict[str, Any], profile: str | None = None) -> SonarConfig:
    """Signal settings from the recording, processing settings from the current code."""
    base = get_profile(profile or meta.get("profile") or "default")
    stored = meta.get("config") or {}
    changes = {k: (tuple(v) if isinstance(v, list) else v) for k, v in stored.items() if k in SIGNAL_KEYS}
    return base.with_(**changes) if changes else base


def load_session(path: Path) -> tuple[Path, dict[str, Any]]:
    """(audio file, session metadata) for a session folder, a session.json or a bare .wav."""
    path = Path(path)
    if path.is_dir():
        meta_path, wav = path / "session.json", path / "audio.wav"
    elif path.suffix == ".json":
        meta_path, wav = path, path.with_name("audio.wav")
    else:
        meta_path, wav = path.with_name("session.json"), path
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("audio"):
        wav = meta_path.parent / meta["audio"]
    if not wav.exists():
        raise FileNotFoundError(f"no audio file at {wav}")
    return wav, meta


def wav_blocks(wav: Path, n_mics: int, block: int = 9600) -> Iterator[np.ndarray]:
    import soundfile as sf

    with sf.SoundFile(str(wav)) as f:
        while True:
            data = f.read(block, dtype="float32", always_2d=True)
            if len(data) == 0:
                return
            yield data[:, :n_mics]


def sim_blocks(cfg: SonarConfig, scene: str, seconds: float, seed: int = 0, cn0_db: float = 66.0, block: int = 9600) -> Iterator[np.ndarray]:
    """Audio of a simulated scene: 'hall' (seated person, fidgets, walkers, noise bursts),
    'hall_empty' (the same hall with nobody at the laptop) or 'quiet' (a still person, nothing else)."""
    from .sim import EchoSimulator, HallNoise, hall_reflectors, person_reflectors

    sim = EchoSimulator(cfg, seed=seed, cn0_db=cn0_db)
    noise = None
    if scene in ("hall", "hall_empty"):
        refl = hall_reflectors(seated=scene == "hall", seed=seed, t_end=seconds + 5.0)
        noise = HallNoise(cfg, sim._noise_sigma, seed=seed, t_end=seconds + 5.0)
    elif scene == "quiet":
        refl = person_reflectors(0.7, chest_db=-30.0, body_db=-28.0, seed=seed)
    else:
        raise ValueError(f"unknown simulated scene {scene!r}")
    for start in range(0, int(seconds * cfg.fs), block):
        rx = sim.render(block, refl, start_sample=start)
        yield rx if noise is None else noise.add(rx, start)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def label_at(events: list[dict[str, Any]], sample: float) -> str:
    for e in events:
        if e["start"] <= sample < e["end"]:
            return f"{e['label']}" + (f"@{e['bpm']:.0f}" if "bpm" in e else "")
    return ""


def print_timeline(trace: Trace, events: list[dict[str, Any]], every: float = 1.0) -> None:
    """One line per `every` seconds: presence and breathing read-outs plus the levels behind them."""
    cfg = trace.cfg
    print("    t  label            | presence  conf range mov still | fast  slow  near   far  dopp | breathing        bpm  raw conf range   amp  snr qual  win")
    upd = trace.breathing
    k = 0
    for t0 in np.arange(0.0, float(trace.t[-1]), every):
        m = trace.between(t0, t0 + every)
        if not np.any(m):
            continue
        i = int(np.flatnonzero(m)[-1])
        last = None
        while k < len(upd) and upd[k][0] < t0 + every:
            last = upd[k][1]
            k += 1
        lab = label_at(events, (t0 + 0.5 * every) * cfg.fs)
        st = trace.state[i] or "(calib)"
        rng = "  -- " if not np.isfinite(trace.range_m[i]) else f"{trace.range_m[i]:5.2f}"
        line = (
            f"{t0:5.0f}  {lab:<16} | {st:<9} {trace.conf[i]:4.2f} {rng} {100 * np.mean(trace.moving[m]):3.0f}% {trace.motionless_s[i]:5.1f}"
            f" | {np.max(trace.fast_db[m]):5.1f} {np.max(trace.slow_db[m]):5.1f} {np.max(trace.near_db[m]):5.1f} {np.max(trace.far_db[m]):5.1f} {np.max(trace.activity_db[m]):5.1f}"
        )
        if last is not None:
            bpm = "  -- " if last.bpm is None else f"{last.bpm:5.1f}"
            raw = "  -- " if last.raw_bpm is None else f"{last.raw_bpm:5.1f}"
            brng = "  -- " if last.range_m is None else f"{last.range_m:5.2f}"
            line += (
                f" | {last.state:<10} {bpm} {raw} {last.confidence:4.2f} {brng} {last.amplitude_mm:5.2f} {last.snr_db:4.1f} {last.quality:4.2f} {last.window_s:4.0f}"
            )
        print(line)


def print_tap_map(cfg: SonarConfig, frames: list[PipelineFrame], every: float = 5.0) -> None:
    """Where things move: fast (gross) and slow (breathing-band) change per distance, in dB over the median tap."""
    cir = [(fr.t, fr.cir) for fr in frames if fr.cir is not None]
    if not cir:
        return
    t = np.array([c[0] for c in cir])
    h = np.stack([c[1] for c in cir]).astype(np.complex128)             # [T, mics, taps]
    rate = cfg.frame_rate
    from scipy import signal as sps

    fast = sps.sosfilt(sps.butter(4, cfg.presence_fast_hp_hz, "highpass", fs=rate, output="sos"), h, axis=0)
    slow = sps.sosfiltfilt(sps.butter(2, [0.1, 0.6], "bandpass", fs=rate, output="sos"), h, axis=0)
    r = cfg.range_m
    cols = [i for i in range(cfg.direct_tap, cfg.n_taps, 3)]
    print("\nwhere things move (dB over the quietest quarter of taps); columns = distance in m")
    print("    t  kind  " + " ".join(f"{r[i]:4.1f}" for i in cols))
    for t0 in np.arange(0.0, float(t[-1]), every):
        m = (t >= t0) & (t < t0 + every)
        if np.sum(m) < 10:
            continue
        for kind, x in (("fast", fast), ("slow", slow), ("level", h)):
            p = np.mean(np.abs(x[m]) ** 2, axis=(0, 1))
            ref = np.percentile(p[cfg.direct_tap + 8:], 25) if kind != "level" else np.percentile(np.mean(np.abs(fast[m]) ** 2, axis=(0, 1)), 25)
            db = 10 * np.log10(p / ref + 1e-30)
            print(f"{t0:5.0f}  {kind:<5} " + " ".join(f"{db[i]:4.0f}" for i in cols))


def print_noise_bands(cfg: SonarConfig, blocks: Iterator[np.ndarray], n_mics: int) -> None:
    """How loud and how bursty the room is beside our own signal, band by band.

    The per-frame noise meter (`dsp.guard_bins`) reads these frequencies. A band whose level
    never moves is probably below the microphone's own noise and tells nothing about the
    room; a band that jumps by 10 dB or more in single frames is where clicks show.
    """
    from .dsp import Prefilter, guard_bins

    pre = Prefilter(cfg, n_mics)
    hz = guard_bins(cfg) * cfg.fs / cfg.frame_len
    edges = np.flatnonzero(np.diff(hz) > 2 * cfg.fs / cfg.frame_len) + 1
    bands = [b for chunk in np.split(np.arange(len(hz)), edges) for b in np.array_split(chunk, max(1, len(chunk) // 12))]
    rows: list[np.ndarray] = []
    tail = np.zeros((0, n_mics))
    for rx in blocks:
        y = np.concatenate([tail, pre.process(np.asarray(rx, dtype=np.float64))])
        n_frames = len(y) // cfg.frame_len
        for k in range(n_frames):
            spec = np.abs(np.fft.rfft(y[k * cfg.frame_len:(k + 1) * cfg.frame_len], axis=0)[guard_bins(cfg)]) ** 2
            rows.append(np.array([np.median(spec[b]) for b in bands]))
        tail = y[n_frames * cfg.frame_len:]
    if len(rows) < 100:
        return
    p = np.stack(rows[50:])                                             # skip the filter start-up
    med = np.median(p, axis=0)
    scale = (cfg.frame_len / 2) ** 2                                    # a full-scale sine reads 0 dB
    print("\nroom noise beside our band, per 20 ms frame (dB re full scale per bin; jumps are relative to the usual level)")
    print("   band (kHz)      usual level   90 % of frames under   99 % under   loudest frame")
    for j, b in enumerate(bands):
        r = 10 * np.log10(np.percentile(p[:, j], [90, 99, 100]) / med[j])
        print(f"   {hz[b[0]] / 1000:5.2f}-{hz[b[-1]] / 1000:5.2f}   {10 * np.log10(med[j] / scale + 1e-30):9.1f}   {r[0]:+17.1f} dB   {r[1]:+7.1f} dB   {r[2]:+10.1f} dB")
    both = np.corrcoef(np.log(p[:, 0]), np.log(p[:, -1]))[0, 1]
    print(f"   lowest and highest band move together: correlation {both:.2f} (near 1 = broadband clicks, near 0 = unrelated noise)")


def report(trace: Trace, events: list[dict[str, Any]], name: str) -> list[Score]:
    """Print the per-label scores of one replay and return them."""
    cfg = trace.cfg
    print(f"\n== {name}: {trace.t[-1]:.0f} s, signal headroom {trace.snr_db:.0f} dB, re-syncs {trace.resyncs} ==")
    scores: list[Score] = []
    for e in events:
        if e.get("kind") == "calibration":
            continue
        t0, t1 = e["start"] / cfg.fs, e["end"] / cfg.fs
        scores.append(score(trace, t0, t1, e["label"], e.get("bpm")))
    ready = np.flatnonzero(trace.state != "")
    if len(ready):
        scores.append(score(trace, float(trace.t[ready[0]]), float(trace.t[-1]) + 1e-3, "whole session"))
    for s in scores:
        print(s.line())
        if s.paced_bpm is not None and s.t1 - s.t0 > 25.0:
            # The analysis window holds up to 30 s, so the first part of a paced block still
            # contains the previous rate. The fair check is the end of the block.
            late = score(trace, s.t1 - 15.0, s.t1, "", s.paced_bpm)
            bpm = "no rate" if late.bpm_median is None else f"median {late.bpm_median:.1f} (error {late.bpm_err_median:.1f}, worst {late.bpm_err_max:.1f})"
            print(f"{'':14}  last 15 s of this block: {late.tracking_share:4.0%} tracked, {bpm}")
    still = [s for s in scores if s.label in STILL_LABELS]
    if still:
        total = sum(s.t1 - s.t0 for s in still)
        onsets = sum(s.moving_onsets for s in still)
        share = sum(s.moving_share * (s.t1 - s.t0) for s in still) / total
        print(f"while asked to sit still ({total:.0f} s): 'moving' fired {onsets} times ({60 * onsets / total:.1f} per minute), on for {share:.1%} of the time; "
              f"breathing restarts {sum(s.restarts for s in still)}")
    return scores


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sonar.replay", description="Replay a recording or a simulated scene and score presence + breathing")
    ap.add_argument("paths", nargs="*", help="session folder(s), a folder of sessions, or audio.wav")
    ap.add_argument("--sim", choices=["hall", "hall_empty", "quiet"], help="use the simulator instead of a recording")
    ap.add_argument("--seconds", type=float, default=60.0, help="length of the simulated scene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cn0", type=float, default=66.0, help="simulator signal-to-noise density (66 = the level measured on the HP)")
    ap.add_argument("--profile", choices=profile_names(), help="signal profile (default: the one stored in the session, else 'default')")
    ap.add_argument("--every", type=float, default=1.0, help="seconds per timeline line")
    ap.add_argument("--quiet", action="store_true", help="scores only, no timeline")
    ap.add_argument("--taps", action="store_true", help="also print where (at which distance) things move")
    ap.add_argument("--noise", action="store_true", help="also print how the room's noise is spread beside our band")
    ap.add_argument("--json", help="write the scores to this file")
    args = ap.parse_args(argv)

    # (name, config, a function that yields the audio blocks afresh, labelled parts, microphones)
    jobs: list[tuple[str, SonarConfig, Any, list[dict[str, Any]], int | None]] = []
    if args.sim:
        cfg = get_profile(args.profile or "default")
        events = [{"label": "present_still" if args.sim != "hall_empty" else "empty", "kind": "scene", "start": int(8 * cfg.fs),
                   "end": int(args.seconds * cfg.fs), **({"bpm": 15.0} if args.sim != "hall_empty" else {})}]
        jobs.append((f"sim:{args.sim} seed {args.seed}", cfg,
                     lambda cfg=cfg: sim_blocks(cfg, args.sim, args.seconds, args.seed, args.cn0), events, None))
    for raw in args.paths:
        root = Path(raw)
        found = sorted(p.parent for p in root.rglob("session.json")) if root.is_dir() and not (root / "session.json").exists() else [root]
        for p in found:
            try:
                wav, meta = load_session(p)
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                print(f"skipping {p}: {exc}", file=sys.stderr)
                continue
            cfg = session_config(meta, args.profile)
            import soundfile as sf

            n_mics = min(cfg.n_mics, sf.info(str(wav)).channels)
            jobs.append((str(p), cfg, lambda wav=wav, n_mics=n_mics: wav_blocks(wav, n_mics), meta.get("events", []), n_mics))
    if not jobs:
        ap.error("give a session path or --sim")

    out: dict[str, Any] = {}
    for name, cfg, blocks, events, n_mics in jobs:
        trace, frames = run_blocks(cfg, blocks(), n_mics=n_mics, keep_cir=args.taps)
        if len(trace.t) == 0:
            print(f"{name}: no audio", file=sys.stderr)
            continue
        if not args.quiet:
            print_timeline(trace, events, args.every)
        if args.taps:
            print_tap_map(trace.cfg, frames)
        if args.noise:
            print_noise_bands(trace.cfg, blocks(), trace.cfg.n_mics)
        scores = report(trace, events, name)
        out[name] = [s.__dict__ for s in scores]
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

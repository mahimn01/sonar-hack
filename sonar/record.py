"""record.py — collect labelled sonar recordings quickly.

Plain words: this plays the sonar signal, records the microphones the whole
time, and tells one person what to do on screen ("get ready ... GO: SWIPE
LEFT"). It saves ONE continuous raw recording per session plus a list of
"between sample A and sample B the person was asked to do X". Because the raw
audio is kept, features can be recomputed later with different settings.

After every trial it shows whether a movement was detected and what the
physics rules think it was. If that guess matches the prompt most of the time,
the hardware and the signal are fine.

How to run (Windows demo laptop, about 4 minutes per gesture session):
    py -3.12 -m sonar.record --person alice --script gestures --reps 12
    py -3.12 -m sonar.record --person alice --script scene        (presence + distractors, about 4 min)
    py -3.12 -m sonar.record --person alice --script breathing    (paced breathing, about 2.5 min)
Without hardware (simulator, for testing the whole chain):
    python -m sonar.record --person sim --backend sim --script gestures --reps 4 --fast

What it changes: creates data/<date-time>_<person>_<script>/ with
    audio.wav      raw microphone audio, 48 kHz, 24-bit, all mic channels
    session.json   who/when, the full signal config, device info, the prompt list
While it runs it holds the speaker and microphone exclusively.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .audio import AudioError, make_backend
from .config import GESTURE_LABELS, SonarConfig, get_profile, profile_names
from .dataset import AUDIO_FILE, SESSION_FILE
from .gestures import Segment, heuristic_classify
from .pipeline import Pipeline

HOW_TO = {
    "swipe_left": "sweep your open hand from RIGHT to LEFT across the screen, ~25 cm in front of it",
    "swipe_right": "sweep your open hand from LEFT to RIGHT across the screen, ~25 cm in front of it",
    "push": "push your open palm quickly TOWARDS the screen and take it straight back",
    "pull": "reach out slowly until your palm is ~15 cm from the screen, then pull it back QUICKLY",
}


class Recorder:
    """Records a backend to disk while running the live pipeline for feedback."""

    def __init__(self, cfg: SonarConfig, backend: Any, folder: Path, meta: dict[str, Any]) -> None:
        import soundfile as sf

        self.backend = backend
        self.pipeline = Pipeline(cfg, n_mics=getattr(backend, "n_mics", cfg.n_mics))
        self.cfg = self.pipeline.cfg
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        self._wav = sf.SoundFile(str(folder / AUDIO_FILE), "w", samplerate=cfg.fs, channels=self.cfg.n_mics, subtype="PCM_24")
        self.meta = dict(meta)
        self.events: list[dict[str, Any]] = []
        self.dropouts: list[int] = []
        self.bursts: list[tuple[Segment, str, float]] = []      # (burst, physics guess, its confidence)
        self.n_samples = 0
        self.peak = 0.0
        self.presence = ""
        self.last_breath: tuple[float | None, float] = (None, 0.0)
        self.error: BaseException | None = None
        self._cond = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None

    # -- background thread: audio -> disk + live pipeline -----------------------------------
    def _loop(self) -> None:
        try:
            while self._running:
                rx = self.backend.read(timeout=0.5)
                if rx is None:
                    continue
                if self.backend.pop_discontinuity():
                    self.dropouts.append(self.n_samples)
                    self.pipeline.notify_discontinuity()
                self._wav.write(rx)
                self.peak = max(self.peak, float(np.max(np.abs(rx))))
                for fr in self.pipeline.process(rx):
                    if fr.presence is not None:
                        self.presence = fr.presence.state
                    if fr.breathing is not None:
                        self.last_breath = (fr.breathing.bpm, fr.breathing.confidence)
                    if fr.segment is not None and fr.window is not None:
                        r = heuristic_classify(fr.window, self.cfg)
                        self.bursts.append((fr.segment, r.label, r.confidence))
                with self._cond:
                    self.n_samples += len(rx)
                    self._cond.notify_all()
        except BaseException as exc:
            self.error = exc
            with self._cond:
                self._cond.notify_all()

    def start(self) -> None:
        self.backend.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="sonar-record", daemon=True)
        self._thread.start()

    # -- timing in AUDIO time (so it also works faster than real time with the simulator) -----
    def wait_until(self, sample: int) -> None:
        with self._cond:
            while self.n_samples < sample:
                if self.error is not None:
                    raise AudioError(f"recording stopped: {self.error}")
                self._cond.wait(timeout=1.0)

    def wait(self, seconds: float) -> None:
        self.wait_until(self.n_samples + int(seconds * self.cfg.fs))

    @property
    def now_s(self) -> float:
        return self.n_samples / self.cfg.fs

    def mark(self, label: str, kind: str, start: int, end: int, **extra: Any) -> None:
        self.events.append({"label": label, "kind": kind, "start": int(start), "end": int(end), **extra})
        self.save_meta()                       # after every trial: a crash never loses the labels

    def bursts_between(self, start: int, end: int) -> list[tuple[Segment, str, float]]:
        f0, f1 = start // self.cfg.frame_len, end // self.cfg.frame_len + 1
        return [b for b in self.bursts if b[0].end >= f0 and b[0].start <= f1]

    def save_meta(self) -> None:
        meta = {
            **self.meta,
            "format": 1,
            "audio": AUDIO_FILE,
            "fs": self.cfg.fs,
            "n_mics": self.cfg.n_mics,
            "n_samples": self.n_samples,
            "config": self.cfg.to_dict(),
            "backend": getattr(self.backend, "info", {}),
            "dropouts": self.dropouts,
            "peak": round(self.peak, 5),
            "events": self.events,
        }
        tmp = self.folder / (SESSION_FILE + ".tmp")
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(self.folder / SESSION_FILE)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.backend.stop()
        self._wav.close()
        self.save_meta()


# ---------------------------------------------------------------------------
# Scripts (what the person is asked to do)
# ---------------------------------------------------------------------------
def _banner(text: str) -> None:
    print("\n" + "=" * 72 + f"\n  {text}\n" + "=" * 72, flush=True)


def calibrate(rec: Recorder, seconds: float = 6.0) -> None:
    _banner("CALIBRATION: sit normally, hands on your lap or the table, do not move")
    start = rec.n_samples
    rec.wait(seconds)
    rec.mark("calibration", "calibration", start, rec.n_samples)
    locked = rec.pipeline.frontend.locked
    print(f"  signal headroom {rec.pipeline.snr_db():.0f} dB, chirp lock {'OK' if locked else 'NOT LOCKED'}, "
          f"input peak {20 * np.log10(rec.peak + 1e-9):.0f} dBFS")
    if rec.peak >= 0.99:
        print("  WARNING: the microphone is clipping. Lower the speaker volume or the mic level.")
    if not locked or rec.pipeline.snr_db() < 25:
        print("  WARNING: weak or missing sonar signal. Run `python -m sonar.selftest` before recording.")


def run_gestures(
    rec: Recorder, gestures: list[str], reps: int, ready_s: float, trial_s: float, rest_s: float,
    rng: np.random.Generator, sim: Any | None,
) -> None:
    trials = [g for g in gestures for _ in range(reps)]
    order = rng.permutation(len(trials))                 # shuffled: classes must not line up with time
    _banner(f"GESTURES: {len(trials)} trials, about {len(trials) * (ready_s + trial_s + rest_s) / 60:.1f} min")
    for g in gestures:
        print(f"  {g:<12} {HOW_TO.get(g, '')}")
    print("  Start each gesture when you see GO. Afterwards let your hand DROP back slowly (no swipe back).")
    rec.wait(2.0)
    hits = 0
    agree = 0
    for n, k in enumerate(order, 1):
        label = trials[k]
        print(f"\n[{n}/{len(trials)}] get ready: {label.upper().replace('_', ' ')}", flush=True)
        rec.wait(ready_s)
        start = rec.n_samples
        print(f"        GO  >>>  {label.upper().replace('_', ' ')}", flush=True)
        if sim is not None:
            sim.perform(label)
        rec.wait(trial_s)
        end = rec.n_samples
        rec.wait(rest_s)                                  # also lets the burst close before we read it out
        found = rec.bursts_between(start, end)
        rec.mark(label, "gesture", start, end, rep=int(n))
        if found:
            seg, guess, conf = max(found, key=lambda b: b[0].peak_db)
            hits += 1
            agree += guess == label
            print(f"        movement detected (peak {seg.peak_db:.0f} dB, {seg.n_frames * 20} ms); physics guess: {guess} ({conf:.2f})")
        else:
            print("        NO movement detected: this trial will be skipped. Move closer / faster / larger.")
    print(f"\nmovement found in {hits}/{len(trials)} trials; physics guess agreed with the prompt in {agree}/{max(hits, 1)}")


SCENE_SCRIPT: list[tuple[str, float, str]] = [
    ("empty", 15.0, "EVERYONE step at least 2 m away from the laptop and stand still"),
    ("walk_in", 6.0, "WALK up to the laptop and sit down in front of it"),
    ("present_still", 25.0, "SIT STILL facing the laptop, breathe normally"),
    ("distractor", 25.0, "TYPE on the keyboard, use the touchpad, reach for a cup, scratch your head"),
    ("walk_out", 6.0, "STAND UP and walk at least 2 m away"),
    ("empty", 10.0, "stay away"),
    ("walk_in", 6.0, "WALK up to the laptop and stand in front of it"),
    ("present_still", 25.0, "STAND STILL facing the laptop"),
    ("distractor", 25.0, "lean in and out, turn to talk to a neighbour, pick something up, adjust your chair"),
    ("distractor", 25.0, "ask a teammate to WALK PAST behind you a few times while you sit still"),
    ("walk_out", 6.0, "walk away"),
    ("empty", 10.0, "stay away"),
]


def run_scene(rec: Recorder) -> None:
    total = sum(s for _, s, _ in SCENE_SCRIPT)
    _banner(f"SCENE: presence and distractors, about {total / 60:.1f} min. Follow the instructions.")
    for label, seconds, text in SCENE_SCRIPT:
        print(f"\n>>> {text}   ({seconds:.0f} s)   [{label}]", flush=True)
        start = rec.n_samples
        rec.wait(seconds)
        rec.mark(label, "scene", start, rec.n_samples)
        print(f"    detector said: presence = {rec.presence or 'n/a'}")


def run_breathing(rec: Recorder, rates: list[float], seconds: float, tty: bool) -> None:
    _banner(f"BREATHING: sit still facing the laptop, 0.5-0.8 m away. Follow the bar: {rates} breaths/min")
    for bpm in rates:
        print(f"\n>>> breathe at {bpm:.0f} per minute for {seconds:.0f} s (IN while the bar grows, OUT while it shrinks)", flush=True)
        rec.wait(3.0)
        start = rec.n_samples
        t0 = rec.now_s
        last_word = ""
        while rec.now_s - t0 < seconds:
            phase = ((rec.now_s - t0) * bpm / 60.0) % 1.0
            inhale = phase < 0.4                                    # 40 % in, 60 % out, like relaxed breathing
            level = phase / 0.4 if inhale else 1.0 - (phase - 0.4) / 0.6
            word = "IN " if inhale else "OUT"
            if tty:
                bar = "#" * int(30 * level)
                print(f"\r    {word} |{bar:<30}|", end="", flush=True)
            elif word != last_word:
                print(f"    {word}", flush=True)
            last_word = word
            rec.wait(0.1)
        rec.mark("breathing", "scene", start, rec.n_samples, bpm=float(bpm))
        bpm_est, conf = rec.last_breath
        shown = "--" if bpm_est is None else f"{bpm_est:.1f}"
        print(f"\n    estimator said: {shown} per minute (confidence {conf:.2f}); truth {bpm:.0f}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sonar.record", description="Record a labelled sonar session")
    ap.add_argument("--person", required=True, help="who is performing (used to hold out whole people for validation)")
    ap.add_argument("--script", choices=["gestures", "scene", "breathing", "all"], default="gestures")
    ap.add_argument("--reps", type=int, default=12, help="repetitions of each gesture")
    ap.add_argument("--gestures", default=",".join(g for g in GESTURE_LABELS if g != "none"))
    ap.add_argument("--out", default="data", help="folder that receives one sub-folder per session")
    ap.add_argument("--profile", choices=profile_names(), default="default")
    ap.add_argument("--backend", choices=["live", "sim"], default="live")
    ap.add_argument("--notes", default="", help="free text: seat distance, room, volume ...")
    ap.add_argument("--ready-s", type=float, default=1.0)
    ap.add_argument("--trial-s", type=float, default=2.5)
    ap.add_argument("--rest-s", type=float, default=1.2)
    ap.add_argument("--breath-rates", default="12,15,20")
    ap.add_argument("--breath-s", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--in-device")
    ap.add_argument("--out-device")
    ap.add_argument("--blocksize", type=int, default=1920)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--fast", action="store_true", help="simulator only: run faster than real time")
    args = ap.parse_args(argv)

    cfg = get_profile(args.profile)
    gestures = [g for g in args.gestures.split(",") if g]
    unknown = [g for g in gestures if g not in GESTURE_LABELS or g == "none"]
    if unknown:
        ap.error(f"unknown gestures {unknown}; choose from {[g for g in GESTURE_LABELS if g != 'none']}")
    seed = int(time.time()) % 100000 if args.seed is None else args.seed
    rng = np.random.default_rng(seed)
    try:
        if args.backend == "live":
            backend = make_backend("live", cfg, in_device=args.in_device, out_device=args.out_device,
                                   blocksize=args.blocksize, gain=args.gain)
        else:
            backend = make_backend("sim", cfg, scene="idle", realtime=not args.fast, seed=seed)
    except AudioError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    session_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{args.person}_{args.script}"
    folder = Path(args.out) / session_id
    meta = {
        "session_id": session_id, "person": args.person, "script": args.script, "notes": args.notes,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "profile": args.profile, "seed": seed,
        "software": __version__, "synthetic": args.backend == "sim",
    }
    rec = Recorder(cfg, backend, folder, meta)
    try:
        rec.start()
    except AudioError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"recording to {folder}  (profile {args.profile}, backend {getattr(backend, 'info', {})})")
    status = 0
    try:
        calibrate(rec)
        if args.script in ("gestures", "all"):
            run_gestures(rec, gestures, args.reps, args.ready_s, args.trial_s, args.rest_s, rng,
                         backend if args.backend == "sim" else None)
        if args.script in ("scene", "all"):
            run_scene(rec)
        if args.script in ("breathing", "all"):
            run_breathing(rec, [float(r) for r in args.breath_rates.split(",")], args.breath_s, sys.stdout.isatty())
        rec.wait(1.0)
    except KeyboardInterrupt:
        print("\ninterrupted: keeping what was recorded so far")
    except AudioError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        status = 1
    finally:
        rec.stop()
    print(f"\nsaved {rec.n_samples / cfg.fs:.0f} s, {len(rec.events)} labelled segments, {len(rec.dropouts)} audio dropouts -> {folder}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

"""engine.py — the realtime engine: microphone audio in, clean JSON events out.

Plain words: this is the program the demo runs. It

  1. opens the audio backend (laptop speakers + microphones, or the simulator,
     or a recorded file),
  2. pushes every block of samples through the pipeline (Doppler pictures,
     echo profile, presence, breathing),
  3. classifies each movement burst (trained model, or physics rules if there
     is no model yet) and applies the anti-jitter rules: confidence threshold,
     dead time, "ignore the hand coming back", "wind-up vs real stroke",
  4. emits events (gesture / presence / breathing / status / spectrogram) to
     every registered Python callback and to every WebSocket client on
     ws://127.0.0.1:8765. The exact event schema is in INTERFACES.md.

How to run (Windows demo laptop):
    py -3.12 -m sonar.engine                               live, model if models/gesture.onnx exists
    py -3.12 -m sonar.engine --physics                     live, physics rules only
    py -3.12 -m sonar.engine --backend sim                 no hardware: scripted simulated visitor
    py -3.12 -m sonar.engine --backend file --file data/<session>/audio.wav

What it changes: while running with the live backend it holds the speaker and
microphone exclusively and plays the near-inaudible sonar signal; it opens a
local WebSocket port. It writes no files.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import __version__
from .audio import AudioError, list_devices, make_backend
from .classifier import load_classifier
from .config import GESTURE_LABELS, SonarConfig, get_profile, profile_names
from .gestures import Segment
from .pipeline import Pipeline

SCHEMA_VERSION = 1
OPPOSITE = {"swipe_left": "swipe_right", "swipe_right": "swipe_left", "push": "pull", "pull": "push"}
Event = dict[str, Any]


class GestureDecider:
    """Turns classified movement bursts into at most one clean gesture each.

    Rules, in order:
      * (in Engine) the burst closed while somebody was walking in  -> ignored
      * (in Engine) the movement was beyond the hand zone (> 0.6 m)  -> ignored
      * "none" or confidence below `min_confidence`               -> ignored
      * within `seg_refractory_frames` of the last gesture        -> ignored
      * the opposite of the last gesture within 1.2 s             -> ignored (the hand coming back)
      * swipes are held for `swipe_hold_s` (0.3 s): if an opposite swipe of at
        least similar strength follows straight away, the first one was the
        wind-up and only the second is reported; if a clearly weaker opposite
        swipe follows, it is the hand returning and only the first is reported.
    """

    def __init__(
        self,
        cfg: SonarConfig,
        classifier: Any,
        min_confidence: float = 0.6,
        swipe_hold_s: float = 0.3,
        prefer_later_margin_db: float = 0.5,
    ) -> None:
        self.cfg = cfg
        self.classifier = classifier
        self.min_confidence = min_confidence
        self._hold = int(round(swipe_hold_s * cfg.frame_rate))
        self._max_wait = int(1.0 * cfg.frame_rate)
        self._margin = prefer_later_margin_db
        self._pending: dict[str, Any] | None = None
        self._last_index = -10**9
        self._last_label = ""
        self.last_decision: dict[str, Any] = {}

    def _blocked(self, label: str, index: int) -> bool:
        since = index - self._last_index
        if since < self.cfg.seg_refractory_frames:
            return True
        return OPPOSITE.get(self._last_label) == label and since < self.cfg.seg_opposite_lockout_frames

    def _emit(self, cand: dict[str, Any]) -> list[Event]:
        if self._blocked(cand["label"], cand["end"]):
            return []
        self._last_index = cand["end"]
        self._last_label = cand["label"]
        return [
            {
                "type": "gesture",
                "label": cand["label"],
                "confidence": round(cand["confidence"], 3),
                "source": self.classifier.source,
                "probs": {k: round(v, 3) for k, v in cand["probs"].items()},
                "duration_s": round(cand["duration_s"], 2),
                "peak_db": round(cand["peak_db"], 1),
            }
        ]

    def on_segment(self, index: int, segment: Segment, window: np.ndarray) -> list[Event]:
        label, conf, probs = self.classifier.predict(window)
        self.last_decision = {"label": label, "confidence": conf, "peak_db": segment.peak_db, "index": index}
        if label == "none" or conf < self.min_confidence:
            return []
        cand = {
            "label": label,
            "confidence": conf,
            "probs": probs,
            "peak_db": segment.peak_db,
            "duration_s": segment.n_frames / self.cfg.frame_rate,
            "end": segment.end,
            "release": index + self._hold,
        }
        if not label.startswith("swipe") or self._hold == 0:
            out: list[Event] = []
            if self._pending is not None:
                out += self._emit(self._pending)
                self._pending = None
            return out + self._emit(cand)

        if self._pending is None:
            if not self._blocked(label, segment.end):
                self._pending = cand
            return []
        first = self._pending
        if OPPOSITE[first["label"]] == label:
            if cand["peak_db"] >= first["peak_db"] - self._margin:
                self._pending = cand            # wind-up, then the real stroke: keep the later one
                return []
            self._pending = None                # stroke, then the weaker return: keep the first
            return self._emit(first)
        self._pending = cand                    # same direction twice: report both
        return self._emit(first)

    def tick(self, index: int, segment_active: bool) -> list[Event]:
        """Call every frame: releases a held swipe once nothing more is coming."""
        p = self._pending
        if p is None or index < p["release"]:
            return []
        if segment_active and index < p["release"] + self._max_wait:
            return []                           # another movement is in progress: wait for its verdict
        self._pending = None
        return self._emit(p)


class Engine:
    def __init__(
        self,
        cfg: SonarConfig,
        backend: Any,
        classifier: Any | None = None,
        min_confidence: float = 0.6,
        swipe_hold_s: float = 0.3,
        emit_spectrogram: bool = True,
        profile: str = "custom",
    ) -> None:
        self.cfg = cfg
        self.backend = backend
        self.pipeline = Pipeline(cfg, n_mics=getattr(backend, "n_mics", cfg.n_mics))
        self.cfg = self.pipeline.cfg
        self.classifier = classifier if classifier is not None else load_classifier(None, self.cfg)
        self.decider = GestureDecider(self.cfg, self.classifier, min_confidence, swipe_hold_s)
        self.emit_spectrogram = emit_spectrogram
        self.profile = profile
        self._callbacks: list[Callable[[Event], None]] = []
        self._seq = 0
        self._rows: list[list[float]] = []
        self._last_presence_emit = -10**9
        self._fps = float(self.cfg.frame_rate)
        self._fps_mark = (time.monotonic(), 0)
        self._frames = 0
        self._dropped = 0
        self._clipped = False
        self._thread: threading.Thread | None = None
        self._running = False
        self._want_recalibrate = False
        self._entering_until = -1
        self._entering_since = -1                      # frame at which the current "entering" spell began
        self.gate_entering = True                      # ignore bursts while somebody is walking in
        # ... but only for this many seconds after they appeared (None = for as long as the presence detector
        # says "entering"). A restless person can keep that state for a long time: in one recorded session it
        # lasted 41 s and swallowed the first 8 prompted gestures.
        self.entering_gate_s: float | None = None
        self.gate_range = True                         # ignore bursts whose movement is beyond the hand zone
        self._zones: deque[tuple[int, float, float]] = deque(maxlen=200)
        self._far_until = -1
        self._bins_hz = [round(float(f), 2) for f in self.cfg.doppler_hz]
        self._last_presence: Any = None
        self._proc_ms = 0.0
        self.min_peak_db = self.cfg.seg_min_peak_db    # bursts weaker than this are never gestures
        self.error: BaseException | None = None

    # -- public API ----------------------------------------------------------------------
    def set_classifier(self, classifier: Any) -> None:
        """Swap the gesture classifier (model or physics rules) and adopt what its training recorded."""
        self.classifier = classifier
        self.decider.classifier = classifier
        meta = getattr(classifier, "meta", {})
        # Real gestures in the training recordings had at least `gesture_peak_db_p10`; anything
        # 6 dB weaker than that (a person walking past far away) is not worth classifying.
        if "gesture_peak_db_p10" in meta:
            self.min_peak_db = max(self.cfg.seg_min_peak_db, float(meta["gesture_peak_db_p10"]) - 6.0)

    def on_event(self, callback: Callable[[Event], None]) -> None:
        """Register a function that receives every event dict (called from the engine thread)."""
        self._callbacks.append(callback)

    def recalibrate(self) -> None:
        """Learn the noise floors again at the next block (thread-safe)."""
        self._want_recalibrate = True

    def hello(self) -> Event:
        """Static description sent to every WebSocket client when it connects."""
        cfg = self.cfg
        return {
            "type": "hello",
            "schema": SCHEMA_VERSION,
            "version": __version__,
            "t": time.time(),
            "labels": [label for label in GESTURE_LABELS if label != "none"],
            "presence_states": ["empty", "entering", "present"],
            "breathing_states": ["tracking", "acquiring", "moving", "no_signal"],
            "frame_rate": cfg.frame_rate,
            "tones_hz": list(cfg.tone_hz),
            "chirp_band_hz": [cfg.chirp_lo_hz, cfg.chirp_hi_hz],
            "doppler_range_hz": [float(cfg.doppler_hz[0]), float(cfg.doppler_hz[-1])],
            "doppler_bins": cfg.doppler_bins,
            "profile": self.profile,
            "backend": getattr(self.backend, "name", "custom"),
            "source": getattr(self.backend, "name", "custom"),
            "classifier": self.classifier.source,
        }

    def handle_command(self, cmd: dict[str, Any]) -> Event | None:
        """Commands a WebSocket client may send."""
        name = cmd.get("cmd")
        if name == "calibrate":
            self.recalibrate()
            return {"type": "ack", "cmd": "calibrate"}
        if name == "hello":
            return self.hello()
        if name == "set_min_confidence":
            try:
                self.decider.min_confidence = float(np.clip(float(cmd.get("value")), 0.0, 1.0))
            except (TypeError, ValueError):
                return {"type": "error", "message": "set_min_confidence needs a numeric 'value'"}
            return {"type": "ack", "cmd": name, "value": self.decider.min_confidence}
        return {"type": "error", "message": f"unknown command {name!r}"}

    # -- event construction ---------------------------------------------------------------
    def _stamp(self, events: list[Event]) -> list[Event]:
        now = time.time()
        for e in events:
            self._seq += 1
            e["t"] = round(now, 3)
            e["seq"] = self._seq
        return events

    def _status(self) -> Event:
        cfg = self.cfg
        last = self.decider.last_decision
        return {
            "type": "status",
            "snr_db": round(self.pipeline.snr_db(), 1),
            "carrier_hz": cfg.tone_hz[0],
            "fps": round(self._fps, 1),
            "calibrating": bool(self.pipeline.calibrating),
            "locked": bool(self.pipeline.frontend.locked),
            "drift_ppm": round(self.pipeline.frontend._chirp.drift_ppm, 2),
            "clipped": self._clipped,
            "dropped": self._dropped,
            "resyncs": self.pipeline.frontend.n_resync,
            "activity_db": round(self.pipeline.segmenter.activity_db, 1),
            "classifier": self.classifier.source,
            "backend": getattr(self.backend, "name", "custom"),
            "source": getattr(self.backend, "name", "custom"),
            "device": getattr(self.backend, "info", {}).get("input", getattr(self.backend, "name", "")),
            "latency_ms": round(self._proc_ms + 1000.0 * getattr(self.backend, "block", getattr(self.backend, "blocksize", 1920)) / cfg.fs, 1),
            "min_peak_db": round(self.min_peak_db, 1),
            "profile": self.profile,
            "tones_hz": list(cfg.tone_hz),
            "chirp_band_hz": [cfg.chirp_lo_hz, cfg.chirp_hi_hz],
            "last_decision": {
                "label": last.get("label"),
                "confidence": round(float(last.get("confidence", 0.0)), 3),
                "peak_db": round(float(last.get("peak_db", 0.0)), 1),
            },
            "version": __version__,
        }

    # -- processing --------------------------------------------------------------------------
    def step(self, rx: np.ndarray) -> list[Event]:
        """Process one block of microphone samples synchronously and return the events it produced."""
        cfg = self.cfg
        fps = int(round(cfg.frame_rate))
        if self._want_recalibrate:
            self._want_recalibrate = False
            self.pipeline.recalibrate()
        events: list[Event] = []
        for fr in self.pipeline.process(rx):
            self._frames += 1
            self._clipped |= fr.clipped

            p = fr.presence
            # Somebody walking up to the laptop is not a gesture: ignore bursts while the
            # presence state is "entering" and for half a second after.
            if p is not None and p.state == "entering":
                if self._entering_since < 0:
                    self._entering_since = fr.index
                if self.entering_gate_s is None or fr.index - self._entering_since <= self.entering_gate_s * fps:
                    self._entering_until = fr.index + fps // 2
            elif p is not None:
                self._entering_since = -1
            if p is not None:
                self._zones.append((fr.index, p.near_db, p.far_db))
            if fr.segment is not None and fr.window is not None:
                blocked = ""
                if fr.segment.peak_db < self.min_peak_db:
                    blocked = "weak"                    # far weaker than a real gesture on this set-up
                elif self.gate_entering and fr.index < self._entering_until:
                    blocked = "entering"
                elif self.gate_range and fr.index < self._far_until:
                    blocked = "far"                     # somebody was just walking by: wait 2 s
                elif self.gate_range:
                    # Range gate (from the chirp): where was the gross movement during this burst?
                    # A hand moves in the near zone; a body walking past or away moves further out.
                    zone = [(n, f) for i, n, f in self._zones if fr.segment.start <= i <= fr.segment.end]
                    if zone:
                        near, far = max(z[0] for z in zone), max(z[1] for z in zone)
                        if far > cfg.gate_far_db and far - near > cfg.gate_far_margin_db:
                            blocked = "far"
                            self._far_until = fr.index + 2 * fps
                if blocked:
                    self.decider.last_decision = {"label": f"blocked:{blocked}", "confidence": 0.0, "peak_db": fr.segment.peak_db}
                else:
                    events += self.decider.on_segment(fr.index, fr.segment, fr.window)
            events += self.decider.tick(fr.index, self.pipeline.segmenter.active)

            if p is not None and (p.changed or fr.index - self._last_presence_emit >= fps):
                self._last_presence_emit = fr.index
                events.append(
                    {
                        "type": "presence",
                        "state": p.state,
                        "confidence": round(p.confidence, 3),
                        "range_m": p.range_m,
                        "moving": p.moving,
                        "motionless_s": round(p.motionless_s, 1),
                        "changed": p.changed,
                    }
                )

            b = fr.breathing
            if b is not None:
                events.append(
                    {
                        "type": "breathing",
                        "bpm": None if b.bpm is None else round(b.bpm, 1),
                        "confidence": round(b.confidence, 3),
                        "state": b.state,
                        "waveform": [round(float(v), 3) for v in b.waveform],
                        "fs": b.fs,
                        "range_m": None if b.range_m is None else round(b.range_m, 2),
                        "amplitude_mm": round(b.amplitude_mm, 2),
                        "window_s": round(b.window_s, 1),
                        # somebody is there, did not walk away, and nothing at all has moved for 5 s
                        "apnea": bool(p is not None and p.state == "present" and p.motionless_s >= 5.0),
                        "motionless_s": 0.0 if p is None else round(p.motionless_s, 1),
                    }
                )

            if self.emit_spectrogram and fr.feat is not None:
                row = fr.feat[:cfg.n_power_channels].mean(axis=0)
                self._rows.append([round(float(v), 3) for v in row])
                if len(self._rows) >= 5:                    # 10 messages per second, 5 time slices each
                    events.append(
                        {
                            "type": "spectrogram",
                            "rows": self._rows,
                            "n_bins": cfg.doppler_bins,
                            "bins_hz": self._bins_hz,
                            "frame_s": cfg.frame_len / cfg.fs,
                        }
                    )
                    self._rows = []

            if self._frames % fps == 0:
                now = time.monotonic()
                t0, n0 = self._fps_mark
                if now > t0:
                    self._fps = (self._frames - n0) / (now - t0)
                self._fps_mark = (now, self._frames)
                events.append(self._status())
                self._clipped = False
        return self._stamp(events)

    def _dispatch(self, events: list[Event]) -> None:
        for e in events:
            for cb in self._callbacks:
                try:
                    cb(e)
                except Exception as exc:                    # a broken listener must not stop the sonar
                    print(f"[engine] event callback failed: {exc!r}", file=sys.stderr)

    def _loop(self) -> None:
        try:
            while self._running:
                rx = self.backend.read(timeout=0.5)
                if rx is None:
                    if getattr(self.backend, "finished", False):
                        break
                    continue
                if self.backend.pop_discontinuity():
                    self._dropped += 1
                    self.pipeline.notify_discontinuity()
                t0 = time.perf_counter()
                events = self.step(rx)
                self._proc_ms += 0.1 * (1000.0 * (time.perf_counter() - t0) - self._proc_ms)
                self._dispatch(events)
        except BaseException as exc:
            self.error = exc
            self._dispatch(self._stamp([{"type": "error", "message": str(exc)}]))
        finally:
            self._running = False

    def start(self) -> None:
        """Open the audio backend and start processing in a background thread."""
        self.backend.start()
        self._running = True
        self._fps_mark = (time.monotonic(), 0)
        self._thread = threading.Thread(target=self._loop, name="sonar-engine", daemon=True)
        self._thread.start()

    @property
    def running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.backend.stop()


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _console_printer(verbose: bool) -> Callable[[Event], None]:
    def show(e: Event) -> None:
        kind = e["type"]
        stamp = time.strftime("%H:%M:%S")
        if kind == "gesture":
            print(f"{stamp}  GESTURE   {e['label']:<12} confidence {e['confidence']:.2f}  ({e['source']}, peak {e['peak_db']} dB)")
        elif kind == "presence" and e["changed"]:
            print(f"{stamp}  PRESENCE  {e['state']:<9} range {e['range_m']} m")
        elif kind == "breathing" and (verbose or e["bpm"] is not None):
            bpm = "--" if e["bpm"] is None else f"{e['bpm']:.1f}"
            print(f"{stamp}  BREATHING {bpm} per min  confidence {e['confidence']:.2f}  {e['state']}")
        elif kind == "status" and verbose:
            print(
                f"{stamp}  status    snr {e['snr_db']} dB  fps {e['fps']}  calibrating {e['calibrating']}  "
                f"locked {e['locked']}  drift {e['drift_ppm']} ppm  activity {e['activity_db']} dB"
            )
        elif kind == "error":
            print(f"{stamp}  ERROR     {e['message']}", file=sys.stderr)
    return show


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sonar.engine", description="Laptop sonar realtime engine")
    ap.add_argument("--backend", choices=["live", "sim", "file"], default="live", help="audio source (default: live)")
    ap.add_argument("--profile", choices=profile_names(), default="default", help="signal profile; must match the model")
    ap.add_argument("--model", default="models/gesture.onnx", help="trained model; physics rules are used if it is missing")
    ap.add_argument("--physics", action="store_true", help="ignore the model and use the physics rules")
    ap.add_argument("--min-confidence", type=float, default=0.6, help="gestures below this confidence are dropped")
    ap.add_argument("--swipe-hold", type=float, default=0.3, help="seconds a swipe is held to tell wind-up from stroke (0 = off)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-range-gate", action="store_true", help="do not veto bursts whose movement is beyond the hand zone")
    ap.add_argument("--no-entering-gate", action="store_true", help="do not veto bursts while somebody is walking in")
    ap.add_argument("--entering-gate-s", type=float,
                    help="veto bursts only for this many seconds after somebody appears (default: as long as they are 'entering')")
    ap.add_argument("--min-peak-db", type=float, help="ignore bursts weaker than this (default: from the model, else 3 dB)")
    ap.add_argument("--no-ws", action="store_true", help="do not start the WebSocket server (console only)")
    ap.add_argument("--no-spectrogram", action="store_true", help="do not send spectrogram events")
    ap.add_argument("--in-device", help="microphone: index or part of its name")
    ap.add_argument("--out-device", help="speaker: index or part of its name")
    ap.add_argument("--blocksize", type=int, default=1920, help="audio block in samples (1920 = 40 ms)")
    ap.add_argument("--gain", type=float, default=1.0, help="scales the transmitted signal (0..1)")
    ap.add_argument("--shared", action="store_true", help="Windows: do NOT use exclusive mode (the mic is low-passed; debugging only)")
    ap.add_argument("--duplex", action="store_true", help="one duplex stream instead of separate input and output streams")
    ap.add_argument("--file", help="recorded audio.wav for --backend file")
    ap.add_argument("--scene", default="demo", choices=["demo", "idle", "empty"], help="simulator scene")
    ap.add_argument("--list-devices", action="store_true", help="print the audio devices and exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print status and unconfident breathing lines")
    args = ap.parse_args(argv)

    if args.list_devices:
        for d in list_devices():
            print(f"[{d['index']:>2}] {d['name']}  ({d['hostapi']}, in {d['in']}, out {d['out']}, {d['default_fs']:.0f} Hz)")
        return 0

    cfg = get_profile(args.profile)
    try:
        if args.backend == "live":
            backend = make_backend(
                "live", cfg, in_device=args.in_device, out_device=args.out_device, exclusive=not args.shared,
                blocksize=args.blocksize, mode="duplex" if args.duplex else "auto", gain=args.gain,
            )
        elif args.backend == "sim":
            backend = make_backend("sim", cfg, scene=args.scene)
        else:
            if not args.file:
                ap.error("--backend file needs --file path/to/audio.wav")
            backend = make_backend("file", cfg, path=args.file)
        engine = Engine(
            cfg, backend, min_confidence=args.min_confidence, swipe_hold_s=args.swipe_hold,
            emit_spectrogram=not args.no_spectrogram, profile=args.profile,
        )
        engine.gate_range = not args.no_range_gate
        engine.gate_entering = not args.no_entering_gate
        engine.entering_gate_s = args.entering_gate_s
        engine.set_classifier(load_classifier(args.model, engine.cfg, force_physics=args.physics))
        if args.min_peak_db is not None:
            engine.min_peak_db = args.min_peak_db
    except (AudioError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    mismatch = getattr(engine.classifier, "mismatch", [])
    if mismatch:
        print(f"WARNING: the model was trained with different settings for {mismatch}; retrain or switch profile.")
    engine.on_event(_console_printer(args.verbose))

    server = None
    if not args.no_ws:
        from .server import EventServer

        server = EventServer(args.host, args.port, on_command=engine.handle_command, hello=engine.hello)
        try:
            server.start()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        engine.on_event(server.broadcast)

    try:
        engine.start()
    except AudioError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if server is not None:
            server.stop()
        return 2
    info = getattr(backend, "info", {})
    print(f"sonar engine {__version__}: backend {args.backend} {info}")
    print(f"classifier: {engine.classifier.source}" + (f" ({Path(args.model).name})" if engine.classifier.source == "model" else ""))
    if server is not None:
        print(f"events on ws://{args.host}:{args.port}  (schema in INTERFACES.md)")
    print("calibrating for about 5 s: keep hands away from the laptop ... Ctrl+C to stop.")
    try:
        while engine.running:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        engine.stop()
        if server is not None:
            server.stop()
    if engine.error is not None:
        print(f"ERROR: {engine.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

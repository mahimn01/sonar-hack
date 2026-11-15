"""Mock sonar engine: realistic fake events over a local WebSocket.

What it does
    Pretends to be the real sonar engine so the slides, the Care Monitor and Sonar Care
    can be built and rehearsed without the laptop's speaker and mic. It streams the
    engine's event schema (INTERFACES.md: hello, gesture, presence, breathing, status,
    spectrogram) on ws://127.0.0.1:8765 and marks everything as simulated
    ("source": "mock"), so the UI shows a "Simulated data" tag and nobody mistakes it
    for real sensing. Like the trained model it knows three gestures: swipe left,
    swipe right and push.

Why
    The real engine only works on the demo MacBook's own speakers and microphone. This
    lets the app layer be developed, tested and rehearsed anywhere.

How to run
    python app/mock_engine.py                      scripted 2-minute loop: someone walks in,
                                                   breathes, gestures, holds their breath, leaves
    python app/mock_engine.py --scenario care      80-second Sonar Care story: arrive, swipe, push, sit still
                                                   (the no-movement alert), a double push (help), leave
    python app/mock_engine.py --scenario steady    person present + gestures every ~8 s (slide rehearsal)
    python app/mock_engine.py --scenario quiet     person present, breathing only
    python app/mock_engine.py --interactive        you fire events from the keyboard (press ? for keys)

    Any connected client can also send JSON commands, used by the tests:
        {"cmd": "gesture", "label": "push", "confidence": 0.9}      {"cmd": "double_push"}
        {"cmd": "presence", "state": "entering"}                    {"cmd": "moving", "seconds": 2}
        {"cmd": "apnea", "on": true}      {"cmd": "freeze", "on": true}      {"cmd": "bpm", "value": 18}
        {"cmd": "snr", "value": 18}       {"cmd": "calibrate"}

What it changes
    Nothing on disk. Opens one local network port (127.0.0.1 only unless --host is given).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field

from websockets.asyncio.server import ServerConnection, broadcast, serve
from websockets.exceptions import ConnectionClosed

VERSION = "mock-1.1"
GESTURES = ("swipe_left", "swipe_right", "push")       # the classes the trained model has (no pull, no wave)
TONES_HZ = [18500.0, 19000.0]                          # left speaker, right speaker (sonar/config.py)
CHIRP_HZ = [19500.0, 21500.0]
KEY_HELP = """
Keys (interactive mode)
  Left / a      swipe_left        Right / d     swipe_right
  Up / w        push              Down / s      two pushes (the help gesture)
  x             low-confidence gesture (should be ignored by the UI)
  1 empty   2 someone entering   3 person present
  m             the person moves for 2 s (resets the no-movement count)
  h             hold breath on/off (triggers the no-breathing alert after ~10 s)
  f             freeze on/off: nothing moves at all, not even breathing
  + / -         breathing rate up / down by 2 per minute
  c             3-second calibration
  ?             this help         q             quit
"""


@dataclass
class Gesture:
    label: str
    start: float          # wall-clock seconds when the hand starts moving
    duration: float
    confidence: float
    sent: bool = False    # the event goes out once the motion is mostly done


@dataclass
class World:
    """Everything the fake room is doing right now."""

    rng: random.Random
    bins: int = 64
    fs: float = 10.0                       # breathing waveform sample rate (Hz), as in sonar/config.py
    presence: str = "empty"
    presence_since: float = field(default_factory=time.time)
    rate_bpm: float = 14.0                 # true breathing rate
    est_bpm: float | None = None           # what the "estimator" reports (lags, like a real one)
    apnea: bool = False
    apnea_since: float = 0.0
    calibrating_until: float = 0.0
    phase: float = 0.0
    amp: float = 0.0                       # current chest-motion amplitude (0 when nobody is there)
    sample_t: float = 0.0                  # time of the last generated waveform sample
    samples: list[tuple[float, float]] = field(default_factory=list)   # recent (t, displacement)
    gestures: list[Gesture] = field(default_factory=list)
    auto_settle_at: float = 0.0            # "entering" turns into "present" at this time
    snr_db: float = 55.0
    snr_target: float = 55.0               # what the signal quality drifts toward (the "snr" command changes it)
    moving_until: float = 0.0              # the person is making a big movement until this time
    frozen: bool = False                   # nothing moves at all, not even breathing
    last_any_motion: float = field(default_factory=time.time)   # for the engine's "motionless_s" counter
    range_m: float = 0.72

    def is_moving(self, now: float) -> bool:
        """Gross body movement right now: walking in, a shift in the chair, a gesture in progress."""
        if self.presence == "entering" or now < self.moving_until:
            return True
        return any(0 <= now - g.start <= g.duration for g in self.gestures)

    def motionless_s(self, now: float) -> float:
        """Like the engine: seconds since ANYTHING moved (breathing counts), 0 while someone breathes or moves."""
        if self.presence == "empty":
            return 0.0
        if not self.frozen:
            self.last_any_motion = now
        return max(0.0, now - self.last_any_motion)

    # ---- breathing ---------------------------------------------------------------

    def _target_amp(self) -> float:
        if self.presence == "present":
            if self.frozen:
                return 0.0
            return 0.05 if self.apnea else 1.0
        if self.presence == "entering":
            return 0.0
        return 0.0

    def advance_breathing(self, now: float) -> None:
        """Generate chest-displacement samples at fs up to `now`."""
        dt = 1.0 / self.fs
        if self.sample_t == 0.0:
            self.sample_t = now - dt
        while self.sample_t + dt <= now:
            self.sample_t += dt
            # Amplitude eases toward its target (a held breath fades over ~1.5 s).
            self.amp += (self._target_amp() - self.amp) * (1 - math.exp(-dt / 0.6))
            jitter = 1 + 0.04 * math.sin(self.sample_t / 7.3)          # natural rate wobble
            self.phase += 2 * math.pi * (self.rate_bpm / 60.0) * jitter * dt
            # Inhale is quicker than exhale: a sine plus a little second harmonic.
            shape = math.sin(self.phase) + 0.22 * math.sin(2 * self.phase + 0.7)
            noise = self.rng.gauss(0, 0.045)
            drift = 0.08 * math.sin(self.sample_t / 11.0)
            self.samples.append((self.sample_t, self.amp * shape * 0.82 + noise + drift * self.amp))
        cutoff = now - 32.0
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.pop(0)

    def breathing_estimate(self, now: float) -> tuple[float | None, float, str]:
        """(bpm, confidence, state) the way the engine's windowed estimator reports it.

        Mirrors sonar/breathing.py: "moving" while someone walks, "acquiring" for the
        first 10 s of stillness, then "tracking". A held breath keeps the old rate for
        a few seconds (the window still holds earlier breaths) before it gives up.
        """
        if self.presence == "entering" or (self.presence == "present" and now < self.moving_until):
            return None, 0.0, "moving"
        if self.presence != "present" or self.frozen:
            return None, 0.0, "no_signal"
        settled = now - max(self.presence_since, self.moving_until)
        if settled < 10.0:                                  # needs ~10 s of stillness first
            return None, round(0.1 + 0.03 * settled, 2), "acquiring"
        if self.apnea:
            held = now - self.apnea_since
            if held > 9.0:
                return None, 0.1, "acquiring"
            return self.est_bpm, max(0.25, 0.9 - 0.08 * held), "tracking"
        self.est_bpm = self.rate_bpm if self.est_bpm is None else self.est_bpm + (self.rate_bpm - self.est_bpm) * 0.15
        return round(self.est_bpm + self.rng.gauss(0, 0.15), 1), round(min(0.97, 0.82 + self.rng.random() * 0.12), 2), "tracking"

    # ---- spectrogram -----------------------------------------------------------------

    def spectro_row(self, now: float) -> list[float]:
        n, c = self.bins, self.bins // 2
        row = [0.035 + 0.05 * self.rng.random() for _ in range(n)]

        def bump(center: float, height: float, width: float) -> None:
            lo, hi = max(0, int(center - 4 * width)), min(n, int(center + 4 * width) + 1)
            for i in range(lo, hi):
                row[i] += height * math.exp(-0.5 * ((i - center) / width) ** 2)

        bump(c, 0.26, 0.7)                                   # leftover direct path at zero Doppler
        if self.presence == "present":
            vel = math.cos(self.phase) * self.amp            # chest velocity follows the breathing phase
            bump(c + 1.4 * vel, 0.22, 1.1)
        if self.presence == "entering":                      # torso plus swinging limbs
            torso = 5.5 + 1.5 * math.sin(now * 2.1)
            bump(c + torso, 0.7, 1.6)
            for _ in range(3):
                bump(c + self.rng.uniform(-6, 15), self.rng.uniform(0.15, 0.45), 1.3)
        for g in self.gestures:
            tau = now - g.start
            if 0 <= tau <= g.duration:
                x = tau / g.duration
                if g.label == "push":
                    v = 14 * math.sin(math.pi * x)
                elif g.label == "swipe_right":
                    v = 11 * math.sin(2 * math.pi * x)
                else:
                    v = -11 * math.sin(2 * math.pi * x)
                fade = math.sin(math.pi * x) ** 0.5
                bump(c + v, 0.95 * fade, 2.0)
                bump(c + 0.45 * v, 0.35 * fade, 3.0)         # forearm and body follow more slowly
        return [round(min(1.0, max(0.0, v)), 3) for v in row]


# The "care" scenario: (second in the 80 s loop, key or command). Chosen to show every part of Sonar Care:
# touch-free control (two swipes and a push), a long sit (the no-movement alert at the 15 s demo setting),
# a double push (the help gesture) and leaving.
CARE_STORY: list[tuple[float, str | dict]] = [
    (6.0, "2"),                                   # someone walks up (settles into "present" 3 s later)
    (14.0, "right"), (19.0, "left"), (24.0, "up"),      # swipe right, swipe left, one push
    (47.0, "m"),                                  # 23 s of sitting still, then a shift in the chair
    (52.0, {"cmd": "double_push"}),               # help
    (66.0, "m"),
    (70.0, "2"), (72.5, "1"),                     # gets up, walks away, room empty
]


class MockEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.care_step = -1
        self.rng = random.Random(args.seed)
        self.world = World(rng=self.rng, bins=args.bins)
        self.clients: set[ServerConnection] = set()
        self.t0 = time.time()
        self.next_auto_gesture = self.t0 + 6.0
        self.scenario = "manual" if args.interactive else args.scenario
        self.last_breath_sent = 0.0
        self.log_enabled = not args.quiet
        if self.scenario not in ("demo", "care"):
            w = self.world
            w.presence, w.presence_since = "present", self.t0 - 25.0
            w.amp, w.sample_t = 1.0, self.t0 - 25.0           # 25 s of breathing history already "recorded"

    # ---- output ---------------------------------------------------------------------

    def log(self, msg: str) -> None:
        if self.log_enabled:
            print(f"[mock {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def send(self, payload: dict) -> None:
        if self.clients:
            broadcast(self.clients, json.dumps(payload, separators=(",", ":")))

    # ---- state changes (keyboard, commands and scenarios all come through here) ----------

    def set_presence(self, state: str, now: float) -> None:
        w = self.world
        if state == w.presence:
            return
        w.presence, w.presence_since = state, now
        if state == "entering":
            w.auto_settle_at = now + 3.0
        if state != "present":
            w.apnea = False
            w.frozen = False
            w.est_bpm = None
        w.last_any_motion = now
        self.log(f"presence -> {state}")
        self.send(self.presence_message(now, changed=True))

    def presence_message(self, now: float, changed: bool = False) -> dict:
        """One presence event with the engine's fields (INTERFACES.md)."""
        w = self.world
        still = w.motionless_s(now)
        conf = 0.86 + self.rng.random() * 0.1
        if w.presence == "present":
            conf *= 1.0 - 0.5 * min(still / 5.0, 1.0)      # like the engine: confidence sags while nothing moves at all
        return {"type": "presence", "state": w.presence, "confidence": round(conf, 2),
                "range_m": None if w.presence == "empty" else round(w.range_m + self.rng.gauss(0, 0.01), 2),
                "moving": w.presence != "empty" and w.is_moving(now), "motionless_s": round(still, 1),
                "changed": changed, "t": now}

    def set_moving(self, seconds: float, now: float) -> None:
        w = self.world
        if w.presence == "empty":
            self.log("movement needs someone in range: press 3 first")
            return
        w.moving_until = now + max(0.2, seconds)
        w.frozen = False
        self.log(f"the person moves for {seconds:.1f} s")

    def set_frozen(self, on: bool, now: float) -> None:
        w = self.world
        if on and w.presence != "present":
            self.log("freeze needs someone present: press 3 first")
            return
        if w.frozen != on:
            w.frozen, w.last_any_motion = on, now
            self.log("freeze ON: nothing moves at all" if on else "moving again")

    def fire_gesture(self, label: str, now: float, confidence: float | None = None) -> None:
        if label not in GESTURES:
            self.log(f"unknown gesture {label!r} ignored")
            return
        conf = confidence if confidence is not None else round(0.78 + self.rng.random() * 0.2, 2)
        self.world.gestures.append(Gesture(label, now, 0.6 if label.startswith("swipe") else 0.5, conf))
        self.log(f"gesture {label} ({conf:.2f})")

    def set_apnea(self, on: bool, now: float) -> None:
        w = self.world
        if w.presence != "present" and on:
            self.log("breath hold needs someone present: press 3 first")
            return
        if w.apnea != on:
            w.apnea, w.apnea_since = on, now
            self.log("breath hold ON (alert expected after ~10 s)" if on else "breathing resumed")

    def set_rate(self, bpm: float) -> None:
        self.world.rate_bpm = max(4.0, min(40.0, bpm))
        self.log(f"breathing rate -> {self.world.rate_bpm:.0f}/min")

    def calibrate(self, now: float, seconds: float = 3.0) -> None:
        self.world.calibrating_until = now + seconds
        self.log(f"calibrating for {seconds:.0f} s")

    def handle_command(self, cmd: dict) -> None:
        now = time.time()
        kind = str(cmd.get("cmd", "")).lower()
        if kind == "gesture":
            conf = cmd.get("confidence")
            self.fire_gesture(str(cmd.get("label", "")), now, float(conf) if conf is not None else None)
        elif kind == "presence":
            self.set_presence(str(cmd.get("state", "")), now)
        elif kind == "apnea":
            self.set_apnea(bool(cmd.get("on", True)), now)
        elif kind == "bpm":
            self.set_rate(float(cmd.get("value", 14)))
        elif kind == "calibrate":
            self.calibrate(now, float(cmd.get("seconds", 3)))
        elif kind == "double_push":
            self.fire_gesture("push", now, 0.95)
            self.fire_gesture("push", now + float(cmd.get("gap", 1.3)), 0.95)
        elif kind == "moving":
            self.set_moving(float(cmd.get("seconds", 2.0)), now)
        elif kind == "freeze":
            self.set_frozen(bool(cmd.get("on", True)), now)
        elif kind == "snr":
            self.world.snr_target = max(0.0, min(80.0, float(cmd.get("value", 55))))
        elif kind in ("ping", "hello", "set_min_confidence"):
            pass                                              # real-engine commands: nothing to do here
        else:
            self.log(f"unknown command {cmd!r}")

    def handle_key(self, key: str) -> None:
        now = time.time()
        k = key.lower()
        mapping = {"left": "swipe_left", "a": "swipe_left", "right": "swipe_right", "d": "swipe_right",
                   "up": "push", "w": "push"}
        if k in mapping:
            self.fire_gesture(mapping[k], now)
        elif k in ("down", "s", "double_push"):
            self.handle_command({"cmd": "double_push"})
        elif k in GESTURES:
            self.fire_gesture(k, now)
        elif k in ("m", "move"):
            self.set_moving(2.0, now)
        elif k in ("f", "freeze"):
            self.set_frozen(not self.world.frozen, now)
        elif k == "x":
            self.fire_gesture(self.rng.choice(GESTURES), now, confidence=0.42)
        elif k in ("1", "empty"):
            self.set_presence("empty", now)
        elif k in ("2", "entering"):
            self.set_presence("entering", now)
        elif k in ("3", "present"):
            self.set_presence("present", now)
        elif k in ("h", "hold"):
            self.set_apnea(not self.world.apnea, now)
        elif k in ("+", "="):
            self.set_rate(self.world.rate_bpm + 2)
        elif k in ("-", "_"):
            self.set_rate(self.world.rate_bpm - 2)
        elif k in ("c", "calibrate"):
            self.calibrate(now)
        elif k in ("?", "help"):
            print(KEY_HELP, flush=True)
        elif k in ("q", "\x03", "quit"):
            raise KeyboardInterrupt
        elif k.strip():
            self.log(f"key {k!r} does nothing (press ? for help)")

    # ---- scripted scenarios ------------------------------------------------------------------

    def run_scenario(self, now: float) -> None:
        w = self.world
        if self.scenario == "demo":
            # A 2-minute story that loops: arrival, breathing, gestures, a breath hold, departure.
            s = (now - self.t0) % 120.0
            if s < 0.2 and w.calibrating_until < now:
                self.calibrate(now, 3.0)
            want = "empty"
            if 10 <= s < 13:
                want = "entering"
            elif 13 <= s < 110:
                want = "present"
            elif 110 <= s < 112:
                want = "entering"
            self.set_presence(want, now)
            hold = 70 <= s < 84
            if w.presence == "present" and hold != w.apnea:
                self.set_apnea(hold, now)
            if s < 1:
                w.rate_bpm = 14.0
            elif 84 <= s < 85:
                w.rate_bpm = 16.0
            gesture_window = 18 <= s < 52
        elif self.scenario == "care":
            # An 80-second Sonar Care story that loops. Each step fires once, when its second comes up.
            s = (now - self.t0) % 80.0
            step = int(s * 4)                                 # quarter-second steps
            if step != self.care_step:
                self.care_step = step
                if s < 0.25:
                    self.calibrate(now, 3.0)
                    w.rate_bpm = 15.0
                for at, what in CARE_STORY:
                    if int(at * 4) == step:
                        self.handle_key(what) if isinstance(what, str) else self.handle_command(what)
            gesture_window = False
        else:
            if w.presence != "present":
                self.set_presence("present", now)
            gesture_window = self.scenario == "steady"
        if gesture_window and now >= self.next_auto_gesture:
            low = self.rng.random() < 0.15
            self.fire_gesture(self.rng.choice(GESTURES), now, confidence=0.45 if low else None)
            self.next_auto_gesture = now + self.rng.uniform(6.0, 8.5)
        elif not gesture_window:
            self.next_auto_gesture = max(self.next_auto_gesture, now + 4.0)

    # ---- main loop ------------------------------------------------------------------------------

    async def tick_loop(self) -> None:
        fps = self.args.fps
        frame = 1.0 / fps
        loop = asyncio.get_running_loop()
        next_at = loop.time()
        rows: list[list[float]] = []
        last_status = last_presence = 0.0
        while True:
            now = time.time()
            w = self.world
            if self.scenario != "manual":
                self.run_scenario(now)
            if w.presence == "entering" and w.auto_settle_at and now >= w.auto_settle_at:
                w.auto_settle_at = 0.0
                leaving = self.scenario == "care" and (now - self.t0) % 80.0 >= 70.0
                if not leaving:
                    self.set_presence("present", now)

            # Spectrogram: one row per frame, sent two at a time.
            w.advance_breathing(now)
            rows.append(w.spectro_row(now))
            if len(rows) >= 2:
                self.send({"type": "spectrogram", "rows": rows, "t": now})
                rows = []

            # Gesture events go out once most of the motion has happened (like a real classifier).
            for g in w.gestures:
                if not g.sent and now - g.start >= g.duration * 0.8:      # also true only once a future gesture has begun
                    g.sent = True
                    self.send({"type": "gesture", "label": g.label, "confidence": g.confidence, "t": now})
            w.gestures = [g for g in w.gestures if now - g.start < g.duration + 0.2]

            # Breathing: like the engine, once a second with the whole analysis window
            # (since the person settled, at most 30 s) scaled to its own peak; or, with
            # --waveform chunk, twice a second with only the new samples.
            window_mode = self.args.waveform == "window"
            if now - self.last_breath_sent >= (1.0 if window_mode else 0.5):
                bpm, conf, state = w.breathing_estimate(now)
                still = w.motionless_s(now)
                # Like the engine: "stopped_s" starts ~4 s after the last breath, and the flag goes up at 6 (about 10 s in).
                stopped = max(0.0, now - w.apnea_since - 4.0) if w.apnea else 0.0
                msg: dict = {"type": "breathing", "bpm": bpm, "confidence": round(conf, 2), "state": state, "fs": w.fs, "t": now,
                             "apnea": w.presence == "present" and (still >= 5.0 or stopped >= 6.0), "motionless_s": round(still, 1),
                             "stopped_s": round(stopped, 1), "held": False}
                if w.presence == "present" and w.samples:
                    if window_mode:
                        since = max(w.presence_since, now - 30.0)
                        chunk = [s for s in w.samples if s[0] >= since]
                    else:
                        chunk = [s for s in w.samples if s[0] > self.last_breath_sent]
                    if chunk:
                        values = [v for _, v in chunk]
                        if window_mode:
                            # Real chest movement over the window (3rd-97th percentile, 1 unit = 4 mm),
                            # sent alongside the peak-scaled waveform exactly like sonar/breathing.py.
                            ordered = sorted(values)
                            lo = ordered[int(0.03 * (len(ordered) - 1) + 0.5)]
                            hi = ordered[int(0.97 * (len(ordered) - 1) + 0.5)]
                            msg["amplitude_mm"] = round(4.0 * (hi - lo), 2)
                            peak = max(abs(v) for v in values) or 1.0
                            values = [v / peak for v in values]
                            msg["window_s"] = round(len(values) / w.fs, 1)
                            msg["range_m"] = round(0.72 + self.rng.gauss(0, 0.01), 2)
                        msg["waveform"] = [round(v, 4) for v in values]
                        msg["t"] = chunk[-1][0]                 # t = time of the last sample
                self.send(msg)
                self.last_breath_sent = now

            # Presence and status at 1 Hz.
            if now - last_presence >= 1.0:
                last_presence = now
                self.send(self.presence_message(now))
            if now - last_status >= 1.0:
                last_status = now
                w.snr_db += (w.snr_target - w.snr_db) * 0.35 + self.rng.gauss(0, 0.4)
                status = {"type": "status", "snr_db": round(w.snr_db, 1), "fps": fps,
                          "calibrating": now < w.calibrating_until, "locked": True, "clipped": False, "dropped": 0,
                          "activity_db": round(14.0 if w.is_moving(now) else abs(self.rng.gauss(0.4, 0.5)), 1),
                          "noise_db": round(abs(self.rng.gauss(0.6, 0.3)), 1), "classifier": "mock", "backend": "mock",
                          "source": "mock", "device": "mock engine", "latency_ms": 42.0, "version": VERSION, "t": now}
                if self.args.tones:
                    status["carrier_hz"] = self.args.tones[0]
                    status["tones_hz"] = self.args.tones
                if self.args.chirp:
                    status["chirp_band_hz"] = self.args.chirp
                self.send(status)

            next_at += frame
            delay = next_at - loop.time()
            if delay < -0.5:                                  # fell far behind (laptop asleep?): resync
                next_at = loop.time()
                delay = 0
            await asyncio.sleep(max(0.0, delay))

    async def handler(self, ws: ServerConnection) -> None:
        self.clients.add(ws)
        peer = ws.remote_address[0] if ws.remote_address else "?"
        self.log(f"client connected from {peer} ({len(self.clients)} total)")
        try:
            await ws.send(json.dumps({"type": "hello", "schema": 1, "version": VERSION, "t": time.time(), "labels": list(GESTURES),
                                      "presence_states": ["empty", "entering", "present"],
                                      "breathing_states": ["tracking", "acquiring", "moving", "no_signal"],
                                      "frame_rate": self.args.fps, "tones_hz": self.args.tones, "chirp_band_hz": self.args.chirp,
                                      "backend": "mock", "source": "mock", "classifier": "mock"}))
            async for message in ws:
                try:
                    cmd = json.loads(message)
                except (TypeError, json.JSONDecodeError):
                    self.log(f"ignored non-JSON message: {str(message)[:60]!r}")
                    continue
                if isinstance(cmd, dict):
                    self.handle_command(cmd)
        except ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            self.log(f"client disconnected ({len(self.clients)} left)")


# ---- keyboard input (runs in a background thread) ------------------------------------------

def _read_keys_windows(post) -> None:
    import msvcrt
    arrows = {"K": "left", "M": "right", "H": "up", "P": "down"}
    while True:
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            post(arrows.get(msvcrt.getwch(), ""))
        else:
            post(ch)


def _read_keys_posix(post) -> None:
    arrows = {"A": "up", "B": "down", "C": "right", "D": "left"}
    fd = sys.stdin.fileno()
    while True:
        ch = os.read(fd, 1).decode(errors="ignore")
        if ch == "\x1b":
            seq = os.read(fd, 2).decode(errors="ignore")
            post(arrows.get(seq[-1:], "") if seq.startswith("[") else "")
        else:
            post(ch)


def _read_lines(post) -> None:
    for line in sys.stdin:
        post(line.strip())


async def main(args: argparse.Namespace) -> None:
    engine = MockEngine(args)
    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    def post(key: str) -> None:
        def run() -> None:
            try:
                engine.handle_key(key)
            except KeyboardInterrupt:
                if not stop.done():
                    stop.set_result(None)
        loop.call_soon_threadsafe(run)

    if args.interactive:
        if sys.platform == "win32" and sys.stdin.isatty():
            reader = _read_keys_windows
        elif sys.stdin.isatty():
            reader = _read_keys_posix
        else:
            reader = _read_lines                               # piped input: one command per line
        threading.Thread(target=reader, args=(post,), daemon=True).start()

    async with serve(engine.handler, args.host, args.port, max_size=2**20) as server:
        engine.log(f"serving ws://{args.host}:{args.port}  scenario={engine.scenario}  waveform={args.waveform}")
        if args.interactive:
            print(KEY_HELP, flush=True)
        ticker = asyncio.create_task(engine.tick_loop())
        try:
            await stop
        finally:
            ticker.cancel()
            server.close()


def _hz_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fake sonar engine for UI development and rehearsal.")
    p.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1; use 0.0.0.0 for phones on the LAN)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--scenario", choices=("demo", "care", "steady", "quiet"), default="demo",
                   help="demo = 2-minute story loop, care = 80 s Sonar Care story, steady = present + gestures, quiet = present, breathing only")
    p.add_argument("--interactive", "-i", action="store_true", help="fire events from the keyboard")
    p.add_argument("--waveform", choices=("window", "chunk"), default="window",
                   help="window = whole analysis window each second, peak-scaled (like sonar/breathing.py); chunk = only new samples")
    p.add_argument("--fps", type=float, default=50.0, help="spectrogram frames per second (engine: 50)")
    p.add_argument("--bins", type=int, default=64, help="Doppler bins per spectrogram row")
    p.add_argument("--tones", type=_hz_list, default=TONES_HZ, help="pilot tones reported in status, e.g. 18500,19000 ('' for none)")
    p.add_argument("--chirp", type=_hz_list, default=CHIRP_HZ, help="chirp band reported in status, e.g. 19500,21500 ('' for none)")
    p.add_argument("--seed", type=int, default=None, help="random seed for repeatable runs")
    p.add_argument("--quiet", action="store_true", help="no per-event logging")
    return p.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    term_state = None
    if arguments.interactive and sys.platform != "win32" and sys.stdin.isatty():
        # cbreak mode: single key presses without Enter, Ctrl+C still works. Restored on exit.
        import termios
        import tty
        term_state = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
    try:
        asyncio.run(main(arguments))
    except KeyboardInterrupt:
        pass
    except OSError as err:
        print(f"[mock] could not start: {err}. Is something else already using port {arguments.port}?", file=sys.stderr)
        sys.exit(1)
    finally:
        if term_state is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, term_state)
        print("[mock] stopped", flush=True)

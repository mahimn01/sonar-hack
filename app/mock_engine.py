"""Mock sonar engine: realistic fake events over a local WebSocket.

What it does
    Pretends to be the real sonar engine so the slides and the Care Monitor can
    be built and rehearsed without the laptop's speaker and mic. It streams the
    provisional event schema (gesture, presence, breathing, status, spectrogram)
    on ws://127.0.0.1:8765 and marks everything as simulated ("source": "mock"),
    so the UI shows a "Simulated data" tag and nobody mistakes it for real sensing.

Why
    The real engine (sonar/) is being built in parallel. This lets the app layer
    be developed, tested and rehearsed independently.

How to run
    python app/mock_engine.py                      scripted 2-minute loop: someone walks in,
                                                   breathes, gestures, holds their breath, leaves
    python app/mock_engine.py --scenario steady    person present + gestures every ~8 s (slide rehearsal)
    python app/mock_engine.py --scenario quiet     person present, breathing only
    python app/mock_engine.py --interactive        you fire events from the keyboard (press ? for keys)

    Any connected client can also send JSON commands, used by the tests:
        {"cmd": "gesture", "label": "push", "confidence": 0.9}
        {"cmd": "presence", "state": "entering"}
        {"cmd": "apnea", "on": true}      {"cmd": "bpm", "value": 18}      {"cmd": "calibrate"}

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

VERSION = "mock-1.0"
GESTURES = ("swipe_left", "swipe_right", "push", "pull")
KEY_HELP = """
Keys (interactive mode)
  Left / a      swipe_left        Right / d     swipe_right
  Up / w        push              Down / s      pull (wave)
  x             low-confidence gesture (should be ignored by the UI)
  1 empty   2 someone entering   3 person present
  h             hold breath on/off (triggers the no-breathing alert after ~10 s)
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

    # ---- breathing ---------------------------------------------------------------

    def _target_amp(self) -> float:
        if self.presence == "present":
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
        if self.presence == "entering":
            return None, 0.0, "moving"
        if self.presence != "present":
            return None, 0.0, "no_signal"
        settled = now - self.presence_since
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
                elif g.label == "pull":
                    v = -14 * math.sin(math.pi * x)
                elif g.label == "swipe_right":
                    v = 11 * math.sin(2 * math.pi * x)
                else:
                    v = -11 * math.sin(2 * math.pi * x)
                fade = math.sin(math.pi * x) ** 0.5
                bump(c + v, 0.95 * fade, 2.0)
                bump(c + 0.45 * v, 0.35 * fade, 3.0)         # forearm and body follow more slowly
        return [round(min(1.0, max(0.0, v)), 3) for v in row]


class MockEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        self.world = World(rng=self.rng, bins=args.bins)
        self.clients: set[ServerConnection] = set()
        self.t0 = time.time()
        self.next_auto_gesture = self.t0 + 6.0
        self.scenario = "manual" if args.interactive else args.scenario
        self.last_breath_sent = 0.0
        self.log_enabled = not args.quiet
        if self.scenario != "demo":
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
            w.est_bpm = None
        self.log(f"presence -> {state}")
        self.send({"type": "presence", "state": state, "confidence": round(0.85 + self.rng.random() * 0.12, 2), "t": now})

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
        else:
            self.log(f"unknown command {cmd!r}")

    def handle_key(self, key: str) -> None:
        now = time.time()
        k = key.lower()
        mapping = {"left": "swipe_left", "a": "swipe_left", "right": "swipe_right", "d": "swipe_right",
                   "up": "push", "w": "push", "down": "pull", "s": "pull"}
        if k in mapping:
            self.fire_gesture(mapping[k], now)
        elif k in ("swipe_left", "swipe_right", "push", "pull"):
            self.fire_gesture(k, now)
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
                self.set_presence("present", now)

            # Spectrogram: one row per frame, sent two at a time.
            w.advance_breathing(now)
            rows.append(w.spectro_row(now))
            if len(rows) >= 2:
                self.send({"type": "spectrogram", "rows": rows, "t": now})
                rows = []

            # Gesture events go out once most of the motion has happened (like a real classifier).
            for g in w.gestures:
                if not g.sent and now - g.start >= g.duration * 0.8:
                    g.sent = True
                    self.send({"type": "gesture", "label": g.label, "confidence": g.confidence, "t": now})
            w.gestures = [g for g in w.gestures if now - g.start < g.duration + 0.2]

            # Breathing: like the engine, once a second with the whole analysis window
            # (since the person settled, at most 30 s) scaled to its own peak; or, with
            # --waveform chunk, twice a second with only the new samples.
            window_mode = self.args.waveform == "window"
            if now - self.last_breath_sent >= (1.0 if window_mode else 0.5):
                bpm, conf, state = w.breathing_estimate(now)
                msg: dict = {"type": "breathing", "bpm": bpm, "confidence": round(conf, 2), "state": state, "fs": w.fs, "t": now}
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
                self.send({"type": "presence", "state": w.presence, "confidence": round(0.86 + self.rng.random() * 0.1, 2), "t": now})
            if now - last_status >= 1.0:
                last_status = now
                w.snr_db += (55.0 - w.snr_db) * 0.2 + self.rng.gauss(0, 0.4)
                status = {"type": "status", "snr_db": round(w.snr_db, 1), "fps": fps,
                          "calibrating": now < w.calibrating_until, "source": "mock", "t": now}
                if self.args.tones:
                    status["tone_hz"] = self.args.tones
                if self.args.chirp:
                    status["chirp_hz"] = self.args.chirp
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
            await ws.send(json.dumps({"type": "hello", "source": "mock", "version": VERSION, "schema": "provisional", "t": time.time()}))
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
    p.add_argument("--scenario", choices=("demo", "steady", "quiet"), default="demo",
                   help="demo = 2-minute story loop, steady = present + gestures, quiet = present, breathing only")
    p.add_argument("--interactive", "-i", action="store_true", help="fire events from the keyboard")
    p.add_argument("--waveform", choices=("window", "chunk"), default="window",
                   help="window = whole analysis window each second, peak-scaled (like sonar/breathing.py); chunk = only new samples")
    p.add_argument("--fps", type=float, default=50.0, help="spectrogram frames per second (engine: 50)")
    p.add_argument("--bins", type=int, default=64, help="Doppler bins per spectrogram row")
    p.add_argument("--tones", type=_hz_list, default=[18000.0, 18500.0], help="pilot tones reported in status, e.g. 18000,18500 ('' for none)")
    p.add_argument("--chirp", type=_hz_list, default=[19000.0, 21500.0], help="chirp band reported in status, e.g. 19000,21500 ('' for none)")
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

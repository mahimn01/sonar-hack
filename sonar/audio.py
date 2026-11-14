"""audio.py — where the microphone samples come from: laptop, simulator or file.

Plain words: three interchangeable "backends" with the same four methods
(`start`, `read`, `stop`, `pop_discontinuity`):

  LiveAudio   plays the sonar signal on the laptop speakers and records the
              microphones. On Windows it opens BOTH devices in WASAPI exclusive
              mode, because in shared mode Windows low-passes the microphone and
              the ultrasound never arrives (measured on the demo laptop).
  SimAudio    the physics simulator pretending to be a laptop (works anywhere,
              used for tests and for developing the UI without hardware).
  FileAudio   replays a recorded session WAV.

What LiveAudio changes on the machine: it holds the speaker and microphone
exclusively while running (other apps cannot play or record) and plays a
near-inaudible 18.5-21.5 kHz signal. Nothing is written to disk.

How playback and recording are lined up: they are two separate streams, so
their offset is unknown and different on every start. We never rely on it:
the receiver finds the direct speaker->microphone sound in the recording and
aligns to that (see dsp.ChirpReceiver). PortAudio does not report dropped
buffers in WASAPI mode, so the receiver also notices by itself when the
alignment jumps and locks again.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import SonarConfig
from .sim import (
    HAND_REST,
    EchoSimulator,
    Reflector,
    WaypointPath,
    breathing_displacement,
    gesture_waypoints,
    hand_reflector,
    min_jerk,
    person_reflectors,
    sway_displacement,
)
from .tx import fade_ramp, make_tx_frame

# Device name fragments tried in order (first match wins). The first entries are
# the demo laptop (HP ENVY x360 15); the rest are generic fallbacks.
DEFAULT_OUTPUT_NAMES = ("Speaker (Realtek", "Speakers (Realtek", "Speaker", "Speakers", "Lautsprecher")
DEFAULT_INPUT_NAMES = ("Microphone Array (Intel", "Microphone Array", "Microphone", "Mikrofon")


class AudioError(RuntimeError):
    """The audio devices could not be opened or stopped delivering samples."""


# ---------------------------------------------------------------------------
# Device lookup
# ---------------------------------------------------------------------------
def list_devices() -> list[dict[str, Any]]:
    """All audio devices with their host API name (for --list-devices)."""
    import sounddevice as sd

    apis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        out.append(
            {
                "index": i,
                "name": d["name"],
                "hostapi": apis[d["hostapi"]]["name"],
                "in": d["max_input_channels"],
                "out": d["max_output_channels"],
                "default_fs": d["default_samplerate"],
            }
        )
    return out


def find_device(kind: str, wanted: str | int | None = None, prefer_hostapi: str | None = None) -> dict[str, Any]:
    """Pick an input or output device by index, by name fragment, or by the built-in name list.

    kind: "input" or "output". On Windows only WASAPI devices are considered
    (exclusive mode exists only there). Raises AudioError with the device list
    if nothing matches.
    """
    key = "in" if kind == "input" else "out"
    if prefer_hostapi is None and sys.platform == "win32":
        prefer_hostapi = "WASAPI"
    devices = [d for d in list_devices() if d[key] > 0]
    pool = [d for d in devices if prefer_hostapi is None or prefer_hostapi.lower() in d["hostapi"].lower()] or devices
    if isinstance(wanted, int) or (isinstance(wanted, str) and wanted.isdigit()):
        for d in devices:
            if d["index"] == int(wanted):
                return d
        raise AudioError(f"no {kind} device with index {wanted}")
    names = [wanted] if wanted else list(DEFAULT_INPUT_NAMES if kind == "input" else DEFAULT_OUTPUT_NAMES)
    for name in names:
        for d in pool:
            if name.lower() in d["name"].lower():
                return d
    if not wanted:
        import sounddevice as sd

        default = sd.default.device[0 if kind == "input" else 1]
        for d in devices:
            if d["index"] == default:
                return d
        if pool:
            return pool[0]
    listing = "\n".join(f"  [{d['index']}] {d['name']} ({d['hostapi']}, in {d['in']}, out {d['out']})" for d in devices)
    raise AudioError(f"no {kind} device matching {wanted!r}. Available:\n{listing}")


# ---------------------------------------------------------------------------
# Live backend
# ---------------------------------------------------------------------------
class LiveAudio:
    """Speaker + microphone through PortAudio (WASAPI exclusive on Windows)."""

    name = "live"

    def __init__(
        self,
        cfg: SonarConfig,
        in_device: str | int | None = None,
        out_device: str | int | None = None,
        exclusive: bool = True,
        blocksize: int = 1920,
        latency: float | str | None = "high",
        mode: str = "auto",
        gain: float = 1.0,
    ) -> None:
        self.cfg = cfg
        self._in_dev = find_device("input", in_device)
        self._out_dev = find_device("output", out_device)
        if self._out_dev["out"] < cfg.n_out:
            raise AudioError(
                f"output device {self._out_dev['name']!r} has {self._out_dev['out']} channel(s) but the profile "
                f"needs {cfg.n_out}; use --profile mono"
            )
        self.n_mics = min(cfg.n_mics, self._in_dev["in"])
        self.exclusive = exclusive and sys.platform == "win32"
        self.blocksize = blocksize
        self.latency = latency
        self.mode = mode
        self._frame = (make_tx_frame(cfg) * gain).astype(np.float32)
        self._tx_pos = 0                       # position inside the TX period
        self._fade_in_left = int(cfg.fade_ms * 1e-3 * cfg.fs)
        self._fade_total = self._fade_in_left
        self._fade_out_left: int | None = None
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=400)
        self._discontinuity = False
        self._streams: list[Any] = []
        self._last_rx = 0.0
        self.info: dict[str, Any] = {}
        self._program: np.ndarray | None = None        # one-shot signal played instead of the loop (self-test)
        self._prog_pos = 0
        self._program_done = threading.Event()
        self.idle_silent = False                       # True: play silence when no program is running

    # -- callbacks (PortAudio threads: keep them tiny) ---------------------------------
    def play_program(self, samples: np.ndarray) -> threading.Event:
        """Play `samples` [n, n_out] once instead of the looped sonar signal (used by the self-test).

        Returns an event that is set when the last sample has been handed to the driver.
        """
        self._program_done = threading.Event()
        self._prog_pos = 0
        self._program = np.ascontiguousarray(samples, dtype=np.float32)
        return self._program_done

    def _fill(self, outdata: np.ndarray, frames: int) -> None:
        prog = self._program
        if prog is not None or self.idle_silent:
            outdata[:] = 0.0
            if prog is not None:
                k = min(frames, len(prog) - self._prog_pos)
                outdata[:k, :self.cfg.n_out] = prog[self._prog_pos:self._prog_pos + k]
                self._prog_pos += k
                if self._prog_pos >= len(prog):
                    self._program = None
                    self._program_done.set()
            return
        n = self.cfg.frame_len
        idx = (self._tx_pos + np.arange(frames)) % n
        self._tx_pos = (self._tx_pos + frames) % n
        block = self._frame[idx]
        if self._fade_in_left > 0:                      # raised-cosine start: no click
            done = self._fade_total - self._fade_in_left
            k = min(frames, self._fade_in_left)
            ramp = fade_ramp(self._fade_total)[done:done + k]
            block[:k] *= ramp[:, None]
            self._fade_in_left -= k
        if self._fade_out_left is not None:             # raised-cosine stop
            k = min(frames, self._fade_out_left)
            ramp = fade_ramp(self._fade_total)[::-1][self._fade_total - self._fade_out_left:][:k]
            block[:k] *= ramp[:, None]
            block[k:] = 0.0
            self._fade_out_left -= k
        outdata[:, :self.cfg.n_out] = block
        outdata[:, self.cfg.n_out:] = 0.0

    def _push(self, indata: np.ndarray) -> None:
        self._last_rx = time.monotonic()
        try:
            self._q.put_nowait(indata[:, :self.n_mics].copy())
        except queue.Full:                              # consumer stalled: drop the oldest block
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(indata[:, :self.n_mics].copy())
            self._discontinuity = True

    def _out_cb(self, outdata: np.ndarray, frames: int, _time: Any, status: Any) -> None:
        if status:
            self._discontinuity = True
        self._fill(outdata, frames)

    def _in_cb(self, indata: np.ndarray, _frames: int, _time: Any, status: Any) -> None:
        if status:
            self._discontinuity = True
        self._push(indata)

    def _duplex_cb(self, indata: np.ndarray, outdata: np.ndarray, frames: int, _time: Any, status: Any) -> None:
        if status:
            self._discontinuity = True
        self._fill(outdata, frames)
        self._push(indata)

    # -- open / close ------------------------------------------------------------------
    def _open(self, mode: str, blocksize: int) -> None:
        import sounddevice as sd

        extra = sd.WasapiSettings(exclusive=True) if self.exclusive else None
        common: dict[str, Any] = dict(
            samplerate=self.cfg.fs, blocksize=blocksize, dtype="float32", latency=self.latency, dither_off=True
        )
        self._streams = []          # filled one by one, so a half-opened pair is always closed again
        if mode == "separate":
            # Two event-driven streams (the robust path for WASAPI exclusive). Input first, so no
            # transmitted sound is missed while the output is still opening.
            self._streams.append(sd.InputStream(
                device=self._in_dev["index"], channels=self.n_mics, extra_settings=extra, callback=self._in_cb, **common
            ))
            self._streams.append(sd.OutputStream(
                device=self._out_dev["index"], channels=self.cfg.n_out, extra_settings=extra, callback=self._out_cb, **common
            ))
        else:
            self._streams.append(sd.Stream(
                device=(self._in_dev["index"], self._out_dev["index"]),
                channels=(self.n_mics, self.cfg.n_out),
                extra_settings=(extra, extra) if extra is not None else None,
                callback=self._duplex_cb,
                **common,
            ))
        for s in self._streams:
            s.start()
        self.blocksize = blocksize
        self.info = {
            "mode": mode,
            "exclusive": self.exclusive,
            "input": self._in_dev["name"],
            "output": self._out_dev["name"],
            "samplerate": self.cfg.fs,
            "blocksize": blocksize,
            "latency": [getattr(s, "latency", None) for s in self._streams],
        }

    def start(self) -> None:
        """Open the devices. Tries separate streams then one duplex stream, each with the requested
        block size and then smaller ones (exclusive mode only accepts sizes the driver likes)."""
        modes = ["separate", "duplex"] if self.mode == "auto" else [self.mode]
        sizes = list(dict.fromkeys([self.blocksize, 960, 480, 0]))        # 0 = let PortAudio choose
        errors = []
        for mode in modes:
            for size in sizes:
                try:
                    self._open(mode, size)
                    self._last_rx = time.monotonic()
                    return
                except Exception as exc:                   # try the next way of opening the devices
                    errors.append(f"{mode}, block {size}: {exc}")
                    self._close()
        hint = ""
        if self.exclusive:
            hint = (
                "\nExclusive mode needs: (1) no other app using the speaker or microphone (close Teams, Discord, "
                "browser tabs playing sound), (2) Sound settings > device > Advanced > 'Allow applications to take "
                "exclusive control' ticked for BOTH devices, (3) 48000 Hz supported by both."
            )
        raise AudioError("could not open the audio devices:\n  " + "\n  ".join(errors) + hint)

    def _close(self) -> None:
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self._streams = []

    def stop(self) -> None:
        if self._streams:
            self._fade_out_left = self._fade_total          # fade out, then stop
            time.sleep(self.cfg.fade_ms * 1e-3 + 2.0 * self.blocksize / self.cfg.fs)
        self._close()

    # -- consumer side -------------------------------------------------------------------
    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Next microphone samples [n, n_mics] float32 (everything queued), or None on timeout."""
        try:
            blocks = [self._q.get(timeout=timeout)]
        except queue.Empty:
            if time.monotonic() - self._last_rx > 3.0:
                raise AudioError("the microphone stream stopped delivering audio (device unplugged or taken over?)")
            return None
        while True:
            try:
                blocks.append(self._q.get_nowait())
            except queue.Empty:
                break
        return np.concatenate(blocks, axis=0)

    def pop_discontinuity(self) -> bool:
        """True once after samples were dropped (the caller should realign)."""
        flag, self._discontinuity = self._discontinuity, False
        return flag


# ---------------------------------------------------------------------------
# Simulator backend
# ---------------------------------------------------------------------------
_DEMO_PERIOD = 62.0
_DEMO_GESTURES = ((10.0, "swipe_right"), (13.5, "push"), (17.0, "swipe_left"))


def _demo_body_y(tm: np.ndarray) -> np.ndarray:
    """Distance of the scripted person along the 62 s demo loop (walk in 4-7 s, leave 50-53 s)."""
    y = np.full_like(tm, 3.2)
    y = np.where((tm >= 4.0) & (tm < 7.0), 3.2 - 2.55 * min_jerk((tm - 4.0) / 3.0), y)
    y = np.where((tm >= 7.0) & (tm < 50.0), 0.65, y)
    y = np.where((tm >= 50.0) & (tm < 53.0), 0.65 + 2.55 * min_jerk((tm - 50.0) / 3.0), y)
    return y


def demo_reflectors(body_db: float = -24.0, hand_db: float = -20.0, bpm: float = 15.0) -> list[Reflector]:
    """A scripted visitor that repeats every 62 s: walks in, does three gestures, sits still
    and breathes at `bpm` for 30 s, leaves. Used by `--backend sim` so the UI can be rehearsed."""
    times = [0.0]
    points = [HAND_REST]
    for t0, label in _DEMO_GESTURES:
        for t, p in gesture_waypoints(label, t0):
            if t > times[-1]:
                times.append(t)
                points.append(p)
    hand = WaypointPath(times, points)

    def body_part(dx: float, dz: float, breathe: bool):
        def f(t: np.ndarray) -> np.ndarray:
            tm = np.mod(t, _DEMO_PERIOD)
            still = (tm >= 7.0) & (tm < 50.0)
            d = np.where(still, sway_displacement(t, 0.0003, 5), 0.0)
            if breathe:
                d = d + np.where(still, breathing_displacement(t, bpm, 0.0025), 0.0)
            bob = np.where(~still & (tm >= 4.0) & (tm < 53.0), 0.02 * np.sin(2 * np.pi * 1.8 * t), 0.0)
            return np.stack([np.full_like(t, dx), _demo_body_y(tm) - d + bob, np.full_like(t, dz)], axis=1)
        return f

    def hand_path(t: np.ndarray) -> np.ndarray:
        tm = np.mod(t, _DEMO_PERIOD)
        p = hand(tm)
        p[:, 1] += _demo_body_y(tm) - 0.65                 # the hand travels with its owner
        return p

    return [
        Reflector(body_part(0.0, 0.22, True), echo_db=body_db, ref_m=0.7),
        Reflector(body_part(0.16, 0.38, False), echo_db=body_db + 2.0, ref_m=0.7),
        Reflector(body_part(0.0, 0.52, False), echo_db=body_db - 2.0, ref_m=0.7),
        *hand_reflector(hand_path, echo_db=hand_db),
    ]


class SimAudio:
    """The simulator as an audio backend.

    scene="demo"  the repeating scripted visitor above.
    scene="idle"  one person sitting still; gestures happen only when `perform()` is called
                  (the recorder uses this so the whole record -> train chain can be tested).
    """

    name = "sim"

    def __init__(
        self,
        cfg: SonarConfig,
        scene: str = "demo",
        realtime: bool = True,
        block: int = 1920,
        seed: int = 0,
        hand_db: float = -20.0,
        body_db: float = -24.0,
        **sim_kwargs: Any,
    ) -> None:
        self.cfg = cfg
        self.n_mics = cfg.n_mics
        self.realtime = realtime
        self.block = block
        sim_kwargs.setdefault("cn0_db", 70.0)
        self._sim = EchoSimulator(cfg, seed=seed, **sim_kwargs)
        self._rng = np.random.default_rng(seed + 1)
        self._pos = 0
        self._t_start = 0.0
        self._lock = threading.Lock()
        self._hand_db = hand_db
        self._hand = WaypointPath([0.0], [HAND_REST])
        if scene == "demo":
            self._reflectors = demo_reflectors(body_db=body_db, hand_db=hand_db)
        elif scene == "idle":
            self._reflectors = person_reflectors(0.65, chest_db=body_db, body_db=body_db + 2.0, seed=seed) + hand_reflector(
                self._hand, echo_db=hand_db
            )
        elif scene == "empty":
            self._reflectors = []
        else:
            raise ValueError(f"unknown scene {scene!r}")
        self.info = {"mode": "sim", "scene": scene}

    @property
    def now_s(self) -> float:
        return self._pos / self.cfg.fs

    def perform(self, label: str, delay_s: float = 0.35, speed: float | None = None) -> None:
        """Make the simulated hand perform a gesture, starting `delay_s` from now (scene="idle")."""
        speed = float(self._rng.uniform(0.8, 1.3)) if speed is None else speed
        with self._lock:
            t0 = max(self.now_s + delay_s, self._hand.times[-1] + 0.05)
            for t, p in gesture_waypoints(label, t0, speed):
                if t > self._hand.times[-1]:
                    jitter = self._rng.normal(0.0, 0.012, size=3) if p != HAND_REST else np.zeros(3)
                    self._hand.append(t, tuple(np.asarray(p) + jitter))

    def start(self) -> None:
        self._t_start = time.monotonic()

    def stop(self) -> None:
        pass

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        if self.realtime:
            due = self._t_start + (self._pos + self.block) / self.cfg.fs
            wait = due - time.monotonic()
            if wait > 0:
                time.sleep(min(wait, timeout))
        with self._lock:
            rx = self._sim.render(self.block, self._reflectors, start_sample=self._pos)
        self._pos += self.block
        return rx

    def pop_discontinuity(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# File backend
# ---------------------------------------------------------------------------
class FileAudio:
    """Replays a recorded session (`audio.wav`) as if it were live."""

    name = "file"

    def __init__(self, cfg: SonarConfig, path: str | Path, realtime: bool = True, block: int = 1920, loop: bool = False) -> None:
        import soundfile as sf

        self.cfg = cfg
        self._sf = sf.SoundFile(str(path))
        if self._sf.samplerate != cfg.fs:
            raise AudioError(f"{path} is {self._sf.samplerate} Hz, expected {cfg.fs} Hz")
        self.n_mics = min(cfg.n_mics, self._sf.channels)
        self.realtime = realtime
        self.block = block
        self.loop = loop
        self._pos = 0
        self._t_start = 0.0
        self.finished = False
        self._jump = False
        self.info = {"mode": "file", "path": str(path), "frames": len(self._sf)}

    def start(self) -> None:
        self._t_start = time.monotonic()

    def stop(self) -> None:
        self._sf.close()

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        if self.finished:
            return None
        if self.realtime:
            wait = self._t_start + (self._pos + self.block) / self.cfg.fs - time.monotonic()
            if wait > 0:
                time.sleep(min(wait, timeout))
        data = self._sf.read(self.block, dtype="float32", always_2d=True)
        if len(data) < self.block:
            if self.loop:
                self._sf.seek(0)
                self._jump = True          # the file wraps: alignment is lost once
            else:
                self.finished = True
            if len(data) == 0:
                return None
        self._pos += len(data)
        return data[:, :self.n_mics]

    def pop_discontinuity(self) -> bool:
        flag, self._jump = self._jump, False
        return flag


def make_backend(kind: str, cfg: SonarConfig, **kwargs: Any) -> LiveAudio | SimAudio | FileAudio:
    """Factory used by the command-line tools."""
    if kind == "live":
        return LiveAudio(cfg, **kwargs)
    if kind == "sim":
        return SimAudio(cfg, **kwargs)
    if kind == "file":
        return FileAudio(cfg, **kwargs)
    raise ValueError(f"unknown backend {kind!r}")

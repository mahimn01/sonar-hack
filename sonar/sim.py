"""sim.py — a small physics simulator of what the laptop microphones hear.

Plain words: instead of a real laptop, this computes the microphone signal from
first principles. Every reflector (a hand, a chest, a wall) returns a delayed,
weaker copy of what the speakers play; when the reflector moves, the delay
changes, and that IS the Doppler shift and the breathing phase we measure.
It lets the tests check the receiver against known ground truth (a hand moving
at exactly 0.5 m/s, a chest breathing at exactly 15 breaths per minute) and
lets the engine and recorder run on a Mac with no audio hardware.

What is modelled: two speakers and two microphones at realistic positions, the
strong direct speaker->mic path, static clutter, moving point reflectors with
1/(d1*d2) spreading, a weaker speaker/mic response in the chirp band, sensor
noise, loud audible hall noise, an unknown playback/record latency and a
speaker-vs-mic clock drift.
What is not: room reverberation tails, reflector shapes, speaker directivity.
(The `datagen/` package has a richer randomised simulator for training data.)

Coordinates (metres), seen by the user facing the screen:
    x = to the user's RIGHT, y = from the screen TOWARDS the user, z = UP,
    origin = middle of the hinge line. Channel 0 = the user's LEFT speaker/mic.

Read-only module: no files, no audio devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import signal as sps

from .config import SPEED_OF_SOUND, SonarConfig
from .tx import make_tx_frame

PathFn = Callable[[np.ndarray], np.ndarray]   # times [n] (s) -> positions [n, 3] (m)
_OVERSAMPLE = 32


# ---------------------------------------------------------------------------
# Geometry and reflectors
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Geometry:
    speakers: np.ndarray     # [n_out, 3]
    mics: np.ndarray         # [n_mics, 3]

    @staticmethod
    def laptop(n_out: int = 2, n_mics: int = 2) -> "Geometry":
        """A 15-inch laptop: speakers above the keyboard, mics beside the webcam."""
        spk = np.array([[-0.12, 0.03, 0.01], [0.12, 0.03, 0.01]])
        mic = np.array([[-0.04, -0.07, 0.21], [0.04, -0.07, 0.21]])
        if n_out == 1:
            spk = spk.mean(axis=0, keepdims=True)
        if n_mics == 1:
            mic = mic.mean(axis=0, keepdims=True)
        return Geometry(spk, mic)


@dataclass
class Reflector:
    """A point that sends back an echo.

    `echo_db` is the echo level relative to the direct speaker->mic path when
    the reflector is `ref_m` away from both speaker and microphone; it scales
    with 1/(d_speaker * d_mic) elsewhere.
    """

    path: PathFn
    echo_db: float = -30.0
    ref_m: float = 0.3
    moving: bool = True


def min_jerk(u: np.ndarray) -> np.ndarray:
    """Smooth 0 -> 1 profile of a natural reaching movement."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


class WaypointPath:
    """Moves through timed waypoints with minimum-jerk segments; rests before/after.

    New waypoints can be appended while a stream is running (the simulator
    backend uses this to perform gestures on demand).
    """

    def __init__(self, times: list[float], points: list[tuple[float, float, float]]) -> None:
        self.times = list(map(float, times))
        self.points = [np.asarray(p, dtype=np.float64) for p in points]

    def append(self, time: float, point: tuple[float, float, float]) -> None:
        if time <= self.times[-1]:
            raise ValueError("waypoints must be appended in time order")
        self.times.append(float(time))
        self.points.append(np.asarray(point, dtype=np.float64))

    def __call__(self, t: np.ndarray) -> np.ndarray:
        times = np.asarray(self.times)
        pts = np.stack(self.points)
        if len(times) == 1:
            return np.repeat(pts, len(t), axis=0)
        i = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
        u = min_jerk((t - times[i]) / (times[i + 1] - times[i]))
        return pts[i] + (pts[i + 1] - pts[i]) * u[:, None]


def static_path(point: tuple[float, float, float]) -> PathFn:
    p = np.asarray(point, dtype=np.float64)
    return lambda t: np.repeat(p[None, :], len(t), axis=0)


def linear_path(start: tuple[float, float, float], velocity: tuple[float, float, float]) -> PathFn:
    """Constant-velocity motion (for the Doppler ground-truth test)."""
    p0 = np.asarray(start, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    return lambda t: p0[None, :] + t[:, None] * v[None, :]


def breathing_displacement(t: np.ndarray, bpm: float, amp_m: float, phase: float = 0.0) -> np.ndarray:
    """Chest-wall displacement (m, positive = towards the laptop).

    Not a pure sine: inhaling is quicker than exhaling, so there is a second
    harmonic, as in real breathing. `amp_m` is half the peak-to-peak movement.
    """
    th = 2 * np.pi * bpm / 60.0 * t + phase
    return amp_m * (np.sin(th) + 0.25 * np.sin(2 * th + 0.6)) / 1.13


def sway_displacement(t: np.ndarray, rms_m: float, seed: int = 0) -> np.ndarray:
    """Slow random body sway (m): broadband noise that falls off above ~0.1 Hz, like real postural sway.

    Built from 160 closely spaced sinusoids with random phases, so it is a smooth
    function of absolute time (blocks rendered separately join seamlessly) and has
    a continuous-looking spectrum rather than a few sharp lines.
    """
    rng = np.random.default_rng(seed + 7919)
    f = np.linspace(0.01, 1.0, 160) + rng.uniform(-0.003, 0.003, size=160)
    a = 1.0 / (1.0 + (f / 0.10) ** 2)                  # amplitude ~ flat below 0.1 Hz, ~1/f^2 above
    ph = rng.uniform(0, 2 * np.pi, size=160)
    s = np.zeros_like(t)
    for fi, ai, pi_ in zip(f, a, ph):
        s += ai * np.sin(2 * np.pi * fi * t + pi_)
    return rms_m * s / np.sqrt(np.sum(a * a) / 2)


def person_reflectors(
    distance_m: float = 0.7,
    bpm: float = 15.0,
    breath_amp_m: float = 0.0025,
    sway_rms_m: float = 0.0005,
    chest_db: float = -40.0,
    body_db: float = -38.0,
    x_m: float = 0.0,
    seed: int = 0,
) -> list[Reflector]:
    """A still person facing the laptop: a breathing chest plus non-breathing body parts.

    All parts share the slow sway; only the chest adds the breathing movement.
    The static parts matter: they are the "static vector" the breathing
    estimator has to remove before it can read the chest phase.
    """
    def chest(t: np.ndarray) -> np.ndarray:
        d = breathing_displacement(t, bpm, breath_amp_m) + sway_displacement(t, sway_rms_m, seed)
        return np.stack([np.full_like(t, x_m), distance_m - d, np.full_like(t, 0.22)], axis=1)

    def shoulders(t: np.ndarray) -> np.ndarray:
        d = sway_displacement(t, sway_rms_m, seed)
        return np.stack([np.full_like(t, x_m + 0.16), distance_m + 0.03 - d, np.full_like(t, 0.38)], axis=1)

    def head(t: np.ndarray) -> np.ndarray:
        d = sway_displacement(t, sway_rms_m, seed)
        return np.stack([np.full_like(t, x_m), distance_m - 0.02 - d, np.full_like(t, 0.52)], axis=1)

    return [
        Reflector(chest, echo_db=chest_db, ref_m=0.7),
        Reflector(shoulders, echo_db=body_db, ref_m=0.7),
        Reflector(head, echo_db=body_db - 4.0, ref_m=0.7),
    ]


# Hand positions used by the scripted gestures (metres, laptop frame).
HAND_REST = (0.10, 0.42, 0.04)
_GESTURE_Y, _GESTURE_Z = 0.26, 0.17


def gesture_waypoints(label: str, t0: float, speed: float = 1.0) -> list[tuple[float, tuple[float, float, float]]]:
    """Timed hand waypoints for one gesture starting at t0 (after the hand has reached its start pose).

    The hand rises from rest to the start pose, performs the stroke, and drops
    back to rest. `speed` > 1 makes everything faster.
    """
    s = 1.0 / speed
    y, z = _GESTURE_Y, _GESTURE_Z
    if label in ("swipe_right", "swipe_left"):
        sign = 1.0 if label == "swipe_right" else -1.0
        a, b = (-0.20 * sign, y, z), (0.20 * sign, y, z)
        # slow wind-up to the start side, quick stroke across, relaxed drop back to rest
        return [(t0, HAND_REST), (t0 + 0.80 * s, a), (t0 + 1.20 * s, b), (t0 + 2.00 * s, HAND_REST)]
    if label == "push":
        a, b = (0.0, 0.36, z), (0.0, 0.14, z)
        return [(t0, HAND_REST), (t0 + 0.80 * s, a), (t0 + 1.08 * s, b), (t0 + 1.40 * s, a), (t0 + 2.20 * s, HAND_REST)]
    if label == "pull":
        # Reach out slowly until the hand is near the screen, then pull it back quickly.
        a, b = (0.0, 0.14, z), (0.0, 0.40, z)
        return [(t0, HAND_REST), (t0 + 1.10 * s, a), (t0 + 1.40 * s, b), (t0 + 2.20 * s, HAND_REST)]
    raise ValueError(f"no scripted trajectory for {label!r}")


def gesture_stroke_window(label: str, t0: float, speed: float = 1.0) -> tuple[float, float]:
    """(start, end) time of the fast, meaningful part of a scripted gesture."""
    s = 1.0 / speed
    spans = {"swipe_right": (0.80, 1.20), "swipe_left": (0.80, 1.20), "push": (0.80, 1.40), "pull": (1.10, 1.40)}
    a, b = spans[label]
    return t0 + a * s, t0 + b * s


def hand_reflector(path: PathFn, echo_db: float = -30.0) -> list[Reflector]:
    """A hand plus a weaker forearm point trailing 12 cm behind it (towards the user)."""
    def forearm(t: np.ndarray) -> np.ndarray:
        return path(t) + np.array([0.02, 0.12, -0.03])[None, :]
    return [Reflector(path, echo_db=echo_db, ref_m=0.3), Reflector(forearm, echo_db=echo_db - 8.0, ref_m=0.3)]


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------
@dataclass
class EchoSimulator:
    cfg: SonarConfig
    geometry: Geometry | None = None
    direct_gain_db: float = -73.0        # RX amplitude / TX amplitude of the direct path (measured: -79 dBFS for a 0.5 tone)
    chirp_response_db: float = -10.0     # speaker+mic are ~10 dB weaker at 19-20 kHz than at 17-18 kHz (measured)
    cn0_db: float = 60.0                 # direct-path power of a 0.5-amplitude tone over noise in 1 Hz (measured: ~60 dB)
    ambient_dbfs: float | None = -40.0   # rms of audible hall noise (speech/music band); None = quiet room
    latency_samples: float = 1234.56     # unknown output+input latency of the audio stack
    clock_ppm: float = 0.0               # speaker clock runs this much faster than the microphone clock
    static_clutter: list[Reflector] = field(default_factory=list)
    seed: int = 0

    def __post_init__(self) -> None:
        cfg = self.cfg
        if self.geometry is None:
            self.geometry = Geometry.laptop(cfg.n_out, cfg.n_mics)
        self._rng = np.random.default_rng(self.seed)
        self._gd = 10.0 ** (self.direct_gain_db / 20.0)
        n = cfg.frame_len
        self._L = n * _OVERSAMPLE
        freqs = np.fft.rfftfreq(n, 1.0 / cfg.fs)
        self._freqs = freqs
        resp = self._response(freqs)
        spec = np.fft.rfft(make_tx_frame(cfg).astype(np.float64), axis=0) * resp[:, None]     # [bins, n_out]
        self._tx_spec = spec
        self._tables = [self._table(spec[:, ch]) for ch in range(cfg.n_out)]
        self._static_tables: list[np.ndarray] | None = None
        self._static_key: tuple[int, ...] | None = None
        # Sensor noise level from the measured carrier-to-noise density.
        carrier_power = (0.5 * self._gd) ** 2 / 2.0
        n0 = carrier_power / 10.0 ** (self.cn0_db / 10.0)
        self._noise_sigma = float(np.sqrt(n0 * cfg.fs / 2.0))
        self._ambient_zi: np.ndarray | None = None
        # Hall noise lives in the audible band; a steep filter keeps it from (unrealistically)
        # raising the ultrasonic noise floor. In-band noise is set by `cn0_db` alone.
        self._ambient_sos = sps.ellip(8, 0.5, 110.0, [150.0, 6000.0], "bandpass", fs=cfg.fs, output="sos")

    # -- transducer response ----------------------------------------------------
    def _response(self, f: np.ndarray) -> np.ndarray:
        """Amplitude response: flat for the tones, `chirp_response_db` lower above ~18.8 kHz."""
        step = 1.0 / (1.0 + np.exp(-(f - 18800.0) / 60.0))
        return 10.0 ** (self.chirp_response_db * step / 20.0)

    def _table(self, spec: np.ndarray) -> np.ndarray:
        """One period, 32x oversampled (exact, the signal is band-limited), padded for cubic lookup."""
        n = self.cfg.frame_len
        big = np.zeros(self._L // 2 + 1, dtype=np.complex128)
        big[:len(spec)] = spec
        table = np.fft.irfft(big, n=self._L) * _OVERSAMPLE
        return np.concatenate([table[-1:], table, table[:3]])

    def _lookup(self, table: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Catmull-Rom interpolation of a periodic table at fractional positions (in table samples)."""
        pos = np.mod(pos, self._L)
        i = np.floor(pos).astype(np.int64)
        f = pos - i
        p0, p1, p2, p3 = table[i], table[i + 1], table[i + 2], table[i + 3]
        return p1 + 0.5 * f * (p2 - p0 + f * (2 * p0 - 5 * p1 + 4 * p2 - p3 + f * (3 * (p1 - p2) + p3 - p0)))

    # -- static part (direct path + things that never move) -----------------------
    def _build_static(self, statics: list[Reflector]) -> list[np.ndarray]:
        geo = self.geometry
        assert geo is not None
        f = self._freqs
        tables = []
        t0 = np.zeros(1)
        for mic in geo.mics:
            spec = np.zeros(len(f), dtype=np.complex128)
            for ch, spk in enumerate(geo.speakers):
                d_direct = np.linalg.norm(spk - mic)
                chan = self._gd * np.exp(-2j * np.pi * f * d_direct / SPEED_OF_SOUND)
                for r in statics:
                    p = r.path(t0)[0]
                    d1, d2 = np.linalg.norm(p - spk), np.linalg.norm(p - mic)
                    g = self._gd * 10.0 ** (r.echo_db / 20.0) * r.ref_m**2 / (d1 * d2)
                    chan = chan + g * np.exp(-2j * np.pi * f * (d1 + d2) / SPEED_OF_SOUND)
                spec += self._tx_spec[:, ch] * chan
            tables.append(self._table(spec))
        return tables

    # -- rendering ------------------------------------------------------------------
    def render(self, n_samples: int, reflectors: list[Reflector] | None = None, start_sample: int = 0) -> np.ndarray:
        """Microphone audio [n_samples, n_mics] float32 for absolute samples start .. start+n."""
        cfg, geo = self.cfg, self.geometry
        assert geo is not None
        reflectors = list(reflectors or [])
        statics = self.static_clutter + [r for r in reflectors if not r.moving]
        movers = [r for r in reflectors if r.moving]
        key = tuple(id(r) for r in statics)
        if self._static_tables is None or key != self._static_key:
            self._static_tables, self._static_key = self._build_static(statics), key

        out = np.zeros((n_samples, cfg.n_mics), dtype=np.float64)
        chunk = 2 * cfg.fs
        for c0 in range(0, n_samples, chunk):
            n_abs = start_sample + c0 + np.arange(min(chunk, n_samples - c0), dtype=np.float64)
            t = n_abs / cfg.fs                                           # wall-clock time (s)
            # Position in the TX period (in audio samples) that a zero-delay path would deliver now.
            u = np.mod(n_abs * (1.0 + self.clock_ppm * 1e-6) - self.latency_samples, cfg.frame_len)
            sl = slice(c0, c0 + len(n_abs))
            for m in range(cfg.n_mics):
                out[sl, m] += self._lookup(self._static_tables[m], u * _OVERSAMPLE)
            if movers:
                # Trajectories are smooth: evaluate at 500 Hz and interpolate the distances.
                tc = np.arange(t[0], t[-1] + 0.004, 0.002)
                for r in movers:
                    pc = r.path(tc)
                    amp0 = self._gd * 10.0 ** (r.echo_db / 20.0) * r.ref_m**2
                    d_mic = [np.interp(t, tc, np.linalg.norm(pc - mic, axis=1)) for mic in geo.mics]
                    for ch, spk in enumerate(geo.speakers):
                        d_spk = np.interp(t, tc, np.linalg.norm(pc - spk, axis=1))
                        for m in range(cfg.n_mics):
                            delay = (d_spk + d_mic[m]) * (cfg.fs / SPEED_OF_SOUND)       # samples
                            out[sl, m] += amp0 / (d_spk * d_mic[m]) * self._lookup(
                                self._tables[ch], (u - delay) * _OVERSAMPLE
                            )
            out[sl] += self._noise_sigma * self._rng.standard_normal((len(n_abs), cfg.n_mics))
            if self.ambient_dbfs is not None:
                white = self._rng.standard_normal((len(n_abs), cfg.n_mics))
                if self._ambient_zi is None:
                    self._ambient_zi = np.zeros((self._ambient_sos.shape[0], 2, cfg.n_mics))
                band, self._ambient_zi = sps.sosfilt(self._ambient_sos, white, axis=0, zi=self._ambient_zi)
                # white noise through this band-pass keeps ~ (5850 Hz / 24000 Hz) of its power
                out[sl] += band * (10.0 ** (self.ambient_dbfs / 20.0) / np.sqrt(5850.0 / 24000.0))
        return out.astype(np.float32)


def simulate(
    cfg: SonarConfig,
    duration_s: float,
    reflectors: list[Reflector] | None = None,
    **sim_kwargs: float | int | None,
) -> np.ndarray:
    """Convenience wrapper: render `duration_s` seconds of microphone audio."""
    sim = EchoSimulator(cfg, **sim_kwargs)  # type: ignore[arg-type]
    return sim.render(int(round(duration_s * cfg.fs)), reflectors)

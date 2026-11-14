"""background.py — lay synthetic gestures over REAL background recordings.

Plain words
-----------
The hardest things to simulate are the boring ones: this laptop's exact speaker
and microphone colouration, its direct speaker-to-mic leakage, the room's
reflections and the hall's noise. But those are trivial to RECORD: play the
probe signal for a few minutes with nobody gesturing and save what the mics
hear. Sound adds up linearly, so we can then add the computed echo of a virtual
hand on top of that real recording. The result has the real laptop, the real
room and the real noise; only the hand is synthetic.

What this file does
    measure(real, core)        reads off the real recording: how strong the direct
                               leakage is at each tone and mic, where the probe
                               signal's period starts, the clock mismatch, and the
                               rough colouration of the chirp band
    inject(real, core, scene)  renders ONLY the moving scene, scaled to the measured
                               leakage level, and adds it to the real audio

Requirements for the real recording: the SAME probe signal must be playing, and
nobody performs gestures (people may sit, type or walk past: those become real
negatives). 2-5 minutes at the demo spot is plenty.

Read-only helper module: returns arrays, writes nothing. `generate.py
--background file.wav` uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import fft as sfft

from .adapter import Core, pre_roll
from .hardware import Hardware, Profile, Response, sample_hardware
from .render import render


@dataclass
class Measured:
    """What a real background recording tells us about the real laptop."""

    direct_gain_db: float  # leakage: received tone amplitude / played tone amplitude (average)
    tone_mic_db: np.ndarray  # [n_tones, n_mics] the same, per tone and microphone
    period_offset: int  # where the strongest (direct) arrival sits inside one signal period
    clock_ppm: float  # microphone clock vs speaker clock
    band_freqs: np.ndarray  # frequencies (Hz) of the probe lines used for the colouration
    band_gain_db: np.ndarray  # smoothed received/played level at those lines, relative to tone 0


class MeasuredResponse(Response):
    """A colouration curve read off a real recording (level only, interpolated)."""

    def __init__(self, freqs: np.ndarray, gain_db: np.ndarray, extra_db: float = 0.0) -> None:
        super().__init__()
        self._f, self._g, self.gain_db = np.asarray(freqs, float), np.asarray(gain_db, float), extra_db

    def __call__(self, f: np.ndarray) -> np.ndarray:
        g = np.interp(np.asarray(f, float), self._f, self._g) + self.gain_db
        return (10.0 ** (g / 20.0)).astype(np.complex128)


def load_audio(path: str | Path) -> np.ndarray:
    """Read a recording as float [n, n_mics]. Supports .wav / .flac / .npy (a float32 .npy is memory-mapped)."""
    path = Path(path)
    if path.suffix == ".npy":
        x = np.load(path, mmap_mode="r")  # memory-mapped: all generator workers share one copy in the page cache
    else:
        import soundfile as sf

        x, _ = sf.read(str(path), dtype="float32", always_2d=True)
    x = np.asarray(x, dtype=np.float32)
    return x[:, None] if x.ndim == 1 else x


def measure(real: np.ndarray, core: Core, seconds: float = 2.0) -> Measured:
    """Read the laptop's direct-leakage levels, period alignment and clock mismatch off a recording."""
    fs, period = core.fs, core.period
    x = np.asarray(real[: int(seconds * fs) // period * period], dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = np.arange(len(x), dtype=np.float64)
    frame = np.asarray(core._frame if core._frame is not None else core.tx(0, period, 0), dtype=np.float64)
    if frame.ndim == 1:
        frame = frame[:, None]
    tx_spec = sfft.rfft(frame.sum(axis=1))
    freqs = sfft.rfftfreq(period, 1.0 / fs)

    # level of each steady tone at each mic, in 0.1 s pieces so a small clock mismatch cannot cancel it
    block = max(period, int(0.1 * fs) // period * period)
    tone_mic = np.zeros((len(core.tones), x.shape[1]))
    drift = []
    for k, f_t in enumerate(core.tones):
        ph = np.exp(-2j * np.pi * np.mod(n * (f_t / fs), 1.0))
        z = (x * ph[:, None])[: len(x) // block * block].reshape(-1, block, x.shape[1]).mean(axis=1) * 2.0
        played = 2.0 * np.abs(tx_spec[int(round(f_t / (fs / period)))]) / period
        tone_mic[k] = 20 * np.log10(np.abs(z).mean(axis=0) / max(played, 1e-12) + 1e-30)
        # clock mismatch = the tone's phase creeping steadily from block to block
        step = np.angle((z[1:] * np.conj(z[:-1])).sum())
        drift.append(step / (2 * np.pi * f_t * block / fs) * 1e6)

    # where inside the period does the direct path arrive? (uses the wideband part of the probe)
    k_frames = min(16, len(x) // period)
    mean_frame = x[: k_frames * period].reshape(k_frames, period, -1).mean(axis=0)
    rx_spec = sfft.rfft(mean_frame, axis=0)
    tone_bins = {int(round(f_t / (fs / period))) for f_t in core.tones}
    lines = np.array([b for b in np.nonzero(np.abs(tx_spec) > 1e-3 * np.abs(tx_spec).max())[0] if b not in tone_bins])
    offset, band_f, band_g = 0, np.array([core.band[0], core.band[1]]), np.zeros(2)
    if len(lines) >= 8:
        spec = np.zeros(len(tx_spec), dtype=np.complex128)
        spec[lines] = (rx_spec[lines] * np.conj(tx_spec[lines])[:, None]).sum(axis=1)
        offset = int(np.argmax(np.abs(sfft.irfft(spec, period))))
        ratio = 20 * np.log10(np.abs(rx_spec[lines]).mean(axis=1) / np.abs(tx_spec[lines]) + 1e-30)
        k = max(1, min(11, len(lines) // 3))
        smooth = np.convolve(np.pad(ratio, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[: len(ratio)]
        band_f, band_g = freqs[lines], smooth - float(tone_mic[0].mean())
    return Measured(float(tone_mic.mean()), tone_mic, offset, float(np.median(drift)), band_f, band_g)


def matched_hardware(rng: np.random.Generator, core: Core, m: Measured, profile: Profile | None = None) -> Hardware:
    """A random virtual laptop whose levels, alignment and colouration follow the real measurements.

    Geometry (mic spacing, speaker direction, echo strength) stays random: a
    background recording cannot tell us those. `profile` carries fixed facts such as
    swapped microphone / speaker channels.
    """
    hw = sample_hardware(rng, profile or Profile(band_hz=core.band), core.fs)
    hw.tx_route = "stereo" if core.n_speakers >= 2 else "both"
    hw.direct_gain_db = m.direct_gain_db
    hw.adc_bits = 0  # the real recording is already quantised; the added echo must stay exact
    hw.clock_ppm = m.clock_ppm
    hw.gain_drift_db, hw.gain_step_db = 0.0, 0.0
    # per-speaker level: each tone tells us how loud "its" speaker arrives
    for s in range(hw.n_spk):
        played = (hw.n_spk - 1 - s) if hw.swap_speakers else s  # render() swaps the channels after this
        mine = [k for k, sp in enumerate(core.tone_speakers) if sp == played]
        level = float(m.tone_mic_db[mine].mean() - m.direct_gain_db) if mine else 0.0
        jitter = rng.normal(0.0, 2.0)  # the echo path is not coloured exactly like the leakage path
        f_all = np.concatenate([[min(core.tones) - 2000.0], [core.tones[k] for k in mine], m.band_freqs])
        g_all = np.concatenate([[level], [level] * len(mine), m.band_gain_db + rng.normal(0.0, 1.5)])
        order = np.argsort(f_all)
        hw.spk_resp[s] = MeasuredResponse(f_all[order], g_all[order], jitter)
    for mic in range(hw.n_mic):
        r = Response()
        real_mic = (hw.n_mic - 1 - mic) if hw.swap_mics else mic  # render() swaps the channels after this
        real_mic = min(real_mic, m.tone_mic_db.shape[1] - 1)  # a one-microphone laptop (MacBook): every virtual mic follows it
        r.gain_db = float(m.tone_mic_db[:, real_mic].mean() - m.direct_gain_db)
        hw.mic_resp[mic] = r
    # line the virtual direct path up with the real one, so echoes sit at the right delay behind it
    d0 = int(round(hw.static_delays[0][0][0] * core.fs))
    hw.latency_samples = int((m.period_offset - d0) % core.period)
    return hw


def inject(
    real: np.ndarray,
    core: Core,
    scene,
    rng: np.random.Generator,
    measured: Measured | None = None,
    hw: Hardware | None = None,
    profile: Profile | None = None,
) -> tuple:
    """Add the echo of a synthetic moving scene to a real recording.

    real : [n, n_mics] excerpt that STARTS on a multiple of the signal period
           (use `random_excerpt`). Returns (mixed audio [n, n_mics] float32, info dict).
    """
    real = np.asarray(real, dtype=np.float32)
    n = len(real)
    measured = measured or measure(real, core)
    hw = hw or matched_hardware(rng, core, measured, profile)
    pre = pre_roll(core, 2500 + core.period)
    out = render(core.tx(pre, n), hw, scene, n, rng, pre=pre, band=core.band, add_noise=False, include_static=False)
    mixed = np.clip(real + out.rx[:, : real.shape[1]], -1.0, 1.0).astype(np.float32)
    # how audible is the added motion against the REAL noise? (same 25 Hz yardstick as render())
    spec = sfft.rfft(real.astype(np.float64), axis=0)
    f = sfft.rfftfreq(n, 1.0 / core.fs)
    quiet = np.ones(len(f), dtype=bool)
    for line in np.arange(0.0, core.fs / 2, core.fs / core.period):
        quiet &= np.abs(f - line) > 6.0  # skip the probe lines themselves
    quiet &= (f > core.band[0] - 400.0) & (f < core.band[1] + 400.0)
    n0 = 2.0 * float(np.median(np.abs(spec[quiet]) ** 2)) / (n * core.fs) / 0.69  # median -> mean of noise power
    events = getattr(scene, "events", None) or [{"t_start": 0.0, "t_end": n / core.fs}]
    sel = np.zeros(n, dtype=bool)
    for ev in events:
        sel[max(0, int(ev["t_start"] * core.fs)) : max(0, int(ev["t_end"] * core.fs))] = True
    p_echo = float(np.mean(out.rx[sel].astype(np.float64) ** 2)) if sel.any() else 0.0
    snr = 10 * np.log10(p_echo / (n0 * 25.0) + 1e-30) if p_echo > 0 else float("nan")
    return mixed, {"hardware": hw, "measured": measured, "echo_snr_db": float(snr)}


def random_excerpt(rng: np.random.Generator, real: np.ndarray, n: int, period: int) -> np.ndarray:
    """A random piece of a long recording that starts on a whole signal period."""
    if len(real) < n:
        raise ValueError(f"background recording is shorter than one clip ({len(real)} < {n} samples)")
    start = int(rng.integers(0, (len(real) - n) // period + 1)) * period
    return real[start : start + n]

"""physics.py — acoustic constants and small geometry helpers.

Plain words: sound travels ~343 m/s, so a hand 30 cm away returns an echo about
1.7 ms after the speaker emits it, and a hand moving at 1 m/s shifts a 20 kHz
tone by roughly 117 Hz (the Doppler effect). Everything the simulator does
follows from path lengths divided by the speed of sound.

Read-only helper module: no files, no audio devices.
"""

from __future__ import annotations

import numpy as np


def speed_of_sound(temp_c: float = 20.0) -> float:
    """Speed of sound in air (m/s) at a given temperature in Celsius."""
    return float(331.3 * np.sqrt(1.0 + temp_c / 273.15))


def air_absorption_db_per_m(
    f_hz: float, temp_c: float = 20.0, rh_pct: float = 50.0, p_kpa: float = 101.325
) -> float:
    """Atmospheric absorption of sound (dB per metre), ISO 9613-1 closed form.

    Near 20 kHz this is about 0.5 dB/m, i.e. tiny for a hand at 30 cm and a few
    dB for a person walking 4 m away. It is also why ultrasonic reverberation
    dies quickly.
    """
    t_k = temp_c + 273.15
    t0, t01 = 293.15, 273.16
    pa = p_kpa / 101.325
    # molar concentration of water vapour (%)
    c_sat = -6.8346 * (t01 / t_k) ** 1.261 + 4.6151
    h = rh_pct * (10.0**c_sat) / pa
    # relaxation frequencies of oxygen and nitrogen
    fr_o = pa * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = pa * (t_k / t0) ** -0.5 * (9.0 + 280.0 * h * np.exp(-4.170 * ((t_k / t0) ** (-1.0 / 3.0) - 1.0)))
    f2 = float(f_hz) ** 2
    alpha = 8.686 * f2 * (
        1.84e-11 / pa * (t_k / t0) ** 0.5
        + (t_k / t0) ** -2.5
        * (
            0.01275 * np.exp(-2239.1 / t_k) / (fr_o + f2 / fr_o)
            + 0.1068 * np.exp(-3352.0 / t_k) / (fr_n + f2 / fr_n)
        )
    )
    return float(alpha)


def doppler_hz(radial_speed_mps: float, f_hz: float, c: float = 343.0) -> float:
    """Two-way Doppler shift for a reflector approaching at `radial_speed_mps`.

    Positive speed = approaching = echo frequency goes UP.
    """
    return 2.0 * radial_speed_mps * f_hz / c


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalise vectors to length 1 (safe for zero-length input)."""
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-9)


def minjerk(u: np.ndarray) -> np.ndarray:
    """Minimum-jerk position profile on u in [0,1]: smooth start, smooth stop.

    Human reaching movements follow this bell-shaped speed curve closely; its
    peak speed is 1.875x the average speed.
    """
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


def smooth_noise(rng: np.random.Generator, n: int, fs: float, cutoff_hz: float) -> np.ndarray:
    """Zero-mean, unit-variance low-pass noise (used for tremor, sway, gain drift).

    Built at a low internal rate (about 16x the cutoff) and interpolated up, so
    a 3-second audio-rate gain track costs microseconds, not a huge FFT.
    """
    if n <= 1:
        return np.zeros(n)
    cutoff_hz = max(float(cutoff_hz), 1e-3)
    rate = min(float(fs), 16.0 * cutoff_hz)
    m = int(np.ceil(n * rate / fs)) + 4
    pad = int(np.ceil(2.0 * rate / cutoff_hz)) + 4
    x = rng.standard_normal(m + 2 * pad)
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / rate)
    spec *= np.exp(-0.5 * (f / cutoff_hz) ** 2)
    y = np.fft.irfft(spec, len(x))[pad : pad + m]
    if rate < fs:
        from scipy.interpolate import CubicSpline

        y = CubicSpline(np.arange(m) / rate, y)(np.arange(n) / fs)
    else:
        y = y[:n]
    y = y - y.mean()
    return y / (y.std() + 1e-12)

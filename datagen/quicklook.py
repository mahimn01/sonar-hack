"""quicklook.py — a minimal Doppler spectrogram, independent of the core package.

Plain words: the speaker plays steady tones near 19 kHz. A moving hand shifts
the echo's pitch up (approaching) or down (moving away) by up to a few hundred
Hz. This module zooms in on each tone and shows how that pitch shift changes
over time: that picture is the "Doppler spectrogram" a human can eyeball.

It is deliberately simple and is NOT the feature the model trains on: the
adapter uses the core package's own feature function when it exists. This one
is for sanity plots, tests and the training-free heuristics.

Read-only helper module.
"""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy.signal import resample_poly

BB_RATE = 2000  # sample rate of the zoomed-in signal around each tone (Hz)


def to_baseband(rx: np.ndarray, fs: int, f_c: float, out_rate: int = BB_RATE) -> np.ndarray:
    """Shift the tone at `f_c` down to 0 Hz and keep +-out_rate/2 around it.

    rx: [n] or [n, n_mic]. Returns complex [n_dec, n_mic].
    """
    x = np.asarray(rx, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = np.arange(x.shape[0], dtype=np.float64)
    mixed = x * np.exp(-2j * np.pi * np.mod(n * (f_c / fs), 1.0))[:, None]
    return resample_poly(mixed, 1, fs // out_rate, axis=0)


def remove_static(bb: np.ndarray, rate: int = BB_RATE, cutoff_hz: float = 6.0, order: int = 2) -> np.ndarray:
    """Remove the unchanging part (direct speaker->mic leakage, walls, desk).

    Static paths sit at exactly 0 Hz after `to_baseband`; this zero-phase
    high-pass keeps only things that MOVE. Raise `cutoff_hz` (e.g. 15) to also
    drop body sway and breathing, which smear a few Hz around the tone.
    """
    n = bb.shape[0]
    f = np.abs(sfft.fftfreq(n, 1.0 / rate)) + 1e-9
    keep = 1.0 / (1.0 + (cutoff_hz / f) ** (2 * order))
    return sfft.ifft(sfft.fft(bb, axis=0) * keep[:, None], axis=0)


def stft(bb: np.ndarray, rate: int = BB_RATE, win_s: float = 0.064, hop_s: float = 0.008) -> tuple:
    """Short-time Fourier transform of a complex baseband signal.

    Returns (S [n_freq, n_time, n_mic] complex, freqs Hz (negative..positive), times s).
    """
    win = int(round(win_s * rate))
    hop = max(1, int(round(hop_s * rate)))
    w = np.hanning(win)
    n_frames = 1 + (bb.shape[0] - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = bb[idx] * w[None, :, None]  # [n_time, win, n_mic]
    spec = sfft.fftshift(sfft.fft(frames, axis=1), axes=1)
    freqs = sfft.fftshift(sfft.fftfreq(win, 1.0 / rate))
    times = (np.arange(n_frames) * hop + win / 2) / rate
    return np.transpose(spec, (1, 0, 2)), freqs, times


def doppler_spectrogram(
    rx: np.ndarray,
    fs: int,
    tones: list | float,
    span_hz: float = 400.0,
    win_s: float = 0.064,
    hop_s: float = 0.008,
) -> tuple:
    """Doppler spectrogram averaged over the probe tones.

    Returns (power [n_freq, n_time, n_mic] linear, freqs Hz, times s, complex STFT of the first tone).
    Positive frequency = reflector approaching.
    """
    tones = [float(tones)] if np.isscalar(tones) else [float(f) for f in tones]
    power, first = None, None
    for f_c in tones:
        bb = remove_static(to_baseband(rx, fs, f_c))
        spec, freqs, times = stft(bb, BB_RATE, win_s, hop_s)
        keep = np.abs(freqs) <= span_hz
        spec, freqs = spec[keep], freqs[keep]
        p = np.abs(spec) ** 2
        power = p if power is None else power + p
        if first is None:
            first = spec
    return power / len(tones), freqs, times, first


def to_db(power: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    """Log scale relative to the loudest cell, clipped at `floor_db`."""
    ref = float(power.max()) + 1e-30
    return np.maximum(10.0 * np.log10(power / ref + 1e-30), floor_db)


def default_tones() -> list:
    """Fallback probe tones used only when the core package is not importable."""
    return [17500.0, 18500.0, 19500.0, 20500.0]


def default_tx(n: int, fs: int = 48000, tones: list | None = None, amp: float = 0.5) -> np.ndarray:
    """Fallback probe signal: a few steady tones, total peak about `amp`."""
    tones = tones or default_tones()
    t = np.arange(n) / fs
    x = sum(np.sin(2 * np.pi * f * t + 1.7 * k) for k, f in enumerate(tones))
    return (amp * x / len(tones)).astype(np.float64)

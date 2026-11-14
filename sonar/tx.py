"""tx.py — the transmitted waveform: two pilot tones plus one periodic chirp.

Plain words: this builds ONE 20 ms period of the speaker signal. The audio
backend loops that period forever. Because every component is an exact
multiple of 50 Hz (= one cycle per period), the loop has no seam: no clicks,
and exactly zero energy below the ultrasonic band.

The chirp is built in the frequency domain: a comb of equal-strength lines
from 19.0 to 21.5 kHz with quadratic ("Schroeder") phases. In the time domain
that is a smooth upward sweep with an almost constant envelope, so it uses the
speaker efficiently and has a perfectly flat spectrum for echo measurement.

Read-only module: no files, no audio devices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SonarConfig


@dataclass(frozen=True)
class TxReference:
    """Everything the receiver needs to know about what was transmitted."""

    frame: np.ndarray          # [frame_len, n_out] float32, one period, ready to loop
    chirp_bins: np.ndarray     # FFT bin numbers of the chirp lines
    chirp_phasors: np.ndarray  # unit complex phase of each chirp line (the matched filter)
    chirp_line_amp: float      # amplitude of each chirp line (cosine amplitude)
    tone_bins: np.ndarray      # FFT bin number of each pilot tone
    crest_factor: float        # peak / rms of the chirp (1.41 = pure sine)


def schroeder_phases(n_lines: int) -> np.ndarray:
    """Quadratic line phases: line m is 'played' at time m/n_lines of the period (an up-sweep)."""
    m = np.arange(n_lines)
    return -np.pi * m * m / n_lines


def make_reference(cfg: SonarConfig) -> TxReference:
    """Build one TX period and the reference the receiver correlates against."""
    cfg.validate()
    n = np.arange(cfg.frame_len)
    frame = np.zeros((cfg.frame_len, cfg.n_out), dtype=np.float64)

    # Pilot tones: plain cosines, one per speaker channel.
    tone_bins = np.array([int(round(f / cfg.bin_hz)) for f in cfg.tone_hz])
    for k, ch in zip(tone_bins, cfg.tone_channels):
        frame[:, ch] += cfg.tone_amp * np.cos(2 * np.pi * k * n / cfg.frame_len)

    # Chirp: sum of comb lines with Schroeder phases, scaled to the requested peak.
    bins = cfg.chirp_bins
    phases = schroeder_phases(len(bins))
    unit = np.zeros(cfg.frame_len)
    for k, ph in zip(bins, phases):
        unit += np.cos(2 * np.pi * k * n / cfg.frame_len + ph)
    peak = float(np.max(np.abs(unit)))
    line_amp = cfg.chirp_amp / peak
    chirp = line_amp * unit
    frame[:, cfg.chirp_channel] += chirp

    crest = float(np.max(np.abs(chirp)) / np.sqrt(np.mean(chirp**2)))
    return TxReference(
        frame=frame.astype(np.float32),
        chirp_bins=bins,
        chirp_phasors=np.exp(1j * phases),
        chirp_line_amp=float(line_amp),
        tone_bins=tone_bins,
        crest_factor=crest,
    )


def make_tx_frame(cfg: SonarConfig) -> np.ndarray:
    """One period of the speaker signal, shape [frame_len, n_out], float32."""
    return make_reference(cfg).frame


def fade_ramp(n_samples: int) -> np.ndarray:
    """Raised-cosine 0 -> 1 ramp used to start/stop the stream without a click."""
    if n_samples <= 0:
        return np.ones(0)
    return 0.5 - 0.5 * np.cos(np.pi * (np.arange(n_samples) + 0.5) / n_samples)


def tx_samples(cfg: SonarConfig, start: int, count: int, frame: np.ndarray | None = None) -> np.ndarray:
    """`count` TX samples starting at absolute sample index `start` (the period just wraps)."""
    frame = make_tx_frame(cfg) if frame is None else frame
    idx = (start + np.arange(count)) % cfg.frame_len
    return frame[idx]

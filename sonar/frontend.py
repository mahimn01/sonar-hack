"""frontend.py — raw microphone audio in, one analysed frame every 20 ms out.

Plain words: this is the one doorway all audio goes through, live or recorded.
Feed it microphone samples in any block size; every 960 samples (20 ms) it
hands back one `FrameOut` holding
  * a Doppler spectrogram column per (tone, microphone), plus a left/right
    direction signal from the microphone pair               -> gestures
  * the echo profile (strength + phase per distance)     -> presence, breathing
Because training data, the tests and the realtime engine all use this same
class, the model always sees features computed exactly as they were in training.

Read-only module: no files, no audio devices.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import SonarConfig
from .dsp import ChirpReceiver, CirFrame, DopplerSTFT, Prefilter, ToneDemod
from .tx import make_reference


@dataclass
class FrameOut:
    index: int                    # frame counter since the stream started
    t: float                      # stream time at the end of this frame (s)
    doppler_db: np.ndarray        # [n_tones * n_mics, doppler_bins] float32, dB re full scale
    lateral: np.ndarray | None    # [doppler_bins] complex64 mic-pair direction signal (None with one mic)
    cir: np.ndarray | None        # [n_mics, n_taps] complex64, None while the chirp receiver is locking
    coherence: float              # chirp lock quality 0..1 (0 while locking)
    drift_ppm: float              # speaker-vs-mic clock offset estimate
    clipped: bool                 # input touched full scale during this frame
    ambient: float = 0.0          # ultrasonic noise power of this frame, measured beside our own signal (0 while locking)


class Frontend:
    def __init__(self, cfg: SonarConfig, n_mics: int | None = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self.n_mics = cfg.n_mics if n_mics is None else n_mics
        self.ref = make_reference(cfg)
        self._pre = Prefilter(cfg, self.n_mics)
        self._demod = ToneDemod(cfg, self.n_mics)
        self._stft = DopplerSTFT(cfg, self.n_mics)
        self._chirp = ChirpReceiver(cfg, self.ref, self.n_mics)
        self._cirs: deque[CirFrame] = deque()
        self._index = 0
        self._clipped = False

    # -- read-outs ---------------------------------------------------------------
    @property
    def locked(self) -> bool:
        """True once the chirp receiver has found the direct path and learned the channel."""
        return self._chirp.state == "run"

    @property
    def n_resync(self) -> int:
        return self._chirp.n_resync

    def carrier_dbfs(self) -> np.ndarray:
        """Static level of each pilot tone at each mic, [n_tones, n_mics] dB re full scale."""
        return self._demod.carrier_dbfs()

    def notify_discontinuity(self) -> None:
        """Call after the audio driver dropped samples: the chirp alignment is no longer valid."""
        self._chirp.relock()
        self._cirs.clear()

    # -- main entry ----------------------------------------------------------------
    def process(self, x: np.ndarray) -> list[FrameOut]:
        """x: [n, n_mics] (or [n] for one mic) float samples in -1..1. Returns completed frames."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] != self.n_mics:
            raise ValueError(f"expected {self.n_mics} microphone channels, got {x.shape[1]}")
        if len(x) == 0:
            return []
        self._clipped |= bool(np.max(np.abs(x)) >= 0.999)

        y = self._pre.process(x)
        cols = self._stft.process(self._demod.process(y))
        self._cirs.extend(self._chirp.process(y))

        # Pair Doppler columns and echo profiles in order. The two paths produce one item per
        # 960 samples each; if profiles ever pile up (after a clock slip) drop the oldest.
        while len(self._cirs) > len(cols) + 2:
            self._cirs.popleft()
        out: list[FrameOut] = []
        for col, lateral in cols:
            cf = self._cirs.popleft() if self._cirs else None
            self._index += 1
            out.append(
                FrameOut(
                    index=self._index,
                    t=self._index * self.cfg.frame_len / self.cfg.fs,
                    doppler_db=col,
                    lateral=lateral,
                    cir=None if cf is None else cf.cir,
                    coherence=0.0 if cf is None else cf.coherence,
                    drift_ppm=self._chirp.drift_ppm,
                    clipped=self._clipped,
                    ambient=0.0 if cf is None else cf.ambient,
                )
            )
            self._clipped = False
        return out


def run_frontend(cfg: SonarConfig, rx: np.ndarray, block: int = 4800) -> list[FrameOut]:
    """Offline helper: push a whole recording through a fresh front end."""
    rx = np.asarray(rx)
    if rx.ndim == 1:
        rx = rx[:, None]
    fe = Frontend(cfg, n_mics=rx.shape[1])
    frames: list[FrameOut] = []
    for i in range(0, len(rx), block):
        frames.extend(fe.process(rx[i:i + block]))
    return frames

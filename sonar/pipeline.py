"""pipeline.py — the whole receive chain in one object (everything except the classifier).

Plain words: feed microphone audio in, get one `PipelineFrame` back per 20 ms
holding everything the rest of the system needs:

    feature column for the gesture model, "a movement burst just ended, here is
    its 1.6 s picture", presence state, breathing result.

The realtime engine, the recorder (live sanity read-out), the dataset builder
and the tests all use this same class, so training features and live features
cannot drift apart.

Read-only module: no files, no audio devices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .breathing import BreathingEstimator, BreathingResult
from .config import SonarConfig
from .frontend import Frontend
from .gestures import DopplerNormalizer, GestureSegmenter, Segment
from .presence import PresenceDetector, PresenceState


@dataclass
class PipelineFrame:
    index: int                           # frame counter since the stream started
    t: float                             # stream time (s)
    calibrating: bool                    # noise floors / chirp lock still being learned
    feat: np.ndarray | None              # [C, F] gesture feature column (None while calibrating)
    activity_db: float                   # hand-movement energy above the noise floor (dB)
    impulse: bool                        # this frame looks like a click / clap
    segment: Segment | None              # set on the frame where a movement burst is closed
    window: np.ndarray | None            # [C, F, T] picture of that burst, ready for the classifier
    presence: PresenceState | None       # None until the chirp receiver is locked and calibrated
    breathing: BreathingResult | None    # set about once per second
    cir: np.ndarray | None               # [n_mics, n_taps] echo profile (None while locking)
    coherence: float
    drift_ppm: float
    clipped: bool


class Pipeline:
    def __init__(self, cfg: SonarConfig, n_mics: int | None = None) -> None:
        self.cfg = cfg if n_mics is None or n_mics == cfg.n_mics else cfg.with_(n_mics=n_mics)
        cfg = self.cfg
        self.frontend = Frontend(cfg)
        self.normalizer = DopplerNormalizer(cfg)
        self.segmenter = GestureSegmenter(cfg)
        self.presence = PresenceDetector(cfg)
        self.breathing = BreathingEstimator(cfg)
        self._prev_segment_end: int | None = None
        self._last_presence: PresenceState | None = None

    # -- control -------------------------------------------------------------------
    def recalibrate(self) -> None:
        """Learn the noise floors again (call with hands away and nobody walking past)."""
        self.normalizer.recalibrate()
        self.presence.recalibrate()
        self.breathing.reset()

    def notify_discontinuity(self) -> None:
        """Audio was dropped: realign the chirp receiver and restart the slow detectors."""
        self.frontend.notify_discontinuity()
        self.breathing.restart("acquiring")

    @property
    def calibrating(self) -> bool:
        return not (self.normalizer.ready and self.frontend.locked and self.presence.calibrated)

    # -- main entry ------------------------------------------------------------------
    def process(self, rx: np.ndarray) -> list[PipelineFrame]:
        out: list[PipelineFrame] = []
        for fr in self.frontend.process(rx):
            feat = None
            segment: Segment | None = None
            window = None
            feats = self.normalizer.features(fr.doppler_db, fr.lateral)
            if feats is not None:
                feat, excess = feats
                segment = self.segmenter.push(fr.index, feat, excess)
                self.normalizer.track(fr.doppler_db, idle=self.segmenter.idle, stuck=self.segmenter.stuck)
                if segment is not None:
                    # Blank everything before the previous burst ended: no stroke is classified twice.
                    window = self.segmenter.window(segment.decided, not_before=self._prev_segment_end)
                    self._prev_segment_end = segment.end + 1

            presence = None
            breathing = None
            if fr.cir is not None:
                # Only a confident, periodic breathing detection counts as proof of life; a weak
                # "something moves slowly" reading is also produced by a person walking off.
                vital = self.breathing.last.state == "tracking"
                presence = self.presence.update(
                    fr.cir, vital=vital, doppler_db=self.segmenter.activity_db if feat is not None else 0.0
                )
                if self.presence.calibrated:
                    breathing = self.breathing.update(fr.cir, moving=presence.moving)
                    self._last_presence = presence
                else:
                    presence = None

            out.append(
                PipelineFrame(
                    index=fr.index,
                    t=fr.t,
                    calibrating=self.calibrating,
                    feat=feat,
                    activity_db=self.segmenter.activity_db if feat is not None else 0.0,
                    impulse=self.segmenter.impulse if feat is not None else False,
                    segment=segment,
                    window=window,
                    presence=presence,
                    breathing=breathing,
                    cir=fr.cir,
                    coherence=fr.coherence,
                    drift_ppm=fr.drift_ppm,
                    clipped=fr.clipped,
                )
            )
        return out

    # -- read-outs for status / self-test ------------------------------------------------
    def snr_db(self) -> float:
        """Weakest pilot-tone level over the Doppler noise floor (dB): the headroom echoes live in."""
        if self.normalizer.floor_db is None:
            return 0.0
        carrier = self.frontend.carrier_dbfs().reshape(-1)                 # [tones*mics]
        floor = np.median(self.normalizer.floor_db, axis=1)                # [tones*mics]
        return float(np.min(carrier - floor))

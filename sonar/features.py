"""features.py — stable, offline entry points to the feature pipeline.

Plain words: three functions for anything that works on recorded or simulated
audio instead of a live stream (the dataset builder, `datagen/`, notebooks):

    stream_features(rx, cfg)        whole recording -> feature columns + movement bursts
    window_from_stream(stream, ...) cut the model input picture for one burst
    clip_to_window(rx, cfg)         one short clip with one gesture -> its model input

They run the SAME `Pipeline` class as the realtime engine, so a model trained on
their output sees exactly what it will see live.

Input audio must start with about 2 s without hand movement (the noise floor is
learned from the first ~1.5 s, like the engine's start-up calibration).

Read-only module: no files, no audio devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SonarConfig
from .gestures import Segment
from .pipeline import Pipeline


@dataclass
class StreamFeatures:
    cfg: SonarConfig
    feat: np.ndarray            # [T, C, F] float32; row i belongs to frame index i + 1; zeros while calibrating
    ready: np.ndarray           # [T] bool: features valid (noise floor learned)
    activity_db: np.ndarray     # [T] movement energy above the noise floor (dB)
    segments: list[Segment] = field(default_factory=list)       # movement bursts, in time order
    not_before: list[int | None] = field(default_factory=list)  # per burst: frames before this are blanked
    presence: list[str] = field(default_factory=list)           # per frame: "empty" | "entering" | "present" | ""
    breathing: list[tuple[float, float | None, float]] = field(default_factory=list)   # (t, bpm, confidence)

    @property
    def n_frames(self) -> int:
        return len(self.feat)

    def frame_of_sample(self, sample: int) -> int:
        """Frame index (1-based, as in Segment) that contains audio sample `sample`."""
        return int(sample // self.cfg.frame_len) + 1


def stream_features(rx: np.ndarray, cfg: SonarConfig, block: int = 9600) -> StreamFeatures:
    """Run a whole recording [n, n_mics] through the pipeline."""
    rx = np.asarray(rx)
    if rx.ndim == 1:
        rx = rx[:, None]
    pipe = Pipeline(cfg, n_mics=rx.shape[1])
    cfg = pipe.cfg
    feats: list[np.ndarray] = []
    ready: list[bool] = []
    act: list[float] = []
    out = StreamFeatures(cfg, np.zeros((0, cfg.n_feat_channels, cfg.doppler_bins), np.float32), np.zeros(0, bool), np.zeros(0))
    prev_end: int | None = None
    zero = np.zeros((cfg.n_feat_channels, cfg.doppler_bins), np.float32)
    for i in range(0, len(rx), block):
        for fr in pipe.process(rx[i:i + block]):
            feats.append(zero if fr.feat is None else fr.feat)
            ready.append(fr.feat is not None)
            act.append(fr.activity_db)
            out.presence.append("" if fr.presence is None else fr.presence.state)
            if fr.breathing is not None:
                out.breathing.append((fr.t, fr.breathing.bpm, fr.breathing.confidence))
            if fr.segment is not None:
                out.segments.append(fr.segment)
                out.not_before.append(prev_end)
                prev_end = fr.segment.end + 1
    if feats:
        out.feat = np.stack(feats)
        out.ready = np.asarray(ready)
        out.activity_db = np.asarray(act)
    return out


def window_from_stream(
    stream: StreamFeatures,
    end_frame: int,
    not_before: int | None = None,
    n_frames: int | None = None,
    pad: int = 0,
) -> np.ndarray:
    """Model input picture [C, F, n_frames + 2*pad] whose un-padded part ends at frame `end_frame`.

    `pad` extra frames on both sides let the trainer shift the picture in time
    as augmentation. Frames outside the recording are zeros; frames before
    `not_before` are blanked exactly as the engine does.
    """
    cfg = stream.cfg
    n = cfg.gesture_frames if n_frames is None else n_frames
    first = end_frame - n + 1 - pad            # frame index of the first column
    last = end_frame + pad
    img = np.zeros((cfg.n_feat_channels, cfg.doppler_bins, last - first + 1), np.float32)
    lo, hi = max(first, 1), min(last, stream.n_frames)
    if hi >= lo:
        img[:, :, lo - first:hi - first + 1] = stream.feat[lo - 1:hi].transpose(1, 2, 0)
    if not_before is not None and not_before > first:
        img[:, :, :min(not_before - first, img.shape[-1])] = 0.0
    return img


def clip_to_window(rx: np.ndarray, cfg: SonarConfig, pad: int = 0) -> tuple[np.ndarray, Segment] | None:
    """One clip containing ONE gesture -> (its model input, the burst). None if nothing moved.

    The strongest movement burst of the clip is taken as the gesture.
    """
    stream = stream_features(rx, cfg)
    if not stream.segments:
        return None
    k = int(np.argmax([s.peak_db for s in stream.segments]))
    seg = stream.segments[k]
    return window_from_stream(stream, seg.decided, stream.not_before[k], pad=pad), seg

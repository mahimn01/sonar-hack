"""adapter.py — the ONLY file in datagen that knows about the core `sonar/` package.

Plain words: the simulator needs three things from the core package:
  1. the exact speaker signal (so synthetic echoes are echoes of the real probe),
  2. the exact feature function (so synthetic and real data look the same to the model),
  3. the label names and the on-disk recording format.
The core package is still being written, so everything it provides is looked up
here, in one place. If the core renames something, this is the only file to fix.
If `sonar/` is missing altogether, a small built-in fallback (four steady tones)
keeps the simulator, the sanity plots and the tests working.

Read-only helper module.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field

import numpy as np

from . import quicklook as ql

# datagen label -> core label. The core gesture model has 5 classes; every
# non-gesture scene is "none" for it, but we keep the detailed scene label too.
GESTURE_MAP = {
    "swipe_left": "swipe_left",
    "swipe_right": "swipe_right",
    "push": "push",
    "pull": "pull",
    "none": "none",
    "wave": "none",  # the core model has no "wave" class (yet): train it as a negative
    "empty": "none",
    "present": "none",
    "entering": "none",
    "leaving": "none",
}
SCENE_MAP = {
    "empty": "empty",
    "present": "present_still",
    "entering": "walk_in",
    "leaving": "walk_out",
    "none": "distractor",
}


@dataclass
class Core:
    """What the simulator needs to know about the probe signal and the features."""

    available: bool
    fs: int = 48000
    period: int = 960  # the speaker signal repeats every `period` samples
    tones: list = field(default_factory=ql.default_tones)  # steady probe tones (Hz)
    tone_speakers: list = field(default_factory=lambda: [0, 0, 0, 0])  # speaker channel of each tone
    band: tuple = (17500.0, 20500.0)  # where the whole probe signal lives
    n_speakers: int = 1
    gesture_labels: tuple = ("none", "swipe_left", "swipe_right", "push", "pull")
    cfg: object | None = None
    _frame: np.ndarray | None = None
    _mods: dict = field(default_factory=dict)

    # ---- speaker signal ---------------------------------------------------
    def tx(self, pre: int, n_out: int, post: int = 480) -> np.ndarray:
        """Speaker samples for output samples [-pre, n_out + post).

        Output sample 0 is the start of a signal period, exactly as in a real
        recording that begins on a frame boundary.
        """
        n_total = pre + n_out + post
        if self.available:
            idx = (np.arange(n_total) - pre) % self.period
            return np.asarray(self._frame, dtype=np.float64)[idx]
        t0 = -pre
        x = ql.default_tx(n_total + 0, self.fs, self.tones)
        # shift the phase so that sample `pre` is time zero (keeps periods aligned)
        n = np.arange(n_total) + t0
        t = n / self.fs
        return (0.5 / len(self.tones)) * sum(np.sin(2 * np.pi * f * t + 1.7 * k) for k, f in enumerate(self.tones))

    # ---- core modules, looked up lazily because they appear one by one -----
    def module(self, name: str) -> object | None:
        """Import `sonar.<name>` if it exists, else None (never raises)."""
        if name not in self._mods:
            try:
                self._mods[name] = importlib.import_module(f"sonar.{name}")
            except Exception:  # missing module or missing optional dependency
                self._mods[name] = None
        return self._mods[name]

    def describe(self) -> str:
        src = "core sonar/ package" if self.available else "built-in fallback (sonar/ not importable)"
        return (
            f"probe signal from {src}: fs={self.fs}, period={self.period} samples, "
            f"tones={self.tones} on speakers {self.tone_speakers}, band={self.band}"
        )


def load_core(profile: str | None = None) -> Core:
    """Read the probe signal from the core package, or fall back to built-in tones."""
    try:
        config = importlib.import_module("sonar.config")
        tx_mod = importlib.import_module("sonar.tx")
        cfg = config.get_profile(profile) if profile else config.default_config()
        frame = np.asarray(tx_mod.make_tx_frame(cfg), dtype=np.float64)
        if frame.ndim == 1:
            frame = frame[:, None]
    except Exception:
        tones = ql.default_tones()
        return Core(available=False, tones=tones, tone_speakers=[0] * len(tones), band=(tones[0], tones[-1]))
    tones = [float(f) for f in getattr(cfg, "tone_hz", ())]
    lo = min([*tones, float(getattr(cfg, "chirp_lo_hz", tones[0] if tones else 18000.0))])
    hi = max([*tones, float(getattr(cfg, "chirp_hi_hz", tones[-1] if tones else 21000.0))])
    return Core(
        available=True,
        fs=int(cfg.fs),
        period=int(frame.shape[0]),
        tones=tones,
        tone_speakers=[int(c) for c in getattr(cfg, "tone_channels", [0] * len(tones))],
        band=(lo, hi),
        n_speakers=int(frame.shape[1]),
        gesture_labels=tuple(getattr(config, "GESTURE_LABELS", ("none", "swipe_left", "swipe_right", "push", "pull"))),
        cfg=cfg,
        _frame=frame,
    )


def pre_roll(core: Core, minimum: int = 4800) -> int:
    """Pre-roll length that is a whole number of signal periods (keeps frames aligned)."""
    return int(np.ceil(minimum / core.period) * core.period)


# --------------------------------------------------------------------------
# Core feature pipeline (used when sonar/ is importable)
# --------------------------------------------------------------------------
def doppler_columns(core: Core, rx: np.ndarray) -> np.ndarray:
    """Raw mic audio -> the core's Doppler columns, [T, C, F] in dB (one column per 20 ms).

    Runs the core's own Prefilter -> ToneDemod -> DopplerSTFT, i.e. exactly the
    tone path of `sonar.frontend.Frontend`, without the chirp receiver (which
    gestures do not use and which costs most of the time).
    """
    dsp = core.module("dsp")
    if dsp is None:
        raise RuntimeError("sonar.dsp is not importable")
    cfg = core.cfg
    x = np.asarray(rx, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    # Best path: the core's own offline entry point (sonar/features.py), written for exactly this use.
    fmod = core.module("features")
    if fmod is not None:
        return wrap_stream(core, fmod.stream_features(x, cfg))
    pre, demod, stft = dsp.Prefilter(cfg, x.shape[1]), dsp.ToneDemod(cfg, x.shape[1]), dsp.DopplerSTFT(cfg, x.shape[1])
    cols = []
    for i in range(0, len(x), 4800):
        for col in stft.process(demod.process(pre.process(x[i : i + 4800]))):
            cols.append(col[0] if isinstance(col, tuple) else col)  # newer cores return (power_db, lateral)
    return np.stack(cols) if cols else np.zeros((0, cfg.n_feat_channels, cfg.doppler_bins), dtype=np.float32)


@dataclass
class Stream:
    """What the core's realtime chain makes of one clip."""

    feat: np.ndarray  # [T, C, F] features in 0..1 (zeros while the noise floor is still being learned)
    ready_from: int  # first frame with valid features
    segments: list  # sonar.gestures.Segment objects (movement bursts the core would classify)
    windows: list  # for each segment: the model input [C, F, gesture_frames]
    frame_rate: float
    frames: list | None = None  # the core's PipelineFrame objects (presence, breathing ...), when available
    core: object | None = None  # the core's own StreamFeatures object, when sonar/features.py exists


def wrap_stream(core: Core, st: object) -> Stream:
    """The core's StreamFeatures object -> the Stream the label code works with (one model input per burst)."""
    fmod = core.module("features")
    windows = [fmod.window_from_stream(st, s.decided, nb) for s, nb in zip(st.segments, st.not_before)]
    feat = np.concatenate([np.zeros((1, *st.feat.shape[1:]), np.float32), st.feat])  # row k = core frame k (1-based)
    ready = np.nonzero(st.ready)[0]
    stream = Stream(feat, int(ready[0]) + 1 if len(ready) else -1, st.segments, windows, float(st.cfg.frame_rate))
    stream.core = st
    return stream


def stream_features(core: Core, rx: np.ndarray) -> Stream:
    """Run the core's realtime chain over a clip exactly as a live stream would.

    Preferred path: `sonar.pipeline.Pipeline`, the one class the core engine,
    recorder and dataset builder all use, so synthetic features are computed by
    literally the same code as live ones. (Older core versions without
    pipeline.py fall back to wiring the same blocks by hand.)

    The clip must begin with ~1.5 s in which nothing is scripted: that is when
    the core learns its noise floor (12 warm-up frames + 60 calibration frames).
    """
    cfg = core.cfg
    x = np.asarray(rx, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    # Best path: the core's own offline entry point (sonar/features.py), written for exactly this use.
    fmod = core.module("features")
    if fmod is not None:
        return wrap_stream(core, fmod.stream_features(x, cfg))
    pipe_mod = core.module("pipeline")
    if pipe_mod is not None:
        pipe = pipe_mod.Pipeline(cfg, n_mics=x.shape[1])
        frames = []
        for i in range(0, len(x), 4800):
            frames.extend(pipe.process(x[i : i + 4800]))
        first = next((f.feat for f in frames if f.feat is not None), None)
        if first is None:
            return Stream(np.zeros((len(frames), 0, 0), np.float32), -1, [], [], float(cfg.frame_rate))
        feat = np.zeros((len(frames) + 1, *first.shape), dtype=np.float32)  # row k = core frame index k (1-based)
        segments, windows, ready_from = [], [], -1
        for f in frames:
            if f.feat is not None:
                if ready_from < 0:
                    ready_from = int(f.index)
                feat[int(f.index)] = f.feat
            if f.segment is not None:
                segments.append(f.segment)
                windows.append(f.window)
        stream = Stream(feat, ready_from, segments, windows, float(cfg.frame_rate))
        stream.frames = frames  # presence / breathing read-outs ride along for free
        return stream

    g = core.module("gestures")
    if g is None:
        raise RuntimeError("neither sonar.pipeline nor sonar.gestures is importable")
    cols = doppler_columns(core, rx)
    norm, seg = g.DopplerNormalizer(cfg), g.GestureSegmenter(cfg)
    wants_excess = "excess_db" in inspect.signature(seg.push).parameters
    feat = np.zeros(cols.shape, dtype=np.float32)
    segments, windows, ready_from = [], [], -1
    for i, col in enumerate(cols):
        if wants_excess:
            ex = norm.excess_db(col)
            if ex is None:
                continue
            f = norm.to_feature(ex)
            s = seg.push(i, f, ex)
        else:
            f = norm.normalize(col)
            if f is None:
                continue
            s = seg.push(i, f)
        if ready_from < 0:
            ready_from = i
        feat[i] = f
        idle = bool(getattr(seg, "idle", not seg.active))
        norm.track(col, idle=idle, stuck=bool(getattr(seg, "stuck", False)))
        if s is not None:
            segments.append(s)
            windows.append(seg.window(s.decided))
    return Stream(feat, ready_from, segments, windows, float(cfg.frame_rate))


def train_pad(core: Core) -> int:
    """Extra frames the core's training pictures carry on each side (for time-shift augmentation)."""
    ds = core.module("dataset")
    return int(getattr(ds, "PAD", 0)) if ds is not None else 0


def padded_window(core: Core, stream: Stream, k: int | None = None, end_frame: int | None = None) -> np.ndarray:
    """Training picture [C, F, T + 2*pad] for burst `k` (or ending at `end_frame`), cut by the core's own function."""
    pad = train_pad(core)
    fmod = core.module("features")
    if fmod is not None and stream.core is not None:
        if k is not None:
            return fmod.window_from_stream(stream.core, stream.segments[k].decided, stream.core.not_before[k], pad=pad)
        return fmod.window_from_stream(stream.core, int(end_frame), pad=pad)
    win = stream.windows[k] if k is not None else window_at(core, stream, int(end_frame))
    return np.pad(win, ((0, 0), (0, 0), (pad, pad)))


def window_at(core: Core, stream: Stream, end_frame: int) -> np.ndarray:
    """Model input [C, F, gesture_frames] whose last column is `end_frame` (zero-padded at the start)."""
    n = int(core.cfg.gesture_frames)
    end = int(np.clip(end_frame, 0, len(stream.feat) - 1))
    chunk = stream.feat[max(0, end - n + 1) : end + 1]
    if len(chunk) < n:
        chunk = np.concatenate([np.zeros((n - len(chunk), *chunk.shape[1:]), dtype=chunk.dtype), chunk])
    return np.transpose(chunk, (1, 2, 0))

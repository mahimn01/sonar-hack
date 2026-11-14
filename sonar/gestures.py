"""gestures.py — from Doppler columns to "a gesture just happened".

Plain words, three small steps:

  1. `DopplerNormalizer` learns how loud each Doppler bin is when nothing moves
     (the noise floor) and rescales every column to "how far above the floor",
     clipped to 0..1. That makes the picture independent of speaker volume.
  2. `GestureSegmenter` watches the amount of movement energy and cuts out the
     short burst where a hand moved (start/stop thresholds, minimum length,
     click rejection). When a burst ends it hands over the last 1.6 s of
     picture: the model input.
  3. `heuristic_classify` is a physics-only classifier that needs no training:
       * a hand moving sideways moves AWAY from one speaker and TOWARDS the
         other, so the two pilot tones show opposite Doppler, and it moves
         towards one microphone and away from the other, which the mic-pair
         "lateral" picture shows directly -> swipe direction (two independent cues);
       * a hand moving at the screen shifts everything the same way: a quick
         thrust and return (push) or a slow reach and a quick retreat (pull).
     It is the fallback when no trained model is present and a sanity check on
     the model when one is.

Feature image layout (what the model sees), shape [C, F, T]:
    channels 0 .. n_tones*n_mics-1   Doppler power, 0..1 (dB above noise floor / 36 dB),
                                     channel = tone * n_mics + mic
    last channel (two mics only)     lateral: -1..1, <0 moving right, >0 moving left,
                                     shown only where there is echo energy
    F = 64 Doppler bins from -250 Hz (moving away) to +242 Hz (approaching)
    T = 80 frames of 20 ms, newest last

Read-only module: no files, no audio devices.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from .config import GESTURE_LABELS, SonarConfig

_LOG_MEAN_BIAS_DB = 2.51      # E[10 log10 X] is 2.51 dB below 10 log10 E[X] for noise power X
_MEDIAN_BIAS_DB = 1.59        # the median of noise power is 1.59 dB below its mean


# ---------------------------------------------------------------------------
# 1. Noise floor and normalisation
# ---------------------------------------------------------------------------
class DopplerNormalizer:
    """Tracks the per-bin noise floor and maps dB columns to 0..1 features."""

    def __init__(self, cfg: SonarConfig, warmup_frames: int = 12, calib_frames: int = 60) -> None:
        self.cfg = cfg
        self.floor_db: np.ndarray | None = None        # [C, F] estimate of the mean noise power in dB
        self._warmup = warmup_frames                   # filters and STFT window still filling
        self._calib_target = calib_frames
        self._calib: list[np.ndarray] = []
        self._alpha = 1.0 / (8.0 * cfg.frame_rate)     # 8 s tracking time constant
        self._alpha_fast = 1.0 / (1.5 * cfg.frame_rate)

    @property
    def ready(self) -> bool:
        return self.floor_db is not None

    def set_floor(self, floor_db: np.ndarray) -> None:
        self.floor_db = np.array(floor_db, dtype=np.float32)

    def recalibrate(self) -> None:
        """Forget the floor and learn it again from the next ~1.2 s."""
        self.floor_db = None
        self._calib = []
        self._warmup = 0

    def excess_db(self, col_db: np.ndarray) -> np.ndarray | None:
        """dB column [C, F] -> dB above the noise floor (unclipped), or None while the floor is being learned."""
        if self.floor_db is None:
            if self._warmup > 0:
                self._warmup -= 1
                return None
            self._calib.append(col_db)
            if len(self._calib) < self._calib_target:
                return None
            # Median over time ignores a brief movement during calibration; a light
            # median across frequency removes the estimation noise of single bins.
            floor = np.median(np.stack(self._calib), axis=0) + _MEDIAN_BIAS_DB
            self.floor_db = median_filter(floor, size=(1, 5), mode="nearest").astype(np.float32)
            self._calib = []
        return col_db - self.floor_db

    def to_feature(self, excess_db: np.ndarray) -> np.ndarray:
        """dB above the floor -> model feature in 0..1."""
        cfg = self.cfg
        return np.clip((excess_db - cfg.feat_floor_margin_db) / cfg.feat_range_db, 0.0, 1.0).astype(np.float32)

    def normalize(self, col_db: np.ndarray) -> np.ndarray | None:
        """dB column [P, F] -> power features in 0..1, or None while the floor is still being learned."""
        ex = self.excess_db(col_db)
        return None if ex is None else self.to_feature(ex)

    def features(self, col_db: np.ndarray, lateral: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
        """One frame -> (model feature column [C, F], dB above floor [P, F]); None while calibrating."""
        ex = self.excess_db(col_db)
        if ex is None:
            return None
        feat = self.to_feature(ex)
        if lateral is not None and self.cfg.has_lateral:
            feat = np.concatenate([feat, lateral_feature(feat, lateral, self.cfg)[None, :]], axis=0)
        return feat, ex

    def track(self, col_db: np.ndarray, idle: bool, stuck: bool = False) -> None:
        """Follow slow changes of the floor. Only while idle, or quickly when activity never ends."""
        if self.floor_db is None or not (idle or stuck):
            return
        a = self._alpha_fast if stuck else self._alpha
        target = col_db + _LOG_MEAN_BIAS_DB
        # Bins far above the floor are movement, not noise: limit how hard they can pull.
        self.floor_db += a * np.clip(target - self.floor_db, -12.0, 6.0 if not stuck else 30.0)


def lateral_mask(power_feat: np.ndarray, cfg: SonarConfig) -> np.ndarray:
    """Where both microphones see echo energy (0..1 per Doppler bin). power_feat: [P, F, ...]."""
    per_mic = power_feat.reshape(cfg.n_tones, cfg.n_mics, *power_feat.shape[1:]).mean(axis=0)
    return np.minimum(1.0, 3.0 * np.sqrt(per_mic[0] * per_mic[1]))


def lateral_feature(power_feat: np.ndarray, lateral: np.ndarray, cfg: SonarConfig) -> np.ndarray:
    """Mic-pair direction picture: phase-turn per frame scaled to -1..1, blanked where there is no echo."""
    return (np.angle(lateral) / np.pi * lateral_mask(power_feat, cfg)).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Segmenter
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    start: int          # frame index where the movement began
    end: int            # frame index of the last active frame
    decided: int        # frame index at which the segment was closed (= window end)
    peak_db: float      # strongest in-band power above the noise floor (dB)
    energy_db: float    # summed activity (dB * frames)
    forced: bool        # True if cut because it exceeded the maximum length

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1


class GestureSegmenter:
    """Energy-based burst detector with hysteresis; keeps the recent feature history.

    Activity of a frame = 10*log10(mean over channels and in-band Doppler bins of
    power / noise-floor power). Pure noise gives about 0 dB (+-0.5 dB); a hand
    whose Doppler ridge is 20 dB above the floor gives roughly 9 dB.
    """

    def __init__(self, cfg: SonarConfig, history_frames: int = 400) -> None:
        self.cfg = cfg
        f = np.abs(cfg.doppler_hz)
        self._band = (f >= cfg.seg_band_hz[0]) & (f <= cfg.seg_band_hz[1])
        self._outer = self._band & (f >= cfg.impulse_outer_hz)
        self._inner = self._band & ~self._outer
        self._pos = self._band & (cfg.doppler_hz > 0)
        self._neg = self._band & (cfg.doppler_hz < 0)
        self._impulse_hold = 0
        self._hist: deque[np.ndarray] = deque(maxlen=history_frames)
        self._hist_index: deque[int] = deque(maxlen=history_frames)
        self.on_thr = cfg.seg_on_db
        self.off_thr = cfg.seg_off_db
        self._idle_mean = 0.0
        self._idle_var = 0.25
        self._active = False
        self._start = 0
        self._last_loud = 0
        self._onset = 0
        self._peak = 0.0
        self._energy = 0.0
        self._long_active = 0
        self.activity_db = 0.0
        self.impulse = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def idle(self) -> bool:
        """Nothing is moving right now (safe moment to update the noise floor)."""
        return not self._active and self.activity_db < self.off_thr and self._impulse_hold == 0

    @property
    def stuck(self) -> bool:
        """Activity has not stopped for 4 s: more likely a new noise level than a gesture."""
        return self._long_active > 4.0 * self.cfg.frame_rate

    def frame_activity(self, excess_db: np.ndarray) -> tuple[float, bool]:
        """(activity in dB, is this frame a broadband click?) for one column of dB-above-floor [C, F]."""
        ratio = 10.0 ** (np.clip(excess_db, -30.0, 60.0) / 10.0)
        inner = float(ratio[:, self._inner].mean())
        outer = float(ratio[:, self._outer].mean())
        pos = float(ratio[:, self._pos].mean()) - 1.0
        neg = float(ratio[:, self._neg].mean()) - 1.0
        act = 10.0 * np.log10(float(ratio[:, self._band].mean()))
        # A click or clap is flat across Doppler: the fast outer bins rise as much as the inner
        # ones, and "towards" and "away" rise together. A hand is a narrow ridge on one side.
        impulse = (
            (inner - 1.0) > 0.4
            and (outer - 1.0) >= self.cfg.impulse_ratio * (inner - 1.0)
            and min(pos, neg) >= 0.5 * max(pos, neg)
        )
        return act, bool(impulse)

    def _update_idle_stats(self, act: float) -> None:
        a = 1.0 / (10.0 * self.cfg.frame_rate)
        self._idle_mean += a * (act - self._idle_mean)
        self._idle_var += a * ((act - self._idle_mean) ** 2 - self._idle_var)
        sd = float(np.sqrt(max(self._idle_var, 1e-6)))
        self.on_thr = max(self.cfg.seg_on_db, self._idle_mean + 4.0 * sd)
        self.off_thr = max(self.cfg.seg_off_db, self._idle_mean + 2.0 * sd)

    def push(self, index: int, feat: np.ndarray, excess_db: np.ndarray) -> Segment | None:
        """Add one frame (model feature column + dB above floor); returns a Segment when a burst closes."""
        cfg = self.cfg
        self._hist.append(feat)
        self._hist_index.append(index)
        act, impulse = self.frame_activity(excess_db)
        self.activity_db, self.impulse = act, impulse
        # A click neither starts nor sustains a gesture, and neither does the 0.2 s of filter
        # ringing that follows a loud one.
        if impulse:
            self._impulse_hold = 10
        held = self._impulse_hold > 0
        if held:
            self._impulse_hold -= 1
            act = -10.0

        if not self._active:
            self._long_active = 0
            if act < self.off_thr and not held:     # learn the idle level from real quiet frames only
                self._update_idle_stats(act)
            self._onset = self._onset + 1 if act >= self.on_thr else 0
            if self._onset >= 2:
                self._active = True
                self._start = index - 1
                self._last_loud = index
                self._peak = act
                self._energy = act
                self._onset = 0
            return None

        self._long_active += 1
        self._peak = max(self._peak, act)
        self._energy += max(act, 0.0)
        if act >= self.off_thr:
            self._last_loud = index
        quiet = index - self._last_loud
        too_long = index - self._start + 1 >= cfg.seg_max_frames
        if quiet < cfg.seg_hang_frames and not too_long:
            return None
        self._active = False
        seg = Segment(self._start, self._last_loud, index, self._peak, self._energy, forced=too_long)
        if seg.n_frames < cfg.seg_min_frames or seg.peak_db < cfg.seg_min_peak_db:
            return None
        return seg

    def window(self, end_index: int, n_frames: int | None = None, not_before: int | None = None) -> np.ndarray | None:
        """Feature image [C, F, n_frames] whose last column is frame `end_index`.

        Columns older than `not_before` are blanked: pass the end of the previous
        burst there so that one stroke is never classified twice (once on its own
        and again as the "history" of the hand dropping back afterwards).
        Returns None if `end_index` is no longer (or not yet) in the history.
        """
        n = self.cfg.gesture_frames if n_frames is None else n_frames
        if not self._hist_index or end_index > self._hist_index[-1]:
            return None
        back = self._hist_index[-1] - end_index
        cols = list(self._hist)
        stop = len(cols) - back
        if stop <= 0:
            return None
        if stop - n < 0:
            pad = [np.zeros_like(cols[0])] * (n - stop)
            chosen = pad + cols[:stop]
        else:
            chosen = cols[stop - n:stop]
        img = np.stack(chosen, axis=-1)
        if not_before is not None:
            first = end_index - n + 1                    # frame index of column 0
            blank = min(max(not_before - first, 0), n)
            img[:, :, :blank] = 0.0
        return img


# ---------------------------------------------------------------------------
# 3. Physics-only classifier
# ---------------------------------------------------------------------------
@dataclass
class HeuristicResult:
    label: str
    confidence: float
    lateral_hz: float        # two-speaker cue: left-tone Doppler minus right-tone Doppler; negative = moving right
    lateral_mic_hz: float    # two-microphone cue: Doppler at mic 0 minus mic 1; negative = moving right
    order_s: float           # time of "moving away" energy minus time of "approaching" energy
    probs: np.ndarray        # pseudo-probabilities in GESTURE_LABELS order


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def heuristic_classify(window: np.ndarray, cfg: SonarConfig) -> HeuristicResult:
    """Classify a feature image [C, F, T] with Doppler sign rules only (no training).

    Direction score = w_speakers * (speaker cue / 15 Hz) + w_mics * (mic cue / 5 Hz);
    15 Hz and 5 Hz are roughly what a slow swipe produces on a 15-inch laptop.
    `cfg.heur_w_speakers` / `cfg.heur_w_mics` flip or mute a cue if the channels
    of a device turn out to be wired the other way round (train.py fits them
    from recorded swipes).
    """
    n_p = cfg.n_power_channels
    power = window[:n_p]
    f = cfg.doppler_hz
    use = np.abs(f) >= 15.0
    # One picture per tone (average the microphones), keep only clear energy.
    per_tone = power.reshape(cfg.n_tones, cfg.n_mics, *power.shape[1:]).mean(axis=1)    # [tones, F, T]
    peak = float(per_tone[:, use, :].max())
    w = np.maximum(per_tone - max(0.08, 0.3 * peak), 0.0) * use[None, :, None]
    e = w.sum(axis=1)                                                                    # [tones, T]
    cen = (w * f[None, :, None]).sum(axis=1) / (e + 1e-9)                                # Doppler centroid (Hz)

    probs = np.full(len(GESTURE_LABELS), 0.02)
    strong_frames = int(np.sum(e.sum(axis=0) > 0.5 * peak))
    if peak < 0.12 or strong_frames < 4:
        probs[0] = 1.0
        return HeuristicResult("none", 1.0, 0.0, 0.0, 0.0, probs / probs.sum())

    # Focus on the stroke itself: the strongest ~0.5 s. The slower wind-up before it and the
    # hand dropping back afterwards are down-weighted so they cannot vote on the direction.
    t = np.arange(window.shape[-1]) / cfg.frame_rate
    e_tot = np.convolve(e.sum(axis=0), np.ones(7) / 7.0, mode="same")
    focus = np.exp(-0.5 * ((t - t[int(np.argmax(e_tot))]) / 0.25) ** 2)
    w = w * focus[None, None, :]
    e = e * focus[None, :]

    # Cue 1 (two speakers): Doppler of the left-speaker tone minus the right-speaker tone.
    lateral = 0.0
    common_rms = float(np.sqrt((e * cen**2).sum() / (e.sum() + 1e-9)))
    if cfg.n_tones >= 2 and cfg.tone_channels[0] != cfg.tone_channels[1]:
        both = np.minimum(e[0], e[1])
        lateral = float((both * (cen[0] - cen[1])).sum() / (both.sum() + 1e-9))         # Hz, <0 = moving right
    # Cue 2 (two microphones): phase-turn rate between the mics where there is energy.
    lateral_mic = 0.0
    if cfg.has_lateral and window.shape[0] > n_p:
        mask = lateral_mask(power, cfg) * use[:, None] * focus[None, :]
        turn = float((window[n_p] * use[:, None] * focus[None, :]).sum() / (mask.sum() + 1e-9))   # mean angle / pi
        lateral_mic = turn * np.pi / (2 * np.pi * cfg.frame_len / cfg.fs)               # Hz
    direction = cfg.heur_w_speakers * lateral / 15.0 + cfg.heur_w_mics * lateral_mic / 5.0

    # When did the hand approach (positive Doppler) and when did it move away (negative),
    # and how fast was each of the two phases?
    w_pos = w[:, f > 0, :].sum(axis=0)                                                   # [F+, T]
    w_neg = w[:, f < 0, :].sum(axis=0)
    pos, neg = w_pos.sum(axis=0), w_neg.sum(axis=0)
    t_pos = float((pos * t).sum() / (pos.sum() + 1e-9))
    t_neg = float((neg * t).sum() / (neg.sum() + 1e-9))
    order = t_neg - t_pos                                                                # >0: approached first
    balance = float(min(pos.sum(), neg.sum()) / (max(pos.sum(), neg.sum()) + 1e-9))
    v_pos = float((w_pos * f[f > 0][:, None]).sum() / (pos.sum() + 1e-9))               # mean approach Doppler (Hz)
    v_neg = float((w_neg * -f[f < 0][:, None]).sum() / (neg.sum() + 1e-9))              # mean retreat Doppler (Hz)

    # Sideways if the direction cues are clear AND not small next to the towards/away Doppler.
    radial_index = common_rms / 40.0
    swipe_score = _sigmoid((abs(direction) - 0.9) / 0.25) * _sigmoid((abs(direction) - 0.7 * radial_index) / 0.2)
    radial_score = (1.0 - swipe_score) * _sigmoid((balance - 0.12) / 0.05) * _sigmoid((abs(order) - 0.05) / 0.02)
    # push = a quick thrust at the screen and back; pull = a slow reach followed by a quick
    # retreat (or a retreat first, if the hand was already near the screen).
    is_push = order > 0 and v_pos >= 0.7 * v_neg

    idx = {name: i for i, name in enumerate(GESTURE_LABELS)}
    probs[idx["swipe_right" if direction < 0 else "swipe_left"]] += swipe_score
    probs[idx["push" if is_push else "pull"]] += radial_score
    probs[idx["none"]] += max(0.0, 1.0 - swipe_score - radial_score)
    probs = probs / probs.sum()
    best = int(np.argmax(probs))
    return HeuristicResult(GESTURE_LABELS[best], float(probs[best]), lateral, lateral_mic, order, probs)

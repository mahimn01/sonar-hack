"""presence.py — is somebody there? (empty / entering / present)

Plain words: the echo profile tells us how strong the echo is at each distance,
50 times a second. An empty room gives the same profile every time. A person
changes it in two ways:

  * gross movement (walking, reaching)   -> the profile changes quickly, and the
                                            pilot tones show Doppler energy
  * "micro movement" (breathing, sway)   -> it changes slowly but never stops

We measure both kinds of change per distance, compare them with their noise
floors, and run a small state machine:

    empty --(gross movement for 0.3 s)--> entering --(still for 1 s)--> present
    present --(walked, then nothing moves for 5 s)--> empty        (they left)
    present --(nothing moves at all, no walking before)--> stays present and
            counts `motionless_s`: the person is probably still there but not
            breathing visibly. The dashboard turns that into an alert. After
            40 s of nothing the detector gives up and says empty.

Only distances inside `presence_max_range_m` (2 m) count for the echo profile,
so people walking further back in the hall do not register there.

No training needed. Read-only module: no files, no audio devices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .config import SonarConfig

_TINY = 1e-30


@dataclass
class PresenceState:
    state: str              # "empty" | "entering" | "present"
    confidence: float       # 0..1
    moving: bool            # gross movement right now
    still_s: float          # seconds since the last gross movement
    range_m: float | None   # distance of the strongest movement, None when empty
    fast_db: float          # gross-movement level above its floor (dB), nearest 2 m
    slow_db: float          # micro-movement level above its floor (dB), nearest 2 m
    near_db: float          # gross-movement level in the hand zone (< 0.6 m)
    changed: bool           # True on the frame where `state` changed
    motionless_s: float = 0.0   # while present: seconds without ANY movement, breathing included


def _alpha(tau_s: float, rate: float) -> float:
    return 1.0 - float(np.exp(-1.0 / (tau_s * rate)))


class PresenceDetector:
    def __init__(self, cfg: SonarConfig, calib_s: float = 4.0) -> None:
        self.cfg = cfg
        rate = cfg.frame_rate
        # Gross movement = what survives a 7 Hz high-pass along time. Breathing and sway change the
        # echo at under 3 Hz and are rejected by > 25 dB, however strong the chest echo is.
        self._hp_sos = sps.butter(4, cfg.presence_fast_hp_hz, "highpass", fs=rate, output="sos")
        self._a_s1 = _alpha(0.25, rate)                        # short average (removes sensor noise)
        self._a_s2 = _alpha(cfg.presence_slow_tau_s, rate)     # long average (the "empty room" picture)
        self._a_pf = _alpha(0.3, rate)                         # smoothing of the gross-movement power
        self._a_ps = _alpha(1.5, rate)                         # smoothing of the micro-movement power
        r = cfg.range_m
        taps = np.arange(cfg.n_taps)
        valid = taps > cfg.direct_tap + 1                      # skip the direct path itself
        self._room = valid & (r <= cfg.presence_max_range_m)
        self._near = valid & (r <= cfg.presence_near_m)
        self._range = r
        self._calib_frames = int(calib_s * rate)
        self.reset()

    # -- state ---------------------------------------------------------------
    def reset(self) -> None:
        self._hp_zi: np.ndarray | None = None
        self._s1: np.ndarray | None = None
        self._s2: np.ndarray | None = None
        self._pf: np.ndarray | None = None        # smoothed gross-movement power per tap
        self._ps: np.ndarray | None = None        # smoothed micro-movement power per tap
        self.floor_fast: np.ndarray | None = None
        self.floor_slow: np.ndarray | None = None
        self._calib_pf: list[np.ndarray] = []
        self._calib_ps: list[np.ndarray] = []
        self._n = 0
        self.state = "empty"
        self._fast_run = 0
        self._quiet_run = 0
        self._last_motion = -1e9      # frame index of the last movement of any kind
        self._last_fast = -1e9
        self._last_micro = -1e9       # frame index of the last micro movement (breathing / sway)
        self._state_since = 0
        self._range_smooth: float | None = None

    @property
    def calibrated(self) -> bool:
        return self.floor_fast is not None

    def recalibrate(self) -> None:
        self.reset()

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _top_mean_db(ratio: np.ndarray, k: int = 3) -> float:
        """dB of the mean of the k largest values (one noisy tap cannot trigger alone)."""
        if ratio.size == 0:
            return -99.0
        top = np.sort(ratio)[-k:]
        return float(10.0 * np.log10(top.mean() + _TINY))

    def _finish_calibration(self) -> None:
        pf = np.stack(self._calib_pf)
        ps = np.stack(self._calib_ps)
        # Low percentile: movement during calibration can only raise the power, never lower it.
        self.floor_fast = np.percentile(pf, 30, axis=0) + _TINY
        # Micro-movement floor: measured, but kept within 3..9 dB of what pure sensor noise
        # predicts, so that a person standing there during calibration cannot blind the detector.
        thermal_slow = 0.055 * self.floor_fast     # short-average noise (0.04) over high-passed noise (0.72)
        self.floor_slow = np.clip(np.percentile(ps, 50, axis=0), 2.0 * thermal_slow, 8.0 * thermal_slow) + _TINY
        self._calib_pf, self._calib_ps = [], []

    # -- main entry ----------------------------------------------------------------
    def update(self, cir: np.ndarray, vital: bool = False, doppler_db: float = 0.0) -> PresenceState:
        """cir: [n_mics, n_taps] complex echo profile of one frame.

        `vital` = the breathing estimator currently sees a breathing-like moving echo.
        It is ~10 dB more sensitive than the per-tap detector here (it combines taps
        and 10-30 s of data), so it keeps a quiet, still person "present".
        `doppler_db` = hand/body movement energy from the pilot tones (gesture
        segmenter activity). The tones are stronger than the chirp, so they see a
        person walking up earlier; they are not range-gated, hence the higher bar.
        """
        cfg = self.cfg
        h = cir.astype(np.complex128)
        self._n += 1
        if self._hp_zi is None:
            self._s1, self._s2 = h.copy(), h.copy()
            self._pf = np.zeros(h.shape[1])
            self._ps = np.zeros(h.shape[1])
            # start the high-pass in steady state for this static picture (no start-up burst)
            self._hp_zi = sps.sosfilt_zi(self._hp_sos)[:, :, None, None] * h[None, None, :, :]

        d_fast, self._hp_zi = sps.sosfilt(self._hp_sos, h[None, :, :], axis=0, zi=self._hp_zi)
        d_fast = d_fast[0]
        self._s1 += self._a_s1 * (h - self._s1)
        self._s2 += self._a_s2 * (h - self._s2)
        d_slow = self._s1 - self._s2
        self._pf += self._a_pf * (np.mean(np.abs(d_fast) ** 2, axis=0) - self._pf)
        self._ps += self._a_ps * (np.mean(np.abs(d_slow) ** 2, axis=0) - self._ps)

        if self.floor_fast is None:
            if self._n > 0.25 * self._calib_frames:        # let the smoothers settle first
                self._calib_pf.append(self._pf.copy())
                self._calib_ps.append(self._ps.copy())
            if self._n >= self._calib_frames:
                self._finish_calibration()
            return PresenceState("empty", 0.0, False, 0.0, None, 0.0, 0.0, 0.0, False)

        assert self.floor_slow is not None
        rf = self._pf / self.floor_fast
        rs = self._ps / self.floor_slow
        fast_db = self._top_mean_db(rf[self._room])
        slow_db = self._top_mean_db(rs[self._room])
        near_db = self._top_mean_db(rf[self._near])
        fast_on = fast_db > cfg.presence_fast_thresh_db or doppler_db > cfg.presence_doppler_thresh_db
        slow_on = slow_db > cfg.presence_slow_thresh_db or vital

        rate = cfg.frame_rate
        prev = self.state
        if fast_on:
            self._fast_run += 1
            self._quiet_run = 0
            self._last_fast = self._n
        else:
            self._quiet_run += 1
            # The scene just stopped changing: take a fresh "background" picture so that only
            # ongoing micro movement (breathing, sway) keeps the slow detector on.
            if self._quiet_run == int(cfg.presence_settle_s * rate):
                self._s2 = self._s1.copy()
                self._ps *= 0.0
                self._fast_run = 0
        if fast_on:
            self._last_motion = self._n
        # Micro movement only counts once the scene has settled after gross movement; right after
        # somebody walked, the "slow" detector is still reacting to the changed picture.
        if slow_on and self._quiet_run > int(cfg.presence_settle_s * rate) + 5:
            self._last_motion = self._n
            self._last_micro = self._n

        if self.state == "empty":
            if self._fast_run >= cfg.presence_enter_s * rate:
                self.state = "entering"
            elif slow_on and self._n - self._state_since > 3.0 * rate and self._quiet_run > 3.0 * rate:
                self.state = "present"          # somebody was already there, sitting still
        elif self.state == "entering":
            if self._quiet_run >= cfg.presence_settle_s * rate:
                self.state = "present"
        elif self.state == "present":
            silent = (self._n - self._last_motion) / rate
            # Was the last thing we saw gross movement (walking away), or did a still person just stop moving?
            walked_last = self._last_fast >= self._last_micro
            if silent > (cfg.presence_absent_s if walked_last else cfg.presence_motionless_s):
                self.state = "empty"
        changed = self.state != prev
        if changed:
            self._state_since = self._n

        # Slowly follow the floors while the room is empty (temperature, fan speed ...).
        if self.state == "empty" and not fast_on and not slow_on:
            self.floor_fast += (self._pf - self.floor_fast) / (20.0 * rate)

        # Where is the movement? (strongest tap inside the room)
        rng: float | None = None
        if self.state != "empty":
            score = np.where(self._room, rf if fast_on else rs, 0.0)
            est = float(self._range[int(np.argmax(score))])
            self._range_smooth = est if self._range_smooth is None else 0.9 * self._range_smooth + 0.1 * est
            rng = round(self._range_smooth, 2)
        else:
            self._range_smooth = None

        still_s = (self._n - self._last_fast) / rate
        motionless = 0.0
        if self.state == "present":
            motionless = (self._n - self._last_motion) / rate
            margin = max(fast_db - cfg.presence_fast_thresh_db, slow_db - cfg.presence_slow_thresh_db, 6.0 if vital else -99.0)
            conf = (0.5 + 0.5 * np.tanh(margin / 6.0)) * (1.0 - 0.5 * min(motionless / cfg.presence_absent_s, 1.0))
        elif self.state == "entering":
            conf = 0.5 + 0.5 * np.tanh((fast_db - cfg.presence_fast_thresh_db) / 6.0)
        else:
            quiet = (self._n - max(self._last_motion, self._state_since)) / rate
            conf = 1.0 - np.exp(-max(quiet, 0.0) / 2.0)
        return PresenceState(
            state=self.state,
            confidence=float(np.clip(conf, 0.0, 1.0)),
            moving=bool(fast_on),
            still_s=float(still_s),
            range_m=rng,
            fast_db=fast_db,
            slow_db=slow_db,
            near_db=near_db,
            changed=changed,
            motionless_s=float(motionless),
        )

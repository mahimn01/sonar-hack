"""presence.py — is somebody there? (empty / entering / present)

Plain words: the echo profile tells us how strong the echo is at each distance,
50 times a second. An empty room gives the same profile every time. A person
changes it in two ways:

  * gross movement (walking, reaching)   -> the profile changes quickly
  * "micro movement" (breathing, sway)   -> it changes slowly but never stops

We measure both kinds of change per distance, compare them with their noise
floors, and run a small state machine:

    empty --(gross movement in front of the laptop for 0.3 s)--> entering --(still for 1 s)--> present
    present --(walked, then nothing moves for 5 s)--> empty        (they left)
    present --(nothing moves at all, no walking before)--> stays present and
            counts `motionless_s`: the person is probably still there but not
            breathing visibly. The dashboard turns that into an alert. After
            40 s of nothing the detector gives up and says empty.

Three things keep this honest in a busy room (learned on the first run in a
hackathon hall, where the first version said "moving" 50 times a minute):

  1. Only "the person zone" counts: 0.3 m to 1.3 m in front of the laptop.
     Other people walking 1.5 m or more away are at other distances and are
     ignored. Once somebody is tracked, only movement up to 0.35 m behind them
     is called theirs. The pilot tones are NOT used here: they cannot tell
     distance, so they hear the whole hall.
  2. The noise level is measured again on EVERY frame, as the middle value over
     all distances. A burst of noise (keys, a wrapper, a clap, a hissing "s")
     raises most distances at once; real movement sits at a few. Every frame is
     divided by its own noise level, and a frame that is noisier than usual
     counts for less in every average (a 2 ms click is all but ignored). A
     second noise meter listens beside our own signal, on frequencies where we
     transmit nothing, and catches the clicks that land on only some of the
     distances. The breathing estimator uses the same per-frame weight.
  3. Switching on needs 8 dB, switching off needs to fall under 5 dB, and each
     distance settles on its own, so a walker at the edge of the zone does not
     wipe out what we know about the person sitting in the middle of it.

No training needed. Read-only module: no files, no audio devices.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .config import SonarConfig

_TINY = 1e-30


@dataclass
class PresenceState:
    state: str              # "empty" | "entering" | "present"
    confidence: float       # 0..1
    moving: bool            # gross movement of the person in front of the laptop, right now
    still_s: float          # seconds since the last gross movement in the person zone
    range_m: float | None   # distance of the person, None when empty
    fast_db: float          # gross-movement level above the noise (dB), anywhere in the person zone
    slow_db: float          # micro-movement level above its floor (dB), person zone
    near_db: float          # gross-movement level in the hand zone (< 0.6 m)
    changed: bool           # True on the frame where `state` changed
    motionless_s: float = 0.0   # while present: seconds without ANY movement, breathing included
    far_db: float = 0.0         # gross-movement level beyond the hand zone (> 0.6 m), the hall included
    person_db: float = 0.0      # gross-movement level at the tracked person's distance (what `moving` is decided on)
    noise_db: float = 0.0       # how far the noise right now is above the calibrated floor (0 = normal)


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
        self._a_pq = _alpha(0.1, rate)                         # quick gross-movement power (tells breathing what to skip)
        self._a_base = _alpha(5.0, rate)                       # how fast "the usual noise level" follows a louder room
        self._a_conf = _alpha(1.5, rate)                       # smoothing of the reported confidence
        self._a_range = _alpha(1.0, rate)                      # smoothing of the reported distance
        r = cfg.range_m
        taps = np.arange(cfg.n_taps)
        valid = taps > cfg.direct_tap + 1                      # skip the direct path itself
        self._person = valid & (r >= cfg.presence_min_range_m) & (r <= cfg.presence_max_range_m)
        self._beyond = self._person & (r > cfg.presence_near_m)   # person zone past arm's length: a body, not a hand
        self._near = valid & (r <= cfg.presence_near_m)
        self._far = valid & (r > cfg.presence_near_m)
        self._ref = taps > cfg.direct_tap + 6                  # the noise level is read from every tap past the direct sound
        self._range = r
        self._calib_frames = int(calib_s * rate)
        self._settle = int(cfg.presence_settle_s * rate)
        self._enter = cfg.presence_enter_s * rate
        self._tap_thr = 10.0 ** (cfg.presence_tap_thresh_db / 10.0)
        self._spread = np.ones(5)                              # a moving echo also spoils 2 taps either side (its main lobe)
        self.reset()

    # -- state ---------------------------------------------------------------
    def reset(self) -> None:
        self._hp_zi: np.ndarray | None = None
        self._s1: np.ndarray | None = None
        self._s2: np.ndarray | None = None
        self._pf: np.ndarray | None = None        # smoothed gross-movement power per tap
        self._pq: np.ndarray | None = None        # the same with a 0.1 s memory
        self._ps: np.ndarray | None = None        # smoothed micro-movement power per tap
        self.floor_fast: np.ndarray | None = None
        self.floor_slow: np.ndarray | None = None
        self._q0 = 1.0                            # what the middle tap of one frame reads in plain sensor noise
        self._base = 1.0                          # the usual noise level lately (either meter), relative to calibration
        self._inband = 1.0                        # the usual noise level lately inside our own band (first meter only)
        self._amb0: float | None = None           # usual out-of-band noise power (learned during calibration)
        self._calib_amb: list[float] = []
        self.noise_now = 1.0                      # noise level of the current frame, relative to calibration
        self.frame_weight = 1.0                   # 1 = a normal frame; towards 0 = drowned in a noise burst
        self._calib_pf: list[np.ndarray] = []
        self._calib_ps: list[np.ndarray] = []
        self._calib_inst: list[np.ndarray] = []
        self._n = 0
        self.state = "empty"
        self._walk_run = 0            # frames of gross movement beyond arm's length (a body walking up); cleared after a quiet second
        self._quiet_run = 0
        self._last_motion = -1e9      # frame index of the last movement of any kind
        self._last_fast = -1e9
        self._last_micro = -1e9       # frame index of the last micro movement (breathing / sway)
        self._state_since = 0
        self._zone_on = False         # gross movement somewhere in the person zone (with hysteresis)
        self._moving = False          # gross movement at the tracked person's distance (with hysteresis)
        self._tap_quiet: np.ndarray | None = None   # per tap: frames since gross movement was last seen there
        self._range_smooth: float | None = None
        self._range_hist: deque[float] = deque(maxlen=int(2.0 * self.cfg.frame_rate))
        self._conf = 0.0
        self._slow_hold = -99.0
        self.fast_ratio: np.ndarray | None = None   # per tap: gross-movement power over the noise right now (for breathing)

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
        return float(10.0 * np.log10(max(float(top.mean()), 1e-10)))

    def _quietest(self, ratio: np.ndarray) -> float:
        """Middle level over the taps: the noise of this frame, whatever moves at a few of them."""
        return float(np.percentile(ratio[self._ref], self.cfg.presence_noise_pct))

    def _finish_calibration(self) -> None:
        pf = np.stack(self._calib_pf)
        ps = np.stack(self._calib_ps)
        # Low percentile: movement during calibration can only raise the power, never lower it.
        self.floor_fast = np.percentile(pf, 30, axis=0) + _TINY
        # Micro-movement floor: measured, but kept within 3..9 dB of what pure sensor noise
        # predicts, so that a person standing there during calibration cannot blind the detector.
        thermal_slow = 0.055 * self.floor_fast     # short-average noise (0.04) over high-passed noise (0.72)
        self.floor_slow = np.clip(np.percentile(ps, 50, axis=0), 2.0 * thermal_slow, 8.0 * thermal_slow) + _TINY
        self._q0 = max(float(np.median([self._quietest(p / self.floor_fast) for p in self._calib_inst])), 1e-3)
        self._pq = self._pf.copy()
        self._amb0 = float(np.median(self._calib_amb)) if self._calib_amb else None
        self._calib_pf, self._calib_ps, self._calib_inst, self._calib_amb = [], [], [], []

    def _strongest_range(self, score: np.ndarray, zone: np.ndarray) -> float:
        """Distance of the strongest tap of `score` inside `zone` (neighbours averaged in)."""
        smooth = np.convolve(score, np.ones(3) / 3.0, mode="same")
        return float(self._range[int(np.argmax(np.where(zone, smooth, 0.0)))])

    # -- main entry ----------------------------------------------------------------
    def update(
        self, cir: np.ndarray, vital: bool = False, doppler_db: float = 0.0, vital_range_m: float | None = None,
        ambient: float = 0.0,
    ) -> PresenceState:
        """cir: [n_mics, n_taps] complex echo profile of one frame.

        `vital` = the breathing estimator currently tracks a breathing rate. It is ~10 dB
        more sensitive than the per-tap detector here (it combines taps and 10-30 s of
        data), so it keeps a quiet, still person "present". `vital_range_m` = where it
        sees that chest, which is the best distance estimate there is.
        `doppler_db` (pilot-tone movement energy) is accepted for compatibility and not
        used: the tones are not range-gated, and in a busy room they report every walker.
        `ambient` = noise power of this frame beside our own signal (`FrameOut.ambient`);
        0 if not available.
        """
        cfg = self.cfg
        h = cir.astype(np.complex128)
        self._n += 1
        if self._hp_zi is None:
            self._s1, self._s2 = h.copy(), h.copy()
            self._pf = np.zeros(h.shape[1])
            self._ps = np.zeros(h.shape[1])
            self._tap_quiet = np.zeros(h.shape[1], dtype=np.int64)
            # start the high-pass in steady state for this static picture (no start-up burst)
            self._hp_zi = sps.sosfilt_zi(self._hp_sos)[:, :, None, None] * h[None, None, :, :]

        d_fast, self._hp_zi = sps.sosfilt(self._hp_sos, h[None, :, :], axis=0, zi=self._hp_zi)
        inst = np.mean(np.abs(d_fast[0]) ** 2, axis=0)     # gross-movement power of this frame, per tap

        if self.floor_fast is None:
            self._s1 += self._a_s1 * (h - self._s1)
            self._s2 += self._a_s2 * (h - self._s2)
            self._pf += self._a_pf * (inst - self._pf)
            self._ps += self._a_ps * (np.mean(np.abs(self._s1 - self._s2) ** 2, axis=0) - self._ps)
            if self._n > 0.25 * self._calib_frames:        # let the smoothers settle first
                self._calib_pf.append(self._pf.copy())
                self._calib_ps.append(self._ps.copy())
                self._calib_inst.append(inst)
                if ambient > 0.0:
                    self._calib_amb.append(ambient)
            if self._n >= self._calib_frames:
                self._finish_calibration()
            return PresenceState("empty", 0.0, False, 0.0, None, 0.0, 0.0, 0.0, False)

        assert self.floor_slow is not None and self._tap_quiet is not None and self._pq is not None
        rate = cfg.frame_rate
        on_db, off_db = cfg.presence_fast_thresh_db, cfg.presence_fast_off_db

        # How noisy is THIS frame? Movement sits at a few distances, a noise burst at most of
        # them (a 2 ms click lands on about two thirds), so the middle value over the taps
        # reads the noise whatever else is going on. The frame is divided by that level, and
        # a frame noisier than usual counts for less.
        noise = max(self._quietest(inst / self.floor_fast) / self._q0, 1e-3)
        # The second meter: a short click lands on only some of the distances (which ones depends
        # on its timing), so the middle tap can miss it. Beside our band nothing hides it.
        burst = noise if not (self._amb0 and ambient > 0.0) else max(noise, ambient / self._amb0)
        weight = min(1.0, self._base / burst)
        self._base += self._a_base * (min(burst, 3.0 * self._base) - self._base)
        self._inband += self._a_base * (min(noise, 3.0 * self._inband) - self._inband)
        self.noise_now, self.frame_weight = burst, weight
        level = inst / noise
        self._pf += self._a_pf * weight * (level - self._pf)
        self._pq += self._a_pq * weight * (level - self._pq)
        self._s1 += self._a_s1 * weight * (h - self._s1)
        self._s2 += self._a_s2 * weight * (h - self._s2)
        d_slow = self._s1 - self._s2
        self._ps += self._a_ps * (np.mean(np.abs(d_slow) ** 2, axis=0) - self._ps)

        rf = self._pf / self.floor_fast                    # gross movement per tap, over the noise
        self.fast_ratio = self._pq / self.floor_fast

        # Each tap settles on its own: a second after gross movement stopped THERE, take a fresh
        # "background" picture of that tap, so that only ongoing micro movement (breathing, sway)
        # keeps the slow detector on. Movement at one distance leaves the other distances alone.
        hot = np.convolve(rf > self._tap_thr, self._spread, mode="same") > 0.5
        self._tap_quiet = np.where(hot, 0, self._tap_quiet + 1)
        fresh = self._tap_quiet == self._settle
        if np.any(fresh):
            self._s2[:, fresh] = self._s1[:, fresh]
            self._ps[fresh] = 0.0
        rs = np.where(self._tap_quiet > self._settle + 5, self._ps / (self.floor_slow * self._inband), 0.0)

        fast_db = self._top_mean_db(rf[self._person])
        walk_db = self._top_mean_db(rf[self._beyond])
        near_db = self._top_mean_db(rf[self._near])
        far_db = self._top_mean_db(rf[self._far])
        slow_db = self._top_mean_db(rs[self._person])

        # "The person moved" = gross movement at their distance. Somebody passing behind them does not count.
        gate = self._person
        if self.state == "present" and self._range_smooth is not None:
            gate = gate & (self._range <= self._range_smooth + cfg.presence_gate_margin_m)
        person_db = self._top_mean_db(rf[gate])
        self._zone_on = fast_db > (off_db if self._zone_on else on_db)
        self._moving = person_db > (off_db if self._moving else on_db)
        zone_on = self._zone_on
        slow_on = slow_db > cfg.presence_slow_thresh_db

        prev = self.state
        if zone_on:
            self._walk_run += int(walk_db > on_db)
            self._quiet_run = 0
            self._last_fast = self._n
            self._last_motion = self._n
        else:
            self._quiet_run += 1
            if self._quiet_run == self._settle:
                self._walk_run = 0
        # A tracked breathing rate only counts as proof of life once the zone has been quiet for a
        # moment: while somebody walks off, the estimator may still be reporting its last window.
        micro = slow_on or (vital and self._quiet_run > self._settle + 5)
        if micro:
            self._last_motion = self._n
            self._last_micro = self._n

        if self.state == "empty":
            # A body walking up shows beyond arm's length. A hand reaching in from the side does
            # not make anybody "enter" (and so cannot get its own gesture vetoed).
            if self._walk_run >= self._enter:
                self.state = "entering"
            elif micro and self._n - self._state_since > 3.0 * rate and self._quiet_run > 3.0 * rate:
                self.state = "present"          # somebody was already there, sitting still
        elif self.state == "entering":
            if self._quiet_run >= self._settle:
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

        # Slowly follow the floors while the room is empty and quiet (temperature, fan speed ...).
        if self.state == "empty" and not zone_on and not micro and weight > 0.8:
            self.floor_fast += (self._pf - self.floor_fast) / (20.0 * rate)

        # Where is the person? Best evidence first; a 2 s median ignores a hand that crosses in front.
        rng: float | None = None
        if self.state != "empty":
            cand: float | None = None
            if self._moving if self.state == "present" else zone_on:
                cand = self._strongest_range(rf, gate)
            elif vital_range_m is not None:
                cand = float(vital_range_m)
            elif slow_on:
                cand = self._strongest_range(rs, self._person)
            if cand is not None:
                self._range_hist.append(cand)
            if self._range_hist:
                med = float(np.median(self._range_hist))
                if self._range_smooth is None:
                    self._range_smooth = med
                else:
                    self._range_smooth += self._a_range * (med - self._range_smooth)
            rng = None if self._range_smooth is None else round(self._range_smooth, 2)
        else:
            self._range_smooth = None
            self._range_hist.clear()

        still_s = (self._n - self._last_fast) / rate
        motionless = 0.0
        if self.state == "present":
            motionless = (self._n - self._last_motion) / rate
            # The slow detector restarts after every fidget; hold its last level for a few seconds
            # so the confidence does not dive each time.
            self._slow_hold = max(slow_db, self._slow_hold - 2.0 / rate)
            margin = max(person_db - on_db, self._slow_hold - cfg.presence_slow_thresh_db, 6.0 if vital else -99.0)
            conf = (0.5 + 0.5 * np.tanh(margin / 6.0)) * (1.0 - 0.5 * min(motionless / cfg.presence_absent_s, 1.0))
        elif self.state == "entering":
            self._slow_hold = -99.0
            conf = 0.5 + 0.5 * np.tanh((fast_db - on_db) / 6.0)
        else:
            self._slow_hold = -99.0
            quiet = (self._n - max(self._last_motion, self._state_since)) / rate
            conf = 1.0 - np.exp(-max(quiet, 0.0) / 2.0)
        conf = float(np.clip(conf, 0.0, 1.0))
        self._conf = conf if changed else self._conf + self._a_conf * (conf - self._conf)
        return PresenceState(
            state=self.state,
            confidence=float(self._conf),
            moving=bool(self._moving if self.state == "present" else zone_on),
            still_s=float(still_s),
            range_m=rng,
            fast_db=fast_db,
            slow_db=slow_db,
            near_db=near_db,
            changed=changed,
            motionless_s=float(motionless),
            far_db=far_db,
            person_db=person_db,
            noise_db=float(10.0 * np.log10(burst)),
        )

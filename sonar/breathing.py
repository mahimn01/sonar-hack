"""breathing.py — breathing rate of a still person from the echo phase.

Plain words: when the chest moves 1 mm towards the laptop, the echo from the
chest arrives a tiny bit earlier, which turns the PHASE of that echo by about
43 degrees (the sound wavelength is only ~17 mm). Breathing moves the chest a
few millimetres, so the phase swings back and forth by a large, easily
measured angle. The steps:

  1. average the 50 echo profiles per second down to 10 per second and low-pass
     them to +-1.6 Hz (breathing is slow; this removes most of the noise);
  2. for every distance tap the echo is  static part + moving part. Subtract
     the time average, then look for the distances where something still moves;
  3. the chest shows up in several neighbouring taps and on both microphones,
     all moving together. A singular value decomposition (SVD) of that small
     group adds them up with the best possible weights -> ONE clean complex
     signal for the chest (this buys ~7-10 dB over any single tap);
  4. over 10-30 s that signal draws an arc of a circle in the complex plane.
     Fit the circle, subtract its centre, and read the angle (phase):
         displacement d = phase * wavelength / (4 pi)
  5. band-pass 0.08-0.8 Hz and find the strongest spectral peak in
     0.1-0.6 Hz (6-36 breaths/min), using the 2nd harmonic as supporting evidence;
  6. report rate, a 0..1 confidence and the waveform.

What happens when the person is not perfectly still (nobody is):

  * A brief disturbance at the chest's distance (a hand moves, a head turns, a
    burst of noise) is SKIPPED, not punished: those tenths of a second are left
    out, the clean stretches on either side are joined, and the measurement
    carries on. Each clean stretch is filtered on its own and gets its own
    "static part", because moving a hand changes the still echo that sits
    under the chest echo.
  * A disturbance of `breath_moving_s` (1.5 s) or more is reported as state
    "moving" while it lasts. The history is kept, but afterwards a rate is only
    shown again once the chest has been seen moving for 2 s (a couple of
    seconds, instead of the 10+ s a fresh start needs).
  * Only a disturbance longer than `breath_gap_s` (3 s: the person really
    changed position) makes the estimator start again from the data after it.
  * Movement at OTHER distances (people walking past further back) does not
    touch the chest's distance and is ignored altogether.
  * If, 4 s after a disturbance, nothing moves at the chest's distance any
    more, the person has left or stopped breathing: no rate is reported.
  * While a brief disturbance passes, the last rate is held for up to
    `breath_hold_s` (3 s) and marked `held`.

Read-only module: no files, no audio devices.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import signal as sps

from .config import SPEED_OF_SOUND, SonarConfig

_TINY = 1e-30


@dataclass
class BreathingResult:
    bpm: float | None            # breaths per minute, None if not confident
    confidence: float            # 0..1
    state: str                   # "tracking" | "acquiring" | "moving" | "no_signal"
    waveform: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))   # chest displacement, -1..1
    fs: float = 10.0             # sample rate of `waveform`
    range_m: float | None = None # distance of the chest tap
    amplitude_mm: float = 0.0    # peak-to-peak chest movement
    window_s: float = 0.0        # seconds of data behind this estimate
    raw_bpm: float | None = None # this window's own estimate before smoothing/gating
    snr_db: float = 0.0          # moving-echo power over sensor noise at the chosen taps
    quality: float = 0.0         # share of the breathing-band power inside the spectral peak (0..1)
    held: bool = False           # True: this repeats the last good rate while a brief disturbance passes
    fresh: bool = True           # the chest was seen moving AFTER the latest disturbance (proof of life right now)
    stopped_s: float = 0.0       # a tracked breathing rhythm stopped this long ago and the chest has been still since
    clean_share: float = 1.0     # share of the analysed window that was undisturbed


def circle_fit(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Taubin circle fit of every column of z ([L, K] complex). Returns centres [K] and radii [K].

    The Taubin fit is nearly unbiased for short arcs (the simple algebraic fit
    shrinks them), which matters for shallow breathing. Columns that cannot be
    fitted (collinear points) fall back to centre = mean.
    """
    scale = np.sqrt(np.mean(np.abs(z) ** 2, axis=0)) + _TINY
    zn = z / scale
    x, y = zn.real, zn.imag
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    u, v = x - xm, y - ym
    w = u * u + v * v
    mxx, myy, mxy = (u * u).mean(axis=0), (v * v).mean(axis=0), (u * v).mean(axis=0)
    mxz, myz, mzz = (u * w).mean(axis=0), (v * w).mean(axis=0), (w * w).mean(axis=0)
    mz = mxx + myy
    cov = mxx * myy - mxy * mxy
    var = mzz - mz * mz
    a3 = 4.0 * mz
    a2 = -3.0 * mz * mz - mzz
    a1 = var * mz + 4.0 * cov * mz - mxz * mxz - myz * myz
    a0 = mxz * (mxz * myy - myz * mxy) + myz * (myz * mxx - mxz * mxy) - var * cov
    xn = np.zeros_like(mz)
    yn = a0.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(25):                                   # Newton steps on the characteristic polynomial
            dy = a1 + xn * (2.0 * a2 + 3.0 * a3 * xn)
            xnew = xn - yn / dy
            ynew = a0 + xnew * (a1 + xnew * (a2 + xnew * a3))
            ok = np.isfinite(xnew) & (np.abs(ynew) < np.abs(yn))
            xn = np.where(ok, xnew, xn)
            yn = np.where(ok, ynew, yn)
        det = xn * xn - xn * mz + cov
        xc = (mxz * (myy - xn) - myz * mxy) / det / 2.0
        yc = (myz * (mxx - xn) - mxz * mxy) / det / 2.0
        r = np.sqrt(xc * xc + yc * yc + mz)
    bad = ~(np.isfinite(xc) & np.isfinite(yc) & np.isfinite(r))
    xc, yc = np.where(bad, 0.0, xc), np.where(bad, 0.0, yc)
    r = np.where(bad, np.sqrt(mz), r)
    return ((xc + xm) + 1j * (yc + ym)) * scale, r * scale


def estimate_rate(
    x: np.ndarray, fs: float, band_hz: tuple[float, float], n_fft: int = 4096, tilt: float = 1.0
) -> tuple[float | None, float]:
    """Dominant rate (Hz) of a displacement waveform inside `band_hz`, and how peaky it is (0..1).

    * The power spectrum is multiplied by f^tilt. Slow body sway and measurement
      drift pile up at the low end (roughly 1/f^2); tilting flattens that so the
      breathing line is judged against a level background.
    * Only true local maxima count: a spectrum that is merely still rising at the
      lower band edge is drift, not breathing.
    * Score(f) = P(f) + 0.4 * P(2f): a real breathing waveform has a 2nd harmonic,
      which protects against reporting double the rate when the waveform is distorted.
    * Quality = share of the 0.05-2 Hz (tilted) power inside the peak and its harmonic.
    """
    n = len(x)
    if n < 16:
        return None, 0.0
    win = np.hanning(n)
    df = fs / n_fft
    freqs = np.arange(n_fft // 2 + 1) * df
    p = np.abs(np.fft.rfft((x - x.mean()) * win, n_fft)) ** 2 * np.maximum(freqs, df) ** tilt
    lo, hi = int(np.ceil(band_hz[0] / df)), int(np.floor(band_hz[1] / df))
    idx = np.arange(lo, hi + 1)
    is_peak = (p[idx] >= p[idx - 1]) & (p[idx] >= p[idx + 1])
    score = np.where(is_peak, p[idx] + 0.4 * p[np.minimum(2 * idx, len(p) - 1)], 0.0)
    if not np.any(is_peak):
        return None, 0.0
    score = np.where(p[idx] >= 0.3 * p[idx][is_peak].max(), score, 0.0)   # the fundamental itself must be present
    j = int(np.argmax(score))
    if score[j] <= 0:
        return None, 0.0
    k1 = int(idx[j])
    a, b, c = p[k1 - 1], p[k1], p[k1 + 1]                                # parabolic refinement of the peak
    den = a - 2 * b + c
    i0 = k1 + (float(np.clip(0.5 * (a - c) / den, -1.0, 1.0)) if den < 0 else 0.0)
    f0 = i0 * df
    half = int(np.ceil(1.6 * fs / n / df))                                # half-width of the Hann main lobe
    k2 = int(round(2 * i0))
    peak = p[max(k1 - half, 0):k1 + half + 1].sum() + p[max(k2 - half, 0):k2 + half + 1].sum()
    total = p[int(0.05 / df):int(2.0 / df)].sum() + _TINY
    return f0, float(min(peak / total, 1.0))


class BreathingEstimator:
    def __init__(self, cfg: SonarConfig) -> None:
        self.cfg = cfg
        r = cfg.range_m
        taps = np.arange(cfg.n_taps)
        self._sel = np.where((r >= cfg.breath_range_m[0]) & (r <= cfg.breath_range_m[1]) & (taps > cfg.direct_tap + 1))[0]
        self._group = max(1, int(round(cfg.frame_rate / cfg.breath_fs)))
        self._fs = cfg.frame_rate / self._group
        self._max_len = int(cfg.breath_max_s * self._fs)
        self._min_len = int(cfg.breath_min_s * self._fs)
        self._every = max(1, int(round(cfg.breath_update_s * self._fs)))
        self._gap_len = max(1, int(round(cfg.breath_gap_s * self._fs)))         # longer disturbance = start again
        self._moving_len = max(1, int(round(cfg.breath_moving_s * self._fs)))   # disturbed this long = "moving"
        self._hold_len = int(round(cfg.breath_hold_s * self._fs))
        self._alive_len = int(round(2.0 * self._fs))     # this much moving chest after a disturbance = still breathing
        self._still_len = int(round(4.0 * self._fs))     # this much motionless chest after a disturbance = gone / not breathing
        self._min_seg = int(round(1.5 * self._fs))                              # shorter clean stretches are not worth joining
        self._dist_thr = 10.0 ** (cfg.breath_disturb_db / 10.0)
        self._wavelength = SPEED_OF_SOUND / cfg.chirp_center_hz
        self._sos = sps.butter(2, [0.08, 0.8], "bandpass", fs=self._fs, output="sos")
        # Neighbouring distance taps are NOT independent looks: the echo profile is computed
        # with ~3 taps per resolution cell, so sensor noise at one tap is almost the same as at
        # the next (correlation 0.94). `_tap_corr[d]` is that correlation at a spacing of d taps.
        k = len(cfg.chirp_bins)
        w2 = sps.get_window(cfg.cir_window, k, fftbins=False) ** 2
        self._tap_corr = np.array([np.sum(w2 * np.exp(2j * np.pi * np.arange(k) * d / cfg.cir_fft)) for d in range(8)]) / w2.sum()
        self._whiteners: dict[int, np.ndarray] = {}
        # Zero-phase low-pass of the complex taps before anything non-linear happens. Two widths
        # are tried: narrow (least noise; quiet breathing) and wide (deep or fast breathing swings
        # the phase quickly and needs the bandwidth). The better-looking result wins.
        self._lp = []
        for hz in (1.3, 3.0):
            sos = sps.butter(2, hz, "lowpass", fs=self._fs, output="sos")
            noise_gain = 2.0 * 0.87 * hz / self._fs      # share of white noise that survives (two-sided band / fs)
            self._lp.append((sos, noise_gain))
        self.reset()

    def reset(self) -> None:
        self._buf: deque[np.ndarray] = deque(maxlen=self._max_len)
        self._dbuf: deque[np.ndarray] = deque(maxlen=self._max_len)    # per slot and tap: strongest gross movement seen
        self._acc: np.ndarray | None = None
        self._dacc: np.ndarray | None = None
        self._acc_n = 0
        self._acc_w = 0.0
        self._prev: np.ndarray | None = None
        self._noise: np.ndarray | None = None       # sensor-noise variance per (mic, tap) at the frame rate
        self._count = 0
        self._stopped = False                       # set by the estimate: the chest is there but has stopped moving
        self._bpm_hist: deque[float] = deque(maxlen=3)
        self._last_tap: int | None = None           # where the chest was last time (preferred next time)
        self._good: tuple[int, BreathingResult] | None = None    # (slot counter, result) of the last reported rate
        self._no_hold = False                       # the last estimate must not be papered over with the held rate
        self._stopped_at: int | None = None         # slot counter when a tracked rhythm was first seen to have stopped
        self.last = BreathingResult(None, 0.0, "acquiring", fs=self._fs)

    def restart(self, state: str = "moving") -> None:
        """Throw away the history (audio was lost, or the scene changed completely) but keep the noise estimate."""
        self._buf.clear()
        self._dbuf.clear()
        self._acc, self._dacc, self._acc_n, self._acc_w = None, None, 0, 0.0
        self._bpm_hist.clear()
        self._count = 0
        self._last_tap = None
        self._good = None
        self._stopped_at = None
        self.last = BreathingResult(None, 0.0, state, fs=self._fs)

    # -- main entry ----------------------------------------------------------------
    def update(
        self, cir: np.ndarray, moving: bool = False, disturb: np.ndarray | None = None, weight: float = 1.0
    ) -> BreathingResult | None:
        """Feed one echo profile [n_mics, n_taps]. Returns a result about once per second, else None.

        `disturb` [n_taps] = gross-movement power over the noise at every distance for this
        frame (`PresenceDetector.fast_ratio`): the estimator skips the moments and distances
        where it is high. `weight` (0..1, `PresenceDetector.frame_weight`) says how much this
        frame is worth: a frame hit by a burst of noise counts for little or nothing.
        `moving` = the person in front of the laptop is moving as a whole; without
        `disturb`, a true `moving` marks every distance as disturbed.
        """
        h = cir[:, self._sel].astype(np.complex128)
        if disturb is not None:
            d = np.asarray(disturb, dtype=np.float64)[self._sel]
        else:
            d = np.full(len(self._sel), np.inf if moving else 0.0)
        # Sensor noise from frame-to-frame differences (movement is far slower than 50 Hz);
        # noisy frames and distances where something is moving right now are left out.
        if self._prev is not None and weight > 0.5:
            d2 = 0.5 * np.abs(h - self._prev) ** 2
            if self._noise is None:
                self._noise = d2 + _TINY
            else:
                calm = (d <= self._dist_thr)[None, :]
                self._noise += 0.01 * calm * (np.minimum(d2, 9.0 * self._noise) - self._noise)
        self._prev = h

        self._acc = weight * h if self._acc is None else self._acc + weight * h
        self._dacc = d if self._dacc is None else np.maximum(self._dacc, d)
        self._acc_n += 1
        self._acc_w += weight
        if self._acc_n < self._group:
            return None
        if self._acc_w > 0.1 * self._group:                # (a slot at this limit is 10x as noisy as a normal one)
            self._buf.append(self._acc / self._acc_w)
            self._dbuf.append(self._dacc)
        else:                                               # the whole slot was noise: repeat the last one, marked unusable
            self._buf.append(self._buf[-1] if self._buf else self._acc * 0.0)
            self._dbuf.append(np.full(len(self._sel), np.inf))
        self._acc, self._dacc, self._acc_n, self._acc_w = None, None, 0, 0.0
        self._count += 1
        if self._count % self._every:
            return None
        self._stopped = False
        self.last = self._finish(self._estimate(), self._stopped)
        return self.last

    def _finish(self, r: BreathingResult, stopped: bool = False) -> BreathingResult:
        """Hold the last good rate for a moment when a brief disturbance interrupts the read-out,
        and keep count of how long a tracked rhythm has been stopped (`stopped` = it is, right now)."""
        if r.bpm is not None:
            self._good = (self._count, r)
            self._stopped_at = None
            return r
        if stopped and self._good is not None and (self._stopped_at is not None or self._count - self._good[0] <= 12 * self._fs):
            if self._stopped_at is None:                    # it was breathing a moment ago, and now the chest is still
                self._stopped_at = self._count
            return replace(r, stopped_s=(self._count - self._stopped_at) / self._fs)
        self._stopped_at = None
        if not self._no_hold and self._good is not None and self._count - self._good[0] <= self._hold_len:
            good = self._good[1]
            return replace(
                good, held=True, fresh=False, confidence=0.8 * good.confidence, window_s=r.window_s,
                waveform=r.waveform if len(r.waveform) else good.waveform,
            )
        return r

    # -- disturbed moments -------------------------------------------------------------
    def _disturbed(self, level: np.ndarray) -> np.ndarray:
        """[L, taps] gross-movement level -> [L, taps] bool "do not use this sample".

        A moving echo also spoils its two neighbours in range (its main lobe) and the slot
        before and after (the detector's own smoothing).
        """
        hot = level > self._dist_thr
        wide = hot.copy()
        for k in (1, 2):
            wide[:, k:] |= hot[:, :-k]
            wide[:, :-k] |= hot[:, k:]
        out = wide.copy()
        out[:-1] |= wide[1:]
        out[1:] |= wide[:-1]
        return out

    @staticmethod
    def _runs(flags: np.ndarray) -> np.ndarray:
        """[k, 2] (start, end-exclusive) of every run of True in a 1-D bool array."""
        edges = np.flatnonzero(np.diff(np.concatenate([[False], flags, [False]]).astype(np.int8)))
        return edges.reshape(-1, 2)

    def _tail(self, bad: np.ndarray) -> int:
        """Length of the run of disturbed slots that reaches up to now (0 if the newest slot is clean)."""
        runs = self._runs(bad)
        return int(runs[-1, 1] - runs[-1, 0]) if len(runs) and runs[-1, 1] == len(bad) else 0

    def _clean_stretches(self, bad: np.ndarray) -> tuple[list[tuple[int, int]], int | None, int | None]:
        """Usable stretches of one chest candidate.

        `bad` [L] marks its disturbed slots. Returns (clean stretches, where the latest
        disturbance ended, where the latest disturbance of "moving" length ended); the last
        two are None when there was none. Everything before the last disturbance longer
        than `breath_gap_s` is dropped (the person really moved: older data describes a
        different posture). Clean stretches shorter than 1.5 s are not used.
        """
        runs = self._runs(bad)
        length = runs[:, 1] - runs[:, 0] if len(runs) else np.zeros(0, dtype=np.int64)
        start = int(runs[length > self._gap_len][-1, 1]) if np.any(length > self._gap_len) else 0
        last_end = int(runs[-1, 1]) if len(runs) else None
        big = runs[(length >= self._moving_len) & (runs[:, 1] > start)] if len(runs) else runs
        big_end = int(big[-1, 1]) if len(big) else None
        good = ~bad
        good[:start] = False
        return [(int(a), int(e)) for a, e in self._runs(good) if e - a >= self._min_seg], last_end, big_end

    # -- the estimate ----------------------------------------------------------------
    def _whitener(self, n_taps: int) -> np.ndarray:
        """[n_taps, 2] matrix that turns a group of neighbouring taps into 2 independent unit-noise looks.

        The noise of `n_taps` neighbours lives almost entirely in 2 directions (their
        eigenvalues are ~4.0 and ~0.9 of 5; the rest is ~0.1). Projecting onto those two and
        scaling each to unit noise loses no echo (an echo has the same shape across taps as
        the noise correlation) and makes "how much is noise" a plain count.
        """
        if n_taps not in self._whiteners:
            i = np.arange(n_taps)
            d = i[:, None] - i[None, :]
            corr = np.where(d >= 0, self._tap_corr[np.abs(d)], np.conj(self._tap_corr[np.abs(d)]))
            lam, vec = np.linalg.eigh(corr)                                  # ascending
            keep = min(2, n_taps)
            self._whiteners[n_taps] = np.conj(vec[:, -keep:]) / np.sqrt(lam[-keep:])[None, :]
        return self._whiteners[n_taps]

    def _combine(self, m: np.ndarray, n_mics: int, band_share: float = 1.0) -> tuple[np.ndarray, float, np.ndarray]:
        """Best-weights sum of a group of taps. Returns (combined complex signal [L], its SNR,
        the weights [n_mics * taps] that produce it: signal = m @ weights).

        `m` [L, n_mics * taps]: the group's taps (mic by mic) with their still part removed
        and scaled by the estimated sensor noise. `band_share` = share of the noise the
        low-pass let through (it sets how many independent samples the L rows are worth).
        All taps of the group see the same chest, so the group is (nearly) one time function
        times one weight per tap; the first SVD component is that time function with
        maximum SNR.

        How much of it is noise is judged twice, and the larger answer wins:
          * the sensor-noise estimate the columns were scaled with (1.0 if it is right), and
          * the group itself: after whitening, the weakest half of the components is what
            the noise really was in this window. That keeps a room that got noisier than the
            estimate knows (bursts, a louder hall) from looking like a moving echo.
        The returned signal has unit noise variance; the SNR is "echo power / noise power",
        0 for plain noise whatever its level.
        """
        per_mic = m.shape[1] // n_mics
        wh = self._whitener(per_mic)
        mw = np.concatenate([m[:, k * per_mic:(k + 1) * per_mic] @ wh for k in range(n_mics)], axis=1)
        u, sv, vh = np.linalg.svd(mw, full_matrices=False)
        n, dims = mw.shape
        ev = sv**2 / n
        # What plain noise would give (from a Monte-Carlo of complex noise matrices): the largest
        # component reads 1 + 1.25*c, the weakest half 1 - 0.95*c, with c = sqrt(dims / samples).
        # The 15 % slack is for an echo that is not perfectly one time function (a chest and a
        # belly move differently) and so leaves a little of itself in the weakest components.
        c = float(np.sqrt(dims / max(n * band_share, dims + 2.0)))
        level = max(1.0, float(np.mean(ev[(dims + 1) // 2:])) / (1.15 * max(1.0 - 0.95 * c, 0.2))) if dims > 1 else 1.0
        g = u[:, 0] * sv[0] / np.sqrt(level)
        snr = max(float(ev[0]) / level - (1.0 + 1.25 * c), 0.0)
        look = np.conj(vh[0]) / np.sqrt(level)                      # weight of each whitened look
        weights = np.concatenate([wh @ look[k * wh.shape[1]:(k + 1) * wh.shape[1]] for k in range(n_mics)])
        return g, snr, weights

    # Tunables of the phase read-out (class attributes so experiments can override them).
    arc_ratio_thr = 0.10      # "how round is the point cloud" above which the circle fit is trusted
    min_snr = 1.2             # a chest candidate needs this much echo power over the noise (after combining);
                              # plain noise reads 0.2 +- 0.3 here (1.2 at the very most), the weakest echoes worth reporting ~1.6
    rate_tilt = 1.0           # spectral tilt used by estimate_rate

    def _same_echo_groups(self, g: np.ndarray, bounds: np.ndarray, gaps: list[int]) -> list[list[int]]:
        """Which clean stretches still share the same still echo? Returns groups of stretch numbers.

        Across a gap the signal either carries on where breathing would have taken it (the
        gap was a burst of noise, or something moved that does not show at this distance),
        or it jumps (a hand now lies somewhere else: the still echo under the chest echo
        changed). "Carries on" = the jump across the gap is no more than 1.5x what the
        fastest tenth of the breathing movement covers in that time, plus a noise allowance.
        """
        n_seg = len(bounds) - 1
        steps = np.concatenate([np.abs(np.diff(g[bounds[i]:bounds[i + 1]])) for i in range(n_seg)])
        fast = float(np.percentile(steps, 90)) if len(steps) else 0.0
        groups = [[0]]
        for i in range(1, n_seg):
            before = g[max(bounds[i] - 2, bounds[i - 1]):bounds[i]].mean()
            after = g[bounds[i]:min(bounds[i] + 2, bounds[i + 1])].mean()
            if abs(after - before) > 1.5 * (gaps[i - 1] + 2) * fast + 2.0:
                groups.append([i])
            else:
                groups[-1].append(i)
        return groups

    def _phase_to_disp(
        self, g: np.ndarray, lens: list[int] | None = None, gaps: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Complex arc -> displacement in metres (positive = towards the laptop).

        Returns (displacement [L], which samples it is valid for [L] bool, method used).
        `g` has unit noise variance (0.5 per real axis) and consists of `lens` clean
        stretches, one after the other, with `gaps[i]` skipped slots between stretch i and
        i+1. Two regimes:
          * long arc (deep breathing, more than ~150 degrees): fit the circle, read the angle;
          * short arc (shallow breathing or weak echo): a circle fit is unreliable there
            (a noisy short arc fits many circles), but a short arc is almost a straight
            line, so project onto that line instead. Rate and shape are right; the
            millimetre scale is then only approximate.
        With several stretches, those that still share the same still echo (see
        `_same_echo_groups`) are read as one piece with a gap in it; where the still echo
        changed, each side gets its own circle centre (or its own mean) and the angles are
        joined to the nearest full turn.
        """
        lens = [len(g)] if lens is None else lens
        k = self._wavelength / (4.0 * np.pi)
        keep = np.ones(len(g), dtype=bool)
        bounds = np.cumsum([0] + list(lens))
        if len(lens) == 1:
            groups = [[0]]
        else:
            groups = self._same_echo_groups(g, bounds, gaps if gaps is not None else [1] * (len(lens) - 1))
        member = [np.concatenate([np.arange(bounds[i], bounds[i + 1]) for i in grp]) for grp in groups]
        gc = g.copy()
        for idx in member:                                          # remove each group's own still part
            gc[idx] -= gc[idx].mean()
        cov = np.cov(np.stack([gc.real, gc.imag]))
        evals, evecs = np.linalg.eigh(cov)                          # ascending
        roundness = max(evals[0] - 0.5, 0.0) / max(evals[1] - 0.5, 1e-9)
        biggest = max(member, key=len)
        centre, radius = circle_fit(g[biggest][:, None])
        if roundness >= self.arc_ratio_thr:
            # A circle needs a fair stretch of arc: groups with under 4 s of data are left out.
            fit = [j for j, idx in enumerate(member) if len(idx) >= int(4.0 * self._fs)] if len(lens) > 1 else [0]
            if fit:
                disp = np.zeros(len(g))
                keep[:] = False
                prev_end: float | None = None
                for j in fit:
                    c = centre if member[j] is biggest else circle_fit(g[member[j]][:, None])[0]
                    for i in groups[j]:
                        th = np.unwrap(np.angle(g[bounds[i]:bounds[i + 1]] - c[0]))
                        if prev_end is not None:                     # same chest: continue on the nearest full turn
                            th += 2.0 * np.pi * np.round((prev_end - th[0]) / (2.0 * np.pi))
                        prev_end = float(th[-1])
                        disp[bounds[i]:bounds[i + 1]] = th * k
                        keep[bounds[i]:bounds[i + 1]] = True
                return disp, keep, "arc"
        axis = evecs[0, 1] + 1j * evecs[1, 1]                       # direction of the line
        proj = (gc * np.conj(axis)).real
        # Sign: moving towards the laptop turns the echo counter-clockwise around the centre.
        radial = g[biggest].mean() - centre[0]
        ccw = 1j * radial / (abs(radial) + _TINY)
        if (ccw * np.conj(axis)).real < 0:
            proj = -proj
        r = max(float(radius[0]), 1.2 * float(np.max(np.abs(proj))))
        return np.arcsin(np.clip(proj / r, -1.0, 1.0)) * k, keep, "line"

    def _swing(self, x: np.ndarray, f0: float) -> float:
        """Amplitude of the oscillation at rate `f0` (Hz) in a stretch of the displacement waveform."""
        w = np.hanning(len(x) + 2)[1:-1]
        t = np.arange(len(x)) / self._fs
        return float(2.0 * np.abs(np.sum(w * (x - np.mean(x)) * np.exp(-2j * np.pi * f0 * t))) / np.sum(w))

    def _estimate(self) -> BreathingResult:
        cfg = self.cfg
        n = len(self._buf)
        window_s = n / self._fs
        self._no_hold = False
        if n < int(4.0 * self._fs) or self._noise is None:
            return BreathingResult(None, 0.0, "acquiring", fs=self._fs, window_s=window_s)

        raw = np.stack(self._buf)                                   # [L, mics, taps]
        n_mics, n_taps = raw.shape[1:]
        bad = self._disturbed(np.stack(self._dbuf))                 # [L, taps]
        # Is the chest's own distance busy right now (before a chest is known: most of the range)?
        if self._last_tap is not None:
            busy = bad[:, max(self._last_tap - 2, 0):self._last_tap + 3].any(axis=1)
        else:
            busy = bad.mean(axis=1) > 0.5
        if self._tail(busy) >= self._moving_len:
            self._no_hold = True                                    # the person is moving: say so, do not show a rate
            return BreathingResult(None, 0.0, "moving", fs=self._fs, window_s=window_s, fresh=False)

        zz = raw.reshape(n, -1)                                     # [L, mics*taps]
        # per tap: clean stretches (only where something was disturbed; elsewhere the whole window is clean)
        dirty = np.flatnonzero(bad.any(axis=0))
        stretches = {int(t): self._clean_stretches(bad[:, t])[0] for t in dirty}
        clean_cols = np.flatnonzero(~np.tile(bad.any(axis=0), n_mics))
        # taps that were disturbed at the same moments are filtered together
        same: dict[tuple[tuple[int, int], ...], list[int]] = {}
        for t, segs in stretches.items():
            same.setdefault(tuple(segs), []).append(t)

        # (rank, f0, quality, disp, tap, snr, span_s, clean share, fresh, waiting for proof of life)
        best: tuple[float, float | None, float, np.ndarray, int, float, float, float, bool, bool] | None = None
        gone = False            # the chest we were following was dropped because nothing moves there any more
        short = False           # a candidate was dropped only because it has too little clean data yet
        for lp_sos, noise_gain in self._lp:
            # Low-pass every clean stretch on its own: nothing of a disturbed moment (and nothing
            # of the jump in the still echo it may have caused) can leak into the samples we use.
            flat = zz.copy()
            flat[:, clean_cols] = sps.sosfiltfilt(lp_sos, zz[:, clean_cols], axis=0)
            for segs, group in same.items():
                cols_g = np.concatenate([m * n_taps + np.asarray(group) for m in range(n_mics)])
                for a, e in segs:
                    flat[a:e, cols_g] = sps.sosfiltfilt(lp_sos, zz[a:e][:, cols_g], axis=0)
            noise = (self._noise / self._group).reshape(-1) * noise_gain + _TINY

            # Where does something move? Excess variance over the sensor noise, per distance tap
            # (within each clean stretch, so a changed still echo does not count as movement).
            zc = flat - flat.mean(axis=0, keepdims=True)
            power = np.mean(np.abs(zc) ** 2, axis=0).reshape(n_mics, n_taps)
            for t, segs in stretches.items():
                total = sum(e - a for a, e in segs)
                if total < int(4.0 * self._fs):
                    power[:, t] = 0.0                               # too little clean data at this distance
                    short = True
                    continue
                cols_t = [m * n_taps + t for m in range(n_mics)]
                acc = np.zeros(n_mics)
                for a, e in segs:
                    seg = flat[a:e][:, cols_t]
                    acc += np.sum(np.abs(seg - seg.mean(axis=0, keepdims=True)) ** 2, axis=0)
                power[:, t] = acc / total
            excess = (power / noise.reshape(n_mics, n_taps) - 1.0).sum(axis=0)
            smooth = np.convolve(excess, np.ones(3) / 3.0, mode="same")
            peaks: list[int] = []
            for t in np.argsort(smooth)[::-1]:
                if smooth[t] < 0.3 * n_mics or len(peaks) >= 3:
                    break
                if all(abs(int(t) - q) > 4 for q in peaks):
                    peaks.append(int(t))

            for t in peaks:
                ours = self._last_tap is not None and abs(t - self._last_tap) <= 3     # the chest we were following
                taps = np.arange(max(t - 2, 0), min(t + 3, n_taps))
                cols = np.concatenate([m * n_taps + taps for m in range(n_mics)])
                segs, last_end, big_end = self._clean_stretches(bad[:, taps].any(axis=1))
                idx = np.concatenate([np.arange(a, e) for a, e in segs]) if segs else np.zeros(0, dtype=np.int64)
                if len(idx) < int(4.0 * self._fs):
                    short = True
                    continue
                scale = np.sqrt(noise[cols])[None, :]               # every column gets unit noise
                parts = [flat[a:e][:, cols] / scale for a, e in segs]
                still = [part.mean(axis=0, keepdims=True) for part in parts]     # each stretch's own still echo
                g, snr, weights = self._combine(np.concatenate([p - m0 for p, m0 in zip(parts, still)]), n_mics, noise_gain)
                if snr < self.min_snr:                              # no moving echo that stands clear of the noise
                    continue

                def moves(part: np.ndarray) -> bool:
                    return float(np.var(part)) - 1.0 >= 0.25 * snr

                # Is it still moving NOW? (breath held, or the person left: the old part of the
                # window would otherwise keep "seeing" breathing for another 30 s)
                fresh, waiting = True, False
                if last_end is None or last_end <= n - int(6.0 * self._fs):
                    if not moves(g[idx >= n - int(6.0 * self._fs)]):
                        gone |= ours
                        continue
                else:
                    # Something disturbed this distance in the last 6 s. Is the chest moving since?
                    after = g[idx >= last_end]
                    fresh = len(after) >= self._alive_len and moves(after)
                    if not fresh and len(after) >= self._still_len:
                        gone |= ours                                # 4 s of nothing after it: gone, or not breathing
                        continue
                    if not fresh and not moves(g[idx >= n - int(6.0 * self._fs)]):
                        continue
                if big_end is not None:
                    # After a disturbance of "moving" length a rate needs fresh proof of life first.
                    since = g[idx >= big_end]
                    if len(since) >= self._still_len and not moves(since):
                        gone |= ours
                        continue
                    waiting = not (len(since) >= self._alive_len and moves(since))

                # The read-out works on the echo WITH its still part (the circle the chest echo
                # draws has its centre there); the stretches are compared to see where it changed.
                lens = [e - a for a, e in segs]
                whole = g + np.concatenate([np.full(ln, (m0 @ weights)[0]) for ln, m0 in zip(lens, still)])
                d_valid, keep, _ = self._phase_to_disp(whole, lens, [b0 - e0 for (_, e0), (b0, _) in zip(segs[:-1], segs[1:])])
                used = idx[keep]
                # A hole longer than `breath_gap_s` (the read-out had to leave a stretch out) cannot
                # be drawn through with a straight line without inventing a slower rhythm: only
                # what comes after the last such hole is used.
                holes = np.flatnonzero(np.diff(used) > self._gap_len + 2)
                if len(holes):
                    used, d_valid, keep = used[holes[-1] + 1:], d_valid[keep][holes[-1] + 1:], slice(None)
                if len(used) < int(4.0 * self._fs):
                    short = True
                    continue
                first = int(used[0])
                disp = np.interp(np.arange(first, n), used, d_valid[keep])           # straight lines across the gaps
                disp = sps.detrend(disp)
                if len(disp) >= 40:
                    disp = sps.sosfiltfilt(self._sos, disp)
                f0, quality = estimate_rate(disp, self._fs, cfg.breath_band_hz, tilt=self.rate_tilt)
                # Prefer staying on the chest we already follow over hopping to a look-alike.
                stay = 1.25 if ours else 1.0
                rank = quality * snr**0.25 * stay
                if best is None or rank > best[0]:
                    best = (rank, f0, quality, disp, t, snr, len(disp) / self._fs, len(used) / len(disp), fresh, waiting)
        if best is None:
            self._no_hold = self._stopped = gone                    # confirmed stillness is not a "brief disturbance"
            state = "acquiring" if short and not gone else "no_signal"     # just started over: still collecting
            return BreathingResult(None, 0.0, state, fs=self._fs, window_s=window_s, fresh=False)
        result = self._report(*best[1:])
        # Follow a candidate once it yields a rate (or while no rate has been had for 10 s); a
        # mere look-alike that shows up while the chest is still must not take its place.
        if result.bpm is not None or self._good is None or self._count - self._good[0] > 10 * self._fs:
            self._last_tap = best[4]
        if result.bpm is None and gone:
            # The chest we were following has stopped; whatever else moves a little is not it.
            self._no_hold = self._stopped = True
            result = replace(result, state="no_signal", fresh=False)
        return result

    def _report(
        self, f0: float | None, quality: float, disp: np.ndarray, tap: int, snr: float, span_s: float, clean: float,
        fresh: bool, waiting: bool,
    ) -> BreathingResult:
        """Turn the best candidate's displacement waveform into the reported result."""
        cfg = self.cfg
        pp_mm = 1000.0 * float(np.percentile(disp, 97) - np.percentile(disp, 3))
        peak = float(np.max(np.abs(disp))) + _TINY
        wave = (disp / peak).astype(np.float32)
        snr_db = 10.0 * np.log10(snr + _TINY)
        rng = float(cfg.range_m[self._sel[tap]])
        if f0 is None or len(disp) < self._min_len or waiting:
            self._no_hold = waiting
            return BreathingResult(
                None, 0.0, "acquiring", wave, self._fs, rng, pp_mm, span_s, None, snr_db, fresh=fresh, clean_share=clean
            )

        raw_bpm = 60.0 * f0
        # Is the rhythm still going? The echo can keep moving a little (sway, pulse) after the
        # breathing has stopped, so "something moves" is not enough: the swing at the breathing
        # rate over the last breath and a half must be at least 40 % of its size over the window.
        #   rhythm gone and the chest nearly still (under 30 % of its usual movement) -> stopped
        #   rhythm gone but the chest still moving (talking, a sigh, a cough)       -> irregular
        n_recent = int(np.clip(1.5 / f0, 6.0, 10.0) * self._fs)
        if len(disp) >= 2 * n_recent and self._swing(disp[-n_recent:], f0) < 0.4 * self._swing(disp, f0):
            self._stopped = float(np.std(disp[-n_recent:])) < 0.3 * float(np.std(disp))
            self._no_hold = self._stopped
            return BreathingResult(
                None, 0.0, "no_signal" if self._stopped else "acquiring", wave, self._fs, rng, pp_mm, span_s, raw_bpm, snr_db,
                quality, fresh=not self._stopped, clean_share=clean,
            )
        # Confidence. In simulation, estimates whose peak holds < 60 % of the band power were
        # the only wrong ones for a seated person, so `quality` 0.6 maps to the reporting limit.
        q_peak = float(np.clip((quality - 0.35) / 0.5, 0.0, 1.0))
        q_len = float(np.clip((f0 * span_s - 1.5) / 1.5, 0.0, 1.0))       # no answer under 1.5 breaths, full trust from 3
        q_snr = float(np.clip((snr_db + 2.0) / 5.0, 0.0, 1.0))           # 0 dB (just detectable) -> 0.4, 3 dB -> full
        q_amp = 1.0 if 0.1 <= pp_mm <= 30.0 else 0.3
        q_clean = float(np.clip((clean - 0.5) / 0.3, 0.0, 1.0))           # more than a fifth of the window skipped: trust less
        self._bpm_hist.append(raw_bpm)
        spread = max(self._bpm_hist) - min(self._bpm_hist)
        stable = len(self._bpm_hist) == self._bpm_hist.maxlen and spread <= 2.0   # 3 updates in a row agree
        conf = q_peak * q_len * float(np.sqrt(q_snr)) * q_amp * q_clean * (1.0 if stable else 0.6)
        bpm = float(np.median(self._bpm_hist)) if (conf >= cfg.breath_min_conf and stable) else None
        state = "tracking" if bpm is not None else "acquiring"
        return BreathingResult(
            bpm, conf, state, wave, self._fs, rng, pp_mm, span_s, raw_bpm, snr_db, quality, fresh=fresh, clean_share=clean
        )

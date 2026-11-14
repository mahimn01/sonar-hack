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

Any gross body movement restarts the measurement (it needs ~8-10 s of stillness).

Read-only module: no files, no audio devices.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

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
        self._wavelength = SPEED_OF_SOUND / cfg.chirp_center_hz
        self._sos = sps.butter(2, [0.08, 0.8], "bandpass", fs=self._fs, output="sos")
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
        self._acc: np.ndarray | None = None
        self._acc_n = 0
        self._prev: np.ndarray | None = None
        self._noise: np.ndarray | None = None       # sensor-noise variance per (mic, tap) at the frame rate
        self._count = 0
        self._bpm_hist: deque[float] = deque(maxlen=3)
        self.last = BreathingResult(None, 0.0, "acquiring", fs=self._fs)

    def restart(self, state: str = "moving") -> None:
        """Body moved: throw away the history (the static echo changed) but keep the noise estimate."""
        self._buf.clear()
        self._acc, self._acc_n = None, 0
        self._bpm_hist.clear()
        self._count = 0
        self.last = BreathingResult(None, 0.0, state, fs=self._fs)

    # -- main entry ----------------------------------------------------------------
    def update(self, cir: np.ndarray, moving: bool = False) -> BreathingResult | None:
        """Feed one echo profile [n_mics, n_taps]. Returns a result about once per second, else None."""
        h = cir[:, self._sel].astype(np.complex128)
        # Sensor noise from frame-to-frame differences (movement is far slower than 50 Hz).
        if self._prev is not None:
            d2 = 0.5 * np.abs(h - self._prev) ** 2
            if self._noise is None:
                self._noise = d2 + _TINY
            else:
                self._noise += 0.01 * (np.minimum(d2, 9.0 * self._noise) - self._noise)
        self._prev = h

        if moving:
            if self.last.state != "moving":
                self.restart("moving")
                return self.last
            return None

        self._acc = h.copy() if self._acc is None else self._acc + h
        self._acc_n += 1
        if self._acc_n < self._group:
            return None
        self._buf.append(self._acc / self._acc_n)
        self._acc, self._acc_n = None, 0
        self._count += 1
        if self._count % self._every:
            return None
        self.last = self._estimate()
        return self.last

    # -- the estimate ----------------------------------------------------------------
    def _combine(self, zc: np.ndarray, noise: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, float]:
        """Best-weights sum of a group of taps. Returns (combined complex signal [L], its SNR).

        All taps of the group see the same chest, so after removing each tap's time
        average the group is (nearly) one time function times one weight per tap.
        The first SVD component is exactly that time function with maximum SNR.
        The SNR is "signal power / noise power" after combining; the 1.8 subtracts
        what the largest component of pure noise would give.
        """
        m = zc[:, cols] / np.sqrt(noise[cols])[None, :]            # every column now has unit noise
        u, sv, _ = np.linalg.svd(m, full_matrices=False)
        g = u[:, 0] * sv[0]
        snr = max(float(sv[0] ** 2 / len(g)) - 1.8, 0.0)
        return g, snr

    # Tunables of the phase read-out (class attributes so experiments can override them).
    arc_ratio_thr = 0.10      # "how round is the point cloud" above which the circle fit is trusted
    rate_tilt = 1.0           # spectral tilt used by estimate_rate

    def _phase_to_disp(self, g: np.ndarray) -> tuple[np.ndarray, str]:
        """Complex arc -> displacement in metres (positive = towards the laptop), and the method used.

        `g` has unit noise variance (0.5 per real axis). Two regimes:
          * long arc (deep breathing, more than ~150 degrees): fit the circle, read the angle;
          * short arc (shallow breathing or weak echo): a circle fit is unreliable there
            (a noisy short arc fits many circles), but a short arc is almost a straight
            line, so project onto that line instead. Rate and shape are right; the
            millimetre scale is then only approximate.
        """
        gc = g - g.mean()
        cov = np.cov(np.stack([gc.real, gc.imag]))
        evals, evecs = np.linalg.eigh(cov)                          # ascending
        roundness = max(evals[0] - 0.5, 0.0) / max(evals[1] - 0.5, 1e-9)
        centre, radius = circle_fit(g[:, None])
        k = self._wavelength / (4.0 * np.pi)
        if roundness >= self.arc_ratio_thr:
            return np.unwrap(np.angle(g - centre[0])) * k, "arc"
        axis = evecs[0, 1] + 1j * evecs[1, 1]                       # direction of the line
        proj = (gc * np.conj(axis)).real
        # Sign: moving towards the laptop turns the echo counter-clockwise around the centre.
        radial = g.mean() - centre[0]
        ccw = 1j * radial / (abs(radial) + _TINY)
        if (ccw * np.conj(axis)).real < 0:
            proj = -proj
        r = max(float(radius[0]), 1.2 * float(np.max(np.abs(proj))))
        return np.arcsin(np.clip(proj / r, -1.0, 1.0)) * k, "line"

    def _estimate(self) -> BreathingResult:
        cfg = self.cfg
        n = len(self._buf)
        window_s = n / self._fs
        if n < int(4.0 * self._fs) or self._noise is None:
            return BreathingResult(None, 0.0, "acquiring", fs=self._fs, window_s=window_s)

        zz = np.stack(self._buf).reshape(n, -1)                     # [L, mics*taps]
        n_mics, n_taps = self._buf[0].shape
        best: tuple[float, float | None, float, np.ndarray, int, float] | None = None
        any_motion = False
        for lp_sos, noise_gain in self._lp:
            flat = sps.sosfiltfilt(lp_sos, zz, axis=0)
            zc = flat - flat.mean(axis=0, keepdims=True)
            noise = (self._noise / self._group).reshape(-1) * noise_gain + _TINY

            # Where does something move? Excess variance over the sensor noise, per distance tap.
            excess = (np.mean(np.abs(zc) ** 2, axis=0) / noise - 1.0).reshape(n_mics, n_taps).sum(axis=0)
            smooth = np.convolve(excess, np.ones(3) / 3.0, mode="same")
            peaks: list[int] = []
            for t in np.argsort(smooth)[::-1]:
                if smooth[t] < 0.3 * n_mics or len(peaks) >= 3:
                    break
                if all(abs(int(t) - q) > 4 for q in peaks):
                    peaks.append(int(t))

            for t in peaks:
                taps = np.arange(max(t - 2, 0), min(t + 3, n_taps))
                cols = np.concatenate([m * n_taps + taps for m in range(n_mics)])
                g, snr = self._combine(zc, noise, cols)
                if snr < 2.5:                                       # combined moving echo < 4 dB over noise
                    continue
                # Is it still moving NOW? (breath held, or the person left: the old part of the
                # window would otherwise keep "seeing" breathing for another 30 s)
                recent = g[-int(6.0 * self._fs):]
                if float(np.var(recent)) - 1.0 < 0.25 * snr:
                    continue
                any_motion = True
                disp = sps.detrend(self._phase_to_disp(g)[0])
                if n >= 40:
                    disp = sps.sosfiltfilt(self._sos, disp)
                f0, quality = estimate_rate(disp, self._fs, cfg.breath_band_hz, tilt=self.rate_tilt)
                rank = quality * snr**0.25
                if best is None or rank > best[0]:
                    best = (rank, f0, quality, disp, t, snr)
        if best is None or not any_motion:
            return BreathingResult(None, 0.0, "no_signal", fs=self._fs, window_s=window_s)
        _, f0, quality, disp, tap, snr = best

        pp_mm = 1000.0 * float(np.percentile(disp, 97) - np.percentile(disp, 3))
        peak = float(np.max(np.abs(disp))) + _TINY
        wave = (disp / peak).astype(np.float32)
        snr_db = 10.0 * np.log10(snr + _TINY)
        rng = float(cfg.range_m[self._sel[tap]])
        if f0 is None or n < self._min_len:
            return BreathingResult(None, 0.0, "acquiring", wave, self._fs, rng, pp_mm, window_s, None, snr_db)

        raw_bpm = 60.0 * f0
        # Confidence. In simulation, estimates whose peak holds < 60 % of the band power were
        # the only wrong ones for a seated person, so `quality` 0.6 maps to the reporting limit.
        q_peak = float(np.clip((quality - 0.35) / 0.5, 0.0, 1.0))
        q_len = float(np.clip((f0 * window_s - 1.5) / 1.5, 0.0, 1.0))     # no answer under 1.5 breaths, full trust from 3
        q_snr = float(np.clip((snr_db - 2.0) / 8.0, 0.0, 1.0))
        q_amp = 1.0 if 0.1 <= pp_mm <= 30.0 else 0.3
        self._bpm_hist.append(raw_bpm)
        spread = max(self._bpm_hist) - min(self._bpm_hist)
        stable = len(self._bpm_hist) == self._bpm_hist.maxlen and spread <= 2.0   # 3 updates in a row agree
        conf = q_peak * q_len * float(np.sqrt(q_snr)) * q_amp * (1.0 if stable else 0.6)
        bpm = float(np.median(self._bpm_hist)) if (conf >= cfg.breath_min_conf and stable) else None
        state = "tracking" if bpm is not None else "acquiring"
        return BreathingResult(bpm, conf, state, wave, self._fs, rng, pp_mm, window_s, raw_bpm, snr_db, quality)

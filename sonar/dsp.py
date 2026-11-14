"""dsp.py — streaming signal-processing blocks of the sonar receiver.

Plain words: these are the four machines every microphone sample goes through.

  1. `Prefilter`     throws away everything below the ultrasonic band (speech,
                     music, keyboard thumps) so a loud hall cannot leak in.
  2. `ToneDemod`     shifts each pilot tone down to 0 Hz and keeps +-300 Hz
                     around it. Movement shows up there as a Doppler shift.
  3. `DopplerSTFT`   turns that into one spectrogram column per 20 ms frame.
  4. `ChirpReceiver` lines the 20 ms frames up with the transmitted chirp,
                     follows any clock drift between speaker and microphone,
                     and returns the echo profile (strength and phase of the
                     echo at each distance).

All blocks accept audio in arbitrary block sizes and keep their own state, so
the same code runs live, on recorded files and inside the tests.

Read-only module: no files, no audio devices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps

from .config import SonarConfig
from .tx import TxReference

_EPS = 1e-30


def db(x: np.ndarray | float, floor: float = _EPS) -> np.ndarray | float:
    """Amplitude -> decibels (20*log10)."""
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))


def db_pow(x: np.ndarray | float, floor: float = _EPS) -> np.ndarray | float:
    """Power -> decibels (10*log10)."""
    return 10.0 * np.log10(np.maximum(x, floor))


# ---------------------------------------------------------------------------
# 1. Prefilter
# ---------------------------------------------------------------------------
def design_prefilter(cfg: SonarConfig) -> np.ndarray:
    """Linear-phase high-pass FIR that passes the sonar band and kills audible sound."""
    span = cfg.doppler_bins / 2 * cfg.bb_fs / cfg.stft_fft
    lowest = min(min(cfg.tone_hz), cfg.chirp_lo_hz)
    pass_edge = lowest - span - cfg.prefilter_margin_hz
    stop_edge = pass_edge - cfg.prefilter_transition_hz
    numtaps, beta = sps.kaiserord(cfg.prefilter_atten_db, cfg.prefilter_transition_hz / (0.5 * cfg.fs))
    numtaps |= 1  # odd length: required for a high-pass with exact linear phase
    return sps.firwin(numtaps, 0.5 * (pass_edge + stop_edge), window=("kaiser", beta), pass_zero=False, fs=cfg.fs)


def guard_bins(cfg: SonarConfig) -> np.ndarray:
    """FFT bins of one frame that lie in the ultrasonic band but carry none of our own signal.

    They are inside the prefilter's passband, at least 350 Hz away from every pilot tone
    (so no Doppler-shifted echo reaches them) and 150 Hz outside the chirp band. What is
    left there is the room's own noise: the per-frame "how noisy is it right now" meter.
    """
    span = cfg.doppler_bins / 2 * cfg.bb_fs / cfg.stft_fft
    lowest = min(min(cfg.tone_hz), cfg.chirp_lo_hz)
    f = np.fft.rfftfreq(cfg.frame_len, 1.0 / cfg.fs)
    ok = (f >= lowest - span - cfg.prefilter_margin_hz + 100.0) & (f <= 0.49 * cfg.fs)
    for tone in cfg.tone_hz:
        ok &= np.abs(f - tone) > span + 100.0
    ok &= (f < cfg.chirp_lo_hz - 150.0) | (f > cfg.chirp_hi_hz + 150.0)
    return np.flatnonzero(ok)


class Prefilter:
    """Streaming high-pass over [n, n_mics] blocks."""

    def __init__(self, cfg: SonarConfig, n_mics: int) -> None:
        self.taps = design_prefilter(cfg)
        self._zi = np.zeros((len(self.taps) - 1, n_mics))

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self._zi = sps.lfilter(self.taps, [1.0], x, axis=0, zi=self._zi)
        return y


# ---------------------------------------------------------------------------
# 2. Tone demodulation (complex band-pass + decimation)
# ---------------------------------------------------------------------------
class ToneDemod:
    """Each pilot tone -> complex baseband at `cfg.bb_fs`, static carrier removed.

    For tone frequency f the output sample m (taken at input index n = m*decim) is
        y[m] = exp(-j 2 pi f n / fs) * sum_i x[n - i] * h[i] * exp(+j 2 pi f i / fs)
    i.e. a band-pass centred on the tone followed by a shift to 0 Hz. A steady
    echo (direct path, furniture) lands exactly on 0 Hz and is removed by a
    5 Hz high-pass; a reflector moving at radial speed v lands on +2 v f / c.
    """

    def __init__(self, cfg: SonarConfig, n_mics: int) -> None:
        self.cfg = cfg
        self.n_mics = n_mics
        width = cfg.bb_fs - 2 * cfg.demod_pass_hz           # pass edge .. first alias of the pass band
        numtaps, beta = sps.kaiserord(cfg.demod_atten_db, width / (0.5 * cfg.fs))
        lowpass = sps.firwin(numtaps, 0.5 * cfg.bb_fs, window=("kaiser", beta), fs=cfg.fs)
        i = np.arange(numtaps)
        self._tones = np.asarray(cfg.tone_hz, dtype=np.float64)
        band = lowpass[None, :] * np.exp(2j * np.pi * self._tones[:, None] * i[None, :] / cfg.fs)
        self._g_rev = band[:, ::-1].copy()                  # [n_tones, numtaps], time-reversed for a dot product
        self._len = numtaps
        self._hist = np.zeros((numtaps - 1, n_mics))
        self._n = 0                                          # absolute index of the next input sample
        self._sos = sps.butter(2, cfg.clutter_hp_hz, "highpass", fs=cfg.bb_fs, output="sos")
        self._zi: np.ndarray | None = None
        self.carrier = np.zeros((cfg.n_tones, n_mics), dtype=np.complex128)   # slow average of the raw baseband
        self._carrier_alpha = 1.0 / (0.5 * cfg.bb_fs)        # 0.5 s time constant

    @property
    def group_delay_samples(self) -> float:
        return 0.5 * (self._len - 1)

    def process(self, x: np.ndarray) -> np.ndarray:
        """x: [n, n_mics] real. Returns [m, n_tones, n_mics] complex baseband (high-passed)."""
        cfg = self.cfg
        buf = np.concatenate([self._hist, x], axis=0)
        first = (-self._n) % cfg.decim                       # offset of the first output inside this block
        starts = np.arange(first, len(x), cfg.decim)         # window start (in buf) of every output
        self._hist = buf[len(buf) - (self._len - 1):]
        n_abs = self._n + starts                             # absolute input index of each output
        self._n += len(x)
        if len(starts) == 0:
            return np.zeros((0, cfg.n_tones, self.n_mics), dtype=np.complex128)

        windows = np.lib.stride_tricks.sliding_window_view(buf, self._len, axis=0)[starts]   # [m, mics, L]
        y = np.einsum("mcl,tl->mtc", windows, self._g_rev, optimize=True)
        phase = np.fmod(self._tones[None, :] * n_abs[:, None], cfg.fs) / cfg.fs               # exact, no drift
        y *= np.exp(-2j * np.pi * phase)[:, :, None]

        # Slow average of the raw baseband = the static carrier (used for level / SNR read-outs).
        self.carrier += (1.0 - np.exp(-len(y) * self._carrier_alpha)) * (y.mean(axis=0) - self.carrier)

        if self._zi is None:                                 # start the high-pass in steady state: no start-up thump
            zi = sps.sosfilt_zi(self._sos)                   # [sections, 2]
            self._zi = zi[:, :, None, None] * y[0][None, None, :, :]
        out, self._zi = sps.sosfilt(self._sos, y, axis=0, zi=self._zi)
        return out

    def carrier_dbfs(self) -> np.ndarray:
        """Level of each tone's static carrier at each mic, dB re full scale, [n_tones, n_mics]."""
        return np.asarray(db(2.0 * self.carrier))


# ---------------------------------------------------------------------------
# 3. Doppler spectrogram columns
# ---------------------------------------------------------------------------
class DopplerSTFT:
    """Baseband samples -> one Doppler column per frame.

    Each column is a pair (power_db, lateral):

    power_db  [n_tones * n_mics, doppler_bins] float32, dB re full scale,
              channel index = tone * n_mics + mic, bins ordered from -250 Hz
              (moving away) to +242 Hz (moving towards the laptop).

    lateral   [doppler_bins] complex64, or None with a single microphone.
              Two-microphone interferometry: with X0, X1 the complex spectra of
              the two mics, R = X0 * conj(X1) holds their phase difference, and
                  lateral = sum over tones of R[now] * conj(R[previous frame])
              holds how fast that difference turns. Its angle is
                  2 pi * hop * (Doppler at mic 0 - Doppler at mic 1),
              positive when the reflector moves towards microphone 0 (the user's
              left), negative when it moves right. The speaker path cancels, so
              this works whatever the speakers are doing.
    """

    def __init__(self, cfg: SonarConfig, n_mics: int) -> None:
        self.cfg = cfg
        self.n_mics = n_mics
        self._win = sps.get_window("hann", cfg.stft_win, fftbins=True)
        self._scale = 2.0 / self._win.sum()                  # a full-scale sine reads 0 dB
        half = cfg.doppler_bins // 2
        self._sel = np.r_[cfg.stft_fft - half:cfg.stft_fft, 0:half]
        self._buf = np.zeros((cfg.stft_win - cfg.stft_hop, cfg.n_tones, n_mics), dtype=np.complex128)
        self._prev_cross: np.ndarray | None = None
        self._smooth = np.ones(3) / 3.0

    def process(self, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray | None]]:
        cfg = self.cfg
        self._buf = np.concatenate([self._buf, y], axis=0)
        cols: list[tuple[np.ndarray, np.ndarray | None]] = []
        while len(self._buf) >= cfg.stft_win:
            seg = self._buf[:cfg.stft_win] * self._win[:, None, None]
            spec = np.fft.fft(seg, n=cfg.stft_fft, axis=0)[self._sel] * self._scale    # [F, tones, mics]
            power = (spec.real**2 + spec.imag**2).transpose(1, 2, 0)                    # [tones, mics, F]
            power_db = db_pow(power).reshape(cfg.n_tones * self.n_mics, -1).astype(np.float32)
            lateral = None
            if self.n_mics >= 2:
                cross = spec[:, :, 0] * np.conj(spec[:, :, 1])                           # [F, tones]
                if self._prev_cross is not None:
                    lag = np.sum(cross * np.conj(self._prev_cross), axis=1)
                    lateral = np.convolve(lag, self._smooth, mode="same").astype(np.complex64)
                else:
                    lateral = np.zeros(len(self._sel), dtype=np.complex64)
                self._prev_cross = cross
            cols.append((power_db, lateral))
            self._buf = self._buf[cfg.stft_hop:]
        return cols


# ---------------------------------------------------------------------------
# 4. Chirp receiver: frame sync, drift tracking, echo profile
# ---------------------------------------------------------------------------
@dataclass
class CirFrame:
    """Echo profile of one 20 ms frame."""

    cir: np.ndarray           # [n_mics, n_taps] complex: echo gain and phase per distance tap
    coherence: float          # 0..1 similarity with the reference channel (1 = locked)
    delay_samples: float      # current fractional delay correction
    drift_ppm: float          # estimated speaker-vs-mic clock offset
    direct_db: np.ndarray     # [n_mics] direct-path gain in dB (RX amplitude / TX amplitude)
    ambient: float = 0.0      # noise power of this frame in the ultrasonic band, measured where we transmit nothing


class ChirpReceiver:
    """Aligns frames to the transmitted chirp and returns the channel impulse response.

    Maths (per frame, per mic). The chirp is a comb of K lines at bins k with
    unit phasors U[k] and amplitude a. For the received frame r[n]:

        H[k]  = FFT(r)[k] * conj(U[k]) / (a * N/2)            channel at line k
        Hc[k] = H[k] * exp(+j 2 pi k delta / N)               undo clock slip `delta` (samples)
        h[n]  = (1/sum W) * sum_m W[m] * Hc[k_lo+m] * exp(+j 2 pi m n / M)   echo profile

    |h[n]| is the amplitude gain of the echo whose travel time is n * N/M
    samples; its phase turns by -4 pi d / lambda when the reflector moves d
    closer or further (lambda = wavelength at the band centre, ~17 mm).

    States: "sync" (find the direct path, park it at `direct_tap`), "ref"
    (learn the static channel and the drift rate), "run".
    """

    def __init__(self, cfg: SonarConfig, ref: TxReference, n_mics: int) -> None:
        self.cfg = cfg
        self.n_mics = n_mics
        self._N = cfg.frame_len
        self._bins = ref.chirp_bins
        k = len(self._bins)
        self._conj_ref = np.conj(ref.chirp_phasors) / (ref.chirp_line_amp * self._N / 2)
        self._win = sps.get_window(cfg.cir_window, k, fftbins=False)
        self._wsum = float(self._win.sum())
        self._kc = float(np.sum(self._win * self._bins) / self._wsum)        # weighted centre bin
        wn = 2 * np.pi * cfg.drift_loop_hz / cfg.frame_rate
        self._kp, self._ki = (2.0 * wn, wn * wn) if cfg.drift_loop_hz > 0 else (0.0, 0.0)
        self._keep = 64                                                        # samples kept before `_start`
        self._guard = guard_bins(cfg)
        self.reset()

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        """Forget everything (called at start)."""
        self._buf = np.zeros((0, self.n_mics))
        self._base = 0            # absolute sample index of _buf[0]
        self._start = 0           # absolute sample index where the next frame starts
        self.n_resync = -1
        self.relock()

    def relock(self) -> None:
        """Forget the alignment but keep the audio (after a dropped buffer or lost lock)."""
        self.state = "sync"
        self._delta = 0.0         # fractional delay still to be undone (samples)
        self._rate = 0.0          # drift in samples per frame
        self._h_ref: np.ndarray | None = None
        self._ref_frames: list[np.ndarray] = []
        self._bad = 0
        self.n_resync += 1

    @property
    def drift_ppm(self) -> float:
        return 1e6 * self._rate / self._N

    # -- helpers -----------------------------------------------------------
    def _spectrum(self, frame: np.ndarray) -> np.ndarray:
        """[N, mics] samples -> [K, mics] channel estimate at the chirp lines."""
        return np.fft.rfft(frame, axis=0)[self._bins] * self._conj_ref[:, None]

    def _take(self, count: int) -> np.ndarray | None:
        lo = self._start - self._base
        if lo < 0 or lo + count > len(self._buf):
            return None
        return self._buf[lo:lo + count]

    def _trim(self) -> None:
        # Never drop more than we hold: right after a coarse sync `_start` can point past the
        # end of the buffer, and `_base` must keep matching the true index of the next sample.
        drop = min(self._start - self._keep - self._base, len(self._buf))
        if drop > 0:
            self._buf = self._buf[drop:]
            self._base += drop

    def _coarse_sync(self, block: np.ndarray) -> None:
        """Find the direct path in the averaged frame and move the frame grid onto it."""
        cfg = self.cfg
        mean_frame = block.reshape(cfg.sync_frames, self._N, self.n_mics).mean(axis=0)
        spec = np.zeros((self._N, self.n_mics), dtype=np.complex128)
        spec[self._bins] = self._spectrum(mean_frame) * self._win[:, None]
        env = np.sum(np.abs(np.fft.ifft(spec, axis=0)) ** 2, axis=1)           # echo power vs delay (samples)
        peak = int(np.argmax(env))
        # The direct path is the EARLIEST strong arrival: look up to 3 ms back for a
        # local maximum within 6 dB of the strongest one.
        for c in (peak - np.arange(int(0.003 * cfg.fs), 0, -1)) % self._N:
            if env[c] >= 0.25 * env[peak] and env[c] >= env[c - 1] and env[c] >= env[(c + 1) % self._N]:
                peak = int(c)
                break
        target = int(round(cfg.direct_tap * cfg.tap_samples))
        self._start += cfg.sync_frames * self._N + ((peak - target) % self._N)
        self.state = "ref"
        self._ref_frames = []

    def _finish_reference(self) -> None:
        """Estimate the drift rate from the reference frames and average them coherently."""
        frames = np.stack(self._ref_frames)                                     # [F, K, mics]
        z = np.sum(frames * np.conj(frames[0])[None], axis=(1, 2))
        psi = np.unwrap(np.angle(z))
        idx = np.arange(len(psi))
        slope = np.polyfit(idx, psi, 1)[0] if self._kp > 0 else 0.0             # rad per frame
        self._rate = -slope * self._N / (2 * np.pi * self._kc)
        comp = np.exp(2j * np.pi * self._bins[None, :, None] * (self._rate * idx)[:, None, None] / self._N)
        self._h_ref = np.mean(frames * comp, axis=0)
        self._delta = self._rate * (len(psi) - 1)       # the run loop adds one more `rate` before its first frame
        self._ref_frames = []
        self.state = "run"

    # -- main entry ----------------------------------------------------------
    def process(self, x: np.ndarray) -> list[CirFrame]:
        """Feed [n, n_mics] prefiltered samples; returns the frames completed by them."""
        cfg = self.cfg
        self._buf = np.concatenate([self._buf, x], axis=0)
        out: list[CirFrame] = []
        while True:
            if self.state == "sync":
                block = self._take(cfg.sync_frames * self._N)
                if block is None:
                    break
                self._coarse_sync(block)
                continue

            if self.state == "ref":
                frame = self._take(self._N)
                if frame is None:
                    break
                self._ref_frames.append(self._spectrum(frame))
                self._start += self._N
                if len(self._ref_frames) >= cfg.ref_frames:
                    self._finish_reference()
                continue

            # ---- run: one frame --------------------------------------------
            delta = self._delta + self._rate
            start = self._start
            while delta > 0.5:              # content is late by more than half a sample: start one sample later
                start += 1
                delta -= 1.0
            while delta < -0.5:
                start -= 1
                delta += 1.0
            start = max(start, self._base)
            if start - self._base + self._N > len(self._buf):
                break                       # not enough audio yet; nothing was committed
            self._delta, self._start = delta, start
            frame = self._buf[start - self._base:start - self._base + self._N]
            spec = np.fft.rfft(frame, axis=0)
            # Ambient noise of this frame: the FFT bins next to our band that carry none of our
            # own signal. A click or a hiss in the room shows here whatever the echoes are doing.
            ambient = float(np.median(np.abs(spec[self._guard]) ** 2)) if len(self._guard) else 0.0
            h_k = spec[self._bins] * self._conj_ref[:, None] * np.exp(2j * np.pi * self._bins * self._delta / self._N)[:, None]
            assert self._h_ref is not None
            z = np.sum(h_k * np.conj(self._h_ref))
            norm = np.sqrt(np.sum(np.abs(h_k) ** 2) * np.sum(np.abs(self._h_ref) ** 2)) + _EPS
            coherence = float(np.abs(z) / norm)
            if self._kp > 0:
                err = -np.angle(z) * self._N / (2 * np.pi * self._kc)           # residual delay (samples)
                self._delta += self._kp * err
                self._rate += self._ki * err

            padded = np.zeros((cfg.cir_fft, self.n_mics), dtype=np.complex128)
            padded[:len(self._bins)] = h_k * self._win[:, None]
            cir = (np.fft.ifft(padded, axis=0)[:cfg.n_taps] * (cfg.cir_fft / self._wsum)).T
            out.append(
                CirFrame(
                    cir=cir.astype(np.complex64),
                    coherence=coherence,
                    delay_samples=float(self._delta),
                    drift_ppm=self.drift_ppm,
                    direct_db=np.asarray(db(np.max(np.abs(cir[:, :cfg.direct_tap + 6]), axis=1))),
                    ambient=ambient,
                )
            )
            self._start += self._N

            # Lost lock (dropped audio, laptop lid moved a lot): start over.
            self._bad = self._bad + 1 if coherence < 0.5 else 0
            if self._bad >= 10:
                self.relock()
                continue
            # Follow slow changes of the static channel (temperature, lid angle).
            if coherence > 0.9:
                self._h_ref += (h_k - self._h_ref) / (20.0 * cfg.frame_rate)
        self._trim()
        return out

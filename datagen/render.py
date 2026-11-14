"""render.py — turn (speaker signal, virtual laptop, moving scene) into microphone audio.

Plain words
-----------
Every echo is just a delayed, quieter copy of what the speaker played:

    mic(t) = sum over paths of   gain(t) * speaker(t - delay(t))

where delay = (speaker->reflector distance + reflector->mic distance) / speed of sound.
A hand moving towards the laptop shortens the delay over time, which is exactly
what produces the Doppler shift. We compute that sum directly, so the result is
correct for ANY probe signal the core package chooses (tones, chirps, both).

How it stays fast and exact
---------------------------
The probe signal lives in a narrow band near 19 kHz. We split it into
    (slow complex envelope) x (fast carrier at f0).
The carrier phase of each path, exp(-j*2*pi*f0*delay), is computed exactly:
it carries all the Doppler and breathing information. The slow envelope is
only looked up at the delayed time (cheap interpolation). Moving paths are
summed at a low sample rate and upsampled once at the end.

The pieces added on top, in order: direct speaker->mic leakage and static room
reflections, microphone colouration, hall noise, clicks, interfering tones,
slow gain wander, clock mismatch, and finally 16-bit rounding.

Read-only helper module: it returns arrays, it never touches audio devices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft
from scipy.interpolate import CubicSpline
from scipy.signal import resample_poly

from .hardware import Hardware
from .physics import smooth_noise

PRE_ROLL = 4800  # default speaker-signal samples needed BEFORE the first output sample (100 ms)
POST_ROLL = 480
SNR_REF_HZ = 25.0  # width of the frequency slice used to quote echo SNR
# body parts whose echo IS the labelled motion (used only to report how audible the gesture is)
KEY_TAGS = ("palm", "finger", "forearm", "walker_torso", "walker_head", "walker_arm", "walker_leg")


@dataclass
class Rendered:
    """Result of one render. `rx` is what a recorder would have saved."""

    rx: np.ndarray  # [n_out, n_mic] float32 in [-1, 1]
    parts: dict  # un-scaled components ("direct = 1" units): static, dynamic
    scale: float  # multiply parts by this to get full-scale units
    f0: float
    dec: int
    echo_snr_db: float = float("nan")  # labelled motion vs in-band noise in a 25 Hz slice (see render())
    noise_density_dbfs: float = float("nan")  # in-band noise, dBFS per Hz


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _route(tx: np.ndarray, hw: Hardware) -> list:
    """Decide what each of the two speakers plays. Returns [left, right]; None = silent."""
    tx = np.asarray(tx, dtype=np.float64)
    if tx.ndim == 1:
        sig = [tx, tx]
        if hw.tx_route == "left":
            sig = [tx, None]
        elif hw.tx_route == "right":
            sig = [None, tx]
    else:
        sig = [tx[:, 0], tx[:, 1] if tx.shape[1] > 1 else tx[:, 0]]
    if hw.swap_speakers:
        sig = sig[::-1]
    return [None if (s is None or not np.any(s)) else s for s in sig]


def occupied_band(tx: np.ndarray, fs: int, floor_db: float = -50.0) -> tuple:
    """Find where the probe signal's energy lives: returns (f_low, f_high) in Hz."""
    x = np.asarray(tx, dtype=np.float64)
    if x.ndim > 1:
        x = x.sum(axis=1) if np.any(x.sum(axis=1)) else x[:, 0]
    n = min(len(x), 1 << 17)
    spec = np.abs(sfft.rfft(x[:n] * np.hanning(n)))
    f = sfft.rfftfreq(n, 1.0 / fs)
    thr = spec.max() * 10.0 ** (floor_db / 20.0)
    idx = np.nonzero(spec > thr)[0]
    if len(idx) == 0:
        return 17000.0, 21000.0
    return float(f[idx[0]]), float(f[idx[-1]])


def _carrier(n0: float, n: int, f0: float, fs: int, step: float = 1.0) -> np.ndarray:
    """exp(j*2*pi*f0*(n0 + step*k)/fs) with the phase kept small for precision."""
    k = np.arange(n, dtype=np.float64)
    cycles = np.mod((n0 + step * k) * (f0 / fs), 1.0)
    return np.exp(2j * np.pi * cycles)


def _to_grid(t_ctrl: np.ndarray, values: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Smoothly resample a control-rate trajectory (cubic spline => smooth velocity)."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0 or values.shape[0] != len(t_ctrl):
        return values
    t_clip = np.clip(t_new, t_ctrl[0], t_ctrl[-1])
    return CubicSpline(t_ctrl, values, axis=0)(t_clip)


def _norm(v: np.ndarray) -> np.ndarray:
    """Length of each row vector (faster than np.linalg.norm for [n, 3] arrays)."""
    return np.sqrt(np.einsum("ij,ij->i", v, v))


def _directivity(cos_t: np.ndarray, n: float, floor: float) -> np.ndarray:
    return floor + (1.0 - floor) * np.maximum(cos_t, 0.0) ** n if n > 0 else np.ones_like(cos_t)


def _max_speed(scene) -> float:
    v = 0.0
    dt = float(scene.t[1] - scene.t[0])
    for sc in scene.scatterers:
        d = np.diff(sc.pos, axis=0)
        v = max(v, float(np.linalg.norm(d, axis=1).max()) / dt)
    return v


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def render(
    tx: np.ndarray,
    hw: Hardware,
    scene,
    n_out: int,
    rng: np.random.Generator | None = None,
    pre: int = PRE_ROLL,
    band: tuple | None = None,
    add_noise: bool = True,
    key_tags: tuple = KEY_TAGS,
    include_static: bool = True,
) -> Rendered:
    """Simulate what the microphones record.

    tx     : speaker samples, shape [pre + n_out + post] (mono) or [.., 2] (stereo).
             Output sample 0 lines up with tx[pre], exactly like a recorder that
             saves RX sample-aligned with TX.
    hw     : the virtual laptop + room (see hardware.py)
    scene  : moving reflectors (see scenes.py); scene.t = 0 is output sample 0
    n_out  : number of microphone samples to produce
    band   : (f_low, f_high) of the probe signal; measured from tx if omitted
    include_static : False = return only the echoes of the moving scene (no direct
             leakage, no room). Used to lay synthetic gestures over REAL recordings.
    """
    rng = rng or np.random.default_rng()
    fs = hw.fs
    spk_sig = _route(tx, hw)
    active = [s for s in range(2) if spk_sig[s] is not None]
    if not active:
        raise ValueError("speaker signal is all zeros")
    n_total = len(spk_sig[active[0]])
    pre_eff = pre - int(hw.latency_samples)
    if pre_eff < 2500 or n_total < pre + n_out + 64:
        raise ValueError("tx too short: need pre-roll >= 2500 + latency and pre + n_out + 64 samples")

    f_lo, f_hi = band if band is not None else occupied_band(np.asarray(tx), fs)
    f0 = 0.5 * (f_lo + f_hi)
    c = hw.c

    # ---- 1. speaker colouration + complex envelope of each speaker signal --
    freqs = sfft.rfftfreq(n_total, 1.0 / fs)
    tx_f, env = {}, {}
    n_idx = np.arange(n_total, dtype=np.float64)
    demod = np.exp(-2j * np.pi * np.mod(n_idx * (f0 / fs), 1.0))
    for s in active:
        spec = sfft.rfft(spk_sig[s]) * hw.spk_resp[s](freqs)
        tx_f[s] = spec
        full = np.zeros(n_total, dtype=np.complex128)
        full[: len(spec)] = 2.0 * spec
        full[0] = spec[0]
        if n_total % 2 == 0:
            full[len(spec) - 1] = spec[-1]
        env[s] = (sfft.ifft(full) * demod).astype(np.complex64)

    # ---- 2. moving reflectors, summed at a low "baseband" sample rate -------
    v_max = _max_speed(scene) if scene.scatterers else 0.0
    f_dmax = max(150.0, 1.25 * 2.0 * v_max * f_hi / c)
    half_width = 0.5 * (f_hi - f_lo) + f_dmax + 300.0
    dec = int(np.clip(np.floor(fs / (2.6 * half_width)), 1, 24))
    margin = 24
    nb = int(np.ceil(n_out / dec)) + 2 * margin
    kb = np.arange(nb, dtype=np.float64) - margin
    tb = kb * dec / fs  # scene time of each baseband sample
    base_idx = kb * dec + pre_eff  # matching index into the speaker buffer
    dyn_bb = np.zeros((hw.n_mic, nb), dtype=np.complex64)
    key_bb = np.zeros((hw.n_mic, nb), dtype=np.complex64)  # only the labelled motion (hand/arm or walker)
    is_key = [False]

    k_amp = 10.0 ** (hw.echo_ref_db / 20.0) * (0.3 * 0.3) ** hw.range_gamma
    alpha = hw.absorption_db_per_m(f0)
    mic_pos = [hw.mics[m] for m in range(hw.n_mic)]
    mic_img = [hw.mics[m] * np.array([1.0, 1.0, -1.0]) for m in range(hw.n_mic)]  # mirror in the desk plane

    def add_path(m: int, s: int, tau: np.ndarray, amp: np.ndarray) -> None:
        """Add one echo path: exact carrier phase, interpolated envelope."""
        idx = base_idx - tau * fs
        i0 = np.floor(idx)
        frac = (idx - i0).astype(np.float32)
        i0 = np.clip(i0.astype(np.int64), 0, n_total - 2)
        e = env[s]
        e_d = e[i0] * (1.0 - frac) + e[i0 + 1] * frac
        ph = -2.0 * np.pi * np.mod(f0 * tau, 1.0)
        rot = (np.cos(ph) + 1j * np.sin(ph)).astype(np.complex64)
        contrib = (amp.astype(np.float32) * rot) * e_d
        dyn_bb[m] += contrib
        if is_key[0]:
            key_bb[m] += contrib

    for sc in scene.scatterers:
        is_key[0] = sc.tag in key_tags
        p = _to_grid(scene.t, sc.pos, tb)  # [nb, 3]
        rho = _to_grid(scene.t, np.asarray(sc.rho, dtype=np.float64), tb)
        nrm = _to_grid(scene.t, sc.normal, tb) if sc.normal is not None else None
        tx_side = {}
        for s in active:
            d = hw.speakers[s] - p
            r = np.maximum(_norm(d), 0.03)
            u = d / r[:, None]
            g = _directivity(-(u @ hw.spk_axis[s]), hw.spk_n, hw.spk_floor)
            tx_side[s] = (r, u, g)
        for m in range(hw.n_mic):
            # how well the microphone hears from the reflector's direction
            d_true = mic_pos[m] - p
            u_true = d_true / np.maximum(_norm(d_true), 0.03)[:, None]
            g_rx = _directivity(-(u_true @ hw.mic_axis), hw.mic_n, hw.mic_floor)
            # the echo reaches the mic directly and, optionally, via a bounce off the desk
            variants = [(mic_pos[m], 1.0)]
            if hw.table_refl > 0.0:
                variants.append((mic_img[m], hw.table_refl))
            for m_pos, coeff in variants:
                d = m_pos - p
                r_rx = np.maximum(_norm(d), 0.03)
                u_rx = d / r_rx[:, None]
                for s in active:
                    r_tx, u_tx, g_tx = tx_side[s]
                    amp = k_amp * coeff * rho * g_tx * g_rx / (r_tx * r_rx) ** hw.range_gamma
                    amp = amp * 10.0 ** (-alpha * (r_tx + r_rx) / 20.0)
                    if nrm is not None and sc.spec_q > 0:
                        bis = u_tx + u_rx
                        bis = bis / np.maximum(_norm(bis), 1e-9)[:, None]
                        cosb = np.abs(np.einsum("ij,ij->i", nrm, bis))
                        amp = amp * (sc.spec_floor + (1.0 - sc.spec_floor) * cosb**sc.spec_q)
                    add_path(m, s, (r_tx + r_rx) / c, amp)

    # optional lid wobble: the direct path itself becomes a (barely) moving path
    wobble = getattr(scene, "lid_wobble", None)
    direct_is_dynamic = wobble is not None and np.any(wobble)
    is_key[0] = False
    if direct_is_dynamic:
        w = _to_grid(scene.t, wobble, tb)
        for m in range(hw.n_mic):
            for s in active:
                tau = hw.static_delays[m][s][0] + w / c
                add_path(m, s, tau, np.full(nb, hw.static_amps[m][s][0]))

    # upsample the moving part to the audio rate and put it back on the carrier
    if dec > 1:
        dyn_up = resample_poly(dyn_bb, dec, 1, axis=1)
    else:
        dyn_up = dyn_bb
    dyn_up = dyn_up[:, margin * dec : margin * dec + n_out]
    car = _carrier(pre_eff, n_out, f0, fs)
    dyn = np.real(dyn_up * car[None, :])  # [n_mic, n_out]

    # ---- 3. direct leakage + static reflections (exact, frequency domain) ---
    band_mask = (freqs > f_lo - 1500.0) & (freqs < f_hi + 1500.0)
    fb = freqs[band_mask]
    # The room response changes slowly with frequency (its echoes are < 0.1 s long), so it is
    # evaluated on a 2 Hz grid and interpolated: 6x cheaper, and still one fixed linear room.
    step = max(1, int(round(2.0 / (fs / n_total))))
    fc = np.append(fb[::step], fb[-1] + (fb[1] - fb[0]) * step)
    static = np.zeros((hw.n_mic, n_out))
    for m in range(hw.n_mic if include_static else 0):
        spec = np.zeros(len(freqs), dtype=np.complex128)
        for s in active:
            delays = np.asarray(hw.static_delays[m][s], dtype=np.float64) + hw.latency_samples / fs
            amps = np.asarray(hw.static_amps[m][s], dtype=np.float64)
            if direct_is_dynamic:
                delays, amps = delays[1:], amps[1:]
            if len(delays):
                # the bulk delay is applied exactly; only the slowly varying remainder is interpolated
                d_ref = float(delays[0])
                h_c = np.exp(-2j * np.pi * fc[:, None] * (delays - d_ref)[None, :]) @ amps
                h = (np.interp(fb, fc, h_c.real) + 1j * np.interp(fb, fc, h_c.imag)) * np.exp(-2j * np.pi * fb * d_ref)
                spec[band_mask] += tx_f[s][band_mask] * h
        static[m] = sfft.irfft(spec, n_total)[pre : pre + n_out]

    # ---- 4. microphone colouration (per mic) ---------------------------------
    clean = static + dyn
    if hw.hum_hz > 0.0:
        # mains ripple on the speaker amplifier wobbles the loudness of everything it plays:
        # steady side lines at +-hum_hz next to every tone (seen in real recordings)
        t_out = np.arange(n_out) / fs
        depth = 2.0 * 10.0 ** (hw.hum_db / 20.0)
        ph = rng.uniform(0.0, 2 * np.pi, 2)
        clean = clean * (1.0 + depth * np.cos(2 * np.pi * hw.hum_hz * t_out + ph[0])
                         + 0.4 * depth * np.cos(4 * np.pi * hw.hum_hz * t_out + ph[1]))[None, :]
    f_out = sfft.rfftfreq(n_out, 1.0 / fs)
    for m in range(hw.n_mic):
        clean[m] = sfft.irfft(sfft.rfft(clean[m]) * hw.mic_resp[m](f_out), n_out)
    parts = {"static": static, "dynamic": dyn}

    # ---- 5. clock mismatch between the speaker (DAC) and mic (ADC) devices ---
    if hw.clock_ppm != 0.0:
        clean = _clock_warp(clean, fs, f0, hw.clock_ppm)

    # ---- 6. scale to converter units, add noise, gain wander, quantise --------
    scale = 10.0 ** (hw.direct_gain_db / 20.0)  # direct path: RX amplitude per unit of TX amplitude
    rx = clean * scale
    snr_db, n0_db = float("nan"), float("nan")
    if add_noise:
        noise = _noise(hw, n_out, rng)
        # how audible is the labelled motion? echo power during the event vs noise in a 25 Hz slice
        in_band = (f_out > f_lo - 500.0) & (f_out < f_hi + 500.0)
        n_spec = sfft.rfft(noise, axis=1)[:, in_band]
        n0 = 2.0 * float(np.mean(np.abs(n_spec) ** 2)) / (n_out * fs)  # power per Hz
        n0_db = 10.0 * np.log10(n0 + 1e-30)
        ev = getattr(scene, "events", None) or [{"t_start": 0.0, "t_end": n_out / fs}]
        sel = np.zeros(nb, dtype=bool)
        for e in ev:
            sel |= (tb >= e["t_start"]) & (tb <= e["t_end"])
        if sel.any() and np.any(key_bb):
            p_key = 0.5 * float(np.mean(np.abs(key_bb[:, sel]) ** 2)) * scale**2
            snr_db = 10.0 * np.log10(p_key / (n0 * SNR_REF_HZ) + 1e-30)
        rx = rx + noise
        rx = rx * _gain_track(hw, n_out, rng)[None, :]
    parts["converter_in"] = rx  # before clipping and rounding: where the GPU parity check compares the two paths
    rx = np.clip(rx, -1.0, 1.0)
    if hw.adc_bits:
        q = float(2 ** (hw.adc_bits - 1))
        rx = np.round(rx * q) / q
    if hw.swap_mics:
        rx = rx[::-1]
    return Rendered(rx=rx.T.astype(np.float32), parts=parts, scale=scale, f0=f0, dec=dec,
                    echo_snr_db=float(snr_db), noise_density_dbfs=float(n0_db))


# --------------------------------------------------------------------------
# nuisance models
# --------------------------------------------------------------------------
def _clock_warp(x: np.ndarray, fs: int, f0: float, ppm: float) -> np.ndarray:
    """Re-sample as if the microphone clock ran `ppm` parts-per-million fast.

    Done on the slow envelope (accurate) with the carrier phase handled exactly.
    """
    n = x.shape[1]
    eps = ppm * 1e-6
    k = np.arange(n, dtype=np.float64)
    pos = np.clip(k * (1.0 + eps), 0, n - 1.001)
    i0 = np.floor(pos).astype(np.int64)
    frac = pos - i0
    out = np.empty_like(x)
    down = np.exp(-2j * np.pi * np.mod(k * (f0 / fs), 1.0))
    up = np.exp(2j * np.pi * np.mod(pos * (f0 / fs), 1.0))
    for m in range(x.shape[0]):
        spec = sfft.fft(x[m])
        spec[n // 2 + 1 :] = 0.0
        spec[1 : (n + 1) // 2] *= 2.0
        e = sfft.ifft(spec) * down
        e_w = e[i0] * (1.0 - frac) + e[np.minimum(i0 + 1, n - 1)] * frac
        out[m] = np.real(e_w * up)
    return out


def _noise(hw: Hardware, n: int, rng: np.random.Generator) -> np.ndarray:
    """Hall + microphone noise for both channels (full-scale units)."""
    fs = hw.fs
    out = rng.standard_normal((hw.n_mic, n)) * 10.0 ** (hw.noise_dbfs / 20.0)
    f = sfft.rfftfreq(n, 1.0 / fs)
    direct = 0.35 * 10.0 ** (hw.direct_gain_db / 20.0)  # roughly the direct tone level at the converter

    # audible hall babble (speech/music-like): loud below a few kHz, a weak tail up high.
    # It is the same sound at both mics (tiny delay difference), so make it coherent.
    if hw.babble_dbfs > -150.0:
        shape = 1.0 / np.sqrt(1.0 + (f / 400.0) ** 2) / (1.0 + (f / 3500.0) ** 2)
        shape[0] = 0.0
        w = sfft.irfft(sfft.rfft(rng.standard_normal(n)) * shape, n)
        envl = 0.35 + np.abs(smooth_noise(rng, n, fs, 4.0))  # syllable-like loudness changes
        w = w * envl
        w *= 10.0 ** (hw.babble_dbfs / 20.0) / (w.std() + 1e-12)
        out += w[None, :] * (1.0 + 0.1 * rng.standard_normal((hw.n_mic, 1)))

    # broadband ultrasonic clicks: keys, zips, claps, chair squeaks
    n_burst = rng.poisson(hw.hf_burst_rate * n / fs) if hw.hf_burst_rate > 0 else 0
    for _ in range(int(n_burst)):
        start = int(rng.integers(0, max(1, n - 64)))
        length = int(min(n - start, fs * rng.uniform(0.002, 0.03)))
        if length < 8:
            continue
        burst = rng.standard_normal(length) * np.exp(-np.arange(length) / (length / rng.uniform(2.0, 6.0)))
        burst -= np.convolve(burst, np.ones(8) / 8.0, mode="same")  # keep the high frequencies
        level = direct * 10.0 ** ((hw.hf_burst_db + rng.normal(0.0, 5.0)) / 20.0)
        burst *= level / (np.abs(burst).max() + 1e-12)
        out[:, start : start + length] += burst[None, :] * (1.0 + 0.2 * rng.standard_normal((hw.n_mic, 1)))

    # steady interfering tones (other gadgets, switching supplies)
    t = np.arange(n) / fs
    for f_t, level_db, drift in hw.tones:
        a = direct * 10.0 ** (level_db / 20.0)
        ph = 2 * np.pi * (f_t * t + 0.5 * drift * t * t)
        for m in range(hw.n_mic):
            out[m] += a * np.cos(ph + rng.uniform(0, 2 * np.pi))
    return out


def _gain_track(hw: Hardware, n: int, rng: np.random.Generator) -> np.ndarray:
    """Slow gain wander plus an optional sudden step (AGC-like behaviour)."""
    g_db = np.zeros(n)
    if hw.gain_drift_db > 0:
        g_db += hw.gain_drift_db * smooth_noise(rng, n, hw.fs, 0.7)
    if hw.gain_step_db != 0.0:
        k0 = int(rng.integers(n // 8, max(n // 8 + 1, 7 * n // 8)))
        ramp = np.clip((np.arange(n) - k0) / (0.02 * hw.fs), 0.0, 1.0)
        g_db += hw.gain_step_db * ramp
    return 10.0 ** (g_db / 20.0)

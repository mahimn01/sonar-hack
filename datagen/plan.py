"""plan.py — everything random about one synthetic clip, as plain numbers (no PyTorch needed).

Plain words
-----------
The GPU renderer (render_torch.py) splits render.py's work in two. This file is the
first half, done on the CPU by ordinary worker processes: it takes the virtual laptop
and the moving scene of one clip and writes down every number the renderer will need,
including the random ones (hum phase, hall noise, gain wander). Those are drawn by
render.py's own functions in render.py's own order, so for the same seed the GPU
renders exactly the clip the CPU renderer would have made.

Read-only helper module: it returns arrays, it never touches files or audio devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hardware import Hardware
from .render import KEY_TAGS, _gain_track, _max_speed, _noise

MARGIN = 24                       # baseband samples of slack on each side (render.py: margin)
RESP_FIELDS = ("f_ref", "tilt_db_per_khz", "shelf_f", "shelf_db", "shelf_width", "ripple_db", "ripple_period",
               "ripple_phase", "lp_f", "lp_order", "delay_s", "gain_db", "phase_rad")


# --------------------------------------------------------------------------
# The plan: everything random about one clip, as plain numpy (picklable)
# --------------------------------------------------------------------------
@dataclass
class ClipPlan:
    n_out: int
    pre: int
    n_total: int
    dec: int                       # baseband decimation render.py would pick for this scene
    include_static: bool
    add_noise: bool
    fs: int
    c: float                       # speed of sound in this clip's room
    alpha: float                   # air absorption, dB per metre
    k_amp: float
    gamma: float
    spk: np.ndarray                # [2, 3]
    spk_axis: np.ndarray           # [2, 3]
    spk_n: float
    spk_floor: float
    mic: np.ndarray                # [n_mic, 3]
    mic_axis: np.ndarray           # [3]
    mic_n: float
    mic_floor: float
    table_refl: float
    spk_col: tuple                 # which column of the speaker signal each speaker plays; -1 = silent
    direct_gain_db: float
    adc_bits: int
    swap_mics: bool
    clock_ppm: float
    latency: int
    hum_hz: float
    hum_db: float
    hum_ph: np.ndarray             # [2] drawn like render.py
    static_delays: np.ndarray      # [n_mic, 2, T] seconds, latency included, padded
    static_amps: np.ndarray        # [n_mic, 2, T], zero where padded
    static_any: np.ndarray         # [n_mic, 2] bool: this (mic, speaker) pair has taps
    spk_resp: list                 # per speaker: ("resp", [13]) or ("table", f, g, extra_db)
    mic_resp: list
    t_ctrl: np.ndarray             # control times of the scene [Tc]
    pos: np.ndarray                # [S, Tc, 3] float64
    rho: np.ndarray                # [S, Tc] float32
    nrm: np.ndarray                # [S, Tc, 3] float32 (zeros where unused)
    use_spec: np.ndarray           # [S] bool
    spec_q: np.ndarray             # [S] float32
    spec_floor: np.ndarray         # [S] float32
    key: np.ndarray                # [S] bool: part of the labelled motion
    wobble: np.ndarray | None      # [Tc] lid wobble in metres, or None
    wobble_tau0: np.ndarray | None  # [n_mic, 2] delay of the direct path
    wobble_amp: np.ndarray | None   # [n_mic, 2]
    events: list = field(default_factory=list)    # [(t_start, t_end)] in seconds
    noise: np.ndarray | None = None  # [n_mic, n_out] hall + microphone noise (full-scale units)
    gain: np.ndarray | None = None   # [n_out] slow gain wander
    real: np.ndarray | None = None   # [n_out, n_real_mics] float32 real background excerpt (overlay clips)
    shm: list | None = None          # where the big arrays sit while they travel in shared memory (gpu_pipeline._to_slot)


def _pack_response(r: object) -> tuple:
    if hasattr(r, "_f"):                                   # background.MeasuredResponse: a table read off a recording
        return ("table", np.asarray(r._f, np.float64), np.asarray(r._g, np.float64), float(r.gain_db))
    return ("resp", np.array([float(getattr(r, k)) for k in RESP_FIELDS], np.float64))


def make_plan(tx: np.ndarray, hw: Hardware, scene, n_out: int, rng: np.random.Generator, pre: int, band: tuple,
              add_noise: bool = True, key_tags: tuple = KEY_TAGS, include_static: bool = True,
              noise_dtype: type = np.float64) -> ClipPlan:
    """Everything render(tx, hw, scene, n_out, rng, pre, band, ...) would use, random draws included.

    The random draws happen here in exactly render()'s order, so `rng` ends in the same state.
    """
    tx = np.asarray(tx)
    if tx.ndim != 2:
        raise ValueError("render_torch needs the core's 2-D speaker signal [n, n_out]")
    fs = hw.fs
    # render._route: column per speaker, swapped if the laptop's speakers are; silent columns play nothing
    cols = [0, 1 if tx.shape[1] > 1 else 0]
    if hw.swap_speakers:
        cols = cols[::-1]
    spk_col = tuple(c if np.any(tx[:, c]) else -1 for c in cols)
    if all(c < 0 for c in spk_col):
        raise ValueError("speaker signal is all zeros")
    n_total = tx.shape[0]
    pre_eff = pre - int(hw.latency_samples)
    if pre_eff < 2500 or n_total < pre + n_out + 64:
        raise ValueError("tx too short: need pre-roll >= 2500 + latency and pre + n_out + 64 samples")
    f_lo, f_hi = band
    f0 = 0.5 * (f_lo + f_hi)
    c = hw.c
    v_max = _max_speed(scene) if scene.scatterers else 0.0
    f_dmax = max(150.0, 1.25 * 2.0 * v_max * f_hi / c)
    half_width = 0.5 * (f_hi - f_lo) + f_dmax + 300.0
    dec = int(np.clip(np.floor(fs / (2.6 * half_width)), 1, 24))
    n_mic = hw.n_mic
    t_ctrl = np.asarray(scene.t, np.float64)
    n_c = len(t_ctrl)

    sc_list = scene.scatterers
    n_s = len(sc_list)
    pos = np.zeros((n_s, n_c, 3))
    rho = np.zeros((n_s, n_c), np.float32)
    nrm = np.zeros((n_s, n_c, 3), np.float32)
    use_spec = np.zeros(n_s, bool)
    spec_q = np.zeros(n_s, np.float32)
    spec_floor = np.ones(n_s, np.float32)
    key = np.zeros(n_s, bool)
    for i, sc in enumerate(sc_list):
        pos[i] = sc.pos
        r = np.asarray(sc.rho, np.float64)
        if r.ndim != 0 and r.shape[0] != n_c:
            raise ValueError("reflectivity track does not match the scene's time axis")
        rho[i] = r                                          # a constant stays constant under the spline
        if sc.normal is not None and sc.spec_q > 0:
            use_spec[i] = True
            nrm[i] = sc.normal
            spec_q[i], spec_floor[i] = sc.spec_q, sc.spec_floor
        key[i] = sc.tag in key_tags

    wobble = getattr(scene, "lid_wobble", None)
    direct_is_dynamic = wobble is not None and bool(np.any(wobble))
    lat_s = hw.latency_samples / fs
    rows_d, rows_a, n_max = [], [], 1
    for m in range(n_mic):
        for s in range(2):
            d = np.asarray(hw.static_delays[m][s], np.float64) + lat_s
            a = np.asarray(hw.static_amps[m][s], np.float64)
            if direct_is_dynamic:
                d, a = d[1:], a[1:]
            rows_d.append(d)
            rows_a.append(a)
            n_max = max(n_max, len(d))
    static_delays = np.zeros((n_mic, 2, n_max))
    static_amps = np.zeros((n_mic, 2, n_max))
    static_any = np.zeros((n_mic, 2), bool)
    for i, (d, a) in enumerate(zip(rows_d, rows_a)):
        m, s = divmod(i, 2)
        if len(d):
            static_delays[m, s, :len(d)] = d
            static_delays[m, s, len(d):] = d[0]             # padding: zero amplitude at the reference delay
            static_amps[m, s, :len(d)] = a
            static_any[m, s] = True
    wob = tau0 = amp0 = None
    if direct_is_dynamic:
        wob = np.asarray(wobble, np.float64)
        tau0 = np.array([[hw.static_delays[m][s][0] for s in range(2)] for m in range(n_mic)], np.float64)
        amp0 = np.array([[hw.static_amps[m][s][0] for s in range(2)] for m in range(n_mic)], np.float64)

    events = [(float(e["t_start"]), float(e["t_end"])) for e in (getattr(scene, "events", None) or [])]

    # ---- the random draws, in render()'s order: hum phase, then noise, then gain wander ----
    hum_ph = rng.uniform(0.0, 2 * np.pi, 2) if hw.hum_hz > 0.0 else np.zeros(2)
    noise = gain = None
    if add_noise:
        noise = _noise(hw, n_out, rng).astype(noise_dtype, copy=False)
        gain = _gain_track(hw, n_out, rng).astype(noise_dtype, copy=False)

    return ClipPlan(
        n_out=int(n_out), pre=int(pre), n_total=int(n_total), dec=dec, include_static=bool(include_static),
        add_noise=bool(add_noise), fs=int(fs), c=float(c), alpha=float(hw.absorption_db_per_m(f0)),
        k_amp=float(10.0 ** (hw.echo_ref_db / 20.0) * (0.3 * 0.3) ** hw.range_gamma), gamma=float(hw.range_gamma),
        spk=np.asarray(hw.speakers, np.float64), spk_axis=np.asarray(hw.spk_axis, np.float64),
        spk_n=float(hw.spk_n), spk_floor=float(hw.spk_floor),
        mic=np.asarray(hw.mics, np.float64), mic_axis=np.asarray(hw.mic_axis, np.float64),
        mic_n=float(hw.mic_n), mic_floor=float(hw.mic_floor), table_refl=float(hw.table_refl), spk_col=spk_col,
        direct_gain_db=float(hw.direct_gain_db), adc_bits=int(hw.adc_bits or 0), swap_mics=bool(hw.swap_mics),
        clock_ppm=float(hw.clock_ppm), latency=int(hw.latency_samples), hum_hz=float(hw.hum_hz), hum_db=float(hw.hum_db),
        hum_ph=np.asarray(hum_ph, np.float64), static_delays=static_delays, static_amps=static_amps, static_any=static_any,
        spk_resp=[_pack_response(r) for r in hw.spk_resp], mic_resp=[_pack_response(r) for r in hw.mic_resp],
        t_ctrl=t_ctrl, pos=pos, rho=rho, nrm=nrm, use_spec=use_spec, spec_q=spec_q, spec_floor=spec_floor, key=key,
        wobble=wob, wobble_tau0=tau0, wobble_amp=amp0, events=events, noise=noise, gain=gain,
    )

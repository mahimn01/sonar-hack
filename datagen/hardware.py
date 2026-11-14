"""hardware.py — one random "virtual laptop + room" per training example.

Plain words: we do not know the exact acoustics of the demo laptop (where the
speakers point, how the microphones respond at 20 kHz, how loud the hall is),
and they change with the room anyway. So instead of guessing one laptop we draw
a different plausible laptop + room for every synthetic example. A model that
copes with all of them treats the real laptop as just one more draw. This is
"domain randomisation".

Coordinate frame (metres), seen by the user sitting in front of the screen:
    x = to the user's RIGHT,  y = from the screen TOWARDS the user,  z = UP
    origin = middle of the hinge line.
Channel convention: microphone channel 0 = the user's LEFT microphone,
speaker channel 0 = the user's LEFT speaker. If the real laptop is wired the
other way round, set `swap_mics=True` / `swap_speakers=True` in the profile.

Read-only helper module: no files, no audio devices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .physics import air_absorption_db_per_m, speed_of_sound, unit


# --------------------------------------------------------------------------
# Frequency responses (speaker and microphone colouration near 17-23 kHz)
# --------------------------------------------------------------------------
@dataclass
class Response:
    """A smooth random gain-vs-frequency curve, normalised to 0 dB at `f_ref`.

    tilt      : steady slope in dB per kHz (laptop speakers usually fall with f)
    shelf_*   : a step somewhere in the band (the demo laptop measured ~10 dB
                less at 19-20 kHz than at 17-18 kHz)
    ripple_*  : comb-like ripple from the grille / enclosure / port
    lp_*      : final low-pass (codec decimation filter close to Nyquist)
    delay_s   : extra pure delay (group delay of the transducer)
    """

    f_ref: float = 19000.0
    tilt_db_per_khz: float = 0.0
    shelf_f: float = 19000.0
    shelf_db: float = 0.0
    shelf_width: float = 500.0
    ripple_db: float = 0.0
    ripple_period: float = 1500.0
    ripple_phase: float = 0.0
    lp_f: float = 23500.0
    lp_order: float = 8.0
    delay_s: float = 0.0
    gain_db: float = 0.0
    phase_rad: float = 0.0

    def _mag_db(self, f: np.ndarray) -> np.ndarray:
        d = self.tilt_db_per_khz * (f - self.f_ref) / 1000.0
        d = d + self.shelf_db / (1.0 + np.exp(-(f - self.shelf_f) / max(self.shelf_width, 1.0)))
        d = d + self.ripple_db * np.cos(2 * np.pi * f / self.ripple_period + self.ripple_phase)
        d = d - 10.0 * np.log10(1.0 + (np.maximum(f, 0.0) / self.lp_f) ** (2 * self.lp_order))
        return d

    def __call__(self, f: np.ndarray) -> np.ndarray:
        """Complex gain at frequencies `f` (Hz)."""
        f = np.asarray(f, dtype=float)
        db = self._mag_db(f) - self._mag_db(np.array([self.f_ref]))[0] + self.gain_db
        mag = 10.0 ** (np.clip(db, -120.0, 40.0) / 20.0)
        return mag * np.exp(-2j * np.pi * f * self.delay_s + 1j * self.phase_rad)


# --------------------------------------------------------------------------
# The virtual laptop + room
# --------------------------------------------------------------------------
@dataclass
class Hardware:
    fs: int = 48000
    temp_c: float = 21.0
    rh_pct: float = 45.0

    # geometry
    speakers: np.ndarray = field(default_factory=lambda: np.array([[-0.12, 0.02, 0.005], [0.12, 0.02, 0.005]]))
    spk_axis: np.ndarray = field(default_factory=lambda: np.array([[0.0, 0.35, 0.94], [0.0, 0.35, 0.94]]))
    spk_n: float = 2.0  # beam sharpness (0 = omni)
    spk_floor: float = 0.15  # leakage far off-axis (linear)
    mics: np.ndarray = field(default_factory=lambda: np.array([[-0.035, -0.08, 0.215], [0.035, -0.08, 0.215]]))
    mic_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.94, 0.34]))
    mic_n: float = 0.5
    mic_floor: float = 0.3
    table_refl: float = 0.5  # amplitude of the echo that bounces off the desk before the mic (0 = off)

    # which speaker plays what: "both" (same signal), "left", "right", or
    # "stereo" (TX has 2 columns: column 0 -> left speaker, column 1 -> right)
    tx_route: str = "both"
    swap_mics: bool = False
    swap_speakers: bool = False

    # echo strength: echo-to-direct ratio (dB) of a unit reflector (a palm
    # facing the laptop) at 0.3 m from speaker and mic, on-axis
    echo_ref_db: float = -20.0
    range_gamma: float = 1.0  # amplitude ~ 1/(Rtx*Rrx)^gamma

    # direct speaker->mic leakage and static room/desk reflections,
    # per (mic, speaker): arrays of delays (s) and amplitudes (direct = ~1)
    static_delays: list = field(default_factory=list)
    static_amps: list = field(default_factory=list)

    spk_resp: list = field(default_factory=list)  # one Response per speaker
    mic_resp: list = field(default_factory=list)  # one Response per mic

    # converter / levels / noise
    direct_gain_db: float = -65.0  # direct speaker->mic leakage: RX amplitude / TX amplitude (measured -73 dB at 36% volume)
    adc_bits: int = 16  # 16 or 24; 0 = no quantisation (float)
    noise_dbfs: float = -92.0  # white mic+ADC noise, total over 0..fs/2
    babble_dbfs: float = -60.0  # audible hall noise (speech/music-like), total
    hf_burst_rate: float = 0.5  # broadband ultrasonic clicks per second (keys, claps, zips)
    hf_burst_db: float = -15.0  # peak level of those clicks relative to the direct leakage
    tones: list = field(default_factory=list)  # narrowband interferers: (f_hz, level_db_re_direct, drift_hz_per_s)
    hum_hz: float = 0.0  # mains ripple on the speaker amp (50/60 Hz or its double); 0 = none
    hum_db: float = -40.0  # level of the resulting side lines relative to the tone
    gain_drift_db: float = 0.5  # slow AGC-like gain wander (rms dB)
    gain_step_db: float = 0.0  # one sudden gain step somewhere in the clip (0 = none)
    clock_ppm: float = 0.0  # ADC clock vs DAC clock mismatch (parts per million)
    latency_samples: int = 0  # constant playback+record latency left in the recording (whole samples)

    # ---- derived --------------------------------------------------------
    @property
    def c(self) -> float:
        return speed_of_sound(self.temp_c)

    def absorption_db_per_m(self, f_hz: float) -> float:
        return air_absorption_db_per_m(f_hz, self.temp_c, self.rh_pct)

    @property
    def n_spk(self) -> int:
        return int(self.speakers.shape[0])

    @property
    def n_mic(self) -> int:
        return int(self.mics.shape[0])

    def summary(self) -> dict:
        """Small JSON-friendly description, stored next to each example."""
        return {
            "temp_c": round(self.temp_c, 2),
            "mic_spacing_m": round(float(np.linalg.norm(self.mics[0] - self.mics[-1])), 4),
            "spk_spacing_m": round(float(np.linalg.norm(self.speakers[0] - self.speakers[-1])), 4),
            "tx_route": self.tx_route,
            "echo_ref_db": round(self.echo_ref_db, 1),
            "direct_gain_db": round(self.direct_gain_db, 1),
            "noise_dbfs": round(self.noise_dbfs, 1),
            "babble_dbfs": round(self.babble_dbfs, 1),
            "adc_bits": self.adc_bits,
            "clock_ppm": round(self.clock_ppm, 2),
            "table_refl": round(self.table_refl, 2),
            "n_tones": len(self.tones),
        }


# --------------------------------------------------------------------------
# Randomisation ranges. Edit these numbers to widen/narrow the virtual world.
# --------------------------------------------------------------------------
@dataclass
class Profile:
    """Ranges for the random virtual laptop. Defaults are a wide guess around a
    15-inch laptop with deck speakers and two bezel microphones (HP ENVY x360)."""

    band_hz: tuple = (17000.0, 21000.0)  # where the probe signal lives (set by the adapter)
    tx_route: str | None = None  # None = random; otherwise force "both" / "left" / "right" / "stereo"
    swap_mics: bool = False
    swap_speakers: bool = False

    mic_spacing: tuple = (0.035, 0.12)  # metres between the two microphones (unknown on the HP)
    mic_height: tuple = (0.19, 0.235)  # how far up the lid the mics sit
    lid_angle_deg: tuple = (95.0, 125.0)  # how far the screen is opened
    spk_half_spacing: tuple = (0.07, 0.15)
    spk_y: tuple = (0.0, 0.20)  # deck position: near hinge .. near front edge
    spk_tilt_deg: tuple = (0.0, 70.0)  # beam axis: straight up .. towards the user
    spk_n: tuple = (0.3, 5.0)
    spk_floor: tuple = (0.05, 0.35)

    echo_ref_db: tuple = (-30.0, 6.0)  # real recordings on other hardware show echoes STRONGER than the leakage
    range_gamma: tuple = (0.75, 1.05)
    table_refl: tuple = (0.0, 0.85)

    direct_gain_db: tuple = (-78.0, -40.0)  # demo laptop measured -73 dB at 36% volume; louder volume -> higher
    p_16bit: float = 0.6
    noise_dbfs: tuple = (-100.0, -86.0)  # demo laptop measured about -98 dBFS in a quiet room
    babble_dbfs: tuple = (-75.0, -30.0)  # -40 dBFS is roughly an 80 dB SPL crowd
    hf_burst_rate: tuple = (0.0, 2.0)
    hf_burst_db: tuple = (-35.0, 0.0)
    p_tone: float = 0.2
    p_hum: float = 0.3  # seen in real recordings: steady lines at +-50 Hz beside the tone
    hum_db: tuple = (-50.0, -24.0)
    gain_drift_db: tuple = (0.0, 1.2)
    p_gain_step: float = 0.08
    clock_ppm: tuple = (-30.0, 30.0)
    p_clock_drift: float = 0.3  # speaker and mic are separate Windows devices; drift is possible
    rt60: tuple = (0.06, 0.35)  # ultrasonic reverberation is short (air absorption)
    latency_samples: tuple = (0, 1900)  # unknown audio-stack latency; the core receiver must find it itself


def _u(rng: np.random.Generator, lo_hi: tuple) -> float:
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _sample_response(rng: np.random.Generator, band: tuple, kind: str) -> Response:
    f_lo, f_hi = band
    f_ref = 0.5 * (f_lo + f_hi)
    if kind == "speaker":
        # speakers roll off towards 20 kHz and often have one abrupt step
        tilt = rng.uniform(-5.0, 1.5)
        shelf_db = rng.uniform(-18.0, 2.0) if rng.random() < 0.7 else 0.0  # measured: ~10 dB less above ~18.8 kHz
        ripple = rng.uniform(0.0, 4.0)
        lp_f = rng.uniform(21500.0, 26000.0)
    else:
        # MEMS mics rise towards their resonance (20-30 kHz); codec filter cuts near Nyquist
        tilt = rng.uniform(-2.5, 3.0)
        shelf_db = rng.uniform(-6.0, 3.0) if rng.random() < 0.3 else 0.0
        ripple = rng.uniform(0.0, 2.0)
        lp_f = rng.uniform(20800.0, 23800.0)
    return Response(
        f_ref=f_ref,
        tilt_db_per_khz=tilt,
        shelf_f=rng.uniform(18300.0, 19600.0) if kind == "speaker" else rng.uniform(f_lo - 500.0, f_hi + 500.0),
        shelf_db=shelf_db,
        shelf_width=rng.uniform(120.0, 600.0),
        ripple_db=ripple,
        ripple_period=rng.uniform(400.0, 3500.0),
        ripple_phase=rng.uniform(0.0, 2 * np.pi),
        lp_f=lp_f,
        lp_order=rng.uniform(4.0, 14.0),
        delay_s=rng.uniform(0.0, 2.5e-4),
    )


def sample_hardware(rng: np.random.Generator, profile: Profile | None = None, fs: int = 48000) -> Hardware:
    """Draw one random virtual laptop + room."""
    p = profile or Profile()
    hw = Hardware(fs=fs)
    hw.temp_c = rng.uniform(17.0, 28.0)
    hw.rh_pct = rng.uniform(20.0, 70.0)

    # ---- geometry -------------------------------------------------------
    lid = np.deg2rad(_u(rng, p.lid_angle_deg))
    h_mic = _u(rng, p.mic_height)
    d_mic = _u(rng, p.mic_spacing)
    # the lid rotates about the hinge (x axis); 90 deg = vertical, more = leaning back
    mic_y, mic_z = h_mic * np.cos(lid), h_mic * np.sin(lid)
    x_off = rng.normal(0.0, 0.006)  # array not perfectly centred
    hw.mics = np.array([[x_off - d_mic / 2, mic_y, mic_z], [x_off + d_mic / 2, mic_y, mic_z]])
    hw.mic_axis = unit(np.array([0.0, np.sin(lid), -np.cos(lid)]))  # screen normal, towards the user
    hw.mic_n = rng.uniform(0.0, 1.5)
    hw.mic_floor = rng.uniform(0.15, 0.6)

    sx = _u(rng, p.spk_half_spacing)
    sy = _u(rng, p.spk_y)
    asym = rng.normal(0.0, 0.008, size=2)
    hw.speakers = np.array([[-sx + asym[0], sy, 0.004], [sx + asym[1], sy + rng.normal(0, 0.004), 0.004]])
    tilt = np.deg2rad(_u(rng, p.spk_tilt_deg))
    splay = np.deg2rad(rng.normal(0.0, 8.0, size=2))  # speakers rarely point exactly alike
    hw.spk_axis = unit(
        np.array(
            [
                [np.sin(splay[0]), np.sin(tilt), np.cos(tilt)],
                [np.sin(splay[1]), np.sin(tilt), np.cos(tilt)],
            ]
        )
    )
    hw.spk_n = _u(rng, p.spk_n)
    hw.spk_floor = _u(rng, p.spk_floor)
    hw.table_refl = max(0.0, _u(rng, p.table_refl)) if rng.random() < 0.8 else 0.0

    hw.tx_route = p.tx_route or str(rng.choice(["both", "both", "left", "right"]))
    hw.swap_mics = p.swap_mics
    hw.swap_speakers = p.swap_speakers

    hw.echo_ref_db = _u(rng, p.echo_ref_db)
    hw.range_gamma = _u(rng, p.range_gamma)

    # ---- direct leakage + static reflections -----------------------------
    c = hw.c
    rt60 = _u(rng, p.rt60)
    n_early = int(rng.integers(2, 7))
    n_late = int(rng.integers(8, 36))
    hw.static_delays, hw.static_amps = [], []
    for m in range(hw.n_mic):
        row_d, row_a = [], []
        for s in range(hw.n_spk):
            d0 = np.linalg.norm(hw.speakers[s] - hw.mics[m]) / c
            a0 = 10.0 ** (rng.uniform(-3.0, 3.0) / 20.0)  # direct path, the reference (~1.0)
            early_d = d0 + rng.uniform(0.1e-3, 3.0e-3, n_early)
            early_a = 10.0 ** (rng.uniform(-22.0, -4.0, n_early) / 20.0)
            late_t = rng.uniform(3.0e-3, min(0.08, rt60 * 0.9), n_late)
            late_a = 10.0 ** ((-18.0 - 60.0 * late_t / rt60 + rng.normal(0, 4.0, n_late)) / 20.0)
            row_d.append(np.concatenate([[d0], early_d, d0 + late_t]))
            row_a.append(np.concatenate([[a0], early_a, late_a]) * rng.choice([-1.0, 1.0], 1 + n_early + n_late))
        hw.static_delays.append(row_d)
        hw.static_amps.append(row_a)
        # keep the direct tap itself positive so "direct = 1" stays meaningful
        for s in range(hw.n_spk):
            hw.static_amps[m][s][0] = abs(hw.static_amps[m][s][0])

    hw.spk_resp = [_sample_response(rng, p.band_hz, "speaker") for _ in range(hw.n_spk)]
    base_mic = _sample_response(rng, p.band_hz, "mic")
    hw.mic_resp = []
    for _ in range(hw.n_mic):
        r = Response(**vars(base_mic))  # the two mics are the same part: small mismatch only
        r.gain_db = rng.normal(0.0, 1.0)
        r.phase_rad = rng.normal(0.0, 0.15)
        r.delay_s = base_mic.delay_s + rng.normal(0.0, 4e-6)
        hw.mic_resp.append(r)

    # ---- levels and noise -------------------------------------------------
    hw.direct_gain_db = _u(rng, p.direct_gain_db)
    hw.adc_bits = 16 if rng.random() < p.p_16bit else 24
    hw.noise_dbfs = _u(rng, p.noise_dbfs)
    hw.babble_dbfs = _u(rng, p.babble_dbfs)
    hw.hf_burst_rate = _u(rng, p.hf_burst_rate) if rng.random() < 0.5 else 0.0
    hw.hf_burst_db = _u(rng, p.hf_burst_db)
    hw.tones = []
    if rng.random() < p.p_tone:
        for _ in range(int(rng.integers(1, 3))):
            hw.tones.append(
                (
                    float(rng.uniform(p.band_hz[0] - 1500.0, min(p.band_hz[1] + 1500.0, fs / 2 - 500.0))),
                    float(rng.uniform(-55.0, -15.0)),
                    float(rng.normal(0.0, 2.0)),
                )
            )
    if rng.random() < p.p_hum:
        hw.hum_hz = float(rng.choice([60.0, 60.0, 120.0, 50.0]))
        hw.hum_db = _u(rng, p.hum_db)
    hw.gain_drift_db = _u(rng, p.gain_drift_db)
    hw.gain_step_db = float(rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 6.0)) if rng.random() < p.p_gain_step else 0.0
    hw.clock_ppm = _u(rng, p.clock_ppm) if rng.random() < p.p_clock_drift else 0.0
    hw.latency_samples = int(rng.integers(p.latency_samples[0], p.latency_samples[1] + 1))
    from . import calibration

    calibration.adjust_hardware(rng, hw)  # a laptop fitted to real recordings, if one is active (else nothing, no draws)
    return hw


def clean_hardware(fs: int = 48000, tx_route: str = "both") -> Hardware:
    """A fixed, quiet, well-behaved laptop: used by tests and by physics checks."""
    hw = Hardware(fs=fs)
    hw.tx_route = tx_route
    hw.spk_n, hw.spk_floor = 0.0, 1.0
    hw.mic_n, hw.mic_floor = 0.0, 1.0
    hw.table_refl = 0.0
    hw.echo_ref_db = -20.0
    c = hw.c
    hw.static_delays = [
        [np.array([np.linalg.norm(hw.speakers[s] - hw.mics[m]) / c]) for s in range(hw.n_spk)] for m in range(hw.n_mic)
    ]
    hw.static_amps = [[np.array([1.0]) for _ in range(hw.n_spk)] for _ in range(hw.n_mic)]
    hw.spk_resp = [Response() for _ in range(hw.n_spk)]
    hw.mic_resp = [Response() for _ in range(hw.n_mic)]
    hw.direct_gain_db = -30.0
    hw.adc_bits = 0
    hw.noise_dbfs = -110.0
    hw.babble_dbfs = -200.0
    hw.hf_burst_rate = 0.0
    hw.gain_drift_db = 0.0
    hw.clock_ppm = 0.0
    return hw

"""config.py — every signal and processing parameter of the sonar in one place.

Plain words: this file is the single source of truth for "what do we play and
how do we listen". Recording, training, the simulator and the realtime engine
all read the same `SonarConfig`, and every recording stores a copy of it, so a
model is never run on features computed with different settings.

The signal (see DESIGN.md for the reasoning):
  * one pilot tone per speaker channel (left 18.5 kHz, right 19.0 kHz) -> Doppler for gestures
  * one periodic chirp 19.5-21.5 kHz on the left channel               -> range-gated echoes
    for presence and breathing
  * everything repeats exactly every `frame_len` samples (20 ms), so one stored
    period can be looped forever with no clicks.

Read-only module: no files, no audio devices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any

import numpy as np

# Classes of the gesture model. "none" is never sent to the UI as a gesture.
GESTURE_LABELS: tuple[str, ...] = ("none", "swipe_left", "swipe_right", "push", "pull")

# Labels a recording session may contain that are not gestures. The dataset
# builder maps all of them to "none" for the gesture model; they are also used
# to tune / check the presence and breathing detectors.
SCENE_LABELS: tuple[str, ...] = (
    "empty",            # nobody within ~2 m
    "present_still",    # a person sitting or standing still
    "walk_in",          # a person walking up to the laptop
    "walk_out",         # a person leaving
    "distractor",       # typing, reaching, drinking, people passing behind ...
    "breathing",        # paced breathing with a known rate (ground truth in metadata)
)

SPEED_OF_SOUND = 343.0  # m/s at ~20 C


@dataclass(frozen=True)
class SonarConfig:
    # ------------------------------------------------------------------ timing
    fs: int = 48000                 # sample rate (Hz); the laptop rejects 96 kHz
    frame_len: int = 960            # one TX period = one processing frame = 20 ms -> 50 frames/s

    # ---------------------------------------------------------------- transmit
    n_out: int = 2                  # speaker channels (0 = user's left, 1 = user's right)
    tone_hz: tuple[float, ...] = (18500.0, 19000.0)   # pilot tones; must be multiples of fs/frame_len
    tone_channels: tuple[int, ...] = (0, 1)           # tone i is played on this speaker channel
    tone_amp: float = 0.35          # digital peak amplitude of each tone (1.0 = full scale)
    chirp_lo_hz: float = 19500.0    # chirp band (inclusive), multiples of fs/frame_len
    chirp_hi_hz: float = 21500.0
    chirp_amp: float = 0.50         # digital peak amplitude of the chirp
    chirp_channel: int = 0          # speaker channel that plays the chirp
    fade_ms: float = 60.0           # raised-cosine fade in/out when the stream starts/stops (no clicks)

    # ----------------------------------------------------------------- receive
    n_mics: int = 2                 # microphone channels (0 = user's left)
    prefilter_atten_db: float = 100.0    # high-pass that removes speech/music before anything else
    prefilter_transition_hz: float = 2500.0
    prefilter_margin_hz: float = 600.0   # passband starts this far below the lowest tone

    # ------------------------------------------------- tone path (gesture Doppler)
    decim: int = 24                 # 48 kHz -> 2 kHz complex baseband per tone
    demod_pass_hz: float = 300.0    # keep +-300 Hz around each tone
    demod_atten_db: float = 90.0
    clutter_hp_hz: float = 5.0      # removes the static carrier (direct path + furniture)
    stft_win: int = 128             # 64 ms Hann window at 2 kHz
    stft_fft: int = 256             # zero-padded -> 7.8 Hz per bin
    doppler_bins: int = 64          # bins kept: -250 Hz .. +242 Hz (about +-2.3 m/s)

    # ------------------------------------------ features fed to the gesture model
    feat_floor_margin_db: float = 3.0    # dB above the noise floor that maps to 0
    feat_range_db: float = 36.0          # dB above that which maps to 1
    gesture_frames: int = 80             # model input length: 80 frames = 1.6 s

    # ------------------------------------------ chirp path (range profile / CIR)
    cir_fft: int = 128              # taps per 20 ms; one tap = 7.5 samples = 5.4 cm of path
    n_taps: int = 96                # taps kept (about 2.6 m of range)
    direct_tap: int = 4             # where frame sync parks the direct speaker->mic path
    cir_window: str = "hann"        # spectral window (sidelobes of the huge direct path)
    sync_frames: int = 16           # frames averaged for the first coarse alignment
    ref_frames: int = 50            # frames used to learn the reference channel / clock drift
    drift_loop_hz: float = 0.05     # bandwidth of the clock-drift tracking loop
    speaker_mic_distance_m: float = 0.25   # only used to print ranges in metres

    # -------------------------------------------------------- gesture segmenter
    seg_band_hz: tuple[float, float] = (20.0, 235.0)   # Doppler band that counts as "hand motion"
    seg_on_db: float = 1.5          # in-band power this many dB above the noise floor opens a segment
    seg_off_db: float = 0.7         # ... and below this it closes
    seg_min_peak_db: float = 3.0    # bursts that never get this strong are ignored
    seg_min_frames: int = 6         # shorter bursts are ignored (0.12 s)
    seg_max_frames: int = 70        # longer motion is cut and classified anyway (1.4 s)
    seg_hang_frames: int = 10       # quiet frames that end a segment (0.2 s) = decision latency
    seg_refractory_frames: int = 20 # dead time after any accepted gesture (0.4 s)
    seg_opposite_lockout_frames: int = 60   # ignore the "hand coming back" gesture for 1.2 s
    impulse_outer_hz: float = 200.0 # clicks/claps light up |f| above this as much as below
    impulse_ratio: float = 0.6      # outer-band excess / inner-band excess that marks a click

    # ------------------------------------------------------------------ presence
    presence_max_range_m: float = 2.0     # "the room" = this far from the laptop
    presence_near_m: float = 0.6          # hand / gesture zone
    presence_enter_s: float = 0.3         # gross motion this long -> "entering"
    presence_settle_s: float = 1.0        # quiet this long after motion -> "present"
    presence_absent_s: float = 5.0        # nothing moves this long after somebody walked -> "empty" (they left)
    presence_motionless_s: float = 40.0   # nothing moves this long with NO walking before -> "empty" (give up)
    presence_doppler_thresh_db: float = 6.0   # pilot-tone movement energy that also counts as gross movement
    presence_fast_hp_hz: float = 7.0      # echo changes faster than this are gross movement (breathing is < 3 Hz)
    presence_slow_tau_s: float = 4.0      # slow background time constant for micro motion
    presence_fast_thresh_db: float = 8.0  # dB above the calibrated floor
    presence_slow_thresh_db: float = 7.0

    # ----------------------------------------------------------------- breathing
    breath_fs: float = 10.0               # slow-time rate of the breathing analysis (Hz)
    breath_range_m: tuple[float, float] = (0.25, 1.6)   # where a chest may be
    breath_band_hz: tuple[float, float] = (0.1, 0.6)    # 6 .. 36 breaths per minute
    breath_min_s: float = 10.0            # shortest still period before a first estimate
    breath_max_s: float = 30.0            # longest analysis window
    breath_update_s: float = 1.0          # how often a new estimate is produced
    breath_min_conf: float = 0.45         # below this the bpm is reported as null

    # ------------------------------------------------- physics-only classifier
    heur_w_speakers: float = 1.0          # sign/weight of the two-speaker direction cue (-1 if L/R speakers are swapped)
    heur_w_mics: float = 1.0              # sign/weight of the two-microphone direction cue (-1 if mic channels are swapped)

    # ------------------------------------------------------------------ derived
    @property
    def frame_rate(self) -> float:
        """Frames per second (50)."""
        return self.fs / self.frame_len

    @property
    def bin_hz(self) -> float:
        """Spacing of the TX comb lines = FFT bin spacing of one frame (50 Hz)."""
        return self.fs / self.frame_len

    @property
    def chirp_bins(self) -> np.ndarray:
        """FFT bin numbers of the chirp lines."""
        lo = int(round(self.chirp_lo_hz / self.bin_hz))
        hi = int(round(self.chirp_hi_hz / self.bin_hz))
        return np.arange(lo, hi + 1)

    @property
    def chirp_center_hz(self) -> float:
        return 0.5 * (self.chirp_lo_hz + self.chirp_hi_hz)

    @property
    def chirp_bandwidth_hz(self) -> float:
        return self.chirp_hi_hz - self.chirp_lo_hz + self.bin_hz

    @property
    def bb_fs(self) -> float:
        """Sample rate of the per-tone complex baseband (2 kHz)."""
        return self.fs / self.decim

    @property
    def stft_hop(self) -> int:
        """Baseband samples per frame (40) -> one Doppler column per frame."""
        return self.frame_len // self.decim

    @property
    def n_tones(self) -> int:
        return len(self.tone_hz)

    @property
    def n_power_channels(self) -> int:
        """Doppler power pictures: one per (tone, microphone)."""
        return self.n_tones * self.n_mics

    @property
    def has_lateral(self) -> bool:
        """With two microphones there is one extra picture: sideways direction from the mic pair."""
        return self.n_mics >= 2

    @property
    def n_feat_channels(self) -> int:
        """Channels of the gesture feature image fed to the model."""
        return self.n_power_channels + (1 if self.has_lateral else 0)

    @property
    def doppler_hz(self) -> np.ndarray:
        """Centre frequency of each kept Doppler bin, low (receding) to high (approaching)."""
        half = self.doppler_bins // 2
        return np.arange(-half, half) * (self.bb_fs / self.stft_fft)

    @property
    def tap_samples(self) -> float:
        """Delay per CIR tap in audio samples (7.5)."""
        return self.frame_len / self.cir_fft

    @property
    def tap_path_m(self) -> float:
        """Extra travel distance per CIR tap (5.4 cm)."""
        return SPEED_OF_SOUND * self.tap_samples / self.fs

    @property
    def range_m(self) -> np.ndarray:
        """Approximate one-way distance (m) of each kept CIR tap from the laptop.

        A reflector at distance r adds about 2r of travel; the direct path
        (speaker straight to mic) already covers `speaker_mic_distance_m`.
        """
        extra = (np.arange(self.n_taps) - self.direct_tap) * self.tap_path_m
        return np.maximum(0.0, 0.5 * (extra + self.speaker_mic_distance_m))

    @property
    def range_resolution_m(self) -> float:
        """Smallest range separation the chirp can resolve, c / 2B (about 6.7 cm)."""
        return SPEED_OF_SOUND / (2.0 * self.chirp_bandwidth_hz)

    def wavelength_m(self, f_hz: float | None = None) -> float:
        return SPEED_OF_SOUND / (self.chirp_center_hz if f_hz is None else f_hz)

    # ------------------------------------------------------------- validation
    def validate(self) -> None:
        """Raise ValueError if the numbers do not fit together."""
        b = self.bin_hz
        for f in (*self.tone_hz, self.chirp_lo_hz, self.chirp_hi_hz):
            if abs(f / b - round(f / b)) > 1e-9:
                raise ValueError(f"{f} Hz is not a multiple of fs/frame_len = {b} Hz")
        if self.frame_len % self.decim:
            raise ValueError("frame_len must be a multiple of decim")
        if len(self.tone_hz) != len(self.tone_channels):
            raise ValueError("tone_hz and tone_channels need the same length")
        if max((*self.tone_channels, self.chirp_channel)) >= self.n_out:
            raise ValueError("a TX component is routed to a speaker channel that does not exist")
        if self.chirp_hi_hz >= self.fs / 2 or self.chirp_lo_hz >= self.chirp_hi_hz:
            raise ValueError("bad chirp band")
        tones = sorted(self.tone_hz)
        span = self.doppler_bins / 2 * self.bb_fs / self.stft_fft
        for a, c in zip(tones, tones[1:]):
            if c - a < 2 * span - 1e-6:
                raise ValueError("tones are closer than twice the Doppler span; their echoes would overlap")
        if min(abs(self.chirp_lo_hz - t) for t in tones) < 2 * span - 1e-6:
            raise ValueError("chirp band is too close to a tone")
        if len(self.chirp_bins) > self.cir_fft:
            raise ValueError("cir_fft must be at least the number of chirp lines")
        peak = [0.0] * self.n_out
        for ch in self.tone_channels:
            peak[ch] += self.tone_amp
        peak[self.chirp_channel] += self.chirp_amp
        if max(peak) > 0.98:
            raise ValueError(f"TX would clip: per-channel peak {peak}")

    # ------------------------------------------------------------ (de)serialise
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SonarConfig":
        known = {f.name: f for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if k not in known:
                continue  # written by a newer/older version; ignore unknown keys
            kwargs[k] = tuple(v) if isinstance(v, list) else v
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def with_(self, **changes: Any) -> "SonarConfig":
        cfg = replace(self, **changes)
        cfg.validate()
        return cfg


# ---------------------------------------------------------------------------
# Named profiles. Record, train and run with the SAME profile.
#   default  tones 18.5/19.0 kHz, chirp 19.5-21.5 kHz. Inaudible to almost everyone.
#   strong   everything 500 Hz lower: ~5 dB more signal on the demo laptop, but an
#            18.0 kHz tone is audible to some people under 25.
#   low      fallback if the self-test shows too little signal above 19 kHz
#            (17.0/17.5 kHz tones are audible to many young people).
#   clean    one component per speaker channel (left = tone, right = chirp): no
#            intermodulation at all, at the price of the two-speaker direction cue.
#   mono     everything from one speaker channel (mono devices).
# ---------------------------------------------------------------------------
_PROFILES: dict[str, dict[str, Any]] = {
    "default": {},
    "strong": {
        "tone_hz": (18000.0, 18500.0),
        "chirp_lo_hz": 19000.0,
        "chirp_hi_hz": 21500.0,
    },
    "low": {
        "tone_hz": (17000.0, 17500.0),
        "chirp_lo_hz": 18000.0,
        "chirp_hi_hz": 20500.0,
    },
    "clean": {
        "tone_hz": (18500.0,),
        "tone_channels": (0,),
        "chirp_channel": 1,
        "tone_amp": 0.7,
        "chirp_amp": 0.8,
    },
    "mono": {
        "n_out": 1,
        "tone_channels": (0, 0),
        "chirp_channel": 0,
        "tone_amp": 0.22,
        "chirp_amp": 0.45,
    },
}


def get_profile(name: str = "default", **overrides: Any) -> SonarConfig:
    """Return a validated config for a named profile, with optional overrides."""
    if name not in _PROFILES:
        raise KeyError(f"unknown profile {name!r}; choose from {sorted(_PROFILES)}")
    cfg = SonarConfig(**{**_PROFILES[name], **overrides})
    cfg.validate()
    return cfg


def default_config() -> SonarConfig:
    return get_profile("default")


def profile_names() -> list[str]:
    return sorted(_PROFILES)

"""heuristics.py — gesture rules that need NO training data.

Plain words
-----------
Three physical facts are enough to tell the main gestures apart:

1. Towards or away?  A hand moving towards the laptop raises the pitch of the
   echo, a hand moving away lowers it. Which side of the tone lights up during
   the fast stroke tells push from pull.

2. Left or right, from the two SPEAKERS.  The left speaker plays one tone and
   the right speaker another. A hand travelling to the right is moving away
   from the left speaker and towards the right one, so the right speaker's
   tone is shifted up more than the left speaker's. The difference between the
   two pitch shifts, added up over the stroke, is positive for "right" and
   negative for "left" wherever the hand is. (Speakers are ~24 cm apart: a
   strong cue that survives even with one microphone.)

3. Left or right, from the two MICROPHONES.  Same idea with the two mics, which
   are only a few cm apart: the phase difference between the mic channels keeps
   turning one way while the hand travels sideways. Weaker than (2) but
   independent of it, and it works for a single-speaker signal.

Why this file exists: it is a safety net that works before any model is
trained, an automatic first-pass labeller for continuous real recordings, and
a yardstick for how much of the problem is physics alone. The core package has
its own physics classifier; this one adds the two-microphone phase cue and is
used here to check the simulator.

Works for steady-tone probes (it zooms in on each tone). Read-only module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quicklook as ql


@dataclass
class Cues:
    """Per-frame measurements derived from the microphone channels."""

    times: np.ndarray  # [T] seconds
    hop_s: float
    energy: np.ndarray  # [T] moving-echo energy (linear), all tones and mics
    fast: np.ndarray  # [T] the part of that energy above ~45 Hz of shift (fast motion only)
    floor: float  # quiet level of `energy`
    fast_floor: float
    centroid_hz: np.ndarray  # [T] average pitch shift: + approaching, - receding
    mic_lateral_hz: np.ndarray  # [T] turn rate of the mic-to-mic phase: + = moving to the user's right
    mic_coherence: np.ndarray  # [T] 0..1, how trustworthy mic_lateral_hz is
    spk_lateral_hz: np.ndarray  # [T] right-speaker shift minus left-speaker shift: + = moving right


def measure(
    rx: np.ndarray,
    fs: int,
    tones: list,
    tone_speakers: list | None = None,
    f_min: float = 20.0,
    f_fast: float = 45.0,
    span_hz: float = 240.0,
    win_s: float = 0.064,
    hop_s: float = 0.008,
    highpass_hz: float = 14.0,
) -> Cues:
    """Compute the physical cues from raw microphone audio and the probe tones.

    tone_speakers : for each tone, 0 if the LEFT speaker plays it, 1 if the RIGHT one.
    highpass_hz   : everything slower than this (breathing, body sway) is ignored.
    """
    tone_speakers = list(tone_speakers) if tone_speakers is not None else [0] * len(tones)
    f_ref = float(tones[0])
    total = fast = num = cross_step = None
    per_tone_c, per_tone_e = [], []
    for f_c in tones:
        bb = ql.remove_static(ql.to_baseband(rx, fs, float(f_c)), cutoff_hz=highpass_hz, order=3)
        spec, freqs, times = ql.stft(bb, ql.BB_RATE, win_s, hop_s)
        keep = (np.abs(freqs) <= span_hz) & (np.abs(freqs) >= f_min)
        spec, f = spec[keep], freqs[keep]
        p = (np.abs(spec) ** 2).sum(axis=2)  # [F, T] summed over mics
        e = p.sum(axis=0)
        c = (f[:, None] * p).sum(axis=0) / (e + 1e-30) * (f_ref / float(f_c))  # as if every tone were f_ref
        per_tone_c.append(c)
        per_tone_e.append(e)
        total = e if total is None else total + e
        fe = p[np.abs(f) >= f_fast].sum(axis=0)
        fast = fe if fast is None else fast + fe
        num = c * e if num is None else num + c * e
        if spec.shape[2] >= 2:
            # phase difference between the mics per Doppler bin, then how it turns frame to frame
            x = spec[:, :, 1] * np.conj(spec[:, :, 0])
            step = (x[:, 1:] * np.conj(x[:, :-1])).sum(axis=0)
            cross_step = step if cross_step is None else cross_step + step
    n_t = len(total)
    centroid = num / (total + 1e-30)

    mic_lat, mic_coh = np.zeros(n_t), np.zeros(n_t)
    if cross_step is not None:
        # average the turning phasor over ~40 ms: lone noisy frames point anywhere,
        # frames that contain a real echo agree with their neighbours
        k = max(1, int(round(0.04 / hop_s)))
        smooth = np.convolve(cross_step, np.ones(k), mode="same")
        raw = np.convolve(np.abs(cross_step), np.ones(k), mode="same") + 1e-30
        mic_lat[1:] = np.angle(smooth) / (2 * np.pi * hop_s)
        mic_coh[1:] = np.abs(smooth) / raw

    spk_lat = np.full(n_t, np.nan)
    left = [i for i, s in enumerate(tone_speakers) if s == 0]
    right = [i for i, s in enumerate(tone_speakers) if s == 1]
    if left and right:
        def avg(idx: list) -> np.ndarray:
            w = np.array([per_tone_e[i] for i in idx])
            return (np.array([per_tone_c[i] for i in idx]) * w).sum(axis=0) / (w.sum(axis=0) + 1e-30)

        spk_lat = avg(right) - avg(left)
    return Cues(
        times=times,
        hop_s=hop_s,
        energy=total,
        fast=fast,
        floor=float(np.percentile(total, 10)) + 1e-30,
        fast_floor=float(np.percentile(fast, 10)) + 1e-30,
        centroid_hz=centroid,
        mic_lateral_hz=mic_lat,
        mic_coherence=mic_coh,
        spk_lateral_hz=spk_lat,
    )


@dataclass
class Decision:
    label: str
    confidence: float
    t_start: float
    t_end: float
    mic_cycles: float  # mic-to-mic phase turns over the stroke, + = to the user's right
    spk_cycles: float  # speaker-to-speaker shift difference added up, + = right (nan if one speaker)
    radial_cycles: float  # net approach (+) or retreat (-) in wavelengths of path
    peak_hz: float  # pitch shift at the strongest moment of the stroke
    snr_db: float


def classify(
    cues: Cues,
    snr_thr_db: float = 10.0,
    min_stroke_s: float = 0.06,
    max_half_s: float = 0.45,
    mic_thr: float = 0.45,
    spk_thr: float = 1.6,
    radial_thr_hz: float = 35.0,
    mic_sign: float = 1.0,
    spk_sign: float = 1.0,
) -> Decision:
    """Rule-based decision for one window that contains at most one gesture.

    The rules look only at the FAST stroke (the moment of largest fast-motion
    energy) so that the slower hand-raise before a gesture and the hand
    dropping afterwards are not mistaken for it.

    snr_thr_db    : how far the fast-motion energy must rise above the quiet level
    mic_thr       : mic-pair phase turns needed to call the stroke sideways
    spk_thr       : speaker-pair cycles needed to call the stroke sideways
    radial_thr_hz : pitch shift needed to call the stroke a push or pull
    mic_sign / spk_sign : set to -1.0 if channel 0 is really the RIGHT mic / speaker
    """
    t, hop = cues.times, cues.hop_s
    k = max(1, int(round(0.05 / hop)))
    fast = np.convolve(cues.fast, np.ones(k) / k, mode="same")
    k0 = int(np.argmax(fast))
    snr = 10.0 * np.log10(fast[k0] / cues.fast_floor + 1e-30)
    none = Decision("none", 0.0, 0.0, 0.0, 0.0, float("nan"), 0.0, 0.0, float(snr))
    if snr < snr_thr_db:
        return none
    # grow the stroke outwards from the peak while the fast energy stays above 12% of it
    lim = int(round(max_half_s / hop))
    a = b = k0
    while a > 0 and k0 - a < lim and fast[a - 1] > 0.12 * fast[k0]:
        a -= 1
    while b < len(fast) - 1 and b - k0 < lim and fast[b + 1] > 0.12 * fast[k0]:
        b += 1
    if (b - a + 1) * hop < min_stroke_s:
        return none
    sl = slice(a, b + 1)
    w = cues.energy[sl] / (cues.energy[sl].sum() + 1e-30)

    radial = float(np.sum(cues.centroid_hz[sl]) * hop)
    good = cues.mic_coherence[sl] > 0.5
    mic_cycles = mic_sign * float(np.sum(cues.mic_lateral_hz[sl] * good) * hop)
    have_spk = not np.all(np.isnan(cues.spk_lateral_hz))
    spk_cycles = spk_sign * float(np.nansum(cues.spk_lateral_hz[sl]) * hop) if have_spk else float("nan")
    c = cues.centroid_hz[sl]
    peak = float(c[int(np.argmax(np.abs(c) * cues.energy[sl]))])
    mean_abs = float(np.sum(np.abs(c) * w))

    # sideways evidence, each cue measured in units of its own threshold
    z_mic = mic_cycles / mic_thr
    z_spk = spk_cycles / spk_thr if have_spk else 0.0
    z = z_spk + 0.5 * z_mic if have_spk else z_mic
    # a push/pull changes the path length a lot and one-sidedly; a swipe goes in-then-out
    one_sided = abs(radial) / (mean_abs * (b - a + 1) * hop + 1e-9)  # 1 = all approach or all retreat
    conf_e = float(np.clip((snr - snr_thr_db) / 15.0, 0.1, 1.0))
    args = (float(t[a]), float(t[b]), mic_cycles, spk_cycles, radial, peak, float(snr))
    if abs(z) >= 1.0 and not (one_sided > 0.8 and abs(z) < 2.0):
        return Decision("swipe_right" if z > 0 else "swipe_left", conf_e * float(np.clip(abs(z) / 3.0, 0.3, 1.0)), *args)
    if abs(peak) >= radial_thr_hz:
        return Decision("push" if peak > 0 else "pull", conf_e * float(np.clip(one_sided, 0.3, 1.0)), *args)
    return Decision("none", 0.0, *args)


def classify_audio(
    rx: np.ndarray, fs: int, tones: list, tone_speakers: list | None = None, span_hz: float = 240.0, **kw
) -> Decision:
    """Convenience: raw microphone audio in, rule-based decision out.

    span_hz: how far from each tone to look. Keep it below half the spacing
    between neighbouring probe components, or one tone's echo is mistaken for
    another's (the core signal has 500 Hz spacing -> 240 Hz).
    """
    return classify(measure(rx, fs, tones, tone_speakers, span_hz=span_hz), **kw)

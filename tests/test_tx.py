"""The transmitted waveform: seamless, band-limited, no clipping."""

import numpy as np
import pytest

from sonar.audio import LiveAudio
from sonar.config import get_profile, profile_names
from sonar.tx import fade_ramp, make_reference, make_tx_frame, tx_samples


@pytest.mark.parametrize("name", profile_names())
def test_profiles_are_valid_and_do_not_clip(name):
    cfg = get_profile(name)
    frame = make_tx_frame(cfg)
    assert frame.shape == (cfg.frame_len, cfg.n_out)
    assert np.abs(frame).max() <= 0.98


def test_no_energy_below_the_ultrasonic_band(cfg):
    frame = make_tx_frame(cfg).astype(np.float64)
    spec = np.abs(np.fft.rfft(np.tile(frame, (8, 1)), axis=0))          # 8 periods in a row, as played
    f = np.fft.rfftfreq(8 * cfg.frame_len, 1 / cfg.fs)
    audible = spec[f < min(cfg.tone_hz) - 100].max()
    assert 20 * np.log10(audible / spec.max()) < -110                   # float32 rounding only: silent to the ear


def test_loop_is_seamless_and_lines_sit_on_the_50hz_grid(cfg):
    ref = make_reference(cfg)
    spec = np.abs(np.fft.rfft(ref.frame.astype(np.float64), axis=0)).sum(axis=1)
    used = set(ref.tone_bins) | set(ref.chirp_bins)
    unused = np.array([k for k in range(len(spec)) if k not in used])
    assert spec[unused].max() < 1e-4 * spec.max()                       # nothing between the intended lines
    assert ref.crest_factor < 2.2                                       # chirp close to constant envelope
    np.testing.assert_array_equal(tx_samples(cfg, 950, 20, ref.frame)[10:], ref.frame[:10])


def test_tones_are_on_separate_speakers_and_chirp_is_flat(cfg):
    ref = make_reference(cfg)
    spec = np.abs(np.fft.rfft(ref.frame.astype(np.float64), axis=0))
    assert spec[ref.tone_bins[0], 0] > 100 * spec[ref.tone_bins[0], 1]  # tone 0 only on the left channel
    assert spec[ref.tone_bins[1], 1] > 100 * spec[ref.tone_bins[1], 0]  # tone 1 only on the right channel
    lines = spec[ref.chirp_bins, cfg.chirp_channel]
    assert lines.max() / lines.min() < 1.001                            # perfectly flat chirp spectrum


def test_live_backend_plays_the_loop_sample_exactly_with_fades(cfg):
    """The output callback must continue the period exactly across calls of any size."""
    audio = LiveAudio.__new__(LiveAudio)                                # no devices needed for this check
    audio.cfg = cfg
    audio._frame = make_tx_frame(cfg)
    audio._tx_pos = 0
    audio._fade_total = audio._fade_in_left = int(cfg.fade_ms * 1e-3 * cfg.fs)
    audio._fade_out_left = None
    audio._program = None
    audio.idle_silent = False
    chunks = []
    for n in (333, 1920, 7, 960, 4096):
        out = np.zeros((n, cfg.n_out), dtype=np.float32)
        audio._fill(out, n)
        chunks.append(out)
    played = np.concatenate(chunks)
    expect = tx_samples(cfg, 0, len(played), audio._frame).copy()
    k = audio._fade_total
    expect[:k] *= fade_ramp(k)[:, None]
    np.testing.assert_allclose(played, expect, atol=1e-6)
    audio._fade_out_left = audio._fade_total                            # now stop: fades to exact silence
    out = np.zeros((k + 500, cfg.n_out), dtype=np.float32)
    audio._fill(out, len(out))
    assert np.abs(out[:10]).max() > 0 and np.abs(out[k:]).max() == 0.0

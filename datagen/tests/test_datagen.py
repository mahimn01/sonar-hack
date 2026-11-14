"""test_datagen.py — quick checks that the synthetic-echo generator obeys the physics.

What it checks (no hardware, no audio devices, about 20 seconds):
  1. air absorption matches the ISO 9613 table value;
  2. a reflector approaching at a known speed gives the textbook Doppler shift;
  3. a reflector moving to the user's right turns the mic-to-mic phase the
     "right" way by the amount geometry predicts, and the right speaker's tone is
     shifted up more than the left speaker's;
  4. an empty room produces no moving energy (the renderer adds no artefacts);
  5. a simulated chest breathing at 15 breaths/min shows up at 0.25 Hz;
  6. the same seed gives the same audio;
  7. with the core `sonar/` package present, a generated clip comes out in the
     exact model-input shape the core defines.

How to run (from the repo root):
    cd ~/Documents/Dev/sonar-hack
    datagen/.venv/bin/python -m pytest datagen/tests -q

Read-only: writes nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from datagen import adapter as ad
from datagen import heuristics as hz
from datagen import quicklook as ql
from datagen.hardware import Profile, clean_hardware, sample_hardware
from datagen.kinematics import Path, Scatterer, Scene, time_axis
from datagen.physics import air_absorption_db_per_m
from datagen.render import PRE_ROLL, render
from datagen.scenes import LABELS, sample_scene

FS = 48000


def _stereo_tones(n: int) -> np.ndarray:
    """18.0 kHz on the left speaker, 18.5 kHz on the right (same layout as the core signal)."""
    t = np.arange(n) / FS
    return np.stack([0.35 * np.cos(2 * np.pi * 18000.0 * t), 0.35 * np.cos(2 * np.pi * 18500.0 * t)], axis=1)


def test_air_absorption_matches_iso_table() -> None:
    # ISO 9613-2 table: 76.6 dB/km at 8 kHz, 20 C, 70 % relative humidity
    assert 1000 * air_absorption_db_per_m(8000.0, 20.0, 70.0) == pytest.approx(76.6, rel=0.03)
    assert 0.4 < air_absorption_db_per_m(20000.0, 20.0, 50.0) < 0.65


def test_doppler_of_approaching_reflector() -> None:
    dur, f_tone, speed = 2.0, 19000.0, 0.5
    n = int(FS * dur)
    tx = 0.5 * np.sin(2 * np.pi * f_tone * np.arange(PRE_ROLL + n + 480) / FS)
    hw = clean_hardware(FS, tx_route="left")
    t = time_axis(dur)
    pos = np.stack([np.full_like(t, -0.12), 1.4 - speed * t, np.full_like(t, 0.1)], axis=1)
    out = render(tx, hw, Scene(t=t, scatterers=[Scatterer(pos, 1.0)]), n, np.random.default_rng(0), band=(f_tone, f_tone))
    power, freqs, times, _ = ql.doppler_spectrogram(out.rx, FS, f_tone, win_s=0.128)
    measured = float(np.median(freqs[np.argmax(power[:, :, 0], axis=0)][len(times) // 4 : 3 * len(times) // 4]))

    def path_len(tq: np.ndarray) -> np.ndarray:
        p = np.stack([np.full_like(tq, -0.12), 1.4 - speed * tq, np.full_like(tq, 0.1)], axis=1)
        return np.linalg.norm(p - hw.speakers[0], axis=1) + np.linalg.norm(p - hw.mics[0], axis=1)

    mid = times[len(times) // 4 : 3 * len(times) // 4]
    expected = float(np.median(-(f_tone / hw.c) * (path_len(mid + 1e-3) - path_len(mid - 1e-3)) / 2e-3))
    assert expected > 40.0  # approaching = positive shift (about 2*v*f/c = 55 Hz)
    assert measured == pytest.approx(expected, abs=8.0)  # one frequency bin


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_left_right_cues_have_the_right_sign(direction: float) -> None:
    dur = 2.0
    n = int(FS * dur)
    tx = _stereo_tones(PRE_ROLL + n + 480)
    hw = clean_hardware(FS, tx_route="stereo")
    t = time_axis(dur)
    path = Path(t, np.array([-0.2 * direction, 0.3, 0.15]), t0=0.8)
    path.move(np.array([0.2 * direction, 0.3, 0.15]), 0.4)
    scene = Scene(t=t, scatterers=[Scatterer(path.pos, 1.0)])
    out = render(tx, hw, scene, n, np.random.default_rng(0), band=(18000.0, 18500.0))
    d = hz.classify_audio(out.rx, FS, [18000.0, 18500.0], [0, 1])

    def mic_diff(p: np.ndarray) -> float:
        return float(np.linalg.norm(p - hw.mics[0]) - np.linalg.norm(p - hw.mics[1]))

    theory = (18000.0 / hw.c) * (mic_diff(path.pos[-1]) - mic_diff(path.pos[0]))  # + for a move to the right
    assert np.sign(d.mic_cycles) == direction and np.sign(d.spk_cycles) == direction
    assert d.mic_cycles == pytest.approx(theory, rel=0.35)
    assert d.label == ("swipe_right" if direction > 0 else "swipe_left")


def test_empty_room_has_no_moving_energy() -> None:
    dur = 1.5
    n = int(FS * dur)
    tx = _stereo_tones(PRE_ROLL + n + 480)
    hw = sample_hardware(np.random.default_rng(3), Profile(band_hz=(18000.0, 18500.0)), FS)
    hw.tx_route, hw.clock_ppm, hw.adc_bits = "stereo", 0.0, 0
    out = render(tx, hw, Scene(t=time_axis(dur)), n, np.random.default_rng(0), band=(18000.0, 18500.0), add_noise=False)
    bb = ql.to_baseband(out.rx, FS, 18000.0)
    spec_all, freqs, _ = ql.stft(bb)  # still contains the static tone at 0 Hz
    spec_mov, _, _ = ql.stft(ql.remove_static(bb, cutoff_hz=10.0, order=3))
    doppler_band = (np.abs(freqs) >= 20.0) & (np.abs(freqs) <= 240.0)  # where gestures live
    direct = np.abs(spec_all[np.argmin(np.abs(freqs)), 10:-10]).mean()
    residue = np.abs(spec_mov[doppler_band][:, 10:-10]).max()
    assert 20 * np.log10(residue / direct) < -70.0


def test_breathing_shows_up_at_the_breathing_rate() -> None:
    dur, bpm = 40.0, 15.0
    n = int(FS * dur)
    tx = 0.5 * np.sin(2 * np.pi * 19000.0 * np.arange(PRE_ROLL + n + 480) / FS)
    hw = clean_hardware(FS, tx_route="left")
    rng = np.random.default_rng(5)
    scene = sample_scene(rng, "present", dur, distractors=False, breathing_only=True, breath_bpm=bpm, breath_mm=3.0,
                         distance=0.6, sway_mm=0.2)
    out = render(tx, hw, scene, n, rng, band=(19000.0, 19000.0))
    bb = ql.to_baseband(out.rx[:, :1], FS, 19000.0, out_rate=100)[200:-200, 0]
    bb = bb - bb.mean()
    # the echo phasor swings along an arc; look along the direction in which it moves most
    xy = np.stack([bb.real, bb.imag])
    series = np.linalg.eigh(np.cov(xy))[1][:, -1] @ xy
    spec = np.abs(np.fft.rfft(series * np.hanning(len(series))))
    f = np.fft.rfftfreq(len(series), 1 / 100)
    band = (f > 0.08) & (f < 0.7)
    assert f[band][np.argmax(spec[band])] * 60.0 == pytest.approx(bpm, abs=1.5)
    assert scene.meta["breath_bpm"] == bpm


def test_same_seed_same_audio_and_all_labels_render() -> None:
    dur = 1.0
    n = int(FS * dur)
    tx = _stereo_tones(PRE_ROLL + n + 480)
    outs = []
    for _ in range(2):
        rng = np.random.default_rng(42)
        hw = sample_hardware(rng, Profile(band_hz=(18000.0, 18500.0), tx_route="stereo"), FS)
        scene = sample_scene(rng, "push", dur, event_center=0.5)
        outs.append(render(tx, hw, scene, n, rng, band=(18000.0, 18500.0)).rx)
    assert np.array_equal(outs[0], outs[1])
    assert outs[0].shape == (n, 2) and np.isfinite(outs[0]).all() and np.abs(outs[0]).max() <= 1.0
    rng = np.random.default_rng(1)
    for label in LABELS:  # every class must at least build a scene
        scene = sample_scene(rng, label, 2.0, lead_in=0.5)
        assert scene.t[0] <= 0.0 and scene.t[-1] >= 2.5
        for ev in scene.events:
            assert ev["t_end"] > ev["t_start"]


def test_generated_clip_matches_the_core_model_input() -> None:
    core = ad.load_core()
    if not core.available or core.module("gestures") is None:
        pytest.skip("core sonar/ package not importable: nothing to match against")
    from datagen import generate

    res = generate.make_clip((123, "push", None, 6.0, True))
    assert res["error"] is None, res["error"]
    assert res["audio"].shape == (int(core.fs * (generate.LEAD_IN + generate.STORY)), core.cfg.n_mics)
    cfg = core.cfg
    for ex in res["examples"]:
        assert ex["window"].shape == (cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames)
        assert -1.0 <= float(ex["window"].min()) and float(ex["window"].max()) <= 1.0  # the lateral channel is signed
        assert ex["label"] in ("push", "none")
        if "padded" in ex:  # the core's training picture carries `pad` spare frames on each side
            pad = ad.train_pad(core)
            assert ex["padded"].shape == (cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames + 2 * pad)
            assert np.allclose(ex["padded"][..., pad : pad + cfg.gesture_frames], ex["window"])

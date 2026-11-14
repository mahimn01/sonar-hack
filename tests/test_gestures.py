"""Gestures: visible spectrogram signature, burst detection, direction cues, click rejection."""

import numpy as np
import pytest

from sonar.dataset import Augmenter
from sonar.gestures import GestureSegmenter, heuristic_classify
from sonar.sim import EchoSimulator, gesture_stroke_window, person_reflectors

from .helpers import classify_bursts, gesture_scene, run_pipeline, strongest

T0 = 6.0


@pytest.fixture(scope="module")
def runs(cfg):
    """Each scripted gesture pushed through the whole pipeline once (shared by the tests below)."""
    out = {}
    for label in ("swipe_left", "swipe_right", "push", "pull"):
        pipe, frames = run_pipeline(cfg, EchoSimulator(cfg, seed=5), gesture_scene(label, T0), seconds=T0 + 4.0)
        out[label] = (frames, classify_bursts(cfg, frames))
    return out


@pytest.mark.parametrize("label", ["swipe_left", "swipe_right", "push", "pull"])
def test_gesture_gives_a_visible_spectrogram_signature(cfg, runs, label):
    frames, bursts = runs[label]
    a, b = gesture_stroke_window(label, T0)
    idle = np.array([fr.activity_db for fr in frames if fr.feat is not None and 2.0 < fr.t < T0 - 0.2])
    stroke = [fr for fr in frames if fr.feat is not None and a <= fr.t <= b + 0.1]
    peak_activity = max(fr.activity_db for fr in stroke)
    peak_feature = max(float(fr.feat[:cfg.n_power_channels].max()) for fr in stroke)
    assert np.percentile(np.abs(idle), 99) < 2.0             # before the gesture: noise only
    assert peak_activity > 4.5                               # the burst stands well above the 3 dB gesture bar ...
    assert peak_feature > 0.4                                # ... and its ridge is > 17 dB above it
    assert bursts, "no movement burst was detected"
    seg = strongest(bursts)[0]
    assert seg.start / cfg.frame_rate < b and seg.end / cfg.frame_rate > a   # the burst overlaps the stroke


@pytest.mark.parametrize("label", ["swipe_left", "swipe_right", "push", "pull"])
def test_physics_classifier_names_the_gesture(cfg, runs, label):
    _, bursts = runs[label]
    _, guess, conf, _, _ = strongest(bursts)
    assert guess == label and conf > 0.6


def test_both_direction_cues_flip_sign_with_swipe_direction(cfg, runs):
    """Two-speaker cue and two-microphone cue: negative = moving right, positive = moving left."""
    _, _, _, spk_l, mic_l = strongest(runs["swipe_left"][1])
    _, _, _, spk_r, mic_r = strongest(runs["swipe_right"][1])
    assert spk_l > 10.0 and spk_r < -10.0                    # Hz
    assert mic_l > 2.0 and mic_r < -2.0
    _, _, _, spk_p, mic_p = strongest(runs["push"][1])
    assert abs(spk_p) < 6.0 and abs(mic_p) < 1.5             # a push is not sideways


def test_two_microphone_cue_alone_is_enough(cfg):
    """Profile 'clean' has ONE tone (no speaker cue): direction must come from the mic pair."""
    from sonar.config import get_profile

    clean = get_profile("clean")
    for label in ("swipe_left", "swipe_right"):
        _, frames = run_pipeline(clean, EchoSimulator(clean, seed=6), gesture_scene(label, T0), seconds=T0 + 4.0)
        _, guess, _, spk, mic = strongest(classify_bursts(clean, frames))
        assert spk == 0.0 and guess == label and (mic > 2.0) == (label == "swipe_left")


def test_mirror_augmentation_turns_left_into_right(cfg, runs):
    frames, _ = runs["swipe_left"]
    fr = max((f for f in frames if f.segment is not None), key=lambda f: f.segment.peak_db)
    aug = Augmenter(cfg, ["none", "swipe_left", "swipe_right", "push", "pull"], pad=0, seed=0)
    mirrored, y = aug.mirror(fr.window, 1)
    assert y == 2 and heuristic_classify(mirrored, cfg).label == "swipe_right"
    reversed_, y = aug.reverse(fr.window, 1)
    assert y == 2 and reversed_.shape == fr.window.shape
    assert heuristic_classify(aug.mirror(*aug.mirror(fr.window, 1))[0], cfg).label == "swipe_left"


def test_still_person_and_clicks_do_not_trigger(cfg):
    """10 s of a seated, breathing person with keyboard-like clicks added: no burst may open."""
    sim = EchoSimulator(cfg, seed=7)
    rx = sim.render(10 * cfg.fs, person_reflectors(0.65, seed=7))
    rng = np.random.default_rng(1)
    for t in (4.0, 5.3, 6.1, 7.7, 8.2):                      # 2 ms broadband clicks, 40 dB above our own signal
        i = int(t * cfg.fs)
        rx[i:i + 96] += 0.01 * rng.standard_normal((96, cfg.n_mics)).astype(np.float32)
    from sonar.pipeline import Pipeline

    pipe = Pipeline(cfg)
    frames = pipe.process(rx)
    assert sum(fr.impulse for fr in frames) >= 5             # the clicks were seen and flagged ...
    assert not any(fr.segment for fr in frames)              # ... and nothing was reported


def test_segmenter_history_window_blanks_old_frames(cfg):
    seg = GestureSegmenter(cfg)
    quiet = np.zeros((cfg.n_power_channels, cfg.doppler_bins), dtype=np.float32)
    for i in range(1, 121):
        seg.push(i, np.full((cfg.n_feat_channels, cfg.doppler_bins), i, dtype=np.float32), quiet)
    w = seg.window(120)
    assert w.shape == (cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames)
    assert w[0, 0, -1] == 120 and w[0, 0, 0] == 41
    w = seg.window(120, not_before=100)
    assert np.all(w[:, :, :59] == 0) and w[0, 0, 59] == 100
    assert seg.window(121) is None

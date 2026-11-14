"""The whole learning chain on simulated recordings: record -> dataset -> train -> ONNX -> classify."""

import json

import numpy as np
import pytest

from sonar import record
from sonar.classifier import OnnxClassifier, PhysicsClassifier, load_classifier
from sonar.dataset import PAD, Augmenter, build_dataset, center_crop, find_sessions, load_session, session_examples
from sonar.features import clip_to_window, stream_features, window_from_stream
from sonar.pipeline import Pipeline
from sonar.train import fit_temperature, main as train_main, remap_labels, report, split_by_group


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    """Three short simulated recording sessions (different 'people'), made with the real recorder CLI."""
    root = tmp_path_factory.mktemp("data")
    for person, seed in (("ann", 11), ("bob", 12), ("cy", 13)):
        code = record.main(["--person", person, "--backend", "sim", "--fast", "--script", "gestures", "--reps", "4",
                            "--out", str(root), "--seed", str(seed)])
        assert code == 0
    return root


@pytest.mark.slow
def test_recorder_writes_complete_sessions(cfg, sessions):
    folders = find_sessions(sessions)
    assert len(folders) == 3
    meta, audio, session_cfg = load_session(folders[0])
    assert session_cfg == cfg                                   # the exact signal config is stored with the audio
    assert audio.shape[1] == cfg.n_mics and abs(len(audio) - meta["n_samples"]) <= 4800
    gestures = [e for e in meta["events"] if e["kind"] == "gesture"]
    assert len(gestures) == 16 and {e["label"] for e in gestures} == {"swipe_left", "swipe_right", "push", "pull"}
    labels = [e["label"] for e in gestures]
    assert labels != sorted(labels)                             # trial order is shuffled
    assert all(0 < e["start"] < e["end"] <= meta["n_samples"] for e in gestures)
    assert meta["events"][0]["kind"] == "calibration" and meta["person"] in ("ann", "bob", "cy")


@pytest.mark.slow
def test_offline_features_equal_live_features(cfg, sessions):
    """The dataset builder must cut exactly the pictures the live engine classifies."""
    _, audio, _ = load_session(find_sessions(sessions)[0])
    audio = audio[:25 * cfg.fs]
    stream = stream_features(audio, cfg)
    live = []
    pipe = Pipeline(cfg)
    for i in range(0, len(audio), 1920):                        # the engine's block size
        live += [fr.window for fr in pipe.process(audio[i:i + 1920]) if fr.window is not None]
    assert len(live) == len(stream.segments) >= 2
    for k, seg in enumerate(stream.segments):
        offline = window_from_stream(stream, seg.decided, stream.not_before[k])
        np.testing.assert_allclose(offline, live[k], atol=1e-5)
        padded = window_from_stream(stream, seg.decided, stream.not_before[k], pad=PAD)
        np.testing.assert_allclose(center_crop(padded, cfg), offline, atol=1e-6)
    win, seg = clip_to_window(audio, cfg)
    assert win.shape == (cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames)


@pytest.mark.slow
def test_dataset_labels_and_split(cfg, sessions):
    data = build_dataset([sessions], verbose=False)
    labels = [str(s) for s in data["labels"]]
    assert labels[0] == "none" and data["X"].shape[1:] == (cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames + 2 * PAD)
    counts = np.bincount(data["y"], minlength=5)
    assert all(c >= 10 for c in counts[1:]) and counts[0] >= 12  # nearly every prompted gesture was found
    assert len(set(data["group"])) == 3
    train, val, how = split_by_group(data, None, 0.25, seed=0)
    assert not set(data["group"][train]) & set(data["group"][val]), "a session leaked into both sides"
    train, val, how = split_by_group(data, "bob", 0.25, seed=0)
    assert set(data["person"][val]) == {"bob"} and "bob" not in set(data["person"][train])
    only3 = remap_labels(data, ["swipe_left", "swipe_right", "push"])
    assert [str(s) for s in only3["labels"]] == ["none", "swipe_left", "swipe_right", "push"]
    assert np.sum(only3["y"] == 0) == counts[0] + counts[4]      # pulls became "none"
    # the report of one session names any prompt whose movement was not found
    _, rep = session_examples(find_sessions(sessions)[0])
    assert rep["gestures"] + rep["missed"] == 16


def test_augmenter_keeps_shape_and_physics(cfg):
    labels = ["none", "swipe_left", "swipe_right", "push", "pull"]
    aug = Augmenter(cfg, labels, seed=1)
    rng = np.random.default_rng(0)
    x = rng.random((cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames + 2 * PAD)).astype(np.float32)
    x[-1] = 2 * x[-1] - 1
    seen = set()
    for _ in range(200):
        out, y = aug(x, 1)
        assert out.shape == (cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames) and out.dtype == np.float32
        assert out[:cfg.n_power_channels].min() >= 0 and out[:cfg.n_power_channels].max() <= 1 and np.abs(out[-1]).max() <= 1
        seen.add(y)
    assert seen == {1, 2}                                        # mirroring swaps swipe_left <-> swipe_right
    assert {aug(x, 3)[1] for _ in range(50)} == {3}              # ... and never relabels a push
    np.testing.assert_array_equal(aug.crop(x, 0), center_crop(x, cfg))
    # a ridge at +78 Hz (bin 42) moves to +94 Hz when the gesture is done 1.2x faster
    ridge = np.zeros_like(x)
    ridge[0, 42, 20:70] = 1.0
    faster = aug.stretch(ridge, 1.2)
    assert int(np.argmax(faster[0].sum(axis=1))) == 44
    # a ridge at +78 Hz ends up at -78 Hz when the movement is played backwards
    assert int(np.argmax(aug.reverse(ridge, 3)[0][0].sum(axis=1))) == 22


def test_metrics_helpers():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    m, table = report(y, pred, ["none", "a", "b"])
    assert m["accuracy"] == pytest.approx(4 / 6, abs=1e-3) and m["confusion"][0] == [1, 1, 0]
    assert m["per_class"]["a"]["recall"] == 1.0 and "predicted" in table
    logits = np.random.default_rng(0).normal(size=(200, 3)) * 8
    labels = logits.argmax(axis=1)
    labels[:60] = (labels[:60] + 1) % 3                          # an over-confident model: 30 % wrong
    assert fit_temperature(logits, labels) > 1.5


@pytest.mark.slow
def test_train_export_and_run_the_onnx_model(cfg, sessions, tmp_path):
    out = tmp_path / "models"
    code = train_main(["--data", str(sessions), "--out", str(out), "--epochs", "12", "--device", "cpu", "--seed", "0"])
    assert code == 0
    meta = json.loads((out / "gesture.json").read_text())
    assert meta["labels"] == ["none", "swipe_left", "swipe_right", "push", "pull"]
    assert meta["input_shape"] == [cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames]
    assert meta["metrics"]["onnx_max_logit_diff"] < 1e-3 and meta["temperature"] >= 1.0
    assert meta["metrics"]["validation"]["accuracy"] >= 0.8      # held-out simulated session
    assert meta["heuristic"]["w_speakers"] > 0 and meta["heuristic"]["w_mics"] > 0   # both cues found, right sign
    assert meta["gesture_peak_db_p10"] > 5.0

    clf = load_classifier(out / "gesture.onnx", cfg)
    assert isinstance(clf, OnnxClassifier) and clf.mismatch == []
    data = build_dataset([sessions], verbose=False)
    x = center_crop(data["X"].astype(np.float32), cfg)
    names = [str(s) for s in data["labels"]]
    hits = sum(clf.predict(xi)[0] == names[yi] for xi, yi in zip(x, data["y"]))
    assert hits / len(x) >= 0.85
    label, conf, probs = clf.predict(x[0])
    assert abs(sum(probs.values()) - 1.0) < 1e-5 and 0.0 <= conf <= 1.0
    # --physics still picks up the cue signs measured during training
    physics = load_classifier(out / "gesture.onnx", cfg, force_physics=True)
    assert isinstance(physics, PhysicsClassifier) and physics.cfg.heur_w_mics == pytest.approx(meta["heuristic"]["w_mics"])

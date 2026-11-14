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


@pytest.mark.slow
def test_unreadable_session_is_skipped_not_fatal(sessions, tmp_path, capsys):
    """A session still being recorded or half copied (no audio yet) must not stop a long training run."""
    broken = tmp_path / "20261114-000000_dan_gestures"
    broken.mkdir()
    (broken / "session.json").write_text((find_sessions(sessions)[0] / "session.json").read_text())
    data = build_dataset([sessions, tmp_path], verbose=False)
    assert len(set(data["group"])) == 3                          # the three good sessions, nothing from the broken one
    assert "skipped session" in capsys.readouterr().err


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


def test_batch_augmenter_matches_numpy_augmenter(cfg):
    """The GPU (torch, whole batch) augmentation must compute exactly what dataset.Augmenter computes."""
    torch = pytest.importorskip("torch")
    from sonar.batch_augment import BatchAugmenter

    labels = ["none", "swipe_left", "swipe_right", "push"]
    na, ta = Augmenter(cfg, labels, seed=0), BatchAugmenter(cfg, labels)
    rng = np.random.default_rng(1)
    n, c, f, t = 6, cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames
    x = rng.random((n, c, f, t + 2 * PAD)).astype(np.float32)
    x[:, :-1] *= rng.random((n, c - 1, f, t + 2 * PAD)) > 0.5          # sparse power pictures, like real ones
    x[:, -1] = 2 * x[:, -1] - 1                                       # lateral picture in -1..1
    xt = torch.from_numpy(x)
    s = np.array([0.84, 0.9, 1.0, 1.07, 1.19, 1.1972])
    np.testing.assert_allclose(ta.stretch(xt.clone(), torch.tensor(s, dtype=torch.float32)).numpy(),
                               np.stack([na.stretch(x[i], float(s[i])) for i in range(n)]), atol=5e-5)
    shift = [-8, -3, 0, 2, 5, 8]
    np.testing.assert_array_equal(ta.crop(xt, torch.tensor(shift)).numpy(),
                                  np.stack([na.crop(x[i], shift[i]) for i in range(n)]))
    xc = np.stack([na.crop(x[i], 0) for i in range(n)])
    xct, y = torch.from_numpy(xc), np.array([0, 1, 2, 3, 1, 0])
    for name in ("mirror", "reverse"):
        got_x, got_y = getattr(ta, name)(xct, torch.tensor(y))
        ref = [getattr(na, name)(xc[i], int(y[i])) for i in range(n)]
        np.testing.assert_array_equal(got_x.numpy(), np.stack([r[0] for r in ref]))
        assert got_y.tolist() == [r[1] for r in ref]
    db = np.array([-9.0, -4.5, -0.3, 0.0, 2.2, 6.0])
    np.testing.assert_allclose(ta.gain(xct, torch.tensor(db, dtype=torch.float32)).numpy(),
                               np.stack([na.gain(xc[i], float(db[i])) for i in range(n)]), atol=1e-6)
    e = rng.exponential(size=(n, cfg.n_power_channels, f, t))
    ref = []
    for i in range(n):
        na.rng = type("FixedRng", (), {"exponential": lambda self, size, e=e[i]: e})()
        ref.append(na.noise(xc[i]))
    np.testing.assert_allclose(ta.noise(xct, torch.tensor(e, dtype=torch.float32)).numpy(), np.stack(ref), atol=1e-6)
    start, width, level = [0, 10, 33, 76, 50, 5], [1, 2, 3, 3, 1, 2], [0.1, 0.2, 0.25, 0.4, 0.33, 0.15]
    ref = xc.copy()
    for i in range(n):
        ref[i, :-1, :, start[i]:start[i] + width[i]] = np.maximum(ref[i, :-1, :, start[i]:start[i] + width[i]], level[i])
    got = ta.click(xct, torch.tensor(start), torch.tensor(width), torch.tensor(level, dtype=torch.float32))
    np.testing.assert_allclose(got.numpy(), ref, atol=1e-7)
    in_time, start, width = [True, False, True, False, True, False], [0, 0, 73, 59, 40, 30], [2, 1, 6, 4, 3, 2]
    ref = xc.copy()
    for i in range(n):
        if in_time[i]:
            ref[i, :, :, start[i]:start[i] + width[i]] = 0.0
        else:
            ref[i, :, start[i]:start[i] + width[i], :] = 0.0
    got = ta.mask(xct, torch.tensor(in_time), torch.tensor(start), torch.tensor(width))
    np.testing.assert_array_equal(got.numpy(), ref)


def test_batch_augmenter_one_microphone(cfg):
    """A MacBook records one microphone channel: 2 pictures, no lateral picture, no mirror."""
    torch = pytest.importorskip("torch")
    from sonar.batch_augment import BatchAugmenter

    mono = cfg.with_(n_mics=1)
    labels = ["none", "swipe_left", "swipe_right", "push"]
    na, ta = Augmenter(mono, labels, seed=0), BatchAugmenter(mono, labels)
    assert mono.n_feat_channels == 2 and not ta.can_mirror
    rng = np.random.default_rng(2)
    x = (rng.random((4, 2, mono.doppler_bins, mono.gesture_frames + 2 * PAD)) * (rng.random((4, 2, 1, 1)) > 0.3)).astype(np.float32)
    xt = torch.from_numpy(x)
    s = [0.86, 1.0, 1.12, 1.19]
    np.testing.assert_allclose(ta.stretch(xt.clone(), torch.tensor(s)).numpy(),
                               np.stack([na.stretch(x[i], s[i]) for i in range(4)]), atol=5e-5)
    xc = np.stack([na.crop(x[i], 0) for i in range(4)])
    got_x, got_y = ta.reverse(torch.from_numpy(xc), torch.tensor([1, 2, 0, 3]))
    np.testing.assert_array_equal(got_x.numpy(), np.stack([na.reverse(xc[i], 0)[0] for i in range(4)]))
    db = [-7.0, -1.0, 0.5, 5.0]
    np.testing.assert_allclose(ta.gain(torch.from_numpy(xc), torch.tensor(db)).numpy(),
                               np.stack([na.gain(xc[i], db[i]) for i in range(4)]), atol=1e-6)
    out, y = ta(xt, torch.tensor([1, 2, 0, 3]))
    assert out.shape == (4, 2, mono.doppler_bins, mono.gesture_frames) and out.min() >= 0 and out.max() <= 1


def test_batch_augmenter_full_recipe(cfg):
    torch = pytest.importorskip("torch")
    from sonar.batch_augment import BatchAugmenter

    labels = ["none", "swipe_left", "swipe_right", "push", "pull"]
    torch.manual_seed(0)
    aug = BatchAugmenter(cfg, labels)
    x = torch.rand(1200, cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames + 2 * PAD).half()
    x[:, -1] = 2 * x[:, -1] - 1
    y = torch.tensor([1, 3] * 600)
    out, y2 = aug(x, y)
    assert out.shape == (1200, cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames) and out.dtype == torch.float32
    p = out[:, :cfg.n_power_channels]
    assert p.min() >= 0 and p.max() <= 1 and out[:, -1].abs().max() <= 1
    swipes, pushes = y2[y == 1], y2[y == 3]
    assert set(swipes.tolist()) == {1, 2} and set(pushes.tolist()) == {3}      # mirror / reverse swap swipes only
    # left -> right needs exactly one of mirror (p 0.5) and reverse (p 0.2): 0.5 * 0.8 + 0.5 * 0.2 = 0.5
    assert abs((swipes == 2).float().mean().item() - 0.5) < 0.07


def test_split_options():
    data = {
        "y": np.array([0, 1, 0, 1, 0, 1, 2, 2]),
        "person": np.array(["ann", "ann", "bob", "bob", "cy", "cy", "synth-1", "synth-2"]),
        "group": np.array(["s1", "s1", "s2", "s2", "s3", "s3", "synth-1", "synth-2"]),
        "source": np.array(["gesture"] * 6 + ["synthetic"] * 2),
    }
    train, val, how = split_by_group(data, "ann,cy", 0.25, seed=0)
    assert sorted(set(data["person"][val])) == ["ann", "cy"] and "ann" not in set(data["person"][train])
    with pytest.raises(ValueError, match="people in the data: \\['ann', 'bob', 'cy'\\]"):   # synthetic ids not listed
        split_by_group(data, "dan", 0.25, seed=0)
    train, val, how = split_by_group(data, None, 0.0, seed=0)
    assert len(val) == 0 and len(train) == 8
    train, val, how = split_by_group(data, None, 0.3, seed=0)
    assert how.startswith("held-out sessions (") and "np.str_" not in how and set(data["source"][val]) == {"gesture"}


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

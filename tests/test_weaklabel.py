"""Weak labels: every burst with every witness, hard negatives, pseudo-labels, and the batched fast path.

The fast path (all bursts judged in one batched pass, the physics rules written with tensor operations)
must give what the original one-burst-at-a-time code gives; several tests here are that comparison.
"""

import json

import numpy as np
import pytest

from sonar import weaklabel
from sonar.batch_judge import BatchJudge, heuristic_batch
from sonar.classifier import OnnxClassifier
from sonar.config import GESTURE_LABELS, get_profile
from sonar.dataset import PAD, center_crop, find_sessions, merge, session_examples
from sonar.features import stream_features, window_from_stream
from sonar.gestures import heuristic_classify
from sonar.train import WEAK_SOURCES, example_weights, keep_weak_out_of_validation
from sonar.weaklabel import WeakConfig, all_bursts, build_tables, build_weak, burst_set_of, session_table, weak_labels, windows_from_stream

from .engine_helpers import camera_sessions, prompted_sessions, scripted_audio, small_model, span, stroke_times, write_session

SCRIPT = [(3.0, "swipe_right", 1.0), (6.5, "push", 1.0), (10.0, "swipe_left", 1.1), (13.5, "push", 1.2), (17.0, "swipe_right", 0.9)]


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    return prompted_sessions(tmp_path_factory)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return small_model(tmp_path_factory) / "gesture.onnx"


@pytest.fixture(scope="module")
def cam(tmp_path_factory):
    return camera_sessions(tmp_path_factory)


@pytest.fixture(scope="module")
def loose(cfg, tmp_path_factory):
    """A recording with gestures in it and no label at all (only the still start is marked)."""
    root = tmp_path_factory.mktemp("loose")
    audio, _ = scripted_audio(cfg, SCRIPT, 21.0, seed=9)
    write_session(root / "20261114-120000_table_engine", cfg, audio, [], person="table", script="engine")
    return root


def test_pictures_of_all_bursts_at_once_equal_the_engine_pictures(cfg):
    audio, _ = scripted_audio(cfg, SCRIPT[:3], 13.0, seed=1)
    stream = stream_features(audio, cfg)
    assert len(stream.segments) >= 3
    x = windows_from_stream(stream)
    assert x.shape == (len(stream.segments), cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames + 2 * PAD) and x.dtype == np.float16
    for k, seg in enumerate(stream.segments):
        one = window_from_stream(stream, seg.decided, stream.not_before[k], pad=PAD)
        np.testing.assert_array_equal(x[k], one.astype(np.float16))
    bs = burst_set_of(stream, "cpu", 0.0)
    np.testing.assert_array_equal(bs.windows()[1], window_from_stream(stream, stream.segments[1].decided, stream.not_before[1]).astype(np.float16))
    assert bs.frame_of_sample(960 * 10) == 11 and len(bs) == len(stream.segments)
    assert windows_from_stream(stream_features(audio[:cfg.fs], cfg)).shape[0] == 0       # one still second: no burst


@pytest.mark.parametrize("profile,mics", [("default", 2), ("clean", 2), ("default", 1), ("mono", 2)])
def test_batched_physics_rules_equal_the_original(profile, mics):
    """heuristic_batch (tensor operations, GPU-ready) against gestures.heuristic_classify, picture by picture."""
    torch = pytest.importorskip("torch")
    cfg = get_profile(profile).with_(n_mics=mics)
    rng = np.random.default_rng(3)
    n, c, f, t = 60, cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames
    x = np.zeros((n, c, f, t), np.float32)
    for i in range(n):                                              # ridges like real strokes, noise, clicks, and nothing at all
        kind = i % 6
        if kind == 5:
            continue
        for ch in range(cfg.n_power_channels):
            centre = 32 + rng.normal(0, 10) + 14 * np.sin(np.linspace(0, rng.uniform(1, 4), t) + rng.uniform(0, 6))
            rows = np.arange(f)[:, None]
            x[i, ch] = rng.uniform(0.1, 1.0) * np.exp(-0.5 * ((rows - centre[None, :]) / rng.uniform(1.5, 5)) ** 2)
            x[i, ch, :, :int(rng.integers(0, 40))] = 0.0
        if kind == 4:
            x[i, :cfg.n_power_channels] = rng.random((cfg.n_power_channels, f, t)) * 0.3
        if cfg.has_lateral:
            x[i, -1] = rng.uniform(-1, 1) * (x[i, 0] > 0.2) * rng.random((f, t))
    out = heuristic_batch(torch.from_numpy(x), cfg)
    ref = [heuristic_classify(w, cfg) for w in x]
    assert out["label"].tolist() == [GESTURE_LABELS.index(r.label) for r in ref]
    assert len({r.label for r in ref}) >= 3                          # the pictures really exercise several verdicts
    np.testing.assert_allclose(out["probs"].numpy(), np.stack([r.probs for r in ref]), atol=1e-6)
    np.testing.assert_allclose(out["conf"].numpy(), [r.confidence for r in ref], atol=1e-6)
    np.testing.assert_allclose(out["lateral_hz"].numpy(), [r.lateral_hz for r in ref], atol=1e-4)
    np.testing.assert_allclose(out["lateral_mic_hz"].numpy(), [r.lateral_mic_hz for r in ref], atol=1e-4)
    np.testing.assert_allclose(out["order_s"].numpy(), [r.order_s for r in ref], atol=1e-6)
    # half precision pictures (how they are kept in memory) give the verdicts of the same pictures in float32
    half = heuristic_batch(torch.from_numpy(x.astype(np.float16)), cfg)
    ref16 = [heuristic_classify(w.astype(np.float16).astype(np.float32), cfg) for w in x]
    assert half["label"].tolist() == [GESTURE_LABELS.index(r.label) for r in ref16]


@pytest.mark.slow
def test_batched_judge_equals_one_burst_at_a_time(cfg, sessions, model):
    pytest.importorskip("torch")
    folder = find_sessions(sessions)[0]
    bs, why = all_bursts([folder])[0]
    assert bs is not None, why
    x = bs.windows()
    judge = BatchJudge(cfg, model, device="cpu")
    assert judge.kind == "torch" and judge.has_model and "batched on cpu" in judge.describe()
    v = judge(x)
    one = OnnxClassifier(model, cfg)
    single = [one.predict(w.astype(np.float32)) for w in x]
    assert [v.labels[i] for i in v.model] == [s[0] for s in single]
    np.testing.assert_allclose(v.model_conf, [s[1] for s in single], atol=1e-4)
    np.testing.assert_allclose(v.model_probs, [[s[2][name] for name in v.labels] for s in single], atol=1e-4)
    ref = [heuristic_classify(w.astype(np.float32), judge.cfg) for w in x]
    assert [GESTURE_LABELS[i] for i in v.physics] == [r.label for r in ref]
    np.testing.assert_allclose(v.physics_probs, np.stack([r.probs for r in ref]), atol=1e-6)
    # the cue weights the trainer measured for this laptop are used, as the engine's physics fallback does
    meta = json.loads(model.with_suffix(".json").read_text())
    assert judge.cfg.heur_w_mics == pytest.approx(meta["heuristic"]["w_mics"]) and judge.temperature == meta["temperature"]
    # machines without PyTorch (the demo laptop): ONNX Runtime in batches + the original physics function, same answers
    plain = BatchJudge(cfg, model, device=None)
    assert plain.kind == "onnx" and "no PyTorch" in plain.describe()
    v2 = plain(x)
    assert v2.model.tolist() == v.model.tolist() and v2.physics.tolist() == v.physics.tolist()
    np.testing.assert_allclose(v2.model_probs, v.model_probs, atol=1e-3)
    assert BatchJudge(cfg, None, device="cpu").kind == "physics" and len(BatchJudge(cfg, None, "cpu")(x[:0]).physics) == 0
    # a model for another microphone count is refused with a reason, not run
    mono = BatchJudge(cfg.with_(n_mics=1), model, device="cpu")
    assert not mono.has_model and "expects pictures of shape" in mono.note


@pytest.mark.slow
def test_roles_are_the_plain_builders(cfg, sessions, cam):
    """The table must say exactly what dataset.session_examples does with every burst."""
    for folder in find_sessions(sessions) + find_sessions(cam):
        examples, report = session_examples(folder)
        table = session_table(folder, device="cpu")
        mine = sorted(b.label for b in table.bursts if b.role == "gesture")
        assert mine == sorted(e.label for e in examples if e.source == "gesture"), folder.name
        assert sum(1 for b in table.bursts if b.role == "none") == report["none"], folder.name
        assert len(table.missed) == report["missed"]
        for b in table.bursts:
            if b.role == "gesture":                                   # same picture, not only the same count
                assert any(e.label == b.label and np.array_equal(e.x.astype(np.float16), table.bset.x[b.k]) for e in examples)
    table = session_table(find_sessions(sessions)[0], device="cpu")
    assert all(b.zone.startswith("trial:") for b in table.bursts if b.role == "gesture") and table.classifier == "physics"
    assert all(b.label_source == "prompt" and b.label_conf == 1.0 for b in table.bursts if b.role == "gesture")
    with pytest.raises(ValueError, match="without the feature stream"):
        table.stream
    assert session_table(find_sessions(sessions)[0], device="cpu", keep_stream=True).stream.n_frames == table.bset.n_frames


@pytest.mark.slow
def test_whatever_fires_in_a_gesture_free_part_is_a_hard_negative(cfg, tmp_path, model):
    """Somebody "reaches around" (here: scripted strokes) while the session says: no gestures. The model fires: hard negatives."""
    audio, _ = scripted_audio(cfg, SCRIPT, 21.0, seed=4)
    write_session(tmp_path / "data" / "20261114-130000_dan_scene", cfg, audio, [span(cfg, "distractor", 2.6, 20.5)], person="dan", script="scene")
    data, rows, summary = build_weak([tmp_path / "data"], model, device="cpu", verbose=False)
    assert data is not None and summary["classifier"] == "model" and summary["by_tag"] == {"hardneg": summary["examples"]}
    assert summary["examples"] >= 4 and set(data["source"]) == {"hardneg"} and set(data["y"]) == {0}
    assert list(data["labels"]) == list(GESTURE_LABELS) and np.allclose(data["weight"], WeakConfig().hardneg_weight)
    assert data["X"].shape[1:] == (cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames + 2 * PAD) and data["X"].dtype == np.float16
    assert set(data["person"]) == {"dan"} and data["weight"].dtype == np.float32
    hard = [r for r in rows if r.get("weak_tag") == "hardneg"]
    assert all(r["zone"] == "free:distractor" and r["model"] != "none" and r["model_conf"] >= 0.6 for r in hard)
    # each one sits on a scripted stroke
    for r in hard:
        assert any(r["t0"] - 1.0 <= a and b <= r["t1"] + 1.0 for _, a, b in stroke_times(SCRIPT))
    # without a model the physics rules stand in for it
    data2, _, summary2 = build_weak([tmp_path / "data"], None, device="cpu", verbose=False)
    assert summary2["classifier"] == "physics" and data2 is not None and set(data2["source"]) == {"hardneg"}
    # the command line writes the set and the hard negatives on their own
    out = tmp_path / "weak.npz"
    assert weaklabel.main([str(tmp_path / "data"), "--model", str(model), "--out", str(out), "--hard-negatives", str(tmp_path / "hard.npz"),
                           "--report", str(tmp_path / "weak.json"), "--device", "cpu", "--workers", "1"]) == 0
    with np.load(tmp_path / "hard.npz") as z:
        assert len(z["y"]) == summary["examples"] and "weight" in z.files
    report = json.loads((tmp_path / "weak.json").read_text())
    assert report["summary"]["timing"]["bursts"] == len(report["bursts"]) and "judge" in report["summary"]["timing"]
    # held-out people can be left out
    assert build_weak([tmp_path / "data"], model, exclude_people=("dan",), device="cpu", verbose=False)[0] is None


@pytest.mark.slow
def test_unlabelled_recordings_get_pseudo_labels_only_when_they_may_hold_gestures(cfg, loose, model):
    data, rows, summary = build_weak([], model, unlabelled=[loose], device="cpu", verbose=False)
    assert data is not None and summary["by_tag"].get("pseudo", 0) >= 3 and set(data["source"]) == {"pseudo"}
    names = [str(n) for n in data["labels"]]
    truth = stroke_times(SCRIPT)
    gestures = [r for r in rows if r.get("weak_tag") == "pseudo" and r["weak_label"] != "none"]
    assert len(gestures) >= 3
    for r in gestures:                                                # every pseudo-label names the stroke that is really there
        hit = [label for label, a, b in truth if r["t0"] - 0.2 <= a and b <= r["t1"] + 0.4]
        assert r["weak_label"] in hit, (r, truth)
        assert 0.15 <= r["weak_weight"] <= WeakConfig().pseudo_weight and r["model_conf"] >= 0.9 and r["physics"] == r["weak_label"]
    assert float(data["weight"].max()) <= WeakConfig().pseudo_weight and {names[i] for i in data["y"]} <= {"none", "swipe_left", "swipe_right", "push"}
    # the same recording placed in the TRAINING folder is "all none" for the plain builder, so there the
    # model's answers are not labels: whatever fires is a hard negative
    data2, _, summary2 = build_weak([loose], model, device="cpu", verbose=False)
    assert data2 is not None and set(summary2["by_tag"]) == {"hardneg"} and set(data2["y"]) == {0}
    # a stricter bar keeps fewer
    strict = build_weak([], model, WeakConfig(pseudo_conf=0.999, pseudo_none=False), unlabelled=[loose], device="cpu", verbose=False)[2]
    assert strict["by_tag"].get("pseudo", 0) <= summary["by_tag"]["pseudo"]
    # declared gesture-free by name: hard negatives again
    free = build_weak([], model, unlabelled=[loose], gesture_free=("table_engine",), device="cpu", verbose=False)[2]
    assert set(free["by_tag"]) == {"hardneg"}


@pytest.mark.slow
def test_camera_sessions_add_seen_non_gestures_and_rescue_confirmed_reviews(cfg, cam, model, tmp_path):
    data, rows, summary = build_weak([cam], model, device="cpu", verbose=False)
    assert data is None or set(data["y"]) == {0}                        # whatever is added from these two sessions is a negative
    # (how many of these a simulated session holds changes from run to run; the rule itself is pinned down by the
    #  hand-built table in test_rules_on_a_hand_built_table)
    aux = [r for r in rows if r.get("weak_tag") == "aux"]
    assert all(r["role"] == "excluded" and r["zone"].startswith("block:") and r["camera"] in ("other", "return")
               and r["camera_stroke"] in ("other", "away") and r["weak_weight"] == WeakConfig().aux_weight for r in aux)
    hard = [r for r in rows if r.get("weak_tag") == "hardneg"]
    assert all(r["zone"].startswith("free:") or r["camera_stroke"] in ("other", "away") for r in hard)
    assert not [r for r in rows if r["role"] == "gesture" and "weak_tag" in r]      # plain examples are never repeated
    table = build_tables(find_sessions(cam), model, "cpu")[0][0]
    assert table.bursts[0].hand_seen is not None and any(b.cam for b in table.bursts)
    # the camera was unsure about one swipe: the event becomes a "review" ...
    import shutil

    src = next(f for f in find_sessions(cam) if f.name.endswith("_free"))
    dst = tmp_path / "data" / src.name
    shutil.copytree(src, dst)
    meta = json.loads((dst / "session.json").read_text())
    ev = next(e for e in meta["events"] if e["kind"] == "gesture" and e["label"] == "swipe_left")
    ev.update(kind="review", label="review", candidate="swipe_left", confidence=0.42, exclude=True)
    ev.pop("tight")
    (dst / "session.json").write_text(json.dumps(meta))
    # ... which the plain builder leaves out, and the weak labeller brings back because model and physics confirm it
    assert sum(1 for e in session_examples(dst)[0] if e.label == "swipe_left") == 1
    table = session_table(dst, model, "cpu")
    rescued = [w for w in weak_labels(table) if w.tag == "rescued"]
    assert len(rescued) == 1 and rescued[0].label == "swipe_left" and 0.5 <= rescued[0].weight <= WeakConfig().rescue_weight
    # a review that names the wrong gesture is not confirmed by anybody and stays out
    ev.update(candidate="push")
    (dst / "session.json").write_text(json.dumps(meta))
    assert not [w for w in weak_labels(session_table(dst, model, "cpu")) if w.tag == "rescued"]


@pytest.mark.slow
def test_cache_workers_and_parity_command(cfg, sessions, cam, model, tmp_path, capsys):
    folders = find_sessions(sessions) + find_sessions(cam)
    cache = tmp_path / "cache"
    first = all_bursts(folders, None, cache, workers=2)
    assert all(bs is not None and bs.how == "cpu" for bs, _ in first) and len(list(cache.glob("*.npz"))) == len(folders)
    again = all_bursts(folders, None, cache, workers=2)
    assert all(bs.how == "cache" for bs, _ in again)
    for (a, _), (b, _) in zip(first, again):
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.seg, b.seg)
        assert a.cfg == b.cfg and a.n_frames == b.n_frames
    # changed audio (another size or time stamp) is cut again, never read from the cache
    assert weaklabel.cache_path(cache, folders[0], "s") != weaklabel.cache_path(cache, folders[1], "s")
    # an unreadable session is reported, the others go on
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "session.json").write_text((folders[0] / "session.json").read_text())
    got = all_bursts([broken, folders[0]], None, None, workers=1)
    assert got[0][0] is None and got[0][1] and got[1][0] is not None
    # step 1 alone (what train_4090.py runs in the background), then the parity check
    assert weaklabel.main([str(sessions), "--tables-only", "--cache", str(tmp_path / "c2"), "--workers", "2", "--device", "cpu"]) == 0
    assert "ready in" in capsys.readouterr().out and len(list((tmp_path / "c2").glob("*.npz"))) == 3
    assert weaklabel.main([str(sessions), str(cam), "--model", str(model), "--parity", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "PARITY OK" in out and "0 different verdicts" in out and "bursts found at the same frames:        yes" in out


def test_training_weights_and_validation_rules():
    y = np.array([0, 0, 0, 1, 1, 2])
    w = np.array([1.0, 1.0, 2.0, 0.5, 0.5, 1.0])
    p = example_weights(y, w, 3)
    assert p[:3].sum() == pytest.approx(1.0) and p[3:5].sum() == pytest.approx(1.0) and p[5] == pytest.approx(1.0)   # classes stay balanced
    assert p[2] == pytest.approx(2 * p[0]) and p[3] == pytest.approx(p[4])                                          # weight counts inside a class
    np.testing.assert_allclose(example_weights(y, np.ones(6), 3), 1.0 / np.bincount(y)[y])                            # all 1: the old rule
    data = {"source": np.array(["gesture", "pseudo", "hardneg", "burst", "aux", "rescued", "idle"])}
    assert keep_weak_out_of_validation(data, np.arange(7)).tolist() == [0, 3, 6]
    assert set(WEAK_SOURCES) == {"pseudo", "rescued", "aux", "hardneg", "retimed"}
    base = {"X": np.zeros((2, 1, 2, 2), np.float16), "y": np.array([0, 1]), "group": np.array(["a", "a"]), "person": np.array(["p", "p"]),
            "source": np.array(["gesture", "burst"]), "peak_db": np.zeros(2, np.float32), "labels": np.array(["none", "x"]),
            "pad": np.array(0), "config": np.array("{}")}
    weak = {**base, "source": np.array(["pseudo", "hardneg"]), "weight": np.array([0.4, 2.0], np.float32)}
    assert "weight" not in merge([base, base])                                                                        # nothing changes without it
    np.testing.assert_allclose(merge([base, weak, base])["weight"], [1, 1, 0.4, 2.0, 1, 1])


@pytest.mark.slow
def test_training_takes_the_weak_set_and_keeps_its_validation_set(cfg, sessions, loose, model, tmp_path):
    from sonar.train import main as train_main

    data, _, _ = build_weak([], model, unlabelled=[loose], device="cpu", verbose=False)
    np.savez_compressed(tmp_path / "weak.npz", **data)
    common = ["--data", str(sessions), "--epochs", "1", "--device", "cpu", "--val-person", "bob"]
    assert train_main([*common, "--out", str(tmp_path / "plain")]) == 0
    assert train_main([*common, "--dataset", str(tmp_path / "weak.npz"), "--out", str(tmp_path / "with_weak")]) == 0
    a = json.loads((tmp_path / "plain" / "metrics.json").read_text())
    b = json.loads((tmp_path / "with_weak" / "metrics.json").read_text())
    assert b["n_val"] == a["n_val"] and b["n_train"] == a["n_train"] + len(data["y"])      # same score set, more training examples


def test_a_prompt_whose_loudest_burst_is_not_the_gesture_gets_the_right_burst_added(cfg):
    """The plain builder takes the loudest burst of a prompt window. If that one is the hand going back, the
    burst the witnesses do read as the prompted gesture is added (the table is built by hand: no audio needed)."""
    from sonar.weaklabel import Burst, BurstSet, SessionTable

    shape = (4, cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames + 2 * PAD)
    bs = BurstSet(cfg, 2000, np.array([[100, 130, 140, 0, -1], [150, 190, 200, 0, 131], [600, 640, 650, 0, 191], [660, 700, 710, 0, 641]]),
                  np.array([12.0, 20.0, 18.0, 10.0], np.float32), np.zeros(shape, np.float16), np.zeros(2000, np.float32))
    meta = {"session_id": "s", "person": "p", "events": [
        {"label": "swipe_left", "kind": "gesture", "start": 90 * 960, "end": 215 * 960, "rep": 1},
        {"label": "push", "kind": "gesture", "start": 590 * 960, "end": 715 * 960, "rep": 2}]}

    def burst(k, role, trial, label, model, conf, physics):
        s = bs.seg[k]
        return Burst(k, int(s[0]), int(s[1]), int(s[2]), float(bs.peak_db[k]), False, role=role, label=label if role == "gesture" else "none",
                     event=trial if role == "gesture" else None, trial=trial, zone="trial:" + label, physics=physics, physics_conf=0.8,
                     model=model, model_conf=conf, label_source="prompt", label_conf=1.0)

    bursts = [
        burst(0, "aux", 0, "swipe_left", "swipe_left", 0.93, "swipe_left"),       # the stroke: quieter, read as the prompted gesture
        burst(1, "gesture", 0, "swipe_left", "swipe_right", 0.88, "swipe_right"),  # the loudest: the hand going back
        burst(2, "gesture", 1, "push", "push", 0.95, "push"),                     # a prompt whose loudest burst is fine
        burst(3, "aux", 1, "push", "push", 0.91, "none"),                        # ... so its other burst is left alone
    ]
    table = SessionTable(__import__("pathlib").Path("."), meta, cfg, bs, bursts, [], "model", ("none", "swipe_left", "swipe_right", "push"))
    found = weak_labels(table)
    assert [(w.k, w.label, w.tag, w.weight) for w in found] == [(0, "swipe_left", "retimed", WeakConfig().retime_weight)]
    assert "read as swipe_right" in found[0].why
    bursts[0].model_conf = 0.55                                               # not sure enough: nothing is added
    assert weak_labels(table) == []
    bursts[0].model_conf, bursts[0].physics = 0.93, "push"                    # the physics rules name a different gesture: nothing is added
    assert weak_labels(table) == []
    table.classifier, bursts[0].physics = "physics", "swipe_left"             # without a model the physics rules decide (0.8 >= 0.6)
    assert [(w.k, w.tag) for w in weak_labels(table)] == [(0, "retimed")]


def test_rules_on_a_hand_built_table(cfg):
    """Every rule that turns the witnesses' opinions into a weighted example, on bursts written out by hand."""
    from pathlib import Path

    from sonar.weaklabel import Burst, BurstSet, SessionTable

    n = 9
    seg = np.array([[100 + 200 * k, 130 + 200 * k, 140 + 200 * k, 0, -1] for k in range(n)])
    bs = BurstSet(cfg, 4000, seg, np.full(n, 15.0, np.float32),
                  np.zeros((n, cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames + 2 * PAD), np.float16), np.zeros(4000, np.float32))

    def burst(k, **kw):
        base = dict(role="none", zone="free:typing", physics="none", physics_conf=1.0, model="none", model_conf=0.95)
        base.update(kw)
        return Burst(k, int(seg[k][0]), int(seg[k][1]), int(seg[k][2]), 15.0, False, **base)

    bursts = [
        burst(0, model="push", model_conf=0.8),                                              # fires while typing: hard negative
        burst(1),                                                                             # quiet about typing: an ordinary negative, not repeated
        burst(2, model="push", model_conf=0.8, cam="review", cam_phys="push"),                # the camera suspects a real gesture: left alone
        burst(3, role="excluded", zone="block:swipe_left", cam="other", cam_phys="other"),    # the camera saw a slow movement: aux negative
        burst(4, role="excluded", zone="block:swipe_left", cam="return", cam_phys="away", model="swipe_right", model_conf=0.9),   # hand dropping back, model fires
        burst(5, role="excluded", zone="block:swipe_left", cam="return", cam_phys="swipe_right"),          # a brisk stroke the other way: ambiguous, left alone
        burst(6, role="excluded", zone="block:push", model="push", model_conf=0.97, physics="push", physics_conf=0.8, hand_seen=0.0),   # camera blind, both agree
        burst(7, role="excluded", zone="block:push", model="push", model_conf=0.97, physics="push", physics_conf=0.8, hand_seen=1.0),   # camera watched: no stroke
        burst(8, role="excluded", zone="block:swipe_left", model="push", model_conf=0.97, physics="push", physics_conf=0.8, hand_seen=0.0),   # not what the block asked for
    ]
    table = SessionTable(Path("."), {"session_id": "s", "person": "p", "events": []}, cfg, bs, bursts, [], "model",
                         ("none", "swipe_left", "swipe_right", "push"))
    wc = WeakConfig()
    got = {w.k: (w.label, w.tag, w.weight) for w in weak_labels(table, wc)}
    assert got == {
        0: ("none", "hardneg", wc.hardneg_weight),
        3: ("none", "aux", wc.aux_weight),
        4: ("none", "hardneg", wc.hardneg_weight * 0.75),
        6: ("push", "pseudo", pytest.approx(wc.pseudo_weight * 0.7, abs=1e-3)),
    }
    # a burst that is too faint to matter is never used
    bursts[0].peak_db = 1.0
    assert 0 not in {w.k for w in weak_labels(table, wc)}
    # a recording with no labels: "all none" when it sits in the training folder, pseudo-labels when given as unlabelled
    loose = [burst(0, zone="unlabelled", model="swipe_left", model_conf=0.99, physics="swipe_left", physics_conf=0.9),
             burst(1, zone="unlabelled"), burst(2, zone="unlabelled", model="push", model_conf=0.99, physics="none")]
    table2 = SessionTable(Path("."), {"session_id": "s", "person": "p", "events": []}, cfg, bs, loose, [], "model", ("none", "swipe_left", "swipe_right", "push"))
    assert {w.k: (w.label, w.tag) for w in weak_labels(table2, wc)} == {0: ("none", "hardneg"), 2: ("none", "hardneg")}
    assert {w.k: (w.label, w.tag) for w in weak_labels(table2, wc, may_hold_gestures=True)} == {0: ("swipe_left", "pseudo"), 1: ("none", "pseudo")}
    # a gesture the model was not trained for is never pseudo-labelled
    assert 0 not in {w.k for w in weak_labels(table2, wc, labels=("push",), may_hold_gestures=True)}

"""Replaying recordings through the real engine: what was recognised, what was missed, and why."""

import json

import pytest

from sonar import gesture_replay
from sonar.dataset import find_sessions
from sonar.gesture_replay import REASONS, replay_session, tally

from .engine_helpers import camera_sessions, prompted_sessions, scripted_audio, small_model, span, write_session


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    return prompted_sessions(tmp_path_factory)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return small_model(tmp_path_factory) / "gesture.onnx"


@pytest.mark.slow
def test_replay_counts_recognised_and_explains_the_misses(cfg, sessions, model):
    folder = find_sessions(sessions)[0]
    res = replay_session(folder, model)
    outcomes = [t["outcome"] for t in res["trials"]]
    assert len(outcomes) == 16 and outcomes.count("recognised") >= 14 and res["classifier"] == "model" and res["person"] == "ann"
    hit = next(t for t in res["trials"] if t["outcome"] == "recognised")
    assert 0.2 < hit["delay_s"] < 3.5 and hit["confidence"] >= 0.6 and hit["label"] in ("swipe_left", "swipe_right", "push", "pull")
    assert res["gate_db"] > 5.0 and res["bursts_seen"] >= 16          # the gate comes from the model file, as in the live engine
    # a gate above every gesture: all blocked as too faint, and that is the reason given
    gated = replay_session(folder, model, min_peak_db=99.0)
    assert all(t["outcome"] == "missed" and t["why"] == REASONS[1] for t in gated["trials"]) and not gated["false_triggers"]
    # an impossible confidence threshold: the model names the right gesture but is "not sure enough"
    unsure = replay_session(folder, model, min_confidence=1.01)
    whys = [t["why"] for t in unsure["trials"]]
    assert all(t["outcome"] == "missed" for t in unsure["trials"]) and whys.count(REASONS[3]) >= 12
    # the physics rules can be replayed the same way (no model file)
    physics = replay_session(folder, None)
    assert physics["classifier"] == "physics" and sum(t["outcome"] == "recognised" for t in physics["trials"]) >= 10
    totals = tally([res, gated])
    assert totals["all"]["trials"] == 32 and totals["all"]["recognised"] == outcomes.count("recognised")
    assert totals["all"]["why"][REASONS[1]] == 16 and totals["by_person"]["ann"]["recall"] == pytest.approx(totals["all"]["recall"])
    assert sum(g["trials"] for g in totals["all"]["by_gesture"].values()) == 32


@pytest.mark.slow
def test_gestures_reported_where_none_was_asked_for_are_false_triggers(cfg, model, tmp_path, tmp_path_factory, capsys):
    script = [(3.0, "swipe_right", 1.0), (6.5, "push", 1.0), (10.0, "swipe_left", 1.1), (13.5, "push", 1.2)]
    audio, _ = scripted_audio(cfg, script, 17.0, seed=7)
    root = tmp_path / "data"
    write_session(root / "20261114-180000_fay_scene", cfg, audio, [span(cfg, "distractor", 2.6, 16.5)], person="fay", script="scene")
    res = replay_session(root / "20261114-180000_fay_scene", model)
    assert not res["trials"] and len(res["false_triggers"]) >= 2
    assert all(f["during"] == "free:distractor" and f["confidence"] >= 0.6 for f in res["false_triggers"])
    t = tally([res])["all"]
    assert t["trials"] == 0 and t["recall"] is None and t["false_per_minute"] > 3.0 and t["false_during"] == {"free:distractor": len(res["false_triggers"])}
    # camera-timed strokes are trials too, and nothing inside a free-form block counts as a false trigger
    cam = next(f for f in find_sessions(camera_sessions(tmp_path_factory)) if f.name.endswith("_free"))
    cres = replay_session(cam, model)
    assert len(cres["trials"]) == 8 and sum(t["outcome"] == "recognised" for t in cres["trials"]) >= 5 and not cres["false_triggers"]
    assert all(t["source"] == "camera" for t in cres["trials"])
    # the command line: per person, the reasons, a report file
    out = tmp_path / "replay.json"
    assert gesture_replay.main([str(root), str(cam), "--model", str(model), "--workers", "2", "--json", str(out), "--min-peak-db", "99"]) == 0
    text = capsys.readouterr().out
    assert "why the missed ones were missed" in text and REASONS[1] in text and "most common" in text and "time:" in text
    saved = json.loads(out.read_text())
    assert saved["totals"]["all"]["recognised"] == 0 and len(saved["sessions"]) == 2
    assert gesture_replay.main([str(root), "--model", str(model), "--people", "nobody"]) == 2

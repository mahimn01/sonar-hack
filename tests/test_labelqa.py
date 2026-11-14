"""Label QA: agreement numbers, flags for single moments worth a look, and the picture."""

import json
import shutil

import pytest

from sonar import labelqa
from sonar.dataset import find_sessions
from sonar.labelqa import one_line, review_session, totals
from sonar.weaklabel import session_table

from .engine_helpers import camera_sessions, prompted_sessions, scripted_audio, small_model, span, write_session


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    return prompted_sessions(tmp_path_factory)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return small_model(tmp_path_factory) / "gesture.onnx"


@pytest.fixture(scope="module")
def cam(tmp_path_factory):
    return camera_sessions(tmp_path_factory)


def free_session(root):
    return next(f for f in find_sessions(root) if f.name.endswith("_free"))


@pytest.mark.slow
def test_agreement_numbers_for_camera_and_prompt_labels(cfg, sessions, cam, model):
    rep = review_session(session_table(free_session(cam), model, "cpu"))
    assert rep["gestures"] == rep["from_camera"] == 8 and rep["from_prompt"] == 0 and rep["no_sonar_movement"] == 0
    row = rep["agreement"]["camera"]
    assert row["labelled"] == 8 and row["physics"]["same"] >= 7 and row["model"]["same"] >= 6
    assert row["physics"]["same"] + row["physics"]["saw_no_gesture"] + row["physics"]["other_gesture"] == 8
    assert rep["swipe_direction"]["camera"]["flipped"] == 0 and rep["swipe_direction"]["camera"]["swipes"] >= 3
    assert rep["sync"]["method"] == "motion" and not rep["verdicts"]
    assert not [f for f in rep["flags"] if f["level"] == "wrong label?"]
    assert "8 gestures (8 camera, 0 prompt)" in one_line(rep) and rep["classifier"] == "model"
    prompted = review_session(session_table(find_sessions(sessions)[0], model, "cpu"))
    assert prompted["from_prompt"] == 16 and prompted["agreement"]["prompt"]["physics"]["same"] >= 14 and prompted["sync"] is None
    both = totals([rep, prompted])
    assert both["sessions"] == 2 and both["all"]["gestures"] == 24 and set(both["by_person"]) == {"v21", "ann"}
    assert both["all"]["agreement"]["camera"]["labelled"] == 8 and both["all"]["agreement"]["prompt"]["labelled"] == 16
    # without a model only the physics rules are compared
    alone = review_session(session_table(free_session(cam), None, "cpu"))
    assert "model" not in alone["agreement"]["camera"] and alone["physics_vs_model"] is None and alone["classifier"] == "physics"


@pytest.mark.slow
def test_flipped_swipe_labels_are_called_out(cfg, cam, model, tmp_path):
    """A camera picture mirrored the wrong way swaps every swipe label: the sonar's own direction cues notice."""
    dst = tmp_path / free_session(cam).name
    shutil.copytree(free_session(cam), dst)
    meta = json.loads((dst / "session.json").read_text())
    swap = {"swipe_left": "swipe_right", "swipe_right": "swipe_left"}
    for ev in meta["events"]:
        if ev.get("kind") == "gesture" and ev["label"] in swap:
            ev["label"] = swap[ev["label"]]
    (dst / "session.json").write_text(json.dumps(meta))
    rep = review_session(session_table(dst, model, "cpu"))
    d = rep["swipe_direction"]["camera"]
    assert d["swipes"] == 4 and d["flipped"] == 4
    assert any("LEFT/RIGHT LOOKS FLIPPED" in v and "mirrored" in v for v in rep["verdicts"])
    wrong = [f for f in rep["flags"] if f["kind"] == "contradicted"]
    assert len(wrong) == 4 and all(f["level"] == "wrong label?" and "model says" in f["text"] and "physics says" in f["text"] for f in wrong)
    assert "4 to check" in one_line(rep)


@pytest.mark.slow
def test_gestures_the_sonar_did_not_see_and_firing_on_negatives_are_flagged(cfg, cam, model, tmp_path):
    # labels that point at moments where nothing moved (as a badly wrong camera clock would produce)
    dst = tmp_path / "shifted" / free_session(cam).name
    shutil.copytree(free_session(cam), dst)
    meta = json.loads((dst / "session.json").read_text())
    for ev in meta["events"]:
        if ev.get("kind") == "gesture":
            ev["start"], ev["end"] = int(1.0 * cfg.fs), int(1.3 * cfg.fs)          # inside the still calibration seconds
    meta["sync"]["method"] = "timestamps"
    (dst / "session.json").write_text(json.dumps(meta))
    rep = review_session(session_table(dst, model, "cpu"))
    assert rep["no_sonar_movement"] == 8 and rep["gestures"] == 0
    assert sum(1 for f in rep["flags"] if f["kind"] == "no sonar movement") == 8
    assert any("NOT lined up" in v for v in rep["verdicts"])
    # strokes inside a part recorded as gesture-free: the model fires, every one is listed as a hard negative
    audio, _ = scripted_audio(cfg, [(3.0, "swipe_right", 1.0), (6.5, "push", 1.0), (10.0, "swipe_left", 1.1)], 14.0, seed=6)
    neg = write_session(tmp_path / "neg" / "20261114-140000_eve_scene", cfg, audio, [span(cfg, "distractor", 2.6, 13.5)], person="eve", script="scene")
    rep = review_session(session_table(neg, model, "cpu"))
    assert rep["gestures"] == 0 and rep["negatives"] >= 3 and rep["fires_on_negatives"] >= 2
    fired = [f for f in rep["flags"] if f["kind"] == "fires on a negative"]
    assert len(fired) == rep["fires_on_negatives"] and all(f["level"] == "hard negative" and "free:distractor" in f["text"] for f in fired)


@pytest.mark.slow
def test_command_line_writes_numbers_and_pictures(cfg, cam, sessions, model, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out = tmp_path / "qa"
    assert labelqa.main([str(free_session(cam)), "--model", str(model), "--out", str(out), "--seconds", "20", "--device", "cpu"]) == 0
    folder = out / free_session(cam).name
    saved = json.loads((folder / "qa.json").read_text())
    pictures = sorted(p.name for p in folder.glob("qa_p*.png"))
    assert pictures == saved["pictures"] and len(pictures) >= 2 and all((folder / p).stat().st_size > 20000 for p in pictures)
    assert len(saved["bursts"]) == saved["bursts"].__len__() and saved["bursts"][0].keys() >= {"role", "zone", "physics", "model", "camera"}
    assert not list(free_session(cam).glob("qa*"))                                   # --out: the session folder is left alone
    # a whole folder, numbers only: one batched pass, a summary file, per-person lines
    assert labelqa.main([str(cam), str(sessions), "--model", str(model), "--no-plot", "--out", str(out), "--device", "cpu", "--workers", "2"]) == 0
    text = capsys.readouterr().out
    assert "=== 5 session(s)" in text and "physics vs camera labels" in text and "model   vs prompt labels" in text and "time:" in text
    summary = json.loads((out / "labelqa_summary.json").read_text())
    assert summary["totals"]["sessions"] == 5 and len(summary["sessions"]) == 5
    assert labelqa.main([str(tmp_path / "nothing-here")]) == 2


@pytest.mark.slow
def test_burst_cutting_audit_and_the_peak_relative_option(cfg, tmp_path, monkeypatch):
    """Very loud echoes (as on the MacBook recordings) make every gesture one long burst chopped at the length limit.
    The audit says so, and the switchable peak-relative cutting gives short bursts again."""
    from sonar.config import SonarConfig, get_profile
    from sonar.labelqa import cutting_report

    from .engine_helpers import room_session

    assert get_profile("default").seg_rel_off_db == 0.0 and SonarConfig.from_dict({"fs": 48000}).seg_rel_off_db == 0.0     # off unless asked for
    mono = cfg.with_(n_mics=1)
    loud = room_session(tmp_path / "20261114-190000_zed_gestures", mono, "zed", seed=77, reps=4, gestures=("swipe_left", "swipe_right", "push"),
                        hand_db=4.0, fidgets=0)
    table = session_table(loud, None, "cpu")
    cut = cutting_report(table)
    assert cut["gestures"] >= 10 and cut["gesture_cut_at_limit_share"] >= 0.6 and cut["gesture_seconds_median"] >= 1.2
    assert cut["gesture_peak_db"]["median"] > 30.0 and 1.0 <= cut["bursts_per_gesture"] <= 4.0 and cut["limit_s"] == 1.4
    assert cut["burst_start_after_prompt_s"]["spread_p10_p90"] >= 0.0 and -0.5 < cut["burst_start_after_prompt_s"]["median"] < 2.5
    assert 0.0 <= cut["saturated_share"] < 0.5 and cut["recognised_in_some_burst"] >= cut["gestures"] - 3
    rep = review_session(table)
    assert rep["cutting"] == cut and any("LONG BURSTS" in v and "SONAR_SEG_REL_OFF_DB" in v for v in rep["verdicts"])
    # switched on for every tool at once by one environment variable (recordings keep their stored settings)
    monkeypatch.setenv("SONAR_SEG_REL_OFF_DB", "12")
    assert get_profile("default").seg_rel_off_db == 12.0
    better = cutting_report(session_table(loud, None, "cpu"))
    assert better["gesture_cut_at_limit_share"] < cut["gesture_cut_at_limit_share"]          # fewer bursts chopped at the limit ...
    assert better["burst_seconds_median"] < cut["burst_seconds_median"] - 0.2                   # ... and clearly shorter bursts overall
    print("cut at the limit:", cut["gesture_cut_at_limit_share"], "->", better["gesture_cut_at_limit_share"],
          "| typical burst length:", cut["burst_seconds_median"], "->", better["burst_seconds_median"])
    monkeypatch.delenv("SONAR_SEG_REL_OFF_DB")
    # at ordinary echo strength a generous setting changes nothing: the absolute thresholds still decide
    from sonar.features import stream_features

    audio, _ = scripted_audio(cfg, [(3.0, "swipe_right", 1.0), (6.5, "push", 1.0), (10.0, "swipe_left", 1.1)], 13.5, seed=2, hand_db=-24.0)
    plain = [(s.start, s.end, s.decided) for s in stream_features(audio, cfg).segments]
    assert plain and plain == [(s.start, s.end, s.decided) for s in stream_features(audio, cfg.with_(seg_rel_off_db=24.0)).segments]

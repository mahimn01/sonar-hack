"""A corrected view of recorded sessions: labels fixed, duplicates taken once, the recordings never touched."""

import json
from pathlib import Path

import numpy as np
import pytest

from sonar import dataview
from sonar.dataset import find_sessions, load_session, session_examples
from sonar.dataview import KNOWN_FLIPPED, make_view
from sonar.labelqa import review_session
from sonar.weaklabel import session_table

from .engine_helpers import prompted_sessions, small_model, span, write_session


def tiny_session(folder: Path, cfg, labels: list[str], person: str = "x") -> Path:
    """A session folder with one second of silence and one prompt per label: enough for everything but the pictures."""
    events = [span(cfg, lab, 0.1 + 0.2 * i, 0.25 + 0.2 * i, kind="gesture") for i, lab in enumerate(labels)]
    return write_session(folder, cfg, np.zeros((cfg.fs, 2), dtype=np.float32), events, person=person)


def gesture_labels(folder: Path) -> list[str]:
    return [e["label"] for e in json.loads((folder / "session.json").read_text())["events"] if e["kind"] == "gesture"]


def files_under(root: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_view_fixes_one_session_and_leaves_the_recordings_alone(cfg, tmp_path):
    root, view = tmp_path / "mac", tmp_path / "view"
    tiny_session(root / "data" / "s1_ann_gestures", cfg, ["swipe_left", "swipe_right", "push"], "ann")
    bob = tiny_session(root / "data" / "s2_bob_gestures", cfg, ["swipe_left", "push", "swipe_right", "swipe_right"], "bob")
    before = files_under(root)
    rows = make_view([root], view, flip_swipes=("s2_bob_gestures",), verbose=False)
    assert files_under(root) == before                                         # nothing recorded was touched
    assert not list(view.rglob("*.wav"))                                       # the view holds labels only
    assert gesture_labels(view / "s1_ann_gestures") == ["swipe_left", "swipe_right", "push"]
    assert gesture_labels(view / "s2_bob_gestures") == ["swipe_right", "push", "swipe_left", "swipe_left"]
    fixed = json.loads((view / "s2_bob_gestures" / "session.json").read_text())
    assert "exchanged" in fixed["label_fix"] and Path(fixed["audio"]) == (bob / "audio.wav").resolve() and fixed["view_of"] == str(bob.resolve())
    assert "label_fix" not in json.loads((view / "s1_ann_gestures" / "session.json").read_text())
    assert [r["action"] for r in sorted(rows, key=lambda r: r["session"])] == ["as recorded", "LEFT/RIGHT EXCHANGED in 3 swipe labels"]
    # the tools that read sessions read the view like the real thing
    assert find_sessions(view) == [view / "s1_ann_gestures", view / "s2_bob_gestures"]
    meta, audio, _ = load_session(view / "s2_bob_gestures")
    assert audio.shape == (cfg.fs, 2) and meta["person"] == "bob"


def test_duplicates_are_taken_once_and_bad_folders_are_named(cfg, tmp_path):
    root = tmp_path / "mac"
    # the same session twice, in both orders of discovery: the copy outside "excluded" must win each time
    tiny_session(root / "data" / "s1_ann_gestures", cfg, ["push"], "ann")
    tiny_session(root / "excluded" / "s1_ann_gestures", cfg, ["swipe_left"], "ann")
    tiny_session(root / "excluded" / "s3_cy_gestures", cfg, ["swipe_left"], "cy")
    tiny_session(root / "kept" / "s3_cy_gestures", cfg, ["push"], "cy")
    tiny_session(root / "data" / "s2_bob_gestures", cfg, ["push"], "bob")
    tiny_session(root / "excluded" / "s4_dee_gestures", cfg, ["swipe_right"], "dee")       # only an excluded copy: still used
    (tiny_session(root / "data" / "s5_no_audio", cfg, ["push"]) / "audio.wav").unlink()
    (root / "data" / "s6_broken").mkdir()
    (root / "data" / "s6_broken" / "session.json").write_text("{ not json")
    view = root / "view"                                                       # inside the data folder, like train_out/dataeng/view could be
    for _ in range(2):                                                         # the second time it must not read its own output
        rows = make_view([root], view, flip_swipes=(), drop=("s2_bob_gestures",), verbose=False)
        assert sorted(p.name for p in view.iterdir()) == ["s1_ann_gestures", "s3_cy_gestures", "s4_dee_gestures"]
        assert gesture_labels(view / "s1_ann_gestures") == ["push"] and gesture_labels(view / "s3_cy_gestures") == ["push"]
        actions = sorted((r["session"], r["action"].split(":")[0].split(" at ")[0]) for r in rows)
        assert actions == [("s1_ann_gestures", "as recorded"), ("s1_ann_gestures", "skipped"), ("s2_bob_gestures", "left out (--drop)"),
                           ("s3_cy_gestures", "as recorded"), ("s3_cy_gestures", "skipped"), ("s4_dee_gestures", "as recorded"),
                           ("s5_no_audio", "skipped"), ("s6_broken", "skipped")]
        skipped = {r["session"]: r for r in rows if r["action"].startswith("skipped")}
        assert "excluded" in Path(skipped["s1_ann_gestures"]["folder"]).parts and "excluded" in Path(skipped["s3_cy_gestures"]["folder"]).parts
        assert "no audio" in skipped["s5_no_audio"]["action"] and "cannot read" in skipped["s6_broken"]["action"]


def test_command_line(cfg, tmp_path, capsys):
    root = tmp_path / "mac"
    tiny_session(root / KNOWN_FLIPPED[0], cfg, ["swipe_left", "swipe_right"], "mahimn")
    assert dataview.main([str(root), "--out", str(tmp_path / "v1")]) == 0     # the known session is fixed without being asked
    assert gesture_labels(tmp_path / "v1" / KNOWN_FLIPPED[0]) == ["swipe_right", "swipe_left"]
    assert "LEFT/RIGHT EXCHANGED in 2 swipe labels" in capsys.readouterr().out
    assert dataview.main([str(root), "--out", str(tmp_path / "v2"), "--flip-swipes", ""]) == 0
    assert gesture_labels(tmp_path / "v2" / KNOWN_FLIPPED[0]) == ["swipe_left", "swipe_right"]
    assert dataview.main([str(tmp_path / "nothing_here"), "--out", str(tmp_path / "v3")]) == 2    # nothing usable: say so


@pytest.mark.slow
def test_a_view_gives_the_same_pictures_and_the_label_check_names_the_fix(cfg, tmp_path, tmp_path_factory):
    sessions = prompted_sessions(tmp_path_factory)
    model = small_model(tmp_path_factory) / "gesture.onnx"
    first, second = find_sessions(sessions)[:2]
    view = tmp_path / "view"
    make_view([sessions], view, flip_swipes=(first.name,), verbose=False)
    swap = {"swipe_left": "swipe_right", "swipe_right": "swipe_left"}
    for folder, fixed in ((first, True), (second, False)):
        real, _ = session_examples(folder)
        seen, _ = session_examples(view / folder.name)
        assert len(real) == len(seen) and all(np.array_equal(a.x, b.x) and a.group == b.group and a.person == b.person for a, b in zip(real, seen))
        assert [b.label for b in seen] == [swap.get(a.label, a.label) if fixed else a.label for a in real]
        assert any(a.label in swap for a in real)
    # the label check on the session whose swipes now go against their labels: it says so, and how to put it right
    rep = review_session(session_table(view / first.name, model, "cpu"))
    d = rep["swipe_direction"]["prompt"]
    assert d["swipes"] >= 6 and d["flipped"] / d["swipes"] >= 0.7
    assert any("LEFT/RIGHT LOOKS FLIPPED" in v and f"sonar.dataview <data> --out <view> --flip-swipes {first.name}" in v for v in rep["verdicts"])
    assert not any("LEFT/RIGHT LOOKS FLIPPED" in v for v in review_session(session_table(view / second.name, model, "cpu"))["verdicts"])

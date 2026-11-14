"""Tests for the fitted laptop (datagen/calibration.py), the real hand paths (datagen/motion.py)
and the labelling of cut-up strokes. Fast: no audio is rendered here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from datagen import calibration
from datagen import generate as g
from datagen.motion import load_library, place_motion

REPO = Path(__file__).resolve().parent.parent
LIBRARY = REPO / "datagen" / "motion_library" / "unica_v1.npz"
CAL = REPO / "calibration" / "macbook_v1.json"


@pytest.fixture(autouse=True)
def _no_calibration_left_behind():
    """A calibration switched on in one test must not leak into the next (it is a process-wide setting)."""
    calibration.activate(None)
    yield
    calibration.activate(None)


def _place(rng: np.random.Generator, clip: dict, mirrored: bool, label: str) -> dict:
    t = np.arange(0.0, 4.7, 0.01)
    return place_motion(rng, t, clip, mirrored, label, np.array([0.0, 0.25, 0.15]), 2.4, np.array([0.2, 0.35, 0.02]), 0.45)


# ------------------------------------------------------------------ real hand paths
def test_library_clips_carry_a_human_peak_speed():
    lib = load_library(str(LIBRARY))
    speeds = np.array([c["peak_speed"] for c in lib.clips if c["label"] != "other"])
    assert len(speeds) > 100
    assert speeds.min() > 0.1 and speeds.max() < 6.0          # metres per second, as tracked from the videos
    assert 0.5 < np.median(speeds) < 1.5


def test_replayed_strokes_keep_a_sensible_tempo():
    """Regression: a loaded clip once had no peak speed, and every stroke was replayed at the fastest tempo allowed (0.07 s)."""
    lib = load_library(str(LIBRARY))
    rng = np.random.default_rng(3)
    seconds = []
    for _ in range(60):
        label = ("swipe_left", "swipe_right", "push", "pull")[int(rng.integers(0, 4))]
        clip, mirrored = lib.draw(rng, label)
        placed = _place(rng, clip, mirrored, label)
        seconds.append(placed["t_end"] - placed["t_start"])
        assert placed["peak_speed"] > 0.0
    seconds = np.array(seconds)
    assert np.median(seconds) > 0.15
    assert (seconds < 0.08).mean() < 0.1                       # the floor is the exception, not the rule


def test_fitted_laptop_never_asks_more_of_a_hand_than_muscles_allow():
    calibration.activate(str(CAL))
    lib = load_library(str(LIBRARY))
    rng = np.random.default_rng(5)
    for _ in range(60):
        label = ("swipe_left", "swipe_right", "push", "pull")[int(rng.integers(0, 4))]
        clip, mirrored = lib.draw(rng, label)
        placed = _place(rng, clip, mirrored, label)
        seconds = placed["t_end"] - placed["t_start"]
        recorded = (clip["stroke"][1] - clip["stroke"][0]) / clip["fps"]
        assert seconds >= 0.079
        assert seconds >= 0.4 * recorded - 1e-9                # at most 2.5 times the recorded tempo
    for travel in (0.1, 0.2, 0.4, 0.56):
        for _ in range(50):
            sec = calibration.stroke_seconds(rng, travel, 0.5)
            assert sec >= max(0.08, np.sqrt(5.77 * travel / 150.0)) - 1e-9      # 150 m/s^2 is the most a hand is given
            assert sec <= 1.4


# ------------------------------------------------------------------ the hooks do nothing without a calibration
def test_hooks_are_silent_without_a_calibration():
    rng = np.random.default_rng(1)
    before = rng.bit_generator.state
    assert calibration.stroke_seconds(rng, 0.3, 0.42) == 0.42
    assert calibration.hand_height(rng, 0.11) == 0.11
    assert calibration.lead_in(1.2, 0.4) == (1.2, 0.4)
    assert calibration.speed_range((0.5, 2.2)) == (0.5, 2.2)
    hw = SimpleNamespace()
    calibration.adjust_hardware(rng, hw)                      # must not touch the laptop
    calibration.adjust_overlay(rng, hw, [3.0, -3.0])
    assert vars(hw) == {}
    assert rng.bit_generator.state == before                  # ... and must not use up random numbers


def test_lead_in_is_short_with_a_calibration():
    calibration.activate(str(CAL))
    assert calibration.lead_in(1.2, 0.4) == (0.7, 0.15)
    assert calibration.lead_in(0.5, 0.1) == (0.5, 0.1)


def test_overlay_gives_both_speakers_the_same_echo_level():
    calibration.activate(str(CAL))
    rng = np.random.default_rng(2)
    hw = SimpleNamespace(spk_resp=[SimpleNamespace(gain_db=0.5), SimpleNamespace(gain_db=-1.0)], echo_ref_db=-20.0)
    calibration.adjust_overlay(rng, hw, [3.2, -3.2])           # the left tone arrives 6.4 dB louder at the microphone
    assert hw.spk_resp[0].gain_db == pytest.approx(0.5 - 3.2)
    assert hw.spk_resp[1].gain_db == pytest.approx(-1.0 + 3.2)
    lo, hi = json.loads(CAL.read_text())["overlay"]["echo_ref_db"]
    assert lo <= hw.echo_ref_db <= hi


# ------------------------------------------------------------------ folds
def test_people_are_sorted_into_humans():
    assert [calibration.human_of(p) for p in ("lance", "lance2", "Lance_2", "aarsh", "mahimn")] == ["lance", "lance", "lance", "aarsh", "mahimn"]
    assert calibration.fold_path(Path("calibration/macbook_v1.json"), "lance") == Path("calibration/macbook_v1_no_lance.json")


def test_newest_picks_the_highest_version_and_no_fold_file(tmp_path: Path):
    for name in ("macbook_v1.json", "macbook_v2.json", "macbook_v10.json", "macbook_v10_no_lance.json", "macbook_v10_folds.report.json"):
        (tmp_path / name).write_text("{}")
    assert calibration.newest(tmp_path).name == "macbook_v10.json"
    assert calibration.newest(tmp_path / "nothing_here") is None


def test_fold_file_is_fitted_without_the_held_out_human(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """fit_folds with made-up trial results: every fold must choose by the OTHER humans' pictures only."""
    slow = {k: 0.0 for k in calibration.KEYS}
    fast = {k: 1.0 * calibration.SCALE[k] for k in calibration.KEYS}
    rows = {"lance": [("swipe_left", fast)] * 3, "lance2": [("swipe_left", fast)] * 3, "aarsh": [("swipe_left", slow)] * 3,
            "mahimn": [("swipe_left", slow)] * 3}

    class FakeSim:
        def __init__(self, *a, **k) -> None: ...
        def __enter__(self): return self
        def __exit__(self, *exc) -> None: ...

    def fake_stats(cal_path: str, n: int, sim, backgrounds: tuple = ()) -> dict:
        scene = json.loads(Path(cal_path).read_text())["scene"]
        is_fast = scene["peak_speed"][1] >= 6.5                # the two fastest speed ranges make "fast" pictures
        return {"swipe_left": dict(fast if is_fast else slow, n=n), "strokes_found": 1.0}

    monkeypatch.setattr(calibration, "Sim", FakeSim)
    monkeypatch.setattr(calibration, "sim_stats", fake_stats)
    base = json.loads(CAL.read_text())
    cals, report = calibration.fit_folds(base, rows, tmp_path, 1, 1, log=lambda text: None)
    assert sorted(cals) == ["aarsh", "lance", "mahimn"]
    assert cals["lance"]["scene"]["peak_speed"][1] < 6.5       # without lance only slow people are left
    assert cals["lance"]["fold"]["people_used"] == ["aarsh", "mahimn"]
    assert cals["lance"]["fold"]["held_out_people"] == ["lance", "lance2"]
    assert cals["aarsh"]["hardware"] == base["hardware"]       # the laptop itself is the same in every fold
    assert cals["aarsh"]["overlay"] == base["overlay"]
    assert len(report["combinations"]) == 15


def test_fold_trial_clips_stay_off_the_held_out_humans_recordings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """With real no-gesture recordings to lay the trial clips over, every fold uses only the OTHER humans' recordings."""
    numbers = {k: 0.0 for k in calibration.KEYS}
    rows = {p: [("swipe_left", numbers)] * 2 for p in ("lance", "lance2", "aarsh", "mahimn")}
    used: dict = {}

    class FakeSim:
        def __init__(self, *a, **k) -> None: ...
        def __enter__(self): return self
        def __exit__(self, *exc) -> None: ...

    def fake_stats(cal_path: str, n: int, sim, backgrounds: tuple = ()) -> dict:
        tag = Path(cal_path).stem.split("_")[-2]               # fold_trial_<time>_<human>_<k>
        used.setdefault(tag, set()).update(backgrounds)
        assert "overlay" in json.loads(Path(cal_path).read_text())
        return {"swipe_left": dict(numbers, n=n), "strokes_found": 1.0}

    monkeypatch.setattr(calibration, "Sim", FakeSim)
    monkeypatch.setattr(calibration, "sim_stats", fake_stats)
    backgrounds = (("l1.npy", "lance"), ("l2.npy", "lance2"), ("a.npy", "aarsh"), ("m.npy", "mahimn"))
    cals, report = calibration.fit_folds(json.loads(CAL.read_text()), rows, tmp_path, 1, 1, log=lambda text: None, backgrounds=backgrounds)
    assert used == {"lance": {"a.npy", "m.npy"}, "aarsh": {"l1.npy", "l2.npy", "m.npy"}, "mahimn": {"l1.npy", "l2.npy", "a.npy"}}
    assert sorted(cals) == ["aarsh", "lance", "mahimn"]
    assert "never laid over that human's own recordings" in cals["lance"]["fold"]["note"]
    assert sorted(report["sim"]) == ["aarsh", "lance", "mahimn"]


# ------------------------------------------------------------------ labels of cut-up strokes
def _stream(bursts: list) -> SimpleNamespace:
    """A stand-in for the feature stream: bursts given as (first frame, last frame) at 50 frames a second."""
    return SimpleNamespace(frame_rate=50.0, segments=[SimpleNamespace(start=a, end=b, peak_db=20.0) for a, b in bursts],
                           windows=[np.zeros((2, 4, 4), np.float32) for _ in bursts])


def test_a_piece_of_a_stroke_is_not_called_none():
    events = [{"label": "swipe_left", "t_start": 3.00, "t_end": 3.40}]
    # burst 1 ends 0.1 s into the stroke (a quarter of it); burst 2 holds most of it; burst 3 is the hand going back
    stream = _stream([(80, 154), (158, 175), (190, 220)])
    old = g.label_segments(stream, events)
    assert [(e["label"], e["kind"]) for e in old] == [("none", "aux"), ("swipe_left", "stroke"), ("none", "aux")]
    new = g.label_segments(stream, events, drop_partial=True)
    assert [(e["label"], e["kind"]) for e in new] == [("swipe_left", "stroke"), ("none", "aux")]
    # nothing holds half of the stroke: no gesture picture, and the piece is not turned into a "none" either
    stream = _stream([(80, 154), (190, 220)])
    assert [(e["label"], e["kind"]) for e in g.label_segments(stream, events, drop_partial=True)] == [("none", "aux")]
    assert [(e["label"], e["kind"]) for e in g.label_segments(stream, events)] == [("none", "aux"), ("none", "aux")]

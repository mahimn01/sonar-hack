"""The camera-labelled recorder end to end, on the simulator with a fake camera that watches the simulated hand.

Checked here: the session folder is what the existing tools expect, the camera's labels land on the right
movements, the clock offset between camera and audio is found, nothing but hand points is saved, and the recorder
still works (with prompted labels) when there is no camera or no MediaPipe.
"""

import json
import sys

import numpy as np
import pytest

from sonar import camrecord
from sonar.camera import FakeCamera
from sonar.camlabel import CamFrame, LabelConfig, load_landmarks
from sonar.camrecord import AudioClock, LiveStrokes, PrivacyFilter, build_events, next_volunteer
from sonar.config import GESTURE_LABELS
from sonar.dataset import build_dataset, find_sessions, load_session, session_examples
from sonar.features import stream_features
from sonar.sim import HAND_REST, EchoSimulator, WaypointPath, gesture_waypoints, person_reflectors

SIM = ["--backend", "sim", "--fast", "--no-window", "--consent", "yes"]


def record(root, *extra: str) -> int:
    return camrecord.main([*SIM, "--out", str(root), *extra])


def only_session(root, script: str):
    folders = [f for f in find_sessions(root) if f.name.endswith("_" + script)]
    assert len(folders) == 1, folders
    return folders[0]


@pytest.fixture(scope="module")
def free_session(tmp_path_factory):
    """One free-form session: 2 sync pushes, then swipe left / swipe right / push twice each, camera 120 ms late."""
    root = tmp_path_factory.mktemp("cam")
    assert record(root, "--person", "v07", "--seed", "21", "--reps", "2", "--rounds", "1", "--fake-latency", "0.12") == 0
    return root


@pytest.mark.slow
def test_free_session_has_the_recorder_format_plus_camera_labels(cfg, free_session):
    folder = only_session(free_session, "free")
    meta, audio, session_cfg = load_session(folder)                    # the existing loader reads it unchanged
    assert session_cfg == cfg and audio.shape[1] == cfg.n_mics and abs(len(audio) - meta["n_samples"]) <= 4800
    for key in ("format", "session_id", "person", "script", "created", "profile", "config", "backend", "dropouts", "peak", "events"):
        assert key in meta, key
    assert meta["person"] == "v07" and meta["script"] == "free" and meta["recorder"] == "camrecord" and meta["synthetic"] is True
    events = meta["events"]
    assert events[0]["kind"] == "calibration" and all(isinstance(e["start"], int) and isinstance(e["end"], int) for e in events)
    assert all(0 <= e["start"] < e["end"] <= meta["n_samples"] for e in events)
    gestures = [e for e in events if e["kind"] == "gesture"]
    counts = {g: sum(1 for e in gestures if e["label"] == g) for g in ("swipe_left", "swipe_right", "push")}
    assert counts == {"swipe_left": 2, "swipe_right": 2, "push": 4}     # 2 sync pushes + 2 in the push block
    assert all(e["source"] == "camera" and e["tight"] is True and e["confidence"] >= 0.8 and e["label"] in GESTURE_LABELS for e in gestures)
    assert all(0.05 < (e["end"] - e["start"]) / cfg.fs < 0.9 for e in gestures)       # the stroke itself, not a 2.5 s window
    assert all(e["speed_mps"] > 0.5 and e["distance_m"] > 0.1 and len(e["direction"]) == 3 for e in gestures)
    # every camera gesture lies in the block that asked for it; blocks are "do not make negatives from me" zones
    blocks = [e for e in events if e["kind"] == "block"]
    assert len(blocks) == 4 and all(b["exclude"] is True and b["label"] == "block" and b["count"] == b["target"] == 2 for b in blocks)
    assert blocks[0]["intent"] == "push" and blocks[0].get("sync") is True
    for g in gestures:
        assert any(b["start"] <= g["t_peak"] <= b["end"] and b["intent"] == g["label"] for b in blocks), g
    # hands going back and wind-ups are kept as plain movements, under names older code never mistakes for gestures
    moves = [e for e in events if e["kind"] == "motion"]
    assert len(moves) >= 8 and {e["label"] for e in moves} <= {"return", "other"}
    # swipe directions are the user's
    assert all(e["direction"][0] > 0.2 for e in gestures if e["label"] == "swipe_right")
    assert all(e["direction"][0] < -0.2 for e in gestures if e["label"] == "swipe_left")
    # consent, camera and label blocks
    assert meta["consent"]["given"] is True and meta["consent"]["volunteer"] == "v07" and len(meta["consent"]["text"]) >= 4
    assert meta["camera"]["backend"] == "fake" and meta["camera"]["pictures_saved"] is False and meta["camera"]["video_saved"] is False
    assert meta["labels"]["source"] == "camera" and meta["labels"]["from_camera"] == 8 and meta["labels"]["review"] == 0
    assert meta["labels"]["physics_check"] == {"camera_gestures": 8, "with_sonar_burst": 8, "physics_agrees": 8}
    assert [p["type"] for p in meta["prompts"]] == ["calibration", "block", "block", "block", "block"]


@pytest.mark.slow
def test_clock_offset_is_measured_and_recorded(cfg, free_session):
    folder = only_session(free_session, "free")
    sync = json.loads((folder / "session.json").read_text())["sync"]
    assert sync["method"] == "motion" and sync["offset_ms"] == pytest.approx(120.0, abs=20.0)      # the fake camera was 120 ms late
    assert sync["residual_ms"] is not None and sync["residual_ms"] < 40.0 and sync["peak_corr"] > 0.6 and sync["n_strokes"] >= 6
    assert len(sync["audio_clock"]) >= 5 and "meaning" in sync
    # the measured offset is remembered per camera + microphone, for sessions in which it cannot be measured
    saved = json.loads((free_session / "camsync.json").read_text())
    assert saved[-1]["offset_ms"] == sync["offset_ms"] and saved[-1]["method"] == "motion"
    assert camrecord.load_sync_prior(free_session, saved[-1]["key"]) == pytest.approx(sync["offset_ms"] / 1e3)
    assert camrecord.load_sync_prior(free_session, "another camera") == 0.0


@pytest.mark.slow
def test_only_hand_points_are_saved(cfg, free_session):
    folder = only_session(free_session, "free")
    assert sorted(p.name for p in folder.iterdir()) == ["audio.wav", "landmarks.jsonl", "session.json"]      # no picture, no video
    header, frames = load_landmarks(folder / "landmarks.jsonl")
    meta = json.loads((folder / "session.json").read_text())
    assert header["width"] == 640 and header["fs"] == cfg.fs and header["hfov_deg"] == 70.0
    seconds = meta["n_samples"] / cfg.fs
    assert len(frames) == pytest.approx(30 * seconds, rel=0.05) and meta["camera"]["frames"] == len(frames)
    samples = np.array([f.sample for f in frames])
    assert np.all(np.diff(samples) > 0) and samples[-1] <= meta["n_samples"] + cfg.fs
    seen = [f for f in frames if f.hands]
    assert len(seen) > 0.8 * len(frames) and seen[0].hands[0].lm.shape == (21, 3) and seen[0].hands[0].world.shape == (21, 3)
    assert max(len(line) for line in (folder / "landmarks.jsonl").read_text().splitlines()) < 4000     # points, not pixels


@pytest.mark.slow
def test_existing_dataset_builder_uses_the_camera_labels(cfg, free_session):
    folder = only_session(free_session, "free")
    examples, report = session_examples(folder)
    assert report["gestures"] == 8 and report["missed"] == 0
    labels = [e.label for e in examples if e.source == "gesture"]
    assert sorted(labels) == ["push"] * 4 + ["swipe_left"] * 2 + ["swipe_right"] * 2
    # free-form blocks never turn into "none" examples: the volunteer may have gestured where the camera could not see
    assert report["none"] == 0 and all(e.source == "idle" for e in examples if e.label == "none")
    # the picture of each gesture is the burst that covers the camera-timed stroke
    meta = json.loads((folder / "session.json").read_text())
    stream = stream_features(load_session(folder)[1], cfg)
    for ev in (e for e in meta["events"] if e["kind"] == "gesture"):
        f0, f1 = stream.frame_of_sample(ev["start"]), stream.frame_of_sample(ev["end"])
        assert any(s.start <= f1 + 3 and s.end >= f0 - 1 for s in stream.segments)
    data = build_dataset([free_session], verbose=False)
    assert int(np.sum(data["y"] != 0)) == 8 and set(data["person"]) == {"v07"}


@pytest.mark.slow
def test_relabel_reproduces_the_labels_from_the_saved_files(free_session, capsys):
    folder = only_session(free_session, "free")
    before = json.loads((folder / "session.json").read_text())
    assert camrecord.main(["--relabel", str(folder)]) == 0
    after = json.loads((folder / "session.json").read_text())
    assert (folder / "session.json.orig").exists() and "relabelled" in after["sync"]
    pick = lambda m: [(e["label"], e["kind"]) for e in m["events"] if e["kind"] in ("gesture", "block", "calibration")]   # noqa: E731
    assert pick(after) == pick(before)
    a = [e["start"] for e in after["events"] if e["kind"] == "gesture"]
    b = [e["start"] for e in before["events"] if e["kind"] == "gesture"]
    assert np.max(np.abs(np.array(a) - np.array(b))) < 0.03 * 48000     # same strokes, same place (offset re-measured offline)
    assert after["sync"]["offset_ms"] == pytest.approx(before["sync"]["offset_ms"], abs=15.0)
    assert "relabelled" in capsys.readouterr().out
    assert camrecord.main(["--relabel", str(free_session)]) == 2        # not a camera session folder


@pytest.mark.slow
def test_strict_trials_get_camera_timing_and_keep_the_prompt_as_fallback(cfg, tmp_path):
    assert record(tmp_path, "--person", "v02", "--script", "gestures", "--reps", "1", "--seed", "5") == 0
    meta = json.loads((only_session(tmp_path, "gestures") / "session.json").read_text())
    prompts = [e for e in meta["events"] if e["kind"] == "prompt"]
    gestures = [e for e in meta["events"] if e["kind"] == "gesture"]
    assert len(prompts) == 3 and all(p["exclude"] and p["label"] == "prompt" for p in prompts)
    assert sorted(g["label"] for g in gestures) == ["push", "swipe_left", "swipe_right"] and all(g["source"] == "camera" for g in gestures)
    for g in gestures:                                                  # each camera-timed stroke sits inside its own trial window
        assert any(p["intent"] == g["label"] and p["start"] - 0.2 * cfg.fs <= g["t_peak"] <= p["end"] + 0.5 * cfg.fs for p in prompts)
    # a trial the camera did not see falls back to the plain prompt window, exactly what sonar.record writes
    fallback = build_events(meta["prompts"], None, cfg.fs, meta["n_samples"])
    plain = [e for e in fallback if e["kind"] == "gesture"]
    assert len(plain) == 3 and all(e["source"] == "prompt" and "tight" not in e and (e["end"] - e["start"]) / cfg.fs == pytest.approx(2.5, abs=0.2)
                                   for e in plain)


@pytest.mark.slow
def test_without_a_camera_it_records_prompted_labels(cfg, tmp_path, capsys):
    assert record(tmp_path, "--person", "v03", "--camera", "none", "--reps", "1", "--seed", "3") == 0
    out = capsys.readouterr().out
    assert "NO CAMERA LABELS" in out and "falling back" in out
    folder = only_session(tmp_path, "free")
    assert sorted(p.name for p in folder.iterdir()) == ["audio.wav", "session.json"]
    meta = json.loads((folder / "session.json").read_text())
    gestures = [e for e in meta["events"] if e["kind"] == "gesture"]
    assert sorted(g["label"] for g in gestures) == ["push", "swipe_left", "swipe_right"] and all(g["source"] == "prompt" for g in gestures)
    assert meta["labels"]["source"] == "prompt" and meta["camera"]["backend"] == "none" and "sync" not in meta
    examples, report = session_examples(folder)
    assert report["gestures"] == 3


@pytest.mark.slow
def test_without_mediapipe_it_says_so_and_falls_back(cfg, tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", None)                 # "import mediapipe" now fails, as on a machine without it
    assert record(tmp_path, "--person", "v04", "--camera", "0", "--script", "gestures", "--reps", "1", "--seed", "4",
                  "--gestures", "push") == 0
    out = capsys.readouterr().out
    assert "NO CAMERA LABELS" in out and "MediaPipe is not installed" in out
    meta = json.loads((only_session(tmp_path, "gestures") / "session.json").read_text())
    assert [e["label"] for e in meta["events"] if e["kind"] == "gesture"] == ["push"]


@pytest.mark.slow
def test_negatives_session_is_all_background(cfg, tmp_path):
    assert record(tmp_path, "--person", "v05", "--script", "negatives", "--negatives-scale", "0.25", "--seed", "8") == 0
    folder = only_session(tmp_path, "negatives")
    meta = json.loads((folder / "session.json").read_text())
    scenes = [e for e in meta["events"] if e["kind"] == "scene"]
    assert [e["activity"] for e in scenes] == ["typing", "reach", "fidget", "lean", "point", "walk"]
    assert all(e["label"] == "distractor" for e in scenes)             # a label the trainer already treats as "none"
    pushes = [e for e in meta["events"] if e["kind"] == "gesture"]
    assert [e["label"] for e in pushes] == ["push", "push"]            # only the two clock-sync pushes at the start
    assert all(e["end"] < scenes[0]["start"] for e in pushes) and meta["sync"]["method"] == "motion"
    examples, report = session_examples(folder)
    assert report["gestures"] == 2 and all(e.label in ("none", "push") for e in examples)


def test_audio_clock_follows_late_blocks_and_recovers_after_a_dropout():
    fs = 48000
    rng = np.random.default_rng(0)
    clock = AudioClock(fs)
    assert clock.sample_at(123.0) == 0.0
    t0, n = 1000.0, 0
    for k in range(400):                                                # 16 s of 40 ms blocks, each delivered 0-25 ms late
        n += 1920
        clock.add(t0 + n / fs + 0.004 + rng.uniform(0.0, 0.025), n)
    assert clock.sample_at(t0 + 10.0) == pytest.approx((10.0 - 0.004) * fs, abs=0.002 * fs)
    assert len(clock.log) >= 15 and clock.log[0][1] == 1920
    clock.reset()                                                       # 0.5 s of audio lost
    for k in range(50):
        n += 1920
        clock.add(t0 + 0.5 + n / fs + 0.004 + rng.uniform(0.0, 0.025), n)
    assert clock.sample_at(t0 + 0.5 + n / fs) == pytest.approx(n - 0.004 * fs, abs=0.003 * fs)


def test_privacy_filter_removes_speech_and_leaves_the_sonar_untouched(cfg):
    sim = EchoSimulator(cfg, seed=4, cn0_db=70.0, ambient_dbfs=-25.0)   # loud hall noise in the audible band
    wp = gesture_waypoints("swipe_right", 4.0)
    path = WaypointPath([0.0] + [w[0] for w in wp], [HAND_REST] + [w[1] for w in wp])
    from sonar.sim import hand_reflector

    audio = sim.render(8 * cfg.fs, hand_reflector(path, echo_db=-22.0) + person_reflectors(0.65))
    filt = PrivacyFilter(cfg, cfg.n_mics)
    parts = [filt.process(audio[i:i + 1920]) for i in range(0, len(audio), 1920)]
    clean = np.concatenate(parts + [filt.flush()])
    assert clean.shape == audio.shape and filt.stop_hz >= 11000.0       # same length: every label still points at the right sample

    def band_db(x: np.ndarray, lo: float, hi: float) -> float:
        seg = x[cfg.fs:5 * cfg.fs, 0].astype(np.float64)
        spec = np.abs(np.fft.rfft(seg * np.blackman(len(seg)))) ** 2     # windowed: the loud hall noise must not smear upwards
        f = np.fft.rfftfreq(4 * cfg.fs, 1 / cfg.fs)
        return float(10 * np.log10(spec[(f >= lo) & (f < hi)].sum() + 1e-30))

    assert band_db(clean, 100, 8000) < band_db(audio, 100, 8000) - 60.0          # speech band: gone
    assert abs(band_db(clean, 18000, 21600) - band_db(audio, 18000, 21600)) < 0.05   # sonar band: untouched
    a, b = stream_features(audio, cfg), stream_features(clean, cfg)
    assert [(s.start, s.end) for s in a.segments] == [(s.start, s.end) for s in b.segments] and len(a.segments) >= 1
    assert float(np.max(np.abs(a.feat - b.feat))) < 0.02


def test_live_counter_reports_each_stroke_once():
    times, pts = [0.0], [HAND_REST]
    for t0, label in ((1.0, "swipe_left"), (4.0, "push")):
        for t, p in gesture_waypoints(label, t0):
            times.append(t)
            pts.append(p)
    now = [0.0]
    cam = FakeCamera(WaypointPath(times, pts), lambda: now[0], latency_s=0.0)
    live = LiveStrokes(cam.width, cam.height, 48000, LabelConfig())
    seen: list[str] = []
    for step in range(1, 71):                                           # the session advancing 0.1 s at a time
        now[0] = 0.1 * step
        while (cf := cam.read()) is not None:
            live.add(CamFrame(0, cf.sample / 48000, cf.sample, cf.hands))
        seen += [s.kind for s in live.poll()]
    assert seen.count("swipe_left") == 1 and seen.count("push") == 1


def test_volunteer_numbers(tmp_path):
    assert next_volunteer(tmp_path) == "v01"
    (tmp_path / "20261114-101500_v01_free").mkdir()
    (tmp_path / "20261114-101700_v12_negatives").mkdir()
    (tmp_path / "20261114-101900_alice_gestures").mkdir()
    assert next_volunteer(tmp_path) == "v13"


@pytest.mark.slow
def test_camera_that_never_sees_the_hand_switches_to_timed_prompts(cfg, tmp_path, capsys, monkeypatch):
    """Hand outside the picture for the whole session: no endless waiting, the prompts label the gestures."""
    from sonar import camera as camera_mod

    seeing = camera_mod.FakeCamera.read

    def blind(self):                                                              # every picture arrives, none shows a hand
        frame = seeing(self)
        if frame is not None:
            frame.hands = []
        return frame

    monkeypatch.setattr(camera_mod.FakeCamera, "read", blind)
    assert record(tmp_path, "--person", "v06", "--reps", "1", "--rounds", "1", "--seed", "6", "--gestures", "push,swipe_left") == 0
    assert "the camera did not see the gestures" in capsys.readouterr().out
    meta = json.loads((only_session(tmp_path, "free") / "session.json").read_text())
    blocks = [e for e in meta["events"] if e["kind"] == "block"]
    assert len(blocks) == 1 and blocks[0]["count"] == 0 and blocks[0]["exclude"] is True      # the sync block nobody saw
    gestures = [e for e in meta["events"] if e["kind"] == "gesture"]
    assert sorted(g["label"] for g in gestures) == ["push", "swipe_left"] and all(g["source"] == "prompt" for g in gestures)
    assert meta["sync"]["method"] == "timestamps" and meta["labels"]["from_camera"] == 0


def test_screen_is_drawn_without_a_window(cfg):
    pytest.importorskip("cv2")
    from sonar.camera import PinholeCamera, hand_points
    from sonar.camlabel import HandObs
    from sonar.camrecord import Screen

    lm, world, _ = PinholeCamera().project(hand_points(np.array([0.05, 0.3, 0.2])))
    view = {"title": "SWIPE LEFT", "text": "Sweep your open hand from RIGHT to LEFT", "hint": "camera saw: swipe left", "count": (2, 3),
            "progress": 0.4, "flash": "OK", "sides": True, "lines": ["line one", "line two"], "status": "status", "footer": "footer"}
    screen = Screen(window=False)
    assert not screen.window and screen.render(None, [], view) == -1               # no window: nothing shown, no key
    picture = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
    canvas = screen.compose(picture, [HandObs("right", 0.9, lm, world)], view)
    assert canvas.shape == (720, 960, 3) and canvas.dtype == np.uint8
    blank = screen.compose(None, [], {"title": "SIT STILL"})
    assert blank.shape == (720, 960, 3) and int(blank.max()) > 200                 # the white title on the dark background
    hidden = Screen(window=False, show_video=False).compose(picture, [], {})
    assert int(hidden[300:400].max()) < 40                                         # --no-video-preview: the picture is not shown

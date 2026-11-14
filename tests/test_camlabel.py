"""The camera auto-labeller on rendered hand tracks: scripted gestures in, labels with the right time and direction out.

No camera and no MediaPipe needed: sonar.camera.FakeCamera draws the 21 hand points of a hand that follows a known
path, exactly as the real tracker would report them (picture coordinates plus the hand's shape in metres).
"""

import json

import numpy as np
import pytest

from sonar.camera import FakeCamera, OpenCVCamera, PinholeCamera, hand_points
from sonar.camlabel import (
    DSP_DELAY_S, CamEvent, CamFrame, HandObs, Intent, LabelConfig, all_strokes, build_tracks, estimate_offset,
    label_session, label_strokes, landmark_header, load_landmarks, palm_state, speed_curve, write_landmarks,
)
from sonar.features import stream_features
from sonar.sim import HAND_REST, EchoSimulator, WaypointPath, gesture_stroke_window, gesture_waypoints, hand_reflector, person_reflectors

FS = 48000
SCRIPT = [(3.0, "swipe_right", 1.0), (7.0, "push", 1.0), (11.0, "swipe_left", 1.2), (15.0, "pull", 1.0),
          (19.0, "swipe_right", 0.8), (23.0, "push", 1.3)]


def scripted_path(script=SCRIPT) -> WaypointPath:
    times, pts = [0.0], [HAND_REST]
    for t0, label, speed in script:
        for t, p in gesture_waypoints(label, t0, speed):
            if t > times[-1]:
                times.append(t)
                pts.append(p)
    return WaypointPath(times, pts)


def film(path, seconds: float, **kwargs) -> tuple[FakeCamera, list[CamFrame]]:
    """Every picture the fake camera takes of `path` in `seconds`."""
    cam = FakeCamera(path, lambda: seconds, fs=FS, **kwargs)
    frames = []
    while (cf := cam.read()) is not None:
        frames.append(CamFrame(len(frames), cf.sample / FS, cf.sample, cf.hands))
    return cam, frames


def gestures_of(events: list[CamEvent]) -> list[CamEvent]:
    return [e for e in events if e.kind == "gesture"]


def test_palm_position_comes_out_in_metres():
    """The hand's known size gives its distance, so positions are real metres whatever the picture size."""
    cam = PinholeCamera(hfov_deg=70.0)
    for palm in ([0.0, 0.25, 0.20], [0.15, 0.40, 0.25], [-0.10, 0.18, 0.18]):
        lm, world, inside = cam.project(hand_points(np.array(palm)))
        pos, centre, edge = palm_state(HandObs("right", 0.9, lm, world), cam.width, cam.height, LabelConfig(hfov_deg=70.0))
        want = np.array(palm) - np.array(cam.pos)                      # right, away from the screen, up
        np.testing.assert_allclose(pos, want, atol=0.012)
        assert 0.0 <= edge <= 1.0 and inside > 0.7
    # the assumed field of view only scales the distance to the screen, never the sideways position
    lm, world, _ = cam.project(hand_points(np.array([0.15, 0.40, 0.25])))
    wide, _, _ = palm_state(HandObs("right", 0.9, lm, world), cam.width, cam.height, LabelConfig(hfov_deg=85.0))
    assert wide[0] == pytest.approx(0.15, abs=0.012) and wide[1] < 0.40
    # without the metric hand shape the nominal hand length is used: rougher, same ballpark
    rough, _, _ = palm_state(HandObs("right", 0.9, lm, None), cam.width, cam.height, LabelConfig())
    assert rough[1] == pytest.approx(0.47, abs=0.08)


def test_scripted_gestures_are_named_and_timed_without_prompts():
    _, frames = film(scripted_path(), 28.0, latency_s=0.0, seed=1)
    strokes, _ = all_strokes(build_tracks(frames, 640, 480, FS))
    found = gestures_of(label_strokes(strokes))
    assert [e.label for e in found] == [label for _, label, _ in SCRIPT]
    for ev, (t0, label, speed) in zip(found, SCRIPT):
        a, b = gesture_stroke_window(label, t0, speed)
        if label == "push":                                            # the scripted push is thrust AND return; the label is the thrust
            b = a + (b - a) * 0.28 / 0.60
        assert a - 0.03 <= ev.t0 <= a + 0.12 and b - 0.12 <= ev.t1 <= b + 0.03, (label, ev.t0, ev.t1, a, b)
        assert ev.confidence >= 0.8 and ev.seen >= 0.9
        # top speed of the scripted stroke (smooth start and stop: 1.875 x distance / time). The camera reads a
        # little low on short strokes because the path is smoothed, most of all towards the screen (the noisy axis)
        true_peak = 1.875 * {"swipe_right": 0.40 / 0.40, "swipe_left": 0.40 / 0.40, "push": 0.22 / 0.28, "pull": 0.26 / 0.30}[label] * speed
        assert 0.6 * true_peak <= ev.speed_mps <= 1.1 * true_peak, (label, ev.speed_mps, true_peak)
    # travel in metres: the scripted swipe crosses 40 cm (the picture edge hides a little of it), the push 22 cm
    assert found[0].distance_m == pytest.approx(0.38, abs=0.05) and found[1].distance_m == pytest.approx(0.20, abs=0.04)
    assert found[0].direction[0] > 0.3 and found[2].direction[0] < -0.3 and found[1].direction[1] < -0.15
    # wind-ups, hands going back and the hand rising from the lap are movements, never gestures
    others = [e for e in label_strokes(strokes) if e.kind == "motion"]
    assert len(others) >= 10 and all(e.label in ("return", "other") for e in others)


def test_prompts_decide_which_stroke_is_the_gesture():
    """In a 'swipe left' block every leftward stroke is the gesture and every rightward one is the hand going back."""
    script = [(2.0, "swipe_left", 1.0), (5.0, "swipe_left", 1.1)]
    _, frames = film(scripted_path(script), 9.0, latency_s=0.0)
    strokes, _ = all_strokes(build_tracks(frames, 640, 480, FS))
    events = label_strokes(strokes, [Intent(1.5, 8.5, "swipe_left")])
    assert [e.label for e in gestures_of(events)] == ["swipe_left", "swipe_left"]
    assert all(e.intent == "swipe_left" and e.confidence > 0.9 for e in gestures_of(events))
    # the hand travelling back to the right afterwards is "the hand going back", not a swipe right
    back = [e for e in events if e.direction[0] > 0.2]
    assert len(back) == 2 and all(e.kind == "motion" and e.label == "return" for e in back)
    # the same strokes in a swipe-RIGHT block: the camera disagrees with the prompt, so nothing is trusted
    wrong = label_strokes(strokes, [Intent(1.5, 8.5, "swipe_right")])
    assert all(e.label == "return" for e in wrong if e.direction[0] < -0.3)
    # a part where no gesture should happen ("negatives"): gesture-like strokes are flagged, never trusted
    flagged = [e for e in label_strokes(strokes, [Intent(0.0, 9.0, "none")]) if e.kind == "review"]
    assert len(flagged) >= 2 and all(e.candidate in ("swipe_left", "swipe_right") and e.confidence < 0.5 for e in flagged)
    assert not gestures_of(label_strokes(strokes, [Intent(0.0, 9.0, "none")]))
    row = flagged[0].to_session_event(FS, n_samples=9 * FS)
    assert row["label"] == "review" and row["exclude"] is True and row["kind"] == "review" and "tight" not in row


def test_direction_is_the_users_and_a_wrong_mirror_setting_flips_it():
    script = [(2.0, "swipe_right", 1.0)]
    _, frames = film(scripted_path(script), 6.0, latency_s=0.0)
    strokes, _ = all_strokes(build_tracks(frames, 640, 480, FS))
    assert [e.label for e in gestures_of(label_strokes(strokes))] == ["swipe_right"]
    # the same pictures NOT turned into the user's view (what a raw webcam picture looks like)
    raw = [CamFrame(f.index, f.t, f.sample, [HandObs(h.hand, h.score, h.lm * [-1, 1, 1] + [1, 0, 0], h.world * [-1, 1, 1]) for h in f.hands])
           for f in frames]
    strokes, _ = all_strokes(build_tracks(raw, 640, 480, FS))
    assert [e.label for e in gestures_of(label_strokes(strokes))] == ["swipe_left"]
    # which is why a normal webcam facing the user is flipped once, and an already mirrored one is not
    assert OpenCVCamera(0, facing="user").flip and not OpenCVCamera(0, facing="user", mirrored_input=True).flip
    assert not OpenCVCamera(0, facing="world").flip


def test_motion_blur_and_a_hand_leaving_the_picture_lower_the_confidence_not_the_label():
    _, frames = film(scripted_path(), 28.0, latency_s=0.0, blur_speed=0.9, seed=3)
    assert sum(1 for f in frames if not f.hands) > 20                   # the fast part of most strokes went unseen
    strokes, _ = all_strokes(build_tracks(frames, 640, 480, FS))
    intents = [Intent(t0 - 0.5, t0 + 3.0, label) for t0, label, _ in SCRIPT]
    prompted = [e for e in label_strokes(strokes, intents) if e.kind in ("gesture", "review") and e.label == e.intent]
    assert [e.label for e in prompted] == [label for _, label, _ in SCRIPT]
    assert min(e.seen for e in prompted) < 0.7 and all(e.confidence < 0.97 for e in prompted if e.seen < 0.7)
    for ev, (t0, label, speed) in zip(prompted, SCRIPT):
        a, b = gesture_stroke_window(label, t0, speed)
        assert a - 0.1 <= ev.t_peak <= b + 0.1
    # a swipe that carries the hand out of the picture: the camera sees the start only
    def out_of_view(t: np.ndarray) -> np.ndarray:
        x = np.clip((t - 2.0) / 0.4, 0.0, 1.0) * 0.6 - 0.05            # 60 cm to the right in 0.4 s: gone after ~25 cm
        return np.stack([x, np.full_like(t, 0.30), np.full_like(t, 0.22)], axis=1)
    _, frames = film(out_of_view, 5.0, latency_s=0.0)
    strokes, _ = all_strokes(build_tracks(frames, 640, 480, FS))
    cut = [s for s in strokes if s.kind == "swipe_right"]
    assert len(cut) == 1 and cut[0].cut_end and cut[0].edge > 0.5
    ev = gestures_of(label_strokes(strokes)) or [e for e in label_strokes(strokes) if e.kind == "review"]
    assert ev[0].label == "swipe_right" and ev[0].confidence < 0.9


def test_two_hands_are_followed_separately():
    path = scripted_path([(2.0, "swipe_left", 1.0)])
    _, frames = film(path, 6.0, latency_s=0.0)
    still = PinholeCamera().project(hand_points(np.array([-0.16, 0.45, 0.20]), hand="left"))
    for f in frames:                                                    # a second, resting hand in every picture
        f.hands.append(HandObs("left", 0.9, still[0], still[1]))
    tracks = build_tracks(frames, 640, 480, FS)
    assert sorted(t.hand for t in tracks) == ["left", "right"]
    strokes, _ = all_strokes(tracks)
    assert [e.label for e in gestures_of(label_strokes(strokes))] == ["swipe_left"]
    assert all(s.hand == "right" for s in strokes)


def test_landmark_file_round_trip(tmp_path):
    _, frames = film(scripted_path([(1.0, "push", 1.0)]), 4.0, latency_s=0.05)
    frames[3].pose = np.zeros((8, 4))
    path = tmp_path / "landmarks.jsonl"
    write_landmarks(path, landmark_header(640, 480, FS, hfov_deg=70.0), frames)
    with open(path, "a") as fh:
        fh.write('{"i": 999, "t": 1.0, "sam')                         # a crash while writing the last line
    header, back = load_landmarks(path)
    assert header["format"] == "sonar-hand-landmarks" and header["width"] == 640 and len(back) == len(frames)
    assert back[3].pose is not None and back[3].pose.shape == (8, 4)
    np.testing.assert_allclose(back[10].hands[0].lm, frames[10].hands[0].lm, atol=1e-4)
    assert abs(back[10].sample - frames[10].sample) <= 0.5 and back[10].hands[0].hand == "right"
    first = json.loads(path.read_text().splitlines()[1])
    assert set(first) == {"i", "t", "sample", "hands"}                 # hand points only: nothing that could hold a picture


def test_session_event_format():
    ev = CamEvent("swipe_left", "gesture", 4.0, 4.4, 4.2, 0.93, 1.5, 0.31, [-0.31, 0.01, 0.0], "right", 1.0, intent="swipe_left")
    row = ev.to_session_event(FS, n_samples=10 * FS, rep=3)
    assert row["label"] == "swipe_left" and row["kind"] == "gesture" and row["tight"] is True and row["rep"] == 3
    assert row["start"] == 4 * FS and row["end"] == int(4.4 * FS) and row["source"] == "camera" and row["confidence"] == 0.93
    assert isinstance(row["start"], int) and "exclude" not in row
    other = CamEvent("other", "motion", 9.9, 10.5, 10.0, 0.0, 0.5, 0.2, [0, 0, 0.2], "right", 1.0).to_session_event(FS, 10 * FS)
    assert other["label"] == "other" and other["end"] <= 10 * FS and "tight" not in other


def test_clock_offset_from_two_movement_curves():
    """Slide a made-up 'hand speed' curve against a delayed copy: the delay must come back."""
    rng = np.random.default_rng(0)
    n = 1500                                                            # 30 s at 50 frames per second
    cam = np.zeros(n)
    for c in rng.uniform(100, 1400, size=12):
        cam += 1.5 * np.exp(-0.5 * ((np.arange(n) - c) / 6.0) ** 2)
    for true_ms in (-120.0, 0.0, 95.0, 260.0):
        lag = (true_ms / 1e3 - DSP_DELAY_S) * 50.0                      # the sonar curve runs DSP_DELAY_S behind the movement
        sonar = 12.0 * np.interp(np.arange(n) + lag, np.arange(n), cam) + rng.normal(0, 0.3, n)
        res = estimate_offset(cam, sonar)
        assert res.method == "motion" and res.offset_s * 1e3 == pytest.approx(true_ms, abs=6.0) and res.peak_corr > 0.9
    # nothing in common: say so instead of inventing an offset
    res = estimate_offset(cam, rng.normal(0, 1, n), prior_s=0.07)
    assert res.method == "timestamps" and res.offset_s == 0.07
    res = estimate_offset(np.zeros(n), rng.normal(0, 1, n))
    assert res.method == "timestamps" and "no hand movement" in res.note


@pytest.mark.slow
def test_labels_line_up_with_the_simulated_sonar(cfg):
    """Camera pictures stamped 140 ms late against the simulated microphones: the offset is found and removed."""
    script = [(3.0, "swipe_right", 1.0), (6.5, "push", 1.0), (10.0, "swipe_left", 1.1), (13.5, "push", 1.2), (17.0, "swipe_right", 0.9)]
    path = scripted_path(script)
    sim = EchoSimulator(cfg, seed=2, cn0_db=70.0)
    audio = sim.render(21 * cfg.fs, hand_reflector(path, echo_db=-20.0) + person_reflectors(0.65, seed=2))
    stream = stream_features(audio, cfg)
    activity = np.concatenate([[0.0], stream.activity_db])
    cam, frames = film(path, 21.0, latency_s=0.140, seed=5)
    intents = [Intent(t0 - 0.3, t0 + 2.8, label) for t0, label, _ in script]
    res = label_session(frames, landmark_header(cam.width, cam.height, cfg.fs, hfov_deg=70.0), activity, intents)
    assert res.sync.method == "motion" and res.sync.offset_s * 1e3 == pytest.approx(140.0, abs=20.0)
    assert res.sync.residual_s is not None and res.sync.residual_s < 0.04 and res.sync.peak_corr > 0.6
    found = gestures_of(res.events)
    assert [e.label for e in found] == [label for _, label, _ in script]
    for ev, (t0, label, speed) in zip(found, script):
        a, b = gesture_stroke_window(label, t0, speed)
        assert a - 0.06 <= ev.t0 and ev.t_peak <= b + 0.03 and a <= ev.t_peak, (label, ev.t0, ev.t_peak, a, b)
        # ... and the sonar's own movement burst covers the camera-timed stroke
        f0, f1 = int(ev.t0 * 50) + 1, int(ev.t1 * 50) + 1
        assert any(s.start <= f1 + 3 and s.end >= f0 - 1 for s in stream.segments), (label, f0, f1)
    # without the sonar curve the camera's own time stamps are used as they are, and that is said
    alone = label_session(frames, landmark_header(cam.width, cam.height, cfg.fs), None, intents)
    assert alone.sync.method == "timestamps" and gestures_of(alone.events)[0].t0 == pytest.approx(found[0].t0 + res.sync.offset_s, abs=0.01)
    assert res.counts()["swipe_right"] == 2 and speed_curve([], 10).sum() == 0.0

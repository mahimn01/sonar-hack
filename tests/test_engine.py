"""Realtime engine: events on a simulated visitor, anti-jitter rules, JSON schema, WebSocket."""

import json
import socket
import threading
import time

import numpy as np
import pytest

from sonar.audio import SimAudio
from sonar.engine import SCHEMA_VERSION, Engine, GestureDecider
from sonar.gestures import Segment
from sonar.server import EventServer, to_json

REQUIRED = {
    "gesture": {"label", "confidence", "source", "probs", "duration_s", "peak_db"},
    "presence": {"state", "confidence", "range_m", "moving", "motionless_s", "changed"},
    "breathing": {"bpm", "confidence", "state", "waveform", "fs", "range_m", "amplitude_mm", "window_s", "apnea", "motionless_s"},
    "status": {"snr_db", "carrier_hz", "fps", "calibrating", "locked", "drift_ppm", "clipped", "dropped", "classifier",
               "backend", "source", "profile", "latency_ms", "chirp_band_hz", "tones_hz"},
    "spectrogram": {"rows", "n_bins", "bins_hz", "frame_s"},
}


@pytest.fixture(scope="module")
def demo_events(cfg):
    """One 62 s loop of the scripted visitor through the engine (faster than real time)."""
    backend = SimAudio(cfg, scene="demo", realtime=False)
    engine = Engine(cfg, backend, profile="default")
    events = []
    for _ in range(int(60 * cfg.fs / backend.block)):
        events += engine.step(backend.read())
    return events


@pytest.mark.slow
def test_demo_visitor_produces_the_expected_story(demo_events):
    gestures = [e["label"] for e in demo_events if e["type"] == "gesture"]
    assert gestures == ["swipe_right", "push", "swipe_left"]              # exactly once each, in order
    states = [e["state"] for e in demo_events if e["type"] == "presence" and e["changed"]]
    assert states == ["entering", "present", "empty"]
    rates = [e["bpm"] for e in demo_events if e["type"] == "breathing" and e["bpm"] is not None]
    assert rates and abs(np.median(rates) - 15.0) < 1.0                    # the visitor breathes at 15/min
    assert all(e["confidence"] >= 0.6 for e in demo_events if e["type"] == "gesture")
    status = [e for e in demo_events if e["type"] == "status"]
    assert status[-1]["locked"] and not status[-1]["calibrating"] and status[-1]["snr_db"] > 30
    assert status[0]["calibrating"]                                        # the first seconds are calibration


@pytest.mark.slow
def test_events_follow_the_documented_schema(demo_events):
    seq = [e["seq"] for e in demo_events]
    assert seq == sorted(seq) and len(set(seq)) == len(seq)
    kinds = {e["type"] for e in demo_events}
    assert kinds == set(REQUIRED)
    for e in demo_events:
        assert REQUIRED[e["type"]] <= set(e), f"{e['type']} lacks {REQUIRED[e['type']] - set(e)}"
        assert isinstance(e["t"], float)
        json.loads(to_json(e))                                             # every event is plain JSON
    spec = next(e for e in demo_events if e["type"] == "spectrogram")
    assert len(spec["rows"]) == 5 and len(spec["rows"][0]) == spec["n_bins"] == len(spec["bins_hz"]) == 64
    assert spec["bins_hz"][0] == -250.0 and spec["bins_hz"][32] == 0.0
    assert all(0.0 <= v <= 1.0 for row in spec["rows"] for v in row)
    wave = next(e for e in demo_events if e["type"] == "breathing" and e["bpm"] is not None)
    assert len(wave["waveform"]) <= 300 and max(abs(v) for v in wave["waveform"]) <= 1.0
    assert wave["fs"] == 10.0


class FakeClassifier:
    source = "model"

    def __init__(self):
        self.next = ("none", 0.0)

    def predict(self, window):
        label, conf = self.next
        return label, conf, {label: conf}


def burst(end, peak=10.0):
    return Segment(start=end - 20, end=end, decided=end + 10, peak_db=peak, energy_db=50.0, forced=False)


def feed(decider, clf, label, conf, end, peak=10.0):
    clf.next = (label, conf)
    return [e["label"] for e in decider.on_segment(end + 10, burst(end, peak), np.zeros((5, 64, 80), np.float32))]


def drain(decider, upto):
    out = []
    for i in range(upto - 60, upto):
        out += [e["label"] for e in decider.tick(i, False)]
    return out


def test_decider_rules(cfg):
    clf = FakeClassifier()
    d = GestureDecider(cfg, clf, min_confidence=0.6, swipe_hold_s=0.3)
    assert feed(d, clf, "push", 0.5, 100) == []                            # below the confidence threshold
    assert feed(d, clf, "none", 0.99, 150) == []                           # "none" is never an event
    assert feed(d, clf, "push", 0.9, 200) == ["push"]                      # push/pull are reported at once
    assert feed(d, clf, "push", 0.9, 210) == []                            # dead time (0.4 s)
    assert feed(d, clf, "pull", 0.9, 240) == []                            # the hand coming back (1.2 s lock-out)
    assert feed(d, clf, "pull", 0.9, 300) == ["pull"]                      # ... but later it is a real pull


def test_decider_tells_wind_up_from_stroke(cfg):
    clf = FakeClassifier()
    d = GestureDecider(cfg, clf, min_confidence=0.6, swipe_hold_s=0.3)
    # wind-up to the left, then the real (stronger) stroke to the right -> only "swipe_right"
    assert feed(d, clf, "swipe_left", 0.9, 100, peak=8.0) == []            # held, not reported yet
    assert feed(d, clf, "swipe_right", 0.9, 125, peak=12.0) == []
    assert drain(d, 220) == ["swipe_right"]
    # a stroke, then the clearly weaker return -> only the stroke
    assert feed(d, clf, "swipe_left", 0.9, 400, peak=12.0) == []
    assert feed(d, clf, "swipe_right", 0.9, 425, peak=7.0) == ["swipe_left"]
    assert drain(d, 520) == []
    # a single swipe is released after the hold time
    assert feed(d, clf, "swipe_right", 0.9, 700) == []
    assert drain(d, 780) == ["swipe_right"]


def test_engine_commands(cfg):
    engine = Engine(cfg, SimAudio(cfg, scene="empty", realtime=False))
    assert engine.handle_command({"cmd": "calibrate"}) == {"type": "ack", "cmd": "calibrate"}
    assert engine.handle_command({"cmd": "set_min_confidence", "value": 0.8})["value"] == 0.8
    assert engine.handle_command({"cmd": "nope"})["type"] == "error"
    hello = engine.hello()
    assert hello["schema"] == SCHEMA_VERSION and hello["labels"] == ["swipe_left", "swipe_right", "push", "pull"]


def test_websocket_broadcast_and_commands(cfg):
    from websockets.sync.client import connect

    with socket.socket() as s:                                             # a free local port
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    engine = Engine(cfg, SimAudio(cfg, scene="empty", realtime=False))
    server = EventServer("127.0.0.1", port, on_command=engine.handle_command, hello=engine.hello)
    server.start()
    try:
        with connect(f"ws://127.0.0.1:{port}") as ws:
            assert json.loads(ws.recv(timeout=3))["type"] == "hello"       # greeting with the static info
            ws.send(json.dumps({"cmd": "ping"}))
            assert json.loads(ws.recv(timeout=3)) == {"type": "pong"}
            ws.send(json.dumps({"cmd": "calibrate"}))
            assert json.loads(ws.recv(timeout=3))["type"] == "ack"
            ws.send("not json")
            assert json.loads(ws.recv(timeout=3))["type"] == "error"
            time.sleep(0.1)
            assert server.n_clients == 1
            server.broadcast({"type": "gesture", "label": "push", "confidence": np.float32(0.9), "t": 1.0})
            got = json.loads(ws.recv(timeout=3))
            assert got["label"] == "push" and abs(got["confidence"] - 0.9) < 1e-6
    finally:
        server.stop()


def test_engine_thread_and_callback_api(cfg):
    """start()/stop() with the realtime simulator: events arrive on the callback."""
    engine = Engine(cfg, SimAudio(cfg, scene="empty", realtime=True))
    seen = []
    done = threading.Event()

    def on_event(e):
        seen.append(e["type"])
        if e["type"] == "status":
            done.set()

    engine.on_event(on_event)
    engine.start()
    try:
        assert done.wait(timeout=6.0), "no status event within 6 s"
    finally:
        engine.stop()
    assert "status" in seen and engine.error is None

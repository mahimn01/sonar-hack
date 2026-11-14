"""Live audio backend logic, exercised with a fake sound library (no devices are opened)."""

import sys
import types

import numpy as np
import pytest

from sonar import audio


class FakeStream:
    live: list = []
    fail = staticmethod(lambda kind, blocksize: False)

    def __init__(self, kind, **kw):
        self.kind, self.kw, self.latency = kind, kw, 0.04
        if FakeStream.fail(kind, kw["blocksize"]):
            raise RuntimeError(f"cannot open {kind} at {kw['blocksize']}")
        FakeStream.live.append(self)

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        FakeStream.live.remove(self)


@pytest.fixture
def live(cfg, monkeypatch):
    fake = types.SimpleNamespace(
        WasapiSettings=lambda exclusive=False: ("wasapi", exclusive),
        InputStream=lambda **kw: FakeStream("in", **kw),
        OutputStream=lambda **kw: FakeStream("out", **kw),
        Stream=lambda **kw: FakeStream("duplex", **kw),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    FakeStream.live = []
    a = audio.LiveAudio.__new__(audio.LiveAudio)
    a.cfg = cfg
    a._in_dev = {"index": 9, "name": "Microphone Array (Intel)", "in": 2}
    a._out_dev = {"index": 8, "name": "Speaker (Realtek)", "out": 2}
    a.n_mics, a.exclusive, a.blocksize, a.latency, a.mode = 2, True, 1920, "high", "auto"
    a._streams, a._last_rx, a.info = [], 0.0, {}
    return a


def test_separate_exclusive_streams_are_the_first_choice(live):
    FakeStream.fail = staticmethod(lambda kind, bs: False)
    live.start()
    assert live.info["mode"] == "separate" and live.info["blocksize"] == 1920
    assert [s.kind for s in FakeStream.live] == ["in", "out"]
    assert all(s.kw["extra_settings"] == ("wasapi", True) and s.kw["samplerate"] == 48000 for s in FakeStream.live)
    assert FakeStream.live[0].kw["device"] == 9 and FakeStream.live[1].kw["device"] == 8


def test_smaller_blocks_then_duplex_are_tried_and_nothing_leaks(live):
    FakeStream.fail = staticmethod(lambda kind, bs: kind == "out" and bs in (1920, 960))
    live.start()
    assert (live.info["mode"], live.info["blocksize"]) == ("separate", 480) and len(FakeStream.live) == 2
    live._close()
    assert FakeStream.live == []                               # the half-opened attempts were closed again

    FakeStream.fail = staticmethod(lambda kind, bs: kind == "out")
    live.blocksize = 1920
    live.start()
    assert live.info["mode"] == "duplex" and FakeStream.live[0].kw["extra_settings"] == (("wasapi", True), ("wasapi", True))
    live._close()

    FakeStream.fail = staticmethod(lambda kind, bs: kind != "in")
    with pytest.raises(audio.AudioError, match="exclusive control"):
        live.start()
    assert FakeStream.live == []                               # the microphone is not left locked


def test_input_queue_drops_oldest_and_flags_it(cfg):
    a = audio.LiveAudio.__new__(audio.LiveAudio)
    a.n_mics = 2
    a._q = audio.queue.Queue(maxsize=3)
    a._discontinuity = False
    a._last_rx = 0.0
    for i in range(5):
        a._push(np.full((4, 2), i, dtype=np.float32))
    assert a.pop_discontinuity() and not a.pop_discontinuity()
    got = a.read(timeout=0.1)
    assert got.shape == (12, 2) and got[0, 0] == 2 and got[-1, 0] == 4     # the two oldest blocks were dropped


def test_file_backend_replays_a_recording(cfg, tmp_path):
    import soundfile as sf

    x = (0.01 * np.random.default_rng(0).standard_normal((5000, 2))).astype(np.float32)
    sf.write(str(tmp_path / "a.wav"), x, cfg.fs, subtype="PCM_24")
    f = audio.FileAudio(cfg, tmp_path / "a.wav", realtime=False, block=1920)
    f.start()
    parts = []
    while (b := f.read()) is not None:
        parts.append(b)
    f.stop()
    out = np.concatenate(parts)
    assert out.shape == x.shape and np.max(np.abs(out - x)) < 2e-7 and f.finished   # 24-bit round trip


def test_find_device_prefers_named_devices(monkeypatch):
    devices = [
        {"index": 3, "name": "Speakers (USB dongle)", "hostapi": "Windows WASAPI", "in": 0, "out": 2, "default_fs": 48000.0},
        {"index": 8, "name": "Speaker (Realtek(R) Audio)", "hostapi": "Windows WASAPI", "in": 0, "out": 2, "default_fs": 48000.0},
        {"index": 9, "name": "Microphone Array (Intel Smart Sound Technology for Digital Microphones)", "hostapi": "Windows WASAPI", "in": 2, "out": 0, "default_fs": 48000.0},
        {"index": 1, "name": "Microphone Array (Intel Smart Sound Technology for Digital Microphones)", "hostapi": "MME", "in": 2, "out": 0, "default_fs": 44100.0},
    ]
    monkeypatch.setattr(audio, "list_devices", lambda: devices)
    assert audio.find_device("output", prefer_hostapi="WASAPI")["index"] == 8        # the demo laptop's speaker by name
    assert audio.find_device("input", prefer_hostapi="WASAPI")["index"] == 9         # the WASAPI one, not MME
    assert audio.find_device("output", "usb", prefer_hostapi="WASAPI")["index"] == 3
    assert audio.find_device("input", 1)["index"] == 1
    with pytest.raises(audio.AudioError, match="Available"):
        audio.find_device("output", "nonexistent", prefer_hostapi="WASAPI")

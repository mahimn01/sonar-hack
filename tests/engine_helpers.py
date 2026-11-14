"""Shared material for the data-engine tests (weak labels, label QA, room adaptation).

Everything is made with the simulator. The slow pieces (three prompted recording sessions and a
small model trained on them) are built once per test run and reused by every test file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sonar import record
from sonar.config import SonarConfig
from sonar.sim import HAND_REST, EchoSimulator, WaypointPath, gesture_stroke_window, gesture_waypoints, hand_reflector, person_reflectors

_CACHE: dict[str, Path] = {}


def prompted_sessions(tmp_path_factory) -> Path:
    """Three simulated sessions made by the prompted recorder (people ann, bob, cy), as test_training.py uses."""
    if "sessions" not in _CACHE:
        root = tmp_path_factory.mktemp("engine_data")
        for person, seed in (("ann", 31), ("bob", 32), ("cy", 33)):
            assert record.main(["--person", person, "--backend", "sim", "--fast", "--script", "gestures", "--reps", "4",
                                "--out", str(root), "--seed", str(seed)]) == 0
        _CACHE["sessions"] = root
    return _CACHE["sessions"]


def small_model(tmp_path_factory) -> Path:
    """A model trained for a few epochs on those sessions: folder with gesture.onnx, gesture.json, gesture.pt."""
    if "model" not in _CACHE:
        from sonar.train import main as train_main

        out = tmp_path_factory.mktemp("engine_model")
        assert train_main(["--data", str(prompted_sessions(tmp_path_factory)), "--out", str(out), "--epochs", "14", "--device", "cpu",
                           "--seed", "0", "--val-fraction", "0"]) == 0
        _CACHE["model"] = out
    return _CACHE["model"]


def scripted_audio(cfg: SonarConfig, script: list[tuple[float, str, float]], seconds: float, seed: int = 0, extra_paths=(),
                   hand_db: float = -20.0, **sim_kwargs: Any) -> tuple[np.ndarray, WaypointPath]:
    """Microphone audio of a seated person whose hand performs `script` = [(start s, gesture, speed), ...]."""
    times, pts = [0.0], [HAND_REST]
    for t0, label, speed in script:
        for t, p in gesture_waypoints(label, t0, speed):
            if t > times[-1]:
                times.append(t)
                pts.append(p)
    path = WaypointPath(times, pts)
    sim_kwargs.setdefault("cn0_db", 70.0)
    sim = EchoSimulator(cfg, seed=seed, **sim_kwargs)
    reflectors = hand_reflector(path, echo_db=hand_db) + person_reflectors(0.65, seed=seed)
    for p in extra_paths:
        reflectors += hand_reflector(p, echo_db=hand_db)
    return sim.render(int(seconds * cfg.fs), reflectors), path


def stroke_times(script: list[tuple[float, str, float]]) -> list[tuple[str, float, float]]:
    """(gesture, start s, end s) of the fast part of each scripted gesture."""
    return [(label, *gesture_stroke_window(label, t0, speed)) for t0, label, speed in script]


def write_session(folder: Path, cfg: SonarConfig, audio: np.ndarray, events: list[dict[str, Any]], person: str = "x",
                  script: str = "gestures", **extra: Any) -> Path:
    """A session folder in the recorder's format (audio.wav + session.json)."""
    import soundfile as sf

    folder.mkdir(parents=True, exist_ok=True)
    sf.write(str(folder / "audio.wav"), audio, cfg.fs, subtype="PCM_24")
    meta = {"format": 1, "session_id": folder.name, "person": person, "script": script, "notes": "", "profile": "default",
            "synthetic": True, "audio": "audio.wav", "fs": cfg.fs, "n_mics": audio.shape[1], "n_samples": len(audio),
            "config": cfg.to_dict(), "backend": {"mode": "sim"}, "dropouts": [], "peak": float(np.max(np.abs(audio))),
            "events": [{"label": "calibration", "kind": "calibration", "start": 0, "end": int(2.5 * cfg.fs)}, *events], **extra}
    (folder / "session.json").write_text(json.dumps(meta, indent=1))
    return folder


def span(cfg: SonarConfig, label: str, t0: float, t1: float, kind: str = "scene", **extra: Any) -> dict[str, Any]:
    return {"label": label, "kind": kind, "start": int(t0 * cfg.fs), "end": int(t1 * cfg.fs), **extra}


def camera_sessions(tmp_path_factory) -> Path:
    """Two simulated camera-labelled sessions (camrecord + the fake camera): one free-form, one gesture-free."""
    if "camera" not in _CACHE:
        from sonar import camrecord

        root = tmp_path_factory.mktemp("engine_cam")
        base = ["--backend", "sim", "--fast", "--no-window", "--consent", "yes", "--out", str(root)]
        assert camrecord.main([*base, "--person", "v21", "--seed", "41", "--reps", "2", "--rounds", "1"]) == 0
        assert camrecord.main([*base, "--person", "v22", "--seed", "42", "--script", "negatives", "--negatives-scale", "0.6"]) == 0
        _CACHE["camera"] = root
    return _CACHE["camera"]


def room_session(folder: Path, cfg: SonarConfig, person: str, seed: int, reps: int = 5, gestures=("swipe_left", "swipe_right", "push", "pull"),
                 hand_db: float = -20.0, offset=(0.0, 0.0, 0.0), speed=(0.8, 1.3), mic_gap: float = 0.08, spk_gap: float = 0.24,
                 fidgets: int = 6, **sim_kwargs: Any) -> Path:
    """A prompted gesture session recorded "somewhere else": another laptop shape, echo strength, noise level, hand position.

    offset  moves every gesture (metres: right, away from the screen, up); mic_gap / spk_gap are the distances between
    the two microphones / speakers of the pretend laptop. Events are prompt windows, as sonar.record writes them.
    After the gestures come `fidgets` unhurried reaches inside a "distractor" part: real non-gesture movement.
    """
    from sonar.sim import Geometry

    rng = np.random.default_rng(seed)
    trials = [g for g in gestures for _ in range(reps)]
    order = rng.permutation(len(trials))
    times, pts = [0.0], [HAND_REST]
    events, t0 = [], 4.0
    for n, k in enumerate(order, 1):
        label = trials[int(k)]
        for t, p in gesture_waypoints(label, t0, float(rng.uniform(*speed))):
            if t > times[-1]:
                times.append(t)
                pts.append(tuple(np.asarray(p) + (np.asarray(offset) if p != HAND_REST else 0.0) + (rng.normal(0, 0.012, 3) if p != HAND_REST else 0.0)))
        events.append({"label": label, "kind": "gesture", "start": int((t0 - 0.35) * cfg.fs), "end": int((t0 + 2.15) * cfg.fs), "rep": n})
        t0 += 4.2
    t_fid = t0
    targets = [(0.30, 0.34, 0.06), (-0.12, 0.30, 0.12), (0.06, 0.24, 0.05), (0.22, 0.44, 0.20)]
    for n in range(fidgets):
        target = tuple(np.asarray(targets[n % 4]) + rng.normal(0, 0.02, 3))
        for t, p in ((t0, HAND_REST), (t0 + 1.0, target), (t0 + 1.5, target), (t0 + 2.6, HAND_REST)):
            if t > times[-1]:
                times.append(t)
                pts.append(p)
        t0 += 3.4
    if fidgets:
        events.append({"label": "distractor", "kind": "scene", "start": int((t_fid + 0.3) * cfg.fs), "end": int(t0 * cfg.fs)})
    path = WaypointPath(times, pts)
    geometry = Geometry(np.array([[-spk_gap / 2, 0.03, 0.01], [spk_gap / 2, 0.03, 0.01]]),
                        np.array([[-mic_gap / 2, -0.07, 0.21], [mic_gap / 2, -0.07, 0.21]]))
    sim_kwargs.setdefault("cn0_db", 70.0)
    sim = EchoSimulator(cfg, seed=seed, geometry=geometry, **sim_kwargs)
    audio = sim.render(int((t0 + 1.0) * cfg.fs), hand_reflector(path, echo_db=hand_db) + person_reflectors(0.65 + offset[1], seed=seed))
    return write_session(folder, cfg, audio, events, person=person, script="gestures")

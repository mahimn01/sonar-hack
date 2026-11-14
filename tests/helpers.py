"""Shared helpers for the tests: scripted simulated scenes and one-call pipeline runs.

Everything here uses the physics simulator (sonar/sim.py); no audio hardware.
"""

from __future__ import annotations

import numpy as np

from sonar.config import SonarConfig
from sonar.gestures import Segment, heuristic_classify
from sonar.pipeline import Pipeline
from sonar.sim import HAND_REST, EchoSimulator, WaypointPath, gesture_waypoints, hand_reflector, person_reflectors

HAND_DB = -22.0     # hand echo relative to the direct path at 30 cm (SoundWave reports about -20 dB on laptops)


def gesture_scene(label: str, t0: float = 6.0, speed: float = 1.0, seed: int = 0):
    """A seated person plus a hand that performs `label` starting at t0 seconds."""
    wp = gesture_waypoints(label, t0, speed)
    path = WaypointPath([0.0] + [w[0] for w in wp], [HAND_REST] + [w[1] for w in wp])
    return hand_reflector(path, echo_db=HAND_DB) + person_reflectors(0.65, seed=seed)


def run_pipeline(cfg: SonarConfig, sim: EchoSimulator, reflectors, seconds: float, block: int = 4800):
    """Render `seconds` of audio block by block and return (pipeline, all frames)."""
    pipe = Pipeline(cfg)
    frames = []
    for start in range(0, int(seconds * cfg.fs), block):
        frames += pipe.process(sim.render(block, reflectors, start_sample=start))
    return pipe, frames


def classify_bursts(cfg: SonarConfig, frames) -> list[tuple[Segment, str, float, float, float]]:
    """(burst, physics label, confidence, speaker cue Hz, mic cue Hz) for every closed burst."""
    out = []
    for fr in frames:
        if fr.segment is not None and fr.window is not None:
            r = heuristic_classify(fr.window, cfg)
            out.append((fr.segment, r.label, r.confidence, r.lateral_hz, r.lateral_mic_hz))
    return out


def strongest(bursts):
    return max(bursts, key=lambda b: b[0].peak_db)


def doppler_peak_hz(cfg: SonarConfig, frames, t0: float, t1: float) -> np.ndarray:
    """Per feature channel: Doppler frequency (Hz) of the strongest bin, averaged over frames t0..t1 s."""
    cols = np.stack([fr.feat[:cfg.n_power_channels] for fr in frames if fr.feat is not None and t0 <= fr.t <= t1])
    return cfg.doppler_hz[np.argmax(cols.mean(axis=0), axis=1)]

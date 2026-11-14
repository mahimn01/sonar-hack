"""A moving reflector must give the right Doppler sign and speed, and static things must give none."""

import numpy as np
import pytest

from sonar.config import SPEED_OF_SOUND
from sonar.sim import EchoSimulator, Geometry, Reflector, linear_path

from .helpers import doppler_peak_hz, run_pipeline


def expected_doppler(cfg, geo, pos, vel):
    """Physics: f_d = -(f/c) * d/dt(|p - speaker| + |p - mic|), per (tone, mic) channel."""
    out = []
    for f, ch in zip(cfg.tone_hz, cfg.tone_channels):
        for mic in geo.mics:
            u_s = (pos - geo.speakers[ch]) / np.linalg.norm(pos - geo.speakers[ch])
            u_m = (pos - mic) / np.linalg.norm(pos - mic)
            out.append(-f / SPEED_OF_SOUND * (vel @ u_s + vel @ u_m))
    return np.array(out)


@pytest.mark.parametrize("speed", [0.4, -0.4, 0.8])
def test_doppler_sign_and_speed(cfg, speed):
    """Reflector moving straight at the laptop (speed > 0) or away from it (speed < 0)."""
    geo = Geometry.laptop()
    t_mid = 3.0
    y_mid = 0.40
    y0 = y_mid + speed * t_mid                                    # so that it passes y_mid at t_mid
    vel = np.array([0.0, -speed, 0.0])
    mover = Reflector(linear_path((0.0, y0, 0.17), tuple(vel)), echo_db=-20.0, ref_m=0.3)
    _, frames = run_pipeline(cfg, EchoSimulator(cfg, seed=1), [mover], seconds=t_mid + 0.3)
    measured = doppler_peak_hz(cfg, frames, t_mid - 0.15, t_mid + 0.15)
    expect = expected_doppler(cfg, geo, np.array([0.0, y_mid, 0.17]), vel)
    assert np.all(np.sign(measured) == np.sign(speed)), f"wrong Doppler sign: {measured}"
    np.testing.assert_allclose(measured, expect, atol=12.0)       # one 7.8 Hz bin plus the 23 Hz window width
    # the simple rule of thumb also holds: about 2 * v * f / c
    assert abs(np.mean(np.abs(measured)) - 2 * abs(speed) * 18750 / SPEED_OF_SOUND) < 15.0


def test_static_scene_has_no_doppler(cfg):
    """Direct path + a wall: the carrier is ~40 dB above the noise yet no Doppler bin lights up."""
    wall = Reflector(lambda t: np.tile([0.0, 0.8, 0.2], (len(t), 1)), echo_db=-20.0, ref_m=0.7, moving=False)
    pipe, frames = run_pipeline(cfg, EchoSimulator(cfg, seed=2), [wall], seconds=6.0)
    act = np.array([fr.activity_db for fr in frames if fr.feat is not None])
    assert pipe.snr_db() > 30.0
    assert np.abs(act[50:]).max() < 2.2                           # noise only (sd 0.5 dB): never near the 3 dB gesture bar
    assert not any(fr.segment for fr in frames)


def test_loud_hall_noise_does_not_leak_in(cfg):
    """Speech-band noise 40 dB louder than our own signal must not raise the Doppler noise floor."""
    quiet_pipe, _ = run_pipeline(cfg, EchoSimulator(cfg, seed=3, ambient_dbfs=None), [], seconds=4.0)
    loud_pipe, _ = run_pipeline(cfg, EchoSimulator(cfg, seed=3, ambient_dbfs=-30.0), [], seconds=4.0)
    assert abs(np.median(loud_pipe.normalizer.floor_db) - np.median(quiet_pipe.normalizer.floor_db)) < 1.0

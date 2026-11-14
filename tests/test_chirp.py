"""Chirp receiver: alignment, range, clock drift, dropped audio."""

import numpy as np
import pytest

from sonar.config import SPEED_OF_SOUND
from sonar.frontend import Frontend, run_frontend
from sonar.sim import EchoSimulator, Geometry, Reflector, static_path


def mean_profile(frames):
    return np.mean(np.abs(np.stack([f.cir for f in frames if f.cir is not None])), axis=0)


@pytest.mark.parametrize("block", [333, 960, 1920, 4800])
@pytest.mark.parametrize("latency", [0.0, 1234.56, 40000.3])
def test_direct_path_lands_on_its_tap_whatever_the_latency_and_block_size(cfg, block, latency):
    rx = EchoSimulator(cfg, latency_samples=latency).render(3 * cfg.fs, [])
    frames = run_frontend(cfg, rx, block=block)
    prof = mean_profile(frames)
    assert [int(np.argmax(p)) for p in prof] == [cfg.direct_tap] * cfg.n_mics
    assert frames[-1].coherence > 0.9


def test_static_reflector_appears_at_the_right_range(cfg):
    geo = Geometry.laptop()
    pos = np.array([0.0, 0.9, 0.2])
    wall = Reflector(static_path(tuple(pos)), echo_db=-12.0, ref_m=0.9, moving=False)
    prof = mean_profile(run_frontend(cfg, EchoSimulator(cfg, cn0_db=75.0).render(4 * cfg.fs, [wall])))[0]
    extra = np.linalg.norm(pos - geo.speakers[cfg.chirp_channel]) + np.linalg.norm(pos - geo.mics[0]) \
        - np.linalg.norm(geo.speakers[cfg.chirp_channel] - geo.mics[0])
    expect_tap = cfg.direct_tap + extra / cfg.tap_path_m
    far = prof.copy()
    far[:cfg.direct_tap + 8] = 0                                   # look beyond the direct path's main lobe
    tap = int(np.argmax(far))
    assert abs(tap - expect_tap) <= 1.5
    assert abs(cfg.range_m[tap] - 0.9) < 0.08                      # reported distance within 8 cm
    # range resolution claimed in the config: c / 2B
    assert abs(cfg.range_resolution_m - SPEED_OF_SOUND / (2 * 2050.0)) < 1e-9


@pytest.mark.parametrize("ppm", [25.0, -40.0])
def test_clock_drift_is_measured_and_removed(cfg, ppm):
    """Speaker and mic on different clocks: the echo profile must still stand perfectly still."""
    rx = EchoSimulator(cfg, clock_ppm=ppm).render(12 * cfg.fs, [])
    frames = [f for f in run_frontend(cfg, rx) if f.cir is not None]
    assert abs(frames[-1].drift_ppm + ppm) < 1.0                   # reported as mic-minus-speaker, hence the sign
    late = np.stack([f.cir for f in frames[-300:]])[:, 0, cfg.direct_tap]
    assert np.std(np.angle(late * np.conj(late.mean()))) < 0.05    # direct-path phase steady to 3 degrees
    assert int(np.argmax(np.abs(late.mean()))) == 0 and frames[-1].coherence > 0.9
    assert int(np.argmax(mean_profile(frames[-300:])[0])) == cfg.direct_tap


def test_receiver_relocks_after_dropped_audio(cfg):
    """PortAudio does not report dropped WASAPI buffers: the receiver has to notice by itself."""
    rx = EchoSimulator(cfg).render(8 * cfg.fs, [])
    cut = np.concatenate([rx[:3 * cfg.fs], rx[3 * cfg.fs + 437:]])  # 437 samples vanish at t = 3 s
    fe = Frontend(cfg)
    frames = []
    for i in range(0, len(cut), 1920):
        frames += fe.process(cut[i:i + 1920])
    assert fe.n_resync == 1                                        # exactly one re-lock
    assert int(np.argmax(mean_profile(frames[-100:])[0])) == cfg.direct_tap
    assert frames[-1].coherence > 0.9

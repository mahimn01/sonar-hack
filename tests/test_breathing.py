"""Breathing: a simulated chest moving a few millimetres must give the right rate.

The simulated conditions are deliberately harsh: the direct speaker->mic path is
40 dB stronger than the chest echo (-50 dB in one case), sensor noise is set to
the level measured on the demo laptop at 36 % volume, there is loud speech-band
hall noise, the non-breathing body parts add a static echo in the same range
bins, and the body sways slowly.
"""

import numpy as np
import pytest

from sonar.breathing import BreathingEstimator, circle_fit, estimate_rate
from sonar.frontend import Frontend
from sonar.sim import EchoSimulator, breathing_displacement, person_reflectors, sway_displacement


def run_breathing(cfg, seconds=40, **kw):
    sim_kw = {k: kw.pop(k) for k in list(kw) if k in ("cn0_db", "clock_ppm", "ambient_dbfs", "latency_samples")}
    refl = person_reflectors(**kw) if kw.pop("person", True) else []
    sim = EchoSimulator(cfg, seed=kw.get("seed", 0), **sim_kw)
    fe, be = Frontend(cfg), BreathingEstimator(cfg)
    results = []
    for i in range(int(seconds)):
        for fr in fe.process(sim.render(cfg.fs, refl, start_sample=i * cfg.fs)):
            if fr.cir is not None:
                r = be.update(fr.cir)
                if r is not None:
                    results.append((fr.t, r))
    return results


@pytest.mark.parametrize(
    "bpm, dist, amp_mm, extra",
    [
        (15.0, 0.7, 2.5, {}),                                   # the case in the brief: 15/min, 5 mm peak-to-peak, 0.7 m
        (12.0, 0.5, 2.5, {}),
        (18.0, 0.6, 1.5, {}),                                   # shallow breathing (3 mm peak-to-peak)
        (20.0, 1.0, 2.5, {"cn0_db": 70.0}),                     # 1 m needs more speaker volume (+10 dB)
        (15.0, 0.7, 2.5, {"clock_ppm": 20.0}),                  # speaker and mic on different clocks
        (15.0, 0.7, 2.5, {"chest_db": -50.0, "body_db": -48.0, "cn0_db": 70.0}),   # direct path 50 dB above the body echoes
        (10.0, 0.6, 5.0, {}),                                   # slow, deep breathing (more than a full phase turn)
    ],
)
def test_breathing_rate_within_one_bpm(cfg, bpm, dist, amp_mm, extra):
    results = run_breathing(cfg, distance_m=dist, bpm=bpm, breath_amp_m=amp_mm / 1000, sway_rms_m=0.0003, **extra)
    t, last = results[-1]
    assert last.state == "tracking", f"not tracking: {last.state}, confidence {last.confidence:.2f}, snr {last.snr_db:.1f} dB"
    assert abs(last.bpm - bpm) <= 1.0, f"estimated {last.bpm:.2f}, truth {bpm}"
    assert abs(last.range_m - dist) < 0.2                       # and it found the chest at the right distance
    # every confident report after 25 s is also within 1 bpm (no flip-flopping)
    late = [r.bpm for tt, r in results if tt > 25 and r.bpm is not None]
    assert late and max(abs(b - bpm) for b in late) <= 1.0
    first = next(tt for tt, r in results if r.bpm is not None)
    assert first < 30.0                                         # first confident rate within 30 s of sitting still


def test_waveform_follows_the_chest_with_the_right_sign(cfg):
    """Positive waveform = chest towards the laptop (inhale)."""
    results = run_breathing(cfg, distance_m=0.6, bpm=14.0, breath_amp_m=0.003, sway_rms_m=0.0, cn0_db=70.0)
    t_end, last = results[-1]
    n = len(last.waveform)
    t = t_end - (n - 1 - np.arange(n)) / last.fs
    truth = breathing_displacement(t, 14.0, 0.003)
    corr = np.corrcoef(last.waveform[20:-20], truth[20:-20])[0, 1]
    assert corr > 0.8
    assert 4.0 < last.amplitude_mm < 8.5                        # true peak-to-peak is about 6 mm


def test_empty_room_reports_no_breathing(cfg):
    results = run_breathing(cfg, seconds=30, person=False)
    assert all(r.bpm is None for _, r in results)
    assert results[-1][1].state in ("no_signal", "acquiring")


def test_circle_fit_recovers_centre_from_a_short_noisy_arc():
    rng = np.random.default_rng(0)
    phase = 0.9 * np.sin(np.linspace(0, 12, 300))               # about 100 degrees of arc, like 2.4 mm of breathing
    z = (3 - 2j) + 1.0 * np.exp(1j * (0.4 + phase)) + 0.03 * (rng.standard_normal(300) + 1j * rng.standard_normal(300))
    centre, radius = circle_fit(z[:, None])
    assert abs(centre[0] - (3 - 2j)) < 0.12 and abs(radius[0] - 1.0) < 0.12


@pytest.mark.parametrize("seed", range(6))
def test_rate_estimator_tolerates_sway_and_prefers_the_fundamental(seed):
    """Slow sway with an rms of 1.5x the breathing amplitude must not pull the rate off.

    (Measured limit of the method: fine up to about 1.5x, unreliable from 2x. A person
    standing free sways more than that, which is why the demo is done seated.)
    """
    fs, t = 10.0, np.arange(300) / 10.0
    breathing = np.sin(2 * np.pi * 0.25 * t) + 0.5 * np.sin(2 * np.pi * 0.5 * t + 1.0)       # strong 2nd harmonic
    f0, quality = estimate_rate(breathing + 1.5 * sway_displacement(t, 1.0, seed=seed), fs, (0.1, 0.6))
    assert abs(60 * f0 - 15.0) < 0.5


def test_rate_estimator_rejects_a_pure_ramp():
    fs, t = 10.0, np.arange(300) / 10.0
    f0, _ = estimate_rate(0.2 * np.sin(2 * np.pi * 0.25 * t) + np.linspace(0, 5, 300), fs, (0.1, 0.6))
    assert abs(60 * f0 - 15.0) < 0.5                                                         # weak breathing on a big drift
    f0, quality = estimate_rate(np.linspace(0, 5, 300) ** 2, fs, (0.1, 0.6))                 # drift only
    assert f0 is None or quality < 0.6

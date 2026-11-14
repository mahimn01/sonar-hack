"""Presence: empty -> entering -> present -> empty on a scripted visitor, and the motionless timer."""

import numpy as np

from sonar.sim import EchoSimulator, Reflector, breathing_displacement, min_jerk, sway_displacement

from .helpers import run_pipeline

T_IN, T_SIT, T_OUT = 8.0, 11.0, 41.0


def visitor(hold_breath_from: float | None = None, leaves: bool = True):
    """Walks in at 8-11 s, sits at 0.65 m breathing 16/min, walks out at 41-44 s (optional)."""
    def body_y(t):
        y = np.full_like(t, 3.0)
        y = np.where((t >= T_IN) & (t < T_SIT), 3.0 - 2.35 * min_jerk((t - T_IN) / 3.0), y)
        y = np.where(t >= T_SIT, 0.65, y)
        if leaves:
            y = np.where(t >= T_OUT, 0.65 + 2.35 * min_jerk((t - T_OUT) / 3.0), y)
        return y

    def part(dx, dz, breathes):
        def f(t):
            still = (t >= T_SIT) & ((t < T_OUT) | (not leaves))
            d = np.where(still, sway_displacement(t, 0.0002, 1), 0.0)
            if breathes:
                b = breathing_displacement(t, 16.0, 0.0025)
                if hold_breath_from is not None:
                    b = np.where(t >= hold_breath_from, breathing_displacement(np.full_like(t, hold_breath_from), 16.0, 0.0025), b)
                    d = np.where(t >= hold_breath_from, 0.0, d)        # perfectly motionless from then on
                d = d + np.where(still, b, 0.0)
            walking = (t >= T_IN) & (t < T_SIT) | (leaves & (t >= T_OUT) & (t < T_OUT + 3))
            bob = np.where(walking, 0.02 * np.sin(2 * np.pi * 1.8 * t), 0.0)
            return np.stack([np.full_like(t, dx), body_y(t) - d + bob, np.full_like(t, dz)], axis=1)
        return f

    return [Reflector(part(0.0, 0.22, True), -24.0, 0.7), Reflector(part(0.16, 0.38, False), -22.0, 0.7),
            Reflector(part(0.0, 0.52, False), -26.0, 0.7)]


def transitions(frames):
    return [(round(fr.t, 1), fr.presence.state) for fr in frames if fr.presence is not None and fr.presence.changed]


def test_walk_in_sit_breathe_leave(cfg):
    pipe, frames = run_pipeline(cfg, EchoSimulator(cfg, seed=3, cn0_db=72.0), visitor(), seconds=56.0)
    seq = transitions(frames)
    assert [s for _, s in seq] == ["entering", "present", "empty"], seq
    t_enter, t_present, t_empty = (t for t, _ in seq)
    assert T_IN < t_enter < T_SIT + 0.5                 # noticed while walking in
    assert T_SIT < t_present < T_SIT + 3.0              # "present" within 3 s of sitting down
    assert T_OUT + 2.0 < t_empty < T_OUT + 3.0 + 7.0    # "empty" within 7 s of being gone
    seated = [fr.presence for fr in frames if fr.presence is not None and 20 < fr.t < 40]
    assert not any(p.moving for p in seated)            # breathing must never count as gross movement
    assert abs(np.median([p.range_m for p in seated]) - 0.7) < 0.15
    # ... and the breathing rate is tracked while seated
    rates = [fr.breathing.bpm for fr in frames if fr.breathing is not None and 28 < fr.t < 40 and fr.breathing.bpm]
    assert rates and abs(np.median(rates) - 16.0) < 1.0


def test_motionless_timer_runs_when_breathing_stops(cfg):
    """Someone who sits there and stops breathing stays 'present' while `motionless_s` counts up
    (the dashboard turns that into the no-breathing alert); 'empty' is only for people who walked away."""
    hold = 30.0
    pipe, frames = run_pipeline(cfg, EchoSimulator(cfg, seed=4, cn0_db=72.0), visitor(hold_breath_from=hold, leaves=False), seconds=52.0)
    at = lambda t: next(fr.presence for fr in frames if fr.presence is not None and fr.t >= t)   # noqa: E731
    assert at(28.0).state == "present" and at(28.0).motionless_s < 2.0
    late = at(50.0)
    assert late.state == "present"
    assert 10.0 < late.motionless_s < 21.0
    breath = [fr.breathing for fr in frames if fr.breathing is not None and fr.t > 46]
    assert all(b.bpm is None for b in breath)           # no rate is invented for a chest that does not move

"""Presence and breathing in a busy room: the regression scene for the first real-hardware run.

On the demo laptop in a hackathon hall the first version said "moving" about 50 times a
minute for a person sitting still, restarted the breathing measurement every 1-3 s (so it
never produced a rate) and reported a range that jumped between 0.2 and 1.8 m. The
simulated hall here reproduces that failure on the old code and must stay fixed:

    seated breather at 0.7 m with fidgeting hands, a turning head and small posture shifts
  + two passers-by who keep crossing 1.5-3 m behind
  + a neighbour sitting 2 m away
  + clicks and hisses in the ultrasonic band (keys, wrappers, claps)
  + sensor noise at the level measured on the laptop (signal headroom ~55 dB)
"""

import numpy as np
import pytest

from sonar.breathing import BreathingEstimator
from sonar.config import get_profile
from sonar.dsp import guard_bins
from sonar.frontend import Frontend
from sonar.pipeline import Pipeline
from sonar.replay import run_blocks, score, sim_blocks
from sonar.sim import (
    EchoSimulator,
    HallNoise,
    Reflector,
    WaypointPath,
    breathing_displacement,
    hall_reflectors,
    person_reflectors,
    sway_displacement,
    walker_reflectors,
)

CN0 = 66.0          # sensor noise at the level measured on the demo laptop


@pytest.fixture(scope="module")
def clean():
    """The profile the demo laptop runs (one pilot tone, chirp on the other speaker)."""
    return get_profile("clean")


@pytest.fixture(scope="module")
def hall(clean):
    """60 s of the busy hall with a person seated at 0.7 m breathing 15 times a minute."""
    trace, _ = run_blocks(clean, sim_blocks(clean, "hall", 60.0, seed=0, cn0_db=CN0))
    return trace


# ---------------------------------------------------------------------------
# The reported failure, end to end
# ---------------------------------------------------------------------------
def test_seated_person_in_a_busy_hall_is_not_called_moving(hall):
    s = score(hall, 10.0, 60.0)
    assert s.present_share == 1.0                        # "present" the whole time, no flapping
    assert s.state_changes == 0
    assert s.moving_onsets <= 1 and s.moving_share < 0.02   # was: 50 onsets a minute, on 20 % of the time
    conf = hall.conf[hall.between(12.0, 60.0)]
    assert conf.min() > 0.5 and np.ptp(conf) < 0.4       # was: swinging between 0.03 and 0.87
    assert abs(s.range_median - 0.72) < 0.1 and s.range_spread < 0.1   # was: jumping between 0.2 and 1.8 m


def test_breathing_is_tracked_through_fidgets_walkers_and_noise(hall):
    s = score(hall, 10.0, 60.0, paced_bpm=15.0)
    assert s.restarts == 0                               # was: a restart every 1-3 s, never a rate
    assert s.first_bpm_s is not None and s.first_bpm_s < 12.0      # first rate within ~17 s of calibration ending
    late = score(hall, 25.0, 60.0, paced_bpm=15.0)
    assert late.tracking_share > 0.95                    # ... and then it stays
    assert late.bpm_err_max <= 1.0                       # every reported rate within 1 breath/min
    assert abs(late.chest_range_median - 0.72) < 0.1


def test_empty_hall_stays_empty(clean):
    """Nobody at the laptop: walkers 1.5-3 m away, a neighbour and noise bursts are not a person,
    and no breathing rate is invented out of noise."""
    trace, _ = run_blocks(clean, sim_blocks(clean, "hall_empty", 50.0, seed=1, cn0_db=CN0))
    s = score(trace, 6.0, 50.0)
    assert s.empty_share == 1.0 and s.moving_onsets == 0
    assert all(b.bpm is None for _, b in trace.breathing)


def test_one_microphone_is_enough(clean):
    """A MacBook gives one microphone channel: the same hall with half the looks."""
    cfg = get_profile("default").with_(n_mics=1)
    trace, _ = run_blocks(cfg, sim_blocks(cfg, "hall", 45.0, seed=2, cn0_db=CN0))
    s = score(trace, 22.0, 45.0, paced_bpm=15.0)
    assert s.present_share == 1.0 and s.moving_onsets <= 1
    assert s.tracking_share > 0.9 and s.bpm_err_max <= 1.0


# ---------------------------------------------------------------------------
# Real movement still counts, and breathing comes back quickly after it
# ---------------------------------------------------------------------------
def mover(seed: int = 0):
    """A seated breather (15/min at 0.70 m) who leans back 8 cm at 22 s, wriggles for 4 s from 36 s,
    and holds their breath from 58 s to 70 s."""
    lean = WaypointPath([0.0, 22.0, 22.7, 400.0], [(0, 0.70, 0), (0, 0.70, 0), (0, 0.78, 0), (0, 0.78, 0)])

    def torso_y(t):
        wriggle = np.where((t >= 36.0) & (t < 40.0), 0.03 * np.sin(2 * np.pi * 1.3 * t) * np.sin(np.pi * (t - 36.0) / 4.0) ** 2, 0.0)
        return lean(t)[:, 1] + wriggle - sway_displacement(t, 0.0004, seed)

    def breath(t):
        frozen = breathing_displacement(np.full_like(t, 58.0), 15.0, 0.0025)
        return np.where((t >= 58.0) & (t < 70.0), frozen, breathing_displacement(t, 15.0, 0.0025))

    def part(dx, dy, dz, share):
        return lambda t: np.stack([np.full_like(t, dx), torso_y(t) + dy - share * breath(t), np.full_like(t, dz)], axis=1)

    return [Reflector(part(0.0, 0.0, 0.22, 1.0), -30.0, 0.7), Reflector(part(0.0, -0.02, 0.05, 1.4), -33.0, 0.7),
            Reflector(part(0.16, 0.03, 0.38, 0.15), -28.0, 0.7), Reflector(part(0.0, -0.02, 0.52, 0.0), -32.0, 0.7)]


def test_real_movement_is_reported_and_breathing_recovers(clean):
    sim = EchoSimulator(clean, seed=2, cn0_db=CN0)            # (seed 2: no noise burst lands on the 0.7 s lean)
    refl = mover() + hall_reflectors(seated=False, seed=2, t_end=90.0)
    noise = HallNoise(clean, sim._noise_sigma, seed=2, t_end=90.0)
    blocks = (noise.add(sim.render(9600, refl, start_sample=s), s) for s in range(0, 82 * clean.fs, 9600))
    trace, _ = run_blocks(clean, blocks)
    states = {round(t): b.state for t, b in trace.breathing}
    rates = {round(t): b.bpm for t, b in trace.breathing}

    # a quick lean (0.7 s): `moving` shows it; the breathing read-out rides through and stays right
    assert np.any(trace.moving[trace.between(22.0, 23.5)])
    assert score(trace, 12.0, 21.0).moving_onsets == 0 and score(trace, 26.0, 35.0).moving_onsets == 0
    assert all(rates[t] is not None and abs(rates[t] - 15.0) <= 1.0 for t in range(26, 36))

    # 4 s of wriggling: breathing says "moving" (no rate), and has a rate again within 12 s of the end
    assert "moving" in {states[t] for t in range(37, 42)}
    assert all(rates[t] is None for t in range(38, 41))
    back = min(t for t in range(40, 58) if rates[t] is not None)
    assert back <= 52 and abs(rates[back] - 15.0) <= 1.5
    assert all(rates[t] is not None and abs(rates[t] - 15.0) <= 1.0 for t in range(54, 58))

    # breath held at 58 s: within 6 s no rate is shown any more, the person stays "present" and
    # `stopped_s` counts up (the alarm signal: "the rhythm we followed has stopped and the chest
    # is still"); breathing again at 70 s -> a rate within 4 s and the counter is back at zero
    stopped = {round(t): b.stopped_s for t, b in trace.breathing}
    assert all(rates[t] is None for t in range(64, 70))
    assert np.all(trace.state[trace.between(58.0, 70.0)] == "present")
    assert stopped[60] == 0.0 and stopped[69] >= 5.0
    assert any(rates[t] is not None and abs(rates[t] - 15.0) <= 1.5 for t in range(71, 75))
    assert stopped[74] == 0.0 and trace.motionless_s[trace.between(73.0, 80.0)].max() < 1.0


# ---------------------------------------------------------------------------
# The pieces, one at a time
# ---------------------------------------------------------------------------
def test_plain_noise_is_never_a_chest_however_loud():
    """The combiner judges the noise level from the data itself: noise 3x or 10x louder than the
    sensor-noise estimate must still read as "no moving echo" (it used to read 5-7 dB)."""
    cfg = get_profile("default")
    be = BreathingEstimator(cfg)
    rng = np.random.default_rng(0)
    wh = be._whitener(5)                                             # [5, 2]: 2 independent looks <- 5 taps
    back = np.linalg.pinv(wh)                                        # looks -> taps with the right tap-to-tap correlation
    for loud in (1.0, 3.0, 10.0):
        worst = 0.0
        for _ in range(40):
            looks = (rng.standard_normal((300, 4)) + 1j * rng.standard_normal((300, 4))) / np.sqrt(2)
            m = np.concatenate([looks[:, :2] @ back, looks[:, 2:] @ back], axis=1) * np.sqrt(loud)
            worst = max(worst, be._combine(m, n_mics=2)[1])
        assert worst < 0.6 * be.min_snr, f"noise {loud}x the estimate reads as an echo of snr {worst:.2f}"
    # ... while a real common movement in the same noise stands out, at its true size: an arc of
    # radius 3 swinging +-1 rad has a moving power of 9 * (1 - J0(1)^2) = 3.7 times the noise
    looks = (rng.standard_normal((300, 4)) + 1j * rng.standard_normal((300, 4))) / np.sqrt(2)
    looks[:, 0] += 3.0 * np.exp(1j * np.sin(2 * np.pi * 0.25 * np.arange(300) / 10.0))
    looks[:, 0] -= looks[:, 0].mean()
    m = np.concatenate([looks[:, :2] @ back, looks[:, 2:] @ back], axis=1)
    g, snr, weights = be._combine(m, n_mics=2)
    assert 3.0 < snr < 4.5 and abs(np.var(g) - (snr + 1.0)) < 0.6
    assert np.allclose(m @ weights, g)                               # the weights reproduce the combined signal


def test_brief_disturbances_are_skipped_not_punished():
    """A 0.4 s disturbance at the chest's distance every 6 s (a hand twitch, a burst of noise): the
    estimator keeps its history and still reports the rate. The first version started over each time."""
    cfg = get_profile("default")
    sim = EchoSimulator(cfg, seed=1, cn0_db=CN0)
    refl = person_reflectors(0.65, bpm=13.0, chest_db=-32.0, body_db=-30.0, seed=1)
    for legacy in (False, True):
        fe, be = Frontend(cfg), BreathingEstimator(cfg)
        results = []
        for i in range(40):
            for fr in fe.process(sim.render(cfg.fs, refl, start_sample=i * cfg.fs)):
                if fr.cir is None:
                    continue
                blip = (fr.t % 6.0) < 0.4
                if legacy:                                               # old-style caller: one flag for every distance
                    r = be.update(fr.cir, moving=blip)
                else:                                                    # per-distance levels, as the pipeline passes them
                    r = be.update(fr.cir, disturb=np.full(cfg.n_taps, 100.0 if blip else 1.0))
                if r is not None:
                    results.append((fr.t, r))
        assert not any(r.state == "moving" for _, r in results)
        late = [r for t, r in results if t > 22.0]
        assert sum(r.bpm is not None for r in late) >= len(late) - 1
        assert all(abs(r.bpm - 13.0) <= 1.0 for r in late if r.bpm is not None)
        assert late[-1].clean_share < 0.97                                 # the skipped moments are accounted for


def test_long_disturbance_restarts_and_says_moving():
    cfg = get_profile("default")
    sim = EchoSimulator(cfg, seed=2, cn0_db=CN0)
    refl = person_reflectors(0.65, bpm=16.0, chest_db=-32.0, body_db=-30.0, seed=2)
    fe, be = Frontend(cfg), BreathingEstimator(cfg)
    results = []
    for i in range(40):
        for fr in fe.process(sim.render(cfg.fs, refl, start_sample=i * cfg.fs)):
            if fr.cir is not None:
                r = be.update(fr.cir, moving=20.0 <= fr.t < 25.0)        # 5 s of movement
                if r is not None:
                    results.append((fr.t, r))
    during = [r for t, r in results if 22.0 <= t < 25.0]
    assert during and all(r.state == "moving" and r.bpm is None for r in during)
    assert all(r.bpm is None for t, r in results if 25.0 <= t < 33.0)     # the old data is gone: measure again
    assert results[-1][1].state == "tracking" and abs(results[-1][1].bpm - 16.0) <= 1.0
    assert results[-1][1].window_s < 16.0                                 # and only the data after the movement is used


def test_frame_noise_meter_sees_clicks_and_not_our_own_signal(clean):
    """The bins beside our band carry none of the transmitted signal, so their level is the room's
    noise: unmoved by a person walking about, up by 10 dB or more in a frame hit by a click."""
    for name in ("default", "clean", "strong", "low", "mono"):
        cfg = get_profile(name)
        f = guard_bins(cfg) * cfg.fs / cfg.frame_len
        assert len(f) >= 20
        assert all(np.min(np.abs(f - t)) > 300.0 for t in cfg.tone_hz)
        assert not np.any((f > cfg.chirp_lo_hz - 100.0) & (f < cfg.chirp_hi_hz + 100.0))
    sim = EchoSimulator(clean, seed=4, cn0_db=CN0)
    rx = sim.render(6 * clean.fs, walker_reflectors(seed=4, t_end=10.0, y_range_m=(0.8, 1.0), start_s=0.0, mean_gap_s=0.5))
    click_at = int(4.5 * clean.fs)
    rx[click_at:click_at + 96] += (sim._noise_sigma * 30.0 * np.random.default_rng(0).standard_normal((96, clean.n_mics))).astype(np.float32)
    fe = Frontend(clean)
    frames = [fr for i in range(0, len(rx), 4800) for fr in fe.process(rx[i:i + 4800]) if fr.cir is not None]
    amb = np.array([fr.ambient for fr in frames])
    t = np.array([fr.t for fr in frames])
    quiet = np.median(amb)
    hit = (t > 4.48) & (t < 4.58)
    assert amb[hit].max() > 10.0 * quiet                               # the click frame stands out
    assert np.percentile(amb[~hit], 99) < 2.5 * quiet                  # a body walking past 0.9 m away does not


def test_noise_bursts_alone_move_nothing(clean):
    """An empty room with clicks and hisses: no movement, no presence, full weight only on clean frames."""
    sim = EchoSimulator(clean, seed=5, cn0_db=CN0)
    noise = HallNoise(clean, sim._noise_sigma, seed=5, t_end=40.0)
    pipe = Pipeline(clean)
    weights, moving, states = [], [], set()
    for start in range(0, 30 * clean.fs, 960):
        for fr in pipe.process(noise.add(sim.render(960, [], start_sample=start), start)):
            if fr.presence is not None:
                weights.append(pipe.presence.frame_weight)
                moving.append(fr.presence.moving)
                states.add(fr.presence.state)
    weights = np.array(weights)
    assert states == {"empty"} and not any(moving)
    assert np.median(weights) > 0.9 and weights.min() < 0.05          # most frames count fully, a click frame hardly at all


def test_passer_by_is_seen_but_is_not_the_person(clean):
    """Somebody with a strong echo crossing 1.4 m away: the echo profile sees the movement (the
    gesture gate needs that), but it is outside the person zone, so nobody "enters" and nothing "moves"."""
    sim = EchoSimulator(clean, seed=6, cn0_db=CN0)
    passer = walker_reflectors(seed=6, t_end=40.0, y_range_m=(1.4, 1.45), start_s=5.0, mean_gap_s=1.0, body_db=-14.0)
    blocks = (sim.render(9600, passer, start_sample=s) for s in range(0, 30 * clean.fs, 9600))
    trace, _ = run_blocks(clean, blocks)
    s = score(trace, 6.0, 30.0)
    assert s.empty_share == 1.0 and s.moving_onsets == 0
    seen = trace.between(6.0, 30.0)
    assert np.max(trace.far_db[seen]) > 8.0 and np.max(trace.fast_db[seen]) < 5.0

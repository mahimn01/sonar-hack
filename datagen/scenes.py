"""scenes.py — one random scene per class.

Plain words: for every training example we invent a small story, e.g. "a
standing right-handed person raises their hand, swipes it 35 cm to the left in
0.3 s at 30 cm from the screen, then lets it drop, while someone walks past
3 m behind". The story is random every time (who, how fast, how far, which
hand, what else is going on) but the LABEL is known because we wrote the story.

Classes produced here
    gestures : swipe_left, swipe_right, push, pull, wave
    negatives: none  (still person, fidgeting, typing, slow drifts, bystanders,
                      clicks, desk bumps, half gestures, hand raises ...)
    presence : empty, present, entering, leaving

Directions are from the USER's point of view: swipe_right = the hand travels
towards the user's right (+x). push = hand moves towards the screen.

Read-only helper module.
"""

from __future__ import annotations

import numpy as np

from .kinematics import (
    CTRL_HZ,
    Path,
    Person,
    Scatterer,
    Scene,
    add_tremor,
    arm_scatterers,
    sample_person,
    time_axis,
    torso_scatterers,
    walker_scatterers,
)
from .physics import unit

GESTURES = ("swipe_left", "swipe_right", "push", "pull", "wave")
PRESENCE = ("empty", "present", "entering", "leaving")
LABELS = GESTURES + ("none",) + PRESENCE

NONE_VARIANTS = {
    # name: probability. Together these define what the model must NOT react to.
    "still": 0.13,
    "empty": 0.08,
    "fidget": 0.17,
    "typing": 0.10,
    "bystander": 0.14,
    "slow_drift": 0.12,
    "hand_raise": 0.12,
    "clicks": 0.07,
    "posture": 0.07,
}


def _logu(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _normals(t: np.ndarray, keys: list, rng: np.random.Generator, wobble: float = 0.05) -> np.ndarray:
    """Palm facing direction over time from (time, direction) keyframes."""
    times = np.array([k[0] for k in keys])
    vecs = np.array([unit(np.asarray(k[1], dtype=float)) for k in keys])
    out = np.stack([np.interp(t, times, vecs[:, i]) for i in range(3)], axis=1)
    if wobble > 0:
        from .physics import smooth_noise

        out = out + wobble * np.stack([smooth_noise(rng, len(t), CTRL_HZ, 1.5) for _ in range(3)], axis=1)
    return unit(out)


def _home(rng: np.random.Generator, person: Person) -> np.ndarray:
    """Where the hand rests: on the desk when seated, hanging by the side when standing."""
    side = 1.0 if person.right_handed else -1.0
    if person.standing and rng.random() < 0.7:
        return person.chest + np.array([side * rng.uniform(0.2, 0.3), rng.uniform(-0.08, 0.05), -rng.uniform(0.3, 0.5)])
    y = rng.uniform(0.14, max(0.2, min(0.45, person.chest[1] - 0.12)))
    return np.array([side * rng.uniform(0.08, 0.3), y, rng.uniform(0.0, 0.07)])


def _zone(rng: np.random.Generator, person: Person) -> tuple:
    """Where in front of the screen the gesture happens: (y_max, z0)."""
    y_max = float(np.clip(person.chest[1] - 0.10, 0.22, 0.62))
    z0 = rng.uniform(0.10, 0.45) if person.standing else rng.uniform(0.05, 0.38)
    return y_max, z0


def _relax(t: np.ndarray, t_from: float, factor: float) -> np.ndarray:
    """Reflectivity multiplier: after `t_from` the hand relaxes (weaker, sloppier echo)."""
    s = np.clip((t - t_from) / 0.15, 0.0, 1.0)
    return 1.0 + (factor - 1.0) * s


def _person_scene(
    rng: np.random.Generator, t: np.ndarray, person: Person, hand: np.ndarray, normal: np.ndarray, hand_rho: np.ndarray | float = 1.0
) -> list:
    """Torso + head + the gesturing arm."""
    torso, breath, offset = torso_scatterers(rng, t, person)
    hand = add_tremor(rng, hand, drift_mm=rng.uniform(0.5, 3.0), tremor_mm=rng.uniform(0.05, 0.4))
    arm = arm_scatterers(rng, t, person, hand, normal, body_offset=offset, hand_rho=hand_rho)
    return torso + arm


def _add_distractors(rng: np.random.Generator, scene: Scene, p_walker: float = 0.22, p_bump: float = 0.12) -> None:
    """Things that happen around any scene in a busy hall."""
    t = scene.t
    if rng.random() < p_walker:
        for _ in range(int(rng.integers(1, 3))):
            scene.scatterers += _bystander(rng, t)
            scene.meta["bystanders"] = scene.meta.get("bystanders", 0) + 1
    if rng.random() < p_bump:
        scene.lid_wobble = _bumps(rng, t, int(rng.integers(1, 4)))
        scene.meta["desk_bumps"] = True


def _bumps(rng: np.random.Generator, t: np.ndarray, n: int) -> np.ndarray:
    """Desk bumps / key presses shake the lid: the mic moves a few micrometres to a fraction of a mm."""
    w = np.zeros(len(t))
    for _ in range(n):
        t0 = rng.uniform(t[0], t[-1])
        amp = _logu(rng, 4e-6, 3e-4)
        f = rng.uniform(4.0, 25.0)
        tau = rng.uniform(0.08, 0.5)
        tt = np.clip(t - t0, 0.0, None)
        w += amp * np.exp(-tt / tau) * np.sin(2 * np.pi * f * tt) * (t >= t0)
    return w


def _bystander(rng: np.random.Generator, t: np.ndarray, near: tuple = (1.4, 5.0)) -> list:
    """Someone walking past in a straight line, 1.4-5 m away."""
    dist = rng.uniform(*near)
    bearing = rng.uniform(-1.3, 1.3)  # mostly in front of / beside the screen
    closest = dist * np.array([np.sin(bearing), np.cos(bearing)])
    direction = unit(np.array([np.cos(bearing), -np.sin(bearing)]) + rng.normal(0, 0.4, 2))
    speed = rng.uniform(0.5, 1.7)
    t_pass = rng.uniform(t[0], t[-1])
    path = closest[None, :] + (t - t_pass)[:, None] * speed * direction[None, :]
    rho = float(np.exp(rng.normal(0.6, 0.4)))  # whole bodies reflect more than a hand
    return walker_scatterers(rng, t, path, chest_z=rng.uniform(0.2, 0.6), rho=rho, legs_visible=rng.uniform(0.0, 0.7))


# --------------------------------------------------------------------------
# Gestures
# --------------------------------------------------------------------------
def _swipe(rng: np.random.Generator, t: np.ndarray, duration: float, direction: float, event_center: float | None) -> Scene:
    person = sample_person(rng)
    y_max, z0 = _zone(rng, person)
    a = rng.uniform(0.08, 0.28)  # half the travel distance
    xc = rng.normal(0.0, 0.05)
    y0 = rng.uniform(0.12, y_max)
    # real recorded swipes last ~0.4-0.8 s and peak near 1 m/s; keep fast flicks too, cap at ~3.5 m/s
    dur = max(_logu(rng, 0.25, 1.1), 1.875 * 2 * a / 3.5)
    start = np.array([xc - direction * a, y0 + rng.normal(0, 0.03), z0 + rng.normal(0, 0.03)])
    end = np.array([xc + direction * a, y0 + rng.normal(0, 0.04), max(0.02, z0 + rng.normal(0, 0.04))])
    bulge = np.array([0.0, rng.normal(0, 0.04), rng.normal(0, 0.03)])
    tc = float(event_center) if event_center is not None else rng.uniform(0.9, max(1.0, duration - 0.9))
    t1, t2 = tc - dur / 2, tc + dur / 2

    # Which hand? Most people sweep with the hand that already rests on the side the stroke
    # STARTS from (no wind-up needed). The rest must first carry the hand across the screen:
    # that wind-up is a real movement in the OPPOSITE direction, only slower and lower.
    if rng.random() < 0.65:
        person.right_handed = direction < 0  # swipe_left starts on the right-hand side
    home = _home(rng, person)
    v_stroke = 1.875 * 2 * a / dur
    hover = rng.random() < (0.25 if v_stroke > 0.8 else 0.5)  # hand already waiting at the start point
    gap = rng.uniform(0.05, 0.5)
    prep = max(rng.uniform(0.35, 1.0), 1.875 * float(np.linalg.norm(start - home)) / (v_stroke * rng.uniform(0.3, 0.75)))
    prep = min(prep, 1.4)  # must fit before the stroke
    path = Path(t, start if hover else home, t0=t1 - (0.0 if hover else prep + gap))
    if not hover:
        path.move(start, prep, bulge=np.array([0.0, rng.uniform(0.0, 0.10), rng.uniform(-0.08, 0.04)]))
        path.hold(gap)
    path.move(end, dur, bulge=bulge, skew=rng.uniform(0.85, 1.25))
    after = rng.choice(["return", "stay"], p=[0.7, 0.3])
    relax_from, relax = t[-1] + 1.0, 1.0
    if after == "return":
        path.hold(rng.uniform(0.05, 0.6))
        relax_from, relax = path.now, rng.uniform(0.3, 0.9)
        # the way back is slower, lower and sloppier than the stroke
        back = max(rng.uniform(0.45, 1.1), 1.875 * float(np.linalg.norm(end - home)) / (v_stroke * rng.uniform(0.3, 0.8)))
        back = min(back, 1.6)
        path.move(home, back, bulge=np.array([0.0, rng.uniform(0.0, 0.08), -rng.uniform(0.0, 0.1)]))

    style = rng.choice(["facing", "edge", "down"], p=[0.4, 0.35, 0.25])
    n0 = {"facing": [0.0, -1.0, 0.15], "edge": [direction, -0.25, 0.0], "down": [0.0, -0.3, -1.0]}[style]
    n0 = np.array(n0) + rng.normal(0, 0.25, 3)
    n1 = n0 + rng.normal(0, 0.45, 3) + np.array([0.6 * direction * rng.random(), 0.0, 0.0])  # wrist flick
    normal = _normals(t, [(t1, n0), (t2, n1)], rng)
    hand_rho = float(np.exp(rng.normal(0.0, 0.3))) * _relax(t, relax_from, relax)

    label = "swipe_right" if direction > 0 else "swipe_left"
    sc = Scene(t=t, scatterers=_person_scene(rng, t, person, path.pos, normal, hand_rho), label=label)
    sc.events = [{"label": label, "t_start": t1, "t_end": t2}]
    sc.meta = {"variant": style, "windup": bool(not hover and (home[0] * direction > 0)),
               "stroke_s": dur, "travel_m": 2 * a, "peak_speed": 1.875 * 2 * a / dur,
               "hand_y": y0, "hand_z": z0, **_person_meta(person)}
    return sc


def _push_pull(rng: np.random.Generator, t: np.ndarray, duration: float, label: str, event_center: float | None) -> Scene:
    person = sample_person(rng)
    y_max, z0 = _zone(rng, person)
    x0 = rng.normal(0.0, 0.06)
    y_far = rng.uniform(0.30, max(0.34, y_max))
    y_near = rng.uniform(0.07, max(0.09, min(0.26, y_far - 0.12)))
    travel = y_far - y_near
    dur = max(_logu(rng, 0.22, 0.9), 1.875 * travel / 3.0)  # real pushes/pulls: ~0.5-0.7 s
    far = np.array([x0 + rng.normal(0, 0.02), y_far, z0 + rng.normal(0, 0.03)])
    near = np.array([x0, y_near, max(0.03, z0 + rng.normal(0, 0.03))])
    bulge = np.array([rng.normal(0, 0.015), 0.0, rng.normal(0, 0.02)])
    tc = float(event_center) if event_center is not None else rng.uniform(0.9, max(1.0, duration - 0.9))
    t1, t2 = tc - dur / 2, tc + dur / 2
    home = _home(rng, person)
    relax_from, relax = t[-1] + 1.0, 1.0

    if label == "push":
        hover = rng.random() < 0.3
        prep, gap = rng.uniform(0.4, 1.0), rng.uniform(0.05, 0.5)
        path = Path(t, far if hover else home, t0=t1 - (0.0 if hover else prep + gap))
        if not hover:
            path.move(far, prep)
            path.hold(gap)
        path.move(near, dur, bulge=bulge, skew=rng.uniform(0.85, 1.25))
        after = rng.choice(["tap", "slow", "stay"], p=[0.45, 0.35, 0.2])
        if after == "tap":  # push and come straight back, a bit slower
            path.hold(rng.uniform(0.0, 0.2))
            path.move(far + rng.normal(0, 0.03, 3), dur * rng.uniform(1.3, 3.0))
        elif after == "slow":
            path.hold(rng.uniform(0.3, 0.9))
            relax_from, relax = path.now, rng.uniform(0.3, 0.9)
            path.move(home, rng.uniform(0.6, 1.4))
        variant = after
    else:
        approach = rng.random() < 0.65  # hand first drifts in slowly, then pulls back fast
        appr, gap = rng.uniform(0.6, 1.5), rng.uniform(0.1, 0.7)
        path = Path(t, home if approach else near, t0=t1 - (appr + gap if approach else 0.0))
        if approach:
            path.move(near, appr)
            path.hold(gap)
        path.move(far, dur, bulge=bulge, skew=rng.uniform(0.85, 1.25))
        after = rng.choice(["stay", "drop"], p=[0.4, 0.6])
        if after == "drop":
            path.hold(rng.uniform(0.2, 0.8))
            relax_from, relax = path.now, rng.uniform(0.3, 0.9)
            path.move(home, rng.uniform(0.6, 1.4))
        variant = ("approach+" if approach else "") + after

    n0 = np.array([0.0, -1.0, 0.12]) + rng.normal(0, 0.22, 3)
    normal = _normals(t, [(t1, n0), (t2, n0 + rng.normal(0, 0.15, 3))], rng)
    hand_rho = float(np.exp(rng.normal(0.0, 0.3))) * _relax(t, relax_from, relax)
    sc = Scene(t=t, scatterers=_person_scene(rng, t, person, path.pos, normal, hand_rho), label=label)
    sc.events = [{"label": label, "t_start": t1, "t_end": t2}]
    sc.meta = {"variant": variant, "stroke_s": dur, "travel_m": travel, "peak_speed": 1.875 * travel / dur,
               "hand_y": y_near, "hand_z": z0, **_person_meta(person)}
    return sc


def _wave(rng: np.random.Generator, t: np.ndarray, duration: float, event_center: float | None) -> Scene:
    person = sample_person(rng)
    y_max, z0 = _zone(rng, person)
    xc, y0 = rng.normal(0.0, 0.07), rng.uniform(0.18, y_max)
    amp = rng.uniform(0.05, 0.14)
    f = rng.uniform(1.6, 3.8)
    halves = int(rng.integers(4, 9))
    dur = halves / (2 * f)
    tc = float(event_center) if event_center is not None else rng.uniform(0.6 + dur / 2, max(0.7 + dur / 2, duration - 0.6 - dur / 2))
    t1, t2 = tc - dur / 2, tc + dur / 2
    centre = np.array([xc, y0, z0 + 0.05])
    home = _home(rng, person)
    prep, gap = rng.uniform(0.35, 0.9), rng.uniform(0.0, 0.3)
    path = Path(t, home, t0=t1 - prep - gap)
    path.move(centre, prep)
    path.hold(gap + dur + rng.uniform(0.05, 0.5))
    relax_from = path.now
    path.move(home, rng.uniform(0.5, 1.1))
    u = np.clip((t - t1) / dur, 0.0, 1.0)
    envl = np.sin(np.pi * u) ** 0.5 * ((t >= t1) & (t <= t2))
    osc = amp * envl * np.sin(2 * np.pi * f * (t - t1))
    pos = path.pos.copy()
    pos[:, 0] += osc
    pos[:, 2] -= 0.6 * osc**2 / max(person.forearm, 0.2)  # the hand swings on an arc about the elbow
    tilt = rng.uniform(0.2, 0.8) * osc / amp  # the palm rocks as the wrist pivots
    normal = unit(np.stack([np.sin(tilt), -np.cos(tilt), np.full(len(t), 0.1)], axis=1) + rng.normal(0, 0.1, 3))
    hand_rho = float(np.exp(rng.normal(0.0, 0.3))) * _relax(t, relax_from, rng.uniform(0.3, 0.9))
    sc = Scene(t=t, scatterers=_person_scene(rng, t, person, pos, normal, hand_rho), label="wave")
    sc.events = [{"label": "wave", "t_start": t1, "t_end": t2}]
    sc.meta = {"variant": "wave", "stroke_s": dur, "wave_hz": f, "travel_m": 2 * amp,
               "peak_speed": 2 * np.pi * f * amp, "hand_y": y0, "hand_z": z0, **_person_meta(person)}
    return sc


def _mocap(rng: np.random.Generator, t: np.ndarray, duration: float, label: str, event_center: float | None, library) -> Scene | None:
    """A gesture (or, for "none", some other hand movement) replayed from a REAL tracked hand path (motion.py).

    The body around the hand (torso, arm, fingers, tremor) is built exactly as for the formula gestures.
    Returns None if the library has no clip for this label: the caller then uses the formula.
    """
    from .motion import place_motion

    drawn = library.draw(rng, "other" if label == "none" else label)
    if drawn is None:
        return None
    clip, mirrored = drawn
    person = sample_person(rng)
    y_max, z0 = _zone(rng, person)
    if label in ("push", "pull"):
        y_far = rng.uniform(0.30, max(0.34, y_max))
        y_near = rng.uniform(0.07, max(0.09, min(0.26, y_far - 0.12)))
        centre = np.array([rng.normal(0.0, 0.06), 0.5 * (y_far + y_near), z0 + rng.normal(0, 0.03)])
    else:
        centre = np.array([rng.normal(0.0, 0.05), rng.uniform(0.12, y_max), z0 + rng.normal(0, 0.03)])
        if label != "none" and rng.random() < 0.65:          # most people sweep with the hand that starts on that side
            person.right_handed = label == "swipe_left"
    tc = float(event_center) if event_center is not None else rng.uniform(0.9, max(1.0, duration - 0.9))
    placed = place_motion(rng, t, clip, mirrored, label, centre, tc, _home(rng, person), y_max)
    relax_from = placed["relax_from"] if placed["after"] == "return" else t[-1] + 1.0
    hand_rho = float(np.exp(rng.normal(0.0 if label != "none" else -0.3, 0.3))) * _relax(t, relax_from, rng.uniform(0.3, 0.9))
    sc = Scene(t=t, scatterers=_person_scene(rng, t, person, placed["pos"], placed["normal"], hand_rho), label=label)
    if label != "none":
        sc.events = [{"label": label, "t_start": placed["t_start"], "t_end": placed["t_end"]}]
    sc.meta = {"variant": "mocap", "motion_source": clip.get("source", ""), "mirrored": bool(mirrored),
               "stroke_s": placed["t_end"] - placed["t_start"], "travel_m": placed["travel_m"], "peak_speed": placed["peak_speed"],
               "hand_y": float(centre[1]), "hand_z": float(centre[2]), **_person_meta(person)}
    return sc


# --------------------------------------------------------------------------
# Negatives ("none")
# --------------------------------------------------------------------------
def _none(rng: np.random.Generator, t: np.ndarray, duration: float, variant: str | None = None) -> Scene:
    names = list(NONE_VARIANTS)
    variant = variant or str(rng.choice(names, p=np.array(list(NONE_VARIANTS.values())) / sum(NONE_VARIANTS.values())))
    sc = Scene(t=t, label="none")
    sc.meta = {"variant": variant}
    if variant == "empty":
        return sc
    if variant == "bystander" and rng.random() < 0.5:  # nobody at the laptop, people walking past
        for _ in range(int(rng.integers(1, 4))):
            sc.scatterers += _bystander(rng, t)
        return sc

    person = sample_person(rng)
    sc.meta.update(_person_meta(person))
    home = _home(rng, person)
    side = 1.0 if person.right_handed else -1.0
    path = Path(t, home, t0=0.0)
    n_keys = [(t[0], np.array([0.0, -0.4, -1.0]) + rng.normal(0, 0.3, 3))]
    hand_rho = float(np.exp(rng.normal(-0.3, 0.4)))
    body_shift = None

    if variant in ("still", "clicks", "bystander"):
        pass
    elif variant == "fidget":  # reach for a cup, scratch the head, adjust glasses: slow and roundabout
        for _ in range(int(rng.integers(1, 4))):
            path.hold(rng.uniform(0.1, 1.0))
            target = person.chest + np.array([side * rng.uniform(-0.1, 0.35), -rng.uniform(0.05, 0.45), rng.uniform(-0.3, 0.4)])
            target[1] = max(target[1], 0.1)
            path.move(target, rng.uniform(0.6, 1.6), bulge=rng.normal(0, 0.04, 3))
            n_keys.append((path.now, rng.normal(0, 1.0, 3)))
            if rng.random() < 0.7:
                path.hold(rng.uniform(0.1, 0.8))
                path.move(home, rng.uniform(0.6, 1.5))
    elif variant == "typing":  # small quick finger/hand hops right above the keyboard
        k = np.array([side * rng.uniform(0.0, 0.12), rng.uniform(0.08, 0.2), rng.uniform(0.01, 0.04)])
        path = Path(t, k)
        while path.now < t[-1]:
            path.hold(rng.uniform(0.05, 0.4))
            hop = k + np.array([rng.normal(0, 0.03), rng.normal(0, 0.02), abs(rng.normal(0, 0.008))])
            path.move(hop, rng.uniform(0.06, 0.18))
        sc.lid_wobble = _bumps(rng, t, int(rng.integers(3, 12)))
    elif variant == "slow_drift":  # looks like a gesture but far too slow / too small to count
        y_max, z0 = _zone(rng, person)
        kind = rng.choice(["lateral", "radial", "tiny"])
        p0 = np.array([rng.uniform(-0.25, 0.25), rng.uniform(0.15, y_max), z0])
        path = Path(t, p0, t0=rng.uniform(0.2, max(0.3, duration - 1.6)))
        if kind == "lateral":
            p1 = p0 + np.array([-np.sign(p0[0] or 1.0) * rng.uniform(0.15, 0.4), rng.normal(0, 0.03), rng.normal(0, 0.03)])
            path.move(p1, rng.uniform(1.4, 2.6))
        elif kind == "radial":
            p1 = p0 + np.array([0.0, rng.choice([-1.0, 1.0]) * rng.uniform(0.1, 0.25), 0.0])
            p1[1] = np.clip(p1[1], 0.08, y_max + 0.05)
            path.move(p1, rng.uniform(1.4, 2.6))
        else:  # a quick but tiny twitch (under ~4 cm)
            path.move(p0 + rng.normal(0, 0.02, 3), rng.uniform(0.12, 0.3))
        n_keys = [(t[0], np.array([0.0, -1.0, 0.1]) + rng.normal(0, 0.3, 3))]
        hand_rho = float(np.exp(rng.normal(0.0, 0.3)))
    elif variant == "hand_raise":  # lifting the hand into position, or dropping it, with no gesture
        y_max, z0 = _zone(rng, person)
        up = np.array([rng.uniform(-0.25, 0.25), rng.uniform(0.15, y_max), z0])
        path = Path(t, home, t0=rng.uniform(0.1, max(0.2, duration - 1.5)))
        path.move(up, rng.uniform(0.35, 1.0), bulge=np.array([0.0, 0.0, rng.uniform(0, 0.06)]))
        if rng.random() < 0.6:
            path.hold(rng.uniform(0.2, 1.2))
            path.move(home, rng.uniform(0.45, 1.2))
        n_keys = [(t[0], np.array([0.0, -1.0, 0.1]) + rng.normal(0, 0.4, 3))]
        hand_rho = float(np.exp(rng.normal(0.0, 0.3)))
    elif variant == "posture":  # the whole body leans or shifts a few centimetres
        t0 = rng.uniform(0.2, max(0.3, duration - 1.0))
        shift = Path(t, np.zeros(3), t0=t0)
        shift.move(np.array([rng.normal(0, 0.04), rng.normal(0, 0.07), rng.normal(0, 0.02)]), rng.uniform(0.5, 1.8))
        body_shift = shift.pos

    normal = _normals(t, n_keys if len(n_keys) > 1 else n_keys + [(t[-1], n_keys[0][1])], rng, wobble=0.1)
    torso, _, offset = torso_scatterers(rng, t, person, offset=body_shift)
    hand = add_tremor(rng, path.pos + (body_shift if body_shift is not None else 0.0), 2.0, 0.3)
    sc.scatterers = torso + arm_scatterers(rng, t, person, hand, normal, body_offset=offset, hand_rho=hand_rho)
    if variant == "bystander":
        for _ in range(int(rng.integers(1, 3))):
            sc.scatterers += _bystander(rng, t)
    return sc


# --------------------------------------------------------------------------
# Presence
# --------------------------------------------------------------------------
def _person_meta(person: Person) -> dict:
    return {
        "standing": bool(person.standing),
        "right_handed": bool(person.right_handed),
        "torso_y": float(person.chest[1]),
        "breath_bpm": float(person.breath_bpm),
        "breath_mm": float(person.breath_mm),
    }


def _present(rng: np.random.Generator, t: np.ndarray, duration: float, breathing_only: bool = False, **over: float | bool | None) -> Scene:
    """A person standing or sitting still in front of the laptop (breathing + small sway)."""
    person = sample_person(rng, near=rng.random() < 0.6)
    for k, v in over.items():
        if k == "distance":
            person.chest[1] = float(v)
        elif v is not None:
            setattr(person, k, v)
    if breathing_only:
        person.sway_mm = min(person.sway_mm, 1.0)
    torso, breath, _ = torso_scatterers(rng, t, person)
    sc = Scene(t=t, scatterers=torso, label="present")
    sc.meta = {"variant": "still", **_person_meta(person), "distance": float(person.chest[1])}
    sc.meta["breath_wave"] = breath
    return sc


def _walk(rng: np.random.Generator, t: np.ndarray, duration: float, entering: bool) -> Scene:
    """Someone walks up to the laptop and stops (entering), or walks away (leaving)."""
    far = rng.uniform(2.5, 5.0)
    bearing = rng.uniform(-1.2, 1.2)
    p_far = far * np.array([np.sin(bearing), np.cos(bearing)])
    p_near = np.array([rng.normal(0, 0.12), rng.uniform(0.5, 1.2)])
    speed = rng.uniform(0.7, 1.6)
    walk_time = np.linalg.norm(p_far - p_near) / speed * 1.3  # min-jerk: peak is faster than average
    a, b = (p_far, p_near) if entering else (p_near, p_far)
    t_start = rng.uniform(0.2, max(0.3, duration - walk_time - 0.3)) if walk_time < duration - 0.6 else rng.uniform(-0.4 * walk_time, 0.3)
    path = Path(t, np.array([a[0], a[1], 0.0]), t0=t_start)
    side_step = rng.normal(0, 0.25, 2)
    path.move(np.array([b[0], b[1], 0.0]), walk_time, bulge=np.array([side_step[0], side_step[1], 0.0]),
              skew=rng.uniform(0.8, 1.2))
    label = "entering" if entering else "leaving"
    sc = Scene(t=t, label=label)
    sc.scatterers = walker_scatterers(rng, t, path.pos[:, :2], chest_z=rng.uniform(0.25, 0.6),
                                      rho=float(np.exp(rng.normal(0.6, 0.35))), legs_visible=rng.uniform(0.0, 0.7))
    sc.events = [{"label": label, "t_start": float(t_start), "t_end": float(t_start + walk_time)}]
    sc.meta = {"variant": label, "walk_speed": speed, "start_dist": far}
    return sc


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def sample_scene(
    rng: np.random.Generator,
    label: str,
    duration: float = 3.0,
    event_center: float | None = None,
    distractors: bool = True,
    lead_in: float = 0.0,
    motion: tuple | None = None,
    **kwargs,
) -> Scene:
    """Draw one random scene of class `label`.

    motion       (MotionLibrary, share): with probability `share` a gesture replays a real tracked hand path
                 from the library instead of the formula stroke (and a "none" clip, half as often, some other
                 real hand movement). None = formulas only, exactly as before.

    duration     : length of the scripted story in seconds.
    lead_in      : quiet seconds added BEFORE the story (person already there,
                   nothing scripted). The returned scene lasts lead_in + duration
                   and all times in it (events included) count from the start
                   of the audio, i.e. the story starts at t = lead_in.
    event_center : middle of the gesture stroke, in story time; random if None.
    """
    t = time_axis(duration, lead_in)
    sc = None
    if motion is not None and label in ("swipe_left", "swipe_right", "push", "pull", "none"):
        library, share = motion
        if rng.random() < (share if label != "none" else 0.5 * share):
            sc = _mocap(rng, t, duration, label, event_center, library)
    if sc is not None:
        pass
    elif label in ("swipe_left", "swipe_right"):
        sc = _swipe(rng, t, duration, -1.0 if label == "swipe_left" else 1.0, event_center)
    elif label in ("push", "pull"):
        sc = _push_pull(rng, t, duration, label, event_center)
    elif label == "wave":
        sc = _wave(rng, t, duration, event_center)
    elif label == "none":
        sc = _none(rng, t, duration, kwargs.get("variant"))
    elif label == "empty":
        sc = Scene(t=t, label="empty", meta={"variant": "empty"})
        if rng.random() < 0.25:  # an "empty" hall still has people far away
            sc.scatterers += _bystander(rng, t, near=(4.0, 8.0))
    elif label == "present":
        sc = _present(rng, t, duration, **kwargs)
    elif label in ("entering", "leaving"):
        sc = _walk(rng, t, duration, entering=label == "entering")
    else:
        raise ValueError(f"unknown label {label!r}; known: {LABELS}")
    if distractors and label not in ("empty",):
        _add_distractors(rng, sc)
    # from here on, time counts from the start of the audio
    sc.t = sc.t + lead_in
    for ev in sc.events:
        ev["t_start"] = float(ev["t_start"] + lead_in)
        ev["t_end"] = float(ev["t_end"] + lead_in)
    sc.meta["label"] = sc.label
    sc.meta["lead_in"] = float(lead_in)
    sc.meta["duration"] = float(lead_in + duration)
    return sc

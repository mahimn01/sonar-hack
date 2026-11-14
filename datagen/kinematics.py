"""kinematics.py — how hands, arms, torsos and walking people move.

Plain words: a gesture is a short, smooth hand movement (fast in the middle,
slow at both ends). The arm follows the hand, the torso sways a little and
breathes, and bystanders walk past. Each body part is represented by a few
reflecting points whose positions we compute over time. The renderer turns
those moving points into echoes.

All positions are in the laptop frame (metres): x = user's right,
y = from the screen towards the user, z = up, origin at the hinge centre.
Trajectories are produced at a slow "control" rate (default 250 Hz) and
smoothly upsampled later.

Read-only helper module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .physics import minjerk, smooth_noise, unit

CTRL_HZ = 250.0


@dataclass
class Scatterer:
    """One reflecting point.

    pos        [T, 3] position over time
    rho        reflectivity (1.0 = a palm facing the laptop); scalar or [T]
    normal     [T, 3] facing direction, for mirror-like parts (palm, chest); None = no preference
    spec_q     how mirror-like it is (0 = reflects the same in all directions)
    spec_floor the part of the echo that survives when it faces the wrong way
    """

    pos: np.ndarray
    rho: np.ndarray | float = 1.0
    normal: np.ndarray | None = None
    spec_q: float = 0.0
    spec_floor: float = 1.0
    tag: str = ""


@dataclass
class Scene:
    t: np.ndarray  # control times (s); t = 0 is the first output sample
    scatterers: list = field(default_factory=list)
    label: str = "none"
    events: list = field(default_factory=list)  # [{"label", "t_start", "t_end"}]
    meta: dict = field(default_factory=dict)
    lid_wobble: np.ndarray | None = None  # [T] metres of speaker<->mic path change (desk bumps)


def time_axis(duration: float, lead_in: float = 0.0, pad: float = 0.06) -> np.ndarray:
    """Control-rate time axis covering a little more than [-lead_in, duration].

    Time 0 is where the scripted story starts; the `lead_in` seconds before it
    are quiet (person present, nothing scripted) so that the receiver can learn
    its noise floor exactly as it does when the real stream starts.
    """
    n = int(np.ceil((duration + lead_in + 2 * pad) * CTRL_HZ)) + 1
    return np.arange(n) / CTRL_HZ - pad - lead_in


# --------------------------------------------------------------------------
# Hand paths
# --------------------------------------------------------------------------
class Path:
    """Build a hand path from holds and smooth point-to-point moves."""

    def __init__(self, t: np.ndarray, start: np.ndarray, t0: float | None = None) -> None:
        self.t = t
        self.pos = np.tile(np.asarray(start, dtype=float), (len(t), 1))
        self.now = float(t[0] if t0 is None else t0)
        self.at = np.asarray(start, dtype=float).copy()

    def hold(self, dur: float) -> "Path":
        self.now += max(dur, 0.0)
        return self

    def move(self, target: np.ndarray, dur: float, bulge: np.ndarray | None = None, skew: float = 1.0) -> tuple:
        """Smooth move to `target` in `dur` seconds. Returns (t_start, t_end).

        bulge : sideways bow of the path (metres, 3-vector) -> arcs instead of straight lines
        skew  : >1 = slow start / fast finish, <1 = the opposite
        """
        target = np.asarray(target, dtype=float)
        dur = max(dur, 0.02)
        u = np.clip((self.t - self.now) / dur, 0.0, 1.0)
        s = minjerk(u**skew)[:, None]
        seg = (target - self.at)[None, :] * s
        if bulge is not None:
            seg = seg + np.asarray(bulge, dtype=float)[None, :] * (4.0 * s * (1.0 - s))
        self.pos += seg
        t_start, self.now = self.now, self.now + dur
        self.at = target.copy()
        return t_start, self.now


def add_tremor(rng: np.random.Generator, pos: np.ndarray, drift_mm: float = 2.0, tremor_mm: float = 0.25) -> np.ndarray:
    """Hands never hold perfectly still: slow drift plus a faint 8-12 Hz tremor."""
    out = pos.copy()
    n = len(pos)
    for k in range(3):
        out[:, k] += 1e-3 * drift_mm * smooth_noise(rng, n, CTRL_HZ, 0.6)
        out[:, k] += 1e-3 * tremor_mm * smooth_noise(rng, n, CTRL_HZ, 9.0)
    return out


# --------------------------------------------------------------------------
# Body model
# --------------------------------------------------------------------------
@dataclass
class Person:
    """A person in front of the laptop (seated or standing)."""

    chest: np.ndarray  # chest centre
    facing: np.ndarray  # unit vector the chest faces (usually towards the screen, -y)
    upper_arm: float = 0.30
    forearm: float = 0.26
    right_handed: bool = True
    breath_bpm: float = 15.0
    breath_mm: float = 3.0
    breath_ie: float = 0.42  # fraction of the cycle spent inhaling
    sway_mm: float = 1.5
    standing: bool = False
    body_rho: float = 1.0  # overall reflectivity (clothing, build)

    @property
    def shoulder(self) -> np.ndarray:
        side = 1.0 if self.right_handed else -1.0
        right = np.cross(np.array([0.0, 0.0, 1.0]), -self.facing)  # person's right-hand side
        # a person facing the screen (-y) has their right hand on the laptop's +x side
        return self.chest + side * 0.19 * unit(right) + np.array([0.0, 0.0, 0.14])


def sample_person(rng: np.random.Generator, near: bool = True) -> Person:
    """Random user: seated or standing, size, breathing, how much they sway."""
    standing = rng.random() < 0.45
    if near:
        y = rng.uniform(0.45, 1.0) if standing else rng.uniform(0.40, 0.78)
    else:
        y = rng.uniform(0.6, 1.6)
    z = rng.uniform(0.28, 0.65) if standing else rng.uniform(0.08, 0.36)
    chest = np.array([rng.normal(0.0, 0.07), y, z])
    yaw = rng.normal(0.0, 0.25)
    facing = unit(np.array([np.sin(yaw), -np.cos(yaw), rng.normal(0.0, 0.08)]))
    return Person(
        chest=chest,
        facing=facing,
        upper_arm=rng.uniform(0.26, 0.34),
        forearm=rng.uniform(0.23, 0.29),
        right_handed=rng.random() < 0.75,
        breath_bpm=rng.uniform(9.0, 24.0),
        breath_mm=float(np.exp(rng.uniform(np.log(1.2), np.log(7.0)))),
        breath_ie=rng.uniform(0.35, 0.5),
        sway_mm=rng.uniform(1.0, 6.0) if standing else rng.uniform(0.3, 2.2),
        standing=standing,
        body_rho=float(np.exp(rng.normal(0.0, 0.35))),
    )


def breathing_wave(rng: np.random.Generator, t: np.ndarray, bpm: float, ie: float = 0.42, irregular: float = 0.06) -> np.ndarray:
    """Chest movement in [0, 1]: quicker inhale, slower exhale, small pause, slightly irregular.

    Returned at the same times as `t`. 1.0 = fully inhaled (chest closest to the laptop).
    """
    dt = float(t[1] - t[0])
    # the breathing rate itself wanders a few percent, as in real breathing
    rate = (bpm / 60.0) * (1.0 + irregular * smooth_noise(rng, len(t), 1.0 / dt, 0.08))
    phase = np.cumsum(rate) * dt + rng.uniform(0.0, 1.0)
    u = np.mod(phase, 1.0)
    inhale = 0.5 - 0.5 * np.cos(np.pi * np.clip(u / ie, 0.0, 1.0))
    exhale_u = np.clip((u - ie) / (1.0 - ie), 0.0, 1.0)
    exhale = np.exp(-3.2 * exhale_u) * (1.0 - exhale_u**3)  # passive, exponential-like exhale
    wave = np.where(u < ie, inhale, exhale)
    depth = 1.0 + 0.12 * smooth_noise(rng, len(t), 1.0 / dt, 0.1)  # breath-to-breath depth changes
    return wave * depth


def torso_scatterers(rng: np.random.Generator, t: np.ndarray, person: Person, offset: np.ndarray | None = None) -> tuple:
    """Chest/abdomen grid + head. Breathes and sways; `offset` [T,3] moves the whole body.

    Returns (scatterers, breathing wave [T], body offset [T,3]).
    """
    n = len(t)
    facing = person.facing
    right = unit(np.cross(np.array([0.0, 0.0, 1.0]), -facing))
    up = np.array([0.0, 0.0, 1.0])
    sway = np.stack(
        [1e-3 * person.sway_mm * smooth_noise(rng, n, CTRL_HZ, rng.uniform(0.15, 0.5)) for _ in range(3)], axis=1
    )
    sway[:, 2] *= 0.3
    base = person.chest[None, :] + sway + (offset if offset is not None else 0.0)
    breath = breathing_wave(rng, t, person.breath_bpm, person.breath_ie)
    out = []
    for gx in (-0.11, 0.0, 0.11):
        for gz in (-0.17, -0.02, 0.12):
            # the belly and lower chest move most, the shoulders least
            w = (0.45 + 0.55 * (1.0 - abs(gx) / 0.14)) * (1.0 if gz < 0.05 else 0.6)
            p = base + gx * right[None, :] + gz * up[None, :]
            p = p + (1e-3 * person.breath_mm * w) * breath[:, None] * facing[None, :]
            rho = person.body_rho * rng.uniform(0.5, 1.4)
            out.append(Scatterer(p, rho, np.tile(facing, (n, 1)), spec_q=1.0, spec_floor=0.45, tag="torso"))
    head = base + np.array([0.0, 0.0, 0.34]) + 0.02 * facing[None, :]
    head = head + 1e-3 * 1.5 * np.stack([smooth_noise(rng, n, CTRL_HZ, 0.4) for _ in range(3)], axis=1)
    out.append(Scatterer(head, person.body_rho * rng.uniform(0.5, 1.0), tag="head"))
    return out, breath, sway + (offset if offset is not None else 0.0)


def _elbow(shoulder: np.ndarray, hand: np.ndarray, l1: float, l2: float, out_dir: np.ndarray) -> np.ndarray:
    """Two-link arm: find the elbow given shoulder and hand (elbow hangs down and outwards)."""
    d_vec = hand - shoulder
    d = np.linalg.norm(d_vec, axis=1)
    d_c = np.clip(d, abs(l1 - l2) + 1e-3, l1 + l2 - 1e-3)
    u = d_vec / np.maximum(d, 1e-6)[:, None]
    a = (l1 * l1 - l2 * l2 + d_c * d_c) / (2.0 * d_c)
    h = np.sqrt(np.maximum(l1 * l1 - a * a, 0.0))
    w = out_dir[None, :] - np.einsum("ij,j->i", u, out_dir)[:, None] * u
    w = w / np.maximum(np.linalg.norm(w, axis=1), 1e-6)[:, None]
    return shoulder + a[:, None] * u + h[:, None] * w


def arm_scatterers(
    rng: np.random.Generator,
    t: np.ndarray,
    person: Person,
    hand: np.ndarray,
    normal: np.ndarray,
    body_offset: np.ndarray | None = None,
    hand_rho: float = 1.0,
) -> list:
    """Hand (palm + fingers), forearm and upper arm following a hand path.

    hand   [T,3] palm-centre path;  normal [T,3] direction the palm faces.
    """
    n = len(t)
    side = 1.0 if person.right_handed else -1.0
    shoulder = person.shoulder[None, :] + (body_offset if body_offset is not None else 0.0)
    shoulder = np.broadcast_to(shoulder, (n, 3))
    right = unit(np.cross(np.array([0.0, 0.0, 1.0]), -person.facing))
    out_dir = unit(np.array([0.0, 0.0, -1.0]) + 0.5 * side * right)
    elbow = _elbow(shoulder, hand, person.upper_arm, person.forearm, out_dir)

    out = []
    # palm frame: fingers point roughly away from the elbow, within the palm plane
    fingers = hand - elbow
    fingers = fingers - np.einsum("ij,ij->i", fingers, normal)[:, None] * normal
    e1 = fingers / np.maximum(np.linalg.norm(fingers, axis=1), 1e-6)[:, None]
    e2 = np.cross(normal, e1)
    q = rng.uniform(1.0, 6.0)  # how mirror-like this person's palm behaves
    floor = rng.uniform(0.12, 0.5)
    # Real recordings show a THICK Doppler trace: at any instant the echo covers every speed
    # from near zero up to the hand's speed. That is the arm: the elbow barely moves, the wrist
    # moves almost as fast as the hand, and the fingers flick a little faster still.
    arm = float(np.exp(rng.uniform(np.log(0.4), np.log(2.5))))  # how strongly THIS person's arm reflects
    out.append(Scatterer(hand, hand_rho, normal, q, floor, "palm"))
    flick = rng.uniform(0.0, 0.12)  # fingers extend during the stroke: a bit faster than the palm
    lead = hand - np.concatenate([hand[:1], hand[:-1]])  # per-step displacement of the palm
    lead = np.cumsum(lead, axis=0) * flick
    for _ in range(int(rng.integers(4, 8))):
        o1, o2 = rng.uniform(-0.03, 0.11), rng.uniform(-0.045, 0.045)
        p = hand + o1 * e1 + o2 * e2 + rng.normal(0.0, 0.006) * normal + lead * rng.uniform(0.0, 1.0)
        out.append(Scatterer(p, hand_rho * rng.uniform(0.15, 0.6), normal, q * 0.5, min(1.0, floor + 0.2), "finger"))
    # forearm and upper arm: several points along each bone
    axis_fore = unit(hand - elbow)
    nrm = unit(np.cross(axis_fore, np.cross(np.array([0.0, -1.0, 0.3]), axis_fore)))
    arm_floor = rng.uniform(0.15, 0.6)  # sleeves and round limbs scatter in all directions
    for frac in (0.15, 0.3, 0.45, 0.6, 0.8, 0.95):
        p = hand + frac * (elbow - hand)
        out.append(Scatterer(p, hand_rho * arm * rng.uniform(0.2, 0.6), nrm, 1.0, arm_floor, "forearm"))
    for frac in (0.25, 0.5, 0.8):
        p = elbow + frac * (shoulder - elbow)
        out.append(Scatterer(p, hand_rho * arm * rng.uniform(0.15, 0.5), tag="upper_arm"))
    return out


def walker_scatterers(
    rng: np.random.Generator,
    t: np.ndarray,
    path_xy: np.ndarray,
    chest_z: float,
    rho: float = 1.0,
    legs_visible: float = 0.5,
) -> list:
    """A walking person following `path_xy` [T,2]: torso bobbing, swinging arms and legs.

    The limbs swing faster and slower than the body, which paints the
    characteristic "micro-Doppler" fringe around a walker's echo.
    """
    n = len(t)
    dt = float(t[1] - t[0])
    vel = np.gradient(path_xy, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    heading = np.zeros((n, 3))
    heading[:, :2] = vel / np.maximum(speed, 1e-3)[:, None]
    stride_hz = 0.6 + 0.55 * np.clip(speed, 0.0, 2.0)  # faster walk => faster steps
    phase = 2 * np.pi * np.cumsum(stride_hz * (speed > 0.08)) * dt + rng.uniform(0, 2 * np.pi)
    moving = np.clip(speed / 0.5, 0.0, 1.0)

    centre = np.column_stack([path_xy, np.full(n, chest_z)])
    centre[:, 2] += 0.02 * moving * np.sin(2 * phase)  # head/torso bob twice per stride
    side = np.column_stack([-heading[:, 1], heading[:, 0], np.zeros(n)])
    facing = heading.copy()
    facing[speed < 0.08] = np.array([0.0, -1.0, 0.0])
    out = []
    for gx in (-0.12, 0.0, 0.12):
        for gz in (-0.18, 0.0, 0.14):
            p = centre + gx * side + np.array([0.0, 0.0, gz])
            out.append(Scatterer(p, rho * rng.uniform(0.5, 1.3), tag="walker_torso"))
    out.append(Scatterer(centre + np.array([0.0, 0.0, 0.35]), rho * rng.uniform(0.5, 1.0), tag="walker_head"))
    swing_arm = rng.uniform(0.10, 0.22)
    swing_leg = rng.uniform(0.22, 0.36)
    for sgn in (-1.0, 1.0):
        arm = centre + sgn * 0.2 * side + np.array([0.0, 0.0, -0.22])
        arm = arm + (sgn * swing_arm * moving * np.sin(phase))[:, None] * heading
        out.append(Scatterer(arm, rho * rng.uniform(0.2, 0.5), tag="walker_arm"))
        if legs_visible > 0:
            for frac, zz in ((0.5, -0.55), (1.0, -0.95)):
                leg = centre + sgn * 0.09 * side + np.array([0.0, 0.0, zz])
                leg = leg + (-sgn * frac * swing_leg * moving * np.sin(phase))[:, None] * heading
                out.append(Scatterer(leg, rho * legs_visible * rng.uniform(0.3, 0.9), tag="walker_leg"))
    return out

"""motion.py — real human hand movements for the simulator (a library of tracked gestures).

Plain words
-----------
kinematics.py invents every gesture from a formula: a smooth stroke, fast in the middle.
Real people are messier: they hesitate, overshoot, curve, flick the wrist, come back
early. This file lets the simulator replay REAL hand paths instead:

  1. a hand tracker (Google MediaPipe, run by mocap_extract.py on gesture videos) gives
     21 hand points per video frame;
  2. `track_to_motion` turns those into one path of the palm in metres in the laptop's
     frame (the camera is treated as the laptop's webcam), smooths out tracker jitter,
     finds the stroke, and names the gesture FROM THE GEOMETRY (which way did the hand
     really travel), not from the video's file name;
  3. `MotionLibrary` stores a few hundred such paths in one small file;
  4. `place_motion` puts one path into a scene: moved to a random spot in front of the
     screen, a little faster or slower, mirrored for the other swipe direction, with a
     smooth bridge from and to where the hand rests.

scenes.py then builds the usual body (torso, arm, fingers) around that hand path, and
the renderer does not know the difference. The formula gestures stay available: a
library is only used when one is given (generate.py --motion, --motion-share).

Coordinates: laptop frame, metres. x = user's right, y = from the screen towards the
user, z = up, origin at the hinge. Camera frame (as in images): x right, y down, z away
from the camera. A webcam picture is not mirrored, so the camera's right is the USER'S
LEFT: x_laptop = -x_camera.

Read-only helper module apart from MotionLibrary.save().
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path as FsPath

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

GESTURES = ("swipe_left", "swipe_right", "push", "pull")
PALM = (0, 5, 9, 13, 17)          # MediaPipe hand points: wrist and the four finger knuckles
MIRROR = {"swipe_left": "swipe_right", "swipe_right": "swipe_left"}


# --------------------------------------------------------------------------
# From tracked hand points to a hand path in the laptop frame
# --------------------------------------------------------------------------
@dataclass
class Camera:
    """The camera the video was shot with, treated as a laptop webcam at the top of the lid."""

    width: int
    height: int
    fov_deg: float = 65.0          # horizontal field of view (typical webcams: 60-78 degrees); sets the depth scale
    lid_deg: float = 108.0         # how far the screen is opened: the camera looks along the screen normal
    height_m: float = 0.215        # camera position up the lid

    @property
    def focal_px(self) -> float:
        return 0.5 * self.width / np.tan(np.deg2rad(self.fov_deg) / 2)


def hand_positions(image: np.ndarray, world: np.ndarray, cam: Camera) -> np.ndarray:
    """Where the hand is, in camera coordinates (metres), for every frame.

    image [T, 21, 2] hand points in the picture, 0..1 (x right, y down).
    world [T, 21, 3] the same points in metres around the hand's centre, camera-aligned axes.
    The tracker knows the hand's true size (world) and how big it appears (image); a pinhole camera
    then fixes where it must be:   f * (X + tx) = u * (Z + tz),   f * (Y + ty) = v * (Z + tz)
    for all 21 points, solved for the offset (tx, ty, tz) by least squares. Returns [T, 21, 3].
    """
    f = cam.focal_px
    u = image[..., 0] * cam.width - 0.5 * cam.width
    v = image[..., 1] * cam.height - 0.5 * cam.height
    x, y, z = world[..., 0], world[..., 1], world[..., 2]
    n_t = image.shape[0]
    a = np.zeros((n_t, 42, 3))
    b = np.zeros((n_t, 42))
    a[:, :21, 0], a[:, :21, 2], b[:, :21] = f, -u, u * z - f * x
    a[:, 21:, 1], a[:, 21:, 2], b[:, 21:] = f, -v, v * z - f * y
    ata = np.einsum("tij,tik->tjk", a, a)
    atb = np.einsum("tij,ti->tj", a, b)
    t = np.linalg.solve(ata + 1e-9 * np.eye(3), atb[..., None])[..., 0]
    return world + t[:, None, :]


def camera_to_laptop(p_cam: np.ndarray, cam: Camera) -> np.ndarray:
    """Camera coordinates -> laptop frame, with the camera at the top of the lid looking along the screen normal."""
    lid = np.deg2rad(cam.lid_deg)
    normal = np.array([0.0, np.sin(lid), -np.cos(lid)])                  # screen normal, towards the user
    up_on_lid = np.array([0.0, np.cos(lid), np.sin(lid)])                # along the lid, away from the hinge
    origin = cam.height_m * up_on_lid
    right = np.array([-1.0, 0.0, 0.0])                                   # camera right = user's left
    return origin + p_cam[..., 0:1] * right + p_cam[..., 1:2] * (-up_on_lid) + p_cam[..., 2:3] * normal


def _smooth(x: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    """Remove tracker jitter (hands do not move faster than a few Hz; single-frame jumps are noise)."""
    win = int(round(seconds * fps)) | 1
    win = max(5, min(win, (len(x) - 1) | 1))
    if win < 5 or len(x) < 7:
        return x
    return savgol_filter(x, win, 3, axis=0, mode="interp")


def track_to_motion(image: np.ndarray, world: np.ndarray, fps: float, cam: Camera, valid: np.ndarray | None = None,
                    hint: str | None = None) -> dict | None:
    """One tracked video clip -> one hand path, or None if it is unusable.

    Returns {"pos" [T, 3] palm centre in the laptop frame (m), "normal" [T, 3] which way the palm faces,
    "fps", "stroke" (first frame, last frame), "label", "travel_m", "peak_speed", "quality"}.
    `hint` is the kind the video claims to show ("swipe", "radial" = push or pull, "other", or a full label): a clip whose
    geometry contradicts it is dropped (wrong label, or the tracker lost the hand).
    """
    n = len(image)
    valid = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    # the longest stretch in which the hand is in view (gaps of up to 4 frames are bridged)
    best, start, gap = (0, 0), None, 0
    for i in range(n + 1):
        if i < n and valid[i]:
            start, gap = (i if start is None else start), 0
            if i - start + 1 > best[1] - best[0]:
                best = (start, i + 1)
        elif start is not None:
            gap += 1
            if gap > 4 or i == n:
                start = None
    lo, hi = best[0], best[1] - 1
    if lo > 0:
        lo += 2                        # the hand is coming into the picture: the first frames see only part of it
    if hi < n - 1:
        hi -= 2
    if hi - lo + 1 < 10:
        return None
    image, world, valid = image[lo:hi + 1].astype(np.float64), world[lo:hi + 1].astype(np.float64), valid[lo:hi + 1]
    t_all = np.arange(len(valid))
    for arr in (image, world):
        flat = arr.reshape(len(arr), -1)
        for c in range(flat.shape[1]):
            flat[~valid, c] = np.interp(t_all[~valid], t_all[valid], flat[valid, c])
    pts = camera_to_laptop(hand_positions(image, world, cam), cam)        # [T, 21, 3]
    palm = pts[:, PALM].mean(axis=1)
    # depth from apparent size is the noisy part: smooth it more than the picture-plane motion
    lid = np.deg2rad(cam.lid_deg)
    axis = np.array([0.0, np.sin(lid), -np.cos(lid)])
    along = palm @ axis
    across = palm - along[:, None] * axis
    palm = _smooth(across, fps, 0.20) + _smooth(along, fps, 0.36)[:, None] * axis
    side = pts[:, 5] - pts[:, 17]
    up = pts[:, 9] - pts[:, 0]
    normal = _smooth(np.cross(side, up), fps, 0.3)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    palm, normal = _complete(palm, normal, fps)

    vel = np.gradient(palm, 1.0 / fps, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    if not np.all(np.isfinite(palm)) or speed.max() > 8.0:               # a tracking jump, not a hand
        return None
    axes = {"swipe": (0,), "swipe_left": (0,), "swipe_right": (0,), "radial": (1,), "push": (1,), "pull": (1,)}.get(hint or "", (0, 1))
    stroke = _find_stroke(palm, vel, fps, axes)
    if stroke is None:
        label, i0, i1, travel = "other", 0, len(palm) - 1, 0.0
    else:
        label, i0, i1, travel = stroke
    if hint and hint != "other":
        family = "swipe" if label.startswith("swipe") else ("radial" if label in ("push", "pull") else label)
        if hint not in (label, family):                                   # "radial" = push or pull, whichever the path shows
            return None
    elif hint == "other":
        label, i0, i1 = "other", 0, len(palm) - 1
    return {"pos": palm.astype(np.float32), "normal": normal.astype(np.float32), "fps": float(fps), "stroke": (int(i0), int(i1)),
            "label": label, "travel_m": float(travel), "peak_speed": float(speed[i0:i1 + 1].max()), "quality": float(valid.mean()),
            "seen_s": float(len(valid) / fps)}


def _complete(palm: np.ndarray, normal: np.ndarray, fps: float, ramp_s: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Give a path that enters or leaves the picture at speed a smooth start and stop.

    A webcam sees only the middle of a wide sweep: the hand is already moving when it comes into view and
    still moving when it leaves. A real hand started from rest and comes to rest, so the missing ends are
    added as a quarter of a second of smooth speeding up / slowing down along the direction of travel.
    The part the camera saw is kept as it is.
    """
    n = int(round(ramp_s * fps))
    tau = (np.arange(1, n + 1) / fps)[:, None]
    ease = tau / 2 + (ramp_s / (2 * np.pi)) * np.sin(np.pi * tau / ramp_s)          # distance for unit speed, cosine ramp
    k = min(3, len(palm) - 1)
    v_end = (palm[-1] - palm[-1 - k]) * fps / k
    v_start = (palm[k] - palm[0]) * fps / k
    if np.linalg.norm(v_end) > 0.2:
        palm = np.concatenate([palm, palm[-1] + v_end * ease])
        normal = np.concatenate([normal, np.repeat(normal[-1:], n, axis=0)])
    if np.linalg.norm(v_start) > 0.2:
        palm = np.concatenate([(palm[0] - v_start * ease)[::-1], palm])
        normal = np.concatenate([np.repeat(normal[:1], n, axis=0), normal])
    return palm, normal


def _find_stroke(palm: np.ndarray, vel: np.ndarray, fps: float, axes: tuple = (0, 1)) -> tuple | None:
    """The one clear movement of a gesture clip: (label, first frame, last frame, travel in m), named from geometry.

    Sideways (x) against towards/away (y, the depth axis): the larger net travel wins. The stroke is the
    stretch around the fastest moment where the hand keeps moving the same way at more than 15 % of that speed.
    """
    best = None
    for axis, names in ((0, ("swipe_left", "swipe_right")), (1, ("push", "pull"))):
        if axis not in axes:
            continue
        v = vel[:, axis]
        k = int(np.argmax(np.abs(v)))
        if abs(v[k]) < 0.12:
            continue
        sign = np.sign(v[k])
        thr = 0.15 * abs(v[k])
        i0 = k
        while i0 > 0 and sign * v[i0 - 1] > thr:
            i0 -= 1
        i1 = k
        while i1 < len(v) - 1 and sign * v[i1 + 1] > thr:
            i1 += 1
        travel = abs(palm[i1, axis] - palm[i0, axis])
        dur = (i1 - i0) / fps
        if travel < 0.05 or not 0.12 <= dur <= 2.0:
            continue
        other = np.abs(palm[i1] - palm[i0])[[a for a in (0, 1, 2) if a != axis]].max()
        score = travel - 0.6 * other                                       # a clean stroke stays on its axis
        if best is None or score > best[0]:
            best = (score, names[1] if sign > 0 else names[0], i0, i1, travel)
    if best is None or best[0] <= 0.02:
        return None
    return best[1:]


# --------------------------------------------------------------------------
# The library file
# --------------------------------------------------------------------------
class MotionLibrary:
    """A few hundred hand paths in one .npz file (float32, a few MB)."""

    def __init__(self, clips: list[dict]) -> None:
        self.clips = clips
        self.by_label: dict[str, list[int]] = {}
        for i, c in enumerate(clips):
            self.by_label.setdefault(c["label"], []).append(i)
            if "peak_speed" not in c:
                # The file stores the path only. The stroke's fastest moment is read off it here, because
                # place_motion sets the tempo from it. (Until 2026-11-14 a loaded clip had no peak speed, which
                # made place_motion replay every clip at the fastest tempo it allows.)
                i0, i1 = c["stroke"]
                step = np.linalg.norm(np.diff(np.asarray(c["pos"][i0:i1 + 1], np.float64), axis=0), axis=1)
                c["peak_speed"] = float(step.max() * c["fps"]) if len(step) else 0.0

    def __len__(self) -> int:
        return len(self.clips)

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in sorted(self.by_label.items())}

    def save(self, path: str | FsPath) -> None:
        clips = self.clips
        lengths = np.array([len(c["pos"]) for c in clips], np.int64)
        np.savez_compressed(
            path, pos=np.concatenate([c["pos"] for c in clips]).astype(np.float32),
            normal=np.concatenate([c["normal"] for c in clips]).astype(np.float32), length=lengths,
            fps=np.array([c["fps"] for c in clips], np.float32), stroke=np.array([c["stroke"] for c in clips], np.int32),
            label=np.array([c["label"] for c in clips]), source=np.array([c.get("source", "") for c in clips]),
            subject=np.array([str(c.get("subject", "")) for c in clips]),
            quality=np.array([c.get("quality", 1.0) for c in clips], np.float32))

    @classmethod
    def load(cls, path: str | FsPath) -> "MotionLibrary":
        with np.load(path, allow_pickle=False) as z:
            ends = np.cumsum(z["length"])
            starts = ends - z["length"]
            clips = [{"pos": z["pos"][a:b], "normal": z["normal"][a:b], "fps": float(z["fps"][i]),
                      "stroke": tuple(int(v) for v in z["stroke"][i]), "label": str(z["label"][i]),
                      "source": str(z["source"][i]), "subject": str(z["subject"][i]), "quality": float(z["quality"][i])}
                     for i, (a, b) in enumerate(zip(starts, ends))]
        return cls(clips)

    def draw(self, rng: np.random.Generator, label: str) -> tuple[dict, bool] | None:
        """A random clip that shows `label`. Swipes may come from a clip of the other direction, mirrored.

        Returns (clip, mirrored) or None if the library has nothing for this label.
        """
        pool = [(i, False) for i in self.by_label.get(label, [])]
        if label in MIRROR:
            pool += [(i, True) for i in self.by_label.get(MIRROR[label], [])]
        if not pool:
            return None
        i, mirrored = pool[int(rng.integers(0, len(pool)))]
        return self.clips[i], mirrored


_LIBRARIES: dict = {}


def load_library(path: str) -> MotionLibrary:
    """Load a library once per process (worker processes share nothing; the file is small)."""
    if path not in _LIBRARIES:
        _LIBRARIES[path] = MotionLibrary.load(path)
    return _LIBRARIES[path]


# --------------------------------------------------------------------------
# One library clip placed into a scene
# --------------------------------------------------------------------------
def place_motion(rng: np.random.Generator, t: np.ndarray, clip: dict, mirrored: bool, label: str, centre: np.ndarray,
                 t_centre: float, home: np.ndarray, y_max: float) -> dict:
    """A library clip as a hand path on the scene's control times `t`.

    centre   where the middle of the stroke should be (laptop frame)
    t_centre when the middle of the stroke happens (s, on the `t` axis)
    home     where the hand rests before and after
    Returns {"pos" [len(t), 3], "normal" [len(t), 3], "t_start", "t_end", "after", "travel_m", "peak_speed", "stretch"}.
    The SHAPE of the path is the real one (how the speed rises and falls, how the path curves, how the palm
    turns). Its size and tempo are set to the demo's gesture range, it is moved in front of the screen and,
    for the other swipe direction, mirrored.
    """
    from .physics import minjerk

    i0, i1 = clip["stroke"]
    fps = clip["fps"]
    # the stroke with the person's own wind-up before it and what they did right after (at most 0.6 s / 0.9 s)
    a, b = max(0, i0 - int(round(0.6 * fps))), min(len(clip["pos"]), i1 + int(round(0.9 * fps)) + 1)
    pos = np.asarray(clip["pos"][a:b], np.float64).copy()
    nrm = np.asarray(clip["normal"][a:b], np.float64).copy()
    i0, i1 = i0 - a, i1 - a
    if mirrored:
        pos[:, 0] *= -1.0
        nrm[:, 0] *= -1.0
    axis = 0 if label.startswith("swipe") else 1
    travel = abs(pos[i1, axis] - pos[i0, axis])
    if label == "other":
        scale, stretch = float(rng.uniform(0.85, 1.15)), float(np.exp(rng.uniform(-0.15, 0.15)))
    else:
        # Size and tempo of the demo's gestures (the same ranges as the formula strokes in scenes.py): public
        # clips are often smaller and slower (in-car style), so the real SHAPE of the stroke is kept and its
        # size and speed are brought into the range people use in front of the laptop.
        want_travel = rng.uniform(0.16, 0.56) if axis == 0 else rng.uniform(0.12, max(0.14, y_max - 0.19))
        from . import calibration

        v_lo, v_hi = calibration.speed_range((0.5, 2.2))                   # the fitted range, if a calibration is active
        want_speed = float(np.exp(rng.uniform(np.log(v_lo), np.log(v_hi))))
        scale = want_travel / max(travel, 1e-6)
        speed = max(float(clip.get("peak_speed", 0.0)) * scale, 1e-6)
        stroke_s = (i1 - i0) / fps
        fastest = 0.07 / max(stroke_s, 1e-6)
        if calibration.ACTIVE is not None:   # at most 2.5x the recorded tempo, and never faster than a hand can be accelerated
            fastest = max(0.4, calibration.min_stroke_seconds(rng, want_travel) / max(stroke_s, 1e-6))
        stretch = float(np.clip(speed / want_speed, fastest, 1.2 / max(stroke_s, 1e-6)))  # > 1 = slower
    mid = 0.5 * (pos[i0] + pos[i1])
    pos = (pos - mid) * scale + centre
    pos[:, 1] = np.maximum(pos[:, 1], 0.05)                              # never through the screen
    pos[:, 2] = np.maximum(pos[:, 2], 0.01)                              # ... or the desk
    tc = np.arange(len(pos)) / fps * stretch
    tc = tc - 0.5 * (tc[i0] + tc[i1]) + t_centre                         # clip time on the scene's time axis
    t_a, t_b = tc[0], tc[-1]
    spline = CubicSpline(tc, pos, axis=0)
    nspl = CubicSpline(tc, nrm, axis=0)
    out = np.tile(home.astype(np.float64), (len(t), 1))
    out_n = np.tile(nrm[0], (len(t), 1))
    inside = (t >= t_a) & (t <= t_b)
    out[inside] = spline(t[inside])
    out_n[inside] = nspl(t[inside])
    # before the clip: the hand comes up from where it rests (smooth, like kinematics.Path.move)
    d_in = float(np.linalg.norm(pos[0] - home))
    prep = float(np.clip(d_in / rng.uniform(0.35, 0.9), 0.3, 1.2))
    if label != "other":
        from . import calibration

        prep = calibration.lead_in(prep, 0.0)[0]                         # unchanged unless a calibration is active
    before = t < t_a
    s = minjerk((t[before] - (t_a - prep)) / prep)[:, None]
    out[before] = home + (pos[0] - home) * s
    # after the clip: the hand drops back (most people), or stays where the clip ended
    after = str(rng.choice(["return", "stay"], p=[0.7, 0.3]))
    later = t > t_b
    out[later] = pos[-1]
    out_n[later] = nrm[-1]
    if after == "return":
        gap = rng.uniform(0.0, 0.4)
        back = float(np.clip(float(np.linalg.norm(pos[-1] - home)) / rng.uniform(0.3, 0.8), 0.4, 1.4))
        s = minjerk((t[later] - (t_b + gap)) / back)[:, None]
        out[later] = pos[-1] + (home - pos[-1]) * s
    out_n /= np.maximum(np.linalg.norm(out_n, axis=1, keepdims=True), 1e-9)
    return {"pos": out, "normal": out_n, "t_start": float(tc[i0]), "t_end": float(tc[i1]), "after": after,
            "travel_m": float(travel * scale), "peak_speed": float(clip.get("peak_speed", 0.0) * scale / stretch),
            "stretch": stretch, "relax_from": float(t_b)}

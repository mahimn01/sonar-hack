"""camlabel.py — turn camera hand tracks into gesture labels for the sonar recordings.

Plain words: while the sonar records, the webcam watches the hand. Hand tracking
gives 21 points of the hand in every camera picture. This module reads those
points (never the pictures) and works out, without any training:

  1. where the palm was in space, in metres (the hand's known size tells how far
     away it is, so sideways travel comes out in real centimetres),
  2. every "stroke": one smooth movement of the hand in one direction, with its
     start, end, speed and distance,
  3. what each stroke was: swipe left / swipe right (sideways), push (towards the
     screen), pull (away, fast), or just "other hand movement",
  4. how sure it is (0..1). Unsure strokes are marked "review", never thrown away,
  5. how far the camera clock is off the audio clock, by sliding the camera's
     "hand speed" curve against the sonar's "movement" curve until they line up.

Directions are the USER's: "right" = the user's right hand side while facing the
screen. The recorder hands this module pictures that are already mirrored into
that view, so x grows towards the user's right.

Words: a "stroke" is one movement in one direction; a "track" is one hand
followed over time; "provisional time" is the camera time before the measured
clock offset is taken out.

Read-only helper module: no camera, no audio devices. Only `write_landmarks`
writes a file (the landmark track of a session).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

LANDMARK_FILE = "landmarks.jsonl"
LANDMARK_FORMAT = "sonar-hand-landmarks"
LANDMARK_VERSION = 1
PALM = (0, 5, 9, 13, 17)                 # wrist + the four finger knuckles: the stiff part of the hand
GESTURES = ("swipe_left", "swipe_right", "push", "pull")
OPPOSITE = {"swipe_left": "swipe_right", "swipe_right": "swipe_left", "push": "pull", "pull": "push"}
# The sonar's movement curve runs this far behind the true movement (filters + the 64 ms
# analysis window); measured with the simulator in tests/test_camlabel.py.
DSP_DELAY_S = 0.040


# ---------------------------------------------------------------------------
# What the camera saw
# ---------------------------------------------------------------------------
@dataclass
class HandObs:
    """One hand in one camera picture."""

    hand: str                          # "left" | "right" (the user's hand) | "unknown"
    score: float                       # how sure the tracker is (0..1)
    lm: np.ndarray                     # [21, 3] x, y in 0..1 of the picture (x towards the user's right, y down), z relative depth
    world: np.ndarray | None = None    # [21, 3] metres, the hand's shape about its own centre (same axes)


@dataclass
class CamFrame:
    """One camera picture, reduced to its hand points."""

    index: int
    t: float                           # seconds since the recorder started (monotonic clock)
    sample: float                      # audio sample index of this picture, BEFORE the measured clock offset is removed
    hands: list[HandObs] = field(default_factory=list)
    pose: np.ndarray | None = None     # optional [8, 4] upper-body joints x, y, z, visibility (no face points)


def frame_to_json(fr: CamFrame) -> str:
    row: dict[str, Any] = {"i": fr.index, "t": round(float(fr.t), 4), "sample": int(round(fr.sample)), "hands": []}
    for h in fr.hands:
        d: dict[str, Any] = {"hand": h.hand, "score": round(float(h.score), 3), "lm": np.round(h.lm, 4).tolist()}
        if h.world is not None:
            d["world"] = np.round(h.world, 4).tolist()
        row["hands"].append(d)
    if fr.pose is not None:
        row["pose"] = np.round(fr.pose, 4).tolist()
    return json.dumps(row, separators=(",", ":"))


def landmark_header(width: int, height: int, fs: int, **extra: Any) -> dict[str, Any]:
    """First line of landmarks.jsonl: how to read the rows that follow."""
    return {
        "format": LANDMARK_FORMAT, "version": LANDMARK_VERSION, "width": int(width), "height": int(height), "fs": int(fs),
        "x_axis": "towards the user's right (picture already mirrored into the user's view)", "y_axis": "down",
        "landmarks": "21 MediaPipe hand points: lm = picture position 0..1 + relative depth; world = metres about the hand centre",
        **extra,
    }


def write_landmarks(path: str | Path, header: dict[str, Any], frames: Iterable[CamFrame]) -> None:
    """Write a whole landmark track at once (the recorder streams the same lines itself)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for fr in frames:
            fh.write(frame_to_json(fr) + "\n")


def load_landmarks(path: str | Path) -> tuple[dict[str, Any], list[CamFrame]]:
    """(header, frames) of a landmarks.jsonl. A half-written last line (a crash) is ignored."""
    header: dict[str, Any] = {}
    frames: list[CamFrame] = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if n == 0 and "format" in row:
                header = row
                continue
            hands = [
                HandObs(h.get("hand", "unknown"), float(h.get("score", 0.0)), np.asarray(h["lm"], dtype=np.float64),
                        np.asarray(h["world"], dtype=np.float64) if "world" in h else None)
                for h in row.get("hands", [])
            ]
            pose = np.asarray(row["pose"], dtype=np.float64) if "pose" in row else None
            frames.append(CamFrame(int(row["i"]), float(row["t"]), float(row["sample"]), hands, pose))
    return header, frames


# ---------------------------------------------------------------------------
# Settings of the labeller (all distances in metres, speeds in metres per second)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelConfig:
    hfov_deg: float = 70.0             # the webcam's horizontal field of view; only the hand-to-camera DISTANCE depends on it
    grid_hz: float = 50.0              # tracks are resampled to the sonar's 50 frames per second
    max_gap_s: float = 0.45            # a hand that vanishes for less than this (motion blur) is still the same movement
    smooth_s: float = 0.035            # smoothing of the palm path before speeds are taken
    min_track_s: float = 0.12          # shorter tracks are tracker flicker
    stroke_min_speed: float = 0.30     # a stroke's top speed must reach this
    stroke_min_m: float = 0.05         # ... and it must travel at least this far
    swipe_min_m: float = 0.10          # sideways travel of a swipe
    swipe_min_speed: float = 0.45
    swipe_purity: float = 0.72         # share of the travel that must be sideways (0.72 = within about 44 degrees)
    push_min_m: float = 0.06           # travel towards the screen of a push
    push_min_speed: float = 0.25
    radial_purity: float = 0.75        # share of the travel that must be straight at the screen (a reach from the lap is not a push)
    pull_min_speed: float = 0.80       # a pull is a QUICK move away; slower ones are the hand going back
    max_stroke_s: float = 1.4          # anything longer is not one gesture stroke
    review_below: float = 0.5          # events with less confidence than this are marked for review
    clean_gesture: float = 0.8         # in a no-gesture part, strokes at least this clean are set aside instead of used as background
    hand_len_m: float = 0.095          # wrist to middle knuckle, used only when the tracker gives no metric hand shape
    sync_max_lag_s: float = 0.6        # the clock offset is searched within +- this
    sync_min_corr: float = 0.30        # below this the two curves do not match well enough to trust the offset


@dataclass
class Track:
    """One hand followed over time (irregular camera times)."""

    t: np.ndarray        # [n] seconds on the audio clock (provisional)
    pos: np.ndarray      # [n, 3] metres from the camera: right, away from the screen, up
    score: np.ndarray    # [n]
    edge: np.ndarray     # [n] 0..1, 1 = the hand touches the border of the picture
    hand: str = "unknown"


@dataclass
class Stroke:
    """One movement of the hand in one direction."""

    t0: float
    t1: float
    t_peak: float
    disp: np.ndarray     # [3] metres: right, away from the screen, up
    path_m: float
    peak_speed: float
    seen: float          # share of the stroke the camera really saw (the rest was bridged across a blur gap)
    hand: str
    score: float
    cut_start: bool = False    # the hand was already moving when the camera first saw it
    cut_end: bool = False      # the hand was still moving when the camera lost it
    edge: float = 0.0
    kind: str = "other"        # physical reading: swipe_left | swipe_right | push | away | other
    purity: float = 0.0

    @property
    def dist(self) -> float:
        return float(np.linalg.norm(self.disp))

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


@dataclass
class CamEvent:
    """One labelled movement. Times are seconds on the audio clock (clock offset already removed)."""

    label: str             # swipe_left | swipe_right | push | pull | return | other
    kind: str              # "gesture" (sure) | "review" (probably a gesture, check it) | "motion" (not a gesture)
    t0: float
    t1: float
    t_peak: float
    confidence: float
    speed_mps: float
    distance_m: float
    direction: list[float]     # metres: right, away from the screen, up
    hand: str
    seen: float
    intent: str | None = None  # what the on-screen prompt asked for at that moment, if anything
    candidate: str | None = None   # for "review": which gesture it probably is
    note: str = ""
    phys: str = "other"            # what the stroke looked like on its own: swipe_left | swipe_right | push | away | other

    def to_session_event(self, fs: int, n_samples: int | None = None, rep: int | None = None) -> dict[str, Any]:
        """The session.json row (same format as the prompted recorder, plus the camera's details)."""
        hi = n_samples if n_samples is not None else 2**62
        start = int(np.clip(round(self.t0 * fs), 0, hi))
        end = int(np.clip(round(self.t1 * fs), start + 1, max(hi, start + 1)))
        ev: dict[str, Any] = {
            # a "review" row must never be read as a gesture by older code, so its label is not a gesture name
            "label": self.label if self.kind != "review" else "review",
            "kind": self.kind, "start": start, "end": end, "source": "camera",
            "confidence": round(float(self.confidence), 3), "t_peak": int(round(self.t_peak * fs)),
            "speed_mps": round(float(self.speed_mps), 2), "distance_m": round(float(self.distance_m), 3),
            "direction": [round(float(v), 3) for v in self.direction], "hand": self.hand, "seen": round(float(self.seen), 2),
        }
        if self.kind == "gesture":
            ev["tight"] = True              # start/end bound the movement itself, not a prompt window
        if self.kind == "review":
            ev["candidate"] = self.candidate or self.label
            ev["exclude"] = True            # neither a gesture example nor a "none" example until somebody checks it
        if rep is not None:
            ev["rep"] = int(rep)
        if self.kind == "motion":
            ev["phys"] = self.phys          # "other" = not gesture-like at all (slow, short or diagonal)
        if self.intent:
            ev["intent"] = self.intent
        if self.note:
            ev["note"] = self.note
        return ev


# ---------------------------------------------------------------------------
# 1. Palm position in metres
# ---------------------------------------------------------------------------
def focal_px(width: int, hfov_deg: float) -> float:
    return 0.5 * width / np.tan(np.radians(hfov_deg) / 2.0)


def palm_state(obs: HandObs, width: int, height: int, cfg: LabelConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """(position [3] in metres from the camera, palm centre [2] in the picture, edge closeness 0..1).

    How the distance is found: the tracker also reports the hand's real shape in metres.
    Comparing the spread of the 21 points in the picture (pixels) with their real spread
    (metres) gives "pixels per metre" at the hand, and the camera's focal length turns that
    into a distance. Sideways and up/down positions need no focal length at all.
    """
    px = obs.lm[:, :2] * np.array([width, height], dtype=np.float64)
    if obs.world is not None and np.isfinite(obs.world).all():
        w = obs.world[:, :2]
        spread_m = float(np.sqrt(np.sum((w - w.mean(axis=0)) ** 2)))
        spread_px = float(np.sqrt(np.sum((px - px.mean(axis=0)) ** 2)))
        scale = spread_px / max(spread_m, 1e-6)                      # pixels per metre at the hand
    else:
        scale = float(np.linalg.norm(px[9] - px[0])) / cfg.hand_len_m
    scale = max(scale, 1e-3)
    centre = obs.lm[list(PALM), :2].mean(axis=0)
    c_px = centre * np.array([width, height]) - 0.5 * np.array([width, height])
    pos = np.array([c_px[0] / scale, focal_px(width, cfg.hfov_deg) / scale, -c_px[1] / scale])
    lo, hi = obs.lm[:, :2].min(axis=0), obs.lm[:, :2].max(axis=0)
    margin = float(min(lo.min(), (1.0 - hi).min()))                  # distance of the hand's box to the picture border
    edge = float(np.clip(1.0 - margin / 0.08, 0.0, 1.0))
    return pos, centre, edge


def build_tracks(frames: list[CamFrame], width: int, height: int, fs: int, cfg: LabelConfig | None = None) -> list[Track]:
    """Follow each hand through the pictures. Times are provisional audio seconds (sample / fs)."""
    cfg = cfg or LabelConfig()
    open_tracks: list[dict[str, Any]] = []
    done: list[dict[str, Any]] = []
    for fr in sorted(frames, key=lambda f: f.sample):
        t = fr.sample / fs
        still_open = []
        for tr in open_tracks:                                   # a hand unseen for too long is a finished track
            (still_open if t - tr["t"][-1] <= cfg.max_gap_s else done).append(tr)
        open_tracks = still_open
        dets = []
        for obs in fr.hands:
            if obs.lm.shape != (21, 3) or not np.isfinite(obs.lm).all():
                continue
            pos, centre, edge = palm_state(obs, width, height, cfg)
            dets.append((obs, pos, centre, edge))
        pairs = []
        for i, (obs, _, centre, _) in enumerate(dets):
            for j, tr in enumerate(open_tracks):
                dt = max(t - tr["t"][-1], 1e-3)
                dist = float(np.linalg.norm(centre - tr["c"]))
                alone = len(dets) == 1 and len(open_tracks) == 1     # one hand before, one hand now: the same hand
                if dist <= 0.12 + 2.5 * dt or alone:
                    pairs.append((dist + (0.15 if obs.hand != tr["hands"][-1] else 0.0), i, j))
        used_d: set[int] = set()
        used_t: set[int] = set()
        for _, i, j in sorted(pairs):
            if i in used_d or j in used_t:
                continue
            used_d.add(i)
            used_t.add(j)
            obs, pos, centre, edge = dets[i]
            tr = open_tracks[j]
            if t <= tr["t"][-1]:
                continue
            tr["t"].append(t); tr["pos"].append(pos); tr["score"].append(obs.score); tr["edge"].append(edge)
            tr["hands"].append(obs.hand); tr["c"] = centre
        for i, (obs, pos, centre, edge) in enumerate(dets):
            if i not in used_d:
                open_tracks.append({"t": [t], "pos": [pos], "score": [obs.score], "edge": [edge], "hands": [obs.hand], "c": centre})
    out = []
    for tr in done + open_tracks:
        if len(tr["t"]) < 3 or tr["t"][-1] - tr["t"][0] < cfg.min_track_s:
            continue
        names, counts = np.unique(tr["hands"], return_counts=True)
        out.append(Track(np.array(tr["t"]), np.array(tr["pos"]), np.array(tr["score"]), np.array(tr["edge"]),
                         str(names[int(np.argmax(counts))])))
    return sorted(out, key=lambda tr: tr.t[0])


# ---------------------------------------------------------------------------
# 2. Strokes
# ---------------------------------------------------------------------------
@dataclass
class GridTrack:
    """A track resampled to the sonar frame rate. Grid point k is at time k / hz."""

    k0: int
    pos: np.ndarray      # [T, 3] smoothed
    vel: np.ndarray      # [T, 3] metres per second
    seen: np.ndarray     # [T] bool: the camera really saw the hand around this moment
    edge: np.ndarray     # [T]
    score: np.ndarray    # [T]
    hand: str
    hz: float

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.vel, axis=1)

    def time(self, k: int | np.ndarray) -> Any:
        return (self.k0 + k) / self.hz


def resample(track: Track, cfg: LabelConfig) -> GridTrack | None:
    hz = cfg.grid_hz
    k0, k1 = int(np.ceil(track.t[0] * hz)), int(np.floor(track.t[-1] * hz))
    if k1 - k0 < 4:
        return None
    tg = np.arange(k0, k1 + 1) / hz
    pos = np.stack([np.interp(tg, track.t, track.pos[:, a]) for a in range(3)], axis=1)
    # "seen": the grid point lies between two pictures taken at the camera's normal pace
    dt = np.diff(track.t)
    normal = 1.8 * float(np.median(dt)) + 1e-3
    j = np.clip(np.searchsorted(track.t, tg, side="right") - 1, 0, len(dt) - 1)
    seen = dt[j] <= max(normal, 0.07)
    sigma = cfg.smooth_s * hz
    sm = gaussian_filter1d(pos, sigma=sigma, axis=0, mode="nearest")
    sm[:, 1] = gaussian_filter1d(pos[:, 1], sigma=1.6 * sigma, mode="nearest")     # the depth reading is the noisy one
    vel = np.gradient(sm, 1.0 / hz, axis=0)
    edge = np.interp(tg, track.t, track.edge)
    score = np.interp(tg, track.t, track.score)
    return GridTrack(k0, sm, vel, seen, edge, score, track.hand, hz)


def find_strokes(g: GridTrack, cfg: LabelConfig | None = None) -> list[Stroke]:
    """Every movement of one hand in one direction: around each speed peak, as long as the hand keeps going that way."""
    cfg = cfg or LabelConfig()
    speed = g.speed
    n = len(speed)
    peaks = list(find_peaks(speed, height=cfg.stroke_min_speed, distance=max(1, int(0.10 * g.hz)), prominence=0.08)[0])
    # a movement cut off by the edge of the track has no peak inside it: its fastest point is the end itself
    if speed[-1] >= cfg.stroke_min_speed and speed[-1] >= speed[max(n - 4, 0):].max() - 1e-9:
        peaks.append(n - 1)
    if speed[0] >= cfg.stroke_min_speed and speed[0] >= speed[:4].max() - 1e-9:
        peaks.append(0)
    spans: list[tuple[int, int, int]] = []
    for k in sorted(set(peaks), key=lambda i: -speed[i]):
        if any(a <= k <= b for a, b, _ in spans):            # already inside a stronger stroke
            continue
        u = g.vel[k] / max(speed[k], 1e-9)
        along = g.vel @ u                                    # speed along this stroke's own direction
        thr = max(0.22 * speed[k], 0.08)
        a = k
        while a > 0 and along[a - 1] > thr and not any(x <= a - 1 <= y for x, y, _ in spans):
            a -= 1
        b = k
        while b < n - 1 and along[b + 1] > thr and not any(x <= b + 1 <= y for x, y, _ in spans):
            b += 1
        spans.append((a, b, k))
    out = []
    for a, b, k in sorted(spans):
        if b - a < 2:
            continue
        disp = g.pos[b] - g.pos[a]
        path = float(np.sum(np.linalg.norm(np.diff(g.pos[a:b + 1], axis=0), axis=1)))
        thr = max(0.22 * speed[k], 0.08)
        st = Stroke(
            t0=float(g.time(a)), t1=float(g.time(b)), t_peak=float(g.time(k)), disp=disp, path_m=path,
            peak_speed=float(speed[k]), seen=float(np.mean(g.seen[a:b + 1])), hand=g.hand,
            score=float(np.mean(g.score[a:b + 1])), cut_start=bool(a == 0 and speed[0] > thr),
            cut_end=bool(b == n - 1 and speed[-1] > thr), edge=float(np.max(g.edge[a:b + 1])),
        )
        if st.dist < cfg.stroke_min_m * (0.5 if (st.cut_start or st.cut_end) else 1.0) or st.duration > cfg.max_stroke_s:
            continue
        classify_stroke(st, cfg)
        out.append(st)
    return out


def classify_stroke(st: Stroke, cfg: LabelConfig) -> None:
    """Fill in the stroke's physical reading (which way the hand went) from its travel in metres."""
    d = st.disp
    total = max(st.dist, 1e-9)
    side, depth = abs(d[0]) / total, abs(d[1]) / total
    cut = st.cut_start or st.cut_end                 # the camera saw only part of it: accept a shorter travel
    k = 0.6 if cut else 1.0
    st.kind, st.purity = "other", max(side, depth)
    if side >= cfg.swipe_purity and abs(d[0]) >= k * cfg.swipe_min_m and st.peak_speed >= cfg.swipe_min_speed:
        st.kind, st.purity = ("swipe_right" if d[0] > 0 else "swipe_left"), side
    elif depth >= cfg.radial_purity and abs(d[1]) >= k * cfg.push_min_m and st.peak_speed >= cfg.push_min_speed:
        st.kind, st.purity = ("push" if d[1] < 0 else "away"), depth


def all_strokes(tracks: list[Track], cfg: LabelConfig | None = None) -> tuple[list[Stroke], list[GridTrack]]:
    cfg = cfg or LabelConfig()
    grids = [g for g in (resample(tr, cfg) for tr in tracks) if g is not None]
    strokes = [s for g in grids for s in find_strokes(g, cfg)]
    return sorted(strokes, key=lambda s: s.t_peak), grids


def speed_curve(grids: list[GridTrack], n: int, hz: float = 50.0) -> np.ndarray:
    """Fastest hand speed (m/s) at every grid point 0..n-1; 0 where the camera saw no hand."""
    out = np.zeros(n)
    for g in grids:
        a, b = max(g.k0, 0), min(g.k0 + len(g.pos), n)
        if b > a:
            out[a:b] = np.maximum(out[a:b], g.speed[a - g.k0:b - g.k0])
    return out


# ---------------------------------------------------------------------------
# 3. Strokes -> labelled events
# ---------------------------------------------------------------------------
@dataclass
class Intent:
    """What the screen asked the volunteer to do between two moments (provisional audio seconds)."""

    t0: float
    t1: float
    label: str             # a gesture name, or "none" for a block in which no gesture should happen


def _base_confidence(st: Stroke, cfg: LabelConfig) -> float:
    need_m = cfg.swipe_min_m if st.kind.startswith("swipe") else cfg.push_min_m
    need_v = cfg.swipe_min_speed if st.kind.startswith("swipe") else cfg.push_min_speed
    need_p = cfg.swipe_purity if st.kind.startswith("swipe") else cfg.radial_purity
    axis = abs(st.disp[0]) if st.kind.startswith("swipe") else abs(st.disp[1])
    c_seen = 0.35 + 0.65 * float(np.clip(st.seen, 0.0, 1.0))
    c_dir = 0.65 + 0.35 * float(np.clip((st.purity - need_p) / max(0.93 - need_p, 1e-6), 0.0, 1.0))
    c_mag = 0.65 + 0.35 * float(np.clip(axis / need_m - 1.0, 0.0, 1.0))
    c_speed = 0.75 + 0.25 * float(np.clip(st.peak_speed / need_v - 1.0, 0.0, 1.0))
    c_track = 0.8 + 0.2 * float(np.clip(st.score, 0.0, 1.0))
    conf = c_seen * c_dir * c_mag * c_speed * c_track
    if st.cut_start or st.cut_end:
        conf *= 0.8
    return float(conf)


def label_strokes(strokes: list[Stroke], intents: list[Intent] | None = None, cfg: LabelConfig | None = None,
                  offset_s: float = 0.0) -> list[CamEvent]:
    """Turn strokes into labelled events. `offset_s` (camera clock minus audio clock) is removed from the times.

    With a prompt ("intent") at that moment: a stroke that matches the prompt is the gesture, the
    opposite stroke next to it is the hand going back, anything else is another movement.
    Without a prompt the camera decides alone: of two opposite strokes close together the clearly
    faster one is the gesture and the other the wind-up or the return; if neither is clearly
    faster both are marked for review.
    """
    cfg = cfg or LabelConfig()
    intents = intents or []
    strokes = sorted(strokes, key=lambda s: s.t_peak)

    def intent_at(t: float) -> str | None:
        for it in intents:
            if it.t0 - 0.2 <= t <= it.t1 + 0.3:
                return it.label
        return None

    events: list[CamEvent] = []
    for i, st in enumerate(strokes):
        want = intent_at(st.t_peak)
        label, kind, conf, note, cand = "other", "motion", 0.0, "", None
        near = [o for o in strokes[max(0, i - 3):i + 4] if o is not st and abs(o.t_peak - st.t_peak) <= 1.2]
        if st.kind in ("swipe_left", "swipe_right", "push", "away"):
            conf = _base_confidence(st, cfg)
            phys = st.kind
            if phys == "away":
                # A move away from the screen is a pull only if it is quick AND not simply the hand going
                # back after a push (an approach just before it that was about as fast or faster).
                came = [o for o in near if o.kind == "push" and 0.0 < st.t_peak - o.t_peak <= 1.5]
                reach_was_slow = all(1.5 * o.peak_speed <= st.peak_speed for o in came)
                phys = "pull" if (st.peak_speed >= cfg.pull_min_speed and reach_was_slow) else "return"
            elif phys == "push":
                # ... and an approach followed by a much quicker retreat was the slow reach before a pull
                went = [o for o in near if o.kind == "away" and 0.0 < o.t_peak - st.t_peak <= 1.5]
                if any(o.peak_speed >= max(1.5 * st.peak_speed, cfg.pull_min_speed) for o in went):
                    phys = "reach"
            if want in GESTURES:
                if phys == want:
                    label, kind = want, "gesture"
                    conf = 1.0 - 0.5 * (1.0 - conf)             # the prompt agrees: halve the doubt
                elif phys in ("return", "reach", OPPOSITE[want]):
                    label, note = "return", f"opposite of the prompted {want}: the hand winding up or going back"
                else:
                    label, kind, cand = phys, "review", phys
                    conf, note = min(conf, 0.45), f"looked like {phys} while the prompt asked for {want}"
            elif want == "none":
                if phys in GESTURES and conf >= cfg.clean_gesture:
                    # a quick, straight, well-seen stroke: probably a real gesture done at the wrong moment
                    label, kind, cand = phys, "review", phys
                    conf, note = min(conf, 0.45), f"looked like a clean {phys} in a part where no gesture was asked for"
                elif phys in ("return", "reach"):
                    label = "return"
                elif phys in GESTURES:
                    # an unhurried reach that only resembles a gesture: exactly the background the model must learn to ignore
                    note = f"resembles {phys} but is slow or crooked: kept as background"
            elif phys in ("return", "reach"):                    # no prompt: the camera decides alone
                label = "return"
            else:
                label, kind = phys, "gesture"
                if phys.startswith("swipe"):
                    rivals = [o for o in near if o.kind == OPPOSITE[phys]]
                    if any(o.peak_speed >= 1.25 * st.peak_speed for o in rivals):
                        label, kind, note = "return", "motion", "a clearly faster opposite stroke is next to it"
                    elif any(o.peak_speed >= 0.8 * st.peak_speed for o in rivals):
                        conf, note = min(conf, 0.45), "an opposite stroke of similar speed is next to it"
            if kind == "gesture" and conf < cfg.review_below:
                kind, cand = "review", label
                note = note or "the camera saw too little of it"
        events.append(CamEvent(
            label=label, kind=kind, t0=st.t0 - offset_s, t1=st.t1 - offset_s, t_peak=st.t_peak - offset_s,
            confidence=round(conf, 3) if kind != "motion" else 0.0, speed_mps=st.peak_speed, distance_m=st.dist,
            direction=[float(v) for v in st.disp], hand=st.hand, seen=st.seen, intent=want, candidate=cand, note=note,
            phys=st.kind,
        ))
    return events


# ---------------------------------------------------------------------------
# 4. Clock offset between camera and audio
# ---------------------------------------------------------------------------
@dataclass
class SyncResult:
    offset_s: float            # camera time minus audio time (positive: camera pictures are stamped late)
    method: str                # "motion" (measured from the two movement curves) | "timestamps" (not measured)
    peak_corr: float = 0.0     # how well the two curves match at the best offset (1 = perfectly)
    residual_s: float | None = None   # spread of the offset from stroke to stroke after correction
    n_strokes: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method, "offset_ms": round(1e3 * self.offset_s, 1), "peak_corr": round(self.peak_corr, 3),
            "residual_ms": None if self.residual_s is None else round(1e3 * self.residual_s, 1),
            "n_strokes": self.n_strokes, "note": self.note,
        }


def _xcorr_lag(cam: np.ndarray, son: np.ndarray, max_lag: int) -> tuple[float, float, np.ndarray]:
    """(best lag in samples with a parabola refinement, correlation there, the whole curve)."""
    a, b = cam - cam.mean(), son - son.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9 or len(a) <= 2 * max_lag + 2:
        return 0.0, 0.0, np.zeros(2 * max_lag + 1)
    r = np.array([float(np.dot(a[max(lag, 0):len(a) + min(lag, 0)], b[max(-lag, 0):len(b) + min(-lag, 0)]))
                  for lag in range(-max_lag, max_lag + 1)]) / (na * nb)
    k = int(np.argmax(r))
    frac = 0.0
    if 0 < k < len(r) - 1:
        den = r[k - 1] - 2 * r[k] + r[k + 1]
        if den < -1e-12:
            frac = float(np.clip(0.5 * (r[k - 1] - r[k + 1]) / den, -0.5, 0.5))
    return k - max_lag + frac, float(r[k]), r


def estimate_offset(cam_speed: np.ndarray, activity_db: np.ndarray, strokes: list[Stroke] | None = None,
                    cfg: LabelConfig | None = None, prior_s: float = 0.0, hz: float = 50.0) -> SyncResult:
    """Clock offset from the camera's hand-speed curve and the sonar's movement curve (both on the 50 Hz frame grid).

    The hand moving is the one thing both instruments see. The camera curve is slid against
    the sonar curve; the offset with the best match is the clock error. The sonar curve is
    known to run DSP_DELAY_S behind the real movement, which is taken out.
    Returns method "timestamps" (offset = `prior_s`) when the curves do not match well enough.
    """
    cfg = cfg or LabelConfig()
    n = min(len(cam_speed), len(activity_db))
    up = 4                                                        # 5 ms steps
    grid = np.arange(n * up) / up
    cam = np.interp(grid, np.arange(n), np.asarray(cam_speed[:n], dtype=np.float64))
    son = np.interp(grid, np.arange(n), np.clip(np.asarray(activity_db[:n], dtype=np.float64), 0.0, 30.0))
    max_lag = int(cfg.sync_max_lag_s * hz * up)
    fail = SyncResult(prior_s, "timestamps", note="camera and sonar movement curves do not match well enough; offset not measured")
    if n < 4 * hz or cam.max() < cfg.stroke_min_speed:
        fail.note = "the camera saw no hand movement; offset not measured"
        return fail
    lag, corr, r = _xcorr_lag(cam, son, max_lag)
    if corr < cfg.sync_min_corr:
        fail.peak_corr = corr
        return fail
    offset = lag / (hz * up) + DSP_DELAY_S
    # stroke by stroke: does each movement agree with the session's offset?
    residual = None
    lags = []
    for st in strokes or []:
        if st.kind == "other" and st.peak_speed < 2 * cfg.stroke_min_speed:
            continue
        c = int(st.t_peak * hz * up)
        a, b = max(c - int(0.9 * hz * up), 0), min(c + int(0.9 * hz * up), len(cam))
        if b - a <= 2 * max_lag // 2 + 8 or son[a:b].max() < 1.5:
            continue
        local, lc, _ = _xcorr_lag(cam[a:b], son[a:b], max_lag // 2)
        if lc >= 0.4:
            lags.append(local / (hz * up) + DSP_DELAY_S)
    if len(lags) >= 3:
        residual = float(1.4826 * np.median(np.abs(np.array(lags) - np.median(lags))))
    return SyncResult(float(offset), "motion", corr, residual, len(lags))


# ---------------------------------------------------------------------------
# 5. One call for a whole session
# ---------------------------------------------------------------------------
@dataclass
class LabelResult:
    events: list[CamEvent]
    sync: SyncResult
    strokes: list[Stroke]
    n_frames: int
    hand_seen_s: float
    config: LabelConfig

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.events:
            key = e.label if e.kind == "gesture" else ("review" if e.kind == "review" else "other movement")
            out[key] = out.get(key, 0) + 1
        return out


def label_session(frames: list[CamFrame], header: dict[str, Any], activity_db: np.ndarray | None,
                  intents: list[Intent] | None = None, cfg: LabelConfig | None = None, prior_offset_s: float = 0.0) -> LabelResult:
    """Landmark frames (+ the sonar's movement curve, + what the prompts asked for) -> labelled events.

    activity_db: the sonar's movement level per 20 ms frame (StreamFeatures.activity_db), or None
    if there is no audio to line up with (then the camera timestamps are used as they are).
    """
    cfg = cfg or LabelConfig()
    if "hfov_deg" in header and cfg.hfov_deg != float(header["hfov_deg"]):
        cfg = replace(cfg, hfov_deg=float(header["hfov_deg"]))
    width, height, fs = int(header.get("width", 640)), int(header.get("height", 480)), int(header.get("fs", 48000))
    tracks = build_tracks(frames, width, height, fs, cfg)
    strokes, grids = all_strokes(tracks, cfg)
    if activity_db is not None and len(activity_db):
        sync = estimate_offset(speed_curve(grids, len(activity_db), cfg.grid_hz), activity_db, strokes, cfg, prior_offset_s, cfg.grid_hz)
    else:
        sync = SyncResult(prior_offset_s, "timestamps", note="no sonar movement curve given; offset not measured")
    events = label_strokes(strokes, intents, cfg, offset_s=sync.offset_s)
    seen_s = float(sum(tr.t[-1] - tr.t[0] for tr in tracks))
    return LabelResult(events, sync, strokes, len(frames), seen_s, cfg)


def label_config_dict(cfg: LabelConfig) -> dict[str, Any]:
    return asdict(cfg)


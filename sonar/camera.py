"""camera.py — where the hand points come from: the webcam, or a stand-in for tests.

Plain words: the camera-labelled recorder (camrecord.py) needs, for every camera
picture, the 21 points of each visible hand. This module provides that from

  OpenCVCamera + MediaPipeHands   the real thing: the laptop webcam read with
                                  OpenCV, hand tracking by Google's MediaPipe.
  FakeCamera                      no camera at all: it "sees" the simulated hand
                                  of the sonar simulator and draws the 21 points
                                  with a pinhole-camera model. Used for the tests
                                  and for rehearsing a session without hardware.

Privacy: pictures never leave this module. The webcam frame is handed to the
tracker and to the on-screen preview and is then forgotten; only the hand
points are passed on. No face points are ever produced (the optional body
tracker keeps shoulders, elbows, wrists and hips only).

Mirroring: every picture is turned into the USER's view before tracking, so
"x grows to the right" means the user's right hand side, exactly like looking
into a mirror. An ordinary webcam facing the user delivers the opposite, so its
picture is flipped once here.

What it changes: opens the webcam while running. The first use downloads the
hand-tracking model (7.8 MB, Apache-2.0, from Google's MediaPipe model store)
into models/mediapipe/. Nothing else is written.
"""

from __future__ import annotations

import hashlib
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .camlabel import HandObs

REPO = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO / "models" / "mediapipe"
MODELS = {
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    ),
    "pose": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
}
POSE_JOINTS = (11, 12, 13, 14, 15, 16, 23, 24)       # shoulders, elbows, wrists, hips: no face points
HAND_LINKS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
              (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))


class CameraUnavailable(RuntimeError):
    """No usable camera or hand tracker; the recorder falls back to prompted labels."""


@dataclass
class CameraFrame:
    """One picture from a camera source."""

    t: float | None = None                 # time.monotonic() when the picture arrived (real cameras)
    sample: float | None = None            # audio sample index, for sources that run on the audio clock (FakeCamera)
    image: np.ndarray | None = None        # BGR picture in the user's view, or None (FakeCamera)
    hands: list[HandObs] | None = None     # already tracked (FakeCamera), or None: run the tracker on `image`


# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------
def ensure_model(kind: str = "hand", folder: Path | None = None, download: bool = True) -> Path:
    """Path of the MediaPipe model file, downloading it once if needed."""
    name, url, sha = MODELS[kind]
    folder = MODEL_DIR if folder is None else Path(folder)
    path = folder / name
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    if not download:
        raise CameraUnavailable(f"{path} is missing")
    folder.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"downloading the {kind}-tracking model once ({url}) ...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, open(tmp, "wb") as fh:
            fh.write(resp.read())
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise CameraUnavailable(
            f"could not download the {kind}-tracking model ({type(exc).__name__}: {exc}).\n"
            f"  On a machine with internet, download {url}\n  and put the file at {path}"
        ) from exc
    got = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if got != sha:
        print(f"note: {name} differs from the copy this code was tested with (sha256 {got[:12]}...); using it anyway")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Real camera
# ---------------------------------------------------------------------------
class OpenCVCamera:
    """The webcam, read on its own thread so every picture is time-stamped the moment it arrives.

    Reading on a thread matters for the clock: if pictures waited in the driver's queue
    while the main loop was busy, their time stamps would be late by a varying amount.
    Only the newest picture is kept; older ones that were never looked at are dropped.
    """

    name = "opencv"

    def __init__(self, index: int | str = 0, width: int = 640, height: int = 480, fps: float = 30.0, api: str = "auto",
                 facing: str = "user", mirrored_input: bool = False) -> None:
        self.index = int(index) if str(index).isdigit() else index
        self.want = (int(width), int(height), float(fps))
        self.api = api
        # an ordinary webcam facing the user shows the user's right hand on the LEFT of its picture
        self.flip = (facing == "user") != bool(mirrored_input)
        self.facing, self.mirrored_input = facing, bool(mirrored_input)
        self.width, self.height = int(width), int(height)
        self._cap: Any = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._running = False
        self.n_frames = 0
        self.n_dropped = 0
        self.info: dict[str, Any] = {}

    def start(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise CameraUnavailable("OpenCV is not installed: pip install -r requirements-camera.txt") from exc
        apis = {"auto": [], "dshow": ["CAP_DSHOW"], "msmf": ["CAP_MSMF"], "avfoundation": ["CAP_AVFOUNDATION"], "any": ["CAP_ANY"]}[self.api]
        if not apis:        # DirectShow opens in a second on Windows where Media Foundation can take ten
            apis = ["CAP_DSHOW", "CAP_MSMF", "CAP_ANY"] if sys.platform == "win32" else (
                ["CAP_AVFOUNDATION", "CAP_ANY"] if sys.platform == "darwin" else ["CAP_ANY"])
        errors = []
        for name in apis:
            cap = cv2.VideoCapture(self.index, getattr(cv2, name))
            if cap is not None and cap.isOpened():
                w, h, fps = self.want
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                good, frame = False, None
                t_end = time.monotonic() + 4.0
                while time.monotonic() < t_end and not good:    # the first pictures can be missing or black while the camera wakes up
                    ok, frame = cap.read()
                    good = bool(ok and frame is not None and frame.size and float(frame.max()) > 0)
                if good:
                    self._cap = cap
                    self.height, self.width = int(frame.shape[0]), int(frame.shape[1])
                    self.info = {"backend": "opencv", "api": name, "index": self.index, "width": self.width, "height": self.height,
                                 "fps_asked": fps, "fps_reported": float(cap.get(cv2.CAP_PROP_FPS)), "facing": self.facing,
                                 "mirrored_input": self.mirrored_input, "flipped_to_user_view": self.flip}
                    break
                errors.append(f"{name}: opened but delivered no picture (camera permission denied, or the camera is in use?)")
                cap.release()
            else:
                errors.append(f"{name}: could not open camera {self.index!r}")
        if self._cap is None:
            hint = ""
            if sys.platform == "darwin":
                hint = "\n  macOS: allow the Terminal (or your IDE) under System Settings > Privacy & Security > Camera, then run again."
            raise CameraUnavailable("no camera picture:\n  " + "\n  ".join(errors) + hint)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="sonar-camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        cap = self._cap
        fails = 0
        while self._running:
            ok = cap.grab()
            t = time.monotonic()                     # stamp at arrival, before the (slower) decoding
            if not ok:
                fails += 1
                if fails > 150:
                    break
                time.sleep(0.01)
                continue
            fails = 0
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            with self._lock:
                if self._latest is not None:
                    self.n_dropped += 1
                self._latest = (frame, t)
                self.n_frames += 1
        self._running = False

    def read(self) -> CameraFrame | None:
        """The newest picture that has not been handed out yet (in the user's view), or None."""
        with self._lock:
            latest, self._latest = self._latest, None
        if latest is None:
            return None
        frame, t = latest
        if self.flip:
            frame = np.ascontiguousarray(frame[:, ::-1])
        return CameraFrame(t=t, image=frame)

    @property
    def alive(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# ---------------------------------------------------------------------------
# Hand tracking
# ---------------------------------------------------------------------------
class MediaPipeHands:
    """MediaPipe hand landmarks (and, if asked, a few body joints) for pictures in the user's view."""

    name = "mediapipe"

    def __init__(self, num_hands: int = 2, pose: bool = False, model_dir: Path | None = None) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise CameraUnavailable("MediaPipe is not installed: pip install -r requirements-camera.txt") from exc
        self._mp = mp
        vision = mp.tasks.vision
        try:
            self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model("hand", model_dir))),
                running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
                min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4, min_tracking_confidence=0.4,
            ))
            self._pose = None
            if pose:
                self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model("pose", model_dir))),
                    running_mode=vision.RunningMode.VIDEO, num_poses=1,
                ))
        except CameraUnavailable:
            raise
        except Exception as exc:
            raise CameraUnavailable(f"MediaPipe could not start ({type(exc).__name__}: {exc})") from exc
        self._last_ms = -1
        self.info = {"tracker": "mediapipe", "version": getattr(mp, "__version__", "?"), "num_hands": num_hands, "pose": bool(pose),
                     "handedness": "MediaPipe names hands as seen in a mirror; pictures are mirrored first, so the name is the user's hand"}

    def process(self, image_bgr: np.ndarray, t: float) -> tuple[list[HandObs], np.ndarray | None]:
        """Hands (and body joints) in one BGR picture that is already in the user's view. `t` in seconds, increasing."""
        mp = self._mp
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ms = max(int(t * 1000.0), self._last_ms + 1)              # MediaPipe wants strictly increasing milliseconds
        self._last_ms = ms
        res = self._hands.detect_for_video(frame, ms)
        hands = []
        for k, pts in enumerate(res.hand_landmarks):
            lm = np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float64)
            world = None
            if k < len(res.hand_world_landmarks):
                world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[k]], dtype=np.float64)
            name, score = "unknown", 0.0
            if k < len(res.handedness) and res.handedness[k]:
                name, score = str(res.handedness[k][0].category_name).lower(), float(res.handedness[k][0].score)
            hands.append(HandObs(name, score, lm, world))
        pose = None
        if self._pose is not None:
            pr = self._pose.detect_for_video(frame, ms)
            if pr.pose_landmarks:
                pts = pr.pose_landmarks[0]
                pose = np.array([[pts[j].x, pts[j].y, pts[j].z, getattr(pts[j], "visibility", 0.0) or 0.0] for j in POSE_JOINTS])
        return hands, pose

    def close(self) -> None:
        for obj in (self._hands, self._pose):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Fake camera (simulator)
# ---------------------------------------------------------------------------
# An open right hand, palm towards the screen, fingers up. Centimetres: (towards the user's
# right, away from the screen, up), wrist at the origin. Rough adult proportions.
_HAND_CM = np.array([
    [0.0, 0.0, 0.0],
    [-2.5, -0.5, 2.0], [-4.5, -1.0, 4.5], [-6.0, -1.2, 7.0], [-7.0, -1.3, 9.0],          # thumb
    [-2.5, 0.0, 9.0], [-2.8, -0.2, 12.5], [-3.0, -0.3, 15.0], [-3.1, -0.4, 17.0],        # index
    [0.0, 0.0, 9.5], [0.0, -0.2, 13.5], [0.0, -0.3, 16.3], [0.0, -0.4, 18.5],            # middle
    [2.2, 0.0, 9.0], [2.5, -0.2, 12.6], [2.7, -0.3, 15.2], [2.8, -0.4, 17.2],            # ring
    [4.2, 0.0, 8.0], [4.9, -0.2, 10.6], [5.3, -0.3, 12.4], [5.5, -0.4, 14.0],            # little finger
]) / 100.0
_PALM = [0, 5, 9, 13, 17]


def hand_points(palm: np.ndarray, hand: str = "right", size: float = 1.0) -> np.ndarray:
    """The 21 points [21, 3] (metres, laptop frame) of an open hand whose palm centre is at `palm`."""
    shape = _HAND_CM * size
    if hand == "left":
        shape = shape * np.array([-1.0, 1.0, 1.0])
    return np.asarray(palm, dtype=np.float64)[None, :] + shape - shape[_PALM].mean(axis=0)


@dataclass
class PinholeCamera:
    """A webcam at the top of the screen, looking at the user. Positions in the laptop frame (metres):
    x to the user's right, y from the screen towards the user, z up, origin at the middle of the hinge."""

    pos: tuple[float, float, float] = (0.0, -0.07, 0.22)
    tilt_deg: float = 0.0                # lid leaning back: the camera looks this much upwards
    hfov_deg: float = 70.0
    width: int = 640
    height: int = 480

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Laptop-frame points [21, 3] -> (lm [21, 3] in the user's view, world [21, 3] metres, share of points in the picture)."""
        th = np.radians(self.tilt_deg)
        fwd = np.array([0.0, np.cos(th), np.sin(th)])
        up = np.array([0.0, -np.sin(th), np.cos(th)])
        d = pts - np.asarray(self.pos)[None, :]
        x, depth, y_up = d[:, 0], d @ fwd, d @ up
        depth = np.maximum(depth, 1e-3)
        f = 0.5 * self.width / np.tan(np.radians(self.hfov_deg) / 2.0)
        lm = np.stack([0.5 + f * x / (depth * self.width), 0.5 - f * y_up / (depth * self.height),
                       (depth - depth[0]) * f / (depth[0] * self.width)], axis=1)
        world = np.stack([x - x.mean(), -(y_up - y_up.mean()), depth - depth.mean()], axis=1)
        inside = float(np.mean((lm[:, 0] >= 0) & (lm[:, 0] <= 1) & (lm[:, 1] >= 0) & (lm[:, 1] <= 1)))
        return lm, world, inside


@dataclass
class FakeCamera:
    """Pretends to be a webcam + hand tracker watching the sonar simulator's hand.

    path        function: times [n] (audio seconds) -> hand positions [n, 3] in the laptop frame
    clock       function: audio seconds simulated so far (pictures are only made for the past)
    latency_s   how late the fake camera stamps its pictures on the audio clock (what the
                recorder's clock sync has to find out)
    The hand is "not detected" when too little of it is inside the picture, and (if blur_speed
    is set) often while it moves faster than that, like a real tracker on a blurred picture.
    """

    path: Callable[[np.ndarray], np.ndarray]
    clock: Callable[[], float]
    fs: int = 48000
    fps: float = 30.0
    latency_s: float = 0.09
    jitter_s: float = 0.003
    noise: float = 0.0015                  # tracker jitter, as a share of the picture width
    blur_speed: float | None = None        # m/s above which detections start to drop out
    min_inside: float = 0.72
    hand: str = "right"
    seed: int = 0
    model: PinholeCamera = field(default_factory=PinholeCamera)
    name: str = "fake"

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._k = 0
        self.width, self.height = self.model.width, self.model.height
        self.n_frames = 0
        self.n_dropped = 0
        self.alive = True
        self.info = {"backend": "fake", "width": self.width, "height": self.height, "fps_asked": self.fps,
                     "latency_s": self.latency_s, "facing": "user", "mirrored_input": False, "flipped_to_user_view": True}

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.alive = False

    def read(self) -> CameraFrame | None:
        tau = self._k / self.fps
        if tau > self.clock() - 0.02:
            return None
        self._k += 1
        self.n_frames += 1
        tt = np.array([tau - 0.01, tau, tau + 0.01])
        p = self.path(tt)
        speed = float(np.linalg.norm(p[2] - p[0]) / 0.02)
        lm, world, inside = self.model.project(hand_points(p[1], self.hand))
        hands: list[HandObs] = []
        blurred = self.blur_speed is not None and self._rng.random() < np.clip((speed - self.blur_speed) / self.blur_speed, 0.0, 0.85)
        if inside >= self.min_inside and not blurred:
            lm = lm + self._rng.normal(0.0, self.noise, size=lm.shape) * np.array([1.0, self.width / self.height, 1.0])
            world = world + self._rng.normal(0.0, 0.0015, size=world.shape)
            hands.append(HandObs(self.hand, 0.95, lm, world))
        stamp = tau + self.latency_s + float(self._rng.normal(0.0, self.jitter_s))
        return CameraFrame(sample=stamp * self.fs, hands=hands)


class NullTracker:
    """Used with sources that already deliver hand points (FakeCamera)."""

    name = "none"
    info: dict[str, Any] = {"tracker": "none"}

    def process(self, image_bgr: np.ndarray, t: float) -> tuple[list[HandObs], np.ndarray | None]:
        return [], None

    def close(self) -> None:
        pass

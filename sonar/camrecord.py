"""camrecord.py — record the sonar while the webcam labels the gestures by itself.

Plain words: this is sonar/record.py with eyes. It plays the sonar signal and
records the microphones exactly like record.py (same audio.wav + session.json,
same devices, profile and block-size options), and at the same time the webcam
watches the volunteer's hand. Hand tracking turns every camera picture into 21
hand points; from those points this program works out when each swipe or push
really happened, which way it went, how fast and how far, and writes that into
session.json as the labels the trainer already understands. So a volunteer can
just do gestures for a minute at their own pace: no "get ready ... GO" timing.

PRIVACY: no picture and no face is ever saved. Only the hand points go to disk
(landmarks.jsonl). `--save-video` (debugging only) is the single exception and
is recorded in session.json. A consent screen is shown first; the volunteer is
a number, never a name.

Sessions you can record (`--script`):
    free        (default) about 60 s: stillness, 2 pushes at the camera (they also line
                the camera clock up with the audio clock), then swipe left / swipe
                right / push in small blocks, counted on screen as the camera sees them
    gestures    strict "get ready ... GO" trials like record.py, with the camera refining
                the timing (this is also what runs when no camera or tracker is available)
    negatives   about 60 s of everyday movement that is NOT a gesture: typing, reaching
                for a cup, fidgeting, leaning, walking off and back. All labelled background
    calibrate   "each gesture 3 times" at the demo table, for sonar/adapt.py

How to run:
    Windows:  py -3.12 -m sonar.camrecord --person v01 --profile clean --blocksize 3840
    macOS:    .venv/bin/python -m sonar.camrecord --person v01 --in-device MacBook --out-device MacBook
    no hardware (simulator + a fake camera that watches the simulated hand):
              python -m sonar.camrecord --person sim --backend sim --fast --no-window --consent yes
    label an existing session again (after changing a threshold; reads landmarks.jsonl + audio.wav):
              python -m sonar.camrecord --relabel data/<session>
Keys in the window: Y agree, M mirror the picture (consent screen only), SPACE skip a step, Q finish early.

What it changes: creates data/<date-time>_<person>_<script>/ with
    audio.wav        raw microphone audio, 48 kHz, 24-bit (identical to record.py;
                     with --ultrasound-only everything below about 12 kHz, i.e. all speech, is removed)
    session.json     as record.py, plus the camera's labels, the clock offset and the consent
    landmarks.jsonl  the hand points of every camera picture (no pictures)
and appends the measured clock offset to <out>/camsync.json. While it runs it holds the
speaker and microphone exclusively and uses the webcam. The first run downloads the
hand-tracking model (7.8 MB) into models/mediapipe/. --relabel rewrites session.json of that
session (the first original is kept as session.json.orig).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import __version__
from .audio import AudioError, make_backend
from .camera import HAND_LINKS, CameraFrame, CameraUnavailable, FakeCamera, MediaPipeHands, NullTracker, OpenCVCamera
from .camlabel import (
    GESTURES, LANDMARK_FILE, CamEvent, CamFrame, HandObs, Intent, LabelConfig, LabelResult, Stroke, all_strokes,
    build_tracks, frame_to_json, label_config_dict, label_session, landmark_header, load_landmarks,
)
from .config import GESTURE_LABELS, SonarConfig, get_profile, profile_names
from .dataset import AUDIO_FILE, SESSION_FILE, load_session
from .record import Recorder, _banner

SYNC_FILE = "camsync.json"
HOW_TO = {
    "swipe_left": "Sweep your open hand from RIGHT to LEFT in front of the screen",
    "swipe_right": "Sweep your open hand from LEFT to RIGHT in front of the screen",
    "push": "Push your open palm at the CAMERA, then take it back",
    "pull": "Reach out slowly, then pull your hand back QUICKLY",
}
NEGATIVES: list[tuple[str, float, str]] = [
    ("typing", 10.0, "TYPE on the keyboard and use the touchpad"),
    ("reach", 10.0, "REACH for a cup or your phone beside the laptop, take a sip, put it back"),
    ("fidget", 8.0, "SCRATCH your head, fix your hair or glasses, rest your chin on your hand"),
    ("lean", 8.0, "LEAN in to read the screen, lean back, turn and talk to a neighbour"),
    ("point", 8.0, "POINT at something on the screen, then wave hello to somebody"),
    ("walk", 12.0, "STAND UP, walk two steps away, come back and sit down"),
]
CONSENT = [
    "This station records two things while you move your hand:",
    "  1. sound ABOVE human hearing from the laptop's microphones{voices}",
    "  2. the position of your hand joints, found by the camera",
    "No photo or video is saved{video}. No name: you are volunteer {person}.",
    "The recording trains a hackathon gesture model and is shared with the team only.",
]


# ---------------------------------------------------------------------------
# Clocks, privacy filter, recorder
# ---------------------------------------------------------------------------
class AudioClock:
    """Which audio sample is being recorded at a given moment of the computer's monotonic clock.

    Every time a block of audio arrives we note (arrival time, samples so far). Blocks arrive
    late by a varying amount but never early, so the EARLIEST relation seen in the last few
    seconds is the true one (plus a small constant delay that the motion sync measures later).
    """

    def __init__(self, fs: int, window_s: float = 10.0) -> None:
        self.fs = fs
        self.window_s = window_s
        self._marks: deque[tuple[float, float]] = deque()
        self._ref: float | None = None
        self.log: list[list[float]] = []            # about one [seconds since start, samples] pair per second
        self.t0: float | None = None

    def add(self, t: float, n_total: int) -> None:
        if self.t0 is None:
            self.t0 = t
        self._marks.append((t, t - n_total / self.fs))
        while self._marks and t - self._marks[0][0] > self.window_s:
            self._marks.popleft()
        self._ref = min(m[1] for m in self._marks)
        if not self.log or t - self.t0 - self.log[-1][0] >= 1.0:
            self.log.append([round(t - self.t0, 4), int(n_total)])

    def reset(self) -> None:
        """Audio was dropped: the old relation no longer holds."""
        self._marks.clear()

    def sample_at(self, t: float) -> float:
        return 0.0 if self._ref is None else (t - self._ref) * self.fs


class PrivacyFilter:
    """Removes everything below the sonar band (all speech) from the audio before it is saved.

    A linear-phase high-pass; the delay it causes is taken out again, so every sample stays at
    its place and the labels still point at the right spot. The sonar processing ignores
    everything below about 15 kHz anyway (its own prefilter), so features do not change.
    """

    def __init__(self, cfg: SonarConfig, n_mics: int) -> None:
        from scipy import signal as sps

        span = cfg.doppler_bins / 2 * cfg.bb_fs / cfg.stft_fft
        prefilter_stop = min(min(cfg.tone_hz), cfg.chirp_lo_hz) - span - cfg.prefilter_margin_hz - cfg.prefilter_transition_hz
        self.pass_hz = float(min(prefilter_stop - 150.0, 14000.0))
        self.stop_hz = self.pass_hz - 2000.0
        numtaps, beta = sps.kaiserord(90.0, 2000.0 / (0.5 * cfg.fs))
        numtaps |= 1
        self.taps = sps.firwin(numtaps, 0.5 * (self.pass_hz + self.stop_hz), window=("kaiser", beta), pass_zero=False, fs=cfg.fs)
        self._delay = (numtaps - 1) // 2
        self._zi = np.zeros((numtaps - 1, n_mics))
        self._skip = self._delay
        self._lfilter = sps.lfilter
        self._n_mics = n_mics

    def process(self, x: np.ndarray) -> np.ndarray:
        y, self._zi = self._lfilter(self.taps, [1.0], np.asarray(x, dtype=np.float64), axis=0, zi=self._zi)
        if self._skip:
            k = min(self._skip, len(y))
            y, self._skip = y[k:], self._skip - k
        return y.astype(np.float32)

    def flush(self) -> np.ndarray:
        return self.process(np.zeros((self._delay, self._n_mics)))


class CamRecorder(Recorder):
    """record.Recorder plus: the audio clock, the movement curve per frame, the optional privacy filter."""

    def __init__(self, cfg: SonarConfig, backend: Any, folder: Path, meta: dict[str, Any], ultrasound_only: bool = False) -> None:
        super().__init__(cfg, backend, folder, meta)
        self.clock = AudioClock(self.cfg.fs)
        self.activity: list[float] = []                 # movement level of frame index i is activity[i - 1]
        self._privacy = PrivacyFilter(self.cfg, self.cfg.n_mics) if ultrasound_only else None
        self.t_start = 0.0

    def _loop(self) -> None:
        from .gestures import heuristic_classify

        try:
            while self._running:
                rx = self.backend.read(timeout=0.5)
                if rx is None:
                    continue
                t = time.monotonic()
                if self.backend.pop_discontinuity():
                    self.dropouts.append(self.n_samples)
                    self.pipeline.notify_discontinuity()
                    self.clock.reset()
                self._wav.write(rx if self._privacy is None else self._privacy.process(rx))
                self.peak = max(self.peak, float(np.max(np.abs(rx))))
                for fr in self.pipeline.process(rx):
                    self.activity.append(float(fr.activity_db))
                    if fr.presence is not None:
                        self.presence = fr.presence.state
                    if fr.segment is not None and fr.window is not None:
                        r = heuristic_classify(fr.window, self.cfg)
                        self.bursts.append((fr.segment, r.label, r.confidence))
                with self._cond:
                    self.n_samples += len(rx)
                    self._cond.notify_all()
                self.clock.add(t, self.n_samples)
        except BaseException as exc:
            self.error = exc
            with self._cond:
                self._cond.notify_all()

    def start(self) -> None:
        self.t_start = time.monotonic()
        super().start()

    def wait_progress(self, timeout: float) -> None:
        """Sleep until more audio has arrived (or the timeout passes)."""
        with self._cond:
            self._cond.wait(timeout=timeout)
        if self.error is not None:
            raise AudioError(f"recording stopped: {self.error}")

    def activity_curve(self) -> np.ndarray:
        """Movement level on the labeller's grid: element k = frame index k (element 0 is unused)."""
        return np.concatenate([[0.0], np.asarray(self.activity, dtype=np.float64)])

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.backend.stop()
        if self._privacy is not None:
            self._wav.write(self._privacy.flush())
        self._wav.close()
        self.save_meta()


# ---------------------------------------------------------------------------
# Live stroke detection (for the on-screen counter)
# ---------------------------------------------------------------------------
class LiveStrokes:
    """Finds finished hand strokes in the last few seconds of camera frames, each reported once."""

    def __init__(self, width: int, height: int, fs: int, cfg: LabelConfig) -> None:
        self.width, self.height, self.fs, self.cfg = width, height, fs, cfg
        self._frames: deque[CamFrame] = deque()
        self._reported: deque[float] = deque(maxlen=64)

    def add(self, frame: CamFrame) -> None:
        self._frames.append(frame)
        while self._frames and frame.sample - self._frames[0].sample > 5.0 * self.fs:
            self._frames.popleft()

    def poll(self) -> list[Stroke]:
        if len(self._frames) < 5:
            return []
        now = self._frames[-1].sample / self.fs
        strokes, _ = all_strokes(build_tracks(list(self._frames), self.width, self.height, self.fs, self.cfg), self.cfg)
        fresh = []
        for st in strokes:
            if now - st.t1 < 0.15 or any(abs(st.t_peak - t) < 0.25 for t in self._reported):
                continue
            self._reported.append(st.t_peak)
            fresh.append(st)
        return fresh


# ---------------------------------------------------------------------------
# The screen (OpenCV window, or the console)
# ---------------------------------------------------------------------------
class Screen:
    """Shows the prompt, the mirrored camera picture with the hand points, and the counters."""

    def __init__(self, window: bool, show_video: bool = True, size: tuple[int, int] = (960, 720)) -> None:
        self.cv2: Any = None
        self.size = size
        self.show_video = show_video
        self._last_console = ""
        if window:
            try:
                import cv2

                cv2.namedWindow("Sonar data station", cv2.WINDOW_AUTOSIZE)
                self.cv2 = cv2
            except Exception as exc:                 # no display, or OpenCV built without a GUI
                print(f"(no window: {type(exc).__name__}: {exc}; prompts are printed here instead)")

    @property
    def window(self) -> bool:
        return self.cv2 is not None

    def console(self, text: str) -> None:
        if text and text != self._last_console:
            print(text, flush=True)
            self._last_console = text

    def render(self, image: np.ndarray | None, hands: list[HandObs], view: dict[str, Any]) -> int:
        """Draw one screen; returns the key pressed (-1 for none)."""
        cv2 = self.cv2
        if cv2 is None:
            return -1
        cv2.imshow("Sonar data station", self.compose(image, hands, view))
        key = int(cv2.waitKey(1))
        return -1 if key < 0 else key & 0xFF

    def compose(self, image: np.ndarray | None, hands: list[HandObs], view: dict[str, Any]) -> np.ndarray:
        """The picture of one screen (BGR). Needs OpenCV, but no window."""
        import cv2

        w, h = self.size
        if image is not None and self.show_video:
            canvas = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        else:
            canvas = np.full((h, w, 3), 24, dtype=np.uint8)
        for obs in hands:
            pts = [(int(p[0] * w), int(p[1] * h)) for p in obs.lm]
            for a, b in HAND_LINKS:
                cv2.line(canvas, pts[a], pts[b], (80, 220, 80), 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(canvas, p, 4, (40, 255, 255), -1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_SIMPLEX

        def bar(y0: int, y1: int) -> None:
            canvas[y0:y1] = (canvas[y0:y1] * 0.3).astype(np.uint8)

        bar(0, 118)
        cv2.putText(canvas, str(view.get("title", ""))[:44], (18, 46), font, 1.25, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, str(view.get("text", ""))[:86], (18, 84), font, 0.62, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(view.get("hint", ""))[:96], (18, 108), font, 0.5, (150, 200, 255), 1, cv2.LINE_AA)
        if view.get("count") is not None:
            done, target = view["count"]
            cv2.putText(canvas, f"{done}/{target}", (w - 190, 78), font, 2.0, (80, 255, 120), 4, cv2.LINE_AA)
        if view.get("progress") is not None:
            cv2.rectangle(canvas, (0, 118), (int(w * float(np.clip(view["progress"], 0, 1))), 126), (80, 200, 255), -1)
        if view.get("flash"):
            size = cv2.getTextSize(view["flash"], font, 2.2, 5)[0]
            cv2.putText(canvas, view["flash"], ((w - size[0]) // 2, h // 2), font, 2.2, (80, 255, 120), 5, cv2.LINE_AA)
        for k, line in enumerate(view.get("lines", [])):
            cv2.putText(canvas, line[:100], (26, 190 + 34 * k), font, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
        if view.get("sides"):
            cv2.putText(canvas, "YOUR LEFT", (14, h - 84), font, 0.8, (255, 200, 80), 2, cv2.LINE_AA)
            cv2.putText(canvas, "YOUR RIGHT", (w - 190, h - 84), font, 0.8, (255, 200, 80), 2, cv2.LINE_AA)
        bar(h - 62, h)
        cv2.putText(canvas, str(view.get("status", ""))[:112], (14, h - 38), font, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, str(view.get("footer", ""))[:112], (14, h - 14), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return canvas

    def close(self) -> None:
        if self.cv2 is not None:
            try:
                self.cv2.destroyAllWindows()
                self.cv2.waitKey(1)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# The station: camera + recorder + what the volunteer is asked to do
# ---------------------------------------------------------------------------
class Station:
    def __init__(self, rec: CamRecorder | None, camera: Any, tracker: Any, screen: Screen, label_cfg: LabelConfig,
                 person: str, fs: int, footer: str, sim: Any = None, video: Any = None) -> None:
        self.rec = rec
        self.camera, self.tracker, self.screen, self.label_cfg = camera, tracker, screen, label_cfg
        self.person, self.fs, self.footer, self.sim = person, fs, footer, sim
        self.video = video
        self.frames: list[CamFrame] = []
        self.prompts: list[dict[str, Any]] = []
        self.live = LiveStrokes(camera.width, camera.height, fs, label_cfg) if camera is not None else None
        self._fresh: list[Stroke] = []
        self._fh: Any = None
        self._image: np.ndarray | None = None
        self._hands: list[HandObs] = []
        self._hand_time = 0.0
        self.view: dict[str, Any] = {}
        self._flash_until = 0.0
        self.quit = False
        self.skip = False
        self.last_key = -1
        self.cam_frames_seen = 0
        self._last_cam = time.monotonic()
        self._last_draw = 0.0
        self._new_picture = False
        self._polls = 0
        self.camera_lost = False
        self.last_count = 0
        self.blind = False                 # the camera does not see the gestures: use timed prompts instead

    @property
    def has_camera(self) -> bool:
        return self.camera is not None and not self.camera_lost

    # -- one turn of the main loop -----------------------------------------------------------
    def open_landmarks(self, path: Path, header: dict[str, Any]) -> None:
        self._fh = open(path, "w", encoding="utf-8")
        self._fh.write(json.dumps(header) + "\n")

    def _take_camera(self) -> None:
        if self.camera is None:
            return
        got = False
        while True:
            cf: CameraFrame | None = self.camera.read()
            if cf is None:
                break
            got = True
            hands, pose = cf.hands, None
            t_now = cf.t if cf.t is not None else time.monotonic()
            if hands is None:
                try:
                    hands, pose = self.tracker.process(cf.image, t_now)
                except Exception as exc:                         # a tracker failure must never cost the audio recording
                    if not self.camera_lost:
                        print(f"WARNING: hand tracking failed ({type(exc).__name__}: {exc}); continuing with prompted labels only", flush=True)
                    self.camera_lost, hands = True, []
            self._image, self._hands = cf.image, hands
            if hands:
                self._hand_time = time.monotonic()
            self.cam_frames_seen += 1
            if self.rec is None or self._fh is None:
                continue                                         # consent screen: preview only, nothing is kept
            sample = cf.sample if cf.sample is not None else self.rec.clock.sample_at(t_now)
            frame = CamFrame(len(self.frames), t_now - self.rec.t_start if cf.t is not None else sample / self.fs, sample, hands, pose)
            self.frames.append(frame)
            self._fh.write(frame_to_json(frame) + "\n")
            if self.live is not None:
                self.live.add(frame)
            if self.video is not None and cf.image is not None:
                self.video.write(cf.image)
            if cf.image is None:
                continue                                         # fake camera: drain every frame that is due
            break                                                # real camera: one picture per turn
        if got:
            self._last_cam = time.monotonic()
            self._new_picture = True
        elif self.rec is not None and not self.camera_lost and getattr(self.camera, "name", "") == "opencv" \
                and time.monotonic() - self._last_cam > 3.0:
            self.camera_lost = True
            print("WARNING: the camera stopped delivering pictures; continuing with prompted labels only", flush=True)
        if got and self.live is not None and self.rec is not None:
            self._polls += 1
            if self.sim is not None or self._polls % 3 == 0:     # ten looks a second are plenty for a counter
                self._fresh += self.live.poll()

    def pump(self) -> None:
        """Camera in, screen out, keys. Called all the time by the session scripts."""
        self._take_camera()
        key = -1
        if self._new_picture or time.monotonic() - self._last_draw >= 0.033:      # at most about 30 screens a second
            view = dict(self.view)
            if time.monotonic() > self._flash_until:
                view.pop("flash", None)
            view["footer"] = self.footer
            if self.rec is not None and self.screen.window:
                cam = "no camera" if not self.has_camera else ("hand in view" if time.monotonic() - self._hand_time < 0.5 else "NO HAND IN VIEW")
                act = self.rec.activity[-1] if self.rec.activity else 0.0
                view["status"] = (f"volunteer {self.person}   {self.rec.now_s:5.1f} s   sonar movement {act:4.1f} dB   "
                                  f"signal {self.rec.pipeline.snr_db():.0f} dB   camera: {cam}")
            key = self.screen.render(self._image, self._hands, view)
            self._last_draw, self._new_picture = time.monotonic(), False
        self.last_key = key
        if key in (ord("q"), ord("Q"), 27):
            self.quit = True
        if key in (ord(" "), ord("n"), ord("N")):
            self.skip = True
        if self.rec is not None:
            self.rec.wait_progress(0.004 if self.has_camera else 0.02)
        else:
            time.sleep(0.005)                                    # consent screen: nothing is recording yet

    def show(self, title: str, text: str = "", hint: str = "", **more: Any) -> None:
        self.view = {"title": title, "text": text, "hint": hint, **more}
        self.screen.console(f"\n>>> {title}" + (f"   {text}" if text else ""))

    def flash(self, text: str, seconds: float = 0.7) -> None:
        self.view["flash"] = text
        self._flash_until = time.monotonic() + seconds

    def take_strokes(self) -> list[Stroke]:
        out, self._fresh = self._fresh, []
        return out

    # -- building blocks of a session (generators: each `yield` hands control back to the main loop) --
    def wait(self, seconds: float, progress: bool = False) -> Iterator[None]:
        assert self.rec is not None
        start = self.rec.n_samples
        end = start + int(seconds * self.fs)
        while self.rec.n_samples < end and not self.quit:
            if progress:
                self.view["progress"] = (self.rec.n_samples - start) / max(end - start, 1)
            yield
        self.view.pop("progress", None)

    def calibrate(self, seconds: float = 6.0) -> Iterator[None]:
        rec = self.rec
        assert rec is not None
        self.show("SIT STILL", "Hands on your lap or the table. Do not move for a few seconds.", "learning the room")
        start = rec.n_samples
        yield from self.wait(seconds, progress=True)
        rec.mark("calibration", "calibration", start, rec.n_samples)
        self.prompts.append({"type": "calibration", "start": start, "end": rec.n_samples})
        locked = rec.pipeline.frontend.locked
        print(f"  signal headroom {rec.pipeline.snr_db():.0f} dB, chirp lock {'OK' if locked else 'NOT LOCKED'}, "
              f"input peak {20 * np.log10(rec.peak + 1e-9):.0f} dBFS", flush=True)
        if rec.peak >= 0.99:
            print("  WARNING: the microphone is clipping. Lower the speaker volume or the mic level.")
        if not locked or rec.pipeline.snr_db() < 25:
            print("  WARNING: weak or missing sonar signal. Run `python -m sonar.selftest` before recording.")

    def block(self, label: str, target: int, sync: bool = False, per_rep_s: float = 5.0) -> Iterator[None]:
        """Free-form block: the volunteer does `label` `target` times at their own pace; the camera counts."""
        rec = self.rec
        assert rec is not None
        name = label.upper().replace("_", " ")
        self.show(name, HOW_TO.get(label, ""), "first, to line the clocks up: do it twice" if sync else "at your own pace; the counter goes up when the camera sees it",
                  count=(0, target))
        self.take_strokes()
        self.skip = False
        start = rec.n_samples
        deadline = start + int((target * per_rep_s + 4.0) * self.fs)
        count, next_sim = 0, start
        while count < target and rec.n_samples < deadline and not (self.skip or self.quit):
            if self.sim is not None and rec.n_samples >= next_sim:
                self.sim.perform(label)
                next_sim = rec.n_samples + int(2.7 * self.fs)
            for st in self.take_strokes():
                self.view["hint"] = f"camera saw: {st.kind.replace('_', ' ')}, {st.peak_speed:.1f} m/s, {100 * st.dist:.0f} cm"
                if st.kind == label or (label == "pull" and st.kind == "away"):
                    count += 1
                    self.view["count"] = (count, target)
                    self.flash("OK")
                    self.screen.console(f"        camera saw {label} {count}/{target}  ({st.peak_speed:.1f} m/s, {100 * st.dist:.0f} cm)")
            yield
        yield from self.wait(1.2 if self.sim is None else 2.0)     # let the last movement finish and its burst close
        end = rec.n_samples
        row = {"type": "block", "intent": label, "start": start, "end": end, "target": target, "count": count, "sync": sync}
        self.prompts.append(row)
        self.last_count = count
        rec.mark("block", "block", start, end, intent=label, target=target, count=count, exclude=True)
        if count < target:
            self.screen.console(f"        only {count} of {target} seen by the camera (keep the hand inside the picture)")

    def trial(self, label: str, n: int, total: int, ready_s: float, trial_s: float, rest_s: float) -> Iterator[None]:
        """Strict prompted trial, exactly like record.py: get ready ... GO ... rest."""
        rec = self.rec
        assert rec is not None
        name = label.upper().replace("_", " ")
        self.show(f"get ready: {name}", HOW_TO.get(label, ""), f"trial {n} of {total}")
        yield from self.wait(ready_s)
        start = rec.n_samples
        self.show(f"GO >>> {name}", HOW_TO.get(label, ""), f"trial {n} of {total}")
        if self.sim is not None:
            self.sim.perform(label)
        yield from self.wait(trial_s, progress=True)
        end = rec.n_samples
        self.show("rest", "let your hand drop back", f"trial {n} of {total}")
        yield from self.wait(rest_s)
        found = rec.bursts_between(start, end)
        self.prompts.append({"type": "trial", "intent": label, "start": start, "end": end, "rep": n})
        rec.mark(label, "gesture", start, end, rep=int(n))          # crash-safe; refined by the camera at the end
        if found:
            seg, guess, conf = max(found, key=lambda b: b[0].peak_db)
            print(f"        movement detected (peak {seg.peak_db:.0f} dB, {seg.n_frames * 20} ms); physics guess: {guess} ({conf:.2f})", flush=True)
        else:
            print("        NO movement detected. Move closer / faster / larger.", flush=True)

    def activity(self, name: str, seconds: float, text: str) -> Iterator[None]:
        """A stretch of everyday movement that must NOT count as a gesture."""
        rec = self.rec
        assert rec is not None
        self.show("NO GESTURES", text, "just be natural; nothing here should trigger the sonar")
        start = rec.n_samples
        self.skip = False
        end_at = start + int(seconds * self.fs)
        while rec.n_samples < end_at and not (self.skip or self.quit):
            self.view["progress"] = (rec.n_samples - start) / max(end_at - start, 1)
            yield
        self.view.pop("progress", None)
        end = rec.n_samples
        self.prompts.append({"type": "activity", "activity": name, "start": start, "end": end})
        rec.mark("distractor", "scene", start, end, activity=name)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# Session scripts
# ---------------------------------------------------------------------------
def script_free(st: Station, gestures: list[str], reps: int, rounds: int, rng: np.random.Generator, ready_s: float = 1.0,
                trial_s: float = 2.5, rest_s: float = 1.2) -> Iterator[None]:
    yield from st.calibrate()
    if "push" in gestures:
        yield from st.block("push", 2, sync=True)
        st.blind = st.last_count == 0 and not st.quit
    per_round = [reps // rounds + (1 if r < reps % rounds else 0) for r in range(rounds)]
    total, done, empty = reps * len(gestures), 0, 0
    for n in per_round:
        if n <= 0:
            continue
        for k in rng.permutation(len(gestures)):               # shuffled: classes must not line up with time
            if st.quit:
                return
            st.blind = st.blind or not st.has_camera
            if st.blind:
                # The camera is not seeing the gestures (hand outside the picture, bad light): do not make the
                # volunteer wait for a counter that never moves. Timed prompts label the rest, as sonar.record does.
                if done == 0 or empty:
                    print("  NOTE: the camera did not see the gestures; the rest of this session uses timed prompts "
                          "(get ready ... GO). Check that the raised hand is inside the picture.", flush=True)
                    empty = 0
                for _ in range(n):
                    done += 1
                    yield from st.trial(gestures[int(k)], done, total, ready_s, trial_s, rest_s)
                continue
            yield from st.block(gestures[int(k)], n)
            done += n
            empty = empty + 1 if st.last_count == 0 else 0
            st.blind = empty >= 2
    st.show("DONE", "Thank you!")
    yield from st.wait(1.0)


def script_trials(st: Station, gestures: list[str], reps: int, rng: np.random.Generator, ready_s: float, trial_s: float,
                  rest_s: float) -> Iterator[None]:
    yield from st.calibrate()
    trials = [g for g in gestures for _ in range(reps)]
    order = rng.permutation(len(trials))
    _banner(f"GESTURES: {len(trials)} trials, about {len(trials) * (ready_s + trial_s + rest_s) / 60:.1f} min")
    yield from st.wait(2.0)
    for n, k in enumerate(order, 1):
        if st.quit:
            return
        yield from st.trial(trials[int(k)], n, len(trials), ready_s, trial_s, rest_s)
    st.show("DONE", "Thank you!")
    yield from st.wait(1.0)


def script_negatives(st: Station, scale: float = 1.0) -> Iterator[None]:
    yield from st.calibrate()
    if st.has_camera:
        yield from st.block("push", 2, sync=True)
    for name, seconds, text in NEGATIVES:
        if st.quit:
            return
        yield from st.activity(name, seconds * scale, text)
    st.show("DONE", "Thank you!")
    yield from st.wait(1.0)


# ---------------------------------------------------------------------------
# From prompts + camera labels to the events of session.json
# ---------------------------------------------------------------------------
def intents_of(prompts: list[dict[str, Any]], fs: int) -> list[Intent]:
    out = []
    for p in prompts:
        if p["type"] in ("block", "trial"):
            out.append(Intent(p["start"] / fs, p["end"] / fs, p["intent"]))
        elif p["type"] in ("activity", "calibration"):
            out.append(Intent(p["start"] / fs, p["end"] / fs, "none"))
    return out


def build_events(prompts: list[dict[str, Any]], result: LabelResult | None, fs: int, n_samples: int) -> list[dict[str, Any]]:
    """The `events` list of session.json: prompts and camera labels merged, in time order.

    * a camera-timed gesture: {"label": gesture, "kind": "gesture", "tight": true, ...}: start/end bound the stroke
    * a free-form block: {"label": "block", "exclude": true}: nothing in it is used as a "none" example,
      because the volunteer may have gestured where the camera could not see
    * an unsure camera reading: {"label": "review", "exclude": true, "candidate": gesture}
    * a strict trial the camera did not see: the plain prompt window, exactly as record.py writes it
    """
    cam = result.events if result is not None else []
    events: list[dict[str, Any]] = []
    for p in prompts:
        if p["type"] == "calibration":
            events.append({"label": "calibration", "kind": "calibration", "start": p["start"], "end": p["end"]})
        elif p["type"] == "block":
            events.append({"label": "block", "kind": "block", "start": p["start"], "end": p["end"], "intent": p["intent"],
                           "target": p["target"], "count": p["count"], "exclude": True, **({"sync": True} if p.get("sync") else {})})
        elif p["type"] == "activity":
            events.append({"label": "distractor", "kind": "scene", "start": p["start"], "end": p["end"], "activity": p["activity"]})
        elif p["type"] == "trial":
            t0, t1 = p["start"] / fs - 0.2, p["end"] / fs + 0.5
            mine = [e for e in cam if t0 <= e.t_peak <= t1 and e.kind in ("gesture", "review")]
            if mine:        # the camera saw it (or saw something else): its timing replaces the prompt window
                events.append({"label": "prompt", "kind": "prompt", "start": p["start"], "end": p["end"], "intent": p["intent"],
                               "rep": p["rep"], "exclude": True})
            else:           # the camera saw nothing: fall back to the prompt window
                events.append({"label": p["intent"], "kind": "gesture", "start": p["start"], "end": p["end"], "rep": p["rep"],
                               "source": "prompt", "confidence": 0.5})
    reps: dict[str, int] = {}
    for e in cam:
        rep = None
        if e.kind == "gesture":
            reps[e.label] = rep = reps.get(e.label, 0) + 1
        ev = e.to_session_event(fs, n_samples, rep)
        if ev["end"] > ev["start"] and ev["start"] < n_samples:
            events.append(ev)
    return sorted(events, key=lambda ev: (ev["start"], ev["end"]))


def physics_agreement(events: list[dict[str, Any]], bursts: list[tuple[Any, str, float]], cfg: SonarConfig) -> tuple[int, int, int]:
    """(camera gestures, of those with a sonar burst, of those where the physics rules name the same gesture)."""
    n = hit = agree = 0
    for ev in events:
        if ev.get("kind") != "gesture" or ev.get("source") != "camera":
            continue
        n += 1
        f0, f1 = ev["start"] // cfg.frame_len + 1, ev["end"] // cfg.frame_len + 1
        best, best_ov = None, 0
        for seg, guess, _ in bursts:
            ov = min(seg.end, f1 + 3) - max(seg.start, f0 - 1) + 1
            if ov > best_ov:
                best, best_ov = guess, ov
        if best is not None:
            hit += 1
            agree += best == ev["label"]
    return n, hit, agree


def load_sync_prior(out: Path, key: str) -> float:
    """Median clock offset measured in earlier sessions with the same camera and microphone (0 if none)."""
    try:
        rows = [r for r in json.loads((out / SYNC_FILE).read_text()) if r.get("key") == key and r.get("method") == "motion"]
    except (OSError, ValueError):
        return 0.0
    return float(np.median([r["offset_ms"] for r in rows[-7:]]) / 1e3) if rows else 0.0


def save_sync(out: Path, key: str, session_id: str, sync: dict[str, Any]) -> None:
    path = out / SYNC_FILE
    try:
        rows = json.loads(path.read_text())
    except (OSError, ValueError):
        rows = []
    rows.append({"key": key, "session": session_id, **sync})
    path.write_text(json.dumps(rows[-200:], indent=1))


def summarize(events: list[dict[str, Any]], result: LabelResult | None, bursts: list[tuple[Any, str, float]] | None,
              cfg: SonarConfig, camera_on: bool) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for ev in events:
        if ev["kind"] == "gesture":
            counts[ev["label"]] = counts.get(ev["label"], 0) + 1
    out: dict[str, Any] = {
        "source": "camera" if camera_on else "prompt", "version": 1, "gestures": counts,
        "from_camera": sum(1 for ev in events if ev["kind"] == "gesture" and ev.get("source") == "camera"),
        "from_prompt_only": sum(1 for ev in events if ev["kind"] == "gesture" and ev.get("source") == "prompt"),
        "review": sum(1 for ev in events if ev["kind"] == "review"),
        "other_movements": sum(1 for ev in events if ev["kind"] == "motion"),
    }
    if result is not None:
        out["camera_frames"] = result.n_frames
        out["hand_in_view_s"] = round(result.hand_seen_s, 1)
        out["config"] = label_config_dict(result.config)
    if bursts is not None:
        n, hit, agree = physics_agreement(events, bursts, cfg)
        out["physics_check"] = {"camera_gestures": n, "with_sonar_burst": hit, "physics_agrees": agree}
    return out


def print_summary(labels: dict[str, Any], sync: dict[str, Any] | None) -> None:
    g = ", ".join(f"{v} {k}" for k, v in sorted(labels["gestures"].items())) or "none"
    print(f"\nlabels: {g}   ({labels['from_camera']} timed by the camera, {labels['from_prompt_only']} from the prompt alone, "
          f"{labels['review']} marked for review, {labels['other_movements']} other hand movements)")
    if sync is not None:
        if sync["method"] == "motion":
            res = "n/a" if sync["residual_ms"] is None else f"{sync['residual_ms']:.0f} ms"
            print(f"clock sync: camera runs {sync['offset_ms']:+.0f} ms against the audio (match {sync['peak_corr']:.2f}, "
                  f"stroke-to-stroke spread {res} over {sync['n_strokes']} strokes); corrected")
        else:
            print(f"clock sync: NOT measured ({sync['note']}); used {sync['offset_ms']:+.0f} ms")
    pc = labels.get("physics_check")
    if pc and pc["camera_gestures"]:
        print(f"sonar check: {pc['with_sonar_burst']}/{pc['camera_gestures']} camera gestures have a sonar movement burst; "
              f"the physics rules name the same gesture in {pc['physics_agrees']}/{max(pc['with_sonar_burst'], 1)}")
        if pc["with_sonar_burst"] >= 4 and pc["physics_agrees"] <= 0.25 * pc["with_sonar_burst"]:
            print("  NOTE: almost none agree. If the SWIPES are the ones that disagree, the camera picture may be mirrored "
                  "(press M on the consent screen) or the microphone/speaker channels are swapped.")


# ---------------------------------------------------------------------------
# Re-label an existing session
# ---------------------------------------------------------------------------
def relabel(folder: Path, hfov: float | None = None, quiet: bool = False) -> int:
    """Run the labeller again on a recorded session (landmarks.jsonl + audio.wav) and rewrite its events."""
    from .features import stream_features

    folder = Path(folder)
    lm_path = folder / LANDMARK_FILE
    if not lm_path.exists():
        print(f"ERROR: {folder} has no {LANDMARK_FILE} (not a camera session)", file=sys.stderr)
        return 2
    meta, audio, cfg = load_session(folder)
    header, frames = load_landmarks(lm_path)
    if hfov:
        header["hfov_deg"] = hfov
    stream = stream_features(audio, cfg)
    act = np.concatenate([[0.0], stream.activity_db])
    prompts = meta.get("prompts", [])
    label_cfg = LabelConfig(hfov_deg=float(header.get("hfov_deg", 70.0)))
    prior = float(meta.get("sync", {}).get("prior_ms", 0.0)) / 1e3
    result = label_session(frames, header, act, intents_of(prompts, cfg.fs), label_cfg, prior)
    events = build_events(prompts, result, cfg.fs, int(meta.get("n_samples", len(audio))))
    bursts = []
    from .gestures import heuristic_classify
    from .features import window_from_stream

    for k, seg in enumerate(stream.segments):
        r = heuristic_classify(window_from_stream(stream, seg.decided, stream.not_before[k]), stream.cfg)
        bursts.append((seg, r.label, r.confidence))
    labels = summarize(events, result, bursts, stream.cfg, camera_on=True)
    orig = folder / (SESSION_FILE + ".orig")
    if not orig.exists():
        orig.write_text((folder / SESSION_FILE).read_text())
    meta.update(events=events, labels=labels, sync={**result.sync.to_dict(), "prior_ms": round(1e3 * prior, 1),
                                                    "relabelled": time.strftime("%Y-%m-%dT%H:%M:%S")})
    tmp = folder / (SESSION_FILE + ".tmp")
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(folder / SESSION_FILE)
    if not quiet:
        print(f"relabelled {folder}")
        print_summary(labels, meta["sync"])
    return 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
class SimVolunteer:
    """In simulator mode: makes the simulated hand perform what the screen asks for."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def perform(self, label: str) -> None:
        self.backend.perform(label)


def next_volunteer(out: Path) -> str:
    """v01, v02, ...: the first number no session folder under `out` uses yet."""
    used = [int(m.group(1)) for p in out.glob("*_v*_*") if (m := re.search(r"_v(\d+)_", p.name))]
    return f"v{max(used, default=0) + 1:02d}"


def open_camera(args: argparse.Namespace, backend: Any, cfg: SonarConfig) -> tuple[Any, Any, str]:
    """(camera, tracker, why not) — camera is None when the session must fall back to prompts."""
    kind = args.camera
    if kind == "none":
        return None, None, "--camera none"
    if kind == "auto":
        kind = "fake" if args.backend == "sim" else "0"
    try:
        if kind == "fake":
            path = getattr(backend, "_hand", None)
            if path is None:
                raise CameraUnavailable("the fake camera needs the simulator backend (--backend sim)")
            cam = FakeCamera(path, lambda: backend.now_s, fs=cfg.fs, fps=args.camera_fps, latency_s=args.fake_latency,
                             blur_speed=args.fake_blur, seed=args.seed or 0)
            return cam, NullTracker(), ""
        w, h = (int(v) for v in args.camera_size.lower().split("x"))
        cam = OpenCVCamera(kind, w, h, args.camera_fps, api=args.camera_api, facing=args.camera_facing,
                           mirrored_input=args.camera_mirrored)
        tracker = MediaPipeHands(num_hands=2, pose=args.pose)        # first: fails fast if MediaPipe is missing
        try:
            cam.start()
        except CameraUnavailable:
            tracker.close()
            raise
        return cam, tracker, ""
    except CameraUnavailable as exc:
        return None, None, str(exc)


def consent_screen(st: Station, args: argparse.Namespace, lines: list[str]) -> tuple[bool, str]:
    """Show what is recorded and wait for a yes. Returns (agreed, how)."""
    if args.consent == "yes":
        return True, "confirmed by the operator (--consent yes)"
    if not st.screen.window:
        print("\n".join(["", *lines, ""]))
        if not sys.stdin.isatty():
            print("ERROR: no window and no keyboard: run with --consent yes once the volunteer has agreed", file=sys.stderr)
            return False, ""
        return input("Type y and press Enter to agree: ").strip().lower().startswith("y"), "typed y in the console"
    st.view = {"title": "BEFORE WE START", "text": "Y = I agree      Q = no thanks      M = mirror the picture",
               "hint": "raise your RIGHT hand: it must show up on the RIGHT side. If it does not, press M.",
               "lines": lines, "sides": True}
    while True:
        st.pump()
        key = st.last_key
        if key in (ord("y"), ord("Y")):
            return True, "key Y on the consent screen"
        if st.quit or key in (ord("n"), ord("N")):
            return False, ""
        if key in (ord("m"), ord("M")) and hasattr(st.camera, "flip"):
            st.camera.flip = not st.camera.flip
            st.camera.mirrored_input = not st.camera.mirrored_input
            st.camera.info.update(mirrored_input=st.camera.mirrored_input, flipped_to_user_view=st.camera.flip)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sonar.camrecord", description="Record a sonar session labelled by the webcam")
    ap.add_argument("--person", help="volunteer id, no names (default: the next free v01, v02, ...)")
    ap.add_argument("--script", choices=["free", "gestures", "negatives", "calibrate"], default="free")
    ap.add_argument("--reps", type=int, help="repetitions of each gesture (default: free 6, gestures 12, calibrate 3)")
    ap.add_argument("--rounds", type=int, default=2, help="free: the repetitions are split over this many rounds")
    ap.add_argument("--gestures", default="swipe_left,swipe_right,push")
    ap.add_argument("--out", default="data", help="folder that receives one sub-folder per session")
    ap.add_argument("--profile", choices=profile_names(), default="default")
    ap.add_argument("--backend", choices=["live", "sim"], default="live")
    ap.add_argument("--notes", default="", help="free text: seat distance, room, volume ...")
    ap.add_argument("--ready-s", type=float, default=1.0)
    ap.add_argument("--trial-s", type=float, default=2.5)
    ap.add_argument("--rest-s", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--in-device")
    ap.add_argument("--out-device")
    ap.add_argument("--blocksize", type=int, default=1920)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--fast", action="store_true", help="simulator only: run faster than real time")
    ap.add_argument("--camera", default="auto", help="auto | none | fake | camera number (default auto: camera 0; the fake one with --backend sim)")
    ap.add_argument("--camera-size", default="640x480")
    ap.add_argument("--camera-fps", type=float, default=30.0)
    ap.add_argument("--camera-api", default="auto", choices=["auto", "dshow", "msmf", "avfoundation", "any"])
    ap.add_argument("--camera-facing", default="user", choices=["user", "world"],
                    help="user: the webcam looks at the volunteer (normal). world: a camera looking the same way as the volunteer")
    ap.add_argument("--camera-mirrored", action="store_true", help="the camera driver already delivers a mirror image")
    ap.add_argument("--hfov", type=float, default=70.0, help="horizontal field of view of the webcam in degrees (only distances to the screen depend on it)")
    ap.add_argument("--pose", action="store_true", help="also track shoulders, elbows, wrists and hips (no face points)")
    ap.add_argument("--no-window", action="store_true", help="no OpenCV window: prompts in the console only")
    ap.add_argument("--no-video-preview", action="store_true", help="show only the hand points, not the camera picture")
    ap.add_argument("--save-video", action="store_true", help="DEBUG ONLY: also save a small video (faces!) as video_debug.mp4")
    ap.add_argument("--ultrasound-only", action="store_true", help="remove everything below ~12 kHz (all speech) from the saved audio")
    ap.add_argument("--consent", choices=["ask", "yes"], default="ask", help="yes: the operator confirms the volunteer agreed")
    ap.add_argument("--negatives-scale", type=float, default=1.0, help="negatives: make every activity this many times longer")
    ap.add_argument("--fake-latency", type=float, default=0.09, help=argparse.SUPPRESS)
    ap.add_argument("--fake-blur", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--relabel", metavar="SESSION", help="label an existing camera session again and exit")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if args.relabel:
        return relabel(Path(args.relabel), hfov=None if args.hfov == 70.0 else args.hfov)

    cfg = get_profile(args.profile)
    gestures = [g for g in args.gestures.split(",") if g]
    unknown = [g for g in gestures if g not in GESTURES]
    if unknown:
        ap.error(f"unknown gestures {unknown}; choose from {list(GESTURES)}")
    seed = int(time.time()) % 100000 if args.seed is None else args.seed
    rng = np.random.default_rng(seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    person = args.person or next_volunteer(out)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", person):
        ap.error("--person: one word of letters, digits, - or _ (a volunteer number, not a name)")
    if args.fast and args.backend != "sim":
        ap.error("--fast only works with --backend sim")
    try:
        if args.backend == "live":
            backend = make_backend("live", cfg, in_device=args.in_device, out_device=args.out_device,
                                   blocksize=args.blocksize, gain=args.gain)
        else:
            backend = make_backend("sim", cfg, scene="idle", realtime=not args.fast, seed=seed)
    except AudioError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    camera, tracker, why = open_camera(args, backend, cfg)
    if camera is None:
        print(f"NO CAMERA LABELS: {why}")
        if args.script in ("free", "calibrate"):
            print("=> falling back to strictly timed prompts (the labels are then the prompt windows, as in sonar.record)")
    screen = Screen(window=not args.no_window, show_video=not args.no_video_preview)
    label_cfg = LabelConfig(hfov_deg=args.hfov)
    footer = "sound above hearing + hand joint positions only. No photo, no video, no name."
    st = Station(None, camera, tracker, screen, label_cfg, person, cfg.fs, footer, sim=SimVolunteer(backend) if args.backend == "sim" else None)
    lines = [ln.format(person=person, voices="" if args.ultrasound_only else " (they also pick up nearby voices)",
                       video=" (DEBUG VIDEO IS ON: a small video IS saved)" if args.save_video else "") for ln in CONSENT]
    if args.backend == "sim":
        agreed, how = True, "simulator (nobody is recorded)"
    else:
        agreed, how = consent_screen(st, args, lines)
    st.quit = st.skip = False
    if not agreed:
        print("no consent: nothing was recorded")
        if camera is not None:
            camera.stop()
            tracker.close()
        screen.close()
        return 3

    session_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{person}_{args.script}"
    folder = out / session_id
    cam_info = dict(getattr(camera, "info", {})) if camera is not None else {"backend": "none", "why": why}
    meta: dict[str, Any] = {
        "session_id": session_id, "person": person, "script": args.script, "notes": args.notes,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "profile": args.profile, "seed": seed,
        "software": __version__, "synthetic": args.backend == "sim", "recorder": "camrecord",
        "consent": {"given": True, "how": how, "volunteer": person, "text": lines, "time": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "camera": {**cam_info, **(getattr(tracker, "info", {}) if tracker is not None else {}), "hfov_deg": args.hfov,
                   "landmarks": LANDMARK_FILE if camera is not None else None, "video_saved": bool(args.save_video and camera is not None),
                   "pictures_saved": False},
        "privacy": {"audio": "ultrasound only (below ~12 kHz removed)" if args.ultrasound_only else "full band, as sonar.record",
                    "faces": "never saved" if not args.save_video else "DEBUG VIDEO SAVED"},
    }
    rec = CamRecorder(cfg, backend, folder, meta, ultrasound_only=args.ultrasound_only)
    st.rec = rec
    if camera is not None:
        try:
            st.open_landmarks(folder / LANDMARK_FILE, landmark_header(camera.width, camera.height, cfg.fs, hfov_deg=args.hfov, camera=cam_info))
            if args.save_video and getattr(camera, "name", "") == "opencv":
                import cv2

                st.video = cv2.VideoWriter(str(folder / "video_debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.camera_fps,
                                           (camera.width, camera.height))
        except Exception as exc:
            print(f"WARNING: camera set-up failed ({exc}); continuing without it")
            camera = st.camera = None
    try:
        rec.start()
    except AudioError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if camera is not None:
            camera.stop()
        screen.close()
        return 2
    print(f"recording to {folder}  (profile {args.profile}, audio {getattr(backend, 'info', {})}, camera {cam_info})")

    reps = args.reps or {"free": 6, "gestures": 12, "calibrate": 3, "negatives": 0}[args.script]
    if args.script in ("free", "calibrate") and st.has_camera:
        script = script_free(st, gestures, reps, 1 if args.script == "calibrate" else max(1, args.rounds), rng,
                             args.ready_s, args.trial_s, args.rest_s)
    elif args.script == "negatives":
        script = script_negatives(st, args.negatives_scale)
    else:
        script = script_trials(st, gestures, reps, rng, args.ready_s, args.trial_s, args.rest_s)
    status = 0
    try:
        while not st.quit:
            st.pump()
            try:
                next(script)
            except StopIteration:
                break
    except KeyboardInterrupt:
        print("\ninterrupted: keeping what was recorded so far")
    except AudioError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        status = 1
    finally:
        rec.stop()
        st.close()
        if st.video is not None:
            st.video.release()
        if camera is not None:
            camera.stop()
        if tracker is not None:
            tracker.close()
        screen.close()

    # ---- labels: prompts + what the camera saw, lined up with the audio ----
    result = None
    sync = None
    key = f"{cam_info.get('backend')}:{cam_info.get('api', '')}:{cam_info.get('index', '')}|{getattr(backend, 'info', {}).get('input', args.backend)}"
    if camera is not None and st.frames:
        prior = load_sync_prior(out, key)
        header = landmark_header(camera.width, camera.height, cfg.fs, hfov_deg=args.hfov)
        result = label_session(st.frames, header, rec.activity_curve(), intents_of(st.prompts, cfg.fs), label_cfg, prior)
        sync = {**result.sync.to_dict(), "prior_ms": round(1e3 * prior, 1), "audio_clock": rec.clock.log[:600],
                "meaning": "offset_ms = camera time stamps minus audio time; already removed from the event times"}
        if result.sync.method == "motion":
            save_sync(out, key, session_id, result.sync.to_dict())
    events = build_events(st.prompts, result, cfg.fs, rec.n_samples)
    labels = summarize(events, result, rec.bursts, rec.cfg, camera_on=result is not None)
    rec.events = events
    rec.meta.update(prompts=st.prompts, labels=labels, **({"sync": sync} if sync is not None else {}))
    rec.meta["camera"].update(frames=len(st.frames), dropped=int(getattr(camera, "n_dropped", 0)) if camera is not None else 0,
                              lost_during_session=bool(st.camera_lost))
    rec.save_meta()
    print(f"\nsaved {rec.n_samples / cfg.fs:.0f} s, {len(rec.dropouts)} audio dropouts, {len(st.frames)} camera pictures -> {folder}")
    print_summary(labels, sync)
    return status


if __name__ == "__main__":
    raise SystemExit(main())

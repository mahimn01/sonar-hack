"""mocap_extract.py — gesture videos -> a library of real hand paths for the simulator.

What it does
    Runs Google's MediaPipe hand tracker over short video clips of people gesturing at a camera
    (a public data set, or clips recorded with the demo laptop's own webcam), turns each clip
    into the path of the palm in metres (motion.py), keeps the clips whose path really shows
    the gesture, and writes them into one small file: the motion library.
    `python -m datagen.generate --motion <library>` then lets a share of the synthetic gestures
    replay these real movements instead of the formula strokes.

Why
    The formula strokes are too tidy. Real hands hesitate, curve and overshoot; a model that
    has seen that variety in training copes better with strangers.

How to run (needs its own small environment, because MediaPipe pins older libraries):
    uv venv --python 3.12 .venv-mocap
    uv pip install --python .venv-mocap/bin/python mediapipe opencv-python-headless numpy scipy
    .venv-mocap/bin/python -m datagen.mocap_extract --clips clips.csv --out datagen/motion/library.npz

    clips.csv has one clip per line:  path,label[,subject[,first_frame,last_frame[,fps]]]
      path   a video file, or a folder of numbered frame pictures
      label  what the clip shows: swipe (either way), swipe_left, swipe_right, push, pull, other
             (other = any hand movement that is NOT one of our gestures; used for "none" clips).
             Which way a swipe goes is read off the tracked path, in the USER'S view (the camera
             picture is treated as a webcam's: the user's right is on the picture's left).
    --mirror-video if the videos were saved mirrored (selfie view).

What it changes
    Writes the library file, a `tracks/` folder next to it (the raw tracker output per clip, so
    the conversion can be repeated without tracking again), and `<library>.report.json`. On first
    use it downloads the tracker model (7.8 MB, Apache 2.0) next to the library. Reads the videos only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def read_frames(path: Path, first: int | None, last: int | None):
    """Yield RGB frames of a video file or of a folder of numbered pictures; also returns the file's own fps."""
    import cv2

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
        files = files[(first or 0):(None if last is None else last + 1)]
        return (cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files), None

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    def frames():
        i = first or 0
        while last is None or i <= last:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()

    return frames(), fps


def track_clip(landmarker, frames, fps: float, mirror: bool) -> dict | None:
    """MediaPipe over one clip: 21 hand points per frame, in the picture and in metres around the hand."""
    import mediapipe as mp

    image, world, valid, size = [], [], [], None
    for i, rgb in enumerate(frames):
        if mirror:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
        size = (rgb.shape[1], rgb.shape[0])
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(round(1000.0 * i / fps)))
        if res.hand_landmarks:
            image.append([[p.x, p.y] for p in res.hand_landmarks[0]])
            world.append([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]])
            valid.append(True)
        else:
            image.append([[0.0, 0.0]] * 21)
            world.append([[0.0, 0.0, 0.0]] * 21)
            valid.append(False)
    if size is None:
        return None
    return {"image": np.asarray(image, np.float32), "world": np.asarray(world, np.float32), "valid": np.asarray(valid, bool),
            "fps": float(fps), "width": size[0], "height": size[1]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gesture videos -> motion library (see the file header)")
    ap.add_argument("--clips", help="CSV: path,label[,subject[,first_frame,last_frame[,fps]]]")
    ap.add_argument("--unica", help="instead of --clips: the folder holding hand_gesture_dataset_videos/ of the University of "
                                    "Cagliari set (Zenodo 16040322, CC BY 4.0); classes 01/02 = sideways, 18/19 = push/pull, rest = other")
    ap.add_argument("--out", required=True, help="library file to write (.npz)")
    ap.add_argument("--root", default="", help="folder the paths in the CSV are relative to")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate when the clip does not say (frame folders)")
    ap.add_argument("--fov", type=float, default=65.0, help="horizontal field of view of the camera in degrees")
    ap.add_argument("--mirror-video", action="store_true", help="the videos are mirrored (selfie view)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many clips (0 = all)")
    ap.add_argument("--retrack", action="store_true", help="run the tracker again even where a saved track exists")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from datagen.motion import Camera, MotionLibrary, track_to_motion

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tracks_dir = out.parent / "tracks"
    tracks_dir.mkdir(exist_ok=True)
    rows = []
    if args.unica:
        kinds = {1: "swipe", 2: "swipe", 18: "radial", 19: "radial"}      # which way exactly is read off the tracked path
        args.root = args.unica
        timing: dict = {}                                                 # (class, user) -> [(first, last)] x 3
        stats = next(Path(args.unica).rglob("hand_gesture_timing_stats.csv"), None)
        if stats is not None:
            with open(stats, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    timing[(int(r["class"]), int(r["user"]))] = [(int(float(r[f"start_frame_{k}"])), int(float(r[f"end_frame_{k}"]))) for k in (1, 2, 3)]
        for video in sorted(Path(args.unica).rglob("class_*/User*_/*.avi")):
            cls, user = int(video.parts[-3].split("_")[1]), video.parts[-2].strip("_")
            row = [str(video.relative_to(args.unica)), kinds.get(cls, "other"), f"unica-{user}"]
            spans = timing.get((cls, int(user.replace("User", ""))))
            rep_no = int(video.stem.split("_")[-1])
            if spans and 1 <= rep_no <= 3:                                # the gesture itself with a third of a second around it
                row += [max(0, spans[rep_no - 1][0] - 10), spans[rep_no - 1][1] + 10, 30]
            rows.append(row)
    elif args.clips:
        with open(args.clips, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if row and not row[0].startswith("#") and row[0] != "path":
                    rows.append(row)
    else:
        ap.error("give --clips or --unica")
    if args.limit:
        rows = rows[:args.limit]

    landmarker = None

    def tracker():
        nonlocal landmarker
        if landmarker is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model = out.parent / "hand_landmarker.task"
            if not model.exists():
                print(f"downloading the hand tracker model to {model}")
                urllib.request.urlretrieve(MODEL_URL, model)
            landmarker = (vision, mp_python.BaseOptions(model_asset_path=str(model)))
        vision, base = landmarker
        opts = vision.HandLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.VIDEO, num_hands=1,
                                            min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4,
                                            min_tracking_confidence=0.4)
        return vision.HandLandmarker.create_from_options(opts)          # a fresh one per clip: its clock restarts at 0

    clips, dropped = [], {"no hand found": 0, "hand lost too often": 0, "path does not show the gesture": 0, "unreadable": 0}
    for n, row in enumerate(rows):
        path, label = Path(args.root) / row[0], row[1].strip()
        subject = row[2].strip() if len(row) > 2 else ""
        first = int(row[3]) if len(row) > 3 and row[3] != "" else None
        last = int(row[4]) if len(row) > 4 and row[4] != "" else None
        clip_id = f"{row[0]}:{'' if first is None else first}-{'' if last is None else last}".replace("/", "_").replace("\\", "_")
        saved = tracks_dir / f"{clip_id}.npz"
        if saved.exists() and not args.retrack:
            with np.load(saved) as z:
                tr = {k: z[k] for k in z.files}
            tr = {**tr, "fps": float(tr["fps"]), "width": int(tr["width"]), "height": int(tr["height"])}
        else:
            try:
                frames, file_fps = read_frames(path, first, last)
                fps = float(row[5]) if len(row) > 5 and row[5] != "" else (file_fps or args.fps)
                with tracker() as lm:
                    tr = track_clip(lm, frames, fps, args.mirror_video)
            except Exception as exc:                                    # a broken file must not stop the run
                print(f"  {row[0]}: skipped ({type(exc).__name__}: {str(exc)[:80]})")
                tr = None
            if tr is None:
                dropped["unreadable"] += 1
                continue
            np.savez_compressed(saved, **tr)
        if not tr["valid"].any():
            dropped["no hand found"] += 1
            continue
        cam = Camera(int(tr["width"]), int(tr["height"]), fov_deg=args.fov)
        motion = track_to_motion(tr["image"], tr["world"], tr["fps"], cam, tr["valid"], hint=label)
        if motion is None:
            dropped["path does not show the gesture"] += 1
            continue
        motion.update(source=clip_id, subject=subject)
        clips.append(motion)
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(rows)} clips tracked, {len(clips)} kept")

    if not clips:
        print("no usable clip: nothing written", file=sys.stderr)
        return 1
    lib = MotionLibrary(clips)
    lib.save(out)
    report = {"clips_in": len(rows), "kept": len(lib), "per_label": lib.counts(), "dropped": dropped,
              "subjects": len({c["subject"] for c in clips}),
              "stroke_seconds_median": float(np.median([(c["stroke"][1] - c["stroke"][0]) / c["fps"] for c in clips if c["label"] != "other"] or [0])),
              "travel_m_median": float(np.median([c["travel_m"] for c in clips if c["label"] != "other"] or [0])),
              "peak_speed_median": float(np.median([c["peak_speed"] for c in clips if c["label"] != "other"] or [0]))}
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {out}: {len(lib)} hand paths {lib.counts()} from {len(rows)} clips; dropped {dropped}")
    print(f"typical stroke: {report['stroke_seconds_median']:.2f} s, {report['travel_m_median']:.2f} m, peak {report['peak_speed_median']:.2f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

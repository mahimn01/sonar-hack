"""weaklabel.py — labels with a confidence for everything the plain dataset builder leaves out.

Plain words: a recording is cut into "bursts" (moments the sonar saw movement). For each burst
several witnesses may have an opinion:

    the prompt     what the screen asked for at that moment (a gesture block, a "no gestures" part)
    the camera     what the hand tracker saw the hand do (camrecord.py sessions)
    the physics    the training-free Doppler rules (gestures.heuristic_classify)
    the model      the current trained model, if there is one

This module (1) puts all of that in one table per session (`build_tables`), which the label QA
(labelqa.py) also uses, and (2) turns the tables into EXTRA training examples that the plain
builder (dataset.py) does not make, each with a weight between 0 and about 2:

    pseudo     a burst nobody labelled, where the model and the physics rules agree it is a gesture
               (unlabelled continuous recordings; gesture blocks where the camera lost the hand)
    rescued    a camera "review" reading that the model or the physics rules confirm
    aux        a hand movement the camera saw that was clearly NOT a gesture (slow, short or
               diagonal, or the hand dropping back): a real negative from inside a gesture session
    hardneg    HARD NEGATIVES: the model fires although no gesture can have happened there
               (typing, reaching, somebody walking past, in a part recorded as gesture-free)
    retimed    a prompt window whose LOUDEST burst (the one the plain builder takes) is not read as
               the prompted gesture, while another burst of the same window is: probably the stroke,
               where the loudest one was the hand going back

Only confident ones are kept. The plain examples (prompted and camera-timed gestures, ordinary
"none" bursts) are NOT repeated here, so this file's output is added to the usual data, never
swapped for it. `sonar.train` uses the `weight` column when it is present.

How it is made fast (it runs on the training box inside scripts/train_4090.py --weak):
  step 1  per session: audio -> Doppler pictures -> bursts. With a CUDA GPU and the GPU front
          end (sonar/features_torch.py) the signal processing runs on the GPU, one session after
          the other, while worker processes finish the bursts; without it, one whole session per
          CPU core. Results are cached per session (--cache), so a re-run only does new sessions,
          and `--tables-only` lets train_4090.py do this step while the GPU pre-trains.
  step 2  the pictures of ALL bursts of ALL sessions go to the device once and both witnesses
          judge them there in one batched pass (batch_judge.py). No per-burst Python.
  step 3  the rules below turn the verdicts into labels (plain bookkeeping, milliseconds).
`--parity` compares the fast path with the original one-burst-at-a-time CPU code on your sessions.

How to run:
    python -m sonar.weaklabel data/ --model models/gesture.onnx --out train_out/weak.npz
    python -m sonar.weaklabel data/ --unlabelled loose_recordings/ --out weak.npz --hard-negatives hard_negatives.npz --report weak.json
    python -m sonar.weaklabel data/ --out weak.npz --exclude-person guest3,guest7      (keep held-out people out)
    python -m sonar.weaklabel data/ --parity                                           (fast path = original code?)
What it changes: writes the .npz / .json files you name and the cache folder. Reads the sessions only.

Output .npz: the dataset format of dataset.py plus `weight` [N] float32. `source` holds the kind
of example (pseudo / rescued / aux / hardneg).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .batch_judge import BatchJudge, pick_device
from .config import GESTURE_LABELS, SCENE_LABELS, SonarConfig
from .dataset import PAD, SESSION_FILE, Example, _feature_key, center_crop, find_sessions, load_session, pack
from .features import StreamFeatures, stream_features

GESTURE_FREE_SCRIPTS = ("scene", "breathing", "negatives")     # sessions in which nobody was asked to gesture
OPPOSITE = {"swipe_left": "swipe_right", "swipe_right": "swipe_left", "push": "pull", "pull": "push"}
# code whose change makes a cached burst set stale
_FEATURE_CODE = ("config.py", "tx.py", "dsp.py", "frontend.py", "gestures.py", "features.py", "pipeline.py", "features_torch.py")


# ---------------------------------------------------------------------------
# Step 1. The bursts of a session (the heavy part: signal processing)
# ---------------------------------------------------------------------------
@dataclass
class BurstSet:
    """Every movement burst of one recording with its picture, as the engine would cut it."""

    cfg: SonarConfig
    n_frames: int
    seg: np.ndarray               # [n, 5] int64: start, end, decided (frame indices), forced (0/1), not_before (-1 = none)
    peak_db: np.ndarray           # [n] float32
    x: np.ndarray                 # [n, C, F, T + 2*PAD] float16: the padded picture of every burst
    activity_db: np.ndarray       # [n_frames] float32: the movement curve
    how: str = "cpu"              # which code made it: "cpu" (the engine's pipeline) | "gpu front end" | "cache"
    seconds: float = 0.0
    stream: StreamFeatures | None = None      # the whole feature stream, only when asked for (label QA pictures)

    def __len__(self) -> int:
        return len(self.seg)

    def frame_of_sample(self, sample: int) -> int:
        return int(sample // self.cfg.frame_len) + 1

    def windows(self) -> np.ndarray:
        """[n, C, F, T]: exactly the pictures the engine classifies."""
        return center_crop(self.x, self.cfg, PAD)


def windows_from_stream(stream: StreamFeatures, pad: int = PAD) -> np.ndarray:
    """features.window_from_stream for every burst of a stream at once: [n, C, F, T + 2*pad] float16."""
    cfg = stream.cfg
    n_b, width = len(stream.segments), cfg.gesture_frames + 2 * pad
    if n_b == 0 or stream.n_frames == 0:
        return np.zeros((0, cfg.n_feat_channels, cfg.doppler_bins, width), np.float16)
    ends = np.array([s.decided for s in stream.segments], dtype=np.int64)
    not_before = np.array([-1 if nb is None else nb for nb in stream.not_before], dtype=np.int64)
    cols = ends[:, None] - (cfg.gesture_frames - 1 + pad) + np.arange(width)[None, :]        # frame index of every column
    valid = (cols >= 1) & (cols <= stream.n_frames) & (cols >= not_before[:, None])
    x = stream.feat[np.clip(cols - 1, 0, stream.n_frames - 1)] * valid[:, :, None, None]     # [n, W, C, F]
    return np.ascontiguousarray(x.transpose(0, 2, 3, 1)).astype(np.float16)


def burst_set_of(stream: StreamFeatures, how: str, seconds: float, keep_stream: bool = False) -> BurstSet:
    seg = np.array([[s.start, s.end, s.decided, int(s.forced), -1 if nb is None else nb]
                    for s, nb in zip(stream.segments, stream.not_before)], dtype=np.int64).reshape(-1, 5)
    return BurstSet(stream.cfg, stream.n_frames, seg, np.array([s.peak_db for s in stream.segments], dtype=np.float32),
                    windows_from_stream(stream), np.asarray(stream.activity_db, dtype=np.float32), how, seconds,
                    stream if keep_stream else None)


def code_key() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in _FEATURE_CODE:
        if (here / name).exists():
            h.update(name.encode())
            h.update((here / name).read_bytes())
    return h.hexdigest()[:12]


def cache_path(cache: Path, folder: Path, session: str, audio_name: str = "audio.wav") -> Path:
    audio = folder / audio_name                    # a full path in session.json (a view made by dataview.py) wins over the folder
    stat = audio.stat() if audio.exists() else None
    key = hashlib.sha256(json.dumps([code_key(), stat.st_size if stat else 0, int(stat.st_mtime) if stat else 0,
                                     os.environ.get("SONAR_SEG_REL_OFF_DB", "")]).encode()).hexdigest()[:12]
    return cache / f"{session}__{key}.npz"


def save_burst_set(path: Path, bs: BurstSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, seg=bs.seg, peak_db=bs.peak_db, x=bs.x, activity_db=bs.activity_db, n_frames=np.array(bs.n_frames),
             config=np.array(json.dumps(bs.cfg.to_dict())), how=np.array(bs.how))
    tmp.replace(path)


def load_burst_set(path: Path) -> BurstSet:
    with np.load(path, allow_pickle=False) as z:
        return BurstSet(SonarConfig.from_dict(json.loads(str(z["config"]))), int(z["n_frames"]), z["seg"], z["peak_db"], z["x"],
                        z["activity_db"], "cache")


def _torch_frontend(cfg: SonarConfig, n_mics: int, device: Any, frontend: str) -> Any:
    """The batched GPU front end (sonar/features_torch.py), when this checkout has it and it is wanted."""
    if frontend == "cpu" or device is None or (frontend == "auto" and device.type != "cuda"):
        return None
    try:
        from .features import stream_from_columns  # noqa: F401
        from .features_torch import TorchFrontend
    except ImportError:
        return None
    return TorchFrontend(cfg, n_mics, device)


def _cpu_job(folder: str, cache_file: str | None) -> tuple[BurstSet | None, str]:
    """Worker process: one whole session through the engine's CPU pipeline."""
    try:
        t0 = time.perf_counter()
        _, audio, cfg = load_session(folder)
        bs = burst_set_of(stream_features(audio, cfg), "cpu", 0.0)
        bs.seconds = time.perf_counter() - t0
        if cache_file:
            save_burst_set(Path(cache_file), bs)
        return bs, ""
    except (OSError, RuntimeError, ValueError, KeyError, IndexError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _segment_job(cfg_dict: dict[str, Any], doppler_db: np.ndarray, lateral: np.ndarray | None, cache_file: str | None,
                 seconds: float) -> tuple[BurstSet | None, str]:
    """Worker process: GPU-made Doppler columns -> bursts, with the engine's own normaliser and segmenter."""
    try:
        from .features import stream_from_columns

        t0 = time.perf_counter()
        bs = burst_set_of(stream_from_columns(SonarConfig.from_dict(cfg_dict), doppler_db, lateral), "gpu front end", 0.0)
        bs.seconds = seconds + time.perf_counter() - t0
        if cache_file:
            save_burst_set(Path(cache_file), bs)
        return bs, ""
    except (OSError, RuntimeError, ValueError, KeyError, IndexError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def all_bursts(folders: list[Path], device: Any = None, cache: Path | None = None, workers: int = 1, frontend: str = "auto",
               keep_stream: bool = False) -> list[tuple[BurstSet | None, str]]:
    """Step 1 for many sessions: (burst set or None, why not) per folder, in the same order."""
    out: list[tuple[BurstSet | None, str] | None] = [None] * len(folders)
    todo: list[int] = []
    files: dict[int, str | None] = {}
    for i, folder in enumerate(folders):
        files[i] = None
        if cache is not None and not keep_stream:
            session, audio_name = folder.name, "audio.wav"
            try:
                meta = json.loads((folder / SESSION_FILE).read_text())
                session, audio_name = meta.get("session_id", folder.name), str(meta.get("audio", "audio.wav"))
            except (OSError, ValueError):
                pass
            path = cache_path(cache, folder, str(session), audio_name)
            files[i] = str(path)
            if path.exists():
                try:
                    out[i] = (load_burst_set(path), "")
                    continue
                except (OSError, ValueError, KeyError):
                    pass
        todo.append(i)
    if keep_stream:                                         # label QA: few sessions, the whole stream is wanted
        for i in todo:
            try:
                t0 = time.perf_counter()
                _, audio, cfg = load_session(folders[i])
                out[i] = (burst_set_of(stream_features(audio, cfg), "cpu", time.perf_counter() - t0, keep_stream=True), "")
            except (OSError, RuntimeError, ValueError, KeyError, IndexError) as exc:
                out[i] = (None, f"{type(exc).__name__}: {exc}")
        todo = []
    if todo:
        workers = max(1, min(workers, len(todo)))
        pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
        futures: dict[int, Any] = {}
        fronts: dict[tuple, Any] = {}
        try:
            for i in todo:
                fe = None
                audio = None
                if frontend != "cpu" and device is not None and (frontend == "torch" or device.type == "cuda"):
                    try:
                        _, audio, cfg = load_session(folders[i])
                        key = (_feature_key(cfg.to_dict()), audio.shape[1])
                        if key not in fronts:
                            fronts[key] = _torch_frontend(cfg, audio.shape[1], device, frontend)
                        fe = fronts[key]
                    except (OSError, RuntimeError, ValueError, KeyError) as exc:
                        out[i] = (None, f"{type(exc).__name__}: {exc}")
                        continue
                if fe is not None:
                    import torch

                    t0 = time.perf_counter()
                    db, lat = fe.columns(torch.from_numpy(np.ascontiguousarray(audio))[None])      # the signal processing, on the GPU
                    args = (fe.cfg.to_dict(), db[0].cpu().numpy(), None if lat is None else lat[0].cpu().numpy(), files[i],
                            time.perf_counter() - t0)
                    futures[i] = pool.submit(_segment_job, *args) if pool else _segment_job(*args)
                else:
                    futures[i] = pool.submit(_cpu_job, str(folders[i]), files[i]) if pool else _cpu_job(str(folders[i]), files[i])
            for i, fut in futures.items():
                out[i] = fut.result() if pool else fut
        finally:
            if pool is not None:
                pool.shutdown()
    return [o if o is not None else (None, "not processed") for o in out]


# ---------------------------------------------------------------------------
# One table per session: every burst and every witness
# ---------------------------------------------------------------------------
@dataclass
class Burst:
    k: int
    start: int                  # frame indices (20 ms each), as in gestures.Segment
    end: int
    decided: int
    peak_db: float
    forced: bool
    role: str = "none"          # what the plain builder does with it: "gesture" | "none" | "aux" | "excluded"
    label: str = "none"         # the plain builder's label ("none" unless role is "gesture")
    label_source: str = ""      # "camera" | "prompt"
    label_conf: float = 0.0
    event: int | None = None    # index into session.json's events
    trial: int | None = None    # the gesture event (prompt window or camera stroke) this burst lies in, chosen or not
    zone: str = "unlabelled"    # what the session says about this moment (see zone_of)
    cam: str = ""               # the camera's reading of the hand: a gesture name, "return", "other", "review"
    cam_phys: str = ""          # the stroke on its own: swipe_left | swipe_right | push | away | other
    cam_conf: float = 0.0
    hand_seen: float | None = None     # share of the camera pictures during the burst that showed a hand (None: no camera)
    physics: str = "none"
    physics_conf: float = 0.0
    model: str = ""             # "" when no trained model was given
    model_conf: float = 0.0
    model_probs: dict[str, float] = field(default_factory=dict)

    @property
    def t0(self) -> float:
        return self.start * 0.02

    @property
    def t1(self) -> float:
        return self.end * 0.02


@dataclass
class SessionTable:
    folder: Path
    meta: dict[str, Any]
    cfg: SonarConfig
    bset: BurstSet
    bursts: list[Burst]
    missed: list[dict[str, Any]]            # gesture events for which the sonar saw no movement at all
    classifier: str                          # "model" | "physics"
    model_labels: tuple[str, ...] = ()

    @property
    def stream(self) -> StreamFeatures:
        if self.bset.stream is None:
            raise ValueError("this table was built without the feature stream (keep_stream=False)")
        return self.bset.stream

    @property
    def session(self) -> str:
        return str(self.meta.get("session_id", self.folder.name))

    @property
    def person(self) -> str:
        return str(self.meta.get("person", "unknown"))


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return min(a1, b1) - max(a0, b0) + 1


def assign_roles(events: list[dict[str, Any]], bs: BurstSet) -> tuple[list[Burst], list[dict[str, Any]]]:
    """Which burst the plain builder (dataset.session_examples) uses for what. Kept in step with it by a test."""
    guard = int(1.0 * bs.cfg.frame_rate)
    start, end = bs.seg[:, 0], bs.seg[:, 1]
    bursts = [Burst(k, int(s[0]), int(s[1]), int(s[2]), float(p), bool(s[3])) for k, (s, p) in enumerate(zip(bs.seg, bs.peak_db))]
    windows: list[tuple[int, int]] = []
    used = np.zeros(len(bursts), dtype=bool)
    missed: list[dict[str, Any]] = []
    for i, ev in enumerate(events):
        f0, f1 = bs.frame_of_sample(ev["start"]), bs.frame_of_sample(ev["end"])
        if ev.get("exclude"):
            windows.append((f0, f1))
            continue
        if ev["label"] not in GESTURE_LABELS or ev["label"] == "none":
            continue
        windows.append((f0 - guard, f1 + 2 * guard))
        inside = np.flatnonzero((end >= f0 - 5) & (start <= f1 + 25))
        if ev.get("tight"):                                                # the burst that covers the stroke; ties: the stronger one
            cover = np.minimum(end[inside], f1 + 3) - np.maximum(start[inside], f0 - 1)
            inside, cover = inside[cover >= 0], cover[cover >= 0]
            best_at = max(range(len(inside)), key=lambda j: (int(cover[j]), float(bs.peak_db[inside[j]]))) if len(inside) else -1
        else:                                                              # the loudest burst in the prompt window
            best_at = int(np.argmax(bs.peak_db[inside])) if len(inside) else -1
        if best_at < 0:
            missed.append({**ev, "index": i})
            continue
        best = int(inside[best_at])
        used[inside] = True
        for k in inside:
            bursts[int(k)].trial = i
        b = bursts[best]
        b.role, b.label, b.event = "gesture", ev["label"], i
        b.label_source = "camera" if ev.get("source") == "camera" else "prompt"
        b.label_conf = float(ev.get("confidence", 1.0))
        for k in inside:
            if k != best and bursts[int(k)].role != "gesture":
                bursts[int(k)].role = "aux"
    for k, b in enumerate(bursts):
        if b.role == "none" and (used[k] or any(a <= b.end and b.start <= z for a, z in windows)):
            b.role = "excluded"
    return bursts, missed


def zone_of(sample: int, meta: dict[str, Any]) -> str:
    """What the session says about one moment.

    "free:<what>"    a part in which no gesture was asked for (calibration, typing, walking in, an empty room ...)
    "block:<g>"      a free-form block of gesture g (camrecord): gestures happen here, also unseen ones
    "trial:<g>"      a timed prompt window for gesture g
    "between"        a gesture session, between two prompts
    "unlabelled"     a recording with no labels at all
    """
    has_gestures = False
    hit = ""
    for ev in meta.get("events", []):
        kind, label = ev.get("kind", ""), ev.get("label", "")
        is_gesture = label in GESTURE_LABELS and label != "none"
        has_gestures |= is_gesture or kind in ("block", "prompt")
        if not (ev["start"] <= sample <= ev["end"]):
            continue
        if kind == "block":
            hit = f"block:{ev.get('intent', '')}"
        elif kind == "prompt":
            hit = hit or f"trial:{ev.get('intent', '')}"
        elif is_gesture and not ev.get("tight"):
            hit = hit or f"trial:{label}"
        elif kind == "calibration" or label in SCENE_LABELS:
            hit = hit or f"free:{ev.get('activity') or label}"
    if hit:
        return hit
    if str(meta.get("script", "")) in GESTURE_FREE_SCRIPTS:
        return f"free:{meta.get('script')}"
    return "between" if has_gestures else "unlabelled"


def _camera_side(folder: Path, meta: dict[str, Any], bs: BurstSet, bursts: list[Burst]) -> None:
    """Fill in what the camera saw during each burst (camrecord sessions only)."""
    cfg = bs.cfg
    cam_events = [ev for ev in meta.get("events", []) if ev.get("source") == "camera"]
    spans = [(bs.frame_of_sample(ev["start"]) - 1, bs.frame_of_sample(ev["end"]) + 3) for ev in cam_events]
    frames_t = frames_hand = None
    lm_path = folder / "landmarks.jsonl"
    if lm_path.exists():
        from .camlabel import load_landmarks

        _, frames = load_landmarks(lm_path)
        offset = float(meta.get("sync", {}).get("offset_ms", 0.0)) / 1e3
        frames_t = np.array([f.sample / cfg.fs - offset for f in frames])
        frames_hand = np.array([bool(f.hands) for f in frames])
    for b in bursts:
        best = 0
        for ev, (f0, f1) in zip(cam_events, spans):               # the camera stroke that overlaps this burst most
            ov = _overlap(b.start, b.end, f0, f1)
            if ov <= 0:
                continue
            if ev.get("kind") == "gesture":
                ov += 1000                                        # a gesture inside the burst outranks the hand going back
            if ov > best:
                best = ov
                b.cam = ev["label"] if ev.get("kind") != "review" else "review"
                b.cam_phys = str(ev.get("phys") or ev.get("candidate") or ev["label"])
                b.cam_conf = float(ev.get("confidence", 0.0))
        if frames_t is not None and frames_hand is not None and len(frames_t):
            sel = (frames_t >= b.t0 - 0.06) & (frames_t <= b.t1 + 0.02)
            b.hand_seen = float(frames_hand[sel].mean()) if sel.any() else 0.0


def build_tables(folders: list[Path], model: str | Path | None = None, device: Any = "auto", cache: Path | None = None,
                 workers: int = 1, frontend: str = "auto", keep_stream: bool = False, compile_model: bool = False,
                 ) -> tuple[list[SessionTable], list[str], dict[str, Any]]:
    """(one table per readable session, the sessions that were skipped and why, timings and how it was computed).

    All bursts of all sessions are judged in ONE batched pass per signal profile.
    """
    dev = pick_device(device) if (device is None or isinstance(device, str)) else device
    skipped: list[str] = []
    t0 = time.perf_counter()
    sets = all_bursts(folders, dev, cache, workers, frontend, keep_stream)
    info: dict[str, Any] = {"seconds": {"bursts": time.perf_counter() - t0}, "how": {}, "audio_minutes": 0.0, "judge": "", "judge_seconds": {}}
    ready: list[tuple[Path, dict[str, Any], BurstSet]] = []
    for folder, (bs, why) in zip(folders, sets):
        if bs is None:
            skipped.append(f"{folder}: {why}")
            continue
        try:
            meta = json.loads((folder / SESSION_FILE).read_text())
        except (OSError, ValueError) as exc:
            skipped.append(f"{folder}: cannot read session.json ({type(exc).__name__})")
            continue
        info["how"][bs.how] = info["how"].get(bs.how, 0) + 1
        info["audio_minutes"] += bs.n_frames / bs.cfg.frame_rate / 60.0
        ready.append((folder, meta, bs))
    groups: dict[tuple, list[int]] = {}
    for i, (_, _, bs) in enumerate(ready):
        groups.setdefault(_feature_key(bs.cfg.to_dict()), []).append(i)
    t_judge = 0.0
    slots: dict[int, SessionTable] = {}
    for members in groups.values():
        cfg = ready[members[0]][2].cfg
        judge = BatchJudge(cfg, model, dev, compile_model=compile_model)
        if judge.note and model:
            print(f"NOTE: {judge.note}", file=sys.stderr)
        x_all = np.concatenate([ready[i][2].windows() for i in members])
        t1 = time.perf_counter()
        verdicts = judge(x_all)                                   # every burst of every session, one batched pass
        t_judge += time.perf_counter() - t1
        info["judge"] = judge.describe()
        info["judge_seconds"].update({k: round(v, 4) for k, v in verdicts.seconds.items()})
        at = 0
        for i in members:
            folder, meta, bs = ready[i]
            bursts, missed = assign_roles(meta.get("events", []), bs)
            for b in bursts:
                j = at + b.k
                b.physics, b.physics_conf = GESTURE_LABELS[int(verdicts.physics[j])], float(verdicts.physics_conf[j])
                if verdicts.model is not None and verdicts.model_conf is not None and verdicts.model_probs is not None:
                    b.model, b.model_conf = verdicts.labels[int(verdicts.model[j])], float(verdicts.model_conf[j])
                    b.model_probs = {name: round(float(p), 4) for name, p in zip(verdicts.labels, verdicts.model_probs[j])}
                b.zone = zone_of(int((b.start + b.end) // 2 * bs.cfg.frame_len), meta)
            at += len(bs)
            _camera_side(folder, meta, bs, bursts)
            slots[i] = SessionTable(folder, meta, bs.cfg, bs, bursts, missed, "model" if judge.has_model else "physics",
                                    judge.labels if judge.has_model else ())
    tables = [slots[i] for i in sorted(slots)]
    info["seconds"]["verdicts"] = t_judge
    info["bursts"] = int(sum(len(t.bursts) for t in tables))
    return tables, skipped, info


def session_table(folder: str | Path, model: str | Path | None = None, device: Any = "auto", keep_stream: bool = False) -> SessionTable:
    """The table of one session (raises ValueError if it cannot be read)."""
    tables, skipped, _ = build_tables([Path(folder)], model, device, keep_stream=keep_stream)
    if not tables:
        raise ValueError(skipped[0] if skipped else f"{folder}: no table")
    return tables[0]


# ---------------------------------------------------------------------------
# Step 3. From the tables to extra weighted examples
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeakConfig:
    min_confidence: float = 0.6     # the engine's threshold: above it the model "fires"
    pseudo_conf: float = 0.90       # the model must be at least this sure before its word is taken as a label
    physics_conf: float = 0.50      # ... and the physics rules must name the same gesture with at least this
    pseudo_weight: float = 0.6      # a pseudo-label never counts as much as a real one
    rescue_weight: float = 0.7
    retime_weight: float = 0.6
    aux_weight: float = 0.6
    hardneg_weight: float = 2.0     # hard negatives are drawn twice as often as an ordinary "none"
    pseudo_none: bool = True        # also keep unlabelled bursts that model AND physics call "none" (weight 0.3)
    min_peak_db: float = 3.0


@dataclass
class WeakLabel:
    k: int
    label: str
    weight: float
    tag: str                 # pseudo | rescued | aux | hardneg | retimed
    why: str


def fires(b: Burst, min_confidence: float, use_model: bool) -> tuple[str, float]:
    """(gesture the classifier would report for this burst, its confidence), or ("none", ...)."""
    if use_model:
        return (b.model, b.model_conf) if b.model not in ("", "none") and b.model_conf >= min_confidence else ("none", b.model_conf)
    return (b.physics, b.physics_conf) if b.physics != "none" and b.physics_conf >= min_confidence else ("none", b.physics_conf)


def weak_labels(table: SessionTable, wc: WeakConfig | None = None, labels: tuple[str, ...] | None = None,
                may_hold_gestures: bool = False) -> list[WeakLabel]:
    """The extra examples of one session (see the top of this file). `labels`: gestures the model is trained for.

    may_hold_gestures: how to read a recording with no labels at all. False (a session in the training
    folder): the plain builder already uses all of it as "none", so whatever fires there is a hard
    negative. True (a recording given as --unlabelled): it may contain real gestures, so the model's
    confident answers become pseudo-labels instead.
    """
    wc = wc or WeakConfig()
    use_model = table.classifier == "model"
    wanted = set(labels) if labels else {g for g in GESTURE_LABELS if g != "none"}
    out: list[WeakLabel] = []
    bs = table.bset
    review = [(bs.frame_of_sample(ev["start"]) - 1, bs.frame_of_sample(ev["end"]) + 3, ev)
              for ev in table.meta.get("events", []) if ev.get("kind") == "review"]
    chosen = {b.event: b for b in table.bursts if b.role == "gesture"}

    def said(b: Burst) -> tuple[str, float]:
        return (b.model, b.model_conf) if use_model else (b.physics, b.physics_conf)

    for b in table.bursts:
        if b.role == "gesture" or b.peak_db < wc.min_peak_db:
            continue                                            # already a plain example / too faint to matter
        # (0) the prompt's loudest burst is not read as the prompted gesture, but this other burst of the same prompt is
        if b.role == "aux" and b.trial in chosen:
            top = chosen[b.trial]
            name, conf = said(b)
            if top.label in wanted and said(top)[0] != top.label and name == top.label and conf >= (0.7 if use_model else 0.6) \
                    and (not use_model or b.physics in (top.label, "none")):
                out.append(WeakLabel(b.k, top.label, wc.retime_weight, "retimed",
                                     f"the loudest burst of this prompt was read as {said(top)[0]}, this one as {name} ({conf:.2f})"))
            continue
        fired, fconf = fires(b, wc.min_confidence, use_model)
        agree = use_model and b.model in wanted and b.model == b.physics and b.model_conf >= wc.pseudo_conf \
            and b.physics_conf >= wc.physics_conf
        zone = b.zone
        if zone == "unlabelled" and not may_hold_gestures:
            zone = "free:a recording without labels"

        # (a) parts recorded as gesture-free: whatever fires there is a hard negative
        if zone.startswith("free:"):
            if b.cam == "review" or b.cam in wanted:
                continue                                        # the camera thinks a gesture happened after all: leave it alone
            if fired != "none":
                out.append(WeakLabel(b.k, "none", wc.hardneg_weight, "hardneg",
                                     f"{'model' if use_model else 'physics'} says {fired} ({fconf:.2f}) during {zone[5:]}"))
            continue

        # (b) a camera "review" reading: keep it if another witness confirms the camera's guess
        rev = next((ev for f0, f1, ev in review if _overlap(b.start, b.end, f0, f1) > 0), None)
        if rev is not None:
            cand = str(rev.get("candidate", ""))
            intent = zone.split(":", 1)[1] if ":" in zone else ""
            if cand in wanted and (not intent or intent == cand or intent == "none"):
                m_yes = use_model and b.model == cand and b.model_conf >= 0.7
                m_no = use_model and b.model in wanted and b.model != cand and b.model_conf >= 0.7
                p_yes = b.physics == cand and b.physics_conf >= wc.physics_conf
                p_no = b.physics in wanted and b.physics != cand and b.physics_conf >= 0.7
                if (m_yes and not p_no) or (p_yes and not m_no):
                    w = wc.rescue_weight * (1.0 if (m_yes and p_yes) else 0.75)
                    out.append(WeakLabel(b.k, cand, round(w, 3), "rescued",
                                         f"camera unsure about {cand} ({rev.get('confidence', 0):.2f}); "
                                         f"{'model ' if m_yes else ''}{'physics ' if p_yes else ''}agree"))
            continue

        # (c) inside a gesture block or between prompts: the camera saw a movement that is clearly not a gesture
        if b.role == "excluded" and b.cam in ("other", "return") and b.cam_phys in ("other", "away"):
            if b.forced or any(o.role == "gesture" and abs(o.k - b.k) == 1 and o.forced for o in table.bursts):
                continue                                        # cut out of one long movement together with a gesture: not clean
            w = wc.hardneg_weight * 0.75 if fired != "none" else wc.aux_weight
            tag = "hardneg" if fired != "none" else "aux"
            what = "the hand dropping back" if b.cam_phys == "away" else "a slow, short or diagonal movement"
            out.append(WeakLabel(b.k, "none", w, tag, f"camera saw {what}" + (f"; classifier said {fired} ({fconf:.2f})" if fired != "none" else "")))
            continue

        # (d) nobody labelled it: take the model's word only if the physics rules say the same
        unknown = zone == "unlabelled" or (b.role == "excluded" and zone.startswith("block:") and b.cam == "")
        if unknown and agree:
            intent = zone.split(":", 1)[1] if zone.startswith("block:") else ""
            if intent and intent != b.model:
                continue
            if b.hand_seen is not None and b.hand_seen > 0.8:
                continue                                        # the camera watched the hand the whole time and saw no stroke
            w = wc.pseudo_weight * float(np.clip((b.model_conf - wc.pseudo_conf) / max(1.0 - wc.pseudo_conf, 1e-6), 0.3, 1.0))
            out.append(WeakLabel(b.k, b.model, round(w, 3), "pseudo", f"model {b.model_conf:.2f} and physics {b.physics_conf:.2f} agree"))
        elif zone == "unlabelled" and wc.pseudo_none and use_model and b.model == "none" and b.model_conf >= wc.pseudo_conf \
                and b.physics == "none":
            out.append(WeakLabel(b.k, "none", 0.3, "pseudo", "model and physics both say none"))
    return out


def examples_of(table: SessionTable, found: list[WeakLabel]) -> tuple[list[Example], np.ndarray]:
    """Padded training pictures for the weak labels, cut exactly as dataset.py cuts them."""
    examples = [Example(table.bset.x[wl.k], wl.label, table.session, table.person, wl.tag, float(table.bset.peak_db[wl.k])) for wl in found]
    return examples, np.asarray([wl.weight for wl in found], dtype=np.float32)


def table_rows(table: SessionTable, found: list[WeakLabel] | None = None) -> list[dict[str, Any]]:
    """The table as plain rows (for reports)."""
    weak = {wl.k: wl for wl in found or []}
    rows = []
    for b in table.bursts:
        row = {"session": table.session, "person": table.person, "t0": round(b.t0, 2), "t1": round(b.t1, 2), "peak_db": round(b.peak_db, 1),
               "role": b.role, "label": b.label, "label_source": b.label_source, "label_conf": round(b.label_conf, 2), "zone": b.zone,
               "camera": b.cam, "camera_stroke": b.cam_phys, "hand_seen": None if b.hand_seen is None else round(b.hand_seen, 2),
               "physics": b.physics, "physics_conf": round(b.physics_conf, 2), "model": b.model, "model_conf": round(b.model_conf, 2)}
        if b.k in weak:
            row.update(weak_label=weak[b.k].label, weak_weight=weak[b.k].weight, weak_tag=weak[b.k].tag, weak_why=weak[b.k].why)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Whole folders
# ---------------------------------------------------------------------------
def pick_sessions(roots: list[str | Path], unlabelled: list[str | Path] | None, exclude_people: tuple[str, ...], profile: str | None,
                  allow_sim: bool, skipped: list[str]) -> list[tuple[Path, bool]]:
    """(session folder, may it hold unlabelled gestures) for everything that passes the filters. Reads session.json only."""
    chosen: list[tuple[Path, bool]] = []
    first_key: tuple | None = None
    for root, loose in [(r, False) for r in roots] + [(r, True) for r in unlabelled or []]:
        for folder in find_sessions(root):
            try:
                meta = json.loads((folder / SESSION_FILE).read_text())
                cfg = SonarConfig.from_dict(meta["config"])
                cfg = cfg.with_(n_mics=int(meta.get("n_mics", cfg.n_mics)))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                skipped.append(f"{folder}: cannot read it ({type(exc).__name__})")
                continue
            if str(meta.get("person", "unknown")) in exclude_people:
                continue
            if profile is not None and str(meta.get("profile", "default")) != profile:
                skipped.append(f"{folder}: recorded with profile {meta.get('profile')}, wanted {profile}")
                continue
            if meta.get("synthetic") and not allow_sim:
                skipped.append(f"{folder}: simulator recording")
                continue
            key = _feature_key(cfg.to_dict())
            first_key = first_key or key
            if key != first_key:
                skipped.append(f"{folder}: recorded with a different signal profile or microphone count")
                continue
            chosen.append((folder, loose))
    return chosen


def build_weak(roots: list[str | Path], model: str | Path | None = None, wc: WeakConfig | None = None,
               exclude_people: tuple[str, ...] = (), labels: tuple[str, ...] | None = None, verbose: bool = True,
               gesture_free: tuple[str, ...] = (), unlabelled: list[str | Path] | None = None,
               profile: str | None = None, allow_sim: bool = True, device: Any = "auto", cache: Path | None = None,
               workers: int = 1, frontend: str = "auto", compile_model: bool = False,
               ) -> tuple[dict[str, np.ndarray] | None, list[dict[str, Any]], dict[str, Any]]:
    """(dataset arrays with `weight`, or None if nothing was found; every table row; a summary).

    roots         the training folder(s): sessions the plain builder also reads
    unlabelled    folders of continuous recordings with no labels that MAY contain gestures (never put
                  those in the training folder: the plain builder would use all of them as "none")
    gesture_free  session names (or parts of names) known to contain no gesture at all, e.g. a recording
                  of the demo table while nobody presented: everything the model fires on is a hard negative
    """
    wc = wc or WeakConfig()
    summary: dict[str, Any] = {"sessions": 0, "skipped": [], "by_tag": {}, "classifier": "physics", "model": str(model) if model else None}
    chosen = pick_sessions(roots, unlabelled, exclude_people, profile, allow_sim, summary["skipped"])
    loose = {folder: flag for folder, flag in chosen}
    tables, skipped, info = build_tables([f for f, _ in chosen], model, device, cache, workers, frontend, compile_model=compile_model)
    summary["skipped"] += skipped
    t0 = time.perf_counter()
    examples: list[Example] = []
    weights: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    cfg_json = ""
    for table in tables:
        cfg_json = cfg_json or json.dumps(table.cfg.to_dict(), sort_keys=True)
        summary["classifier"] = table.classifier
        if any(name in table.session for name in gesture_free):
            for b in table.bursts:
                if b.role != "gesture":
                    b.zone = "free:declared gesture-free"
        trained_for = labels or (tuple(n for n in table.model_labels if n != "none") or None)
        found = weak_labels(table, wc, trained_for, may_hold_gestures=loose.get(table.folder, False))
        ex, w = examples_of(table, found)
        examples += ex
        weights.append(w)
        rows += table_rows(table, found)
        summary["sessions"] += 1
        tags: dict[str, int] = {}
        for wl in found:
            tags[wl.tag] = tags.get(wl.tag, 0) + 1
            summary["by_tag"][wl.tag] = summary["by_tag"].get(wl.tag, 0) + 1
        if verbose:
            shown = ", ".join(f"{v} {k}" for k, v in sorted(tags.items())) or "nothing extra"
            print(f"{table.session}: {len(table.bursts)} bursts -> {shown}")
    seconds = {**info["seconds"], "rules": time.perf_counter() - t0}
    summary["timing"] = {"audio_minutes": round(info["audio_minutes"], 2), "bursts": info.get("bursts", 0), "burst_sets": info["how"],
                         "judge": info["judge"], "seconds": {k: round(v, 3) for k, v in seconds.items()},
                         "judge_seconds": info["judge_seconds"]}
    if not examples:
        return None, rows, summary
    data = pack(examples, cfg_json)
    data["weight"] = np.concatenate(weights).astype(np.float32)
    summary["examples"] = int(len(data["y"]))
    summary["by_label"] = {str(n): int(np.sum(data["y"] == i)) for i, n in enumerate(data["labels"]) if np.any(data["y"] == i)}
    return data, rows, summary


def timing_line(t: dict[str, Any]) -> str:
    how = ", ".join(f"{n} by {k}" for k, n in t["burst_sets"].items()) or "none"
    s = t["seconds"]
    return (f"time: {t['audio_minutes']} min of audio, {t['bursts']} bursts; step 1 (audio -> bursts: {how}) {s.get('bursts', 0):.2f} s; "
            f"step 2 (verdicts: {t['judge']}) {s.get('verdicts', 0):.3f} s; step 3 (rules) {s.get('rules', 0):.3f} s")


# ---------------------------------------------------------------------------
# Parity: the fast path against the original one-burst-at-a-time CPU code
# ---------------------------------------------------------------------------
PARITY_LIMITS = {"pictures": 2e-3, "physics": 1e-4, "model": 5e-3}


def parity(folders: list[Path], model: str | Path | None, device: Any, frontend: str = "auto", limit: int = 6) -> bool:
    """Compare, on real sessions: (1) bursts and pictures from the fast step 1 against the engine's CPU
    pipeline, (2) the batched physics rules against gestures.heuristic_classify, (3) the batched model
    against classifier.OnnxClassifier one burst at a time. Prints the worst difference of each; True if all pass."""
    from .classifier import OnnxClassifier
    from .gestures import heuristic_classify

    dev = pick_device(device) if (device is None or isinstance(device, str)) else device
    ok = True
    worst = {"pictures": 0.0, "physics": 0.0, "model": 0.0}
    counts = {"sessions": 0, "bursts": 0, "burst_mismatch": 0, "physics_label_diff": 0, "model_label_diff": 0}
    front_used = "cpu"
    judge: BatchJudge | None = None
    onnx: Any = None
    for folder in folders[:limit]:
        try:
            _, audio, cfg = load_session(folder)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            print(f"  skipped {folder.name}: {type(exc).__name__}: {exc}")
            continue
        ref = burst_set_of(stream_features(audio, cfg), "cpu", 0.0)
        fast, why = all_bursts([folder], dev, None, 1, frontend)[0]
        if fast is None:
            print(f"  FAILED on {folder.name}: {why}")
            ok = False
            continue
        front_used = fast.how
        counts["sessions"] += 1
        counts["bursts"] += len(ref)
        if len(fast) != len(ref) or not np.array_equal(fast.seg[:, :3], ref.seg[:, :3]):
            counts["burst_mismatch"] += 1
        elif len(ref):
            worst["pictures"] = max(worst["pictures"], float(np.max(np.abs(fast.x.astype(np.float32) - ref.x.astype(np.float32)))))
        if judge is None:
            judge = BatchJudge(ref.cfg, model, dev)
            if judge.has_model and model:
                onnx = OnnxClassifier(model, ref.cfg)
        x = ref.windows()
        if len(x) == 0:
            continue
        v = judge(x)
        one = [heuristic_classify(w.astype(np.float32), judge.cfg) for w in x]
        counts["physics_label_diff"] += int(sum(GESTURE_LABELS.index(r.label) != int(p) for r, p in zip(one, v.physics)))
        worst["physics"] = max(worst["physics"], float(np.max(np.abs(np.stack([r.probs for r in one]) - v.physics_probs))))
        if onnx is not None and v.model is not None and v.model_probs is not None:
            single = [onnx.predict(w.astype(np.float32)) for w in x]
            counts["model_label_diff"] += int(sum(s[0] != v.labels[int(m)] for s, m in zip(single, v.model)))
            worst["model"] = max(worst["model"], float(np.max(np.abs(np.array([[s[2][n] for n in v.labels] for s in single]) - v.model_probs))))
    print(f"parity on {counts['sessions']} session(s), {counts['bursts']} bursts; step 1 by: {front_used}; "
          f"judge: {judge.describe() if judge else 'n/a'}")
    same = "yes" if not counts["burst_mismatch"] else f"NO, {counts['burst_mismatch']} session(s) differ"
    print(f"  bursts found at the same frames:        {same}")
    print(f"  pictures, worst difference:             {worst['pictures']:.2e}   (limit {PARITY_LIMITS['pictures']:.0e})")
    print(f"  physics rules, worst probability diff:  {worst['physics']:.2e}   (limit {PARITY_LIMITS['physics']:.0e}), "
          f"{counts['physics_label_diff']} different verdicts")
    if onnx is not None:
        print(f"  model, worst probability difference:    {worst['model']:.2e}   (limit {PARITY_LIMITS['model']:.0e}), "
              f"{counts['model_label_diff']} different verdicts")
    else:
        print("  model: no trained model given, not compared")
    ok = ok and counts["sessions"] > 0 and not counts["burst_mismatch"] and not counts["physics_label_diff"] \
        and not counts["model_label_diff"] and all(worst[k] <= PARITY_LIMITS[k] for k in worst)
    print("PARITY OK" if ok else "PARITY FAILED")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sonar.weaklabel", description="Extra weighted training examples: pseudo-labels and hard negatives")
    ap.add_argument("roots", nargs="*", default=[], help="the training folder(s): session folders or folders containing sessions")
    ap.add_argument("--unlabelled", nargs="*", default=[],
                    help="folders of continuous recordings without labels that may contain gestures (pseudo-labelled; keep them OUT of the training folder)")
    ap.add_argument("--out", default="weak.npz", help="dataset of the extra examples (with a weight column)")
    ap.add_argument("--model", default="models/gesture.onnx", help="current model (physics rules only if the file is missing)")
    ap.add_argument("--hard-negatives", help="also write the hard negatives alone to this .npz")
    ap.add_argument("--report", help="write every burst with every witness's opinion to this .json")
    ap.add_argument("--exclude-person", default="", help="leave these people out (the held-out people of the honest score)")
    ap.add_argument("--labels", default="", help="gestures the model is trained for (default: all)")
    ap.add_argument("--gesture-free", default="", help="session names (comma-separated, parts are enough) known to hold no gesture")
    ap.add_argument("--min-confidence", type=float, default=0.6, help="the engine's threshold: above it the model 'fires'")
    ap.add_argument("--pseudo-conf", type=float, default=0.90, help="how sure the model must be before its word becomes a label")
    ap.add_argument("--no-pseudo-none", action="store_true", help="do not keep unlabelled bursts that everything calls 'none'")
    ap.add_argument("--profile", help="only use sessions recorded with this signal profile")
    ap.add_argument("--no-sim-sessions", action="store_true", help="skip simulator recordings")
    ap.add_argument("--device", default="auto", help="auto (CUDA if there is one, else CPU) | cuda | cpu | mps | none (no PyTorch)")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16), help="CPU processes for step 1 (audio -> bursts)")
    ap.add_argument("--cache", help="folder that keeps the bursts of each session between runs")
    ap.add_argument("--frontend", default="auto", choices=["auto", "torch", "cpu"],
                    help="step 1 signal processing: auto = the GPU front end on CUDA if this checkout has it, else the engine's CPU pipeline")
    ap.add_argument("--tables-only", action="store_true", help="only do step 1 and fill --cache (train_4090.py runs this while the GPU pre-trains)")
    ap.add_argument("--compile", action="store_true", help="try torch.compile for the model pass (rarely worth it; falls back by itself)")
    ap.add_argument("--parity", action="store_true", help="compare the fast path with the original CPU code on up to 6 sessions and exit")
    args = ap.parse_args(argv)
    if not args.roots and not args.unlabelled:
        ap.error("give at least one folder")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    cache = Path(args.cache) if args.cache else None
    people = tuple(p for p in args.exclude_person.split(",") if p)
    labels = tuple(g for g in args.labels.split(",") if g) or None
    model = args.model if args.model and Path(args.model).exists() else None

    if args.parity or args.tables_only:
        skipped: list[str] = []
        chosen = [f for f, _ in pick_sessions(args.roots, args.unlabelled, people, args.profile, not args.no_sim_sessions, skipped)]
        if not chosen:
            print("no usable session found" + "".join(f"\n  {s}" for s in skipped), file=sys.stderr)
            return 2
        if args.parity:
            return 0 if parity(chosen, model, args.device, args.frontend) else 1
        t0 = time.perf_counter()
        sets = all_bursts(chosen, pick_device(args.device), cache, args.workers, args.frontend)
        done = [bs for bs, _ in sets if bs is not None]
        how: dict[str, int] = {}
        for bs in done:
            how[bs.how] = how.get(bs.how, 0) + 1
        for folder, (bs, why) in zip(chosen, sets):
            if bs is None:
                print(f"WARNING: skipped {folder}: {why}", file=sys.stderr)
        print(f"bursts of {len(done)} session(s) ready in {time.perf_counter() - t0:.1f} s "
              f"({', '.join(f'{n} by {k}' for k, n in how.items()) or 'none'}; {sum(len(bs) for bs in done)} bursts)"
              + (f" -> {cache}" if cache else ""))
        return 0

    wc = WeakConfig(min_confidence=args.min_confidence, pseudo_conf=args.pseudo_conf, pseudo_none=not args.no_pseudo_none)
    data, rows, summary = build_weak(args.roots, model, wc, people, labels,
                                     gesture_free=tuple(s for s in args.gesture_free.split(",") if s), unlabelled=args.unlabelled,
                                     profile=args.profile, allow_sim=not args.no_sim_sessions, device=args.device, cache=cache,
                                     workers=args.workers, frontend=args.frontend, compile_model=args.compile)
    for line in summary["skipped"]:
        print(f"WARNING: skipped {line}", file=sys.stderr)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({"summary": summary, "bursts": rows}, indent=1))
    print(timing_line(summary["timing"]))
    if data is None:
        print(f"no extra examples found in {summary['sessions']} session(s) (classifier: {summary['classifier']})")
        return 0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    print(f"wrote {args.out}: {summary['examples']} extra examples {summary['by_tag']} labelled {summary['by_label']} "
          f"from {summary['sessions']} session(s), classifier: {summary['classifier']}")
    if args.hard_negatives:
        hard = data["source"] == "hardneg"
        if hard.any():
            sub = {k: (v[hard] if k in ("X", "y", "group", "person", "source", "peak_db", "weight") else v) for k, v in data.items()}
            np.savez_compressed(args.hard_negatives, **sub)
            what: dict[str, int] = {}
            for r in rows:
                if r.get("weak_tag") == "hardneg":
                    what[r["zone"]] = what.get(r["zone"], 0) + 1
            print(f"wrote {args.hard_negatives}: {int(hard.sum())} hard negatives; where they happened: {what}")
        else:
            print("no hard negatives: the classifier did not fire in any gesture-free part")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

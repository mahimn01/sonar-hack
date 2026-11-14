"""dataset.py — recorded sessions -> labelled training pictures (+ augmentation).

Plain words: a recording session is one long WAV plus a list of "between sample
A and sample B the person was asked to do X". This module

  1. runs the whole WAV through the same pipeline the live engine uses,
  2. finds the movement bursts, and for every prompted gesture takes the
     STRONGEST burst inside its prompt window as the example of that gesture
     (so the labels line up with the movement automatically, no hand cutting),
  3. turns every other burst (reaching, typing, walking in ...) into a "none"
     example, plus a few pictures of nothing at all,
  4. saves everything as one `.npz` file.

`Augmenter` makes each picture slightly different every time it is used for
training (shifted, mirrored left<->right, faster/slower, quieter, noisier), so
a few hundred real examples behave like many more.

How to run:
    python -m sonar.dataset data/ --out dataset.npz        (all sessions under data/)
What it changes: writes the one .npz file you name. Reads the sessions only.

Dataset file (`np.load(path, allow_pickle=False)`):
    X        [N, C, F, T + 2*pad] float16   feature pictures (see gestures.py for the layout)
    y        [N] int64                      index into `labels`
    labels   [K] str                        class names, "none" first
    group    [N] str                        session id (splits are made by session, never inside one)
    person   [N] str
    source   [N] str                        "gesture" | "burst" | "idle" | "aux" | "synthetic"
    peak_db  [N] float32                    strength of the burst
    pad      int                            extra frames on each side for time-shift augmentation
    config   str                            JSON of the SonarConfig used
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .config import GESTURE_LABELS, SonarConfig
from .features import StreamFeatures, stream_features, window_from_stream

PAD = 8                       # frames of slack on each side (0.16 s)
SESSION_FILE = "session.json"
AUDIO_FILE = "audio.wav"


# ---------------------------------------------------------------------------
# Sessions on disk
# ---------------------------------------------------------------------------
def find_sessions(root: str | Path) -> list[Path]:
    """All session folders (those containing session.json) under `root`, sorted."""
    root = Path(root)
    if (root / SESSION_FILE).exists():
        return [root]
    return sorted(p.parent for p in root.rglob(SESSION_FILE))


def load_session(folder: str | Path) -> tuple[dict[str, Any], np.ndarray, SonarConfig]:
    """(metadata, audio [n, n_mics] float32, config the session was recorded with)."""
    import soundfile as sf

    folder = Path(folder)
    meta = json.loads((folder / SESSION_FILE).read_text())
    audio, fs = sf.read(str(folder / meta.get("audio", AUDIO_FILE)), dtype="float32", always_2d=True)
    cfg = SonarConfig.from_dict(meta["config"])
    if fs != cfg.fs:
        raise ValueError(f"{folder}: audio is {fs} Hz but the session config says {cfg.fs} Hz")
    if audio.shape[1] != cfg.n_mics:
        cfg = cfg.with_(n_mics=audio.shape[1])
    return meta, audio, cfg


@dataclass
class Example:
    x: np.ndarray        # [C, F, T + 2*PAD]
    label: str
    group: str
    person: str
    source: str
    peak_db: float


def session_examples(
    folder: str | Path,
    aux_as_none: bool = False,
    n_idle: int = 8,
    seed: int = 0,
    stream: StreamFeatures | None = None,
) -> tuple[list[Example], dict[str, Any]]:
    """Cut one session into labelled examples. Returns (examples, a small report)."""
    meta, audio, cfg = load_session(folder)
    stream = stream_features(audio, cfg) if stream is None else stream
    group = meta.get("session_id", Path(folder).name)
    person = meta.get("person", "unknown")
    rng = np.random.default_rng(seed)
    guard = int(1.0 * cfg.frame_rate)
    used = np.zeros(len(stream.segments), dtype=bool)
    examples: list[Example] = []
    report: dict[str, Any] = {"session": group, "gestures": 0, "missed": 0, "none": 0, "idle": 0, "missed_trials": []}

    def cut(k: int) -> np.ndarray:
        seg = stream.segments[k]
        return window_from_stream(stream, seg.decided, stream.not_before[k], pad=PAD)

    # 1. prompted gestures: strongest burst inside (a little beyond) the prompt window
    windows: list[tuple[int, int]] = []
    for ev in meta.get("events", []):
        f0, f1 = stream.frame_of_sample(ev["start"]), stream.frame_of_sample(ev["end"])
        if ev.get("exclude"):
            # camera recorder (camrecord.py): a reading it was unsure about, or a free-form block in
            # which the volunteer may have gestured unseen. Not a gesture example, and no "none" either.
            windows.append((f0, f1))
            continue
        if ev["label"] not in GESTURE_LABELS or ev["label"] == "none":
            continue
        windows.append((f0 - guard, f1 + 2 * guard))
        inside = [k for k, s in enumerate(stream.segments) if s.end >= f0 - 5 and s.start <= f1 + 25]
        rank = lambda k: stream.segments[k].peak_db                     # noqa: E731
        if ev.get("tight"):
            # start/end bound the stroke itself (timed by the camera), not a 2.5 s prompt window:
            # take the burst that covers the stroke, not the loudest one nearby (often the hand going back)
            cover = lambda k: min(stream.segments[k].end, f1 + 3) - max(stream.segments[k].start, f0 - 1)   # noqa: E731
            inside = [k for k in inside if cover(k) >= 0]
            rank = lambda k: (cover(k), stream.segments[k].peak_db)     # noqa: E731
        if not inside:
            report["missed"] += 1
            report["missed_trials"].append({"label": ev["label"], "rep": ev.get("rep"), "t": round(ev["start"] / cfg.fs, 1)})
            continue
        best = max(inside, key=rank)
        used[inside] = True
        examples.append(Example(cut(best), ev["label"], group, person, "gesture", float(stream.segments[best].peak_db)))
        report["gestures"] += 1
        if aux_as_none:                     # wind-up / hand-return bursts of the same trial
            for k in inside:
                if k != best:
                    examples.append(Example(cut(k), "none", group, person, "aux", float(stream.segments[k].peak_db)))

    # 2. every other burst is a negative
    for k, seg in enumerate(stream.segments):
        if used[k] or any(a <= seg.end and seg.start <= b for a, b in windows):
            continue
        examples.append(Example(cut(k), "none", group, person, "burst", float(seg.peak_db)))
        report["none"] += 1

    # 3. a few pictures of nothing (quiet moments), so "no movement" is firmly "none"
    quiet = np.where(stream.ready & (stream.activity_db < 1.0))[0] + 1
    quiet = quiet[(quiet > cfg.gesture_frames + PAD) & (quiet < stream.n_frames - PAD)]
    quiet = np.array([q for q in quiet if not any(a <= q <= b for a, b in windows)], dtype=int)
    if len(quiet):
        for q in rng.choice(quiet, size=min(n_idle, len(quiet)), replace=False):
            examples.append(Example(window_from_stream(stream, int(q), pad=PAD), "none", group, person, "idle", 0.0))
            report["idle"] += 1
    return examples, report


def build_dataset(
    roots: list[str | Path], aux_as_none: bool = False, n_idle: int = 8, verbose: bool = True
) -> dict[str, np.ndarray]:
    """All sessions under `roots` -> the dataset arrays described at the top of this file.

    A session that cannot be read (still being recorded, half copied, damaged) is skipped with a
    warning; a session recorded with a different signal profile stops the build.
    """
    examples: list[Example] = []
    cfg_json = ""
    skipped: list[str] = []
    for root in roots:
        for folder in find_sessions(root):
            try:
                meta, _, cfg = load_session(folder)
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                print(f"WARNING: skipped session {folder}: cannot read it ({type(exc).__name__}: {exc})", file=sys.stderr)
                skipped.append(str(folder))
                continue
            this = json.dumps(cfg.to_dict(), sort_keys=True)
            if cfg_json and _feature_key(json.loads(this)) != _feature_key(json.loads(cfg_json)):
                raise ValueError(f"{folder} was recorded with a different signal profile than the other sessions")
            cfg_json = this
            try:
                ex, rep = session_examples(folder, aux_as_none=aux_as_none, n_idle=n_idle)
            except (OSError, RuntimeError, ValueError, KeyError, IndexError) as exc:
                print(f"WARNING: skipped session {folder}: {type(exc).__name__}: {exc}", file=sys.stderr)
                skipped.append(str(folder))
                continue
            examples += ex
            if verbose:
                print(
                    f"{rep['session']}: {rep['gestures']} gestures, {rep['missed']} prompts with no movement found, "
                    f"{rep['none']} other bursts, {rep['idle']} idle"
                )
    if skipped:
        print(f"WARNING: {len(skipped)} session(s) skipped, see above", file=sys.stderr)
    if not examples:
        raise ValueError(f"no usable sessions under {[str(r) for r in roots]}")
    return pack(examples, cfg_json)


def _feature_key(cfg_dict: dict[str, Any]) -> tuple:
    from .classifier import FEATURE_FIELDS

    return tuple(json.dumps(cfg_dict.get(k)) for k in FEATURE_FIELDS)


def pack(examples: list[Example], cfg_json: str) -> dict[str, np.ndarray]:
    labels = list(GESTURE_LABELS)
    return {
        "X": np.stack([e.x for e in examples]).astype(np.float16),
        "y": np.array([labels.index(e.label) for e in examples], dtype=np.int64),
        "labels": np.array(labels),
        "group": np.array([e.group for e in examples]),
        "person": np.array([e.person for e in examples]),
        "source": np.array([e.source for e in examples]),
        "peak_db": np.array([e.peak_db for e in examples], dtype=np.float32),
        "pad": np.array(PAD),
        "config": np.array(cfg_json),
    }


def merge(datasets: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate datasets (e.g. real sessions + synthetic examples from datagen)."""
    first = datasets[0]
    for d in datasets[1:]:
        if list(d["labels"]) != list(first["labels"]) or d["X"].shape[1:] != first["X"].shape[1:]:
            raise ValueError("datasets have different labels or picture shapes")
    out = {k: np.concatenate([d[k] for d in datasets]) for k in ("X", "y", "group", "person", "source", "peak_db")}
    out.update({k: first[k] for k in ("labels", "pad", "config")})
    return out


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
class Augmenter:
    """Random, physically sensible variations of one feature picture.

    Every transform mirrors something that really varies between people and
    days: when the gesture happens inside the window, which way it goes, how
    fast, how loud the echo is, how noisy the room is.
    """

    def __init__(self, cfg: SonarConfig, labels: list[str], pad: int = PAD, seed: int | None = None, strength: float = 1.0) -> None:
        self.cfg = cfg
        self.labels = list(labels)
        self.pad = pad
        self.rng = np.random.default_rng(seed)
        self.strength = strength
        self.n_p = cfg.n_power_channels
        self.has_lat = cfg.has_lateral
        self._flip_f = (cfg.doppler_bins - np.arange(cfg.doppler_bins)) % cfg.doppler_bins   # f -> -f on the bin grid
        self._swap = {}
        if "swipe_left" in self.labels and "swipe_right" in self.labels:
            a, b = self.labels.index("swipe_left"), self.labels.index("swipe_right")
            self._swap = {a: b, b: a}
        # channel order after exchanging left<->right: tones swapped (if there are two on
        # different speakers) and microphones swapped
        tones = list(range(cfg.n_tones))
        if cfg.n_tones == 2 and cfg.tone_channels[0] != cfg.tone_channels[1]:
            tones = [1, 0]
        mics = list(range(cfg.n_mics))[::-1]
        self._mirror_idx = [t * cfg.n_mics + m for t in tones for m in mics]
        self._can_mirror = cfg.n_mics == 2
        self._swipes = set(self._swap)
        self._none = self.labels.index("none") if "none" in self.labels else -1

    # -- individual transforms ---------------------------------------------------------
    def mirror(self, x: np.ndarray, y: int) -> tuple[np.ndarray, int]:
        """Exchange left and right: a swipe left becomes a swipe right."""
        out = x.copy()
        out[:self.n_p] = x[self._mirror_idx]
        if self.has_lat:
            out[self.n_p] = -x[self.n_p]
        return out, self._swap.get(y, y)

    def reverse(self, x: np.ndarray, y: int) -> tuple[np.ndarray, int]:
        """Play the movement backwards: time reversed, every speed sign reversed."""
        out = x[:, self._flip_f, ::-1].copy()
        if self.has_lat:
            out[self.n_p] = -out[self.n_p]
        return out, self._swap.get(y, y)

    def stretch(self, x: np.ndarray, s: float) -> np.ndarray:
        """The same gesture done s times faster: shorter in time, proportionally higher Doppler."""
        f0 = self.cfg.doppler_bins / 2.0                                     # the 0 Hz bin stays put
        t0 = x.shape[-1] - 1 - self.pad - self.cfg.seg_hang_frames           # ... and so does the end of the burst
        out = np.empty_like(x)
        for c in range(x.shape[0]):
            # output(f, t) = input(f0 + (f - f0)/s, t0 + (t - t0)*s)
            out[c] = ndimage.affine_transform(
                x[c], [1.0 / s, s], offset=[f0 - f0 / s, t0 - t0 * s], order=1, mode="constant", cval=0.0
            )
        if self.has_lat:
            out[self.n_p] = np.clip(out[self.n_p] * s, -1.0, 1.0)     # sideways Doppler grows with speed too
        return out

    def gain(self, x: np.ndarray, db: float) -> np.ndarray:
        """Echo `db` louder or quieter (features are dB above the floor / feat_range_db)."""
        out = x.copy()
        p = out[:self.n_p]
        shifted = p + db / self.cfg.feat_range_db
        out[:self.n_p] = np.where(p > 0, np.clip(shifted, 0.0, 1.0), 0.0)
        if self.has_lat and db < 0:          # the lateral picture is masked by echo strength
            scale = np.clip((p.mean(axis=0) + db / self.cfg.feat_range_db) / (p.mean(axis=0) + 1e-6), 0.0, 1.0)
            out[self.n_p] *= np.minimum(1.0, scale * 3.0)
        return out

    def noise(self, x: np.ndarray) -> np.ndarray:
        """Fresh noise-floor speckle (what an idle spectrogram looks like)."""
        cfg = self.cfg
        out = x.copy()
        e = self.rng.exponential(size=out[:self.n_p].shape)
        speckle = np.clip((10 * np.log10(e + 1e-12) - cfg.feat_floor_margin_db) / cfg.feat_range_db, 0.0, 1.0)
        out[:self.n_p] = np.maximum(out[:self.n_p], speckle.astype(out.dtype))
        return out

    def click(self, x: np.ndarray) -> np.ndarray:
        """A broadband click (keyboard, clap): a faint vertical stripe 1-3 frames wide."""
        out = x.copy()
        t = int(self.rng.integers(0, x.shape[-1] - 3))
        w = int(self.rng.integers(1, 4))
        level = float(self.rng.uniform(0.1, 0.4))
        out[:self.n_p, :, t:t + w] = np.maximum(out[:self.n_p, :, t:t + w], level)
        return out

    def mask(self, x: np.ndarray) -> np.ndarray:
        """Hide a short time strip or a few Doppler bins (stops the model relying on one spot)."""
        out = x.copy()
        if self.rng.random() < 0.5:
            t = int(self.rng.integers(0, x.shape[-1] - 6))
            out[:, :, t:t + int(self.rng.integers(2, 7))] = 0.0
        else:
            f = int(self.rng.integers(0, x.shape[1] - 4))
            out[:, f:f + int(self.rng.integers(1, 5)), :] = 0.0
        return out

    # -- the full recipe -------------------------------------------------------------------
    def crop(self, x: np.ndarray, shift: int = 0) -> np.ndarray:
        """Cut the model-sized window out of the padded picture (shift 0 = exactly what the engine sees)."""
        t = self.cfg.gesture_frames
        start = int(np.clip(self.pad - shift, 0, x.shape[-1] - t))
        return x[:, :, start:start + t]

    def __call__(self, x: np.ndarray, y: int) -> tuple[np.ndarray, int]:
        rng, k = self.rng, self.strength
        x = np.asarray(x, dtype=np.float32)
        if rng.random() < 0.5 * k:
            x = self.stretch(x, float(np.exp(rng.uniform(-0.18, 0.18))))
        x = self.crop(x, int(rng.integers(-self.pad, self.pad + 1)) if k > 0 else 0)
        if self._can_mirror and rng.random() < 0.5 * min(k, 1.0):
            x, y = self.mirror(x, y)
        if (y in self._swipes or y == self._none) and rng.random() < 0.2 * k:
            x, y = self.reverse(x, y)
        if rng.random() < 0.7 * k:
            x = self.gain(x, float(rng.uniform(-9.0, 6.0)))
        if rng.random() < 0.5 * k:
            x = self.noise(x)
        if rng.random() < 0.15 * k:
            x = self.click(x)
        if rng.random() < 0.25 * k:
            x = self.mask(x)
        return np.ascontiguousarray(x, dtype=np.float32), y


def center_crop(x: np.ndarray, cfg: SonarConfig, pad: int = PAD) -> np.ndarray:
    """[..., T + 2*pad] -> [..., T]: the un-shifted window, i.e. exactly what the engine classifies."""
    return x[..., pad:pad + cfg.gesture_frames]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sonar.dataset", description="Build a training dataset from recorded sessions")
    ap.add_argument("roots", nargs="+", help="session folders or folders containing sessions")
    ap.add_argument("--out", default="dataset.npz")
    ap.add_argument("--aux-none", action="store_true", help="also use wind-up / hand-return bursts of gesture trials as 'none'")
    ap.add_argument("--idle", type=int, default=8, help="quiet 'nothing happens' examples per session")
    args = ap.parse_args(argv)
    try:
        data = build_dataset(args.roots, aux_as_none=args.aux_none, n_idle=args.idle)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    counts = {str(name): int(np.sum(data["y"] == i)) for i, name in enumerate(data["labels"])}
    print(f"wrote {args.out}: {len(data['y'])} examples {counts}, picture shape {data['X'].shape[1:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

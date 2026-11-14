"""classifier.py — "which gesture is in this 1.6 s picture?"

Plain words: two interchangeable classifiers with the same `predict` method.

  OnnxClassifier     the trained CNN, run with onnxruntime on the CPU (well under
                     1 ms per gesture). Needs `gesture.onnx` plus `gesture.json`
                     (labels, confidence temperature, the config it was trained with).
  PhysicsClassifier  the training-free Doppler-sign rules from gestures.py.
                     Used when no model file exists yet, or with `--physics`.

`load_classifier` picks the model if the file is there, otherwise the physics rules.

Read-only module (it only reads the model files).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import GESTURE_LABELS, SonarConfig
from .gestures import heuristic_classify

# Fields that change what the feature picture looks like. A model must be run with the
# same values it was trained with.
FEATURE_FIELDS = (
    "fs", "frame_len", "tone_hz", "tone_channels", "n_mics", "decim", "clutter_hp_hz", "stft_win", "stft_fft",
    "doppler_bins", "feat_floor_margin_db", "feat_range_db", "gesture_frames",
)


class PhysicsClassifier:
    source = "physics"

    def __init__(self, cfg: SonarConfig, meta: dict[str, Any] | None = None) -> None:
        self.meta = meta or {}
        # train.py measures on recorded swipes which way round this laptop's speakers / mics are.
        h = self.meta.get("heuristic", {})
        if "w_speakers" in h and "w_mics" in h and (h["w_speakers"] or h["w_mics"]):
            cfg = cfg.with_(heur_w_speakers=float(h["w_speakers"]), heur_w_mics=float(h["w_mics"]))
        self.cfg = cfg
        self.labels = GESTURE_LABELS

    def predict(self, window: np.ndarray) -> tuple[str, float, dict[str, float]]:
        r = heuristic_classify(window, self.cfg)
        return r.label, r.confidence, {name: float(p) for name, p in zip(GESTURE_LABELS, r.probs)}


class OnnxClassifier:
    source = "model"

    def __init__(self, path: str | Path, cfg: SonarConfig) -> None:
        import onnxruntime as ort

        path = Path(path)
        meta_path = path.with_suffix(".json")
        self.meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.labels = tuple(self.meta.get("labels", GESTURE_LABELS))
        self.temperature = float(self.meta.get("temperature", 1.0))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1            # tiny model: one thread is fastest and leaves the CPU to the DSP
        self._sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        shape = self._sess.get_inputs()[0].shape
        want = (cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames)
        if tuple(shape[1:]) != want:
            raise ValueError(f"model {path} expects input {tuple(shape[1:])} but this profile produces {want}")
        self.mismatch = [
            k for k in FEATURE_FIELDS
            if "config" in self.meta and k in self.meta["config"] and _norm(self.meta["config"][k]) != _norm(getattr(cfg, k))
        ]

    def predict(self, window: np.ndarray) -> tuple[str, float, dict[str, float]]:
        logits = self._sess.run(None, {self._input: window[None].astype(np.float32)})[0][0]
        z = logits / self.temperature
        z = z - z.max()
        p = np.exp(z) / np.exp(z).sum()
        best = int(np.argmax(p))
        return self.labels[best], float(p[best]), {name: float(v) for name, v in zip(self.labels, p)}


def _norm(v: Any) -> Any:
    return list(v) if isinstance(v, (tuple, list)) else v


def load_classifier(model: str | Path | None, cfg: SonarConfig, force_physics: bool = False) -> PhysicsClassifier | OnnxClassifier:
    """The trained model if `model` exists (and physics is not forced), else the physics rules.

    The physics rules still read `<model>.json` if it exists, to pick up the cue signs and
    the typical gesture strength measured from the recordings.
    """
    if not force_physics and model is not None and Path(model).exists():
        return OnnxClassifier(model, cfg)
    meta: dict[str, Any] = {}
    if model is not None and Path(model).with_suffix(".json").exists():
        meta = json.loads(Path(model).with_suffix(".json").read_text())
    return PhysicsClassifier(cfg, meta)

"""batch_judge.py — what the physics rules and the trained model say about MANY bursts at once.

Plain words: the label QA and the weak labeller need two opinions about every movement burst
of every session: the training-free physics rules (gestures.heuristic_classify) and the
current model. Asking one burst at a time is a Python loop over thousands of pictures.
This file answers for the whole stack of pictures in one go:

    heuristic_batch   the physics rules for a stack [N, C, F, T], written with PyTorch tensor
                      operations, so it runs on the GPU (or on the CPU) with no per-burst loop.
                      It is the same arithmetic as gestures.heuristic_classify; a test and
                      `python -m sonar.weaklabel --parity` compare the two.
    BatchJudge        loads the model once and judges a stack: with PyTorch from gesture.pt on
                      the chosen device, or else with ONNX Runtime in batches; plus the physics
                      rules. The pictures are moved to the device once and stay there.

Fallbacks: without PyTorch (the demo laptop) the model runs through ONNX Runtime in batches
and the physics rules fall back to the original one-burst-at-a-time function. Results are
the same either way; only the speed differs.

Read-only module: it reads the model files, writes nothing.
"""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .classifier import FEATURE_FIELDS, _norm
from .config import GESTURE_LABELS, SonarConfig
from .gestures import heuristic_classify

CHUNK = 4096             # bursts per step on the device (about 0.5 GB of float64 work memory for the physics rules)


def have_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def pick_device(name: str | None = "auto") -> Any:
    """A torch.device, or None when PyTorch is not installed. auto = CUDA if there is one, else the CPU.

    Apple's MPS is only used when asked for by name: it has no float64, and the Mac Studio's GPU is shared.
    """
    if name in (None, "none") or not have_torch():
        return None
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@contextlib.contextmanager
def exact_gpu_maths() -> Iterator[None]:
    """Judging must give the numbers the engine gives: no reduced-precision (TF32) convolutions meanwhile."""
    import torch

    old = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


# ---------------------------------------------------------------------------
# The physics rules for a whole stack of pictures
# ---------------------------------------------------------------------------
def heuristic_batch(x: Any, cfg: SonarConfig) -> dict[str, Any]:
    """gestures.heuristic_classify for a stack x [N, C, F, T] (a torch tensor on any device).

    Returns torch tensors on the same device: `label` [N] (index into GESTURE_LABELS), `conf` [N],
    `probs` [N, 5], and the cues `lateral_hz`, `lateral_mic_hz`, `order_s` [N].
    Every line mirrors the single-picture function; see there for what the numbers mean.
    """
    import torch

    n, c, n_f, n_t = x.shape
    dev = x.device
    dt = torch.float32 if dev.type == "mps" else torch.float64
    n_p, tones, mics = cfg.n_power_channels, cfg.n_tones, cfg.n_mics
    f_np = np.asarray(cfg.doppler_hz, dtype=np.float64)
    f = torch.from_numpy(f_np).to(dev, dt)
    use = torch.from_numpy(np.abs(f_np) >= 15.0).to(dev)
    use_f = use.to(dt)
    pos_idx = torch.from_numpy(np.flatnonzero(f_np > 0)).to(dev)
    neg_idx = torch.from_numpy(np.flatnonzero(f_np < 0)).to(dev)
    use_idx = torch.from_numpy(np.flatnonzero(np.abs(f_np) >= 15.0)).to(dev)
    t = torch.arange(n_t, device=dev, dtype=dt) / cfg.frame_rate
    box = torch.full((1, 1, 7), 1.0 / 7.0, device=dev, dtype=dt)

    xw = x.to(dt)
    power = xw[:, :n_p].reshape(n, tones, mics, n_f, n_t)
    per_tone = power.mean(dim=2)                                              # [N, tones, F, T]
    peak = per_tone.index_select(2, use_idx).amax(dim=(1, 2, 3))              # [N]
    w = (per_tone - torch.clamp(0.3 * peak, min=0.08)[:, None, None, None]).clamp_min(0.0) * use_f[None, None, :, None]
    e = w.sum(dim=2)                                                          # [N, tones, T]
    cen = (w * f[None, None, :, None]).sum(dim=2) / (e + 1e-9)                # Doppler centroid (Hz)
    e_sum = e.sum(dim=1)                                                      # [N, T]
    nothing = (peak < 0.12) | ((e_sum > 0.5 * peak[:, None]).sum(dim=1) < 4)

    # focus on the stroke itself: the strongest ~0.5 s
    e_tot = torch.nn.functional.conv1d(e_sum[:, None, :], box, padding=3)[:, 0]
    focus = torch.exp(-0.5 * ((t[None, :] - t[e_tot.argmax(dim=1)][:, None]) / 0.25) ** 2)       # [N, T]
    w = w * focus[:, None, None, :]
    e = e * focus[:, None, :]

    common_rms = torch.sqrt((e * cen**2).sum(dim=(1, 2)) / (e.sum(dim=(1, 2)) + 1e-9))
    lateral = torch.zeros(n, device=dev, dtype=dt)
    if tones >= 2 and cfg.tone_channels[0] != cfg.tone_channels[1]:            # cue 1: the two speakers
        both = torch.minimum(e[:, 0], e[:, 1])
        lateral = (both * (cen[:, 0] - cen[:, 1])).sum(dim=1) / (both.sum(dim=1) + 1e-9)
    lateral_mic = torch.zeros(n, device=dev, dtype=dt)
    if cfg.has_lateral and c > n_p:                                            # cue 2: the two microphones
        per_mic = power.mean(dim=1)                                            # [N, mics, F, T]
        gate = use_f[None, :, None] * focus[:, None, :]
        mask = torch.clamp(3.0 * torch.sqrt(per_mic[:, 0] * per_mic[:, 1]), max=1.0) * gate
        turn = (xw[:, n_p] * gate).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-9)
        lateral_mic = turn * np.pi / (2 * np.pi * cfg.frame_len / cfg.fs)
    direction = cfg.heur_w_speakers * lateral / 15.0 + cfg.heur_w_mics * lateral_mic / 5.0

    w_pos = w.index_select(2, pos_idx).sum(dim=1)                              # [N, F+, T]
    w_neg = w.index_select(2, neg_idx).sum(dim=1)
    pos, neg = w_pos.sum(dim=1), w_neg.sum(dim=1)                              # [N, T]
    ps, ns = pos.sum(dim=1), neg.sum(dim=1)
    order = (neg * t).sum(dim=1) / (ns + 1e-9) - (pos * t).sum(dim=1) / (ps + 1e-9)     # >0: approached first
    balance = torch.minimum(ps, ns) / (torch.maximum(ps, ns) + 1e-9)
    v_pos = (w_pos * f[pos_idx][None, :, None]).sum(dim=(1, 2)) / (ps + 1e-9)
    v_neg = (w_neg * -f[neg_idx][None, :, None]).sum(dim=(1, 2)) / (ns + 1e-9)

    radial_index = common_rms / 40.0
    swipe = torch.sigmoid((direction.abs() - 0.9) / 0.25) * torch.sigmoid((direction.abs() - 0.7 * radial_index) / 0.2)
    radial = (1.0 - swipe) * torch.sigmoid((balance - 0.12) / 0.05) * torch.sigmoid((order.abs() - 0.05) / 0.02)
    is_push = (order > 0) & (v_pos >= 0.7 * v_neg)
    right = direction < 0

    idx = {name: i for i, name in enumerate(GESTURE_LABELS)}
    probs = torch.full((n, len(GESTURE_LABELS)), 0.02, device=dev, dtype=dt)
    probs[:, idx["swipe_right"]] += swipe * right
    probs[:, idx["swipe_left"]] += swipe * ~right
    probs[:, idx["push"]] += radial * is_push
    probs[:, idx["pull"]] += radial * ~is_push
    probs[:, idx["none"]] += (1.0 - swipe - radial).clamp_min(0.0)
    blank = torch.full((len(GESTURE_LABELS),), 0.02, device=dev, dtype=dt)
    blank[idx["none"]] = 1.0
    probs = torch.where(nothing[:, None], blank[None, :], probs)
    probs = probs / probs.sum(dim=1, keepdim=True)
    conf, label = probs.max(dim=1)
    zero = torch.zeros_like(lateral)
    return {
        "label": label, "conf": torch.where(nothing, torch.ones_like(conf), conf), "probs": probs,
        "lateral_hz": torch.where(nothing, zero, lateral), "lateral_mic_hz": torch.where(nothing, zero, lateral_mic),
        "order_s": torch.where(nothing, zero, order),
    }


# ---------------------------------------------------------------------------
# Both witnesses for a stack of pictures
# ---------------------------------------------------------------------------
@dataclass
class Verdicts:
    physics: np.ndarray                 # [N] index into GESTURE_LABELS
    physics_conf: np.ndarray            # [N]
    physics_probs: np.ndarray           # [N, 5]
    model: np.ndarray | None = None     # [N] index into `labels` (None: no trained model)
    model_conf: np.ndarray | None = None
    model_probs: np.ndarray | None = None       # [N, K]
    labels: tuple[str, ...] = ()        # the model's class names
    seconds: dict[str, float] = field(default_factory=dict)


class BatchJudge:
    """Loads the model once; judges stacks of burst pictures [N, C, F, T] (numpy, any float type)."""

    def __init__(self, cfg: SonarConfig, model: str | Path | None = None, device: Any = "auto", compile_model: bool = False) -> None:
        self.device = pick_device(device) if (device is None or isinstance(device, str)) else device
        self.meta: dict[str, Any] = {}
        self.kind = "physics"                    # which engine runs the model: "torch" | "onnx" | "physics" (no model)
        self.labels: tuple[str, ...] = ()
        self.temperature = 1.0
        self.note = ""
        self._net: Any = None
        self._sess: Any = None
        path = Path(model) if model else None
        if path is not None and path.with_suffix(".json").exists():
            self.meta = json.loads(path.with_suffix(".json").read_text())
        h = self.meta.get("heuristic", {})          # which way round this laptop's speakers / microphones are (train.py)
        if "w_speakers" in h and "w_mics" in h and (h["w_speakers"] or h["w_mics"]):
            cfg = cfg.with_(heur_w_speakers=float(h["w_speakers"]), heur_w_mics=float(h["w_mics"]))
        self.cfg = cfg
        if path is None or not path.exists():
            return
        want = [cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames]
        if list(self.meta.get("input_shape", want)) != want:
            self.note = f"{path.name} expects pictures of shape {self.meta.get('input_shape')}, these sessions give {want}"
            return
        bad = [k for k in FEATURE_FIELDS if "config" in self.meta and k in self.meta["config"]
               and _norm(self.meta["config"][k]) != _norm(getattr(cfg, k))]
        if bad:
            self.note = f"{path.name} was trained with different signal settings for {bad}"
            return
        self.labels = tuple(self.meta.get("labels", GESTURE_LABELS))
        self.temperature = float(self.meta.get("temperature", 1.0))
        pt = path.with_suffix(".pt")
        if self.device is not None and pt.exists():
            import torch

            from .model import GestureNet

            state = torch.load(str(pt), map_location="cpu")
            width = int(state.get("width", 24)) if isinstance(state, dict) and "model" in state else 24
            weights = state["model"] if isinstance(state, dict) and "model" in state else state
            net = GestureNet(cfg.n_feat_channels, len(self.labels), width=width)
            net.load_state_dict(weights)
            self._net = net.to(self.device).eval()
            self.kind = "torch"
            if compile_model:                          # rarely worth it: one pass over a few thousand pictures
                try:
                    compiled = torch.compile(self._net)
                    with torch.no_grad():
                        compiled(torch.zeros((2, *want), device=self.device))
                    self._net = compiled
                    self.note = "torch.compile on"
                except Exception as exc:               # no Triton / no compiler: plain PyTorch
                    self.note = f"torch.compile not available ({type(exc).__name__})"
        else:
            import onnxruntime as ort

            providers = [p for p in ("CUDAExecutionProvider",) if p in ort.get_available_providers()
                         and self.device is not None and self.device.type == "cuda"] + ["CPUExecutionProvider"]
            self._sess = ort.InferenceSession(str(path), providers=providers)
            self.kind = "onnx"

    @property
    def has_model(self) -> bool:
        return self.kind in ("torch", "onnx")

    def describe(self) -> str:
        where = "no PyTorch: one burst at a time on the CPU" if self.device is None else f"batched on {self.device}"
        model = {"torch": f"PyTorch on {self.device}", "onnx": "ONNX Runtime, batched", "physics": "none"}[self.kind]
        return f"physics rules {where}; model: {model}" + (f" ({self.note})" if self.note else "")

    # -- the two witnesses ------------------------------------------------------------------
    def _physics(self, x: np.ndarray, staged: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if staged is None:                              # no PyTorch on this machine: the original function, burst by burst
            res = [heuristic_classify(np.asarray(w, dtype=np.float32), self.cfg) for w in x]
            return (np.array([GESTURE_LABELS.index(r.label) for r in res], dtype=np.int64),
                    np.array([r.confidence for r in res]), np.stack([r.probs for r in res]) if res else np.zeros((0, 5)))
        import torch

        with torch.no_grad():
            parts = [heuristic_batch(staged[i:i + CHUNK], self.cfg) for i in range(0, len(staged), CHUNK)]
            label = torch.cat([p["label"] for p in parts]).cpu().numpy()
            conf = torch.cat([p["conf"] for p in parts]).double().cpu().numpy()
            probs = torch.cat([p["probs"] for p in parts]).double().cpu().numpy()
        return label, conf, probs

    def _model(self, x: np.ndarray, staged: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.kind == "torch":
            import torch

            with torch.no_grad(), exact_gpu_maths():
                logits = torch.cat([self._net(staged[i:i + CHUNK].float()) for i in range(0, len(staged), CHUNK)])
                probs = torch.softmax(logits.double() / self.temperature, dim=1)
                conf, label = probs.max(dim=1)
                return label.cpu().numpy(), conf.cpu().numpy(), probs.cpu().numpy()
        logits = np.concatenate([self._sess.run(None, {"input": np.ascontiguousarray(x[i:i + 1024], dtype=np.float32)})[0]
                                 for i in range(0, len(x), 1024)]).astype(np.float64)
        z = logits / self.temperature
        z -= z.max(axis=1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return probs.argmax(axis=1), probs.max(axis=1), probs

    def __call__(self, x: np.ndarray) -> Verdicts:
        """x: [N, C, F, T] pictures exactly as the engine classifies them (the un-padded window)."""
        n = len(x)
        k = max(len(self.labels), 1)
        if n == 0:
            empty = Verdicts(np.zeros(0, np.int64), np.zeros(0), np.zeros((0, len(GESTURE_LABELS))), labels=self.labels)
            if self.has_model:
                empty.model, empty.model_conf, empty.model_probs = np.zeros(0, np.int64), np.zeros(0), np.zeros((0, k))
            return empty
        seconds: dict[str, float] = {}
        staged = None
        if self.device is not None:
            import torch

            t0 = time.perf_counter()
            staged = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)        # once; everything below stays on the device
            seconds["to device"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = Verdicts(*self._physics(x, staged), labels=self.labels)
        seconds["physics rules"] = time.perf_counter() - t0
        if self.has_model:
            t0 = time.perf_counter()
            out.model, out.model_conf, out.model_probs = self._model(x, staged)
            seconds["model"] = time.perf_counter() - t0
        out.seconds = seconds
        return out

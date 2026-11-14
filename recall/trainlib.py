"""trainlib.py — a small training loop and honest scoring for the recall experiments.

Plain words: this trains the same small network the engine runs (sonar.model.GestureNet) with the same
augmentation code (sonar.batch_augment), but with two differences from sonar.train that the experiments need:
  * every example has its own sampling weight, so real recordings and synthetic clips can be mixed in a chosen
    ratio instead of whatever their counts happen to be;
  * nothing is ever chosen by looking at the test person: a fixed number of steps, the last model, and a
    confidence temperature that comes from elsewhere.
Read-only on the data; writes nothing by itself.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .common import REPO

sys.path.insert(0, str(REPO))

from sonar.config import SonarConfig  # noqa: E402

CFG = SonarConfig(n_mics=1)          # the MacBook sessions: default profile, one microphone


@dataclass
class TrainSpec:
    epochs: int = 60
    steps: int = 0                   # steps per epoch; 0 = the released rule max(20, ceil(n_train / batch))
    batch: int = 0                   # 0 = the released default: 256 on CUDA, 64 elsewhere
    lr: float = 0.0                  # 0 = the released rule 2e-3 * sqrt(batch / 64)
    lr_scale: float = 1.0            # multiplies that learning rate (gentler fine-tuning)
    width: int = 24
    seed: int = 0
    strength: float = 1.0            # augmentation strength (sonar.batch_augment)
    label_smoothing: float = 0.05
    weight_decay: float = 1e-3
    dropout: float = 0.3
    threads: int = 4
    device: str = "auto"             # auto | cuda | mps | cpu
    aug: dict = field(default_factory=dict)      # extra augmentation options (see recall.augment)


def pick_device(name: str = "auto") -> Any:
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")       # MPS only when asked for: this Mac is shared


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train(x: Any, y: Any, w: Any, labels: tuple[str, ...], spec: TrainSpec,
          init: Any = None, cfg: SonarConfig = CFG, verbose: bool = False) -> Any:
    """Pictures x [N, C, 64, 96] (float16, numpy or a tensor already on the device), labels y [N], sampling
    weights w [N] (any scale) -> trained model (eval mode, on the CPU).

    Same network, optimiser, schedule, label smoothing and augmentation code as sonar.train. On CUDA: data
    resident on the GPU, bf16 autocast, batch 256 with the learning rate scaled by sqrt(batch / 64), as there.
    """
    import torch
    from torch import nn

    from sonar.model import GestureNet

    from .augment import make_augmenter

    device = pick_device(spec.device)
    cuda = device.type == "cuda"
    if not cuda:
        torch.set_num_threads(spec.threads)
    torch.manual_seed(spec.seed)
    if cuda:
        torch.cuda.manual_seed(spec.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    batch = spec.batch or (256 if cuda else 64)
    lr = (spec.lr or 2e-3 * float(np.sqrt(batch / 64))) * spec.lr_scale
    xt = x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x, dtype=np.float16))
    yt = y if torch.is_tensor(y) else torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
    xt, yt = xt.to(device), yt.to(device)
    wt = w if torch.is_tensor(w) else torch.from_numpy(np.asarray(w, dtype=np.float64))
    p = (wt.double() / wt.double().sum()).float().to(device)
    n_train = int((p > 0).sum())
    steps = spec.steps or max(20, int(np.ceil(n_train / batch)))
    model = GestureNet(xt.shape[1], len(labels), width=spec.width, dropout=spec.dropout).to(device)
    if init is not None:
        model.load_state_dict(init)
    aug = make_augmenter(cfg, list(labels), device, strength=spec.strength, **spec.aug)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=spec.weight_decay, **({"fused": True} if cuda else {}))
    total = spec.epochs * steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=spec.label_smoothing)
    use_amp = cuda and torch.cuda.is_bf16_supported()
    t0 = time.time()
    model.train()
    for step in range(total):
        i = torch.multinomial(p, batch, replacement=True)
        xb, yb = aug(xt[i], yt[i])
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(xb)
            loss = loss_fn(out.float(), yb)
        else:
            loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step + 1) % max(1, total // 6) == 0:
            print(f"    step {step + 1}/{total} loss {float(loss.detach()):.3f} ({time.time() - t0:.0f} s)", flush=True)
    return model.float().cpu().eval()


def logits_of(model: Any, x: np.ndarray, pad: int = 8, n_t: int = 80, device: str = "cpu") -> np.ndarray:
    """Logits for the un-shifted window (exactly what the engine would classify), always in float32."""
    import torch

    dev = torch.device(device)
    model = model.to(dev)
    xc = np.ascontiguousarray(np.asarray(x)[..., pad:pad + n_t], dtype=np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(xc), 1024):
            out.append(model(torch.from_numpy(xc[i:i + 1024]).to(dev)).float().cpu().numpy())
    model.cpu()
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature (>= 1) that makes the confidences honest on data the model was NOT trained on."""
    best_t, best = 1.0, np.inf
    for t in np.exp(np.linspace(0.0, np.log(8.0), 80)):
        p = softmax(logits, t)
        nll = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
        if nll < best:
            best_t, best = float(t), float(nll)
    return best_t


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval for a share k/n."""
    if n == 0:
        return 0.0, 1.0
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def score(logits: np.ndarray, y: np.ndarray, labels: tuple[str, ...], t: float = 1.0, thr: float = 0.6) -> dict:
    """The released pipeline's two headline numbers plus breakdowns.

    recall  = share of real gestures named correctly with confidence >= thr
    false   = share of non-gesture examples that would fire some gesture with confidence >= thr
    """
    p = softmax(logits, t)
    arg = p.argmax(axis=1)
    pred = np.where(p.max(axis=1) >= thr, arg, 0)
    g = y != 0
    k = len(labels)
    conf = np.zeros((k, k), int)
    for a, b in zip(y, pred):
        conf[a, b] += 1
    raw = np.zeros((k, k), int)
    for a, b in zip(y, arg):
        raw[a, b] += 1
    hit, n_g = int((pred[g] == y[g]).sum()), int(g.sum())
    fa, n_n = int((pred[~g] != 0).sum()), int((~g).sum())
    return {
        "recall": hit / max(n_g, 1), "recall_ci": wilson(hit, n_g), "n_gesture": n_g, "hits": hit,
        "false": fa / max(n_n, 1), "false_ci": wilson(fa, n_n), "n_none": n_n, "false_n": fa,
        "acc_raw": float((arg == y).mean()) if len(y) else 0.0,
        "per_class": {labels[c]: (int(conf[c, c]), int(conf[c].sum())) for c in range(1, k)},
        "confusion": conf, "confusion_raw": raw,
        "wrong_gesture": int(((pred[g] != y[g]) & (pred[g] != 0)).sum()),      # gestures answered with ANOTHER gesture
        "silent": int((pred[g] == 0).sum()),                                   # gestures answered with nothing
    }


def recall_at_false(logits: np.ndarray, y: np.ndarray, budget: float, t: float = 1.0) -> tuple[float, float]:
    """(best recall, threshold) over confidence thresholds whose false-trigger share stays <= budget.

    This looks at the test labels, so it is an upper bound on what threshold tuning could give: report it as such.
    """
    p = softmax(logits, t)
    arg, conf = p.argmax(axis=1), p.max(axis=1)
    g = y != 0
    best = (0.0, 1.0)
    for thr in np.unique(np.concatenate([conf, [0.0, 1.01]])):
        pred = np.where(conf >= thr, arg, 0)
        if (~g).sum() and (pred[~g] != 0).mean() > budget:
            continue
        r = float((pred[g] == y[g]).mean()) if g.any() else 0.0
        if r > best[0]:
            best = (r, float(thr))
    return best

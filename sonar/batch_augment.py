"""batch_augment.py — the training augmentation for a whole batch at once, on the training device.

Plain words: this makes the same random, physically sensible variations as
`dataset.Augmenter` (faster/slower, time shift, left<->right mirror, played
backwards, louder/quieter, fresh noise speckle, a click, a small mask), with the
same probabilities and the same maths, but for a whole batch in one go with
torch. On a GPU the batch never leaves the card, so the training loop no longer
waits for a Python loop on one CPU core (that loop managed about 2,350 examples a
second on the Mac Studio, and it ran in turn with every GPU step).

Every transform is checked against the numpy version in tests/test_training.py.
Each one is applied to every example and then kept only where its coin flip
said so (an unselected example gets the identity, e.g. a stretch by exactly 1),
so the GPU never has to report back to the CPU in the middle of a step.

Needs PyTorch (training machine only). Read-only: no files, no audio devices.
"""

from __future__ import annotations

import torch

from .config import SonarConfig
from .dataset import PAD


def _lerp_axis(x: torch.Tensor, pos: torch.Tensor, dim: int) -> torch.Tensor:
    """Linear interpolation of x [B, C, F, T] along `dim` at positions pos [B, n]; 0 outside [0, size-1].

    Same rule as scipy.ndimage with order=1, mode="constant", cval=0.
    """
    size = x.shape[dim]
    lo = pos.floor()
    frac = pos - lo
    lo_i = lo.long()
    inside = (pos >= 0) & (pos <= size - 1)
    shape = [x.shape[0], 1, 1, 1]
    shape[dim] = pos.shape[1]
    full = list(x.shape)
    full[dim] = pos.shape[1]
    g0 = x.gather(dim, lo_i.clamp(0, size - 1).view(shape).expand(full))
    g1 = x.gather(dim, (lo_i + 1).clamp(0, size - 1).view(shape).expand(full))
    frac = frac.view(shape)
    return (g0 * (1.0 - frac) + g1 * frac) * inside.view(shape)


class BatchAugmenter:
    """`dataset.Augmenter` for a batch: x [B, C, F, T + 2*pad] -> [B, C, F, T] float32, labels [B]."""

    def __init__(self, cfg: SonarConfig, labels: list[str], pad: int = PAD, strength: float = 1.0,
                 device: torch.device | str = "cpu") -> None:
        dev = torch.device(device)
        labels = list(labels)
        self.cfg = cfg
        self.pad = pad
        self.k = strength
        self.n_p = cfg.n_power_channels
        self.has_lat = cfg.has_lateral
        self.can_mirror = cfg.n_mics == 2
        # label after a left<->right exchange: swipe_left <-> swipe_right, everything else unchanged
        swap = list(range(len(labels)))
        if "swipe_left" in labels and "swipe_right" in labels:
            a, b = labels.index("swipe_left"), labels.index("swipe_right")
            swap[a], swap[b] = b, a
        none = labels.index("none") if "none" in labels else -1
        self.swap = torch.tensor(swap, dtype=torch.long, device=dev)
        # only swipes and "none" may be played backwards (a backwards push is a pull)
        self.reversible = torch.tensor([swap[i] != i or i == none for i in range(len(labels))], device=dev)
        # channel order after exchanging left<->right: tones swapped (if on different speakers), mics swapped
        tones = list(range(cfg.n_tones))
        if cfg.n_tones == 2 and cfg.tone_channels[0] != cfg.tone_channels[1]:
            tones = [1, 0]
        mics = list(range(cfg.n_mics))[::-1]
        order = [t * cfg.n_mics + m for t in tones for m in mics] + list(range(self.n_p, cfg.n_feat_channels))
        self.mirror_idx = torch.tensor(order, dtype=torch.long, device=dev)
        f = cfg.doppler_bins
        self.flip_f = torch.tensor([(f - i) % f for i in range(f)], dtype=torch.long, device=dev)   # f -> -f

    # -- individual transforms (fixed parameters, so they can be compared with the numpy version) --
    def stretch(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Each example done s times faster: shorter in time, proportionally higher Doppler (s = 1: unchanged)."""
        _, _, n_f, n_t = x.shape
        f0 = n_f / 2.0                                                    # the 0 Hz bin stays put
        t0 = n_t - 1 - self.pad - self.cfg.seg_hang_frames                # ... and so does the end of the burst
        f = torch.arange(n_f, device=x.device, dtype=torch.float32)
        t = torch.arange(n_t, device=x.device, dtype=torch.float32)
        out = _lerp_axis(x, f0 + (f[None, :] - f0) / s[:, None], dim=2)   # output(f, t) = input(f0 + (f - f0)/s,
        out = _lerp_axis(out, t0 + (t[None, :] - t0) * s[:, None], dim=3)  #                       t0 + (t - t0)*s)
        if self.has_lat:                                                  # sideways Doppler grows with speed too
            out[:, self.n_p] = (out[:, self.n_p] * s[:, None, None]).clamp(-1.0, 1.0)
        return out

    def crop(self, x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Cut the model-sized window out of the padded picture (shift 0 = exactly what the engine sees)."""
        n_t = self.cfg.gesture_frames
        start = (self.pad - shift).clamp(0, x.shape[-1] - n_t)
        idx = start[:, None] + torch.arange(n_t, device=x.device)[None, :]
        return x.gather(3, idx[:, None, None, :].expand(x.shape[0], x.shape[1], x.shape[2], n_t))

    def mirror(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Exchange left and right: a swipe left becomes a swipe right."""
        out = x[:, self.mirror_idx]
        if self.has_lat:
            out[:, self.n_p] = -out[:, self.n_p]
        return out, self.swap[y]

    def reverse(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Play the movement backwards: time reversed, every speed sign reversed."""
        out = x[:, :, self.flip_f].flip(-1)
        if self.has_lat:
            out[:, self.n_p] = -out[:, self.n_p]
        return out, self.swap[y]

    def gain(self, x: torch.Tensor, db: torch.Tensor) -> torch.Tensor:
        """Echo `db` louder or quieter (features are dB above the floor / feat_range_db)."""
        step = (db / self.cfg.feat_range_db).view(-1, 1, 1, 1)
        p = x[:, :self.n_p]
        p = torch.where(p > 0, (p + step).clamp(0.0, 1.0), torch.zeros_like(p))
        if not self.has_lat:
            return p
        # the lateral picture is masked by echo strength: fade it where the (new) echo got weaker
        mean = p.mean(dim=1)
        scale = ((mean + step[:, 0]) / (mean + 1e-6)).clamp(0.0, 1.0)
        fade = torch.where((db < 0).view(-1, 1, 1), torch.clamp(scale * 3.0, max=1.0), torch.ones_like(scale))
        return torch.cat([p, (x[:, self.n_p] * fade)[:, None], x[:, self.n_p + 1:]], dim=1)

    def noise(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Fresh noise-floor speckle from exponential samples e [B, n_power, F, T]."""
        cfg = self.cfg
        speckle = ((10.0 * torch.log10(e + 1e-12) - cfg.feat_floor_margin_db) / cfg.feat_range_db).clamp(0.0, 1.0)
        return torch.cat([torch.maximum(x[:, :self.n_p], speckle), x[:, self.n_p:]], dim=1)

    def click(self, x: torch.Tensor, t: torch.Tensor, width: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        """A broadband click: frames [t, t + width) of the power pictures raised to at least `level`."""
        frames = torch.arange(x.shape[-1], device=x.device)[None, :]
        hit = ((frames >= t[:, None]) & (frames < (t + width)[:, None]))[:, None, None, :]
        p = x[:, :self.n_p]
        p = torch.where(hit, torch.maximum(p, level.view(-1, 1, 1, 1)), p)
        return torch.cat([p, x[:, self.n_p:]], dim=1)

    def mask(self, x: torch.Tensor, in_time: torch.Tensor, start: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
        """Zero a short time strip (in_time) or a few Doppler bins, in every channel."""
        frames = torch.arange(x.shape[-1], device=x.device)[None, :]
        bins = torch.arange(x.shape[-2], device=x.device)[None, :]
        end = (start + width)[:, None]
        cut_t = in_time[:, None] & (frames >= start[:, None]) & (frames < end)
        cut_f = ~in_time[:, None] & (bins >= start[:, None]) & (bins < end)
        keep = ~(cut_t[:, None, None, :] | cut_f[:, None, :, None])
        return torch.where(keep, x, torch.zeros_like(x))

    # -- the full recipe (same order and probabilities as dataset.Augmenter.__call__) ---------------
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.k
        x = x.float()
        n, dev = x.shape[0], x.device
        n_f, n_t = self.cfg.doppler_bins, self.cfg.gesture_frames

        def coin(p: float) -> torch.Tensor:
            return torch.rand(n, device=dev) < p

        def pick(sel: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.where(sel.view(-1, *([1] * (a.dim() - 1))), a, b)

        # faster / slower by up to +-18 % (half of the examples; the others are stretched by exactly 1)
        factor = torch.exp(torch.empty(n, device=dev).uniform_(-0.18, 0.18))
        x = self.stretch(x, torch.where(coin(0.5 * k), factor, torch.ones_like(factor)))
        # time shift by up to +-pad frames while cutting out the model window
        shift = torch.randint(-self.pad, self.pad + 1, (n,), device=dev) if k > 0 else torch.zeros(n, dtype=torch.long, device=dev)
        x = self.crop(x, shift)
        if self.can_mirror:
            sel = coin(0.5 * min(k, 1.0))
            xm, ym = self.mirror(x, y)
            x, y = pick(sel, xm, x), pick(sel, ym, y)
        sel = self.reversible[y] & coin(0.2 * k)
        xr, yr = self.reverse(x, y)
        x, y = pick(sel, xr, x), pick(sel, yr, y)
        x = pick(coin(0.7 * k), self.gain(x, torch.empty(n, device=dev).uniform_(-9.0, 6.0)), x)
        e = -torch.log1p(-torch.rand(n, self.n_p, n_f, n_t, device=dev))          # exponential(1), never inf
        x = pick(coin(0.5 * k), self.noise(x, e), x)
        x = pick(coin(0.15 * k), self.click(x, torch.randint(0, n_t - 3, (n,), device=dev),
                                            torch.randint(1, 4, (n,), device=dev),
                                            torch.empty(n, device=dev).uniform_(0.1, 0.4)), x)
        in_time = coin(0.5)
        start = torch.where(in_time, torch.randint(0, n_t - 6, (n,), device=dev), torch.randint(0, n_f - 4, (n,), device=dev))
        width = torch.where(in_time, torch.randint(2, 7, (n,), device=dev), torch.randint(1, 5, (n,), device=dev))
        x = pick(coin(0.25 * k), self.mask(x, in_time, start, width), x)
        return x.contiguous(), y

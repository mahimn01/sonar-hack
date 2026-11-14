"""augment.py — the training augmentation used by the recall experiments (batched, on the training device).

Plain words: `make_augmenter` returns the released augmenter (sonar.batch_augment.BatchAugmenter) unless an
experiment asks for a change. The one change so far is the SPEED range: the released recipe makes a gesture
up to 18 % faster or slower, but the recorded people differ by about 5x (lance's swipes shift the tones by
+-500..700 Hz, mahimn's by 100..250 Hz), so an experiment can widen it.

Options (all optional):
    speed=(lo, hi)     stretch factor range (log-uniform), e.g. (0.6, 1.8); default the released exp(+-0.18)
    speed_p=0.5        share of examples that are stretched (released: 0.5)
    reverse_p=0.2      share of swipes / "none" pictures played backwards with the label exchanged (released: 0.2)
    gain=(-9.0, 6.0)   echo level change in dB (released)
Read-only: no files, no audio devices.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from sonar.batch_augment import BatchAugmenter
from sonar.config import SonarConfig


class TunableAugmenter(BatchAugmenter):
    """BatchAugmenter with the speed range, reverse share and gain range exposed. Same order of steps."""

    def __init__(self, cfg: SonarConfig, labels: list[str], device: Any, strength: float = 1.0,
                 speed: tuple[float, float] | None = None, speed_p: float = 0.5, reverse_p: float = 0.2,
                 gain: tuple[float, float] = (-9.0, 6.0)) -> None:
        super().__init__(cfg, labels, strength=strength, device=device)
        self.speed = (math.exp(-0.18), math.exp(0.18)) if speed is None else (float(speed[0]), float(speed[1]))
        self.speed_p, self.reverse_p, self.gain_range = float(speed_p), float(reverse_p), (float(gain[0]), float(gain[1]))

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.k
        x = x.float()
        n, dev = x.shape[0], x.device
        n_f, n_t = self.cfg.doppler_bins, self.cfg.gesture_frames

        def coin(p: float) -> torch.Tensor:
            return torch.rand(n, device=dev) < p

        def pick(sel: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.where(sel.view(-1, *([1] * (a.dim() - 1))), a, b)

        lo, hi = math.log(self.speed[0]), math.log(self.speed[1])
        factor = torch.exp(torch.empty(n, device=dev).uniform_(lo, hi))
        x = self.stretch(x, torch.where(coin(self.speed_p * k), factor, torch.ones_like(factor)))
        shift = torch.randint(-self.pad, self.pad + 1, (n,), device=dev) if k > 0 else torch.zeros(n, dtype=torch.long, device=dev)
        x = self.crop(x, shift)
        if self.can_mirror:
            sel = coin(0.5 * min(k, 1.0))
            xm, ym = self.mirror(x, y)
            x, y = pick(sel, xm, x), pick(sel, ym, y)
        sel = self.reversible[y] & coin(self.reverse_p * k)
        xr, yr = self.reverse(x, y)
        x, y = pick(sel, xr, x), pick(sel, yr, y)
        x = pick(coin(0.7 * k), self.gain(x, torch.empty(n, device=dev).uniform_(*self.gain_range)), x)
        e = -torch.log1p(-torch.rand(n, self.n_p, n_f, n_t, device=dev))
        x = pick(coin(0.5 * k), self.noise(x, e), x)
        x = pick(coin(0.15 * k), self.click(x, torch.randint(0, n_t - 3, (n,), device=dev),
                                            torch.randint(1, 4, (n,), device=dev),
                                            torch.empty(n, device=dev).uniform_(0.1, 0.4)), x)
        in_time = coin(0.5)
        start = torch.where(in_time, torch.randint(0, n_t - 6, (n,), device=dev), torch.randint(0, n_f - 4, (n,), device=dev))
        width = torch.where(in_time, torch.randint(2, 7, (n,), device=dev), torch.randint(1, 5, (n,), device=dev))
        x = pick(coin(0.25 * k), self.mask(x, in_time, start, width), x)
        return x.contiguous(), y


def make_augmenter(cfg: SonarConfig, labels: list[str], device: Any, strength: float = 1.0, **opts: Any) -> Any:
    """The released augmenter when no option is given (bit-for-bit the same code path), else the tunable one."""
    if not opts:
        return BatchAugmenter(cfg, labels, strength=strength, device=device)
    return TunableAugmenter(cfg, labels, device, strength=strength, **opts)

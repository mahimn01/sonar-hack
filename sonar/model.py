"""model.py — the gesture classifier: a small convolutional network.

Plain words: the input is a stack of small pictures (Doppler frequency on one
axis, 1.6 s of time on the other; one picture per tone and microphone plus one
"which way sideways" picture). Four convolution blocks look for shapes in those
pictures (a ridge that goes up then down = push; opposite ridges on the two
tones = swipe), then one linear layer names the gesture.

Why this small (about 135 000 weights): we only have a few hundred real
examples, the classes differ in simple physical ways, and it must run on a
laptop CPU in well under a millisecond.

Needs PyTorch (training machine only). The demo laptop runs the exported ONNX
file through `classifier.py` and never imports this module.
"""

from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class GestureNet(nn.Module):
    """Input [B, C, 64, 80] (channels, Doppler bins, frames) -> logits [B, n_classes]."""

    def __init__(self, in_channels: int, n_classes: int, width: int = 24, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(in_channels, width),          # 64x80 -> 32x40
            _block(width, 2 * width),            # -> 16x20
            _block(2 * width, 4 * width),        # -> 8x10
            _block(4 * width, 4 * width),        # -> 4x5
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * width, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_device(name: str = "auto") -> torch.device:
    """cuda (RTX 4090) > mps (Mac Studio) > cpu, unless a device is named explicitly."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

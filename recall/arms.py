"""arms.py — the experiments ("arms"), each one a recipe for training on a table of examples.

How to run:   .venv/bin/python -m recall.arms <arm> [--seeds 0,1,2] [--nested]
Every arm is scored by recall.lopo.run_lopo (leave one human out). Read-only on the data.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .data import LABELS, Table, all_official
from .lopo import run_lopo
from .trainlib import TrainSpec, train


def class_balanced(y: np.ndarray, n_classes: int = len(LABELS)) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return 1.0 / np.maximum(counts[y], 1.0)


def fit_real_only(spec: TrainSpec):
    """Only the real recordings of the training humans, every class drawn equally often, from scratch."""
    def fit(t: Table, seed: int) -> Any:
        y = t.y()
        s = TrainSpec(**{**spec.__dict__, "seed": seed})
        return train(t.X, y, class_balanced(y), LABELS, s)
    return fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    table = all_official()
    if args.arm == "real_only":
        run_lopo("real_only", table, fit_real_only(TrainSpec(epochs=args.epochs)), seeds=seeds, nested=args.nested)
    else:
        raise SystemExit(f"unknown arm {args.arm}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# train_4090.sh — one command for the whole gesture-model training (Linux / macOS).
#
# What it does: synthetic data -> pre-training -> fine-tuning on every recorded session
# under data/ (if any) -> export to models/ and models-release/ -> engine check.
# All the logic lives in scripts/train_4090.py; this file only finds the project's Python.
#
# How to run, from anywhere:
#   scripts/train_4090.sh                              # full run on the GPU (CUDA)
#   scripts/train_4090.sh --sweep                      # also try widths 16/24/32 x seeds 0/1/2
#   scripts/train_4090.sh --device cpu --size tiny     # few-minute smoke test without a GPU
#   scripts/train_4090.sh --help                       # every option
#
# What it changes: train_out/, models/, models-release/. Reads data/ only. No audio devices.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "No Python environment at .venv/ - do TRAINING.md steps 2 and 3 first." >&2
  exit 1
fi
exec "$PY" scripts/train_4090.py "$@"

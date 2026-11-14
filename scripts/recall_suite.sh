#!/usr/bin/env bash
# recall_suite.sh - the same as recall_suite.ps1 for Linux / macOS.
# How to run (from the repo folder):   scripts/recall_suite.sh            (full run on CUDA)
#                                      scripts/recall_suite.sh --device cpu --smoke
# Writes only under train_out/recall/.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m recall.suite run "$@"

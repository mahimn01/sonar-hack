#!/usr/bin/env bash
# run_best_mac.sh — start the sonar engine with the current best gesture model (the one-microphone MacBook model).
#
# What it does: runs `python -m sonar.engine` with the model in models-release-mac-recall/ and the engine settings
# that the recordings support on a MacBook:
#   * one microphone channel;
#   * a movement burst also ends once it is 12 dB below its own peak (SONAR_SEG_REL_OFF_DB=12). On this MacBook the
#     hand is so loud that the engine's burst otherwise runs into its 1.4 s limit and is chopped at an arbitrary
#     point (111 of 218 recorded gestures). With the same model and no retraining this alone took the held-out
#     person from 0.65 to 0.73 of gestures recognised, with fewer false triggers;
#   * range gate off (it blocked fast swipes and never blocked a non-gesture);
#   * "somebody is walking in" gate limited to 3 s (it used to swallow gestures for up to 41 s).
# It plays the near-inaudible 18.5-21.5 kHz sonar signal on the speakers, listens on the microphone, prints every
# gesture and serves events on ws://127.0.0.1:8765 for the UI. Stop with Ctrl+C. It writes no files.
#
# How to run (from anywhere):
#   scripts/run_best_mac.sh                      on a MacBook: its own speakers and microphone
#   scripts/run_best_mac.sh --backend sim        no sound: the simulated visitor (checks that everything loads)
#   scripts/run_best_mac.sh -v                   also print one status line per second
# On a Mac that is not a MacBook the system's default microphone and speakers are used; choose others with
#   SONAR_IN_DEVICE="part of the name" SONAR_OUT_DEVICE="part of the name" scripts/run_best_mac.sh
#
# The model was trained on ONE laptop (Lance's MacBook Pro). On any other speakers/microphone it is not expected to
# work: left/right comes from where that laptop's two speakers and its microphone sit.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1            # so every line reaches a log file at once when the output is piped
export SONAR_SEG_REL_OFF_DB="${SONAR_SEG_REL_OFF_DB:-12}"   # peak-relative burst end (0 = the engine's old cutting)
DEVICES=()
if [[ "$(sysctl -n hw.model 2>/dev/null)" == MacBook* ]]; then
  DEVICES=(--in-device "${SONAR_IN_DEVICE:-MacBook}" --out-device "${SONAR_OUT_DEVICE:-MacBook}")
else
  [ -n "${SONAR_IN_DEVICE:-}" ] && DEVICES+=(--in-device "$SONAR_IN_DEVICE")
  [ -n "${SONAR_OUT_DEVICE:-}" ] && DEVICES+=(--out-device "$SONAR_OUT_DEVICE")
fi
exec .venv/bin/python -m sonar.engine --model models-release-mac-recall/gesture.onnx --mics 1 --no-range-gate --entering-gate-s 3 ${DEVICES[@]+"${DEVICES[@]}"} "$@"

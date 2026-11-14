#!/usr/bin/env bash
# mac_record_all.sh — guided recording of one person on the MacBook: scene, then gestures, then breathing.
#
# What it does: asks for the person's name and how many repetitions of each gesture, optionally runs the
# 1-minute speaker/microphone self-test, then records three sessions with sonar/record.py, pausing before
# each one so the person can get ready:
#   1. scene      empty room, walking in, sitting still, typing and reaching, walking out   (~3.5 min)
#   2. gestures   swipe left, swipe right, push, REPS times each, shuffled                (~0.2 min per trial)
#   3. breathing  paced breathing at 12, 15 and 20 breaths per minute                     (~2.5 min)
# Why on the MacBook itself: a model only works on the laptop whose recordings trained it (the HP's
# recordings sound different and have two microphones; a MacBook gives one).
#
# How to run, from anywhere (one person per run; give every person their own name):
#   scripts/mac_record_all.sh
# Optional settings (environment variables):
#   SONAR_PROFILE=strong      signal profile, if the self-test recommends one (use the same one for training and the demo)
#   SONAR_GESTURES=swipe_left,swipe_right,push,pull   to record pull as well
#   SONAR_IN_DEVICE / SONAR_OUT_DEVICE   part of the device names (default "MacBook" = the built-in mic and speakers)
#   SONAR_OUT=data            where the session folders go
#   SONAR_BACKEND=sim         rehearse the whole flow with the simulator (no sound, nothing recorded for real)
#
# What it changes: creates data/<date-time>_<person>_<script>/ (audio.wav + session.json) per session.
# While recording it plays a near-inaudible 18.5-21.5 kHz signal on the speakers and uses the microphone.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "No Python environment at .venv/ - see TRAINING.md, 'Demo on a MacBook', step 1." >&2
  exit 1
fi
PROFILE="${SONAR_PROFILE:-default}"
GESTURES="${SONAR_GESTURES:-swipe_left,swipe_right,push}"
IN_DEV="${SONAR_IN_DEVICE:-MacBook}"
OUT_DEV="${SONAR_OUT_DEVICE:-MacBook}"
OUT="${SONAR_OUT:-data}"
DEVICES=(--in-device "$IN_DEV" --out-device "$OUT_DEV")
FAST=()
if [ "${SONAR_BACKEND:-live}" = "sim" ]; then
  DEVICES=(--backend sim)
  FAST=(--fast)
  echo "(rehearsal with the simulator: no sound is played and nothing real is recorded)"
fi

echo "Guided sonar recording on this Mac. Before you start:"
echo "  * disconnect AirPods / Bluetooth headphones; use the MacBook's own speakers and microphone"
echo "  * set the volume once (start around 60 %) and do not change it again: recording, training and demo"
echo "    must all use the same volume. Charger in. Lid at the angle it will have in the demo."
echo "  * the first time, macOS asks to let the Terminal use the microphone: allow it"
echo "    (otherwise the recording is silent: System Settings > Privacy & Security > Microphone)"
echo

read -r -p "Person's name (one word, letters/digits, e.g. alice): " PERSON
if [[ ! "$PERSON" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Use one word: letters, digits, - or _ only." >&2
  exit 1
fi
read -r -p "Repetitions of each gesture [12]: " REPS
REPS="${REPS:-12}"
if [[ ! "$REPS" =~ ^[0-9]+$ ]] || [ "$REPS" -lt 1 ]; then
  echo "Repetitions must be a whole number, 1 or more." >&2
  exit 1
fi
read -r -p "Notes for these sessions (seat distance, volume, room) [Enter to skip]: " NOTES
read -r -p "Run the 1-minute speaker/microphone self-test first? (do it once per day / after moving the laptop) [y/N]: " SELFTEST

if [[ "$SELFTEST" =~ ^[Yy] ]]; then
  echo
  echo "=== self-test: keep quiet and still for a minute"
  if ! "$PY" -m sonar.selftest --profile "$PROFILE" "${DEVICES[@]}"; then
    echo
    echo "The self-test reported a problem (see its lines above; README section 1 says what to change)."
  fi
  read -r -p "Continue with the recordings? [Y/n]: " GO
  if [[ "$GO" =~ ^[Nn] ]]; then
    exit 0
  fi
fi

# record <script> [extra record.py options]: one session, after the person says they are ready.
# Ctrl+C during a session ends that session early and keeps what was recorded; Ctrl+C at a prompt quits.
record() {
  local script=$1
  shift
  echo
  read -r -p ">>> Next: the $script session for $PERSON. Press Enter when ready... " _
  "$PY" -m sonar.record --person "$PERSON" --script "$script" --profile "$PROFILE" --notes "$NOTES" --out "$OUT" \
    "${DEVICES[@]}" ${FAST[@]+"${FAST[@]}"} "$@"
}

record scene
record gestures --reps "$REPS" --gestures "$GESTURES"
record breathing

echo
echo "Done: three sessions for $PERSON are in $OUT/:"
ls -d "$OUT"/*_"$PERSON"_* 2>/dev/null | tail -3 || true
echo
echo "Next person: run scripts/mac_record_all.sh again with their name."
echo "Train here on the MacBook:  scripts/train_4090.sh --device mps --size small --release models-release-mac"
echo "Or send data/ to the 4090:  see TRAINING.md, 'Demo on a MacBook'."

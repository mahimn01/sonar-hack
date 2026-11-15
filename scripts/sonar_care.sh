#!/usr/bin/env bash
# sonar_care.sh — start Sonar Care with one command.
#
# What it does:
#   1. makes sure there is a Python environment in .venv (creates it if missing);
#   2. installs the packages Sonar Care needs, only if some are missing (first run: about a minute);
#   3. starts the app: the sonar engine, the Sonar Care service, the menu bar item, and opens the dashboard.
#
# How to run (from anywhere):
#   scripts/sonar_care.sh                    live, on a MacBook: its own speakers and microphone
#   scripts/sonar_care.sh --backend sim      no sound: the engine's simulated visitor
#   scripts/sonar_care.sh --mock             no engine: scripted fake events, to rehearse the screens anywhere
#   scripts/sonar_care.sh --dry-run          show which key each gesture WOULD press, press nothing
#   scripts/sonar_care.sh --attach ws://127.0.0.1:8766     use an engine that is already running
#   scripts/sonar_care.sh --headless         no menu bar item
#   scripts/sonar_care.sh --help             every option
#
# What it changes on this machine:
#   * may create .venv and install Python packages INTO IT (nothing global);
#   * plays the near-inaudible 18.5-21.5 kHz sonar signal on the speakers (not with --mock / --attach / --backend sim);
#   * presses keys in the app in front when it sees a gesture (not with --dry-run). macOS asks once for the
#     Accessibility permission of the terminal you started this from;
#   * writes settings to ~/Library/Application Support/SonarCare/ and logs to ~/Library/Logs/SonarCare/.
# Stop with Ctrl+C or "Quit Sonar Care" in the menu: the engine it started stops too.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

say() { printf '[sonar-care] %s\n' "$*"; }
die() { printf '[sonar-care] ERROR: %s\n' "$*" >&2; exit 1; }

# ---- 1. the Python environment -------------------------------------------------------------------------------
if [ ! -x "$PY" ]; then
  say "no .venv yet: creating one"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
  else
    BASE=""
    for cand in python3.12 python3.13 python3.11 python3.10 python3; do
      found="$(command -v "$cand" 2>/dev/null || true)"
      [ -n "$found" ] || continue
      # /usr/bin/python3 is only a stub that pops up an Xcode install dialog when the command line tools are missing.
      if [ "$found" = "/usr/bin/python3" ] && ! xcode-select -p >/dev/null 2>&1; then continue; fi
      if "$found" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then BASE="$found"; break; fi
    done
    [ -n "$BASE" ] || die "no Python 3.10+ found. Install uv (no Xcode needed), then run this again:
       curl -LsSf https://astral.sh/uv/install.sh | sh"
    "$BASE" -m venv .venv
  fi
fi

pip_install() {
  if command -v uv >/dev/null 2>&1; then uv pip install --python "$PY" "$@"; else "$PY" -m pip install --disable-pip-version-check "$@"; fi
}
has() { "$PY" -c "import $1" >/dev/null 2>&1; }

# ---- 2. packages, only when something is missing ----------------------------------------------------------------
NEED_ENGINE=1
for arg in "$@"; do
  case "$arg" in --mock|--attach|--attach=*|--help|-h|--version) NEED_ENGINE=0 ;; esac
done
if ! has websockets; then
  say "installing the app's packages (first run)"
  pip_install -r app/requirements-care.txt || pip_install "websockets>=13" || die "could not install websockets. Check the internet connection."
fi
if [ "$NEED_ENGINE" = 1 ] && ! "$PY" -c "import numpy, scipy, sounddevice, soundfile, onnxruntime" >/dev/null 2>&1; then
  say "installing the sonar engine's packages (first run, about a minute)"
  pip_install -r requirements.txt || die "could not install the engine's packages (requirements.txt). Without them only --mock works."
fi
if [ "$(uname -s)" = "Darwin" ] && ! has rumps; then
  say "installing the menu bar library (optional)"
  pip_install -r app/requirements-care.txt >/dev/null 2>&1 || say "could not install it: Sonar Care will run without a menu bar item"
fi

# ---- 3. go -----------------------------------------------------------------------------------------------------
if [ "$NEED_ENGINE" = 1 ] && [ ! -f models-candidate-mac-calm50/gesture.onnx ] && [ ! -f models-release-mac-recall/gesture.onnx ]; then
  die "no gesture model found (models-candidate-mac-calm50/gesture.onnx). Run 'git pull', or try: scripts/sonar_care.sh --mock"
fi
exec "$PY" app/sonar_care.py "$@"

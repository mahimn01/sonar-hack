"""Sonar Care: start the app.

What it does
    Runs the Sonar Care service: touch-free control (gestures press keys), room monitoring
    (no-movement and no-breathing alerts) and the help gesture (two pushes), with the
    dashboard at http://127.0.0.1:8800/care.html. The code lives in app/care/.

How to run (from the repo root; scripts/sonar_care.sh does this for you)
    .venv/bin/python app/sonar_care.py --help
    .venv/bin/python app/sonar_care.py --headless --engine-port 8771 --dry-run

What it changes
    Presses keys in the frontmost app when a gesture is seen (not with --dry-run), shows
    notifications and plays sounds for alerts (not with --quiet), writes its settings file
    and an event log. See app/care/service.py for the details.
"""

from __future__ import annotations

from care.main import main

if __name__ == "__main__":
    raise SystemExit(main())

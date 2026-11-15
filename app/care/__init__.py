"""Sonar Care: the background service behind the menu bar app and the dashboard.

What it is
    A small Python service that sits between the sonar engine and the people using it.
    It listens to the engine's events and turns them into three things:
      1. touch-free control: a hand gesture presses a real key in whatever app is in front;
      2. room monitoring: caregiver alerts ("no movement", "no breathing");
      3. the help gesture: two pushes in a row raise a "help requested" alert.

How the pieces fit (every file starts with its own plain-English header)
    settings.py    what the user chose, saved as JSON
    rules.py       every decision, as pure functions of events and time (no I/O, unit-tested with a fake clock)
    actions.py     pressing keys and media keys on macOS, or only logging them in dry-run mode
    notify.py      macOS notification + sound for an alert, rate-limited
    client.py      the WebSocket connection to the engine (reconnects by itself)
    relay.py       a WebSocket the dashboard connects to: engine events + care events on one stream
    api.py         the local web server: the pages and a small JSON API
    supervisor.py  starts, watches and restarts the engine process
    service.py     wires all of the above together
    menubar.py     the macOS menu bar item
    main.py        the command line

Run it with scripts/sonar_care.sh (or `python app/sonar_care.py --help`).
"""

__version__ = "1.0.0"

"""main.py: the Sonar Care command line.

What it does
    Starts the Sonar Care service (see service.py), the local web server and, unless told
    otherwise, opens the dashboard. In this phase it attaches to an engine that is already
    running; starting and supervising the engine and the menu bar item arrive in phase 2.

How to run (from the repo root)
    .venv/bin/python app/sonar_care.py --headless --engine-port 8771 --dry-run
    .venv/bin/python app/sonar_care.py --attach ws://127.0.0.1:8766 --dry-run --no-browser

    --attach URL        connect to this engine address
    --engine-port N     connect to ws://127.0.0.1:N (default 8765)
    --port N            web server port (default 8800; the next free one is used if busy)
    --dry-run           log the key each gesture WOULD press, press nothing
    --quiet             no notification and no sound for alerts
    --headless          no menu bar item (also: --no-menubar)
    --no-browser        do not open the dashboard
    --lan               let other devices on the network view the dashboard and acknowledge alerts
    --settings PATH     use this settings file instead of the default one

What it changes
    See service.py. Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import __version__
from . import settings as S
from .api import make_server, serve_in_thread
from .service import Options, Service


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sonar_care", description="Sonar Care: touch-free control, room monitoring and a help gesture from the laptop's sonar.")
    ap.add_argument("--attach", metavar="URL", help="connect to an engine at this WebSocket address")
    ap.add_argument("--engine-port", type=int, default=8765, help="engine port on this machine (default 8765)")
    ap.add_argument("--port", type=int, default=8800, help="web server port (default 8800)")
    ap.add_argument("--dry-run", action="store_true", help="log key presses instead of sending them")
    ap.add_argument("--quiet", action="store_true", help="no notification and no sound for alerts")
    ap.add_argument("--headless", "--no-menubar", dest="headless", action="store_true", help="run without the menu bar item")
    ap.add_argument("--no-browser", action="store_true", help="do not open the dashboard")
    ap.add_argument("--lan", action="store_true", help="also listen on the local network (view and acknowledge only)")
    ap.add_argument("--settings", type=Path, default=None, help="settings file to use")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every care event")
    ap.add_argument("--version", action="version", version=f"Sonar Care {__version__}")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    engine_url = args.attach or f"ws://127.0.0.1:{args.engine_port}"
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    opts = Options(engine_url=engine_url, host=host, port=args.port, dry_run=args.dry_run, quiet=args.quiet,
                   mode="attach", settings_path=args.settings, verbose=args.verbose)

    service = Service(opts)
    if args.verbose:
        def show(message: dict) -> None:
            if message["type"] != "care_state":
                print(f"[care] {message}", flush=True)
        service.on_message(show)
    try:
        service.start()
        server = make_server(service, host, args.port, lan=args.lan)
    except (RuntimeError, OSError) as err:
        print(f"[care] could not start: {err}", file=sys.stderr)
        service.stop()
        return 2
    service.http_port = server.server_address[1]
    serve_in_thread(server)

    base = f"http://127.0.0.1:{service.http_port}/"
    print(f"Sonar Care {__version__}")
    print(f"  dashboard   {base}care.html")
    print(f"  engine      {engine_url}  (attached; not started by this app)")
    print(f"  settings    {args.settings or S.settings_path()}")
    print(f"  event log   {S.log_dir() / 'events.jsonl'}")
    if args.dry_run:
        print("  DRY RUN     gestures are logged, no key is pressed")
    print("  Ctrl+C to stop")
    if not args.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(base + "care.html",)).start()

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        while not stop.is_set():
            time.sleep(0.2)
    finally:
        server.shutdown()
        server.server_close()
        service.stop()
        print("[care] stopped")
    return 0

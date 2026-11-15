"""main.py: the Sonar Care command line.

What it does
    Starts everything: the sonar engine (and keeps it alive), the Sonar Care service, the
    local web server and the menu bar item, then opens the dashboard.

How to run (scripts/sonar_care.sh does this for you, from anywhere)
    scripts/sonar_care.sh                      live: this MacBook's speakers and microphone
    scripts/sonar_care.sh --backend sim        no sound: the engine's simulated visitor
    scripts/sonar_care.sh --mock               no engine at all: scripted fake events (rehearse the screens)
    scripts/sonar_care.sh --attach ws://127.0.0.1:8766     use an engine that is already running; start none

    --engine-port N     port for the engine (default 8765). If one is already listening there, it is used as is.
    --port N            web server port (default 8800; the next free one is used if busy)
    --dry-run           log the key each gesture WOULD press, press nothing
    --quiet             no notification, no sound, no voice for alerts
    --headless          no menu bar item (same as --no-menubar)
    --no-browser        do not open the dashboard
    --browser           open the dashboard in the default browser instead of a Chrome app window
    --lan               let other devices on the network view the dashboard and acknowledge alerts
    --scene NAME        simulator scene with --backend sim (demo, hall, ...), or mock scenario with --mock
    --engine-args "..." extra arguments passed to the engine
    --settings PATH     use this settings file instead of the default one
    -v                  print every care event

What it changes
    Starts the engine process; presses keys when a gesture is seen (not with --dry-run); shows
    notifications and plays sounds for alerts (not with --quiet); writes its settings file and
    logs. Stop with Ctrl+C or Quit in the menu: the engine it started is stopped too.
"""

from __future__ import annotations

import argparse
import shlex
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__, menubar
from . import settings as S
from .api import make_server, serve_in_thread
from .service import Options, Service
from .supervisor import EngineSupervisor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import find_browser  # noqa: E402  (Chrome / Edge lookup shared with the slides server)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sonar_care", description="Sonar Care: touch-free control, room monitoring and a help gesture from the laptop's sonar.")
    ap.add_argument("--backend", choices=("live", "sim"), default="live", help="engine audio: live (speakers + mic) or sim (simulated visitor, no sound)")
    ap.add_argument("--mock", action="store_true", help="scripted fake engine instead of the real one (to rehearse the screens)")
    ap.add_argument("--attach", metavar="URL", help="use the engine at this WebSocket address; start none")
    ap.add_argument("--engine-port", type=int, default=8765, help="engine port (default 8765)")
    ap.add_argument("--scene", default=None, help="simulator scene (--backend sim) or mock scenario (--mock)")
    ap.add_argument("--engine-args", default="", help="extra arguments for the engine, as one quoted string")
    ap.add_argument("--port", type=int, default=8800, help="web server port (default 8800)")
    ap.add_argument("--dry-run", action="store_true", help="log key presses instead of sending them")
    ap.add_argument("--quiet", action="store_true", help="no notification, sound or voice for alerts")
    ap.add_argument("--headless", "--no-menubar", dest="headless", action="store_true", help="run without the menu bar item")
    ap.add_argument("--no-browser", action="store_true", help="do not open the dashboard")
    ap.add_argument("--browser", action="store_true", help="open the dashboard in the default browser, not a Chrome app window")
    ap.add_argument("--lan", action="store_true", help="also listen on the local network (view and acknowledge only)")
    ap.add_argument("--settings", type=Path, default=None, help="settings file to use")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every care event")
    ap.add_argument("--version", action="version", version=f"Sonar Care {__version__}")
    return ap


def open_page(url: str, app_window: bool = True) -> None:
    """Open a page: a Chrome/Edge app window (no tabs, no address bar) if there is one, else the default browser."""
    if app_window:
        exe = find_browser()
        if exe:
            try:
                subprocess.Popen([exe, f"--app={url}", "--window-size=1440,900"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except OSError:
                pass                                       # fall through to the default browser
    if not webbrowser.open(url):
        print(f"[care] could not open a browser; open {url} yourself.", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    if args.attach:
        mode, engine_url = "attach", args.attach
        if urlsplit(engine_url).scheme not in ("ws", "wss"):
            print(f"[care] --attach needs a ws:// address, got {engine_url!r}", file=sys.stderr)
            return 2
    else:
        mode = "mock" if args.mock else args.backend
        engine_url = f"ws://127.0.0.1:{args.engine_port}"
    host = "0.0.0.0" if args.lan else "127.0.0.1"
    opts = Options(engine_url=engine_url, host=host, port=args.port, dry_run=args.dry_run, quiet=args.quiet,
                   mode=mode, settings_path=args.settings, verbose=args.verbose)

    supervisor = None
    if mode != "attach":
        supervisor = EngineSupervisor(mode, args.engine_port, S.log_dir() / "engine.log", scene=args.scene,
                                      extra=shlex.split(args.engine_args))
    service = Service(opts, supervisor=supervisor)
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
    info = service.engine_info
    how = {"live": "this laptop's speakers and microphone", "sim": "simulated visitor (no sound)", "mock": "scripted fake events",
           "attach": "already running; not started by this app"}[mode]
    print(f"Sonar Care {__version__}")
    print(f"  dashboard   {base}care.html")
    print(f"  slides      {base}presenter.html?ws={engine_url}")
    print(f"  engine      {engine_url}  ({how})" + (f"  state: {info.get('state')}" if supervisor else ""))
    if supervisor is not None:
        print(f"  engine log  {supervisor.log_path}")
    print(f"  settings    {args.settings or S.settings_path()}")
    print(f"  event log   {S.log_dir() / 'events.jsonl'}")
    keys = service.actions.status()
    if args.dry_run:
        print("  DRY RUN     gestures are logged, no key is pressed")
    elif keys["mode"] == "blocked":
        print(f"  KEYS BLOCKED  allow the app you started this from in {keys['settings_path']}")
    print("  Ctrl+C (or Quit in the menu) to stop")

    def show_page(fragment: str = "") -> None:
        open_page(base + "care.html" + fragment, app_window=not args.browser)

    if not args.no_browser:
        first = "" if service.get_settings()["onboarded"] else "#/setup"
        threading.Timer(0.6, show_page, args=(first,)).start()

    stopping = threading.Event()

    def shut_down() -> None:
        if stopping.is_set():
            return
        stopping.set()
        server.shutdown()
        server.server_close()
        service.stop()
        print("[care] stopped", flush=True)

    def on_signal(*_: object) -> None:
        stopping_now = not stopping.is_set()
        shut_down()
        if stopping_now and not args.headless and menubar.available():
            try:
                import rumps
                rumps.quit_application()
            except Exception:                              # noqa: BLE001  (already quitting)
                pass

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, on_signal)            # the terminal window was closed

    shown = False
    if not args.headless:
        presets = [(k, v["label"]) for k, v in S.PRESETS.items()]
        shown = menubar.run(service, show_page, presets, shut_down)     # returns when Quit is chosen, or False if unavailable
        if not shown and not stopping.is_set():
            print("[care] no menu bar item (rumps is not installed or there is no display); running without it", flush=True)
    if not shown:
        try:
            while not stopping.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
    shut_down()
    return 0

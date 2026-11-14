"""Static web server for the Laptop Sonar slides and Care Monitor.

What it does
    Serves the zero-build front end in app/web on http://127.0.0.1:8000 and opens
    it in the default browser. Uses only the Python standard library, so it runs
    even if nothing else installed at the venue.

Why
    The pages talk to the sonar engine over a WebSocket (ws://127.0.0.1:8765 by
    default). Serving them over http (not file://) keeps fonts and the clock sync
    reliable, and lets a phone on the same network open the monitor.

How to run (Windows: use `py -3.12` instead of `python`)
    python app/serve.py                     presenter, expects the real engine on port 8765
    python app/serve.py --mock              also start the mock engine (needs: pip install websockets)
    python app/serve.py --mock --scenario steady
    python app/serve.py --view monitor      open the Care Monitor instead of the slides
    python app/serve.py --app               open Chrome/Edge as a clean app window (no tabs or address bar)
    python app/serve.py --ws ws://127.0.0.1:9000   engine on another port

What it changes
    Nothing on disk. Listens on 127.0.0.1 only, unless --host 0.0.0.0 is given.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import importlib.util
import os
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
WEB_DIR = APP_DIR / "web"

# Windows can map .js to text/plain through the registry; force correct types.
MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static files with no caching (edits show up on reload) and quiet logs."""

    protocol_version = "HTTP/1.1"    # keep-alive: one connection serves many files
    verbose = False

    def guess_type(self, path: str | os.PathLike[str]) -> str:
        ext = os.path.splitext(str(path))[1].lower()
        return MIME_TYPES.get(ext) or super().guess_type(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 (stdlib signature)
        status = str(args[1]) if len(args) > 1 else ""
        if self.verbose or not status.startswith(("2", "3")):
            sys.stderr.write(f"[serve] {self.address_string()} {format % args}\n")


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = False      # fail fast on a busy port so we can try the next one
    # The stdlib default backlog is 5. A page load fires a burst of parallel requests, and
    # macOS and Windows reset connections that overflow the queue, which breaks the page.
    request_queue_size = 128


def bind(host: str, port: int, tries: int = 10) -> Server:
    """Bind to `port`, or the next free one if it is taken."""
    handler = functools.partial(Handler, directory=str(WEB_DIR))
    last_err: OSError | None = None
    for p in range(port, port + tries):
        try:
            return Server((host, p), handler)
        except OSError as err:
            last_err = err
    raise SystemExit(f"[serve] no free port in {port}-{port + tries - 1}: {last_err}")


def find_browser() -> str | None:
    """Chrome or Edge executable for --app / --kiosk windows."""
    candidates: list[str] = []
    if sys.platform == "win32":
        pf = [os.environ.get(k, "") for k in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
        for base in filter(None, pf):
            candidates += [os.path.join(base, r"Google\Chrome\Application\chrome.exe"),
                           os.path.join(base, r"Microsoft\Edge\Application\msedge.exe")]
    elif sys.platform == "darwin":
        candidates += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                       "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
    for name in ("chrome", "google-chrome", "chromium", "msedge", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    return next((c for c in candidates if os.path.exists(c)), None)


def open_view(url: str, mode: str) -> None:
    """Open the page: default browser, or a Chrome/Edge app or kiosk window."""
    if mode in ("app", "kiosk"):
        exe = find_browser()
        if exe:
            flag = [f"--app={url}"] if mode == "app" else ["--kiosk", url]
            subprocess.Popen([exe, *flag, "--start-maximized"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        print("[serve] Chrome/Edge not found; opening the default browser instead.")
    if not webbrowser.open(url):
        print(f"[serve] could not open a browser; open {url} yourself.")


def start_mock(args: argparse.Namespace) -> subprocess.Popen | None:
    if importlib.util.find_spec("websockets") is None:
        print("[serve] --mock needs the websockets package: pip install -r app/requirements.txt")
        return None
    cmd = [sys.executable, str(APP_DIR / "mock_engine.py"), "--scenario", args.scenario, "--port", str(args.mock_port)]
    print(f"[serve] starting mock engine: scenario={args.scenario} port={args.mock_port}")
    return subprocess.Popen(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the Laptop Sonar web app.")
    ap.add_argument("--host", default="127.0.0.1", help="interface (0.0.0.0 lets phones on the LAN open the monitor)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--view", choices=("presenter", "monitor", "index"), default="presenter")
    ap.add_argument("--ws", default=None, help="engine WebSocket URL (default ws://<page host>:8765)")
    ap.add_argument("--theme", choices=("light", "dark"), default=None)
    ap.add_argument("--app", action="store_true", help="open as a Chrome/Edge app window")
    ap.add_argument("--kiosk", action="store_true", help="open full-screen kiosk (exit with Alt+F4)")
    ap.add_argument("--no-browser", action="store_true", help="just serve; do not open anything")
    ap.add_argument("--mock", action="store_true", help="also run the mock engine")
    ap.add_argument("--scenario", choices=("demo", "steady", "quiet"), default="demo", help="mock scenario (with --mock)")
    ap.add_argument("--mock-port", type=int, default=8765)
    ap.add_argument("--verbose", action="store_true", help="log every request")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if not (WEB_DIR / "presenter.html").exists():
        raise SystemExit(f"[serve] web files not found in {WEB_DIR}")
    Handler.verbose = args.verbose

    # Start the HTTP server first so the browser never opens onto a dead port.
    server = bind(args.host, args.port)
    port = server.server_address[1]
    page_host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    query = {}
    if args.ws:
        query["ws"] = args.ws
    elif args.mock and args.mock_port != 8765:
        query["ws"] = f"ws://{page_host}:{args.mock_port}"
    if args.theme:
        query["theme"] = args.theme
    qs = ("?" + urllib.parse.urlencode(query)) if query else ""
    base = f"http://{page_host}:{port}/"
    url = base + ("" if args.view == "index" else f"{args.view}.html") + qs

    mock = start_mock(args) if args.mock else None

    print(f"[serve] Laptop Sonar web app on {base}")
    print(f"        presenter: {base}presenter.html{qs}")
    print(f"        monitor:   {base}monitor.html{qs}")
    print("        keys on the slides: arrows = navigate, O = live panel, F = fullscreen, ? = help")
    print("        Ctrl+C to stop")
    if not args.no_browser:
        mode = "kiosk" if args.kiosk else "app" if args.app else "browser"
        threading.Timer(0.8 if mock else 0.2, open_view, args=(url, mode)).start()

    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if mock and mock.poll() is None:
            mock.terminate()
            try:
                mock.wait(timeout=3)
            except subprocess.TimeoutExpired:
                mock.kill()
        print("[serve] stopped")
        time.sleep(0.1)


if __name__ == "__main__":
    main()

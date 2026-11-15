"""api.py: the local web server for Sonar Care: the pages plus a small JSON API.

What it does
    Serves the files in app/web (the dashboard, the slides, the launcher) and answers:

      GET  /api/boot.js          tiny script the pages load first: where the relay WebSocket is
      GET  /api/state            the current state (engine, presence, breathing, alerts, ...)
      GET  /api/settings         the settings + everything the settings page needs to draw itself
      POST /api/settings         change settings; body = a partial JSON object
      POST /api/ack              acknowledge alerts; body = {"id": "help-3"} or {} for all
      POST /api/help/cancel      cancel a help request that is still counting down
      POST /api/help/send        send a help request that is still counting down, at once
      POST /api/calibrate        ask the engine to recalibrate (about 5 s, keep still)
      POST /api/practice         body = {"active": true|false}: tutorial mode (nothing is pressed or raised)
      POST /api/engine           body = {"action": "start"|"stop"|"restart"}: the engine this app manages
      POST /api/accessibility    open System Settings on the Accessibility page (and ask macOS to list this app)
      POST /api/test-action      body = {"action": "right", "delay_s": 3}: press one key after a delay, to try a mapping
      POST /api/test-alert       announce a harmless test alert (banner, chime, voice), to check it is noticed
      GET  /api/timeline         recent timeline entries (?limit=100)

    Every reply is JSON. Errors come back as {"error": "..."} with a 4xx status.

Why plain http.server
    No extra install, and the same static-file code the slides already use (app/serve.py).

Safety
    It listens on 127.0.0.1 only (unless started with --lan). A POST must be JSON and must come
    from one of our own pages: another website open in the browser cannot change settings or
    make this app press keys. With --lan, other devices may view and acknowledge, nothing else.

What it changes
    Nothing by itself; POST requests change settings or acknowledge alerts through service.py.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import WEB_DIR, Handler as StaticHandler, Server  # noqa: E402  (static files, MIME types, no caching)

from . import __version__  # noqa: E402
from . import settings as S  # noqa: E402

MAX_BODY = 64 * 1024
LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
LOCAL_NAMES = {"127.0.0.1", "localhost", "[::1]", "::1"}
LAN_POSTS = {"/api/ack", "/api/help/cancel", "/api/help/send"}     # what another device on the network may do with --lan


class ApiHandler(StaticHandler):
    """Static files from app/web, plus /api/*. `service` and `lan` are set on the class by make_server()."""

    service: Any = None
    lan = False
    server_version = f"SonarCare/{__version__}"

    # ---- helpers --------------------------------------------------------------------------------

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def _host_name(self) -> str:
        """The host part of the Host header, without the port."""
        host = self.headers.get("Host", "")
        if host.startswith("["):                           # [::1]:8800
            return host.split("]")[0] + "]"
        return host.rsplit(":", 1)[0] if ":" in host else host

    def _from_this_machine(self) -> bool:
        return self.client_address[0] in LOOPBACK

    def _request_allowed(self, post: bool, path: str) -> str | None:
        """None if the request may go ahead, else why not."""
        name = self._host_name()
        if not self.lan and name not in LOCAL_NAMES:
            return "this address is not served"            # stops DNS-rebinding tricks against the local API
        if not post:
            return None
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            return "send JSON (Content-Type: application/json)"
        origin = self.headers.get("Origin")
        if origin is not None:
            try:
                origin_host = urlsplit(origin).netloc
            except ValueError:
                return "bad Origin"
            if origin_host != self.headers.get("Host"):
                return "requests from other websites are not accepted"
        if not self._from_this_machine() and path not in LAN_POSTS:
            return "only this computer can change that"
        return None

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError("request too large")
        raw = self.rfile.read(length) if length else b""
        data = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        if not isinstance(data, dict):
            raise ValueError("the body must be a JSON object")
        return data

    # ---- GET ------------------------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (stdlib name)
        url = urlsplit(self.path)
        if not url.path.startswith("/api/"):
            super().do_GET()
            return
        why = self._request_allowed(False, url.path)
        if why:
            self._fail(403, why)
            return
        svc = self.service
        try:
            if url.path == "/api/boot.js":
                self._boot_js()
            elif url.path == "/api/state":
                self._send_json(svc.snapshot())
            elif url.path == "/api/settings":
                self._send_json({"settings": svc.get_settings(), "meta": S.describe(), "path": str(svc.opts.settings_path or S.settings_path())})
            elif url.path == "/api/timeline":
                limit = int((parse_qs(url.query).get("limit") or ["100"])[0])
                self._send_json({"timeline": svc.timeline(limit)})
            else:
                self._fail(404, "no such endpoint")
        except (ValueError, TimeoutError) as err:
            self._fail(400, str(err) or type(err).__name__)

    def _boot_js(self) -> None:
        """Where the page finds the relay, as a script it can load before anything else."""
        svc = self.service
        host = self._host_name() or "127.0.0.1"
        boot = {
            "version": __version__, "relay": f"ws://{host}:{svc.relay_port}", "engine": svc.opts.engine_url,
            "dryRun": svc.actions.dry_run, "mode": svc.opts.mode, "local": self._from_this_machine(),
        }
        body = ("window.SonarCareBoot = " + json.dumps(boot, separators=(",", ":")) + ";\n").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/javascript; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- POST -----------------------------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802 (stdlib name)
        url = urlsplit(self.path)
        if not url.path.startswith("/api/"):
            self._fail(404, "no such endpoint")
            return
        why = self._request_allowed(True, url.path)
        if why:
            self._fail(403, why)
            return
        svc = self.service
        try:
            data = self._body()
            if url.path == "/api/settings":
                self._send_json({"settings": svc.update_settings(data)})
            elif url.path == "/api/ack":
                alert_id = data.get("id")
                self._send_json({"acknowledged": svc.acknowledge(str(alert_id) if alert_id is not None else None)})
            elif url.path == "/api/help/cancel":
                self._send_json({"cancelled": svc.cancel_help()})
            elif url.path == "/api/help/send":
                self._send_json({"sent": svc.send_help_now()})
            elif url.path == "/api/calibrate":
                self._send_json({"sent": svc.calibrate()})
            elif url.path == "/api/practice":
                self._send_json({"practice": svc.set_practice(bool(data.get("active")))})
            elif url.path == "/api/engine":
                self._send_json({"engine": svc.engine_command(str(data.get("action")))})
            elif url.path == "/api/accessibility":
                self._send_json({"opened": svc.open_accessibility(), "keys": svc.actions.status()})
            elif url.path == "/api/test-alert":
                self._send_json(svc.test_alert())
            elif url.path == "/api/test-action":
                delay = data.get("delay_s", 0)
                if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not 0 <= delay <= 10:
                    raise ValueError("delay_s must be a number from 0 to 10")
                self._send_json(svc.test_action(str(data.get("action")), float(delay)))
            else:
                self._fail(404, "no such endpoint")
        except S.SettingsError as err:
            self._fail(400, str(err))
        except (ValueError, UnicodeDecodeError) as err:
            self._fail(400, f"bad request: {err}")
        except TimeoutError:
            self._fail(503, "the service is busy; try again")

    def do_OPTIONS(self) -> None:  # noqa: N802 (stdlib name)
        self._fail(403, "cross-origin requests are not accepted")   # no CORS: only our own pages may call the API


def make_server(service: Any, host: str, port: int, lan: bool = False, tries: int = 20) -> Server:
    """Bind the web server on `port`, or the next free one. Call .serve_forever() on the result."""
    handler = type("BoundApiHandler", (ApiHandler,), {"service": service, "lan": lan})

    def factory(*args: Any, **kwargs: Any) -> ApiHandler:
        return handler(*args, directory=str(WEB_DIR), **kwargs)

    last: OSError | None = None
    for p in range(port, port + tries):
        if p == service.relay_port:
            continue
        try:
            return Server((host, p), factory)
        except OSError as err:
            last = err
    raise OSError(f"no free port for the web server in {port}-{port + tries - 1}: {last}")


def serve_in_thread(server: Server) -> threading.Thread:
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.25), name="sonar-care-http", daemon=True)
    thread.start()
    return thread

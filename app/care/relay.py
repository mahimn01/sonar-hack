"""relay.py: the WebSocket the dashboard connects to.

What it does
    Passes every engine message on to the browser unchanged, and adds Sonar Care's own
    messages on the same stream (care_state, care_alert, care_action, care_timeline,
    care_settings). A page therefore needs one connection and sees engine data and
    care decisions in the order they happened. A newly connected page first receives a
    snapshot (who the engine is, the current state, open alerts, the recent timeline).

    Pages may send the engine's commands ({"cmd": "calibrate"}, ...); they are forwarded.

Why not let the page talk to the engine directly
    The alerts are decided in this service, not in the browser. One stream keeps the page
    simple and keeps the slides (which do talk to the engine directly) independent of us.

What it changes
    Opens one listening TCP port on 127.0.0.1 (or on all interfaces with --lan). Browser
    pages from other websites are refused (Origin check), so a web page you happen to
    have open cannot read the sensor.
"""

from __future__ import annotations

import http
import json
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from websockets.asyncio.server import ServerConnection, broadcast, serve
from websockets.exceptions import ConnectionClosed

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


class Relay:
    def __init__(self, host: str, port: int, snapshot: Callable[[], list[str]],
                 on_command: Callable[[dict[str, Any]], Awaitable[None]], extra_hosts: set[str] | None = None) -> None:
        self.host = host
        self.port = port
        self._snapshot = snapshot            # messages a new page gets first
        self._on_command = on_command
        self._allowed_hosts = LOCAL_HOSTS | (extra_hosts or set())
        self._clients: set[ServerConnection] = set()
        self._server: Any = None

    @property
    def n_clients(self) -> int:
        return len(self._clients)

    def _origin_ok(self, origin: str | None) -> bool:
        """Scripts, tests and curl send no Origin. A browser page must come from this machine."""
        if origin is None:
            return True
        try:
            host = urlsplit(origin).hostname or ""
        except ValueError:
            return False
        return host in self._allowed_hosts or f"[{host}]" in self._allowed_hosts

    def _check_origin(self, connection: ServerConnection, request: Any) -> Any:
        if not self._origin_ok(request.headers.get("Origin")):
            return connection.respond(http.HTTPStatus.FORBIDDEN, "Sonar Care only talks to pages from this computer.\n")
        return None

    async def _handler(self, ws: ServerConnection) -> None:
        try:
            for text in self._snapshot():
                await ws.send(text)
        except ConnectionClosed:
            return
        self._clients.add(ws)
        try:
            async for message in ws:
                try:
                    cmd = json.loads(message)
                except (TypeError, ValueError):
                    continue
                if isinstance(cmd, dict):
                    await self._on_command(cmd)
        except ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)

    async def start(self) -> None:
        self._server = await serve(self._handler, self.host, self.port, process_request=self._check_origin,
                                   max_size=64 * 1024, ping_interval=20, ping_timeout=20)

    def broadcast(self, text: str) -> None:
        """Send one message to every connected page. Never waits for a slow one."""
        if self._clients:
            broadcast(self._clients, text)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

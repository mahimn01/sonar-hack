"""server.py — pushes the engine's events to the UI over a local WebSocket.

Plain words: a tiny broadcast server. Every connected client (the dashboard,
the slides page, a key-press bridge) receives every event as one line of JSON
text. A client may also send a command such as {"cmd": "calibrate"}.

It runs in its own thread with its own asyncio loop, so the audio processing
thread just calls `broadcast(event)` and never waits for a slow browser.

What it changes on the machine: opens a listening TCP port on 127.0.0.1 (local
machine only by default). Nothing is written to disk.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Callable

import numpy as np


def to_json(event: dict[str, Any]) -> str:
    """Compact JSON; numpy numbers and arrays are converted to plain Python."""

    def default(o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"cannot serialise {type(o).__name__}")

    return json.dumps(event, default=default, separators=(",", ":"))


class EventServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        on_command: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
        hello: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self._on_command = on_command
        self._hello = hello
        self._clients: set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self.error: BaseException | None = None

    @property
    def n_clients(self) -> int:
        return len(self._clients)

    # -- asyncio side ----------------------------------------------------------------
    async def _handler(self, ws: Any) -> None:
        self._clients.add(ws)
        try:
            if self._hello is not None:
                await ws.send(to_json(self._hello()))
            async for message in ws:
                try:
                    cmd = json.loads(message)
                except (ValueError, TypeError):
                    await ws.send(to_json({"type": "error", "message": "commands must be JSON objects"}))
                    continue
                if not isinstance(cmd, dict):
                    await ws.send(to_json({"type": "error", "message": "commands must be JSON objects"}))
                    continue
                if cmd.get("cmd") == "ping":
                    await ws.send(to_json({"type": "pong"}))
                    continue
                reply = self._on_command(cmd) if self._on_command is not None else None
                if reply is not None:
                    await ws.send(to_json(reply))
        except Exception:
            pass                                   # a client that disconnects mid-send is not our problem
        finally:
            self._clients.discard(ws)

    async def _main(self) -> None:
        from websockets.asyncio.server import serve

        self._stop = asyncio.Event()
        try:
            async with serve(self._handler, self.host, self.port):
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:               # e.g. port already in use
            self.error = exc
            self._ready.set()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())
        self._loop.close()

    # -- thread-safe API ----------------------------------------------------------------
    def start(self, timeout: float = 5.0) -> None:
        self._thread = threading.Thread(target=self._run, name="sonar-ws", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("WebSocket server did not start")
        if self.error is not None:
            raise RuntimeError(f"WebSocket server could not listen on {self.host}:{self.port}: {self.error}")

    def broadcast(self, event: dict[str, Any]) -> None:
        """Send one event to every connected client (safe to call from any thread)."""
        if self._loop is None or not self._clients:
            return
        text = to_json(event)

        def send() -> None:
            from websockets.asyncio.server import broadcast

            broadcast(list(self._clients), text)   # skips clients that are closing; never blocks

        self._loop.call_soon_threadsafe(send)

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=3.0)

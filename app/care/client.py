"""client.py: the one WebSocket connection from Sonar Care to the sonar engine.

What it does
    Connects to the engine's event stream (ws://127.0.0.1:8765 by default), hands every message
    to the service as text, and reconnects by itself whenever the engine goes away: after
    0.25 s, then 0.5 s, 1 s, capped at 2 s, forever. It can also send the engine a command
    such as {"cmd": "calibrate"}.

Why
    The engine may be started after the app, restarted by the supervisor, or crash. The app
    must heal without anyone touching it.

What it changes
    Nothing on disk. It only talks to the engine on the local machine.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Callable

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI


class EngineClient:
    """One reconnecting WebSocket. All methods must be called on the service's event loop."""

    def __init__(self, url: str, on_message: Callable[[str], None], on_state: Callable[[str], None]) -> None:
        self.url = url
        self._on_message = on_message
        self._on_state = on_state            # "connecting" | "open" | "closed"
        self._ws: Any = None
        self._stopped = False
        self.state = "connecting"
        self.attempts = 0                    # failed attempts since the last good connection

    def _set(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self._on_state(state)

    async def run(self) -> None:
        """Connect, read until the connection ends, wait, repeat. Returns only after stop()."""
        kwargs: dict[str, Any] = {"open_timeout": 3, "max_size": 4 * 1024 * 1024}
        if "proxy" in inspect.signature(connect).parameters:
            kwargs["proxy"] = None           # never route localhost through a system proxy
        while not self._stopped:
            self._set("connecting")
            try:
                async with connect(self.url, **kwargs) as ws:
                    self._ws = ws
                    self.attempts = 0
                    self._set("open")
                    async for message in ws:
                        if isinstance(message, str):          # binary frames are not part of the schema
                            self._on_message(message)
            except InvalidURI as err:
                print(f"[care] bad engine address {self.url!r}: {err}", flush=True)
            except (OSError, asyncio.TimeoutError, ConnectionClosed, InvalidHandshake):
                pass                                          # engine not there (yet): retry below
            finally:
                self._ws = None
            if self._stopped:
                break
            self._set("closed")
            delay = min(2.0, 0.25 * 2 ** min(self.attempts, 4))
            self.attempts += 1
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break

    async def send(self, command: dict[str, Any]) -> bool:
        """Send one command to the engine. False if it is not connected right now."""
        ws = self._ws
        if ws is None:
            return False
        try:
            await ws.send(json.dumps(command, separators=(",", ":")))
            return True
        except (ConnectionClosed, OSError):
            return False

    async def stop(self) -> None:
        self._stopped = True
        ws = self._ws
        if ws is not None:
            await ws.close()

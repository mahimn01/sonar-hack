"""service.py: wires Sonar Care together and keeps it running.

What it does
    One background thread runs everything that reacts to the engine:
      engine events -> rules.py decides -> this file carries the decisions out
        * a gesture that should act  -> actions.py presses the key (or logs it in a dry run)
        * an alert raised            -> notify.py shows a notification and plays a sound
        * anything worth remembering -> one line in the timeline (memory + a log file)
    and it tells every open dashboard about all of it through relay.py.

    Other threads (the web server in api.py, the menu bar) use the small thread-safe
    methods at the bottom: snapshot(), update_settings(), acknowledge(), calibrate(), ...

Why one loop
    The rules and the settings are only ever touched from this loop, so no locks are needed
    and an alert can never be decided twice.

What it changes
    * presses keys (through actions.py) unless dry_run is set;
    * shows notifications / plays sounds (through notify.py) unless quiet is set;
    * writes settings.json when a setting changes, and appends to events.jsonl in the log folder;
    * opens two local ports: the web server and the relay WebSocket;
    * starts, restarts and stops the engine process through supervisor.py (when it was given one).
"""

from __future__ import annotations

import asyncio
import collections
import json
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import adapt  # noqa: E402  (the engine's message format lives there, nowhere else)

from . import __version__  # noqa: E402
from . import settings as S  # noqa: E402
from .actions import Actions  # noqa: E402
from .client import EngineClient  # noqa: E402
from .notify import Notifier  # noqa: E402
from .relay import Relay  # noqa: E402
from .rules import CareRules, fmt_duration  # noqa: E402
from .supervisor import EngineSupervisor  # noqa: E402

GESTURE_TEXT = {"swipe_left": "Swipe left", "swipe_right": "Swipe right", "push": "Push"}

# Signal quality from the engine's SNR (INTERFACES.md: healthy 40+, gestures unreliable below ~25).
SNR_GOOD_DB = 35.0
SNR_WEAK_DB = 25.0

TICK_S = 0.25                 # how often time-based rules are checked
STATE_EVERY_S = 1.0           # how often the dashboard gets a full state message when nothing changes
STALL_S = 3.0                 # connected but silent this long = "no data"
TIMELINE_KEEP = 300


@dataclass
class Options:
    """How this run was started (the command line fills this in)."""

    engine_url: str = "ws://127.0.0.1:8765"
    host: str = "127.0.0.1"           # web server and relay listen here ("0.0.0.0" with --lan)
    port: int = 8800                  # web server; the relay takes the next free port after it
    dry_run: bool = False             # log key presses instead of sending them
    quiet: bool = False               # no notification, no sound
    mode: str = "attach"              # live | sim | mock | attach: how the engine is provided
    settings_path: Path | None = None
    verbose: bool = False
    clock: Callable[[], float] = field(default=time.time, repr=False)


def free_port(host: str, start: int, tries: int = 20) -> int:
    """First port from `start` upward that nothing is listening on."""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host if host != "0.0.0.0" else "", port))
            except OSError:
                continue
            return port
    raise OSError(f"no free port in {start}-{start + tries - 1}")


def signal_quality(snr_db: float | None) -> str:
    if snr_db is None:
        return "unknown"
    return "good" if snr_db >= SNR_GOOD_DB else "fair" if snr_db >= SNR_WEAK_DB else "weak"


class Service:
    """The running app. start() it, read snapshot(), stop() it."""

    def __init__(self, opts: Options, actions: Actions | None = None, notifier: Notifier | None = None,
                 supervisor: EngineSupervisor | None = None) -> None:
        self.opts = opts
        self.settings = S.load(opts.settings_path)
        self.actions = actions or Actions(dry_run=opts.dry_run)
        self.notifier = notifier or Notifier(enabled=not opts.quiet)
        self.supervisor = supervisor                     # None = the engine is somebody else's (--attach)
        self.relay_port = 0
        self.http_port = opts.port
        self.engine_info: dict[str, Any] = {"managed": False, "state": "external", "url": opts.engine_url}
        self._engine_state = "external"

        self._now = opts.clock
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._error: BaseException | None = None

        self._rules: CareRules | None = None
        self._client: EngineClient | None = None
        self._relay: Relay | None = None
        self._status: dict[str, Any] = {}
        self._hello_text: str | None = None
        self._last_msg_at = 0.0
        self._opened_at = 0.0
        self._health = "connecting"
        self._last_action: dict[str, Any] | None = None
        self._timeline: collections.deque[dict[str, Any]] = collections.deque(maxlen=TIMELINE_KEEP)
        self._timeline_seq = 0
        self._rev = 0
        self._dirty = True
        self._state: dict[str, Any] = {"type": "care_state", "t": self._now(), "version": __version__, "starting": True}
        self._log_path = S.log_dir() / "events.jsonl"
        self._log_failed = False
        self._listeners: list[Callable[[dict[str, Any]], None]] = []

    # ---- life cycle ------------------------------------------------------------------------------

    def start(self, timeout: float = 10.0) -> None:
        """Start the background loop; returns once the relay is listening."""
        self._thread = threading.Thread(target=self._run, name="sonar-care", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("Sonar Care did not start in time")
        if self._error is not None:
            raise RuntimeError(f"Sonar Care could not start: {self._error}")

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except BaseException as exc:                      # anything here means the app is not running: report it
            self._error = exc
            self._ready.set()
        finally:
            self._loop.close()

    async def _main(self) -> None:
        self._stop = asyncio.Event()
        now = self._now()
        self._rules = CareRules(self.settings, now)
        self.relay_port = free_port(self.opts.host, self.opts.port + 1)
        self._relay = Relay(self.opts.host, self.relay_port, self._hello_messages, self._on_page_command)
        await self._relay.start()
        self._client = EngineClient(self.opts.engine_url, self._on_engine_message, self._on_link_state)
        tasks = [asyncio.create_task(self._client.run()), asyncio.create_task(self._ticker())]
        self._add_timeline({"t": now, "kind": "system", "level": "info", "icon": "ping", "text": "Sonar Care started",
                            "detail": "dry run: no keys are pressed" if self.opts.dry_run else None})
        if self.supervisor is not None:
            self.supervisor.start()
            self._watch_engine(now)
        self._publish_state(now)
        self._ready.set()
        try:
            await self._stop.wait()
        finally:
            await self._client.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._relay.stop()
            if self.supervisor is not None:
                await asyncio.get_running_loop().run_in_executor(None, self.supervisor.shutdown)

    # ---- engine -> rules -------------------------------------------------------------------------

    def _on_link_state(self, state: str) -> None:
        if state == "open":
            self._opened_at = self._now()
        self._dirty = True

    def _on_engine_message(self, text: str) -> None:
        now = self._now()
        self._last_msg_at = now
        self._relay.broadcast(text)
        if text.startswith('{"type":"spectrogram"'):      # 10 a second, only the page draws them
            return
        for evt in adapt(text, now):
            kind = evt.get("kind")
            if kind == "spectrogram" or kind == "unknown":
                continue
            if kind == "hello":
                self._hello_text = text
                continue
            if kind == "status":
                self._status = evt
            self._apply(self._rules.on_event(evt, now))

    async def _ticker(self) -> None:
        last_state = 0.0
        while True:
            now = self._now()
            if self._client.state != "open":
                health = "connecting" if self._health == "connecting" and not self._last_msg_at else "offline"
            elif now - max(self._last_msg_at, self._opened_at) > STALL_S:
                health = "stalled"
            else:
                health = "live" if self._last_msg_at >= self._opened_at else "connecting"
            if health != self._health:
                self._health = health
                self._dirty = True
            if self.supervisor is not None and self.supervisor.poll():
                self._watch_engine(now)
            self._apply(self._rules.on_link(health == "live", now))
            self._apply(self._rules.tick(now))
            if self._dirty or now - last_state >= STATE_EVERY_S:
                self._publish_state(now)
                last_state = now
            await asyncio.sleep(TICK_S)

    def _watch_engine(self, now: float) -> None:
        """The engine process changed state: remember it and write a timeline line a person can read."""
        info = self.supervisor.info()
        self.engine_info = info
        self._dirty = True
        state = info["state"]
        if state == self._engine_state:
            return
        before, self._engine_state = self._engine_state, state
        line = {
            "starting": ("info", f"Starting the engine ({info['model']})", None),
            "running": ("ok", "Engine running", f"model {info['model']}" if info["model"] else None),
            "attached": ("info", "Using an engine that was already running", f"port {info['port']}"),
            "restarting": ("attention", "The engine stopped unexpectedly; restarting", info["last_error"]),
            "failed": ("alert", "The engine could not be started", info["last_error"]),
            "stopped": ("info", "Engine stopped", None) if before != "external" else None,
        }.get(state)
        if line:
            self._add_timeline({"t": now, "kind": "system", "level": line[0], "icon": "engine", "text": line[1], "detail": line[2]})

    # ---- rules -> the world ------------------------------------------------------------------------

    def _apply(self, effects: list[dict[str, Any]]) -> None:
        changed = False
        for e in effects:
            kind = e["effect"]
            if kind == "gesture":
                self._do_gesture(e)
                changed = True
            elif kind == "alert":
                self._do_alert(e)
                changed = True
            elif kind == "timeline":
                self._add_timeline(e["entry"])
        if changed:                                       # the state follows a decision at once, not at the next tick
            self._publish_state(self._now())

    def _do_gesture(self, e: dict[str, Any]) -> None:
        gesture = e["gesture"]
        msg: dict[str, Any] = {"type": "care_action", "t": e["t"], "gesture": gesture, "confidence": e.get("confidence"),
                               "result": e["result"], "reason": e.get("reason")}
        for key in ("action", "delayed_s", "help_until", "double_push", "gap_s"):
            if e.get(key) is not None:
                msg[key] = e[key]
        if e.get("action"):
            msg["action_label"] = S.action_label(e["action"])
        if e["result"] == "act":
            outcome = self.actions.perform(e["action"])
            msg["result"], msg["detail"] = outcome["result"], outcome["detail"]
            note = {"dry_run": "dry run: no key pressed", "blocked": outcome["detail"], "failed": outcome["detail"]}.get(outcome["result"])
            if e.get("delayed_s"):
                waited = f"after waiting {e['delayed_s']:.1f} s for a second push"
                note = f"{note}; {waited}" if note else waited
            self._add_timeline({"t": e["t"], "kind": "gesture", "level": "info" if outcome["result"] in ("sent", "dry_run") else "attention",
                                "icon": gesture, "text": f"{GESTURE_TEXT.get(gesture, gesture)}: {msg['action_label']}", "detail": note})
        self._last_action = msg
        self._broadcast(msg)
        self._dirty = True

    def _do_alert(self, e: dict[str, Any]) -> None:
        now = self._now()
        alert, change = e["alert"], e["change"]
        self._broadcast({"type": "care_alert", "t": now, "change": change, "alert": alert})
        if change in ("raised", "reminder"):
            cfg = self.settings["alerts"]
            seconds = self.notifier.alert(alert, reminder=change == "reminder", notification=cfg["notification"],
                                          sound=cfg["sound"], volume=cfg["volume"], speak=cfg["speak"], now=now)
            if seconds:
                self._rules.sound_played(now, seconds + 0.5)
        self._dirty = True

    def _add_timeline(self, entry: dict[str, Any]) -> None:
        self._timeline_seq += 1
        entry = {"id": self._timeline_seq, **{k: v for k, v in entry.items() if v is not None}}
        self._timeline.append(entry)
        self._broadcast({"type": "care_timeline", "t": entry["t"], "entry": entry})
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        except OSError as err:
            if not self._log_failed:                      # say it once; the app keeps working without the file
                self._log_failed = True
                print(f"[care] cannot write the event log {self._log_path}: {err}", file=sys.stderr, flush=True)

    def _broadcast(self, message: dict[str, Any]) -> None:
        if self._relay is not None:
            self._relay.broadcast(json.dumps(message, separators=(",", ":")))
        for listener in self._listeners:
            try:
                listener(message)
            except Exception as err:                      # a broken listener must not stop the service
                print(f"[care] listener failed: {err!r}", file=sys.stderr, flush=True)

    # ---- state for the dashboard and the menu bar -----------------------------------------------------

    def _build_state(self, now: float) -> dict[str, Any]:
        st, s = self._status, self.settings
        snr = st.get("snrDb")
        mapping = S.mapping(s)
        state: dict[str, Any] = {
            "type": "care_state", "t": now, "version": __version__,
            "dry_run": self.actions.dry_run, "quiet": not self.notifier.enabled, "mode": self.opts.mode,
            "engine": dict(self.engine_info),
            "link": {"state": self._client.state if self._client else "connecting", "health": self._health,
                     "age_s": None if not self._last_msg_at else round(now - self._last_msg_at, 1)},
            "sonar": {
                "snr_db": snr, "signal": signal_quality(snr), "calibrating": st.get("calibrating") is True,
                "locked": st.get("locked"), "source": st.get("source"), "noise_db": st.get("noiseDb"),
                "fps": st.get("fps"), "clipped": st.get("clipped"), "device": st.get("device"),
            },
            "control": {
                "enabled": s["control"]["enabled"], "preset": s["control"]["preset"],
                "preset_label": S.PRESETS[s["control"]["preset"]]["label"],
                "mapping": {g: {"action": a, "label": S.action_label(a)} for g, a in mapping.items()},
                "keys": self.actions.status(),
            },
            "monitoring": s["monitoring"],
            "room": s["room"], "onboarded": s["onboarded"], "settings_rev": self._rev,
            "pages": self._relay.n_clients if self._relay else 0,
        }
        state.update(self._rules.snapshot(now))
        if self._last_action is not None:
            state["last_gesture"] = {k: self._last_action.get(k) for k in ("gesture", "confidence", "result", "reason", "action", "action_label", "t")}
        return state

    def _publish_state(self, now: float) -> None:
        self._state = self._build_state(now)
        self._dirty = False
        self._broadcast(self._state)

    def _hello_messages(self) -> list[str]:
        """What a page gets the moment it connects."""
        now = self._now()
        out = []
        if self._hello_text:
            out.append(self._hello_text)                  # who the engine is (it only says so once per connection)
        out.append(json.dumps({"type": "care_hello", "t": now, "version": __version__, "dry_run": self.actions.dry_run,
                               "mode": self.opts.mode, "gestures": list(S.GESTURES),
                               "timeline": list(self._timeline)[-80:]}, separators=(",", ":")))
        out.append(json.dumps({"type": "care_settings", "t": now, "rev": self._rev, "settings": self.settings}, separators=(",", ":")))
        out.append(json.dumps(self._build_state(now), separators=(",", ":")))
        return out

    async def _on_page_command(self, cmd: dict[str, Any]) -> None:
        """A page sent an engine command (calibrate, ping, or a mock-engine test command): pass it on."""
        if self._client is not None:
            await self._client.send(cmd)

    # ---- thread-safe API (web server threads, menu bar) -------------------------------------------------

    def _call(self, coro: Coroutine[Any, Any, Any], timeout: float = 5.0) -> Any:
        if self._loop is None:
            raise RuntimeError("Sonar Care is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def snapshot(self) -> dict[str, Any]:
        """The latest state (at most a second old). Safe from any thread; never blocks."""
        return self._state

    def on_message(self, listener: Callable[[dict[str, Any]], None]) -> None:
        """Call `listener(message)` (on the service loop) for every care_* message. Keep it quick."""
        self._listeners.append(listener)

    def get_settings(self) -> dict[str, Any]:
        return self.settings

    def timeline(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._timeline)[-max(1, min(limit, TIMELINE_KEEP)):]

    def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Change settings (partial is fine). Raises settings.SettingsError with a readable reason."""
        async def do() -> dict[str, Any]:
            now = self._now()
            old = self.settings
            new = S.merge(old, patch)
            if new == old:
                return old
            S.save(new, self.opts.settings_path)
            self.settings = new
            self._rev += 1
            self._apply(self._rules.apply_settings(new, now))
            for text in _describe_change(old, new):
                self._add_timeline({"t": now, "kind": "system", "level": "info", "icon": "settings", "text": text})
            self._broadcast({"type": "care_settings", "t": now, "rev": self._rev, "settings": new})
            self._publish_state(now)
            return new
        return self._call(do())

    def acknowledge(self, alert_id: str | None = None) -> int:
        """Acknowledge one alert, or all that need it. Returns how many were acknowledged."""
        async def do() -> int:
            effects = self._rules.acknowledge(self._now(), alert_id)
            self._apply(effects)
            self._publish_state(self._now())
            return sum(1 for e in effects if e["effect"] == "alert")
        return self._call(do())

    def cancel_help(self) -> bool:
        """Cancel a help request that is still counting down. False if there was none."""
        async def do() -> bool:
            effects = self._rules.cancel_help(self._now())
            self._apply(effects)
            self._publish_state(self._now())
            return bool(effects)
        return self._call(do())

    def calibrate(self) -> bool:
        """Ask the engine to learn its noise floors again (about 5 s; keep still). False if it is not connected."""
        async def do() -> bool:
            return await self._client.send({"cmd": "calibrate"})
        return self._call(do())

    def set_practice(self, active: bool) -> bool:
        """Tutorial on/off: while on, gestures are recognised but press nothing and raise no alert."""
        async def do() -> bool:
            self._apply(self._rules.set_practice(active, self._now()))
            self._dirty = True
            return self._rules.practice
        return self._call(do())

    def engine_command(self, action: str) -> dict[str, Any]:
        """start | stop | restart the engine this app manages. Raises ValueError if there is none to manage."""
        if self.supervisor is None:
            raise ValueError("this engine was started elsewhere; Sonar Care does not manage it")
        if action not in ("start", "stop", "restart"):
            raise ValueError("action must be start, stop or restart")

        async def do() -> dict[str, Any]:
            getattr(self.supervisor, action)()
            self._watch_engine(self._now())
            self._publish_state(self._now())
            return self.supervisor.info()
        return self._call(do())

    def open_accessibility(self) -> bool:
        """Open System Settings on the Accessibility page, and ask macOS to list this app there."""
        self.actions.request_access()
        return self.actions.open_settings()

    def test_action(self, action: str, delay_s: float = 0.0) -> dict[str, Any]:
        """Press one action after `delay_s` (so there is time to click into another app). Returns the outcome."""
        if action not in S.ACTIONS or action == "off":
            raise ValueError(f"unknown action {action!r}")
        if delay_s > 0:
            time.sleep(min(delay_s, 10.0))                # runs on the web server's thread, not on the service loop
        outcome = self.actions.perform(action)

        async def note() -> None:
            self._add_timeline({"t": self._now(), "kind": "gesture", "level": "info" if outcome["result"] in ("sent", "dry_run") else "attention",
                                "icon": "test", "text": f"Test: {S.action_label(action)}", "detail": outcome["detail"]})
        self._call(note())
        return outcome

    def inject(self, event: dict[str, Any]) -> None:
        """Feed one normalised event straight into the rules (tests only)."""
        async def do() -> None:
            self._apply(self._rules.on_event(event, self._now()))
        self._call(do())


def _describe_change(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Timeline lines for the setting changes a caregiver would want to see later."""
    out = []
    if old["control"]["enabled"] != new["control"]["enabled"]:
        out.append("Gesture control turned " + ("on" if new["control"]["enabled"] else "off"))
    if old["monitoring"]["enabled"] != new["monitoring"]["enabled"]:
        out.append("Room monitoring turned " + ("on" if new["monitoring"]["enabled"] else "off"))
    if S.mapping(old) != S.mapping(new):
        out.append(f"Gestures now control: {S.PRESETS[new['control']['preset']]['label']}")
    a, b = old["monitoring"]["no_movement"], new["monitoring"]["no_movement"]
    if (a["enabled"], a["seconds"]) != (b["enabled"], b["seconds"]):
        out.append(f"No-movement alert: after {fmt_duration(b['seconds'])}" if b["enabled"] else "No-movement alert turned off")
    if old["monitoring"]["help"]["enabled"] != new["monitoring"]["help"]["enabled"]:
        out.append("Help gesture turned " + ("on" if new["monitoring"]["help"]["enabled"] else "off"))
    return out

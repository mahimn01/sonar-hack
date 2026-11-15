"""menubar.py: the Sonar Care item in the macOS menu bar.

What it does
    Shows the sensor's state at a glance and gives one-click control:

        ◉ Sonar                         <- title changes with the state (HELP, Calibrating, Weak signal, Offline)
        Sonar live · signal good (43 dB)
        Presence: someone here · still 40 s
        Breathing: 15/min (early estimate)
        Last gesture: Swipe right, Right arrow · 12 s ago
        ---------------------------------
        ✓ Gesture control      ✓ Room monitoring      Gestures control > Slides / Media / Scroll / Custom
        Open Dashboard   Open Settings   Run Setup…   Calibrate
        Acknowledge alert   Cancel help request        (only while there is one)
        Stop engine / Start engine
        Quit Sonar Care

    The decisions about WHAT to show live in menu_model(), a plain function of the service's
    state that is unit-tested. The rumps code below only draws that model.

Why rumps
    It gives a real menu bar item from plain Python: no Xcode, no app bundle, no signing.
    If rumps (or pyobjc) is missing, run() returns False and the app carries on without a
    menu bar item (the same as --headless).

What it changes
    Nothing by itself. The menu items call the service (settings, acknowledge, engine start/stop).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable

GESTURE_TEXT = {"swipe_left": "Swipe left", "swipe_right": "Swipe right", "push": "Push"}
RESULT_TEXT = {"sent": "", "dry_run": " (dry run)", "blocked": " (blocked)", "failed": " (failed)", "held": " (waiting for a second push)",
               "armed": " (waiting for a second push)", "help": " (help)", "ignored": " (ignored)", "paused": " (paused)",
               "practice": " (practice)", "off": " (control is off)", "unmapped": " (no action)"}


def _ago(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 2:
        return "just now"
    if s < 60:
        return f"{s} s ago"
    if s < 3600:
        return f"{s // 60} min ago"
    return f"{s // 3600} h ago"


def _dur(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s} s"
    return f"{s // 60} min {s % 60:02d} s" if s < 3600 else f"{s // 3600} h {s % 3600 // 60} min"


def menu_model(state: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    """Everything the menu shows, worked out from one care_state snapshot. Pure: no clock, no system calls."""
    now = state.get("t", time.time()) if now is None else now
    engine = state.get("engine") or {}
    link = state.get("link") or {}
    sonar = state.get("sonar") or {}
    presence = state.get("presence") or {}
    breathing = state.get("breathing") or {}
    still = (state.get("stillness") or {}).get("still_s")
    control = state.get("control") or {}
    monitoring = state.get("monitoring") or {}
    alerts = [a for a in state.get("alerts") or [] if a.get("state") in ("pending", "active")]
    health = link.get("health", "connecting")
    estate = engine.get("state", "external")
    snr = sonar.get("snr_db")

    # ---- line 1: the sensor ---------------------------------------------------------------------
    if state.get("starting"):
        sensor, level = "Starting…", "starting"
    elif estate == "failed":
        sensor, level = "Engine could not start", "offline"
    elif estate in ("starting", "restarting"):
        sensor, level = ("Engine starting…" if estate == "starting" else f"Engine restarting in {engine.get('retry_in_s') or 0:.0f} s"), "starting"
    elif estate == "stopped" and health != "live":
        sensor, level = "Engine stopped", "offline"
    elif health == "connecting":
        sensor, level = "Connecting to the sensor…", "starting"
    elif health != "live":
        sensor, level = "Sensor offline", "offline"
    elif sonar.get("calibrating"):
        sensor, level = "Calibrating: keep still", "calibrating"
    elif sonar.get("signal") == "weak":
        sensor, level = f"Weak signal · {snr:.0f} dB" if snr is not None else "Weak signal", "weak"
    else:
        quality = {"good": "signal good", "fair": "signal fair"}.get(sonar.get("signal"), "signal")
        sensor, level = f"Sonar live · {quality}" + (f" ({snr:.0f} dB)" if snr is not None else ""), "live"
    if sonar.get("source") not in (None, "live") and level == "live":
        sensor += " · simulated"

    # ---- title: the loudest thing wins --------------------------------------------------------------
    top = alerts[0] if alerts else None
    if top is not None:
        if top["kind"] == "help":
            left = max(0, round((top.get("escalate_at") or now) - now)) if top["state"] == "pending" else 0
            title = f"▲ Help in {left} s" if top["state"] == "pending" else "▲ HELP"
        else:
            title = "▲ " + str(top.get("title") or "Alert")
        level = "alert"
    else:
        title = {"live": "◉ Sonar", "calibrating": "◌ Calibrating", "weak": "◍ Weak signal", "offline": "○ Sonar off",
                 "starting": "◌ Sonar"}[level]

    # ---- the other status lines -----------------------------------------------------------------------
    lines = [sensor]
    if estate == "failed" and engine.get("last_error"):
        lines.append(str(engine["last_error"])[:90])
    if health == "live":
        p = presence.get("state")
        if p == "present" and presence.get("confirmed"):
            text = "Presence: someone here"
            if still is not None and still >= 5:
                text += f" · still {_dur(still)}"
            elif presence.get("moving"):
                text += " · moving"
        elif p == "present":
            text = "Presence: checking (weak sign of someone)"
        elif p == "entering":
            text = "Presence: someone arriving"
        elif p == "empty":
            text = "Presence: nobody in range"
        else:
            text = "Presence: waiting"
        lines.append(text)
        bpm = breathing.get("bpm")
        lines.append(f"Breathing: {bpm:.0f}/min (early estimate)" if bpm is not None else "Breathing: — (no confident reading)")
    last = state.get("last_gesture")
    if last and last.get("gesture"):
        what = GESTURE_TEXT.get(last["gesture"], str(last["gesture"]))
        if last.get("action_label") and last.get("result") in ("sent", "dry_run", "blocked", "failed", "held"):
            what += f", {last['action_label']}"
        lines.append(f"Last gesture: {what}{RESULT_TEXT.get(last.get('result'), '')} · {_ago(now - (last.get('t') or now))}")
    else:
        lines.append("Last gesture: none yet")
    for a in alerts:
        lines.append(("⚠ " if a["level"] == "attention" else "▲ ") + str(a.get("title")) + (" (sending soon)" if a["state"] == "pending" else ""))

    keys = control.get("keys") or {}
    notes = []
    if state.get("dry_run"):
        notes.append("Dry run: no keys are pressed")
    elif keys.get("mode") == "blocked":
        notes.append("Key presses are blocked: allow in Accessibility…")
    if state.get("practice"):
        notes.append("Practice mode: nothing is pressed")

    if estate == "attached" or not engine.get("managed", False):
        engine_item, engine_action = "Engine: started elsewhere", None
    elif estate in ("running", "starting", "restarting", "stopping"):
        engine_item, engine_action = "Stop engine", "stop"
    else:
        engine_item, engine_action = "Start engine", "start"

    return {
        "title": title, "level": level, "lines": lines, "notes": notes,
        "control_on": bool(control.get("enabled")), "monitoring_on": bool(monitoring.get("enabled")),
        "preset": control.get("preset"),
        "needs_access": keys.get("mode") == "blocked",
        "can_ack": any(a["state"] == "active" for a in alerts),
        "can_cancel_help": any(a["state"] == "pending" and a["kind"] == "help" for a in alerts),
        "engine_item": engine_item, "engine_action": engine_action,
        "can_calibrate": health == "live",
    }


def available() -> bool:
    """Can a menu bar item be shown on this machine?"""
    if sys.platform != "darwin":
        return False
    try:
        import rumps  # noqa: F401
    except Exception:                                      # noqa: BLE001  (missing or broken install: run headless)
        return False
    return True


def run(service: Any, open_page: Callable[[str], None], presets: list[tuple[str, str]], on_quit: Callable[[], None]) -> bool:
    """Show the menu bar item and run until Quit. Returns False at once if it cannot be shown (caller goes headless).

    Must be called on the main thread: macOS only lets the main thread own the menu bar.
    """
    if not available():
        return False
    import rumps

    class App(rumps.App):
        def __init__(self) -> None:
            super().__init__("Sonar Care", title="◌ Sonar", quit_button=None)
            self.status_items = [rumps.MenuItem(" ") for _ in range(8)]
            self.control = rumps.MenuItem("Gesture control", callback=self.toggle_control)
            self.monitoring = rumps.MenuItem("Room monitoring", callback=self.toggle_monitoring)
            self.preset_menu = rumps.MenuItem("Gestures control")
            self.preset_items = {}
            for preset_id, label in presets:
                item = rumps.MenuItem(label, callback=self.pick_preset)
                item.preset_id = preset_id
                self.preset_items[preset_id] = item
                self.preset_menu.add(item)
            self.ack = rumps.MenuItem("Acknowledge alert", callback=self.acknowledge)
            self.cancel = rumps.MenuItem("Cancel help request", callback=self.cancel_help)
            self.access = rumps.MenuItem("Allow key presses (Accessibility)…", callback=self.open_access)
            self.calibrate = rumps.MenuItem("Calibrate", callback=self.do_calibrate)
            self.engine = rumps.MenuItem("Stop engine", callback=self.toggle_engine)
            self.menu = [
                *self.status_items, None,
                self.ack, self.cancel, self.access,
                self.control, self.monitoring, self.preset_menu, None,
                rumps.MenuItem("Open Dashboard", callback=lambda _: open_page("")),
                rumps.MenuItem("Open Settings", callback=lambda _: open_page("#/settings")),
                rumps.MenuItem("Run Setup…", callback=lambda _: open_page("#/setup")),
                self.calibrate, None,
                self.engine, None,
                rumps.MenuItem("Quit Sonar Care", callback=self.quit),
            ]
            self.model: dict[str, Any] = {}
            self.timer = rumps.Timer(self.refresh, 1.0)
            self.timer.start()

        # ---- drawing ----
        def refresh(self, _: Any = None) -> None:
            try:
                self._draw(menu_model(service.snapshot(), time.time()))
            except Exception as err:                       # noqa: BLE001  (never let a drawing bug kill the menu)
                print(f"[care] menu refresh failed: {err!r}", file=sys.stderr, flush=True)

        def _draw(self, m: dict[str, Any]) -> None:
            if not self.model and os.environ.get("SONAR_CARE_DEBUG"):
                print(f"[care] menu bar item is up: title {m['title']!r}, lines {m['lines']}", flush=True)
            self.model = m
            if self.title != m["title"]:
                self.title = m["title"]
            texts = m["lines"] + m["notes"]
            for i, item in enumerate(self.status_items):
                text = texts[i] if i < len(texts) else None
                self._show(item, text is not None)
                if text is not None and item.title != text:
                    item.title = text
            self.control.state = 1 if m["control_on"] else 0
            self.monitoring.state = 1 if m["monitoring_on"] else 0
            for preset_id, item in self.preset_items.items():
                item.state = 1 if preset_id == m["preset"] else 0
            self._show(self.ack, m["can_ack"])
            self._show(self.cancel, m["can_cancel_help"])
            self._show(self.access, m["needs_access"])
            self.calibrate.set_callback(self.do_calibrate if m["can_calibrate"] else None)
            self.engine.title = m["engine_item"]
            self.engine.set_callback(self.toggle_engine if m["engine_action"] else None)

        @staticmethod
        def _show(item: Any, visible: bool) -> None:
            try:
                item._menuitem.setHidden_(not visible)     # rumps has no public "hide" for an item
            except Exception:                              # noqa: BLE001
                pass

        # ---- actions (each talks to the service; a failure is printed, never raised into Cocoa) ----
        def _do(self, what: str, fn: Callable[[], Any]) -> None:
            try:
                fn()
            except Exception as err:                       # noqa: BLE001
                print(f"[care] {what} failed: {err!r}", file=sys.stderr, flush=True)
            self.refresh()

        def toggle_control(self, _: Any) -> None:
            self._do("toggle gesture control", lambda: service.update_settings({"control": {"enabled": not self.model.get("control_on")}}))

        def toggle_monitoring(self, _: Any) -> None:
            self._do("toggle monitoring", lambda: service.update_settings({"monitoring": {"enabled": not self.model.get("monitoring_on")}}))

        def pick_preset(self, sender: Any) -> None:
            self._do("choose preset", lambda: service.update_settings({"control": {"preset": sender.preset_id}}))

        def acknowledge(self, _: Any) -> None:
            self._do("acknowledge", service.acknowledge)

        def cancel_help(self, _: Any) -> None:
            self._do("cancel help", service.cancel_help)

        def open_access(self, _: Any) -> None:
            self._do("open Accessibility settings", service.open_accessibility)

        def do_calibrate(self, _: Any) -> None:
            self._do("calibrate", service.calibrate)

        def toggle_engine(self, _: Any) -> None:
            action = self.model.get("engine_action")
            if action:
                self._do(f"{action} engine", lambda: service.engine_command(action))

        def quit(self, _: Any) -> None:
            self.timer.stop()
            try:
                on_quit()
            finally:
                rumps.quit_application()

    try:
        try:                                               # a menu bar item only: no Dock icon, no app menu
            from AppKit import NSApplication
            NSApplication.sharedApplication().setActivationPolicy_(1)
        except Exception:                                  # noqa: BLE001
            pass
        # rumps turns Ctrl+C into "terminate the app", which ends the process without returning here.
        # Whatever way the app ends, stop the service and the engine first.
        try:
            rumps.events.before_quit.register(on_quit)
        except Exception:                                  # noqa: BLE001  (older rumps: the Quit item and SIGTERM still clean up)
            pass
        App().run()
    except Exception as err:                               # noqa: BLE001  (no window server, broken pyobjc, ...)
        print(f"[care] menu bar unavailable ({err!r}); continuing without it", file=sys.stderr, flush=True)
        return False
    return True

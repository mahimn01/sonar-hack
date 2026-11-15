"""actions.py: turns "press Right arrow" into a real key press on this Mac (or only logs it).

What it does
    perform("right") presses and releases that key in whatever app is in front. Action names
    are the ones in settings.ACTIONS ("right", "space", "media_play_pause", ...).

    Three ways to run:
      * dry run   nothing is pressed; every action is logged as "would press ...". Used for
                  tests, for --dry-run, and on any system that is not macOS.
      * live      keys are sent with macOS CoreGraphics (no extra install, no code signing);
                  media keys with AppKit when pyobjc is installed, else by asking Music/Spotify.
      * blocked   macOS has not granted Accessibility permission, so key presses would be
                  silently dropped. perform() says so instead of pretending.

Why
    This is the only file that can type into other apps, so it is the only file that needs
    the Accessibility permission and the only place a dry run has to be enforced.

What it changes
    Live mode sends key events to the frontmost app. Nothing on disk. (Phase 1 ships the dry
    run; the live macOS backends are added in phase 2.)
"""

from __future__ import annotations

import sys
import time
from typing import Any

from . import settings as S


class Actions:
    """Performs actions, or records what it would have done. Safe to call from any thread."""

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self.log: list[dict[str, Any]] = []       # the last 100 actions, newest last (tests and the dry-run view read it)

    # ---- what the settings page shows ---------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Whether key presses can reach other apps right now, in words fit for the UI."""
        if self.dry_run:
            return {"mode": "dry_run", "trusted": None, "backend": "dry run", "can_press": False,
                    "note": "Dry run: actions are logged, no key is pressed."}
        return {"mode": "unavailable", "trusted": None, "backend": "none", "can_press": False,
                "note": f"Key presses are not available on {sys.platform}."}

    # ---- pressing -----------------------------------------------------------------------------

    def perform(self, action: str) -> dict[str, Any]:
        """Do one action. Returns {"result": "sent" | "dry_run" | "blocked" | "failed", "detail": str}."""
        label = S.action_label(action)
        if action not in S.ACTIONS or action == "off":
            out = {"result": "failed", "detail": f"unknown action {action!r}"}
        elif self.dry_run:
            out = {"result": "dry_run", "detail": f"would press {label}"}
        else:
            out = {"result": "blocked", "detail": "key presses are not available on this system"}
        self.log.append({"t": time.time(), "action": action, "label": label, **out})
        del self.log[:-100]
        return out

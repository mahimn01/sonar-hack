"""notify.py: tells a caregiver about an alert outside the browser (notification + sound + voice).

What it does
    For a raised alert: a macOS notification banner, a system sound, and the alert's title
    spoken aloud ("Help requested"). For a reminder: the sound again. Everything uses tools
    that ship with macOS (osascript, afplay, say): nothing to install, nothing to sign.

    It is rate-limited: the same alert cannot sound more than once every few seconds, and no
    two sounds start within two seconds of each other, whatever the rules ask for.

    With enabled=False (tests, --quiet) it only records what it would have done.

Why
    An alert must be noticed when no browser window is open or the dashboard is behind the
    slides, and by a caregiver who is not looking at the screen.

Why the caller gets "seconds of sound" back
    The sonar listens through the same speakers. A loud alert sound can look like movement, so
    the rules ignore movement for as long as the sound lasts (see rules.CareRules.sound_played).

What it changes
    Shows notifications and plays audio through the speakers. Nothing on disk.
    Notifications from osascript appear under "Script Editor" in System Settings > Notifications;
    if they are switched off there, or a Focus mode is on, only the sound and voice get through.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

MIN_INTERVAL_S = 4.0          # never announce the same alert more often than this
MIN_GAP_S = 2.0               # never start two sounds closer together than this
SOUND_DIR = Path("/System/Library/Sounds")
# level -> (sound file, seconds it lasts)
SOUNDS: dict[str, tuple[str, float]] = {"alert": ("Sosumi.aiff", 1.0), "attention": ("Glass.aiff", 1.6)}
SPEECH_SECONDS = 1.8          # "Help requested", "No movement", ...

Runner = Callable[[list[str]], None]


def _spawn(cmd: list[str]) -> None:
    """Start a helper program and do not wait for it (an alert must never block the service)."""
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


def notification_command(title: str, subtitle: str, body: str) -> list[str]:
    """osascript command for a banner. The texts travel as arguments, never inside the script, so quotes cannot break it."""
    return ["osascript",
            "-e", "on run argv",
            "-e", "display notification (item 3 of argv) with title (item 1 of argv) subtitle (item 2 of argv)",
            "-e", "end run",
            title, subtitle, body]


def sound_command(level: str, volume: float, speak: str | None = None) -> tuple[list[str], float]:
    """Command for the chime, optionally followed by a spoken line. Returns (command, seconds of audio)."""
    name, seconds = SOUNDS.get(level, SOUNDS["attention"])
    vol = f"{max(0.0, min(1.0, volume)):.2f}"
    path = str(SOUND_DIR / name)
    if not speak:
        return ["afplay", "-v", vol, path], seconds
    # One small shell line so the voice starts after the chime. The texts are arguments ($1..$3), never part of the script.
    return ["/bin/sh", "-c", 'afplay -v "$1" "$2"; say "[[volm $1]] $3"', "sonar-care", vol, path, speak], seconds + SPEECH_SECONDS


class Notifier:
    """Notification + sound + voice for alerts. Safe to call from the service loop; never raises."""

    def __init__(self, enabled: bool = True, runner: Runner | None = None) -> None:
        self.enabled = enabled and (sys.platform == "darwin" or runner is not None)
        self._run = runner or _spawn
        self.sent: list[dict[str, Any]] = []      # what was (or would have been) announced, newest last
        self._last: dict[str, float] = {}
        self._last_sound_at = -1e9

    def alert(self, alert: dict[str, Any], *, reminder: bool, notification: bool, sound: bool, volume: float,
              speak: bool = True, now: float | None = None) -> float:
        """Announce one alert. Returns the seconds of audio started (0 if none), so the caller can ignore that time."""
        now = time.time() if now is None else now
        key = str(alert.get("id"))
        if now - self._last.get(key, -1e9) < MIN_INTERVAL_S:
            return 0.0
        self._last[key] = now
        title = str(alert.get("title") or "Alert")
        detail = str(alert.get("detail") or "")
        show = bool(notification and not reminder)
        play = bool(sound and now - self._last_sound_at >= MIN_GAP_S)
        say = bool(play and speak and not reminder)
        record = {"t": now, "id": key, "kind": alert.get("kind"), "title": title, "detail": detail, "reminder": reminder,
                  "notification": show, "sound": play, "speech": say, "delivered": False, "seconds": 0.0}
        self.sent.append(record)
        del self.sent[:-50]
        if not self.enabled:
            return 0.0
        seconds = 0.0
        try:
            if show:
                self._run(notification_command("Sonar Care", title, detail))
            if play:
                cmd, seconds = sound_command(str(alert.get("level") or "attention"), volume, title if say else None)
                self._run(cmd)
                self._last_sound_at = now
            record["delivered"] = True
            record["seconds"] = seconds
        except OSError as err:                            # a missing tool must never take the service down
            print(f"[care] could not announce the alert: {err}", file=sys.stderr, flush=True)
        return seconds

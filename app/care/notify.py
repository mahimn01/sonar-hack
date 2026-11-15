"""notify.py: tells a caregiver about an alert outside the browser (notification + sound).

What it does
    alert(...) shows a macOS notification and plays a system sound for a raised alert, and
    plays the sound again for a reminder. It is rate-limited: the same alert cannot fire more
    than once every few seconds, whatever the rules ask for.

    With enabled=False (tests, --quiet) it only records what it would have done.

Why
    An alert must be noticed when no browser window is open or the dashboard is behind the
    slides.

What it changes
    Shows notifications and plays audio through the speakers. Nothing on disk. (Phase 1 ships
    the recording version; the macOS notification and sound are added in phase 2.)
"""

from __future__ import annotations

import time
from typing import Any

MIN_INTERVAL_S = 4.0          # never notify for the same alert more often than this
SOUND_SECONDS = 1.5           # how long an alert sound lasts (the rules ignore movement for this long)


class Notifier:
    """Notification + sound for alerts. Safe to call from the service loop; never raises."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.sent: list[dict[str, Any]] = []      # what was (or would have been) shown, newest last
        self._last: dict[str, float] = {}

    def alert(self, alert: dict[str, Any], *, reminder: bool, notification: bool, sound: bool, volume: float,
              now: float | None = None) -> float:
        """Announce one alert. Returns the seconds of sound played (0 if none), so the caller can ignore that time."""
        now = time.time() if now is None else now
        key = str(alert.get("id"))
        if now - self._last.get(key, -1e9) < MIN_INTERVAL_S:
            return 0.0
        self._last[key] = now
        record = {"t": now, "id": key, "kind": alert.get("kind"), "title": alert.get("title"), "detail": alert.get("detail"),
                  "reminder": reminder, "notification": bool(notification and not reminder), "sound": bool(sound),
                  "delivered": False}
        self.sent.append(record)
        del self.sent[:-50]
        return 0.0

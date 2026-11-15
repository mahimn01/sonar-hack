"""rules.py: every decision Sonar Care makes, in one place, with no input or output.

What it does
    Takes the engine's events (already normalised by app/adapter.py), the user's settings and the
    current time, and decides:
      * which gesture presses which key, and which gestures to ignore (too unsure, too soon after
        the last one, the hand coming back after a swipe);
      * when two pushes in a row mean "help requested";
      * when to raise "No movement" (someone is present, nothing has moved, breathing not confirmed);
      * when to raise "No breathing movement" (a steady breathing rate was seen, then it stopped);
      * when an alert is acknowledged, resolves by itself, needs a reminder, or can be put away.
    It returns a list of "effects" (plain dicts) that service.py carries out: press a key, show and
    sound an alert, add a line to the timeline.

Why it is separate
    Alerts must be decided in exactly one place, so they fire the same way whether or not a browser
    is open. Keeping this file free of clocks, sockets and system calls means every rule can be
    tested by feeding events and moving a fake clock (app/tests/test_care_rules.py).

How time works here
    Every method takes `now` (seconds, any steady clock). Nothing in this file reads the real clock.

What it changes
    Nothing outside its own memory.

The help gesture and the push action (the documented decision)
    A push can mean "press the push key" or the first half of "help". With push_timing = "wait"
    (the default) a single push presses its key only after the help window (2.5 s) has passed
    without a second push, so one physical action never does both. With "instant" the key is
    pressed at once and a second push then also raises help. Swipes always act immediately.
    While a help alert is open, gestures press nothing; another push repeats the request.

    What counts as "two pushes": the model has no "pull" class, so the hand coming back after a
    push is very often reported as a second push 0.4-1.0 s later (measured: 4 of 10 single pushes).
    A push event sooner than min_gap_s (1.0 s) after the first one is therefore the same push. The
    second push must arrive between min_gap_s and window_s after the first.

    Because about 1 single push in 12 still looks like two, a help request first shows on screen
    for confirm_s (3 s) with a Cancel button. Only a person at the screen can cancel it; a gesture
    never can, and another push sends it at once. A real call must never be dropped by accident.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keys_bridge import GestureGate  # noqa: E402  (confidence floor, cooldown, return-stroke filter)

from . import settings as S  # noqa: E402

Effect = dict[str, Any]

# How sure the engine must be that somebody is really there before the room monitor trusts a
# "present" (a noisy room can report a faint "present" with 5 % confidence and nobody in it).
PRESENCE_CONFIRM_CONFIDENCE = 0.6
PRESENCE_CONFIRM_EVENTS = 2            # that many presence messages in a row (they arrive once a second)

# No-breathing alarm (same numbers as web/js/config.js, so both screens behave alike).
BREATH_MIN_CONFIDENCE = 0.4            # a rate below this confidence does not count as "steady breathing"
BREATH_MOTION_HOLD_S = 6.0             # no alarm this soon after a gesture, a presence change or body movement
BREATH_ARM_S = 5.0                     # a steady rate must have been on screen this long before the alarm arms
BREATH_FRESH_S = 2.5                   # ... and the last good reading must be this recent

LINK_GRACE_S = 3.0                     # the engine may be silent this long before monitoring pauses
START_GRACE_S = 20.0                   # ... and this long while it starts up for the first time
RESOLVED_LINGER_S = 20.0               # a self-resolved alert nobody acknowledged stays visible this long
ACKED_LINGER_S = 8.0                   # an acknowledged help request stays visible this long ("answered in 12 s")
MAX_REMINDERS = 10
PRACTICE_TIMEOUT_S = 90.0              # the tutorial must say "still here" this often, or practice mode ends

ALERT_PRIORITY = {"help": 0, "no_breathing": 1, "no_movement": 2}


def fmt_duration(seconds: float) -> str:
    """12 -> '12 s', 75 -> '1 min 15 s', 3700 -> '1 h 1 min'."""
    s = max(0, int(seconds + 0.5))
    if s < 60:
        return f"{s} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} min {s:02d} s" if s else f"{m} min"
    h, m = divmod(m, 60)
    return f"{h} h {m} min"


# ---- breathing waveform memory (Python twin of the Timeline in web/js/store.js) ---------------------

class WaveTimeline:
    """Recent chest-movement samples with their times. Overlapping windows replace older samples."""

    def __init__(self, max_seconds: float = 60.0) -> None:
        self.t: list[float] = []
        self.v: list[float] = []
        self.max = max_seconds

    def merge(self, samples: list[float], fs: float, t_last: float, t0: float | None = None) -> None:
        n = len(samples)
        if not n or not fs or fs <= 0:
            return
        dt = 1.0 / fs
        start = t0 if t0 is not None else t_last - (n - 1) * dt
        cut = len(self.t)
        while cut > 0 and self.t[cut - 1] >= start - dt * 0.5:
            cut -= 1
        del self.t[cut:], self.v[cut:]
        if cut > 0 and start - self.t[cut - 1] > max(1.5, 4 * dt):     # a gap: break the line
            self.t.append(self.t[cut - 1] + dt)
            self.v.append(math.nan)
        for i, value in enumerate(samples):
            self.t.append(start + i * dt)
            self.v.append(value)
        limit = self.t[-1] - self.max
        drop = 0
        while drop < len(self.t) and self.t[drop] < limit:
            drop += 1
        if drop:
            del self.t[:drop], self.v[:drop]

    def last_time(self) -> float | None:
        return self.t[-1] if self.t else None

    def index_at(self, x: float) -> int:
        """Index of the last sample at or before time x, -1 if none."""
        lo, hi, ans = 0, len(self.t) - 1, -1
        while lo <= hi:
            mid = (lo + hi) >> 1
            if self.t[mid] <= x:
                ans, lo = mid, mid + 1
            else:
                hi = mid - 1
        return ans

    def rms(self, t0: float, t1: float) -> float:
        """Spread around the mean over [t0, t1]; NaN with fewer than 4 samples."""
        total = total2 = 0.0
        n = 0
        i = max(0, self.index_at(t0))
        while i < len(self.t) and self.t[i] <= t1:
            v = self.v[i]
            if self.t[i] >= t0 and math.isfinite(v):
                total += v
                total2 += v * v
                n += 1
            i += 1
        if n < 4:
            return math.nan
        mean = total / n
        return math.sqrt(max(0.0, total2 / n - mean * mean))

    def clear(self) -> None:
        self.t.clear()
        self.v.clear()


def to_millimetres(wave: list[float], amplitude_mm: float | None) -> list[float]:
    """Undo the engine's "scale each window to its own peak".

    The engine stretches every window to full height, so a held breath would be stretched back up.
    When it also reports the real peak-to-peak chest movement in millimetres, rescale the window so
    the flat-line check below works on real size.
    """
    if amplitude_mm is None or not amplitude_mm >= 0 or len(wave) < 5:
        return wave
    finite = sorted(v for v in wave if math.isfinite(v))
    if len(finite) < 5:
        return wave
    hi = finite[int(0.97 * (len(finite) - 1) + 0.5)]
    lo = finite[int(0.03 * (len(finite) - 1) + 0.5)]
    pp = hi - lo
    if not pp > 1e-9:
        return wave
    k = amplitude_mm / pp
    return [v * k for v in wave]


# ---- alerts -------------------------------------------------------------------------------------

class AlertBook:
    """The open alerts and their life cycle: raised -> acknowledged and/or resolved -> put away.

    At most one alert per kind is open at a time. "help" stays until a person acknowledges it.
    "no_movement" and "no_breathing" also resolve by themselves when the condition ends.
    """

    def __init__(self) -> None:
        self._by_kind: dict[str, dict[str, Any]] = {}
        self._seq = 0

    def get(self, kind: str) -> dict[str, Any] | None:
        return self._by_kind.get(kind)

    def is_open(self, kind: str) -> bool:
        """The condition is still going on (acknowledged or not)."""
        a = self._by_kind.get(kind)
        return a is not None and a["resolved_at"] is None and not (kind == "help" and a["acked_at"] is not None)

    def needs_attention(self, kind: str) -> bool:
        """Raised (or about to be) and nobody has acknowledged it yet."""
        a = self._by_kind.get(kind)
        return a is not None and a["state"] in ("pending", "active")

    def is_pending(self, kind: str) -> bool:
        """Seen, shown on screen with a Cancel button, not yet sent out."""
        a = self._by_kind.get(kind)
        return a is not None and a["state"] == "pending"

    def listing(self) -> list[dict[str, Any]]:
        return sorted((dict(a) for a in self._by_kind.values()), key=lambda a: (ALERT_PRIORITY.get(a["kind"], 9), a["raised_at"]))

    def raise_(self, kind: str, level: str, title: str, detail: str, since: float, now: float,
               pending_s: float = 0.0, **extra: Any) -> list[Effect]:
        """Open an alert. With pending_s > 0 it first waits that long on screen (cancellable) before it goes out."""
        out: list[Effect] = []
        old = self._by_kind.pop(kind, None)
        if old is not None:                                 # a finished one of the same kind was still on screen
            out.append({"effect": "alert", "change": "dismissed", "alert": dict(old)})
        self._seq += 1
        pending = pending_s > 0
        alert = {
            "id": f"{kind}-{self._seq}", "kind": kind, "level": level, "state": "pending" if pending else "active",
            "title": title, "detail": detail,
            "since": since,              # when the condition began (first push, last movement, last breath)
            "raised_at": now,            # when it was detected
            "active_at": None if pending else now,           # when it went out (sound, notification)
            "escalate_at": now + pending_s if pending else None,
            "acked_at": None, "resolved_at": None, "resolution": None,
            "ack_s": None,               # seconds from going out to being acknowledged
            "count": 1, "reminders": 0, "last_sound_at": now,
        }
        alert.update(extra)
        self._by_kind[kind] = alert
        return out + [{"effect": "alert", "change": "pending" if pending else "raised", "alert": dict(alert)}]

    def escalate(self, kind: str, now: float) -> list[Effect]:
        """A pending alert goes out now (its cancel time ran out, or the person asked again)."""
        a = self._by_kind.get(kind)
        if a is None or a["state"] != "pending":
            return []
        a["state"], a["active_at"], a["escalate_at"], a["last_sound_at"] = "active", now, None, now
        return [{"effect": "alert", "change": "raised", "alert": dict(a)}]

    def cancel(self, kind: str, now: float, why: str) -> list[Effect]:
        """Drop a pending alert before it went out."""
        a = self._by_kind.get(kind)
        if a is None or a["state"] != "pending":
            return []
        del self._by_kind[kind]
        a["state"], a["resolved_at"], a["resolution"], a["escalate_at"] = "cancelled", now, why, None
        return [{"effect": "alert", "change": "dismissed", "alert": dict(a)},
                _timeline("alert", "info", "check", f"{a['title']}: {why}", now, alert_id=a["id"])]

    def update(self, kind: str, now: float, change: str = "updated", **fields: Any) -> list[Effect]:
        """Change fields of an open alert. change="reminder" also asks for the sound again."""
        a = self._by_kind.get(kind)
        if a is None or all(a.get(k) == v for k, v in fields.items()):
            return []
        a.update(fields)
        if change == "reminder":
            a["last_sound_at"] = now
        return [{"effect": "alert", "change": change, "alert": dict(a)}]

    def resolve(self, kind: str, now: float, resolution: str) -> list[Effect]:
        a = self._by_kind.get(kind)
        if a is None or a["resolved_at"] is not None:
            return []
        a["resolved_at"], a["resolution"] = now, resolution
        lasted = fmt_duration(now - a["since"])
        out: list[Effect] = []
        if a["acked_at"] is not None:                       # somebody already saw it: nothing left to show
            a["state"] = "resolved"
            del self._by_kind[kind]
            out.append({"effect": "alert", "change": "dismissed", "alert": dict(a)})
        else:
            a["state"] = "resolved"
            out.append({"effect": "alert", "change": "resolved", "alert": dict(a)})
        out.append(_timeline("alert", "ok", "check", f"{a['title']}: {resolution}", now, detail=f"lasted {lasted}", alert_id=a["id"]))
        return out

    def acknowledge(self, now: float, alert_id: str | None = None) -> list[Effect]:
        """Acknowledge one alert by id, or every alert that still needs it."""
        out: list[Effect] = []
        for kind in list(self._by_kind):
            a = self._by_kind[kind]
            if alert_id is not None and a["id"] != alert_id:
                continue
            if a["acked_at"] is not None or a["state"] == "pending":     # a pending one is cancelled, not acknowledged
                continue
            a["acked_at"] = now
            a["ack_s"] = round(now - a["active_at"], 1)
            waited = fmt_duration(a["ack_s"])
            if a["resolved_at"] is not None:                # already over: acknowledging puts it away
                del self._by_kind[kind]
                out.append({"effect": "alert", "change": "dismissed", "alert": dict(a)})
                out.append(_timeline("alert", "info", "check", f"{a['title']}: acknowledged", now, detail=f"after {waited}", alert_id=a["id"]))
                continue
            a["state"] = "acknowledged"
            out.append({"effect": "alert", "change": "acknowledged", "alert": dict(a)})
            out.append(_timeline("alert", "ok", "check", f"{a['title']}: acknowledged after {waited}", now, alert_id=a["id"]))
        return out

    def tick(self, now: float, repeat_s: float) -> list[Effect]:
        """Reminders for unanswered red alerts, and putting finished alerts away."""
        out: list[Effect] = []
        for kind in list(self._by_kind):
            a = self._by_kind[kind]
            if a["state"] == "pending" and now >= a["escalate_at"]:
                out += self.escalate(kind, now)
                out.append(_timeline("alert", "alert", kind, a["title"], now, detail=a["detail"], alert_id=a["id"]))
                continue
            if a["state"] == "active" and a["level"] == "alert" and repeat_s > 0 and a["reminders"] < MAX_REMINDERS \
                    and now - a["last_sound_at"] >= repeat_s:
                a["reminders"] += 1
                a["last_sound_at"] = now
                out.append({"effect": "alert", "change": "reminder", "alert": dict(a)})
            done_at = a["resolved_at"] if a["state"] == "resolved" else a["acked_at"] if kind == "help" and a["acked_at"] is not None else None
            linger = RESOLVED_LINGER_S if a["state"] == "resolved" else ACKED_LINGER_S
            if done_at is not None and now - done_at >= linger:
                del self._by_kind[kind]
                out.append({"effect": "alert", "change": "dismissed", "alert": dict(a)})
        return out


def _timeline(kind: str, level: str, icon: str, text: str, now: float, **extra: Any) -> Effect:
    """One line for the event timeline. level: info | ok | attention | alert."""
    entry = {"t": now, "kind": kind, "level": level, "icon": icon, "text": text}
    entry.update({k: v for k, v in extra.items() if v is not None})
    return {"effect": "timeline", "entry": entry}


# ---- the rules ----------------------------------------------------------------------------------------

class CareRules:
    """All of Sonar Care's decisions. Feed it events and time; carry out the effects it returns."""

    def __init__(self, settings: dict[str, Any], now: float) -> None:
        self.settings = settings
        self.alerts = AlertBook()
        self._gate = GestureGate()
        self._configure_gate()

        # engine link and signal
        self.connected = False
        self._ever_connected = False
        self._link_changed_at = now
        self._link_logged = False          # "Sensor offline" was written for the current outage
        self.calibrating = False
        self._paused_until = now           # monitors ignore stillness before this time (after reconnect / calibration)

        # presence
        self.presence = "unknown"          # empty | entering | present | unknown
        self.presence_since = now
        self.presence_confidence: float | None = None
        self.range_m: float | None = None
        self.moving = False
        self.motionless_s: float | None = None     # the engine's own "nothing at all moved" counter
        self.confirmed = False             # somebody is really there (see PRESENCE_CONFIRM_*)
        self._confident_in_a_row = 0

        # signs of life (for "No movement")
        self.last_motion_at = now          # gross movement: walking, reaching, a gesture
        self.last_breath_at: float | None = None   # a confidently tracked breathing rate
        self.last_motion_what = "start"

        # breathing (for "No breathing movement"; the same logic as web/js/store.js)
        self.wave = WaveTimeline(60.0)
        self.bpm: float | None = None
        self.breath_state: str | None = None
        self.breath_confidence: float | None = None
        self._breath_logged = False        # "Breathing measured" was written for this visit
        self._b_motion_at = now
        self._b_valid_streak_start = 0.0
        self._b_last_valid_at = 0.0
        self._b_armed = False
        self._b_baseline: float | None = None
        self._b_flat_since = 0.0
        self._b_engine_apnea = False

        # gestures
        self.last_gesture: dict[str, Any] | None = None
        self._pending_push: dict[str, Any] | None = None    # first push of a possible "help" pair
        self._guard_until = 0.0            # ignore movement while our own alert sound plays
        self._practice_until = 0.0         # the tutorial is running: recognise, but press nothing and raise nothing
        self._practice_push: float | None = None

    # ---- settings -----------------------------------------------------------------------------

    def _configure_gate(self) -> None:
        c = self.settings["control"]
        self._gate.min_confidence = float(c["min_confidence"])
        self._gate.cooldown_s = float(c["cooldown_s"])
        self._gate.return_stroke_s = float(c["return_stroke_s"])

    def apply_settings(self, settings: dict[str, Any], now: float) -> list[Effect]:
        """Swap in new settings. Anything that no longer applies is resolved right away."""
        self.settings = settings
        self._configure_gate()
        out: list[Effect] = []
        mon = settings["monitoring"]
        if not (mon["enabled"] and mon["help"]["enabled"]):
            out += self._flush_pending_push(now)
            out += self.alerts.cancel("help", now, "help gesture turned off")
        if not (mon["enabled"] and mon["no_movement"]["enabled"]):
            out += self.alerts.resolve("no_movement", now, "monitoring turned off")
        if not (mon["enabled"] and mon["no_breathing"]["enabled"]):
            out += self.alerts.resolve("no_breathing", now, "monitoring turned off")
        return out

    # ---- helpers -------------------------------------------------------------------------------

    @property
    def practice(self) -> bool:
        return self._practice_until > 0.0

    def _help_on(self) -> bool:
        mon = self.settings["monitoring"]
        return bool(mon["enabled"] and mon["help"]["enabled"])

    def _guarded(self, now: float) -> bool:
        return now < self._guard_until

    def sound_played(self, now: float, seconds: float = 1.5) -> None:
        """Our own alert sound is playing: movement seen during it may be the sound itself, so skip it."""
        self._guard_until = max(self._guard_until, now + seconds)

    def _saw_motion(self, now: float, what: str) -> None:
        self.last_motion_at = now
        self.last_motion_what = what

    def _b_disarm(self) -> None:
        self._b_armed = False
        self._b_valid_streak_start = 0.0
        self._b_flat_since = 0.0
        self._b_baseline = None

    # ---- inputs -------------------------------------------------------------------------------

    def on_link(self, connected: bool, now: float) -> list[Effect]:
        """The connection to the engine opened or closed."""
        if connected == self.connected:
            return []
        self.connected = connected
        self._link_changed_at = now
        out: list[Effect] = []
        if connected:
            self._ever_connected = True
            self._paused_until = now + 1.0
            self._saw_motion(now, "reconnect")          # never count the outage as stillness
            self._b_motion_at = now
            if self._link_logged:
                out.append(_timeline("system", "ok", "link", "Sensor connected", now))
            self._link_logged = False
        return out

    def on_event(self, evt: dict[str, Any], now: float) -> list[Effect]:
        """One normalised engine event (the output of app/adapter.py)."""
        kind = evt.get("kind")
        if kind == "gesture":
            return self._on_gesture(evt, now)
        if kind == "presence":
            return self._on_presence(evt, now)
        if kind == "breathing":
            return self._on_breathing(evt, now)
        if kind == "status":
            return self._on_status(evt, now)
        return []

    # ---- status -----------------------------------------------------------------------------------

    def _on_status(self, s: dict[str, Any], now: float) -> list[Effect]:
        out: list[Effect] = []
        cal = s.get("calibrating") is True
        if cal != self.calibrating:
            self.calibrating = cal
            if cal:
                self._b_disarm()
                out.append(_timeline("system", "info", "calibrate", "Calibrating: keep still for a few seconds", now))
            else:
                self._paused_until = now + 1.0
                self._saw_motion(now, "calibration")
                self._b_motion_at = now
                out.append(_timeline("system", "ok", "check", "Calibration complete", now))
        return out

    # ---- presence ---------------------------------------------------------------------------------

    def _on_presence(self, p: dict[str, Any], now: float) -> list[Effect]:
        state = p.get("state")
        if state not in ("empty", "entering", "present"):
            return []
        out: list[Effect] = []
        prev = self.presence
        self.presence_confidence = p.get("confidence")
        self.range_m = p.get("rangeM")
        self.motionless_s = p.get("motionlessS")
        moving = p.get("moving") is True
        guarded = self._guarded(now)

        if state != prev:
            self.presence = state
            self.presence_since = now
            self._b_motion_at = now                       # same as the web monitor: any change disarms the alarm
            self._b_disarm()
            if state == "empty":
                was_confirmed = self.confirmed
                self.confirmed = False
                self._confident_in_a_row = 0
                self.moving = False
                self.wave.clear()
                self.bpm = None
                self._breath_logged = False
                self.last_breath_at = None
                out += self.alerts.resolve("no_movement", now, "nobody in range any more")
                out += self.alerts.resolve("no_breathing", now, "nobody in range any more")
                if was_confirmed:
                    out.append(_timeline("presence", "info", "empty", "Room is empty", now))
                return out
            if prev in ("empty", "unknown"):              # a new visit starts
                self.confirmed = False
                self._confident_in_a_row = 0
                self.last_breath_at = None
                self._saw_motion(now, "arrival")
            elif not guarded:                             # entering <-> present: they walked or settled
                self._saw_motion(now, "walking")

        if state == "empty":                              # the engine's confidence here is "sure the room is empty"
            self.moving = False
            self._confident_in_a_row = 0
            return out
        self.moving = moving
        if (moving or state == "entering") and not guarded:
            self._saw_motion(now, "moving")

        conf = self.presence_confidence
        if conf is not None and conf >= PRESENCE_CONFIRM_CONFIDENCE:
            self._confident_in_a_row += 1
        else:
            self._confident_in_a_row = 0
        if not self.confirmed and self._confident_in_a_row >= PRESENCE_CONFIRM_EVENTS:
            self.confirmed = True
            out.append(_timeline("presence", "info", "enter", "Someone is here", now,
                                 detail=None if self.range_m is None else f"about {self.range_m:.1f} m from the laptop"))
        return out

    # ---- breathing --------------------------------------------------------------------------------

    def _valid_breath(self, b: dict[str, Any]) -> bool:
        if b.get("bpm") is None or b.get("apnea") is True:
            return False
        if b.get("breathState") and b["breathState"] != "breathing":
            return False
        conf = b.get("confidence")
        return conf is None or conf >= BREATH_MIN_CONFIDENCE

    def _on_breathing(self, b: dict[str, Any], now: float) -> list[Effect]:
        out: list[Effect] = []
        self.breath_state = b.get("breathState")
        self.breath_confidence = b.get("confidence")
        self._b_engine_apnea = b.get("apnea") is True or b.get("breathState") == "apnea"
        if b.get("motionlessS") is not None:
            self.motionless_s = b["motionlessS"]
        if b.get("breathState") == "motion":              # the engine saw the body move
            self._b_motion_at = now
            self._b_disarm()
            if not self._guarded(now):
                self._saw_motion(now, "moving")
        if self._valid_breath(b):
            if not self._b_valid_streak_start or now - self._b_last_valid_at > 3.0:
                self._b_valid_streak_start = now
            self._b_last_valid_at = now
            self.bpm = b["bpm"]
            self.last_breath_at = now
            if not self._breath_logged and self.presence == "present":
                self._breath_logged = True
                out.append(_timeline("breathing", "ok", "breath", f"Breathing measured: about {round(self.bpm)} breaths/min", now,
                                     detail="early estimate, experimental"))
        elif b.get("bpm") is None and now - self._b_last_valid_at > 6.0:
            self.bpm = None
        wave = b.get("waveform")
        if wave and b.get("fs"):
            t_last = now if b.get("tLocal") else b.get("t", now)
            t0 = None if b.get("tLocal") else b.get("t0")
            self.wave.merge(to_millimetres(wave, b.get("amplitude")), b["fs"], t_last, t0)
        return out

    # ---- gestures ---------------------------------------------------------------------------------

    def _on_gesture(self, g: dict[str, Any], now: float) -> list[Effect]:
        label = g.get("label")
        if label in (None, "none"):
            return []
        conf = g.get("confidence")
        c = 1.0 if conf is None else float(conf)
        base: Effect = {"effect": "gesture", "gesture": label, "confidence": conf, "t": now}

        self._b_motion_at = now                           # a hand moved: this is not a breathing pause
        self._b_disarm()
        if self._guarded(now):
            return [dict(base, result="ignored", reason="alert sound playing")]
        self._saw_motion(now, "gesture")

        if label not in S.GESTURES:
            return [dict(base, result="ignored", reason="unsupported gesture")]
        if c < float(self.settings["control"]["min_confidence"]):
            return self._remember([dict(base, result="ignored", reason="low confidence")])
        if self.practice:
            return self._remember(self._practice_gesture(label, base, now))

        if self.alerts.needs_attention("help"):
            out: list[Effect] = []
            if self.alerts.is_pending("help"):
                # Not sent yet. Another push sends it at once; nothing a hand does can cancel it.
                a = self.alerts.get("help")
                min_gap = float(self.settings["monitoring"]["help"]["min_gap_s"])
                if label == "push" and self._help_on() and now - a.get("last_request_at", a["raised_at"]) >= min_gap:
                    out += self.alerts.escalate("help", now)
                    out.append(_timeline("alert", "alert", "help", a["title"], now, detail="sent early by another push", alert_id=a["id"]))
                    return self._remember(out + [dict(base, result="help", reason="confirmed by another push")])
                return self._remember([dict(base, result="paused", reason="help request pending")])
            if label == "push" and self._help_on():
                out += self._help_again(now)
            return self._remember(out + [dict(base, result="paused", reason="help alert is open")])

        if label == "push" and self._help_on():
            return self._remember(self._push_with_help(c, base, now))
        return self._remember(self._flush_pending_push(now) + self._control(label, c, base, now))

    def _remember(self, effects: list[Effect]) -> list[Effect]:
        """Keep the newest gesture decision for the menu bar and the dashboard."""
        for e in effects:
            if e.get("effect") == "gesture":
                self.last_gesture = {k: e.get(k) for k in ("gesture", "confidence", "result", "reason", "action", "t")}
        return effects

    def _control(self, label: str, c: float, base: Effect, now: float) -> list[Effect]:
        """A gesture that should press its key (if gesture control is on, mapped and not filtered out)."""
        if not self.settings["control"]["enabled"]:
            return [dict(base, result="off", reason="gesture control is off")]
        action = S.mapping(self.settings)[label]
        if action == "off":
            return [dict(base, result="unmapped", reason="no action for this gesture")]
        reason = self._judge(label, c, now)
        if reason:
            return [dict(base, result="ignored", reason=reason)]
        return [dict(base, result="act", action=action)]

    def _judge(self, label: str, c: float, now: float) -> str | None:
        """None if this gesture may act, else why not. Adds "same movement" to the key bridge's three filters."""
        repeat_s = float(self.settings["control"]["repeat_s"])
        if self._gate._last_label == label and now - self._gate._last_at < repeat_s:
            return "too soon" if now - self._gate._last_at < self._gate.cooldown_s else "same movement"
        return self._gate.judge(label, c, now)

    def _push_with_help(self, c: float, base: Effect, now: float) -> list[Effect]:
        """A push while the help gesture is on: second of a pair = help, otherwise it starts the window."""
        cfg = self.settings["monitoring"]["help"]
        window, min_gap = float(cfg["window_s"]), float(cfg["min_gap_s"])
        out: list[Effect] = []
        pend = self._pending_push
        if pend is not None:
            gap = now - pend["t"]
            if gap < min_gap:
                return [dict(base, result="ignored", reason="same push")]
            if gap <= window:
                self._pending_push = None
                self._gate._last_label, self._gate._last_at = "push", now     # whatever follows waits out the cooldown
                out.append(dict(base, result="help", reason="second push", gap_s=round(gap, 2)))
                return out + self._raise_help(now, first_push_at=pend["t"])
            out += self._flush_pending_push(now)          # the window had already closed

        held = None
        if not self.settings["control"]["enabled"]:
            first = dict(base, result="armed", reason="gesture control is off")
        else:
            action = S.mapping(self.settings)["push"]
            if action == "off":
                first = dict(base, result="armed", reason="push has no action")
            else:
                reason = self._judge("push", c, now)
                if reason:
                    first = dict(base, result="ignored", reason=reason)
                elif cfg["push_timing"] == "instant":
                    first = dict(base, result="act", action=action)
                else:
                    held = action
                    first = dict(base, result="held", action=action, reason="waiting for a second push")
        self._pending_push = {"t": now, "held": held, "confidence": base["confidence"]}
        first["help_until"] = now + window
        return out + [first]

    def _flush_pending_push(self, now: float) -> list[Effect]:
        """The help window is over (or another gesture arrived): a held single push presses its key now."""
        pend = self._pending_push
        if pend is None:
            return []
        if pend["held"] is None:
            # nothing was held back; keep the window open only while it lasts
            if now - pend["t"] > float(self.settings["monitoring"]["help"]["window_s"]) or not self._help_on():
                self._pending_push = None
            return []
        self._pending_push = None
        return [{"effect": "gesture", "gesture": "push", "confidence": pend["confidence"], "t": now, "result": "act",
                 "action": pend["held"], "delayed_s": round(now - pend["t"], 2)}]

    def _raise_help(self, now: float, first_push_at: float) -> list[Effect]:
        confirm_s = float(self.settings["monitoring"]["help"]["confirm_s"])
        out = self.alerts.raise_("help", "alert", "Help requested", "Two pushes in a row at the laptop", first_push_at, now,
                                 pending_s=confirm_s, last_request_at=now)
        alert_id = self.alerts.get("help")["id"]
        if confirm_s > 0:
            out.append(_timeline("alert", "attention", "help", "Help gesture seen", now,
                                 detail=f"alerting in {fmt_duration(confirm_s)} unless cancelled on screen", alert_id=alert_id))
        else:
            out.append(_timeline("alert", "alert", "help", "Help requested", now, detail="Two pushes in a row at the laptop", alert_id=alert_id))
        return out

    def cancel_help(self, now: float) -> list[Effect]:
        """Somebody pressed Cancel on screen while the request was still pending."""
        return self.alerts.cancel("help", now, "cancelled on screen")

    def _help_again(self, now: float) -> list[Effect]:
        a = self.alerts.get("help")
        min_gap = float(self.settings["monitoring"]["help"]["min_gap_s"])
        if a is None or now - a.get("last_request_at", a["raised_at"]) < max(min_gap, 1.0):
            return []
        out = self.alerts.update("help", now, change="reminder", count=a["count"] + 1, last_request_at=now,
                                 detail=f"Requested {a['count'] + 1} times, still waiting")
        out.append(_timeline("alert", "alert", "help", "Help requested again", now, alert_id=a["id"]))
        return out

    # ---- practice mode (the tutorial) ---------------------------------------------------------------

    def set_practice(self, active: bool, now: float) -> list[Effect]:
        """Tutorial on: gestures are recognised and reported, but press nothing and raise nothing."""
        was = self.practice
        self._practice_until = now + PRACTICE_TIMEOUT_S if active else 0.0
        if active != was:
            self._practice_push = None
            self._pending_push = None                     # a push held from before must not fire mid-lesson
            self._lesson_boundary(now)
        return []

    def _lesson_boundary(self, now: float) -> None:
        """A lesson started or ended: the monitors start counting from here, not from before it."""
        self._saw_motion(now, "practice")
        self._b_motion_at = now
        self._b_disarm()

    def _practice_gesture(self, label: str, base: Effect, now: float) -> list[Effect]:
        res = dict(base, result="practice")
        if label == "push":
            cfg = self.settings["monitoring"]["help"]
            first = self._practice_push
            if first is not None and now - first < float(cfg["min_gap_s"]):
                return [dict(base, result="ignored", reason="same push", practice=True)]
            if first is not None and now - first <= float(cfg["window_s"]):
                self._practice_push = None
                res["double_push"] = True
                res["gap_s"] = round(now - first, 2)
            else:
                self._practice_push = now
                res["help_until"] = now + float(cfg["window_s"])
        return [res]

    # ---- time -------------------------------------------------------------------------------------

    def acknowledge(self, now: float, alert_id: str | None = None) -> list[Effect]:
        return self.alerts.acknowledge(now, alert_id)

    def tick(self, now: float) -> list[Effect]:
        """Call a few times a second: everything that depends on time passing."""
        out: list[Effect] = []
        if self._practice_until and now >= self._practice_until:
            self._practice_until = 0.0
            self._practice_push = None
            self._lesson_boundary(now)

        # A single push whose help window ran out presses its key now.
        pend = self._pending_push
        if pend is not None and now - pend["t"] > float(self.settings["monitoring"]["help"]["window_s"]):
            out += self._flush_pending_push(now)
            self._pending_push = None

        # Engine link: say so once if it stays away.
        grace = LINK_GRACE_S if self._ever_connected else START_GRACE_S
        offline = not self.connected and now - self._link_changed_at > grace
        if offline and not self._link_logged:
            self._link_logged = True
            self._b_disarm()
            out.append(_timeline("system", "attention", "link", "Sensor offline", now))

        out += self._tick_no_movement(now, offline)
        out += self._tick_no_breathing(now, offline)
        out += self.alerts.tick(now, float(self.settings["alerts"]["repeat_s"]))
        return out

    # ---- "No movement" -----------------------------------------------------------------------------

    def still_for(self, now: float) -> float | None:
        """Seconds since the last sign of life, or None when that question does not apply right now."""
        if self.presence != "present" or not self.confirmed or not self.connected or self.calibrating:
            return None
        cfg = self.settings["monitoring"]["no_movement"]
        evidence = self.last_motion_at
        if cfg["breathing_counts"] and self.last_breath_at is not None:
            evidence = max(evidence, self.last_breath_at)
        return max(0.0, now - max(evidence, self._paused_until))

    def _tick_no_movement(self, now: float, offline: bool) -> list[Effect]:
        mon = self.settings["monitoring"]
        cfg = mon["no_movement"]
        if not (mon["enabled"] and cfg["enabled"]) or self.practice:
            return self.alerts.resolve("no_movement", now, "monitoring turned off") if self.alerts.is_open("no_movement") and not self.practice else []
        if offline:
            return self.alerts.resolve("no_movement", now, "sensor offline")
        still = self.still_for(now)
        limit = float(cfg["seconds"])
        if self.alerts.is_open("no_movement"):
            if self.presence == "entering":                   # they got up and are walking
                return self.alerts.resolve("no_movement", now, "movement seen again")
            if still is not None and still < limit:           # a sign of life came back
                seen = "breathing detected again" if cfg["breathing_counts"] and self.last_breath_at is not None \
                    and self.last_breath_at >= self.last_motion_at else "movement seen again"
                return self.alerts.resolve("no_movement", now, seen)
            return []
        if still is None or still < limit:
            return []
        since = now - still
        breathing_now = self.last_breath_at is not None and now - self.last_breath_at < 5.0
        if breathing_now:
            detail = f"Someone is present but has not moved for {fmt_duration(still)}. Breathing is still detected."
        else:
            detail = f"Someone is present but nothing has moved for {fmt_duration(still)}. Breathing is not confirmed."
        out = self.alerts.raise_("no_movement", "attention", "No movement", detail, since, now, threshold_s=limit,
                                 breathing_detected=breathing_now)
        out.append(_timeline("alert", "attention", "still", f"No movement for {fmt_duration(still)}", now,
                             detail="breathing still detected" if breathing_now else "breathing not confirmed",
                             alert_id=self.alerts.get("no_movement")["id"]))
        return out

    # ---- "No breathing movement" (the web monitor's alarm, moved here so it has one home) ------------

    def _tick_no_breathing(self, now: float, offline: bool) -> list[Effect]:
        mon = self.settings["monitoring"]
        cfg = mon["no_breathing"]
        enabled = bool(mon["enabled"] and cfg["enabled"]) and not self.practice

        still = (self.presence == "present" and now - self._b_motion_at >= BREATH_MOTION_HOLD_S
                 and not self.calibrating and not offline and self.connected)
        if not still:
            if self._b_armed:
                self._b_disarm()
        elif (not self._b_armed and self._b_valid_streak_start and now - self._b_valid_streak_start >= BREATH_ARM_S
              and now - self._b_last_valid_at < BREATH_FRESH_S):
            self._b_armed = True

        # Flat-line check: the chest movement collapses well below this person's own normal.
        t_now = self.wave.last_time()
        if self._b_armed and t_now is not None:
            recent = self.wave.rms(t_now - 3.0, t_now)
            if math.isfinite(recent):
                if self._b_baseline is None:
                    self._b_baseline = recent
                if not self._b_flat_since and recent > self._b_baseline * 0.5:
                    self._b_baseline += (recent - self._b_baseline) * 0.05
                if recent < self._b_baseline * 0.3:
                    if not self._b_flat_since:
                        self._b_flat_since = now - 3.0
                elif recent > self._b_baseline * 0.5:
                    self._b_flat_since = 0.0

        # Paused if: the engine says so, the line is flat, or good readings stopped arriving.
        since = 0.0
        if self._b_armed:
            limit = float(cfg["seconds"])
            if self._b_engine_apnea:
                since = min(now, self._b_last_valid_at or now)
            if self._b_flat_since and now - self._b_flat_since >= limit:
                since = min(since, self._b_flat_since) if since else self._b_flat_since
            if self._b_last_valid_at and now - self._b_last_valid_at >= limit and not self._b_flat_since:
                moving = (t_now is not None and self._b_baseline is not None
                          and self.wave.rms(t_now - 3.0, t_now) > self._b_baseline * 2.5)
                if not moving:
                    since = min(since, self._b_last_valid_at) if since else self._b_last_valid_at

        if not enabled:
            return self.alerts.resolve("no_breathing", now, "monitoring turned off") if self.alerts.is_open("no_breathing") and not self.practice else []
        if since and not self.alerts.is_open("no_breathing"):
            paused = fmt_duration(now - since)
            out = self.alerts.raise_("no_breathing", "alert", "No breathing movement",
                                     f"Breathing was steady, then no breathing movement for {paused}. The person is present and still.",
                                     since, now, threshold_s=float(cfg["seconds"]))
            out.append(_timeline("alert", "alert", "breath", f"No breathing movement for {paused}", now,
                                 alert_id=self.alerts.get("no_breathing")["id"]))
            return out
        if not since and self.alerts.is_open("no_breathing"):
            why = "movement seen" if now - self._b_motion_at < BREATH_MOTION_HOLD_S else "breathing resumed"
            return self.alerts.resolve("no_breathing", now, why)
        return []

    # ---- what the rest of the app may read -----------------------------------------------------------

    def snapshot(self, now: float) -> dict[str, Any]:
        """The decisions' current state, for the dashboard and the menu bar."""
        pend = self._pending_push
        help_cfg = self.settings["monitoring"]["help"]
        still = self.still_for(now)
        breathing_shown = self.bpm is not None and now - self._b_last_valid_at < 6.0
        return {
            "presence": {
                "state": self.presence, "since": self.presence_since, "confidence": self.presence_confidence,
                "range_m": self.range_m, "moving": self.moving, "confirmed": self.confirmed,
                "motionless_s": self.motionless_s,
            },
            "breathing": {
                "bpm": round(self.bpm, 1) if breathing_shown else None,
                "state": self.breath_state, "confidence": self.breath_confidence, "armed": self._b_armed,
            },
            "stillness": {
                "still_s": None if still is None else round(still, 1),
                "limit_s": float(self.settings["monitoring"]["no_movement"]["seconds"]),
                "last_movement": self.last_motion_what,
            },
            "help": {
                "pending": pend is not None,
                "since": None if pend is None else pend["t"],
                "until": None if pend is None else pend["t"] + float(help_cfg["window_s"]),
                "held_action": None if pend is None else pend["held"],
            },
            "practice": self.practice,
            "last_gesture": self.last_gesture,
            "alerts": self.alerts.listing(),
        }

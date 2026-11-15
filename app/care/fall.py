"""fall.py: EXPERIMENTAL fall detector. Rules only, no training. Pure: no clock, no I/O.

What it looks for
    A fall, seen by sonar, is one short, strong burst of movement that goes ONE way and does not come
    back, followed by stillness. This file watches the engine's Doppler strip (the `spectrogram` events:
    64 speed bins, 50 rows a second) and raises a "possible fall" when all of these hold:

      1. a burst of movement that lasts 0.4 to 2.5 s        (the fast part of a fall is 0.5-2.5 s)
      2. big: a lot of the speed bins light up              (a body or a fast hand, not a fidget)
      3. with fast movement in it                            (energy beyond +/-100 Hz, about 0.9 m/s)
      4. one way, toward OR away, counted from 1.5 s before  (a hand comes back, a sit-down goes down and
         the burst to 1.2 s after it                          back, a walk past goes toward and then away)
      5. the model did not call it a gesture
      6. no train of smaller movements right after it        (a jump lands, then recovers in steps)
      7. then stillness for 15 s (settable, 5-60 s),         (stillness alone is never enough)
         slow movement included
      8. and the engine does not say the room is empty       (that is someone leaving, not lying there)
    A change in the person's measured distance across the burst adds confidence but is never required.

What 36 minutes of real MacBook recordings taught us (12 sessions, 4 people, nobody fell)
    * A hand gesture 30 cm from the laptop is as big on the strip as a body 1.5 m away (pushes read
      0.5, people walking read 0.3-0.5, fidgeting while sitting still reads 0.1-0.2). "Bigger than a
      gesture" therefore does not exist here; rule 2 only removes the small stuff.
    * The engine's distance zones and its presence flag do not separate a hand from a body on the
      one-microphone MacBook either, so they are not used as gates (rule 8 is the one soft exception).
    * The engine's "moving" flag is on 61% of the time while people sit still, so it cannot define
      stillness. The strip can: its slowest bins (8-23 Hz, 0.07-0.2 m/s) are dark when someone sits
      still and lit when they shift about, so rule 7 watches those as well.
    * What does separate a gesture from a fall is rule 4: real swipes and most pushes go out AND back.
    * How many false alarms the recordings give at each setting is in docs/SONAR_CARE.md. The ones
      that remain are people stepping away from the laptop and then standing still: see below.

Why it lives here and not in the engine
    Everything it needs is already in the engine's event stream at full frame rate, so the engine is
    not touched at all: gestures, presence and breathing cannot be disturbed, it works with an engine
    that is already running, and every threshold can be changed from the Settings page while it runs.
    An engine-side detector would see the same Doppler picture plus the distance zones, which (above)
    do not help on this hardware.

What it gets wrong (say so in the pitch)
    Misses:   * a fall side-on to the laptop, or straight down: little speed toward or away from it;
              * a fall behind furniture or far from the laptop: the echo is too weak (the reach is not measured);
              * a slow slide to the floor: no fast burst;
              * a fall that ends out of the sensor's view while the engine reports the room empty.
    Fires on: * someone stepping quickly away from the laptop (or up to it) and then nothing moving
                in view for the whole stillness time: to this sensor that IS a fall;
              * a bag dropped in front of the laptop that then lies still;
              * a hand pushed toward the laptop and held there, when the model did not recognise it.
    It is a prototype rule, checked on real recordings of ordinary movement and on synthetic falls.
    No real fall has been recorded with it.

How to use it
    det = FallDetector(config)                 # config: see DEFAULTS and for_sensitivity()
    events = det.on_rows(rows, bins_hz, t)     # every spectrogram message; t = time of the newest row
    events += det.on_gesture(t)                # every gesture the engine reports
    det.on_presence(state, range_m, t)         # every presence message
    events += det.tick(now)                    # a few times a second
    Each event is a dict {"stage": "candidate" | "confirmed" | "dismissed" | "moved", ...} (see _event()).
    "moved" comes at most once after a "confirmed": the strip showed movement again (the alert stays up).

What it changes
    Nothing outside its own memory.

Sources for the starting thresholds
    Lian et al., "Fall Detection via Inaudible Acoustic Sensing", IMWUT 2021, doi:10.1145/3478094.
    Erol, Amin et al., SPIE 2018; Erol & Amin, EUSIPCO 2016 (radar fall detection: one-sidedness of the
    Doppler spectrum, extreme Doppler, the range change as a supporting cue, a stillness timer).
"""

from __future__ import annotations

from collections import deque
from typing import Any

# Every threshold. Levels are 0..1 where 1 = "every moving speed bin at the top of the engine's scale".
DEFAULTS: dict[str, float] = {
    "stillness_s": 15.0,       # stillness after the burst before the alert goes out
    "burst_level": 0.25,       # peak level a burst must reach (real fidgeting 0.1-0.2, walking 0.3-0.5, gestures 0.5)
    "fast_level": 0.20,        # ... and this much of it beyond +/- fast_hz
    "one_sided": 0.35,         # 0 = as much toward as away, 1 = all one way
    "min_s": 0.4,              # shortest burst
    "max_s": 2.5,              # longest burst (longer = walking, tidying, not a fall)
    "on_level": 0.08,          # a burst starts when the level rises above this ...
    "off_level": 0.05,         # ... and ends when it stays below this for end_gap_s
    "end_gap_s": 0.30,
    "min_hz": 25.0,            # ignore the slowest bins: breathing, sway, the speaker's own leak
    "fast_hz": 100.0,          # "fast" movement: about 0.9 m/s toward or away
    "look_back_s": 1.5,        # the one-way test also covers this long before the burst ...
    "settle_s": 1.2,           # ... and this long after it; same-way movement in that time is the body settling
    "quiet_level": 0.07,       # stillness = the level stays under this, apart from blips shorter than quiet_break_s ...
    "quiet_break_s": 0.5,
    "big_level": 0.15,         # ... and never reaches this for 0.1 s (a hand coming back is short but big) ...
    "slow_hz": 7.0,            # ... and the slow bins (slow_hz up to min_hz: someone shifting about) stay under
    "slow_level": 0.15,        #     slow_level, apart from blips shorter than slow_break_s
    "slow_break_s": 1.0,
    "recovery_s": 3.0,         # a jump: 2 or more smaller movements within this time after the burst
    "recovery_ratio": 0.30,    # "smaller movement" = at least this fraction of the burst's peak
    "gesture_veto_s": 1.2,     # a gesture reported this close to the burst means it was a hand
    "range_jump_m": 0.35,      # a distance change across the burst that supports "fall" (never required)
    "need_presence": 1.0,      # 1 = drop the alert if the engine says the room is empty when the stillness time is up
}
# How "sensitivity" scales the two size thresholds: high = fires more easily.
SENSITIVITY: dict[str, float] = {"low": 1.3, "medium": 1.0, "high": 0.72}
FULL_SCALE_DB = 36.0           # the strip's 0..1 covers this many dB (config.feat_range_db), starting 3 dB above the floor


def for_sensitivity(config: dict[str, Any], sensitivity: str) -> dict[str, float]:
    """The thresholds actually used: the configured ones with the size thresholds scaled by the sensitivity."""
    out = {**DEFAULTS, **{k: float(v) for k, v in config.items() if k in DEFAULTS}}
    k = SENSITIVITY.get(sensitivity, 1.0)
    out["burst_level"] = min(0.95, out["burst_level"] * k)
    out["fast_level"] = min(0.9, out["fast_level"] * k)
    return out


def _one_sided(pos: float, neg: float) -> tuple[float, str]:
    """(0..1 how one-way the movement energy is, which way)."""
    total = pos + neg
    return (abs(pos - neg) / total if total > 1e-9 else 0.0), ("toward" if pos > neg else "away")


class FallDetector:
    """Feed it Doppler rows, gestures, presence and time; it returns fall events. See the module header."""

    def __init__(self, config: dict[str, float] | None = None) -> None:
        self.cfg: dict[str, float] = {**DEFAULTS, **(config or {})}
        self._seq = 0
        self._frame_s = 0.02
        self._masks: tuple[list[int], list[int], list[int], list[int]] | None = None    # toward, away, fast and slow bins
        self._n_bins = 0
        self._hist: deque[tuple[float, float, float]] = deque()   # (time, toward energy, away energy) of the last seconds
        # the burst being built
        self._burst: dict[str, Any] | None = None
        self._below_since: float | None = None
        # the candidate waiting for stillness
        self._cand: dict[str, Any] | None = None
        self._after: dict[str, Any] | None = None             # the confirmed fall we are still watching for movement
        # context
        self._last_gesture_at = -1e9
        self._range_log: list[tuple[float, float]] = []       # (t, range_m), newest last
        self._presence, self._presence_at = "unknown", -1e9
        self._level, self._slow = 0.0, 0.0
        self.last_burst: dict[str, Any] | None = None         # the newest finished burst and why it did or did not count
        self.last_row_at = 0.0

    def configure(self, config: dict[str, float]) -> None:
        """New thresholds (from the Settings page). Takes effect on the next row."""
        self.cfg = {**DEFAULTS, **config}
        self._masks = None                                     # min_hz / fast_hz may have moved

    # ---- frames -------------------------------------------------------------------------------------

    def _build_masks(self, bins_hz: list[float] | None, n: int) -> None:
        if not bins_hz or len(bins_hz) != n:                   # the engine's layout: -250 Hz .. +242 Hz, 0 Hz in the middle
            step = 500.0 / n
            bins_hz = [-250.0 + i * step for i in range(n)]
        lo, fast = self.cfg["min_hz"], self.cfg["fast_hz"]
        toward = [i for i, f in enumerate(bins_hz) if f >= lo]
        away = [i for i, f in enumerate(bins_hz) if f <= -lo]
        fast_bins = [i for i, f in enumerate(bins_hz) if abs(f) >= fast]
        slow_bins = [i for i, f in enumerate(bins_hz) if self.cfg["slow_hz"] <= abs(f) < lo]
        self._masks, self._n_bins = (toward, away, fast_bins, slow_bins), n

    def on_rows(self, rows: list[list[float]], bins_hz: list[float] | None, t_last: float, frame_s: float | None = None) -> list[dict[str, Any]]:
        """New Doppler rows (oldest first); t_last = the time of the newest one."""
        if not rows:
            return []
        if frame_s and frame_s > 0:
            self._frame_s = float(frame_s)
        n = len(rows[-1])
        if self._masks is None or n != self._n_bins:
            self._build_masks(bins_hz, n)
        toward, away, fast_bins, slow_bins = self._masks
        n_moving = max(1, len(toward) + len(away))
        out: list[dict[str, Any]] = []
        for k, row in enumerate(rows):
            if len(row) != n:
                continue
            t = t_last - (len(rows) - 1 - k) * self._frame_s
            pos = sum(row[i] for i in toward)
            neg = sum(row[i] for i in away)
            level = (pos + neg) / n_moving
            fast = sum(row[i] for i in fast_bins) / max(1, len(fast_bins))
            slow = sum(row[i] for i in slow_bins) / len(slow_bins) if slow_bins else 0.0
            out += self._frame(t, level, pos, neg, fast, slow)
        self.last_row_at = t_last
        return out

    def _frame(self, t: float, level: float, pos: float, neg: float, fast: float, slow: float) -> list[dict[str, Any]]:
        cfg = self.cfg
        self._level, self._slow = level, slow
        out: list[dict[str, Any]] = []
        self._hist.append((t, pos, neg))
        keep = cfg["look_back_s"] + cfg["max_s"] + cfg["settle_s"] + 6.0
        while self._hist and (t - self._hist[0][0] > keep or self._hist[0][0] > t):
            self._hist.popleft()

        # 1. cut the stream into bursts: from the first frame above on_level to the last one above off_level
        b = self._burst
        if b is None:
            if level >= cfg["on_level"]:
                self._burst = {"start": t, "end": t, "peak": level, "pos": pos, "neg": neg, "fast": fast}
                self._below_since = None
        else:
            if level >= cfg["off_level"]:
                b["end"] = t
                self._below_since = None
            elif self._below_since is None:
                self._below_since = t
            b["peak"] = max(b["peak"], level)
            b["fast"] = max(b["fast"], fast)
            b["pos"] += pos
            b["neg"] += neg
            if self._below_since is not None and t - self._below_since >= cfg["end_gap_s"]:
                self._burst, self._below_since = None, None
                out += self._burst_done(b, t)

        # 2. a candidate is waiting
        c = self._cand
        if c is not None and t > c["burst_end"]:
            since = t - c["burst_end"]
            # 2a. once the body has had time to settle: was the whole event one-way, or did it come back?
            if not c["window_ok"] and since >= cfg["settle_s"]:
                lo, hi = c["burst_start"] - cfg["look_back_s"], c["burst_end"] + cfg["settle_s"]
                p = sum(hp for (ht, hp, _) in self._hist if lo <= ht <= hi)
                n = sum(hn for (ht, _, hn) in self._hist if lo <= ht <= hi)
                c["one_sided"], c["direction"] = _one_sided(p, n)
                if c["one_sided"] < cfg["one_sided"]:
                    return out + self._dismiss(t, "it moved back the other way within a second or two (a reach, a gesture or sitting down)")
                c["window_ok"] = True
            # 2b. is the person staying still?
            c["loud_run_s"] = c["loud_run_s"] + self._frame_s if level >= cfg["quiet_level"] else 0.0
            c["big_run_s"] = c["big_run_s"] + self._frame_s if level >= cfg["big_level"] else 0.0
            c["slow_run_s"] = c["slow_run_s"] + self._frame_s if slow >= cfg["slow_level"] else 0.0
            if level >= cfg["quiet_level"]:
                if since <= cfg["recovery_s"] and level >= cfg["recovery_ratio"] * c["peak"] and not c["in_peak"]:
                    c["in_peak"] = True
                    c["recoveries"] += 1
            else:
                c["in_peak"] = False
            if c["recoveries"] >= 2:
                out += self._dismiss(t, "it was followed by a train of smaller movements (a jump or a stumble that recovered)")
            elif since > cfg["settle_s"] and (c["loud_run_s"] >= cfg["quiet_break_s"] or c["big_run_s"] >= 0.1):
                out += self._dismiss(t, "movement was seen again")
            elif since > cfg["settle_s"] and c["slow_run_s"] >= cfg["slow_break_s"]:
                out += self._dismiss(t, "slow movement was seen again (someone shifting about, not lying still)")

        # 3. a fall was reported: say once when movement comes back (the alert itself stays until a person answers it)
        a = self._after
        if a is not None:
            a["loud_run_s"] = a["loud_run_s"] + self._frame_s if level >= cfg["quiet_level"] else 0.0
            a["big_run_s"] = a["big_run_s"] + self._frame_s if level >= cfg["big_level"] else 0.0
            if a["loud_run_s"] >= cfg["quiet_break_s"] or a["big_run_s"] >= 0.1:
                self._after = None
                out.append(self._event("moved", t, a["cand"], stillness_s=t - a["cand"]["burst_end"]))
            elif t - a["since"] > 600.0:
                self._after = None
        return out

    # ---- bursts ---------------------------------------------------------------------------------------

    def _burst_done(self, b: dict[str, Any], now: float) -> list[dict[str, Any]]:
        cfg = self.cfg
        duration = b["end"] - b["start"] + self._frame_s
        one_sided, direction = _one_sided(b["pos"], b["neg"])
        why = None
        if duration < cfg["min_s"]:
            why = f"too short ({duration:.1f} s; needs {cfg['min_s']:.1f} s)"
        elif duration > cfg["max_s"]:
            why = f"too long ({duration:.1f} s; looks like walking or ongoing movement)"
        elif b["peak"] < cfg["burst_level"]:
            why = f"not big enough (level {b['peak']:.2f}, needs {cfg['burst_level']:.2f})"
        elif b["fast"] < cfg["fast_level"]:
            why = f"not fast enough (fast level {b['fast']:.2f}, needs {cfg['fast_level']:.2f})"
        elif one_sided < cfg["one_sided"]:
            why = f"moved both ways (one-way score {one_sided:.2f}, needs {cfg['one_sided']:.2f}: a swipe, sitting down, walking past)"
        elif b["start"] - cfg["gesture_veto_s"] <= self._last_gesture_at <= b["end"] + cfg["gesture_veto_s"]:
            why = "the model recognised it as a hand gesture"
        summary = {"t": b["end"], "duration_s": round(duration, 2), "level": round(b["peak"], 3), "fast_level": round(b["fast"], 3),
                   "one_sided": round(one_sided, 2), "direction": direction, "counted": why is None,
                   "why": why or "counted: big, fast and one-way", "outcome": None if why is None else "ignored"}
        self.last_burst = summary
        if why is not None:
            return []
        out = self._dismiss(now, "a newer movement took over")       # a second big burst while waiting: the newer one is judged
        summary["outcome"] = "waiting for stillness"
        self._cand = {"burst_start": b["start"], "burst_end": b["end"], "peak": b["peak"], "fast": b["fast"], "one_sided": one_sided,
                      "direction": direction, "duration": duration, "loud_run_s": 0.0, "big_run_s": 0.0, "slow_run_s": 0.0, "recoveries": 0, "in_peak": False,
                      "window_ok": False, "range_before": self._range_before(b["start"]), "summary": summary}
        return out + [self._event("candidate", now, self._cand, stillness_s=max(0.0, now - b["end"]))]

    # ---- context ----------------------------------------------------------------------------------------

    def on_gesture(self, t: float) -> list[dict[str, Any]]:
        """The engine reported a hand gesture at time t."""
        self._last_gesture_at = t
        c = self._cand
        if c is not None and t - c["burst_end"] <= self.cfg["gesture_veto_s"]:
            return self._dismiss(t, "the model recognised it as a hand gesture")
        if c is not None:
            return self._dismiss(t, "a hand gesture was seen afterwards")
        return []

    def on_presence(self, state: str, range_m: float | None, t: float) -> list[dict[str, Any]]:
        """A presence message. Only the state and the distance are kept; see the header for why nothing here is a gate."""
        self._presence, self._presence_at = state, t
        if range_m is not None and state != "empty":
            self._range_log.append((t, float(range_m)))
            del self._range_log[:-30]
        return []

    def reset(self, t: float) -> list[dict[str, Any]]:
        """The stream stopped or the engine recalibrated: forget what was in progress."""
        self._burst, self._below_since, self._after = None, None, None
        self._hist.clear()
        return self._dismiss(t, "the sensor was interrupted")

    def _range_before(self, t: float) -> float | None:
        vals = [r for (rt, r) in self._range_log if t - 6.0 <= rt <= t]
        return sorted(vals)[len(vals) // 2] if vals else None

    def _range_after(self, t0: float) -> float | None:
        vals = [r for (rt, r) in self._range_log if rt >= t0 + 2.0]     # the engine's distance is a 2 s median
        return sorted(vals)[len(vals) // 2] if vals else None

    # ---- time ---------------------------------------------------------------------------------------------

    def tick(self, now: float) -> list[dict[str, Any]]:
        """Call a few times a second: confirms a candidate once the stillness time has passed."""
        c = self._cand
        if c is None:
            return []
        still = now - c["burst_end"]
        if self.last_row_at and now - self.last_row_at > 3.0:     # no Doppler rows: we cannot vouch for stillness
            return self._dismiss(now, "the Doppler stream stopped")
        if still < self.cfg["stillness_s"] or not c["window_ok"]:
            return []
        if self.cfg["need_presence"] >= 0.5 and self._presence == "empty" and now - self._presence_at < 5.0:
            return self._dismiss(now, "nobody is in view any more (more likely someone left than fell)")
        self._cand = None
        self._after = {"since": now, "cand": c, "loud_run_s": 0.0, "big_run_s": 0.0}
        c["summary"]["outcome"] = f"possible fall: still for {still:.0f} s"
        return [self._event("confirmed", now, c, stillness_s=still)]

    def _dismiss(self, t: float, why: str) -> list[dict[str, Any]]:
        c = self._cand
        if c is None:
            return []
        self._cand = None
        still = max(0.0, t - c["burst_end"])
        c["summary"]["outcome"] = f"dropped after {still:.1f} s: {why}"
        return [self._event("dismissed", t, c, stillness_s=still, reason=why)]

    # ---- output ---------------------------------------------------------------------------------------------

    def _event(self, stage: str, now: float, c: dict[str, Any], stillness_s: float, reason: str | None = None) -> dict[str, Any]:
        cfg = self.cfg
        after = self._range_after(c["burst_end"])
        before = c["range_before"]
        jump = abs(after - before) if after is not None and before is not None else None
        range_vote = 1.0 if jump is not None and jump >= cfg["range_jump_m"] else 0.0
        size = min(1.0, max(0.0, (c["peak"] - cfg["burst_level"]) / max(cfg["burst_level"], 1e-6)))
        fast = min(1.0, c["fast"] / max(cfg["fast_level"] * 2.0, 1e-6))
        confidence = 0.30 + 0.25 * size + 0.20 * c["one_sided"] + 0.15 * fast + 0.10 * range_vote
        self._seq += 1
        return {
            "stage": stage, "t": now, "seq": self._seq,
            "burst_t": c["burst_end"], "burst_level": round(c["peak"], 3),
            "burst_db": round(FULL_SCALE_DB * c["peak"], 1),      # the same thing as dB on the engine's strip scale
            "fast_level": round(c["fast"], 3), "one_sided": round(c["one_sided"], 2), "direction": c["direction"],
            "duration_s": round(c["duration"], 2), "stillness_s": round(stillness_s, 1), "needs_s": cfg["stillness_s"],
            "range_before_m": None if before is None else round(before, 2), "range_after_m": None if after is None else round(after, 2),
            "confidence": round(min(1.0, confidence), 2), "reason": reason,
        }

    def snapshot(self, now: float) -> dict[str, Any]:
        """For the Settings page's live read-out and the dashboard's "watching" line."""
        c = self._cand
        return {
            "level": round(self._level, 3), "slow_level": round(self._slow, 3),
            "watching": None if c is None else {"since": c["burst_end"], "still_s": round(max(0.0, now - c["burst_end"]), 1),
                                                "needs_s": self.cfg["stillness_s"], "direction": c["direction"]},
            "last_burst": None if self.last_burst is None else dict(self.last_burst),
            "has_rows": bool(self.last_row_at) and now - self.last_row_at < 3.0,
            "thresholds": {k: round(self.cfg[k], 3) for k in ("burst_level", "fast_level", "one_sided", "min_s", "max_s", "stillness_s")},
        }

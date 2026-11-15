"""Engine-event adapter (Python twin of web/js/adapter.js).

What it does: turns one raw WebSocket message from the sonar engine into a list
of normalised events, so Python tools (keys_bridge.py, the tests) never read
raw engine field names directly.

Why: the engine's final schema lives in INTERFACES.md and may still change.
When it does, update the alias tables here and in web/js/adapter.js, then run
`python -m unittest discover -s app/tests` to check both still agree on
app/tests/fixtures/engine_events.json.

Changes nothing on disk; pure functions only.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any

VERSION = "provisional-2"

TYPE_ALIASES = {
    # Sonar Care's own messages (app/care/service.py), on the same stream as the engine's.
    "care_hello": "careHello", "care_state": "careState", "care_alert": "careAlert", "care_action": "careAction",
    "care_timeline": "careTimeline", "care_settings": "careSettings",
    "fall": "fall", "care_fall": "fall", "possible_fall": "fall",
    "gesture": "gesture", "gestures": "gesture",
    "presence": "presence", "occupancy": "presence", "room": "presence",
    "breathing": "breathing", "breath": "breathing", "respiration": "breathing", "resp": "breathing", "vitals": "breathing",
    "status": "status", "health": "status", "signal": "status", "stats": "status",
    "spectrogram": "spectrogram", "doppler": "spectrogram", "spec": "spectrogram",
    "hello": "hello", "info": "hello", "meta": "hello",
}

GESTURE_ALIASES = {
    "swipe_left": "swipe_left", "left": "swipe_left", "swipeleft": "swipe_left", "left_swipe": "swipe_left",
    "swipe_right": "swipe_right", "right": "swipe_right", "swiperight": "swipe_right", "right_swipe": "swipe_right",
    "push": "push", "toward": "push", "towards": "push", "approach": "push",
    "pull": "pull", "wave": "pull", "pull_wave": "pull", "pullwave": "pull", "away": "pull", "retreat": "pull",
    "none": "none", "no_gesture": "none", "nogesture": "none", "idle": "none", "background": "none",
    "noise": "none", "other": "none", "null": "none",
}

PRESENCE_ALIASES = {
    "empty": "empty", "empty_room": "empty", "absent": "empty", "none": "empty", "vacant": "empty",
    "nobody": "empty", "no_person": "empty",
    "present": "present", "person_present": "present", "person_present_still": "present", "present_still": "present",
    "still": "present", "occupied": "present", "sitting": "present", "standing": "present", "stationary": "present",
    "entering": "entering", "person_walking_in": "entering", "walking_in": "entering", "walking": "entering",
    "moving": "entering", "enter": "entering", "entered": "entering", "arrival": "entering", "motion": "entering",
}

BREATH_STATE_ALIASES = {
    "breathing": "breathing", "normal": "breathing", "ok": "breathing", "tracking": "breathing", "locked": "breathing",
    "apnea": "apnea", "apnoea": "apnea", "breath_hold": "apnea", "no_breathing": "apnea", "stopped": "apnea", "flat": "apnea",
    "motion": "motion", "moving": "motion", "unstable": "motion",
    "absent": "absent", "empty": "absent", "no_person": "absent",
    "measuring": "measuring", "acquiring": "measuring", "searching": "measuring", "warming_up": "measuring",
    "calibrating": "measuring", "no_signal": "measuring", "nosignal": "measuring", "weak": "measuring",
}

GESTURES = ("swipe_left", "swipe_right", "push", "pull")


def _key(v: Any) -> str:
    s = "" if v is None else str(v).strip().lower()
    return re.sub(r"[\s\-/]+", "_", s)


def _pick(d: dict, names: tuple[str, ...]) -> Any:
    for n in names:
        if d.get(n) is not None:
            return d[n]
    return None


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, str) and v.strip():
        try:
            f = float(v)
        except ValueError:
            return None
        return f if math.isfinite(f) else None
    return None


def _bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        k = _key(v)
        if k in ("true", "yes", "on", "1", "running", "active", "in_progress"):
            return True
        if k in ("false", "no", "off", "0", "done", "idle", "complete", "finished"):
            return False
    return None


def _conf(v: Any) -> float | None:
    n = _num(v)
    if n is None:
        return None
    if 1 < n <= 100:
        n /= 100
    return max(0.0, min(1.0, n))


def _time(d: dict, now_s: float) -> tuple[float, bool]:
    t = _num(_pick(d, ("t", "ts", "time", "timestamp", "t_s", "unix")))
    if t is None:
        ms = _num(_pick(d, ("t_ms", "ts_ms", "time_ms")))
        if ms is not None:
            t = ms / 1000
    if t is not None and t > 1e12:
        t /= 1000
    if t is None or t < 1e8:
        return now_s, True
    return t, False


def _str_or_none(v: Any) -> str | None:
    return None if v is None else str(v)


def _gesture(m: dict, t: float, local: bool) -> dict:
    raw = _pick(m, ("label", "gesture", "name", "class", "cls", "prediction", "pred", "value"))
    label = GESTURE_ALIASES.get(_key(raw)) or ("none" if raw is None else "unknown")
    return {"kind": "gesture", "label": label, "rawLabel": _str_or_none(raw),
            "confidence": _conf(_pick(m, ("confidence", "conf", "p", "prob", "probability", "score"))),
            "durationS": _num(_pick(m, ("duration_s", "duration"))),
            "peakDb": _num(_pick(m, ("peak_db", "peak"))),
            "source": _str_or_none(_pick(m, ("source", "classifier"))),
            "t": t, "tLocal": local}


def _presence(m: dict, t: float, local: bool) -> dict:
    raw = _pick(m, ("state", "presence", "label", "class", "status", "value"))
    state = PRESENCE_ALIASES.get(_key(raw))
    if state is None and isinstance(raw, bool):
        state = "present" if raw else "empty"
    return {"kind": "presence", "state": state or "unknown", "rawState": _str_or_none(raw),
            "confidence": _conf(_pick(m, ("confidence", "conf", "p", "prob", "probability", "score"))),
            "rangeM": _num(_pick(m, ("range_m", "distance_m", "range", "distance"))),
            "moving": _bool(_pick(m, ("moving", "is_moving"))),
            "motionlessS": _num(_pick(m, ("motionless_s", "still_s", "motionless"))),
            "changed": _bool(_pick(m, ("changed",))),
            "t": t, "tLocal": local}


def _breathing(m: dict, t: float, local: bool) -> dict:
    bpm = _num(_pick(m, ("bpm", "rate_bpm", "breaths_per_min", "breathing_rate", "rr", "resp_rate", "rate", "value")))
    if bpm is not None and not (0 < bpm <= 120):
        bpm = None
    wf = _pick(m, ("waveform", "wave", "samples", "displacement", "signal", "trace", "phase"))
    fs = _num(_pick(m, ("fs", "waveform_fs", "fs_hz", "sample_rate_hz", "sample_rate", "srate")))
    t0 = _num(_pick(m, ("t0", "waveform_t0", "t_start", "start_t")))
    if isinstance(wf, dict):
        fs = fs if fs is not None else _num(_pick(wf, ("fs", "sample_rate", "fs_hz")))
        t0 = t0 if t0 is not None else _num(_pick(wf, ("t0", "t_start")))
        wf = _pick(wf, ("samples", "data", "values", "y"))
    waveform = None
    if isinstance(wf, list) and wf:
        waveform = [x if x is not None else math.nan for x in (_num(v) for v in wf)]
    if t0 is not None and t0 > 1e12:
        t0 /= 1000
    apnea = _bool(_pick(m, ("apnea", "apnoea", "breath_hold", "no_breathing")))
    state = BREATH_STATE_ALIASES.get(_key(_pick(m, ("breath_state", "breathing_state", "state"))))
    if apnea is True:
        state = "apnea"
    return {"kind": "breathing", "bpm": bpm,
            "confidence": _conf(_pick(m, ("confidence", "conf", "quality", "p", "score"))),
            "waveform": waveform, "fs": fs if fs and fs > 0 else None, "t0": t0,
            "apnea": apnea, "breathState": state,
            "amplitude": _num(_pick(m, ("amplitude", "amplitude_mm", "rms", "displacement_mm"))),
            "rangeM": _num(_pick(m, ("range_m", "distance_m", "range", "distance"))),
            "windowS": _num(_pick(m, ("window_s", "window", "window_sec"))),
            "held": _bool(_pick(m, ("held",))),
            "stoppedS": _num(_pick(m, ("stopped_s",))),
            "motionlessS": _num(_pick(m, ("motionless_s", "still_s"))),
            "t": t, "tLocal": local}


def _scalar(v: Any) -> tuple[float | None, list[float] | None]:
    n = _num(v)
    if n is not None:
        return n, None
    vals: list[float] = []
    if isinstance(v, list):
        vals = [x for x in (_num(i) for i in v) if x is not None]
    elif isinstance(v, dict):
        vals = [x for x in (_num(i) for i in v.values()) if x is not None]
    if not vals:
        return None, None
    return sorted(vals)[len(vals) // 2], vals


def _last_decision(v: Any) -> dict | None:
    """The classifier's verdict on the latest movement burst, including rejected ones ("none", "blocked:weak")."""
    if not isinstance(v, dict) or v.get("label") is None:
        return None
    return {"label": str(v["label"]), "confidence": _conf(v.get("confidence")), "peakDb": _num(v.get("peak_db"))}


def _status(m: dict, t: float, local: bool) -> dict:
    snr, snr_list = _scalar(_pick(m, ("snr_db", "snr", "snrDb", "snr_dB")))
    car, car_list = _scalar(_pick(m, ("tones_hz", "carriers_hz", "carrier_hz", "carrier", "f0", "f0_hz", "tone_hz", "freq_hz")))
    band = _pick(m, ("band_hz", "band", "chirp_band_hz", "chirp_hz", "sweep_hz"))
    band_hz = None
    if isinstance(band, list) and len(band) >= 2 and _num(band[0]) is not None and _num(band[1]) is not None:
        lo, hi = _num(band[0]), _num(band[1])
        band_hz = [min(lo, hi), max(lo, hi)]
    else:
        flo = _num(_pick(m, ("f_lo", "f_low", "f_min", "fmin", "f_start")))
        fhi = _num(_pick(m, ("f_hi", "f_high", "f_max", "fmax", "f_end")))
        if flo is not None and fhi is not None:
            band_hz = [min(flo, fhi), max(flo, fhi)]
    cal = _pick(m, ("calibrating", "is_calibrating", "calibration"))
    return {"kind": "status", "snrDb": snr, "snrList": snr_list, "carrierHz": car, "carriersHz": car_list,
            "bandHz": band_hz, "fps": _num(_pick(m, ("fps", "frame_rate", "frames_per_s", "hop_hz", "update_hz"))),
            "calibrating": None if cal is None else _bool(cal),
            "source": _str_or_none(_pick(m, ("source", "mode", "origin"))),
            "device": _str_or_none(_pick(m, ("device", "mic", "input_device"))),
            "latencyMs": _num(_pick(m, ("latency_ms", "lag_ms", "pipeline_latency_ms"))),
            "locked": _bool(_pick(m, ("locked", "lock"))),
            "clipped": _bool(_pick(m, ("clipped", "clipping"))),
            "dropped": _num(_pick(m, ("dropped", "dropouts"))),
            "activityDb": _num(_pick(m, ("activity_db", "activity"))),
            "noiseDb": _num(_pick(m, ("noise_db",))),
            "classifier": _str_or_none(_pick(m, ("classifier",))),
            "lastDecision": _last_decision(_pick(m, ("last_decision",))),
            "t": t, "tLocal": local}


def _spectrogram(m: dict, t: float, local: bool) -> dict:
    rows = _pick(m, ("rows", "frames", "columns", "data", "spec", "S", "values"))
    if rows is None:
        one = _pick(m, ("row", "frame", "column"))
        rows = [one] if one is not None else None
    if not isinstance(rows, list) or not rows:
        return {"kind": "unknown", "raw": m, "reason": "spectrogram without rows"}
    if not isinstance(rows[0], list):
        rows = [rows]
    out = [[x if x is not None else math.nan for x in (_num(v) for v in r)] for r in rows if isinstance(r, list) and r]
    finite = [v for r in out for v in r if math.isfinite(v)]
    if not out or not finite:
        return {"kind": "unknown", "raw": m, "reason": "spectrogram rows empty"}
    lo, hi = min(finite), max(finite)
    autoscaled = lo < -0.01 or hi > 1.5
    if autoscaled:
        a, b = (max(lo, hi - 80), hi) if (hi <= 0.5 and lo < -1) else (lo, hi)
        span = (b - a) or 1.0
        out = [[max(0.0, min(1.0, (v - a) / span)) if math.isfinite(v) else 0.0 for v in r] for r in out]
    else:
        out = [[max(0.0, min(1.0, v)) if math.isfinite(v) else 0.0 for v in r] for r in out]
    axis = _pick(m, ("axis", "label", "units"))
    bins = _pick(m, ("bins_hz", "freqs_hz", "doppler_hz", "f_hz", "axis_hz"))
    return {"kind": "spectrogram", "rows": out,
            "binsHz": [(_num(v) if _num(v) is not None else math.nan) for v in bins] if isinstance(bins, list) else None,
            "axisLabel": _str_or_none(axis), "autoscaled": autoscaled,
            "frameS": _num(_pick(m, ("frame_s", "hop_s", "dt"))), "t": t, "tLocal": local}


def _hello(m: dict, t: float, local: bool) -> dict:
    labels = _pick(m, ("labels", "gestures", "classes"))
    return {"kind": "hello", "source": _str_or_none(_pick(m, ("source", "mode", "origin", "engine"))),
            "version": _str_or_none(_pick(m, ("version", "engine_version"))),
            "schema": _str_or_none(_pick(m, ("schema", "schema_version", "protocol"))),
            "labels": [str(x) for x in labels] if isinstance(labels, list) else None,
            "classifier": _str_or_none(_pick(m, ("classifier",))),
            "t": t, "tLocal": local}


# ---- Sonar Care's own messages -------------------------------------------------------------------
# Field names on the wire are snake_case (see docs/SONAR_CARE.md); the app uses camelCase like the rest.

def _d(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def _care_alert(a: Any) -> dict | None:
    if not isinstance(a, dict):
        return None
    return {"id": _str_or_none(a.get("id")), "kind": _str_or_none(a.get("kind")), "level": _str_or_none(a.get("level")),
            "state": _str_or_none(a.get("state")), "title": _str_or_none(a.get("title")), "detail": _str_or_none(a.get("detail")),
            "since": _num(a.get("since")), "raisedAt": _num(a.get("raised_at")), "activeAt": _num(a.get("active_at")),
            "escalateAt": _num(a.get("escalate_at")), "ackedAt": _num(a.get("acked_at")), "resolvedAt": _num(a.get("resolved_at")),
            "resolution": _str_or_none(a.get("resolution")), "ackS": _num(a.get("ack_s")), "count": _num(a.get("count")),
            "thresholdS": _num(a.get("threshold_s")), "breathingDetected": _bool(a.get("breathing_detected")),
            "experimental": _bool(a.get("experimental")) is True, "direction": _str_or_none(a.get("direction")),
            "confidence": _conf(a.get("confidence")), "movedAt": _num(a.get("moved_at"))}


def _care_entry(e: Any) -> dict | None:
    if not isinstance(e, dict):
        return None
    return {"id": _num(e.get("id")), "t": _num(e.get("t")), "kind": _str_or_none(e.get("kind")), "level": _str_or_none(e.get("level")),
            "icon": _str_or_none(e.get("icon")), "text": _str_or_none(e.get("text")), "detail": _str_or_none(e.get("detail")),
            "alertId": _str_or_none(e.get("alert_id"))}


def _care_gesture(g: Any) -> dict | None:
    if not isinstance(g, dict) or g.get("gesture") is None:
        return None
    return {"gesture": str(g["gesture"]), "confidence": _conf(g.get("confidence")), "result": _str_or_none(g.get("result")),
            "reason": _str_or_none(g.get("reason")), "action": _str_or_none(g.get("action")),
            "actionLabel": _str_or_none(g.get("action_label")), "t": _num(g.get("t"))}


def _care_hello(m: dict, t: float, local: bool) -> dict:
    gestures = m.get("gestures")
    timeline = m.get("timeline")
    return {"kind": "careHello", "version": _str_or_none(m.get("version")), "dryRun": _bool(m.get("dry_run")),
            "mode": _str_or_none(m.get("mode")), "gestures": [str(g) for g in gestures] if isinstance(gestures, list) else [],
            "timeline": [e for e in (_care_entry(x) for x in timeline) if e] if isinstance(timeline, list) else [],
            "t": t, "tLocal": local}


def _care_settings(m: dict, t: float, local: bool) -> dict:
    # The settings document belongs to the local API (GET/POST /api/settings) and is passed through as it is.
    return {"kind": "careSettings", "rev": _num(m.get("rev")), "settings": _d(m.get("settings")), "t": t, "tLocal": local}


def _care_action(m: dict, t: float, local: bool) -> dict:
    return {"kind": "careAction", "gesture": _str_or_none(m.get("gesture")), "confidence": _conf(m.get("confidence")),
            "result": _str_or_none(m.get("result")), "reason": _str_or_none(m.get("reason")), "action": _str_or_none(m.get("action")),
            "actionLabel": _str_or_none(m.get("action_label")), "detail": _str_or_none(m.get("detail")),
            "delayedS": _num(m.get("delayed_s")), "helpUntil": _num(m.get("help_until")), "doublePush": _bool(m.get("double_push")),
            "gapS": _num(m.get("gap_s")), "t": t, "tLocal": local}


def _fall_burst(b: Any) -> dict | None:
    if not isinstance(b, dict):
        return None
    return {"t": _num(b.get("t")), "durationS": _num(b.get("duration_s")), "level": _num(b.get("level")),
            "fastLevel": _num(b.get("fast_level")), "oneSided": _num(b.get("one_sided")), "direction": _str_or_none(b.get("direction")),
            "counted": _bool(b.get("counted")) is True, "why": _str_or_none(b.get("why")), "outcome": _str_or_none(b.get("outcome"))}


def _fall_state(f: Any) -> dict:
    """The experimental fall alert's live read-out inside care_state (app/care/fall.py snapshot())."""
    f = _d(f)
    w, th = f.get("watching"), _d(f.get("thresholds"))
    return {"enabled": _bool(f.get("enabled")) is True, "sensitivity": _str_or_none(f.get("sensitivity")),
            "level": _num(f.get("level")), "slowLevel": _num(f.get("slow_level")), "hasRows": _bool(f.get("has_rows")) is True,
            "error": _str_or_none(f.get("error")),
            "watching": {"since": _num(w.get("since")), "stillS": _num(w.get("still_s")), "needsS": _num(w.get("needs_s")),
                         "direction": _str_or_none(w.get("direction"))} if isinstance(w, dict) else None,
            "lastBurst": _fall_burst(f.get("last_burst")),
            "thresholds": {"burstLevel": _num(th.get("burst_level")), "fastLevel": _num(th.get("fast_level")),
                           "oneSided": _num(th.get("one_sided")), "minS": _num(th.get("min_s")), "maxS": _num(th.get("max_s")),
                           "stillnessS": _num(th.get("stillness_s"))}}


def _fall(m: dict, t: float, local: bool) -> dict:
    """The experimental "possible fall" event. Today Sonar Care's own detector sends it on the relay, not the engine."""
    stage = _str_or_none(_pick(m, ("stage", "state", "phase")))
    stage = stage.lower() if stage else None
    if stage not in ("candidate", "confirmed", "dismissed", "moved"):
        return {"kind": "unknown", "raw": m, "reason": "fall event without a known stage"}
    direction = _str_or_none(m.get("direction"))
    return {"kind": "fall", "stage": stage, "seq": _num(m.get("seq")), "burstT": _num(m.get("burst_t")),
            "burstDb": _num(m.get("burst_db")), "burstLevel": _num(m.get("burst_level")), "fastLevel": _num(m.get("fast_level")),
            "oneSided": _num(m.get("one_sided")), "direction": direction if direction in ("toward", "away") else None,
            "durationS": _num(m.get("duration_s")), "stillnessS": _num(m.get("stillness_s")), "needsS": _num(m.get("needs_s")),
            "rangeBeforeM": _num(m.get("range_before_m")), "rangeAfterM": _num(m.get("range_after_m")),
            "confidence": _conf(m.get("confidence")), "reason": _str_or_none(m.get("reason")),
            "experimental": True, "t": t, "tLocal": local}


def _care_alert_event(m: dict, t: float, local: bool) -> dict:
    return {"kind": "careAlert", "change": _str_or_none(m.get("change")), "alert": _care_alert(m.get("alert")), "t": t, "tLocal": local}


def _care_timeline(m: dict, t: float, local: bool) -> dict:
    return {"kind": "careTimeline", "entry": _care_entry(m.get("entry")), "t": t, "tLocal": local}


def _care_state(m: dict, t: float, local: bool) -> dict:
    eng, link, sonar, pres, br = _d(m.get("engine")), _d(m.get("link")), _d(m.get("sonar")), _d(m.get("presence")), _d(m.get("breathing"))
    still, hlp, ctl, mon = _d(m.get("stillness")), _d(m.get("help")), _d(m.get("control")), _d(m.get("monitoring"))
    keys, mh, mm, mb = _d(ctl.get("keys")), _d(mon.get("help")), _d(mon.get("no_movement")), _d(mon.get("no_breathing"))
    mf = _d(mon.get("fall"))
    mapping = {str(g): {"action": _str_or_none(_d(v).get("action")), "label": _str_or_none(_d(v).get("label"))}
               for g, v in _d(ctl.get("mapping")).items()}
    alerts = m.get("alerts")
    return {
        "kind": "careState", "version": _str_or_none(m.get("version")), "starting": _bool(m.get("starting")) is True,
        "dryRun": _bool(m.get("dry_run")), "quiet": _bool(m.get("quiet")), "mode": _str_or_none(m.get("mode")),
        "engine": {"managed": _bool(eng.get("managed")), "state": _str_or_none(eng.get("state")), "mode": _str_or_none(eng.get("mode")),
                   "port": _num(eng.get("port")), "url": _str_or_none(eng.get("url")), "pid": _num(eng.get("pid")),
                   "model": _str_or_none(eng.get("model")), "restarts": _num(eng.get("restarts")),
                   "lastError": _str_or_none(eng.get("last_error")), "uptimeS": _num(eng.get("uptime_s")),
                   "retryInS": _num(eng.get("retry_in_s"))},
        "link": {"state": _str_or_none(link.get("state")), "health": _str_or_none(link.get("health")), "ageS": _num(link.get("age_s"))},
        "sonar": {"snrDb": _num(sonar.get("snr_db")), "signal": _str_or_none(sonar.get("signal")),
                  "calibrating": _bool(sonar.get("calibrating")) is True, "locked": _bool(sonar.get("locked")),
                  "source": _str_or_none(sonar.get("source")), "noiseDb": _num(sonar.get("noise_db")), "fps": _num(sonar.get("fps")),
                  "clipped": _bool(sonar.get("clipped")), "device": _str_or_none(sonar.get("device"))},
        "presence": {"state": _str_or_none(pres.get("state")), "since": _num(pres.get("since")), "confidence": _conf(pres.get("confidence")),
                     "rangeM": _num(pres.get("range_m")), "moving": _bool(pres.get("moving")) is True,
                     "confirmed": _bool(pres.get("confirmed")) is True, "motionlessS": _num(pres.get("motionless_s"))},
        "breathing": {"bpm": _num(br.get("bpm")), "state": _str_or_none(br.get("state")), "confidence": _conf(br.get("confidence")),
                      "armed": _bool(br.get("armed")) is True},
        "stillness": {"stillS": _num(still.get("still_s")), "limitS": _num(still.get("limit_s")),
                      "lastMovement": _str_or_none(still.get("last_movement"))},
        "help": {"pending": _bool(hlp.get("pending")) is True, "since": _num(hlp.get("since")), "until": _num(hlp.get("until")),
                 "heldAction": _str_or_none(hlp.get("held_action"))},
        "practice": _bool(m.get("practice")) is True,
        "fall": _fall_state(m.get("fall")),
        "lastGesture": _care_gesture(m.get("last_gesture")),
        "alerts": [a for a in (_care_alert(x) for x in alerts) if a] if isinstance(alerts, list) else [],
        "control": {"enabled": _bool(ctl.get("enabled")) is True, "preset": _str_or_none(ctl.get("preset")),
                    "presetLabel": _str_or_none(ctl.get("preset_label")), "mapping": mapping,
                    "keys": {"mode": _str_or_none(keys.get("mode")), "trusted": _bool(keys.get("trusted")),
                             "backend": _str_or_none(keys.get("backend")), "canPress": _bool(keys.get("can_press")) is True,
                             "media": _str_or_none(keys.get("media")), "note": _str_or_none(keys.get("note")),
                             "settingsPath": _str_or_none(keys.get("settings_path"))}},
        "monitoring": {"enabled": _bool(mon.get("enabled")) is True,
                       "help": {"enabled": _bool(mh.get("enabled")) is True, "windowS": _num(mh.get("window_s")),
                                "minGapS": _num(mh.get("min_gap_s")), "confirmS": _num(mh.get("confirm_s")),
                                "pushTiming": _str_or_none(mh.get("push_timing"))},
                       "noMovement": {"enabled": _bool(mm.get("enabled")) is True, "seconds": _num(mm.get("seconds")),
                                      "breathingCounts": _bool(mm.get("breathing_counts")) is True},
                       "noBreathing": {"enabled": _bool(mb.get("enabled")) is True, "seconds": _num(mb.get("seconds"))},
                       "fall": {"enabled": _bool(mf.get("enabled")) is True, "sensitivity": _str_or_none(mf.get("sensitivity")),
                                "stillnessS": _num(mf.get("stillness_s")), "burstLevel": _num(mf.get("burst_level")),
                                "fastLevel": _num(mf.get("fast_level")), "oneSided": _num(mf.get("one_sided")),
                                "minS": _num(mf.get("min_s")), "maxS": _num(mf.get("max_s"))}},
        "room": _str_or_none(m.get("room")), "onboarded": _bool(m.get("onboarded")) is True,
        "settingsRev": _num(m.get("settings_rev")), "pages": _num(m.get("pages")),
        "t": t, "tLocal": local,
    }


_PARSERS = {"gesture": _gesture, "presence": _presence, "breathing": _breathing,
            "status": _status, "spectrogram": _spectrogram, "hello": _hello, "fall": _fall,
            "careHello": _care_hello, "careState": _care_state, "careAlert": _care_alert_event, "careAction": _care_action,
            "careTimeline": _care_timeline, "careSettings": _care_settings}


def _adapt_obj(m: Any, now_s: float, out: list[dict], depth: int) -> None:
    if depth > 3:
        out.append({"kind": "unknown", "raw": m, "reason": "nested too deep"})
        return
    if isinstance(m, list):
        for item in m:
            _adapt_obj(item, now_s, out, depth + 1)
        return
    if not isinstance(m, dict):
        out.append({"kind": "unknown", "raw": m, "reason": "not an object"})
        return
    if isinstance(m.get("events"), list):
        _adapt_obj(m["events"], now_s, out, depth + 1)
        return

    type_raw = _pick(m, ("type", "event", "kind", "topic", "msg_type"))
    body = m
    inner = _pick(m, ("data", "payload", "body"))
    if type_raw is not None and isinstance(inner, dict) and not _key(type_raw).startswith("care_"):
        body = dict(inner)
        for k, v in m.items():
            if k not in body and k not in ("data", "payload", "body"):
                body[k] = v

    if type_raw is None:
        found = False
        for name, sub in m.items():
            kind = TYPE_ALIASES.get(_key(name))
            if kind and isinstance(sub, dict):
                sub = dict(sub)
                if sub.get("t") is None and m.get("t") is not None:
                    sub["t"] = m["t"]
                sub["type"] = kind
                _adapt_obj(sub, now_s, out, depth + 1)
                found = True
        if not found:
            out.append({"kind": "unknown", "raw": m, "reason": "no type field"})
        return

    kind = TYPE_ALIASES.get(_key(type_raw))
    if kind is None:
        out.append({"kind": "unknown", "raw": m, "reason": f'unknown type "{type_raw}"'})
        return
    t, local = _time(body, now_s)
    out.append(_PARSERS[kind](body, t, local))


def adapt(raw: str | bytes | dict | list, now_s: float | None = None) -> list[dict]:
    """Return the normalised events contained in one raw engine message."""
    if now_s is None:
        now_s = time.time()
    msg: Any = raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return [{"kind": "unknown", "raw": raw, "reason": "invalid JSON"}]
    out: list[dict] = []
    _adapt_obj(msg, now_s, out, 0)
    return out

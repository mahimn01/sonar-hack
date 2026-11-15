"""settings.py: what the user chose in Sonar Care, saved as one JSON file.

What it does
    Holds every user-facing choice: which key each gesture presses, whether gesture control and
    room monitoring are on, the alert thresholds, and whether the first-run setup was finished.
    It loads the file, fills in anything missing with a safe default, rejects values that make no
    sense (a negative timeout, an unknown key name) and writes changes back atomically.

Why
    The menu bar, the dashboard and the settings page all change the same choices. One validated
    file means a bad value from any of them can never reach the part that presses keys.

Where the file lives
    macOS:  ~/Library/Application Support/SonarCare/settings.json
    else:   ~/.sonar_care/settings.json
    Set SONAR_CARE_HOME to use another folder (the tests do, so they never touch the real file).

What it changes
    Only that one JSON file (and it creates the folder). Reading never writes.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

# The three gestures the trained model knows. There is no "pull" and no "wave" class.
GESTURES: tuple[str, ...] = ("swipe_left", "swipe_right", "push")

# Every action a gesture can be mapped to: id -> (label shown in the UI, group).
# "off" means the gesture does nothing. Key names follow app/keys_bridge.py where they overlap.
ACTIONS: dict[str, tuple[str, str]] = {
    "off": ("Off", "none"),
    "right": ("Right arrow", "keys"),
    "left": ("Left arrow", "keys"),
    "up": ("Up arrow", "keys"),
    "down": ("Down arrow", "keys"),
    "space": ("Space", "keys"),
    "return": ("Return", "keys"),
    "escape": ("Escape", "keys"),
    "tab": ("Tab", "keys"),
    "page_down": ("Page down", "keys"),
    "page_up": ("Page up", "keys"),
    "home": ("Home", "keys"),
    "end": ("End", "keys"),
    "b": ("B (blank the slide)", "keys"),
    "f": ("F (full screen)", "keys"),
    "k": ("K (play / pause in YouTube)", "keys"),
    "media_play_pause": ("Play / pause", "media"),
    "media_next": ("Next track", "media"),
    "media_previous": ("Previous track", "media"),
}
# No volume or mute on purpose: the sonar plays through the same speakers at a fixed volume, and changing it
# would change the sensor's own signal.

# Ready-made mappings. "custom" uses control.custom instead.
PRESETS: dict[str, dict[str, Any]] = {
    "slides": {
        "label": "Slides / PDF",
        "hint": "Keynote, PowerPoint, Google Slides, Preview",
        "map": {"swipe_right": "right", "swipe_left": "left", "push": "space"},
    },
    "media": {
        "label": "Media",
        "hint": "Music, Spotify, podcasts, video players",
        "map": {"swipe_right": "media_next", "swipe_left": "media_previous", "push": "media_play_pause"},
    },
    "scroll": {
        "label": "Scroll",
        "hint": "Web pages, documents, recipes",
        "map": {"swipe_right": "page_down", "swipe_left": "page_up", "push": "off"},
    },
    "custom": {
        "label": "Custom",
        "hint": "Pick a key for each gesture",
        "map": {},
    },
}

# Quick thresholds for the "no movement" alert. "demo" is short on purpose and does not let detected
# breathing cancel the alert, so it can be shown on stage without depending on the experimental breathing reading.
NO_MOVEMENT_PRESETS: dict[str, dict[str, Any]] = {
    "demo": {"label": "Demo", "seconds": 15, "breathing_counts": False},
    "standard": {"label": "Standard", "seconds": 60, "breathing_counts": True},
    "relaxed": {"label": "Relaxed", "seconds": 300, "breathing_counts": True},
}

DEFAULTS: dict[str, Any] = {
    "version": SCHEMA_VERSION,
    "onboarded": False,              # the first-run setup was completed (or skipped)
    "room": "Living room",           # label shown on the dashboard
    "control": {
        "enabled": True,             # gestures press keys
        "preset": "slides",
        "custom": {"swipe_right": "right", "swipe_left": "left", "push": "space"},
        "min_confidence": 0.6,       # ignore gestures the model is less sure about than this (0..1)
        "cooldown_s": 0.7,           # ignore any gesture this soon after one that acted
        "return_stroke_s": 1.5,      # ignore the opposite swipe for this long (the hand coming back)
        "repeat_s": 1.3,             # the same gesture again this soon is the same movement, reported twice
    },
    "monitoring": {
        "enabled": True,             # caregiver alerts and the help gesture
        "help": {
            "enabled": True,
            "window_s": 2.5,         # two pushes this close together = "help requested"
            "min_gap_s": 1.0,        # a second push event sooner than this is the same physical push (see below)
            "confirm_s": 3.0,        # seconds to cancel on screen before the alert goes out (0 = at once)
            "push_timing": "wait",   # "wait": a single push acts only after the window passed (never both);
                                     # "instant": a single push acts at once, a double push then also raises help
        },
        "no_movement": {
            "enabled": True,
            "seconds": 60,           # someone is present and nothing has moved for this long
            "breathing_counts": True,  # a confidently tracked breathing rate counts as a sign of life
        },
        "no_breathing": {
            "enabled": True,
            "seconds": 10,           # breathing movement stopped for this long (the clinical length of an apnea)
        },
        "fall": {                    # EXPERIMENTAL, rules only (app/care/fall.py). Off until someone switches it on.
            "enabled": False,
            "sensitivity": "medium", # low | medium | high: scales the two size thresholds below
            "stillness_s": 15,       # stillness after the sudden movement before the alert goes out
            "burst_level": 0.25,     # how big the movement must be (0..1). Real recordings: fidgeting 0.1-0.2,
                                     # walking 0.3-0.5, a hand gesture at the laptop 0.5
            "fast_level": 0.20,      # how much of it must be faster than about 0.9 m/s
            "one_sided": 0.35,       # 0 = as much toward as away, 1 = all one way
            "min_s": 0.4,            # shortest and ...
            "max_s": 2.5,            # ... longest sudden movement that counts
        },
    },
    "alerts": {
        "notification": True,        # macOS notification banner
        "sound": True,               # a system sound through the speakers
        "speak": True,               # then say the alert's title aloud ("Help requested")
        "volume": 0.7,               # 0..1
        "repeat_s": 20,              # remind again while a help request is not acknowledged (0 = never)
    },
    "ui": {"theme": "auto"},         # auto | light | dark
}

# Why these numbers (measured on 13 recorded sessions, 4 people, with the calm50 model on 2026-11-15):
#   * One physical push is reported as TWO push events in about 4 of 10 cases, because the model has no "pull"
#     class and the hand coming back also looks like a push. The gaps were 0.40-0.98 s in 27 of 33 pairs, so
#     min_gap_s = 1.0 and repeat_s = 1.3 treat those as one movement.
#   * With min_gap_s = 1.0 and window_s = 2.5, about 1 single push in 12 still looks like a double push (the hand
#     was pulled back after a pause). That is why the help alert has a short on-screen cancel time (confirm_s).
#   * The opposite swipe never arrived inside 1.2 s (the engine blocks it) but did at 1.24 s: return_stroke_s = 1.5.

# (low, high) limits for every number a user can set.
_LIMITS: dict[str, tuple[float, float]] = {
    "control.min_confidence": (0.3, 0.99),
    "control.cooldown_s": (0.0, 5.0),
    "control.return_stroke_s": (0.0, 5.0),
    "control.repeat_s": (0.0, 5.0),
    "monitoring.help.window_s": (1.5, 6.0),
    "monitoring.help.min_gap_s": (0.2, 1.5),
    "monitoring.help.confirm_s": (0.0, 10.0),
    "monitoring.no_movement.seconds": (5, 3600),
    "monitoring.no_breathing.seconds": (5, 120),
    "monitoring.fall.stillness_s": (5, 60),
    "monitoring.fall.burst_level": (0.05, 0.95),
    "monitoring.fall.fast_level": (0.0, 0.9),
    "monitoring.fall.one_sided": (0.0, 0.95),
    "monitoring.fall.min_s": (0.2, 2.0),
    "monitoring.fall.max_s": (1.0, 8.0),
    "alerts.volume": (0.0, 1.0),
    "alerts.repeat_s": (0, 600),
}
_CHOICES: dict[str, tuple[str, ...]] = {
    "control.preset": tuple(PRESETS),
    "monitoring.help.push_timing": ("wait", "instant"),
    "monitoring.fall.sensitivity": ("low", "medium", "high"),
    "ui.theme": ("auto", "light", "dark"),
}


class SettingsError(ValueError):
    """A setting was rejected. The message says which one and why, in words fit for the UI."""


def home_dir() -> Path:
    """Folder that holds settings.json and the event log."""
    override = os.environ.get("SONAR_CARE_HOME")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SonarCare"
    return Path.home() / ".sonar_care"


def log_dir() -> Path:
    """Folder for the engine log, the service log and the event history."""
    if os.environ.get("SONAR_CARE_LOGS"):
        return Path(os.environ["SONAR_CARE_LOGS"]).expanduser()
    if os.environ.get("SONAR_CARE_HOME"):
        return Path(os.environ["SONAR_CARE_HOME"]).expanduser() / "logs"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "SonarCare"
    return Path.home() / ".sonar_care" / "logs"


def settings_path() -> Path:
    return home_dir() / "settings.json"


def _walk(tree: dict[str, Any], prefix: str = ""):
    """Yield (dotted path, value) for every leaf in a nested dict."""
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _walk(value, path + ".")
        else:
            yield path, value


def _check_leaf(path: str, value: Any, default: Any) -> Any:
    """Return `value` cleaned up for `path`, or raise SettingsError."""
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise SettingsError(f"{path} must be true or false")
        return value
    if isinstance(default, (int, float)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
            raise SettingsError(f"{path} must be a number")
        low, high = _LIMITS.get(path, (float("-inf"), float("inf")))
        if not low <= value <= high:
            raise SettingsError(f"{path} must be between {low:g} and {high:g}")
        return type(default)(value) if isinstance(default, int) and float(value).is_integer() else float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise SettingsError(f"{path} must be text")
        if path in _CHOICES and value not in _CHOICES[path]:
            raise SettingsError(f"{path} must be one of: {', '.join(_CHOICES[path])}")
        if path.startswith("control.custom."):
            if value not in ACTIONS:
                raise SettingsError(f"{path}: unknown action {value!r}")
            return value
        if path == "room":
            return value.strip()[:60] or DEFAULTS["room"]
        return value
    raise SettingsError(f"{path} cannot be changed")


def merge(base: dict[str, Any], patch: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
    """Return `base` with `patch` laid over it, every changed value validated.

    `patch` may be partial ({"control": {"preset": "media"}}). With strict=True an unknown key or
    a bad value raises SettingsError; with strict=False (loading a file from disk) it is dropped,
    so a hand-edited or older file can never stop the app from starting.
    """
    if not isinstance(patch, dict):
        raise SettingsError("settings must be a JSON object")
    out = copy.deepcopy(base)

    def lay(target: dict[str, Any], defaults: dict[str, Any], src: dict[str, Any], prefix: str) -> None:
        for key, value in src.items():
            path = f"{prefix}{key}"
            if key not in defaults:
                if strict:
                    raise SettingsError(f"unknown setting {path!r}")
                continue
            if isinstance(defaults[key], dict):
                if not isinstance(value, dict):
                    if strict:
                        raise SettingsError(f"{path} must be an object")
                    continue
                lay(target[key], defaults[key], value, path + ".")
                continue
            try:
                target[key] = _check_leaf(path, value, defaults[key])
            except SettingsError:
                if strict:
                    raise

    lay(out, DEFAULTS, patch, "")
    out["version"] = SCHEMA_VERSION
    return out


def defaults() -> dict[str, Any]:
    return copy.deepcopy(DEFAULTS)


def load(path: Path | None = None) -> dict[str, Any]:
    """Read the settings file. A missing or damaged file gives the defaults (never an exception)."""
    path = path or settings_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return defaults()
    except (OSError, ValueError) as err:
        print(f"[care] settings file {path} could not be read ({err}); using defaults", file=sys.stderr)
        return defaults()
    if not isinstance(raw, dict):
        return defaults()
    return merge(DEFAULTS, raw, strict=False)


def save(settings: dict[str, Any], path: Path | None = None) -> Path:
    """Write the settings atomically (temp file, then rename) so a crash never leaves half a file."""
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".settings-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def mapping(settings: dict[str, Any]) -> dict[str, str]:
    """The action each gesture triggers right now: {"swipe_left": "left", ...}. Missing = "off"."""
    control = settings["control"]
    chosen = control["custom"] if control["preset"] == "custom" else PRESETS[control["preset"]]["map"]
    return {g: (chosen.get(g) if chosen.get(g) in ACTIONS else "off") for g in GESTURES}


def action_label(action: str) -> str:
    return ACTIONS.get(action, (action, ""))[0]


def describe() -> dict[str, Any]:
    """Everything the settings page needs to draw its controls (sent with GET /api/settings)."""
    return {
        "gestures": list(GESTURES),
        "actions": [{"id": k, "label": v[0], "group": v[1]} for k, v in ACTIONS.items()],
        "presets": [{"id": k, "label": v["label"], "hint": v["hint"], "map": {g: v["map"].get(g, "off") for g in GESTURES}}
                    for k, v in PRESETS.items()],
        "no_movement_presets": [{"id": k, **v} for k, v in NO_MOVEMENT_PRESETS.items()],
        "limits": {k: list(v) for k, v in _LIMITS.items()},
        "defaults": defaults(),
    }

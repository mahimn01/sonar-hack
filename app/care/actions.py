"""actions.py: turns "press Right arrow" into a real key press on this Mac (or only logs it).

What it does
    perform("right") presses and releases that key in whatever app is in front. Action names
    are the ones in settings.ACTIONS ("right", "space", "media_play_pause", ...).

    Three ways it can run:
      * dry run   nothing is pressed; every action is logged as "would press ...". Used for
                  tests, for --dry-run, and on any system that is not macOS.
      * live      keys are sent with macOS CoreGraphics, loaded straight from the system with
                  ctypes: nothing to install, nothing to sign. Media keys (play/pause, next,
                  previous) use AppKit when pyobjc is installed, and otherwise ask Spotify or
                  Music directly.
      * blocked   macOS has not granted Accessibility permission. The press is still attempted
                  (a wrong "not allowed" answer must never break a demo) but reported as blocked,
                  with the place in System Settings to fix it.

Why
    This is the only file that can type into other apps, so it is the only file that needs
    the Accessibility permission and the only place a dry run has to be enforced.

The permission
    System Settings > Privacy & Security > Accessibility: switch on the app Sonar Care was started
    from (Terminal, iTerm, ...). macOS attributes the key presses to that app.

What it changes
    Live mode sends key events to the frontmost app. Nothing on disk.

Not here on purpose
    Volume and mute. The sonar plays through the same speakers at a fixed volume; changing it
    would change the sensor's own signal.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Protocol

from . import settings as S

# macOS virtual key codes (ANSI layout).
KEYCODES: dict[str, int] = {
    "left": 123, "right": 124, "down": 125, "up": 126,
    "space": 49, "return": 36, "escape": 53, "tab": 48,
    "page_up": 116, "page_down": 121, "home": 115, "end": 119,
    "b": 11, "f": 3, "k": 40,
}
# Media keys (NX_KEYTYPE_* in IOKit's ev_keymap.h).
MEDIA_KEYS: dict[str, int] = {"media_play_pause": 16, "media_next": 17, "media_previous": 18}
# What to ask a music app to do when media keys cannot be sent.
MEDIA_SCRIPT: dict[str, str] = {"media_play_pause": "playpause", "media_next": "next track", "media_previous": "previous track"}

ACCESSIBILITY_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
ACCESSIBILITY_PATH = "System Settings > Privacy & Security > Accessibility"
TRUST_CACHE_S = 2.0


class Backend(Protocol):
    """Something that can press keys. The real one is MacBackend; tests pass a fake."""

    name: str

    def trusted(self) -> bool | None: ...          # may this process post key events? None = cannot tell
    def key(self, keycode: int) -> None: ...        # press and release one key
    def media(self, action: str) -> str: ...        # press a media key; returns how ("media key", "Spotify", ...)
    def media_how(self) -> str: ...                 # how media keys would be sent right now
    def request_access(self) -> bool: ...           # ask macOS to show its permission prompt


class ActionError(RuntimeError):
    """An action could not be carried out; the message is fit for the UI."""


class MacBackend:
    """Key presses through CoreGraphics (ctypes). Works on a stock Mac with the system Python or any venv."""

    name = "CoreGraphics"

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise ActionError(f"key presses need macOS (this is {sys.platform})")
        path = ctypes.util.find_library("ApplicationServices") or "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        cf_path = ctypes.util.find_library("CoreFoundation") or "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        try:
            self._cg = ctypes.CDLL(path)
            self._cf = ctypes.CDLL(cf_path)
        except OSError as err:
            raise ActionError(f"could not load the macOS event library: {err}") from err
        cg, cf = self._cg, self._cf
        cg.CGEventSourceCreate.restype = ctypes.c_void_p
        cg.CGEventSourceCreate.argtypes = [ctypes.c_int32]
        cg.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
        cg.CGEventCreateKeyboardEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool]
        cg.CGEventPost.restype = None
        cg.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        cg.AXIsProcessTrusted.restype = ctypes.c_bool
        cg.AXIsProcessTrusted.argtypes = []
        cf.CFRelease.restype = None
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        self._preflight = self._optional("CGPreflightPostEventAccess")      # macOS 10.15+
        self._request = self._optional("CGRequestPostEventAccess")
        self._appkit: Any = None
        self._quartz: Any = None
        try:                                               # media keys need a system-defined NSEvent: pyobjc, if installed
            import AppKit
            import Quartz
            self._appkit, self._quartz = AppKit, Quartz
        except Exception:                                  # noqa: BLE001  (any import problem means "not available")
            pass

    def _optional(self, name: str) -> Any:
        try:
            fn = getattr(self._cg, name)
        except AttributeError:
            return None
        fn.restype = ctypes.c_bool
        fn.argtypes = []
        return fn

    def trusted(self) -> bool | None:
        try:
            if self._preflight is not None:
                return bool(self._preflight())
            return bool(self._cg.AXIsProcessTrusted())
        except Exception:                                  # noqa: BLE001
            return None

    def request_access(self) -> bool:
        """Show macOS's own "allow this app to control your computer" prompt (it appears once per app)."""
        if self._request is None:
            return False
        try:
            return bool(self._request())
        except Exception:                                  # noqa: BLE001
            return False

    def make_key_events(self, keycode: int) -> tuple[int, int, int]:
        """Create (source, key-down, key-up) without posting them. The caller must release all three."""
        source = self._cg.CGEventSourceCreate(1)           # 1 = the hardware keyboard's state, like a real key
        down = self._cg.CGEventCreateKeyboardEvent(source, keycode, True)
        up = self._cg.CGEventCreateKeyboardEvent(source, keycode, False)
        if not down or not up:
            for ref in (down, up, source):
                if ref:
                    self._cf.CFRelease(ref)
            raise ActionError("macOS refused to create the key event")
        return source, down, up

    def release(self, *refs: int) -> None:
        for ref in refs:
            if ref:
                self._cf.CFRelease(ref)

    def key(self, keycode: int) -> None:
        source, down, up = self.make_key_events(keycode)
        try:
            self._cg.CGEventPost(0, down)                  # 0 = where hardware key presses enter the system
            self._cg.CGEventPost(0, up)
        finally:
            self.release(down, up, source)

    def media_how(self) -> str:
        if self._appkit is not None:
            return "media key"
        app = _running_music_app()
        return app or "unavailable (start Spotify or Music, or install pyobjc)"

    def media(self, action: str) -> str:
        if self._appkit is not None:
            code = MEDIA_KEYS[action]
            for down in (True, False):
                state = 0xA if down else 0xB
                event = self._appkit.NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                    14, (0, 0), state << 8, 0, 0, None, 8, (code << 16) | (state << 8), -1)
                self._quartz.CGEventPost(0, event.CGEvent())
            return "media key"
        app = _running_music_app()
        if app is None:
            raise ActionError("no media key support: start Spotify or Music, or install pyobjc")
        done = subprocess.run(["osascript", "-e", f'tell application "{app}" to {MEDIA_SCRIPT[action]}'],
                              capture_output=True, text=True, timeout=5)
        if done.returncode != 0:
            raise ActionError(f"{app} did not accept the command: {done.stderr.strip()[:120]}")
        return app


def _running_music_app() -> str | None:
    """Spotify or Music, whichever is running (never starts one)."""
    if not shutil.which("pgrep"):
        return None
    for app in ("Spotify", "Music"):
        if subprocess.run(["pgrep", "-x", app], capture_output=True).returncode == 0:
            return app
    return None


def make_backend() -> Backend | None:
    """The real key-press backend for this machine, or None when there is none (not a Mac)."""
    try:
        return MacBackend()
    except ActionError as err:
        print(f"[care] key presses unavailable: {err}", file=sys.stderr, flush=True)
        return None


class Actions:
    """Performs actions, or records what it would have done. Safe to call from any thread."""

    def __init__(self, dry_run: bool = False, backend: Backend | None = None) -> None:
        self.dry_run = dry_run
        self._backend = backend if backend is not None else (None if dry_run else make_backend())
        self._lock = threading.Lock()
        self._trust: tuple[float, bool | None] = (0.0, None)
        self.log: list[dict[str, Any]] = []       # the last 100 actions, newest last (tests and the dry-run view read it)

    # ---- permission -----------------------------------------------------------------------------

    def trusted(self) -> bool | None:
        """May we post key events? True / False / None (unknown). Checked at most every 2 s."""
        if self._backend is None:
            return None
        at, value = self._trust
        if time.monotonic() - at > TRUST_CACHE_S:
            value = self._backend.trusted()
            self._trust = (time.monotonic(), value)
        return value

    def request_access(self) -> bool:
        """Ask macOS to show its permission prompt. False in a dry run or when there is no backend."""
        if self.dry_run or self._backend is None:
            return False
        self._trust = (0.0, None)
        return self._backend.request_access()

    def open_settings(self) -> bool:
        """Open the Accessibility page of System Settings."""
        if sys.platform != "darwin":
            return False
        try:
            subprocess.Popen(["open", ACCESSIBILITY_URL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            return False

    def status(self) -> dict[str, Any]:
        """Whether key presses can reach other apps right now, in words fit for the UI."""
        base = {"settings_path": ACCESSIBILITY_PATH, "platform": sys.platform}
        if self.dry_run:
            return {**base, "mode": "dry_run", "trusted": None, "backend": "dry run", "can_press": False, "media": "dry run",
                    "note": "Dry run: actions are logged, no key is pressed."}
        if self._backend is None:
            return {**base, "mode": "unavailable", "trusted": None, "backend": "none", "can_press": False, "media": "unavailable",
                    "note": "Key presses are not available on this system."}
        trusted = self.trusted()
        if trusted is False:
            note = f"macOS is blocking key presses. Turn on the app you started Sonar Care from in {ACCESSIBILITY_PATH}."
        elif trusted is None:
            note = "Could not check the Accessibility permission. Key presses are sent anyway."
        else:
            note = "Key presses are allowed."
        return {**base, "mode": "blocked" if trusted is False else "live", "trusted": trusted, "backend": self._backend.name,
                "can_press": trusted is not False, "media": self._backend.media_how(), "note": note}

    # ---- pressing -------------------------------------------------------------------------------

    def perform(self, action: str) -> dict[str, Any]:
        """Do one action. Returns {"result": "sent" | "dry_run" | "blocked" | "failed", "detail": str}."""
        label = S.action_label(action)
        if action not in S.ACTIONS or action == "off" or (action not in KEYCODES and action not in MEDIA_KEYS):
            out = {"result": "failed", "detail": f"unknown action {action!r}"}
        elif self.dry_run:
            out = {"result": "dry_run", "detail": f"would press {label}"}
        elif self._backend is None:
            out = {"result": "failed", "detail": "key presses are not available on this system"}
        else:
            out = self._press(action, label)
        with self._lock:
            self.log.append({"t": time.time(), "action": action, "label": label, **out})
            del self.log[:-100]
        return out

    def _press(self, action: str, label: str) -> dict[str, Any]:
        trusted = self.trusted()
        try:
            if action in MEDIA_KEYS:
                how = self._backend.media(action)
                sent = f"pressed {label}" if how == "media key" else f"{label} sent to {how}"
                if how != "media key":                    # asking an app directly needs no Accessibility permission
                    return {"result": "sent", "detail": sent}
            else:
                self._backend.key(KEYCODES[action])
                sent = f"pressed {label}"
        except ActionError as err:
            return {"result": "failed", "detail": str(err)}
        except Exception as err:                           # noqa: BLE001  (a system call failed: report it, keep running)
            return {"result": "failed", "detail": f"{type(err).__name__}: {err}"}
        if trusted is False:
            return {"result": "blocked", "detail": f"macOS is blocking key presses: allow them in {ACCESSIBILITY_PATH}"}
        return {"result": "sent", "detail": sent}

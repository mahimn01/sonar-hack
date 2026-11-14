"""Gesture-to-keyboard bridge: drive PowerPoint or Google Slides with sonar gestures.

What it does
    Listens to the sonar engine's WebSocket and, when a gesture is recognised,
    presses a key on this computer:
        swipe_right -> Right arrow (next slide)
        swipe_left  -> Left arrow (previous slide)
        push, pull  -> nothing, unless you map them with --map
    It applies the same safety filters as the web slides: a minimum confidence,
    a short cooldown, and ignoring the hand's return stroke after a swipe.

Why
    So the team can present from PowerPoint, Keynote or Google Slides instead of
    the built-in web slides (app/web/presenter.html).

How to run (Windows: use `py -3.12` instead of `python`)
    python app/keys_bridge.py                          real engine on ws://127.0.0.1:8765
    python app/keys_bridge.py --dry-run                print the key it would press, press nothing
    python app/keys_bridge.py --map push=b             push also blanks the screen in PowerPoint
    python app/keys_bridge.py --map swipe_right=page_down,swipe_left=page_up
    Then click into the slideshow window within the arm delay (default 3 s).

What it changes
    Sends key presses to whichever window has focus. Nothing on disk.
    Do not also drive the web presenter page with gestures at the same time.
    macOS: the terminal needs Accessibility permission. Windows: if PowerPoint runs
    as administrator, run this as administrator too.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter import GESTURES, adapt  # noqa: E402  (local module next to this file)

DEFAULT_MAP = {"swipe_right": "right", "swipe_left": "left"}
OPPOSITE = {"swipe_left": "swipe_right", "swipe_right": "swipe_left", "push": "pull", "pull": "push"}


@dataclass
class GestureGate:
    """Decides whether a gesture should fire (mirrors the web store's filters)."""

    min_confidence: float = 0.6
    cooldown_s: float = 0.7
    return_stroke_s: float = 1.1
    _last_label: str | None = None
    _last_at: float = -1e9

    def judge(self, label: str, confidence: float | None, now: float) -> str | None:
        """Return None if accepted, else the reason it was rejected."""
        if label not in GESTURES:
            return "not a gesture"
        if (1.0 if confidence is None else confidence) < self.min_confidence:
            return "low confidence"
        if now - self._last_at < self.cooldown_s:
            return "too soon"
        if self._last_label and OPPOSITE[label] == self._last_label and now - self._last_at < self.return_stroke_s:
            return "return stroke"
        self._last_label, self._last_at = label, now
        return None


def parse_map(text: str | None) -> dict[str, str]:
    """'push=b,swipe_right=page_down' -> {'push': 'b', 'swipe_right': 'page_down'} on top of the defaults."""
    mapping = dict(DEFAULT_MAP)
    if not text:
        return mapping
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise SystemExit(f"--map entry {part!r} must look like label=key")
        label, key = (s.strip().lower() for s in part.split("=", 1))
        if label not in GESTURES:
            raise SystemExit(f"--map: unknown gesture {label!r}; use one of {', '.join(GESTURES)}")
        if key in ("", "none", "off"):
            mapping.pop(label, None)
        else:
            mapping[label] = key
    return mapping


class KeyPresser:
    """Presses keys with pynput, or only prints them in dry-run mode."""

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run
        self.kb = None
        self.Key = None
        if not dry_run:
            try:
                from pynput.keyboard import Controller, Key
            except ImportError as err:
                raise SystemExit(f"pynput is not installed ({err}). Run: pip install -r app/requirements.txt") from err
            self.kb, self.Key = Controller(), Key

    def resolve(self, name: str) -> object:
        if self.Key is None:
            return name
        if len(name) == 1:
            return name
        key = getattr(self.Key, name, None)
        if key is None:
            raise SystemExit(f"unknown key name {name!r} (examples: right, left, page_down, space, b)")
        return key

    def press(self, name: str) -> None:
        if self.dry_run:
            print(f"[bridge] would press: {name}", flush=True)
            return
        key = self.resolve(name)
        self.kb.press(key)
        self.kb.release(key)
        print(f"[bridge] pressed: {name}", flush=True)


async def run(args: argparse.Namespace) -> None:
    try:
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed, InvalidURI
    except ImportError as err:
        raise SystemExit(f"websockets>=13 is needed ({err}). Run: pip install -r app/requirements.txt") from err
    connect_kwargs = {"open_timeout": 3}
    if "proxy" in inspect.signature(connect).parameters:
        connect_kwargs["proxy"] = None            # never route localhost through a system proxy

    mapping = parse_map(args.map)
    presser = KeyPresser(args.dry_run)
    for name in mapping.values():                 # fail now, not mid-talk, on a bad key name
        presser.resolve(name)
    gate = GestureGate(args.min_conf, args.cooldown, args.return_ms / 1000.0)

    print(f"[bridge] mapping: {', '.join(f'{k} -> {v}' for k, v in mapping.items()) or '(none)'}")
    if not args.dry_run and args.arm_delay > 0:
        print(f"[bridge] click into your slideshow now; keys go live in {args.arm_delay:.0f} s")
        await asyncio.sleep(args.arm_delay)

    attempt = 0
    while True:
        try:
            async with connect(args.ws, **connect_kwargs) as ws:
                attempt = 0
                print(f"[bridge] connected to {args.ws}; waiting for gestures (Ctrl+C to stop)", flush=True)
                async for message in ws:
                    for evt in adapt(message):
                        if evt["kind"] != "gesture" or evt["label"] == "none":
                            continue
                        reason = gate.judge(evt["label"], evt["confidence"], time.monotonic())
                        conf = "?" if evt["confidence"] is None else f"{evt['confidence']:.2f}"
                        if reason:
                            print(f"[bridge] {evt['label']} ({conf}) ignored: {reason}", flush=True)
                            continue
                        key = mapping.get(evt["label"])
                        if key:
                            presser.press(key)
                        else:
                            print(f"[bridge] {evt['label']} ({conf}) has no key mapped", flush=True)
        except InvalidURI as err:
            raise SystemExit(f"[bridge] bad --ws URL: {err}") from err
        except (OSError, asyncio.TimeoutError, ConnectionClosed) as err:
            delay = min(2.0, 0.25 * 2 ** attempt)
            attempt += 1
            print(f"[bridge] engine not reachable ({type(err).__name__}); retrying in {delay:.1f} s", flush=True)
            await asyncio.sleep(delay)


def main() -> None:
    ap = argparse.ArgumentParser(description="Press keys when the sonar engine sees a gesture.")
    ap.add_argument("--ws", default="ws://127.0.0.1:8765", help="engine WebSocket URL")
    ap.add_argument("--map", default=None, help="extra mappings, e.g. push=b,swipe_right=page_down (key=none to disable)")
    ap.add_argument("--min-conf", type=float, default=0.6, help="ignore gestures below this confidence")
    ap.add_argument("--cooldown", type=float, default=0.7, help="seconds to ignore gestures after one fires")
    ap.add_argument("--return-ms", type=float, default=1100, help="ignore the opposite gesture for this long (hand returning)")
    ap.add_argument("--arm-delay", type=float, default=3.0, help="seconds to wait before pressing keys, to focus the slides")
    ap.add_argument("--dry-run", action="store_true", help="print keys instead of pressing them")
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("[bridge] stopped")


if __name__ == "__main__":
    main()

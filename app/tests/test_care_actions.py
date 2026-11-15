"""Tests for app/care/actions.py (key presses) and app/care/notify.py (notification, sound, voice).

What it checks
    - a dry run logs "would press ..." and never touches the keyboard;
    - live mode sends the right macOS key code, or the right media key, through the backend;
    - a missing Accessibility permission is reported as "blocked" (and the press is still attempted);
    - failures come back as a readable result, never as an exception;
    - alerts are announced with the right commands, rate-limited, and never when quiet.

Run: python -m pytest app/tests/test_care_actions.py -q
PRESSES NO KEYS, shows no notification and plays no sound: the backend and the command runner are fakes.
The one macOS-only test builds key events and releases them without posting them.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from care import actions as A  # noqa: E402
from care import notify as N  # noqa: E402
from care import settings as S  # noqa: E402


class FakeBackend:
    """Records what would have been pressed."""

    name = "fake"

    def __init__(self, trusted: bool | None = True, media: str = "media key", fail: Exception | None = None) -> None:
        self._trusted, self._media, self._fail = trusted, media, fail
        self.keys: list[int] = []
        self.medias: list[str] = []
        self.trust_checks = 0
        self.asked = 0

    def trusted(self) -> bool | None:
        self.trust_checks += 1
        return self._trusted

    def key(self, keycode: int) -> None:
        if self._fail:
            raise self._fail
        self.keys.append(keycode)

    def media(self, action: str) -> str:
        if self._fail:
            raise self._fail
        self.medias.append(action)
        return self._media

    def media_how(self) -> str:
        return self._media

    def request_access(self) -> bool:
        self.asked += 1
        return True


class KeyTables(unittest.TestCase):
    def test_every_action_in_the_settings_can_be_pressed(self) -> None:
        for action in S.ACTIONS:
            if action != "off":
                self.assertTrue(action in A.KEYCODES or action in A.MEDIA_KEYS, action)
        self.assertEqual((A.KEYCODES["left"], A.KEYCODES["right"], A.KEYCODES["space"]), (123, 124, 49))
        self.assertEqual((A.KEYCODES["page_down"], A.KEYCODES["page_up"], A.KEYCODES["escape"]), (121, 116, 53))
        self.assertEqual(A.MEDIA_KEYS, {"media_play_pause": 16, "media_next": 17, "media_previous": 18})

    def test_no_volume_actions(self) -> None:
        # Changing the volume would change the sonar's own signal, so no gesture may do it.
        self.assertFalse([a for a in S.ACTIONS if "volume" in a or "mute" in a])


class DryRun(unittest.TestCase):
    def test_logs_and_presses_nothing(self) -> None:
        backend = FakeBackend()
        acts = A.Actions(dry_run=True, backend=backend)
        self.assertEqual(acts.perform("right"), {"result": "dry_run", "detail": "would press Right arrow"})
        self.assertEqual(acts.perform("media_play_pause")["detail"], "would press Play / pause")
        self.assertEqual((backend.keys, backend.medias), ([], []))
        self.assertEqual([e["action"] for e in acts.log], ["right", "media_play_pause"])
        status = acts.status()
        self.assertEqual((status["mode"], status["can_press"], status["trusted"]), ("dry_run", False, None))
        self.assertFalse(acts.request_access())

    def test_no_backend_is_created_for_a_dry_run(self) -> None:
        self.assertIsNone(A.Actions(dry_run=True)._backend)

    def test_log_is_bounded(self) -> None:
        acts = A.Actions(dry_run=True)
        for _ in range(130):
            acts.perform("space")
        self.assertEqual(len(acts.log), 100)


class Live(unittest.TestCase):
    def test_keys_and_media_keys_reach_the_backend(self) -> None:
        backend = FakeBackend()
        acts = A.Actions(dry_run=False, backend=backend)
        self.assertEqual(acts.perform("right"), {"result": "sent", "detail": "pressed Right arrow"})
        self.assertEqual(acts.perform("space")["result"], "sent")
        self.assertEqual(acts.perform("media_next"), {"result": "sent", "detail": "pressed Next track"})
        self.assertEqual((backend.keys, backend.medias), ([124, 49], ["media_next"]))
        status = acts.status()
        self.assertEqual((status["mode"], status["can_press"], status["backend"], status["media"]), ("live", True, "fake", "media key"))

    def test_media_through_a_music_app_needs_no_permission(self) -> None:
        acts = A.Actions(dry_run=False, backend=FakeBackend(trusted=False, media="Spotify"))
        self.assertEqual(acts.perform("media_play_pause"), {"result": "sent", "detail": "Play / pause sent to Spotify"})

    def test_missing_permission_is_reported_but_the_press_is_still_tried(self) -> None:
        backend = FakeBackend(trusted=False)
        acts = A.Actions(dry_run=False, backend=backend)
        out = acts.perform("right")
        self.assertEqual(out["result"], "blocked")
        self.assertIn("Privacy & Security > Accessibility", out["detail"])
        self.assertEqual(backend.keys, [124], "a wrong 'not allowed' answer from macOS must never break the demo")
        status = acts.status()
        self.assertEqual((status["mode"], status["can_press"], status["trusted"]), ("blocked", False, False))
        self.assertIn("Accessibility", status["note"])

    def test_unknown_permission_state_still_presses(self) -> None:
        acts = A.Actions(dry_run=False, backend=FakeBackend(trusted=None))
        self.assertEqual(acts.perform("left")["result"], "sent")
        self.assertEqual(acts.status()["mode"], "live")

    def test_bad_actions_fail_without_pressing(self) -> None:
        backend = FakeBackend()
        acts = A.Actions(dry_run=False, backend=backend)
        for action in ("off", "volume_up", "rm -rf", ""):
            self.assertEqual(acts.perform(action)["result"], "failed", action)
        self.assertEqual(backend.keys, [])

    def test_backend_errors_become_results_not_exceptions(self) -> None:
        told = A.Actions(dry_run=False, backend=FakeBackend(fail=A.ActionError("macOS refused to create the key event")))
        self.assertEqual(told.perform("right"), {"result": "failed", "detail": "macOS refused to create the key event"})
        odd = A.Actions(dry_run=False, backend=FakeBackend(fail=OSError("boom")))
        self.assertEqual(odd.perform("media_next"), {"result": "failed", "detail": "OSError: boom"})

    def test_permission_is_checked_at_most_every_two_seconds(self) -> None:
        backend = FakeBackend()
        acts = A.Actions(dry_run=False, backend=backend)
        for _ in range(20):
            acts.perform("right")
            acts.status()
        self.assertEqual(backend.trust_checks, 1)
        acts.request_access()                              # asking again forgets the cached answer
        acts.status()
        self.assertEqual((backend.asked, backend.trust_checks), (1, 2))

    def test_no_backend_on_this_system(self) -> None:
        acts = A.Actions(dry_run=False, backend=None)
        if acts._backend is None:                          # e.g. Linux: nothing to press with
            self.assertEqual(acts.perform("right")["result"], "failed")
            self.assertEqual(acts.status()["mode"], "unavailable")

    @unittest.skipUnless(sys.platform == "darwin", "the real backend is macOS only")
    def test_real_backend_builds_key_events_without_posting_them(self) -> None:
        backend = A.MacBackend()
        self.assertIn(backend.trusted(), (True, False, None))
        for name, code in A.KEYCODES.items():
            source, down, up = backend.make_key_events(code)
            self.assertTrue(source and down and up, name)
            backend.release(down, up, source)              # created and released; never posted
        self.assertIsInstance(backend.media_how(), str)


def alert(kind: str = "help", level: str = "alert", ident: str = "help-1", title: str = "Help requested",
          detail: str = "Two pushes in a row at the laptop") -> dict[str, Any]:
    return {"id": ident, "kind": kind, "level": level, "title": title, "detail": detail}


class Notifications(unittest.TestCase):
    def setUp(self) -> None:
        self.ran: list[list[str]] = []
        self.notifier = N.Notifier(enabled=True, runner=self.ran.append)
        self.opts = dict(notification=True, sound=True, volume=0.7, speak=True)

    def test_raised_alert_shows_a_banner_and_plays_chime_then_voice(self) -> None:
        seconds = self.notifier.alert(alert(), reminder=False, now=100.0, **self.opts)
        self.assertEqual(len(self.ran), 2)
        banner, audio = self.ran
        self.assertEqual(banner[0], "osascript")
        self.assertEqual(banner[-3:], ["Sonar Care", "Help requested", "Two pushes in a row at the laptop"])
        self.assertEqual(audio[:2], ["/bin/sh", "-c"])
        self.assertEqual(audio[-3:], ["0.70", "/System/Library/Sounds/Sosumi.aiff", "Help requested"])
        self.assertGreater(seconds, 2.0)                   # chime + voice: the rules ignore movement for this long
        self.assertEqual((self.notifier.sent[-1]["delivered"], self.notifier.sent[-1]["speech"]), (True, True))

    def test_texts_travel_as_arguments_never_inside_a_script(self) -> None:
        nasty = 'He said "help"; do shell script "rm -rf ~" \\ $(reboot) `id`'
        self.notifier.alert(alert(title=nasty, detail=nasty), reminder=False, now=100.0, **self.opts)
        banner, audio = self.ran
        self.assertNotIn(nasty, " ".join(banner[: banner.index("Sonar Care")]))     # the osascript program itself is fixed
        self.assertEqual(banner[-1], nasty)
        self.assertNotIn(nasty, audio[2])                                             # the sh program is fixed too
        self.assertEqual(audio[-1], nasty)

    def test_attention_alert_uses_the_softer_sound(self) -> None:
        self.notifier.alert(alert("no_movement", "attention", "no_movement-2", "No movement"), reminder=False, now=100.0, **self.opts)
        self.assertIn("/System/Library/Sounds/Glass.aiff", self.ran[1])

    def test_reminder_is_sound_only(self) -> None:
        self.notifier.alert(alert(), reminder=False, now=100.0, **self.opts)
        self.ran.clear()
        seconds = self.notifier.alert(alert(), reminder=True, now=125.0, **self.opts)
        self.assertEqual(len(self.ran), 1)
        self.assertEqual(self.ran[0][0], "afplay")
        self.assertEqual(seconds, 1.0)

    def test_rate_limits(self) -> None:
        self.assertGreater(self.notifier.alert(alert(), reminder=False, now=100.0, **self.opts), 0)
        self.ran.clear()
        self.assertEqual(self.notifier.alert(alert(), reminder=True, now=102.0, **self.opts), 0.0)     # same alert, too soon
        self.assertEqual(self.ran, [])
        # A different alert a second later: its banner shows, but two sounds never start within 2 s of each other.
        seconds = self.notifier.alert(alert("no_movement", "attention", "no_movement-2", "No movement"), reminder=False, now=101.0, **self.opts)
        self.assertEqual((len(self.ran), self.ran[0][0], seconds), (1, "osascript", 0.0))

    def test_each_channel_can_be_switched_off(self) -> None:
        self.notifier.alert(alert(ident="a"), reminder=False, now=100.0, notification=False, sound=True, volume=0.3, speak=False)
        self.assertEqual(self.ran, [["afplay", "-v", "0.30", "/System/Library/Sounds/Sosumi.aiff"]])
        self.ran.clear()
        self.assertEqual(self.notifier.alert(alert(ident="b"), reminder=False, now=110.0, notification=True, sound=False, volume=1, speak=True), 0.0)
        self.assertEqual([c[0] for c in self.ran], ["osascript"])

    def test_quiet_records_but_does_nothing(self) -> None:
        quiet = N.Notifier(enabled=False, runner=self.ran.append)
        self.assertEqual(quiet.alert(alert(), reminder=False, now=100.0, **self.opts), 0.0)
        self.assertEqual(self.ran, [])
        self.assertEqual((quiet.sent[-1]["kind"], quiet.sent[-1]["delivered"]), ("help", False))

    def test_a_missing_tool_never_raises(self) -> None:
        def broken(cmd: list[str]) -> None:
            raise FileNotFoundError(cmd[0])
        notifier = N.Notifier(enabled=True, runner=broken)
        self.assertEqual(notifier.alert(alert(), reminder=False, now=100.0, **self.opts), 0.0)
        self.assertFalse(notifier.sent[-1]["delivered"])

    def test_volume_is_clamped(self) -> None:
        self.assertEqual(N.sound_command("alert", 7)[0][2], "1.00")
        self.assertEqual(N.sound_command("alert", -1)[0][2], "0.00")


if __name__ == "__main__":
    unittest.main()

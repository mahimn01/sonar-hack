"""Tests for app/care/rules.py: every Sonar Care decision, driven with a fake clock.

What it checks
    - gestures press the mapped key, and the filters (confidence, cooldown, return stroke) hold;
    - the help gesture: two pushes raise help, one push never does both "key" and "help";
    - "No movement": raised only for a confirmed, still person; resolved by movement, acknowledged, re-raised;
    - "No breathing movement": the same scenarios as the web monitor's own tests (app/tests/store.test.js);
    - alert life cycle: acknowledge, time-to-acknowledge, reminders, putting finished alerts away.

Run: python -m pytest app/tests/test_care_rules.py -q      (or: python -m unittest discover -s app/tests)
Read-only: no clock, no network, no key presses. Engine messages go through app/adapter.py first, as in the app.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from adapter import adapt  # noqa: E402
from care import rules as R  # noqa: E402
from care import settings as S  # noqa: E402

T0 = 1794600000.0


class Rig:
    """A fake engine + a fake clock around one CareRules."""

    def __init__(self, patch: dict[str, Any] | None = None, connect: bool = True) -> None:
        self.now = T0
        self.settings = S.merge(S.defaults(), patch or {})
        self.rules = R.CareRules(self.settings, self.now)
        self.effects: list[dict[str, Any]] = []
        if connect:
            self.keep(self.rules.on_link(True, self.now))

    def keep(self, effects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.effects += effects
        return effects

    def raw(self, msg: dict[str, Any]) -> list[dict[str, Any]]:
        """Send one engine-shaped message through the adapter into the rules."""
        out: list[dict[str, Any]] = []
        for evt in adapt(json.dumps({"t": self.now, **msg}), self.now):
            out += self.rules.on_event(evt, self.now)
        return self.keep(out)

    def tick(self) -> list[dict[str, Any]]:
        return self.keep(self.rules.tick(self.now))

    def wait(self, seconds: float, step: float = 0.25) -> list[dict[str, Any]]:
        """Let time pass with nothing arriving."""
        out: list[dict[str, Any]] = []
        for _ in range(int(round(seconds / step))):
            self.now += step
            out += self.tick()
        return out

    def gesture(self, label: str, confidence: float | None = 0.9) -> list[dict[str, Any]]:
        return self.raw({"type": "gesture", "label": label, "confidence": confidence})

    def presence(self, state: str = "present", confidence: float = 0.95, moving: bool = False, range_m: float | None = 0.6,
                 motionless_s: float = 0.0) -> list[dict[str, Any]]:
        return self.raw({"type": "presence", "state": state, "confidence": confidence, "moving": moving,
                         "range_m": range_m, "motionless_s": motionless_s})

    def sit(self, seconds: int, moving: bool = False, breathing: dict[str, Any] | None = None, confidence: float = 0.95,
            state: str = "present") -> list[dict[str, Any]]:
        """`seconds` of a person in front of the laptop: one presence (and breathing) message a second, 4 ticks a second."""
        out: list[dict[str, Any]] = []
        for _ in range(seconds):
            self.now += 1.0
            out += self.presence(state, confidence, moving)
            if breathing is not None:
                out += self.raw({"type": "breathing", **breathing})
            for _ in range(4):
                out += self.tick()
        return out

    def apply(self, patch: dict[str, Any]) -> list[dict[str, Any]]:
        self.settings = S.merge(self.settings, patch)
        return self.keep(self.rules.apply_settings(self.settings, self.now))

    # ---- reading the effects --------------------------------------------------------------------
    def acts(self, effects: list[dict[str, Any]] | None = None) -> list[tuple[str, str]]:
        return [(e["gesture"], e["action"]) for e in (self.effects if effects is None else effects)
                if e["effect"] == "gesture" and e["result"] == "act"]

    def results(self, effects: list[dict[str, Any]] | None = None) -> list[tuple[str, str, str | None]]:
        return [(e["gesture"], e["result"], e.get("reason")) for e in (self.effects if effects is None else effects)
                if e["effect"] == "gesture"]

    def alert_changes(self, kind: str | None = None, effects: list[dict[str, Any]] | None = None) -> list[tuple[str, str]]:
        return [(e["alert"]["kind"], e["change"]) for e in (self.effects if effects is None else effects)
                if e["effect"] == "alert" and (kind is None or e["alert"]["kind"] == kind)]

    def timeline(self) -> list[str]:
        return [e["entry"]["text"] for e in self.effects if e["effect"] == "timeline"]

    def open_alert(self, kind: str) -> dict[str, Any] | None:
        return self.rules.alerts.get(kind)


NO_HELP = {"monitoring": {"help": {"enabled": False}}}
HELP_NOW = {"monitoring": {"help": {"confirm_s": 0}}}          # no cancel countdown: the alert goes out at once


class GestureControl(unittest.TestCase):
    def test_swipes_press_the_slides_keys(self) -> None:
        rig = Rig(NO_HELP)
        rig.gesture("swipe_right")
        rig.wait(2)
        rig.gesture("swipe_left")
        self.assertEqual(rig.acts(), [("swipe_right", "right"), ("swipe_left", "left")])

    def test_push_presses_space_at_once_when_help_is_off(self) -> None:
        rig = Rig(NO_HELP)
        self.assertEqual(rig.acts(rig.gesture("push")), [("push", "space")])

    def test_low_confidence_is_ignored(self) -> None:
        rig = Rig(NO_HELP)
        self.assertEqual(rig.results(rig.gesture("swipe_right", 0.4)), [("swipe_right", "ignored", "low confidence")])

    def test_cooldown_and_return_stroke(self) -> None:
        rig = Rig(NO_HELP)
        rig.gesture("swipe_right")
        rig.now += 0.3
        self.assertEqual(rig.results(rig.gesture("swipe_right"))[0][2], "too soon")
        rig.now += 0.6                      # 0.9 s after the swipe: past the cooldown, inside the return-stroke window
        self.assertEqual(rig.results(rig.gesture("swipe_left"))[0][2], "return stroke")
        rig.now += 0.35                     # 1.25 s: the hand still coming back (seen at 1.24 s in the recordings)
        self.assertEqual(rig.results(rig.gesture("swipe_left"))[0][2], "return stroke")
        rig.now += 0.35                     # 1.6 s: a real swipe back
        self.assertEqual(rig.acts(rig.gesture("swipe_left")), [("swipe_left", "left")])

    def test_one_movement_reported_twice_acts_once(self) -> None:
        # Measured: one push is reported as two push events 0.40-0.98 s apart in 4 of 10 cases; swipes repeat too.
        rig = Rig(NO_HELP)
        rig.gesture("push")
        rig.now += 0.5
        self.assertEqual(rig.results(rig.gesture("push"))[0][2], "too soon")
        rig.now += 0.4                      # 0.9 s: past the cooldown, still the same movement
        self.assertEqual(rig.results(rig.gesture("push"))[0][2], "same movement")
        rig.now += 0.5                      # 1.4 s: a second, deliberate push
        rig.gesture("push")
        self.assertEqual(rig.acts(), [("push", "space"), ("push", "space")])
        rig.wait(3)
        rig.gesture("swipe_right")
        rig.now += 1.2
        self.assertEqual(rig.results(rig.gesture("swipe_right"))[0][2], "same movement")

    def test_control_off_presses_nothing(self) -> None:
        rig = Rig({"control": {"enabled": False}, **NO_HELP})
        self.assertEqual(rig.results(rig.gesture("swipe_right")), [("swipe_right", "off", "gesture control is off")])
        self.assertEqual(rig.acts(), [])

    def test_presets_and_custom_mapping(self) -> None:
        media = Rig({"control": {"preset": "media"}, **NO_HELP})
        media.gesture("swipe_right"); media.wait(2); media.gesture("push"); media.wait(2); media.gesture("swipe_left")
        self.assertEqual(media.acts(), [("swipe_right", "media_next"), ("push", "media_play_pause"), ("swipe_left", "media_previous")])

        scroll = Rig({"control": {"preset": "scroll"}, **NO_HELP})
        self.assertEqual(scroll.acts(scroll.gesture("swipe_right")), [("swipe_right", "page_down")])
        scroll.wait(2)
        self.assertEqual(scroll.results(scroll.gesture("push")), [("push", "unmapped", "no action for this gesture")])

        custom = Rig({"control": {"preset": "custom", "custom": {"swipe_right": "k", "swipe_left": "off", "push": "escape"}}, **NO_HELP})
        self.assertEqual(custom.acts(custom.gesture("swipe_right")), [("swipe_right", "k")])
        custom.wait(2)
        self.assertEqual(custom.results(custom.gesture("swipe_left"))[0][1], "unmapped")

    def test_gestures_the_model_does_not_have_are_ignored(self) -> None:
        rig = Rig(NO_HELP)
        self.assertEqual(rig.results(rig.gesture("pull")), [("pull", "ignored", "unsupported gesture")])
        self.assertEqual(rig.raw({"type": "gesture", "label": "none", "confidence": 0.99}), [])

    def test_a_gesture_during_our_own_alert_sound_is_ignored(self) -> None:
        rig = Rig(NO_HELP)
        rig.rules.sound_played(rig.now, 1.5)
        rig.now += 1.0
        self.assertEqual(rig.results(rig.gesture("swipe_right")), [("swipe_right", "ignored", "alert sound playing")])
        rig.now += 1.0
        self.assertEqual(rig.acts(rig.gesture("swipe_right")), [("swipe_right", "right")])

    def test_last_gesture_is_remembered_for_the_menu(self) -> None:
        rig = Rig(NO_HELP)
        rig.gesture("swipe_right", 0.93)
        last = rig.rules.snapshot(rig.now)["last_gesture"]
        self.assertEqual((last["gesture"], last["result"], last["action"], last["confidence"]), ("swipe_right", "act", "right", 0.93))


class HelpGesture(unittest.TestCase):
    def test_single_push_acts_only_after_the_help_window(self) -> None:
        rig = Rig()
        first = rig.gesture("push")
        self.assertEqual(rig.results(first), [("push", "held", "waiting for a second push")])
        self.assertAlmostEqual(first[0]["help_until"], rig.now + 2.5)
        self.assertTrue(rig.rules.snapshot(rig.now)["help"]["pending"])
        rig.wait(2.25)
        self.assertEqual(rig.acts(), [], "nothing may be pressed inside the window")
        rig.wait(0.5)
        self.assertEqual(rig.acts(), [("push", "space")])
        self.assertGreater(rig.effects[-1]["delayed_s"], 2.5)
        self.assertFalse(rig.rules.snapshot(rig.now)["help"]["pending"])
        self.assertIsNone(rig.open_alert("help"))

    def test_double_push_raises_help_and_presses_nothing(self) -> None:
        rig = Rig(HELP_NOW)
        rig.gesture("push")
        rig.now += 1.2
        second = rig.gesture("push")
        self.assertEqual(rig.results(second), [("push", "help", "second push")])
        self.assertEqual(rig.alert_changes("help"), [("help", "raised")])
        rig.wait(6)
        self.assertEqual(rig.acts(), [], "a double push must never also press the push key")
        alert = rig.open_alert("help")
        self.assertEqual((alert["level"], alert["state"], alert["title"]), ("alert", "active", "Help requested"))
        self.assertEqual(alert["since"], T0)                       # the first push
        self.assertIn("Help requested", rig.timeline())

    def test_the_same_physical_push_reported_twice_is_not_help(self) -> None:
        rig = Rig(HELP_NOW)
        rig.gesture("push")
        rig.now += 0.68                                            # the hand coming back: the typical repeat in the recordings
        self.assertEqual(rig.results(rig.gesture("push")), [("push", "ignored", "same push")])
        self.assertIsNone(rig.open_alert("help"))
        rig.wait(2.5)
        self.assertEqual(rig.acts(), [("push", "space")], "one push, one key press, no help")
        self.assertIsNone(rig.open_alert("help"))

    def test_a_quick_second_push_still_counts_through_its_return_stroke(self) -> None:
        rig = Rig(HELP_NOW)
        rig.gesture("push")
        rig.now += 0.8                                             # second push started early: still inside the same-push gap
        self.assertEqual(rig.results(rig.gesture("push")), [("push", "ignored", "same push")])
        rig.now += 0.7                                             # ... but its own return stroke lands at 1.5 s
        rig.gesture("push")
        self.assertIsNotNone(rig.open_alert("help"))

    def test_second_push_after_the_window_is_a_new_single_push(self) -> None:
        rig = Rig()
        rig.gesture("push")
        rig.wait(3.0)
        self.assertEqual(rig.acts(), [("push", "space")])
        second = rig.gesture("push")
        self.assertEqual(rig.results(second), [("push", "held", "waiting for a second push")])
        self.assertIsNone(rig.open_alert("help"))
        rig.wait(3.0)
        self.assertEqual(rig.acts(), [("push", "space"), ("push", "space")])

    def test_late_second_push_without_a_tick_in_between(self) -> None:
        rig = Rig()
        rig.gesture("push")
        rig.now += 4.0                                             # no tick ran: the old push must still act, in order
        out = rig.gesture("push")
        self.assertEqual(rig.results(out), [("push", "act", None), ("push", "held", "waiting for a second push")])
        self.assertIsNone(rig.open_alert("help"))

    def test_instant_timing_acts_once_then_help(self) -> None:
        rig = Rig({"monitoring": {"help": {"push_timing": "instant", "confirm_s": 0}}})
        self.assertEqual(rig.acts(rig.gesture("push")), [("push", "space")])
        rig.now += 1.0
        rig.gesture("push")
        rig.wait(6)
        self.assertEqual(rig.acts(), [("push", "space")], "exactly one key press, from the first push")
        self.assertIsNotNone(rig.open_alert("help"))

    def test_help_works_with_gesture_control_off(self) -> None:
        rig = Rig({"control": {"enabled": False}, **HELP_NOW})
        self.assertEqual(rig.results(rig.gesture("push")), [("push", "armed", "gesture control is off")])
        rig.now += 1.5
        rig.gesture("push")
        self.assertIsNotNone(rig.open_alert("help"))
        self.assertEqual(rig.acts(), [])

    def test_help_works_right_after_a_swipe(self) -> None:
        rig = Rig(HELP_NOW)
        rig.gesture("swipe_right")
        rig.now += 0.4                                             # inside the cooldown: the push presses nothing ...
        self.assertEqual(rig.results(rig.gesture("push")), [("push", "ignored", "too soon")])
        rig.now += 1.0                                             # ... but it still counts as the first of a pair
        rig.gesture("push")
        self.assertIsNotNone(rig.open_alert("help"))
        self.assertEqual(rig.acts(), [("swipe_right", "right")])

    def test_no_help_when_the_help_gesture_or_monitoring_is_off(self) -> None:
        for patch in (NO_HELP, {"monitoring": {"enabled": False}}):
            rig = Rig(patch)
            rig.gesture("push")
            rig.now += 2.0
            rig.gesture("push")
            rig.wait(4)
            self.assertIsNone(rig.open_alert("help"), patch)
            self.assertEqual(rig.acts(), [("push", "space"), ("push", "space")])

    def test_a_swipe_inside_the_window_lets_the_held_push_act_first(self) -> None:
        rig = Rig()
        rig.gesture("push")
        rig.now += 1.0
        out = rig.gesture("swipe_right")
        self.assertEqual(rig.acts(out), [("push", "space"), ("swipe_right", "right")])
        rig.now += 1.0                                             # that push was used as a key press: it cannot pair up any more
        rig.gesture("push")
        self.assertIsNone(rig.open_alert("help"))

    def test_open_help_alert_pauses_control_and_another_push_asks_again(self) -> None:
        rig = Rig(HELP_NOW)
        rig.gesture("push"); rig.now += 1.0; rig.gesture("push")
        rig.now += 2.0
        self.assertEqual(rig.results(rig.gesture("swipe_right")), [("swipe_right", "paused", "help alert is open")])
        rig.now += 2.0
        again = rig.gesture("push")
        self.assertEqual(rig.alert_changes("help", again), [("help", "reminder")])
        self.assertEqual(rig.open_alert("help")["count"], 2)
        self.assertIn("Help requested again", rig.timeline())
        self.assertEqual(rig.acts(), [])

        rig.now += 5.0
        acked = rig.keep(rig.rules.acknowledge(rig.now))
        self.assertEqual(rig.alert_changes("help", acked), [("help", "acknowledged")])
        self.assertEqual(rig.open_alert("help")["ack_s"], 9.0)
        self.assertTrue(any(t.startswith("Help requested: acknowledged after 9 s") for t in rig.timeline()), rig.timeline())
        rig.now += 1.0
        self.assertEqual(rig.acts(rig.gesture("swipe_right")), [("swipe_right", "right")])
        gone = rig.wait(8)                                         # the answered request leaves the screen by itself
        self.assertEqual(rig.alert_changes("help", gone), [("help", "dismissed")])
        self.assertIsNone(rig.open_alert("help"))

    def test_acknowledge_by_id(self) -> None:
        rig = Rig(HELP_NOW)
        rig.gesture("push"); rig.now += 1.0; rig.gesture("push")
        self.assertEqual(rig.rules.acknowledge(rig.now, "no-such-id"), [])
        self.assertEqual(rig.open_alert("help")["state"], "active")
        rig.keep(rig.rules.acknowledge(rig.now, rig.open_alert("help")["id"]))
        self.assertEqual(rig.open_alert("help")["state"], "acknowledged")

    def test_unanswered_help_reminds_at_the_repeat_interval(self) -> None:
        rig = Rig({"alerts": {"repeat_s": 20}, **HELP_NOW})
        rig.gesture("push"); rig.now += 1.0; rig.gesture("push")
        rig.wait(65)
        self.assertEqual(rig.alert_changes("help").count(("help", "reminder")), 3)
        quiet = Rig({"alerts": {"repeat_s": 0}, **HELP_NOW})
        quiet.gesture("push"); quiet.now += 1.0; quiet.gesture("push")
        quiet.wait(65)
        self.assertEqual(quiet.alert_changes("help"), [("help", "raised")])

    def test_a_new_request_replaces_an_answered_one_still_on_screen(self) -> None:
        rig = Rig(HELP_NOW)
        rig.gesture("push"); rig.now += 1.0; rig.gesture("push")
        rig.now += 1.0
        rig.keep(rig.rules.acknowledge(rig.now))
        rig.now += 2.0
        rig.gesture("push"); rig.now += 1.0
        out = rig.gesture("push")
        self.assertEqual(rig.alert_changes("help", out), [("help", "dismissed"), ("help", "raised")])
        self.assertEqual(rig.open_alert("help")["id"], "help-2")

    def test_help_waits_on_screen_before_it_goes_out(self) -> None:
        rig = Rig()                                                # default: 3 s to cancel
        rig.gesture("push"); rig.now += 1.2
        out = rig.gesture("push")
        self.assertEqual(rig.alert_changes("help", out), [("help", "pending")])
        alert = rig.open_alert("help")
        self.assertEqual((alert["state"], alert["active_at"]), ("pending", None))
        self.assertAlmostEqual(alert["escalate_at"], rig.now + 3.0)
        self.assertIn("Help gesture seen", rig.timeline())
        rig.wait(2.75)
        self.assertEqual(rig.open_alert("help")["state"], "pending")
        out = rig.wait(0.5)
        self.assertEqual(rig.alert_changes("help", out), [("help", "raised")])
        alert = rig.open_alert("help")
        self.assertEqual(alert["state"], "active")
        self.assertIn("Help requested", rig.timeline())
        rig.now += 4.0
        rig.keep(rig.rules.acknowledge(rig.now))
        self.assertAlmostEqual(rig.open_alert("help")["ack_s"], 4.0, delta=0.3)   # counted from when it went out
        self.assertEqual(rig.acts(), [])

    def test_cancel_on_screen_drops_a_pending_request(self) -> None:
        rig = Rig()
        rig.gesture("push"); rig.now += 1.2; rig.gesture("push")
        rig.now += 1.0
        out = rig.keep(rig.rules.cancel_help(rig.now))
        self.assertEqual(rig.alert_changes("help", out), [("help", "dismissed")])
        self.assertEqual(out[0]["alert"]["state"], "cancelled")
        self.assertIsNone(rig.open_alert("help"))
        self.assertIn("Help requested: cancelled on screen", rig.timeline())
        later = rig.wait(10)
        self.assertEqual(rig.alert_changes("help", later), [])
        self.assertEqual(rig.acts(rig.gesture("swipe_right")), [("swipe_right", "right")])
        self.assertEqual(rig.rules.cancel_help(rig.now), [], "nothing to cancel once it is gone")

    def test_a_hand_can_send_a_pending_request_early_but_never_cancel_it(self) -> None:
        rig = Rig()
        rig.gesture("push"); rig.now += 1.2; rig.gesture("push")
        rig.now += 0.6
        self.assertEqual(rig.results(rig.gesture("swipe_left")), [("swipe_left", "paused", "help request pending")])
        self.assertEqual(rig.open_alert("help")["state"], "pending")
        rig.now += 0.6                                             # 1.2 s after the request: another push
        out = rig.gesture("push")
        self.assertEqual(rig.alert_changes("help", out), [("help", "raised")])
        self.assertEqual(rig.open_alert("help")["state"], "active")

    def test_acknowledge_does_not_touch_a_pending_request(self) -> None:
        rig = Rig()
        rig.gesture("push"); rig.now += 1.2; rig.gesture("push")
        self.assertEqual(rig.rules.acknowledge(rig.now), [])
        self.assertEqual(rig.open_alert("help")["state"], "pending")

    def test_turning_the_help_gesture_off_cancels_a_pending_request(self) -> None:
        rig = Rig()
        rig.gesture("push"); rig.now += 1.2; rig.gesture("push")
        out = rig.apply(NO_HELP)
        self.assertEqual(rig.alert_changes("help", out), [("help", "dismissed")])

    def test_turning_help_off_releases_a_held_push(self) -> None:
        rig = Rig()
        rig.gesture("push")
        rig.now += 0.5
        self.assertEqual(rig.acts(rig.apply(NO_HELP)), [("push", "space")])


class NoMovement(unittest.TestCase):
    def test_raises_after_the_limit_for_a_confirmed_still_person(self) -> None:
        rig = Rig()
        rig.sit(3, moving=True)                                    # they sit down ...
        rig.sit(58)
        self.assertIsNone(rig.open_alert("no_movement"), "58 s still: not yet")
        self.assertAlmostEqual(rig.rules.snapshot(rig.now)["stillness"]["still_s"], 58.0, delta=1.0)
        rig.sit(3)
        alert = rig.open_alert("no_movement")
        self.assertIsNotNone(alert)
        self.assertEqual((alert["level"], alert["state"], alert["title"]), ("attention", "active", "No movement"))
        self.assertAlmostEqual(alert["since"], T0 + 3, delta=0.01)   # the last movement
        self.assertIn("Breathing is not confirmed", alert["detail"])
        self.assertTrue(any(t.startswith("No movement for 1 min") for t in rig.timeline()), rig.timeline())

    def test_movement_and_gestures_keep_it_quiet(self) -> None:
        rig = Rig(NO_HELP)
        for _ in range(4):
            rig.sit(40)
            rig.sit(1, moving=True)                                # a shift in the chair
        for _ in range(4):
            rig.sit(40)
            rig.gesture("swipe_right")
        self.assertEqual(rig.alert_changes("no_movement"), [])

    def test_tracked_breathing_counts_as_a_sign_of_life_by_default(self) -> None:
        rig = Rig()
        rig.sit(120, breathing={"bpm": 15, "confidence": 0.9, "state": "tracking"})
        self.assertEqual(rig.alert_changes("no_movement"), [])
        rig.sit(70, breathing={"bpm": None, "confidence": 0.1, "state": "acquiring"})   # the reading is lost
        self.assertIsNotNone(rig.open_alert("no_movement"))

    def test_demo_setting_alerts_on_any_stillness_and_says_breathing_is_seen(self) -> None:
        rig = Rig({"monitoring": {"no_movement": {"seconds": 15, "breathing_counts": False}}})
        rig.sit(14, breathing={"bpm": 15, "confidence": 0.9, "state": "tracking"})
        self.assertIsNone(rig.open_alert("no_movement"))
        rig.sit(2, breathing={"bpm": 15, "confidence": 0.9, "state": "tracking"})
        alert = rig.open_alert("no_movement")
        self.assertIsNotNone(alert)
        self.assertIn("Breathing is still detected", alert["detail"])
        self.assertTrue(alert["breathing_detected"])

    def test_a_faint_unconfirmed_presence_never_raises(self) -> None:
        rig = Rig({"monitoring": {"no_movement": {"seconds": 15}}})
        rig.sit(120, confidence=0.05)                              # what a noisy empty room looks like
        self.assertEqual(rig.alert_changes("no_movement"), [])
        self.assertIsNone(rig.rules.snapshot(rig.now)["stillness"]["still_s"])
        self.assertNotIn("Someone is here", rig.timeline())

    def test_resolves_when_they_move_then_leaves_the_screen(self) -> None:
        rig = Rig({"monitoring": {"no_movement": {"seconds": 15}}})
        rig.sit(17)
        self.assertIsNotNone(rig.open_alert("no_movement"))
        out = rig.sit(1, moving=True)
        self.assertEqual(rig.alert_changes("no_movement", out), [("no_movement", "resolved")])
        alert = rig.open_alert("no_movement")
        self.assertEqual((alert["state"], alert["resolution"]), ("resolved", "movement seen again"))
        self.assertTrue(any(t == "No movement: movement seen again" for t in rig.timeline()), rig.timeline())
        gone = rig.sit(12) + rig.sit(9, moving=True)                # visible for 20 s so a caregiver still sees it happened
        self.assertEqual(rig.alert_changes("no_movement", gone), [("no_movement", "dismissed")])

    def test_acknowledged_while_still_stays_quiet_until_movement(self) -> None:
        rig = Rig({"monitoring": {"no_movement": {"seconds": 15}}})
        rig.sit(17)
        acked = rig.keep(rig.rules.acknowledge(rig.now))
        self.assertEqual(rig.alert_changes("no_movement", acked), [("no_movement", "acknowledged")])
        more = rig.sit(60)
        self.assertEqual(rig.alert_changes("no_movement", more), [], "no second alert for the same stillness")
        self.assertEqual(rig.open_alert("no_movement")["state"], "acknowledged")
        out = rig.sit(1, moving=True)
        self.assertEqual(rig.alert_changes("no_movement", out), [("no_movement", "dismissed")])
        self.assertIsNone(rig.open_alert("no_movement"))

    def test_acknowledging_a_resolved_alert_puts_it_away(self) -> None:
        rig = Rig({"monitoring": {"no_movement": {"seconds": 15}}})
        rig.sit(17); rig.sit(1, moving=True)
        out = rig.keep(rig.rules.acknowledge(rig.now))
        self.assertEqual(rig.alert_changes("no_movement", out), [("no_movement", "dismissed")])
        self.assertIsNone(rig.open_alert("no_movement"))

    def test_a_second_spell_of_stillness_raises_again(self) -> None:
        rig = Rig({"monitoring": {"no_movement": {"seconds": 15}}})
        rig.sit(17); rig.sit(1, moving=True); rig.sit(17)
        self.assertEqual(rig.alert_changes("no_movement"),
                         [("no_movement", "raised"), ("no_movement", "resolved"), ("no_movement", "dismissed"), ("no_movement", "raised")])
        self.assertEqual(rig.open_alert("no_movement")["state"], "active")

    def test_leaving_resolves_it(self) -> None:
        rig = Rig({"monitoring": {"no_movement": {"seconds": 15}}})
        rig.sit(17)
        rig.now += 1
        out = rig.presence("empty", range_m=None)
        self.assertEqual(rig.alert_changes("no_movement", out), [("no_movement", "resolved")])
        self.assertEqual(rig.open_alert("no_movement")["resolution"], "nobody in range any more")
        self.assertIn("Room is empty", rig.timeline())

    def test_getting_up_and_walking_resolves_it(self) -> None:
        rig = Rig({"monitoring": {"no_movement": {"seconds": 15}}})
        rig.sit(17)
        out = rig.sit(1, state="entering", moving=True)
        self.assertEqual(rig.alert_changes("no_movement", out), [("no_movement", "resolved")])

    def test_sensor_outage_and_calibration_are_not_stillness(self) -> None:
        rig = Rig({"monitoring": {"no_movement": {"seconds": 15}}})
        rig.sit(10)
        rig.keep(rig.rules.on_link(False, rig.now))
        rig.wait(30)                                               # the engine is away for 30 s
        self.assertEqual(rig.alert_changes("no_movement"), [])
        self.assertIn("Sensor offline", rig.timeline())
        rig.keep(rig.rules.on_link(True, rig.now))
        self.assertIn("Sensor connected", rig.timeline())
        rig.sit(12)
        self.assertEqual(rig.alert_changes("no_movement"), [], "the count restarts when the sensor is back")
        rig.sit(5)
        self.assertIsNotNone(rig.open_alert("no_movement"))

        cal = Rig({"monitoring": {"no_movement": {"seconds": 15}}})
        cal.sit(10)
        cal.raw({"type": "status", "snr_db": 40, "calibrating": True})
        cal.sit(10)
        self.assertEqual(cal.alert_changes("no_movement"), [])
        cal.raw({"type": "status", "snr_db": 40, "calibrating": False})
        cal.sit(12)
        self.assertEqual(cal.alert_changes("no_movement"), [])
        cal.sit(5)
        self.assertIsNotNone(cal.open_alert("no_movement"))
        self.assertIn("Calibration complete", cal.timeline())

    def test_outage_resolves_an_open_alert(self) -> None:
        rig = Rig({"monitoring": {"no_movement": {"seconds": 15}}})
        rig.sit(17)
        rig.keep(rig.rules.on_link(False, rig.now))
        out = rig.wait(5)
        self.assertEqual(rig.alert_changes("no_movement", out), [("no_movement", "resolved")])
        self.assertEqual(rig.open_alert("no_movement")["resolution"], "sensor offline")

    def test_monitoring_off_never_raises_and_resolves_an_open_one(self) -> None:
        off = Rig({"monitoring": {"enabled": False, "no_movement": {"seconds": 15}}})
        off.sit(60)
        self.assertEqual(off.alert_changes(), [])
        rig = Rig({"monitoring": {"no_movement": {"seconds": 15}}})
        rig.sit(17)
        out = rig.apply({"monitoring": {"no_movement": {"enabled": False}}})
        self.assertEqual(rig.alert_changes("no_movement", out), [("no_movement", "resolved")])

    def test_movement_caused_by_our_own_alert_sound_does_not_resolve(self) -> None:
        rig = Rig({"monitoring": {"no_movement": {"seconds": 15}}})
        rig.sit(17)
        rig.rules.sound_played(rig.now, 2.5)                       # the service played the alert sound
        out = rig.sit(1, moving=True)                              # the speaker's own burst looks like movement
        self.assertEqual(rig.alert_changes("no_movement", out), [])
        self.assertEqual(rig.open_alert("no_movement")["state"], "active")
        out = rig.sit(3, moving=True)                              # real movement after the sound
        self.assertEqual(rig.alert_changes("no_movement", out), [("no_movement", "resolved")])

    def test_presence_lines_in_the_timeline(self) -> None:
        rig = Rig()
        rig.presence("empty", range_m=None)
        rig.sit(1, state="entering", moving=True)
        rig.sit(3)
        self.assertEqual(rig.timeline().count("Someone is here"), 1)
        rig.now += 1
        rig.presence("empty", range_m=None)
        self.assertEqual(rig.timeline()[-1], "Room is empty")


class BreathRig(Rig):
    """Generates the breathing messages the web monitor's tests use (app/tests/store.test.js)."""

    def __init__(self, patch: dict[str, Any] | None = None) -> None:
        super().__init__(patch)
        self.phase = 0.0
        self.amp = 1.0
        self.hist: list[float] = []
        self.seed = 1

    def chunks(self, seconds: float, bpm: float | None = 14, conf: float = 0.9) -> None:
        """Every 0.5 s: ten new 20 Hz samples (a client that sends only what is new)."""
        for _ in range(int(seconds * 2)):
            chunk = []
            for _ in range(10):
                self.phase += 2 * math.pi * (14 / 60) / 20
                chunk.append(self.amp * math.sin(self.phase) + 0.01 * math.cos(7 * self.phase))
            self.now += 0.5
            self.raw({"type": "breathing", "bpm": bpm, "confidence": conf, "waveform": chunk, "fs": 20})
            self.raw({"type": "status", "snr_db": 55, "carrier_hz": 19000, "fps": 30, "calibrating": False})
            self.tick()

    def _noise(self) -> float:
        self.seed = (self.seed * 16807) % 2147483647
        return (self.seed / 2147483647 - 0.5) * 0.02

    def second(self, bpm: float | None, state: str, noise: bool = False, amplitude: bool = False) -> None:
        """One second the way sonar/breathing.py sends it: the whole window (up to 30 s at 10 Hz) scaled to its peak."""
        fs = 10
        for _ in range(fs):
            self.phase += 2 * math.pi * (14 / 60) / fs
            self.hist.append(self.amp * math.sin(self.phase) + (self._noise() if noise else 0.004 * math.sin(17 * self.phase)))
        del self.hist[:-30 * fs]
        peak = max(abs(v) for v in self.hist)
        msg: dict[str, Any] = {"type": "breathing", "bpm": bpm, "confidence": (0.8 if bpm else 0.2), "state": state, "fs": fs,
                               "waveform": [v / peak for v in self.hist]}
        if amplitude:
            ordered = sorted(self.hist)
            msg["amplitude_mm"] = 4 * (ordered[int(0.97 * (len(ordered) - 1) + 0.5)] - ordered[int(0.03 * (len(ordered) - 1) + 0.5)])
        self.now += 1.0
        self.raw(msg)
        for _ in range(4):
            self.tick()


class NoBreathing(unittest.TestCase):
    def test_arms_on_steady_breathing_fires_after_10_s_flat_and_clears(self) -> None:
        rig = BreathRig()
        rig.presence("present")
        rig.chunks(14)
        self.assertTrue(rig.rules.snapshot(rig.now)["breathing"]["armed"], "should arm after steady breathing while still")
        self.assertIsNone(rig.open_alert("no_breathing"))

        rig.amp = 0.02                      # breath hold: the chest stops, the estimator still says 14
        rig.chunks(6)
        self.assertIsNone(rig.open_alert("no_breathing"), "must not fire before 10 s")
        rig.chunks(6)
        alert = rig.open_alert("no_breathing")
        self.assertIsNotNone(alert, "should fire by ~10-12 s of no breathing")
        self.assertEqual((alert["level"], alert["title"]), ("alert", "No breathing movement"))
        self.assertTrue(any(t.startswith("No breathing movement for") for t in rig.timeline()))

        rig.amp = 1.0
        rig.chunks(4)
        self.assertEqual(rig.open_alert("no_breathing")["resolution"], "breathing resumed")
        self.assertIn("No breathing movement: breathing resumed", rig.timeline())

    def test_stays_quiet_while_someone_is_gesturing(self) -> None:
        rig = BreathRig(NO_HELP)
        rig.presence("present")
        rig.chunks(14)
        rig.amp = 0.02
        for _ in range(8):
            rig.gesture("push")
            rig.chunks(2)
        self.assertEqual(rig.alert_changes("no_breathing"), [], "gestures mean motion, not a breathing pause")

    def test_no_alert_for_a_person_who_never_showed_steady_breathing(self) -> None:
        rig = BreathRig()
        rig.amp = 0.02
        rig.presence("present")
        rig.chunks(20, conf=0.2)
        self.assertEqual(rig.alert_changes("no_breathing"), [])

    def test_engine_shaped_data(self) -> None:
        rig = BreathRig()
        rig.presence("present")
        for _ in range(20):
            rig.second(14, "tracking")
        self.assertTrue(rig.rules.snapshot(rig.now)["breathing"]["armed"])
        rig.amp = 0.01                      # breath hold; the estimator keeps reporting 14 for a while
        fired_at = None
        for s in range(1, 21):
            rig.second(None if s >= 9 else 14, "acquiring" if s >= 9 else "tracking")
            if rig.open_alert("no_breathing") is not None:
                fired_at = s
                break
        self.assertTrue(fired_at is not None and 9 <= fired_at <= 14, f"fired after {fired_at} s")
        rig.amp = 1.0
        for _ in range(5):
            rig.second(14, "tracking")
        self.assertIsNotNone(rig.open_alert("no_breathing")["resolved_at"], "clears once breathing resumes")

    def test_engine_reported_body_motion_suppresses_the_alert(self) -> None:
        rig = BreathRig()
        rig.presence("present")
        wave = [math.sin(i / 3) for i in range(100)]
        for _ in range(20):
            rig.now += 1.0
            rig.raw({"type": "breathing", "bpm": 14, "confidence": 0.8, "state": "tracking", "fs": 10, "waveform": wave})
            rig.tick()
        self.assertTrue(rig.rules.snapshot(rig.now)["breathing"]["armed"])
        for _ in range(15):
            rig.now += 1.0
            rig.raw({"type": "breathing", "bpm": None, "confidence": 0, "state": "moving", "fs": 10})
            rig.tick()
        self.assertEqual(rig.alert_changes("no_breathing"), [])

    def test_a_hold_longer_than_the_engine_window_stays_flagged(self) -> None:
        rig = BreathRig()
        rig.presence("present")
        for _ in range(25):
            rig.second(14, "tracking", noise=True, amplitude=True)
        rig.amp = 0.0
        for s in range(1, 46):
            rig.second(None if s >= 9 else 14, "acquiring" if s >= 9 else "tracking", noise=True, amplitude=True)
        # 45 s in: the engine's window holds nothing but noise, scaled to full height. Still an alarm, still flat.
        alert = rig.open_alert("no_breathing")
        self.assertTrue(alert is not None and alert["resolved_at"] is None, "alert must hold through a long breath-hold")
        wave = rig.rules.wave
        self.assertLess(wave.rms(wave.t[-1] - 5, wave.t[-1]), 0.2)

    def test_engine_apnea_flag_alone_needs_a_steady_rate_first(self) -> None:
        rig = BreathRig()
        rig.presence("present")
        for _ in range(30):                                        # a noisy room: "present", nothing moving, the flag raised
            rig.now += 1.0
            rig.raw({"type": "breathing", "bpm": None, "confidence": 0.0, "state": "no_signal", "apnea": True, "fs": 10, "waveform": []})
            rig.tick()
        self.assertEqual(rig.alert_changes("no_breathing"), [])

    def test_disabled_in_settings(self) -> None:
        rig = BreathRig({"monitoring": {"no_breathing": {"enabled": False}}})
        rig.presence("present")
        rig.chunks(14)
        rig.amp = 0.02
        rig.chunks(16)
        self.assertEqual(rig.alert_changes("no_breathing"), [])

    def test_breathing_measured_is_logged_once_per_visit(self) -> None:
        rig = Rig()
        rig.sit(10, breathing={"bpm": 15.4, "confidence": 0.9, "state": "tracking"})
        self.assertEqual([t for t in rig.timeline() if t.startswith("Breathing measured")], ["Breathing measured: about 15 breaths/min"])
        self.assertEqual(rig.rules.snapshot(rig.now)["breathing"]["bpm"], 15.4)
        rig.sit(8, breathing={"bpm": None, "confidence": 0.0, "state": "no_signal"})
        self.assertIsNone(rig.rules.snapshot(rig.now)["breathing"]["bpm"], "a stale rate is never shown")


class WaveMemory(unittest.TestCase):
    def test_merges_full_windows_and_chunks_without_duplicates(self) -> None:
        tl = R.WaveTimeline(60)
        fs, t = 20, 1000.0
        tl.merge([1, 2, 3, 4], fs, t + 0.15)
        tl.merge([3, 4, 5, 6], fs, t + 0.25)                       # overlapping window
        self.assertEqual(tl.v, [1, 2, 3, 4, 5, 6])
        tl.merge([7], fs, t + 0.30)                                # incremental chunk
        self.assertEqual(len(tl.v), 7)
        tl.merge([9], fs, t + 5)                                   # gap -> NaN break
        self.assertTrue(math.isnan(tl.v[-2]))

    def test_keeps_only_the_last_minute(self) -> None:
        tl = R.WaveTimeline(60)
        for k in range(100):
            tl.merge([0.0] * 10, 10, 1000.0 + k)
        self.assertLessEqual(tl.t[-1] - tl.t[0], 60.0)


class PracticeMode(unittest.TestCase):
    def test_tutorial_recognises_but_presses_and_raises_nothing(self) -> None:
        rig = Rig()
        rig.keep(rig.rules.set_practice(True, rig.now))
        self.assertEqual(rig.results(rig.gesture("swipe_right")), [("swipe_right", "practice", None)])
        rig.now += 1.5
        first = rig.gesture("push")
        self.assertEqual(rig.results(first), [("push", "practice", None)])
        self.assertNotIn("double_push", first[0])
        rig.now += 1.2
        second = rig.gesture("push")
        self.assertTrue(second[0]["double_push"])
        rig.sit(100)                                               # stillness during a lesson is not an alert either
        self.assertEqual(rig.acts(), [])
        self.assertEqual(rig.alert_changes(), [])

    def test_practice_ends_by_itself_if_the_page_goes_away(self) -> None:
        rig = Rig(NO_HELP)
        rig.keep(rig.rules.set_practice(True, rig.now))
        rig.wait(R.PRACTICE_TIMEOUT_S + 1)
        self.assertFalse(rig.rules.practice)
        self.assertEqual(rig.acts(rig.gesture("swipe_right")), [("swipe_right", "right")])

    def test_a_push_held_from_before_the_lesson_never_fires(self) -> None:
        rig = Rig()
        rig.gesture("push")
        rig.keep(rig.rules.set_practice(True, rig.now))
        rig.wait(5)
        rig.keep(rig.rules.set_practice(False, rig.now))
        rig.wait(5)
        self.assertEqual(rig.acts(), [])


class Formatting(unittest.TestCase):
    def test_durations(self) -> None:
        self.assertEqual([R.fmt_duration(x) for x in (0, 9.4, 59.6, 60, 75, 3600, 3700)],
                         ["0 s", "9 s", "1 min", "1 min", "1 min 15 s", "1 h 0 min", "1 h 1 min"])


if __name__ == "__main__":
    unittest.main()

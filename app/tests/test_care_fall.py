"""Tests for the EXPERIMENTAL fall alert: the detector (app/care/fall.py) and the alert it raises (rules.py).

What it checks
    With synthetic Doppler rows (sizes copied from real recordings; nobody was dropped for these tests):
      - a big, fast, one-way burst followed by stillness is confirmed after the stillness time, not before;
      - things that must NOT fire: a fidget, a swipe (both ways), a push and its return stroke, a gesture the
        model recognised, a burst followed by movement or by slow shifting, walking past, sitting down,
        a jump, an empty room afterwards;
      - the sensitivity setting, the distance-change vote, a lost Doppler stream, the read-out used for tuning.
    With REAL rows (fixtures/fall_real_rows.json: six windows cut from recorded MacBook sessions, nobody fell):
      - real pushes, swipes, walking in, sitting still and slow shifting raise nothing;
      - the known weak spot is pinned: a person stepping away and standing still looks like a fall.
    The alert: raised only when the feature is on, stays until acknowledged, keeps "No movement" quiet.

Run: python -m pytest app/tests/test_care_fall.py -q
Read-only: no clock, no network, no sound. What these tests cannot tell you: how a REAL fall looks to the
MacBook's speakers and microphone. No real fall has been recorded.
"""

from __future__ import annotations

import base64
import json
import statistics
import sys
import unittest
import zlib
from pathlib import Path
from typing import Any

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from care import fall as F  # noqa: E402
from care import rules as R  # noqa: E402
from care import settings as S  # noqa: E402

BINS = [-250.0 + i * 7.8125 for i in range(64)]          # the engine's Doppler bins: -250 Hz (away) .. +242 Hz (toward)
FRAME = 0.02                                             # 50 rows a second
REAL = APP / "tests" / "fixtures" / "fall_real_rows.json"


def row(*lumps: tuple[float, float, float], floor: float = 0.005) -> list[float]:
    """One Doppler row: a quiet floor plus lumps (centre Hz, half-width Hz, height 0..1), flat-topped.
    The floor is the real one: a still room reads 0.001 (median) to 0.012 (90th percentile)."""
    out = []
    for f in BINS:
        v = floor
        for centre, half, height in lumps:
            d = abs(f - centre)
            if d <= half:
                v = max(v, height)
            elif d <= half + 16:
                v = max(v, height * (1 - (d - half) / 16))
        out.append(min(1.0, v))
    return out


# Sizes below are the detector's "level" for the row; real recordings: fidget 0.1-0.2, walking 0.3-0.5, gestures 0.5.
QUIET = row()
BODY_TOWARD = row((135.0, 95.0, 0.8))                    # one way, +40 .. +230 Hz: level 0.37
BODY_AWAY = row((-135.0, 95.0, 0.8))
FIDGET = row((60.0, 30.0, 0.5))                          # level 0.09: someone shifting in a chair
SWIPE = row((120.0, 110.0, 0.9), (-120.0, 110.0, 0.9))   # a hand right at the laptop: both ways at once, level 0.8
SHIFTING = row((15.0, 8.0, 0.6), (-15.0, 8.0, 0.6))      # slow movement only: the 8-23 Hz bins


class Rig:
    def __init__(self, sensitivity: str = "medium", **config: float) -> None:
        self.det = F.FallDetector(F.for_sensitivity(config, sensitivity))
        self.now = 1000.0
        self.events: list[dict[str, Any]] = []

    def feed(self, a_row: list[float], seconds: float) -> "Rig":
        """`seconds` of the same row, delivered like the engine does: 5 rows per message, with ticks in between."""
        return self.play([a_row] * int(round(seconds / FRAME)))

    def play(self, rows: list[list[float]]) -> "Rig":
        for k in range(0, len(rows), 5):
            chunk = rows[k:k + 5]
            self.now += len(chunk) * FRAME
            self.events += self.det.on_rows(chunk, BINS, self.now, FRAME)
            self.events += self.det.tick(self.now)
        return self

    def stages(self) -> list[str]:
        return [e["stage"] for e in self.events]

    def last(self, stage: str) -> dict[str, Any]:
        return [e for e in self.events if e["stage"] == stage][-1]


class Detector(unittest.TestCase):
    def test_a_fall_is_confirmed_only_after_the_stillness_time(self) -> None:
        rig = Rig().feed(QUIET, 3).feed(BODY_TOWARD, 1.2)
        self.assertEqual(rig.stages(), [], "nothing while the burst is still going on")
        rig.feed(QUIET, 1.0)
        self.assertEqual(rig.stages(), ["candidate"])
        cand = rig.last("candidate")
        self.assertEqual((cand["direction"], cand["needs_s"]), ("toward", 15.0))
        self.assertAlmostEqual(cand["burst_level"], 0.37, delta=0.01)
        self.assertAlmostEqual(cand["duration_s"], 1.2, delta=0.1)
        self.assertGreaterEqual(cand["one_sided"], 0.95)
        rig.feed(QUIET, 13.5)
        self.assertEqual(rig.stages(), ["candidate"], "14.5 s of stillness: not yet")
        self.assertAlmostEqual(rig.det.snapshot(rig.now)["watching"]["still_s"], 14.5, delta=0.2)
        rig.feed(QUIET, 1.0)
        self.assertEqual(rig.stages(), ["candidate", "confirmed"])
        done = rig.last("confirmed")
        self.assertGreaterEqual(done["stillness_s"], 15.0)
        self.assertTrue(0.3 <= done["confidence"] <= 1.0)
        self.assertAlmostEqual(done["burst_db"], 36 * done["burst_level"], delta=0.1)
        self.assertIsNone(rig.det.snapshot(rig.now)["watching"])
        self.assertIn("possible fall", rig.det.snapshot(rig.now)["last_burst"]["outcome"])
        rig.feed(QUIET, 30)
        self.assertEqual(rig.stages(), ["candidate", "confirmed"], "one fall, one alert")
        rig.feed(FIDGET, 1.0).feed(QUIET, 5).feed(FIDGET, 1.0)
        self.assertEqual(rig.stages(), ["candidate", "confirmed", "moved"], "movement afterwards is reported once")
        self.assertAlmostEqual(rig.last("moved")["stillness_s"], 46.0, delta=0.3)       # half a second into the first fidget

    def test_away_from_the_laptop_counts_too(self) -> None:
        rig = Rig().feed(QUIET, 2).feed(BODY_AWAY, 1.0).feed(QUIET, 17)
        self.assertEqual(rig.stages(), ["candidate", "confirmed"])
        self.assertEqual(rig.last("confirmed")["direction"], "away")

    def test_a_fidget_is_too_small(self) -> None:
        rig = Rig().feed(QUIET, 2).feed(FIDGET, 0.6).feed(QUIET, 20)
        self.assertEqual(rig.stages(), [])
        self.assertFalse(rig.det.last_burst["counted"])
        self.assertIn("not big enough", rig.det.last_burst["why"])
        self.assertEqual(rig.det.last_burst["outcome"], "ignored")

    def test_a_swipe_goes_both_ways(self) -> None:
        """A hand at the laptop is BIGGER on the strip than a body across the room: size cannot reject it, direction does."""
        rig = Rig().feed(QUIET, 2).feed(SWIPE, 0.7).feed(QUIET, 20)
        self.assertEqual(rig.stages(), [])
        self.assertGreater(rig.det.last_burst["level"], 0.7)
        self.assertIn("moved both ways", rig.det.last_burst["why"])

    def test_a_push_and_its_return_stroke_is_not_a_fall(self) -> None:
        for gap in (0.2, 0.5, 0.9):                                 # real pushes: the hand comes back 0.4-1.0 s later
            rig = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 0.5).feed(QUIET, gap).feed(BODY_AWAY, 0.5).feed(QUIET, 20)
            self.assertNotIn("confirmed", rig.stages(), f"gap {gap}")
        self.assertIn("moved back the other way", rig.last("dismissed")["reason"])
        back_first = Rig().feed(QUIET, 2).feed(BODY_AWAY, 0.3).feed(QUIET, 0.5).feed(BODY_TOWARD, 0.6).feed(QUIET, 20)
        self.assertEqual(back_first.stages(), ["candidate", "dismissed"], "the look-back sees the wind-up before the burst")

    def test_a_burst_the_model_called_a_gesture_is_not_a_fall(self) -> None:
        rig = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 1.0).feed(QUIET, 0.4)
        self.assertEqual(rig.stages(), ["candidate"])
        rig.events += rig.det.on_gesture(rig.now)                   # the engine names a gesture 0.4 s after the hand stopped
        self.assertEqual(rig.stages(), ["candidate", "dismissed"])
        self.assertIn("hand gesture", rig.last("dismissed")["reason"])
        early = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 0.9)
        early.events += early.det.on_gesture(early.now)             # ... or while the burst is still open
        early.feed(QUIET, 20)
        self.assertEqual(early.stages(), [])
        self.assertIn("hand gesture", early.det.last_burst["why"])
        late = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 1.0).feed(QUIET, 6)
        late.events += late.det.on_gesture(late.now)                # a gesture 6 s later: someone is up and waving
        self.assertEqual(late.stages(), ["candidate", "dismissed"])

    def test_a_burst_followed_by_movement_is_dropped(self) -> None:
        rig = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 1.2).feed(QUIET, 3).feed(FIDGET, 1.0).feed(QUIET, 20)
        self.assertEqual(rig.stages(), ["candidate", "dismissed"])
        self.assertEqual(rig.last("dismissed")["reason"], "movement was seen again")
        blip = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 1.2).feed(QUIET, 3).feed(FIDGET, 0.3).feed(QUIET, 20)
        self.assertEqual(blip.stages(), ["candidate", "confirmed"], "a small blip under half a second is not 'moving again'")
        hand = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 1.2).feed(QUIET, 3).feed(BODY_TOWARD, 0.2).feed(QUIET, 20)
        self.assertEqual(hand.stages(), ["candidate", "dismissed"], "a short but BIG movement is")

    def test_slow_shifting_afterwards_is_not_stillness(self) -> None:
        rig = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 1.0).feed(QUIET, 2).feed(SHIFTING, 3.0).feed(QUIET, 20)
        self.assertEqual(rig.stages(), ["candidate", "dismissed"])
        self.assertIn("slow movement", rig.last("dismissed")["reason"])
        self.assertLess(Rig().feed(SHIFTING, 1).det.snapshot(0)["level"], 0.02)
        self.assertGreater(Rig().feed(SHIFTING, 1).det.snapshot(0)["slow_level"], 0.5)

    def test_walking_past_does_not_fire(self) -> None:
        there_and_back = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 1.0).feed(BODY_AWAY, 1.0).feed(QUIET, 20)
        self.assertEqual(there_and_back.stages(), [])
        self.assertIn("moved both ways", there_and_back.det.last_burst["why"])
        strolling = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 4.0).feed(QUIET, 20)
        self.assertEqual(strolling.stages(), [])
        self.assertIn("too long", strolling.det.last_burst["why"])

    def test_sitting_down_moves_both_ways(self) -> None:
        down, back = row((100.0, 80.0, 0.85)), row((-100.0, 80.0, 0.85))    # moderate (level 0.31), toward then away
        rig = Rig().feed(QUIET, 2).feed(down, 0.6).feed(back, 0.6).feed(QUIET, 20)
        self.assertEqual(rig.stages(), [])
        self.assertIn("moved both ways", rig.det.last_burst["why"])
        self.assertLess(rig.det.last_burst["one_sided"], 0.2)

    def test_a_jump_is_a_burst_then_smaller_movements(self) -> None:
        small = row((70.0, 60.0, 0.5))                              # the landing and the recovery steps
        rig = (Rig().feed(QUIET, 2).feed(BODY_TOWARD, 0.8).feed(QUIET, 0.4).feed(small, 0.24).feed(QUIET, 0.2)
               .feed(small, 0.24).feed(QUIET, 20))
        self.assertEqual(rig.stages(), ["candidate", "dismissed"])
        self.assertIn("smaller movements", rig.last("dismissed")["reason"])

    def test_one_settling_movement_right_after_is_allowed(self) -> None:
        settle = row((50.0, 40.0, 0.4))                             # the body settling within the first second
        rig = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 1.0).feed(QUIET, 0.36).feed(settle, 0.2).feed(QUIET, 16)
        self.assertEqual(rig.stages(), ["candidate", "confirmed"])

    def test_too_short_and_too_long(self) -> None:
        self.assertIn("too short", Rig().feed(QUIET, 1).feed(BODY_TOWARD, 0.2).feed(QUIET, 2).det.last_burst["why"])
        self.assertIn("too long", Rig().feed(QUIET, 1).feed(BODY_TOWARD, 3.0).feed(QUIET, 2).det.last_burst["why"])

    def test_sensitivity_scales_the_size_needed(self) -> None:
        small_body, mid_body = row((135.0, 95.0, 0.45)), row((135.0, 95.0, 0.62))      # levels 0.21 and 0.29
        self.assertEqual(Rig("medium").feed(QUIET, 1).feed(small_body, 1.0).feed(QUIET, 17).stages(), [], "0.21 is under 0.25")
        self.assertEqual(Rig("high").feed(QUIET, 1).feed(small_body, 1.0).feed(QUIET, 17).stages(), ["candidate", "confirmed"])
        self.assertEqual(Rig("medium").feed(QUIET, 1).feed(mid_body, 1.0).feed(QUIET, 17).stages(), ["candidate", "confirmed"])
        self.assertEqual(Rig("low").feed(QUIET, 1).feed(mid_body, 1.0).feed(QUIET, 17).stages(), [], "0.29 is under the low setting's 0.325")
        self.assertAlmostEqual(F.for_sensitivity({}, "high")["burst_level"], 0.25 * 0.72)
        self.assertEqual(F.for_sensitivity({"stillness_s": 10, "not_a_setting": 1}, "medium")["stillness_s"], 10.0)
        self.assertNotIn("not_a_setting", F.for_sensitivity({"not_a_setting": 1}, "medium"))

    def test_stillness_time_is_a_setting(self) -> None:
        rig = Rig(stillness_s=10).feed(QUIET, 1).feed(BODY_TOWARD, 1.0).feed(QUIET, 9.6)
        self.assertEqual(rig.stages(), ["candidate"])
        rig.feed(QUIET, 0.8)
        self.assertEqual(rig.stages(), ["candidate", "confirmed"])

    def test_a_change_in_distance_adds_confidence_but_is_not_needed(self) -> None:
        plain = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 1.0).feed(QUIET, 17)
        moved = Rig()
        for _ in range(4):
            moved.feed(QUIET, 0.5)
            moved.det.on_presence("present", 0.60, moved.now)
        moved.feed(BODY_TOWARD, 1.0)
        for _ in range(34):
            moved.feed(QUIET, 0.5)
            moved.det.on_presence("present", 1.15, moved.now)
        self.assertEqual(moved.stages(), ["candidate", "confirmed"])
        done = moved.last("confirmed")
        self.assertEqual((done["range_before_m"], done["range_after_m"]), (0.6, 1.15))
        self.assertGreater(done["confidence"], plain.last("confirmed")["confidence"])

    def test_an_empty_room_afterwards_is_someone_leaving(self) -> None:
        rig = Rig().feed(QUIET, 2).feed(BODY_AWAY, 1.5)
        for _ in range(34):
            rig.feed(QUIET, 0.5)
            rig.det.on_presence("empty", None, rig.now)
        self.assertEqual(rig.stages(), ["candidate", "dismissed"])
        self.assertIn("nobody is in view", rig.last("dismissed")["reason"])
        stays = Rig().feed(QUIET, 2).feed(BODY_AWAY, 1.5)
        for _ in range(34):
            stays.feed(QUIET, 0.5)
            stays.det.on_presence("present", 1.2, stays.now)
        self.assertEqual(stays.stages(), ["candidate", "confirmed"])

    def test_no_doppler_stream_no_claim_of_stillness(self) -> None:
        rig = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 1.0).feed(QUIET, 1.0)
        rig.now += 5.0                                              # the stream stops (engine restart, --no-spectrogram)
        rig.events += rig.det.tick(rig.now)
        self.assertEqual(rig.stages(), ["candidate", "dismissed"])
        self.assertIn("Doppler stream stopped", rig.last("dismissed")["reason"])
        self.assertEqual(rig.det.reset(rig.now), [])
        again = Rig().feed(QUIET, 2).feed(BODY_TOWARD, 1.0).feed(QUIET, 1.0)
        again.events += again.det.reset(again.now)                  # calibration, or the link dropping
        self.assertEqual(again.stages(), ["candidate", "dismissed"])

    def test_the_read_out_used_for_tuning(self) -> None:
        rig = Rig().feed(QUIET, 1)
        snap = rig.det.snapshot(rig.now)
        self.assertEqual((snap["last_burst"], snap["watching"], snap["has_rows"]), (None, None, True))
        self.assertLess(snap["level"], 0.02)
        rig.feed(BODY_TOWARD, 0.5)
        self.assertGreater(rig.det.snapshot(rig.now)["level"], 0.3)
        rig.feed(QUIET, 1)
        burst = rig.det.snapshot(rig.now)["last_burst"]
        self.assertEqual(set(burst), {"t", "duration_s", "level", "fast_level", "one_sided", "direction", "counted", "why", "outcome"})
        self.assertEqual(burst["outcome"], "waiting for stillness")
        self.assertEqual(rig.det.snapshot(rig.now)["thresholds"]["burst_level"], 0.25)
        rig.feed(FIDGET, 1.0).feed(QUIET, 1)
        self.assertIn("not big enough", rig.det.snapshot(rig.now)["last_burst"]["why"])

    def test_new_thresholds_apply_while_running(self) -> None:
        rig = Rig().feed(QUIET, 1).feed(FIDGET, 0.6).feed(QUIET, 1)
        self.assertIn("not big enough", rig.det.last_burst["why"])
        rig.det.configure(F.for_sensitivity({"burst_level": 0.06, "fast_level": 0.0}, "medium"))
        rig.feed(FIDGET, 0.6).feed(QUIET, 1)
        self.assertTrue(rig.det.last_burst["counted"])

    def test_rows_of_another_width_and_missing_bin_list(self) -> None:
        det = F.FallDetector()
        wide = [0.0] * 32
        for i in range(20, 31):
            wide[i] = 0.9                                           # the top third of a 32-bin strip = moving toward, fast
        t, out = 100.0, []
        for a_row, seconds in ((([0.0] * 32), 1.0), (wide, 1.0), (([0.0] * 32), 16.5)):
            for _ in range(int(seconds / 0.1)):
                t += 0.1
                out += det.on_rows([a_row] * 5, None, t) + det.tick(t)
        self.assertEqual([e["stage"] for e in out], ["candidate", "confirmed"])
        self.assertEqual(det.on_rows([], None, t), [])


class RealRecordings(unittest.TestCase):
    """Six windows of real engine rows from recorded MacBook sessions (four people, nobody fell).
    Cut on 2026-11-15 from sessions in mac/data by replaying the audio through sonar.pipeline and keeping the row
    the engine sends (the mean of the two tone channels), stored as row x 255. What each window holds is in its "what".
    The file is JSON (not .npz) on purpose: the repo's .gitignore drops *.npz, and this must arrive with a git pull."""

    @classmethod
    def setUpClass(cls) -> None:
        fixture = json.loads(REAL.read_text())
        n = fixture["bins"]
        cls.meta, cls.rows = {}, {}
        for w in fixture["windows"]:
            raw = zlib.decompress(base64.b64decode(w["rows_b64"]))          # one byte per bin, row after row
            cls.meta[w["name"]] = w
            cls.rows[w["name"]] = [[v / 255 for v in raw[k:k + n]] for k in range(0, len(raw), n)]

    def run_window(self, name: str, sensitivity: str = "medium", nothing_after_s: float = 0.0, **config: float) -> Rig:
        rig = Rig(sensitivity, **config).play(self.rows[name])
        return rig.feed([0.0] * 64, nothing_after_s) if nothing_after_s else rig

    def test_the_fixture_is_what_it_says(self) -> None:
        self.assertEqual(sorted(self.meta), ["gestures_run", "one_way_then_shifting", "push_and_back", "sitting_still", "stepping_away", "walk_in_and_sit"])
        self.assertEqual(len(self.rows["sitting_still"]), 1500)
        self.assertEqual(len(self.rows["sitting_still"][0]), 64)

    def test_a_still_room_really_is_dark_on_the_strip(self) -> None:
        rig = Rig()
        levels = []
        for k in range(0, 1500, 5):
            rig.play(self.rows["sitting_still"][k:k + 5])
            levels.append(rig.det.snapshot(rig.now)["level"])
        self.assertLess(statistics.median(levels), 0.01)
        self.assertEqual(rig.stages(), [])

    def test_a_real_push_and_its_return_stroke(self) -> None:
        for sensitivity in ("medium", "high"):
            rig = self.run_window("push_and_back", sensitivity, nothing_after_s=20)
            self.assertEqual(rig.stages(), ["candidate", "dismissed"], sensitivity)
            self.assertIn("moved back the other way", rig.last("dismissed")["reason"])

    def test_a_run_of_real_swipes_and_pushes(self) -> None:
        for sensitivity in ("medium", "high"):
            rig = self.run_window("gestures_run", sensitivity, nothing_after_s=20, stillness_s=5)
            self.assertNotIn("confirmed", rig.stages(), sensitivity)
        self.assertGreater(max(e["burst_level"] for e in rig.events), 0.3, "the gestures are big: it is not size that rejects them")

    def test_real_walking_in_and_sitting_down(self) -> None:
        for sensitivity in ("medium", "high"):
            self.assertNotIn("confirmed", self.run_window("walk_in_and_sit", sensitivity, stillness_s=8).stages(), sensitivity)
        self.assertEqual(self.run_window("walk_in_and_sit").stages(), [], "at the default setting it is not even a candidate")

    def test_a_real_one_way_movement_followed_by_slow_shifting(self) -> None:
        rig = self.run_window("one_way_then_shifting", stillness_s=8)
        self.assertNotIn("confirmed", rig.stages())
        self.assertTrue(any("slow movement" in (e["reason"] or "") for e in rig.events), "the fast bins are quiet here; only the slow bins show the person is moving")

    def test_known_weak_spot_stepping_away_and_standing_still_looks_like_a_fall(self) -> None:
        """Pinned on purpose. The person steps back from the laptop in 2 s and stands still for 11 s. To this sensor that is
        a one-way burst followed by stillness, which is the rule. If a change to the detector makes this pass, update the docs."""
        rig = self.run_window("stepping_away", stillness_s=8)
        self.assertEqual(rig.stages(), ["candidate", "confirmed", "moved"], "confirmed after 8 s; 'moved' when they walk back 4 s later")
        self.assertEqual(rig.last("confirmed")["direction"], "away")
        self.assertEqual(self.run_window("stepping_away").stages(), ["candidate", "dismissed"], "at the default 15 s they had walked back by then")


class Alert(unittest.TestCase):
    """rules.py: what happens with the detector's verdict."""

    CONFIRMED = {"stage": "confirmed", "burst_t": 990.0, "direction": "toward", "stillness_s": 15.0, "confidence": 0.7}

    def make(self, enabled: bool = True) -> R.CareRules:
        return R.CareRules(S.merge(S.defaults(), {"monitoring": {"fall": {"enabled": enabled}}}), 1000.0)

    def test_off_by_default(self) -> None:
        self.assertFalse(S.defaults()["monitoring"]["fall"]["enabled"])

    def test_the_settings_start_from_the_thresholds_checked_on_real_recordings(self) -> None:
        saved = S.defaults()["monitoring"]["fall"]
        for key in ("stillness_s", "burst_level", "fast_level", "one_sided", "min_s", "max_s"):
            self.assertEqual(float(saved[key]), F.DEFAULTS[key], key)
        tuned = S.merge(S.defaults(), {"monitoring": {"fall": {"min_s": 0.3, "sensitivity": "high"}}})["monitoring"]["fall"]
        self.assertEqual(F.for_sensitivity({k: v for k, v in tuned.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
                                           tuned["sensitivity"])["min_s"], 0.3)
        with self.assertRaises(S.SettingsError):
            S.merge(S.defaults(), {"monitoring": {"fall": {"sensitivity": "extreme"}}})
        self.assertEqual(self.make(enabled=False).on_fall(self.CONFIRMED, 1005.0), [])

    def test_confirmed_raises_a_red_alert_marked_experimental(self) -> None:
        rules = self.make()
        self.assertEqual(rules.on_fall({"stage": "candidate"}, 1001.0), [])
        self.assertEqual(rules.on_fall({"stage": "dismissed"}, 1002.0), [])
        out = rules.on_fall(self.CONFIRMED, 1005.0)
        alert = rules.alerts.get("fall")
        self.assertEqual((alert["kind"], alert["level"], alert["state"], alert["title"]), ("fall", "alert", "active", "Possible fall"))
        self.assertIn("Experimental", alert["detail"])
        self.assertIn("toward the laptop", alert["detail"])
        self.assertTrue(alert["experimental"])
        self.assertEqual([e["entry"]["text"] for e in out if e["effect"] == "timeline"], ["Possible fall (experimental)"])
        self.assertEqual(rules.alerts.listing()[0]["kind"], "fall")

    def test_stays_until_acknowledged_even_when_movement_returns(self) -> None:
        rules = self.make()
        rules.on_link(True, 1000.0)
        rules.on_fall(self.CONFIRMED, 1005.0)
        for t in range(1006, 1030):
            rules.on_event({"kind": "presence", "state": "present", "confidence": 0.9, "moving": True}, float(t))
            rules.tick(float(t))
        self.assertNotIn("Movement was seen", rules.alerts.get("fall")["detail"], "the engine's moving flag is not trusted for this")
        rules.on_fall({"stage": "moved"}, 1012.0)                   # the detector saw movement on the strip
        rules.on_fall({"stage": "moved"}, 1020.0)
        alert = rules.alerts.get("fall")
        self.assertEqual(alert["state"], "active", "a person must look at it; it never clears by itself")
        self.assertEqual(alert["detail"].count("Movement was seen again"), 1)
        self.assertIn("Movement was seen again 7 s later", alert["detail"])
        rules.acknowledge(1030.0)
        self.assertEqual(rules.alerts.get("fall")["state"], "acknowledged")
        gone = rules.tick(1030.0 + R.ACKED_LINGER_S + 0.1)
        self.assertEqual([(e["alert"]["kind"], e["change"]) for e in gone if e["effect"] == "alert"], [("fall", "dismissed")])

    def test_a_second_fall_while_one_is_open_reminds_instead_of_stacking(self) -> None:
        rules = self.make()
        rules.on_fall(self.CONFIRMED, 1005.0)
        out = rules.on_fall(self.CONFIRMED, 1040.0)
        self.assertEqual([(e["alert"]["count"], e["change"]) for e in out if e["effect"] == "alert"], [(2, "reminder")])

    def test_no_movement_stays_quiet_while_the_fall_alert_is_open(self) -> None:
        patch = {"monitoring": {"fall": {"enabled": True}, "no_movement": {"seconds": 15, "breathing_counts": False}}}
        rules = R.CareRules(S.merge(S.defaults(), patch), 1000.0)
        rules.on_link(True, 1000.0)
        raised = []
        for t in range(1001, 1060):
            if t == 1010:
                rules.on_fall(self.CONFIRMED, float(t))
            for e in rules.on_event({"kind": "presence", "state": "present", "confidence": 0.9, "moving": False}, float(t)) + rules.tick(float(t)):
                if e["effect"] == "alert" and e["change"] == "raised":
                    raised.append(e["alert"]["kind"])
        self.assertEqual(raised, [], "one event, one alert: the fall alert already covers the stillness")
        self.assertEqual([a["kind"] for a in rules.alerts.listing()], ["fall"])

    def test_not_raised_during_practice(self) -> None:
        rules = self.make()
        rules.set_practice(True, 1000.0)
        self.assertEqual(rules.on_fall(self.CONFIRMED, 1005.0), [])


if __name__ == "__main__":
    unittest.main()

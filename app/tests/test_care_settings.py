"""Tests for app/care/settings.py: defaults, validation, saving and the gesture-to-key mapping.

Run: python -m pytest app/tests/test_care_settings.py -q
Writes only inside a temporary folder; the real settings file is never touched.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from care import settings as S  # noqa: E402


class Defaults(unittest.TestCase):
    def test_defaults_pass_their_own_validation(self) -> None:
        self.assertEqual(S.merge(S.defaults(), S.defaults()), S.defaults())

    def test_defaults_match_the_product_decisions(self) -> None:
        d = S.defaults()
        self.assertEqual(S.mapping(d), {"swipe_left": "left", "swipe_right": "right", "push": "space"})
        self.assertEqual((d["monitoring"]["help"]["window_s"], d["monitoring"]["help"]["min_gap_s"]), (2.5, 1.0))
        self.assertEqual(d["monitoring"]["no_movement"]["seconds"], 60)
        self.assertEqual(d["monitoring"]["no_breathing"]["seconds"], 10)
        self.assertFalse(d["onboarded"])

    def test_every_preset_uses_known_gestures_and_actions(self) -> None:
        for name, preset in S.PRESETS.items():
            for gesture, action in preset["map"].items():
                self.assertIn(gesture, S.GESTURES, name)
                self.assertIn(action, S.ACTIONS, name)
        self.assertNotIn("pull", S.GESTURES)

    def test_describe_lists_what_the_settings_page_needs(self) -> None:
        info = S.describe()
        self.assertEqual(info["gestures"], ["swipe_left", "swipe_right", "push"])
        self.assertEqual([p["id"] for p in info["presets"]], ["slides", "media", "scroll", "custom"])
        self.assertEqual(next(p for p in info["presets"] if p["id"] == "scroll")["map"]["push"], "off")
        self.assertEqual(next(p for p in info["no_movement_presets"] if p["id"] == "demo")["seconds"], 15)
        json.dumps(info)                                           # must be plain JSON


class Merge(unittest.TestCase):
    def test_partial_patch_changes_only_what_it_names(self) -> None:
        out = S.merge(S.defaults(), {"control": {"preset": "media"}, "monitoring": {"no_movement": {"seconds": 15}}})
        self.assertEqual(out["control"]["preset"], "media")
        self.assertEqual(out["monitoring"]["no_movement"]["seconds"], 15)
        self.assertTrue(out["monitoring"]["no_movement"]["breathing_counts"])
        self.assertEqual(S.defaults()["control"]["preset"], "slides", "the defaults themselves must not change")

    def test_mapping_follows_preset_and_custom(self) -> None:
        media = S.merge(S.defaults(), {"control": {"preset": "media"}})
        self.assertEqual(S.mapping(media)["push"], "media_play_pause")
        custom = S.merge(S.defaults(), {"control": {"preset": "custom", "custom": {"push": "off", "swipe_left": "k"}}})
        self.assertEqual(S.mapping(custom), {"swipe_left": "k", "swipe_right": "right", "push": "off"})

    def test_bad_values_are_rejected_with_a_reason(self) -> None:
        bad = [
            ({"control": {"preset": "games"}}, "one of"),
            ({"control": {"min_confidence": 2}}, "between"),
            ({"control": {"min_confidence": "high"}}, "number"),
            ({"control": {"enabled": "yes"}}, "true or false"),
            ({"control": {"custom": {"push": "self_destruct"}}}, "unknown action"),
            ({"control": {"custom": {"pull": "left"}}}, "unknown setting"),
            ({"monitoring": {"no_movement": {"seconds": 1}}}, "between"),
            ({"monitoring": {"help": {"window_s": 60}}}, "between"),
            ({"monitoring": {"help": {"push_timing": "sometimes"}}}, "one of"),
            ({"monitoring": "on"}, "object"),
            ({"nonsense": 1}, "unknown setting"),
            ({"alerts": {"volume": True}}, "number"),
        ]
        for patch, word in bad:
            with self.subTest(patch=patch):
                with self.assertRaises(S.SettingsError) as ctx:
                    S.merge(S.defaults(), patch)
                self.assertIn(word, str(ctx.exception))
        with self.assertRaises(S.SettingsError):
            S.merge(S.defaults(), ["not", "an", "object"])        # type: ignore[arg-type]

    def test_numbers_keep_sensible_types(self) -> None:
        out = S.merge(S.defaults(), {"monitoring": {"no_movement": {"seconds": 30.0}, "help": {"window_s": 3}}})
        self.assertEqual(out["monitoring"]["no_movement"]["seconds"], 30)
        self.assertIsInstance(out["monitoring"]["no_movement"]["seconds"], int)
        self.assertIsInstance(out["monitoring"]["help"]["window_s"], float)

    def test_room_name_is_trimmed_and_never_empty(self) -> None:
        self.assertEqual(S.merge(S.defaults(), {"room": "  Bedroom 2  "})["room"], "Bedroom 2")
        self.assertEqual(S.merge(S.defaults(), {"room": "   "})["room"], "Living room")
        self.assertEqual(len(S.merge(S.defaults(), {"room": "x" * 200})["room"]), 60)


class Files(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.environ.get("SONAR_CARE_HOME")
        os.environ["SONAR_CARE_HOME"] = self.tmp.name

    def tearDown(self) -> None:
        if self.old is None:
            os.environ.pop("SONAR_CARE_HOME", None)
        else:
            os.environ["SONAR_CARE_HOME"] = self.old
        self.tmp.cleanup()

    def test_paths_follow_the_override(self) -> None:
        self.assertEqual(S.settings_path(), Path(self.tmp.name) / "settings.json")
        self.assertEqual(S.log_dir(), Path(self.tmp.name) / "logs")

    def test_missing_file_gives_defaults_and_writes_nothing(self) -> None:
        self.assertEqual(S.load(), S.defaults())
        self.assertFalse(S.settings_path().exists())

    def test_save_then_load_round_trip(self) -> None:
        changed = S.merge(S.defaults(), {"onboarded": True, "control": {"preset": "scroll"}, "room": "Ward 3"})
        path = S.save(changed)
        self.assertEqual(path, S.settings_path())
        self.assertEqual(S.load(), changed)
        self.assertEqual([p.name for p in path.parent.iterdir()], ["settings.json"], "no temp file left behind")

    def test_damaged_or_old_files_never_stop_the_app(self) -> None:
        path = S.settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ this is not json", encoding="utf-8")
        self.assertEqual(S.load(), S.defaults())
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        self.assertEqual(S.load(), S.defaults())
        # A file from another version: unknown keys and a bad value are dropped, good values are kept.
        path.write_text(json.dumps({"version": 0, "room": "Den", "control": {"preset": "media", "min_confidence": 7, "turbo": True},
                                    "legacy": {"x": 1}}), encoding="utf-8")
        loaded = S.load()
        self.assertEqual((loaded["room"], loaded["control"]["preset"]), ("Den", "media"))
        self.assertEqual(loaded["control"]["min_confidence"], 0.6)
        self.assertEqual(loaded["version"], S.SCHEMA_VERSION)
        self.assertNotIn("legacy", loaded)


if __name__ == "__main__":
    unittest.main()

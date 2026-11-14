"""Unit tests for keys_bridge.py's gesture filter and key mapping.

Run: python -m unittest discover -s app/tests -v
Read-only; presses no keys.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from keys_bridge import DEFAULT_MAP, GestureGate, parse_map  # noqa: E402


class GateTests(unittest.TestCase):
    def test_accepts_confident_gesture(self) -> None:
        self.assertIsNone(GestureGate().judge("swipe_right", 0.9, 10.0))

    def test_rejects_low_confidence(self) -> None:
        self.assertEqual(GestureGate().judge("swipe_right", 0.4, 10.0), "low confidence")

    def test_missing_confidence_is_trusted(self) -> None:
        self.assertIsNone(GestureGate().judge("push", None, 10.0))

    def test_cooldown_then_ok(self) -> None:
        g = GestureGate(cooldown_s=0.7, return_stroke_s=1.1)
        self.assertIsNone(g.judge("swipe_right", 0.9, 10.0))
        self.assertEqual(g.judge("swipe_right", 0.9, 10.3), "too soon")
        self.assertIsNone(g.judge("swipe_right", 0.9, 10.8))

    def test_return_stroke_suppressed(self) -> None:
        g = GestureGate(cooldown_s=0.7, return_stroke_s=1.1)
        self.assertIsNone(g.judge("swipe_right", 0.9, 10.0))
        self.assertEqual(g.judge("swipe_left", 0.9, 10.9), "return stroke")
        self.assertIsNone(g.judge("swipe_left", 0.9, 11.2))

    def test_unknown_label(self) -> None:
        self.assertEqual(GestureGate().judge("clap", 0.9, 1.0), "not a gesture")


class MapTests(unittest.TestCase):
    def test_default(self) -> None:
        self.assertEqual(parse_map(None), DEFAULT_MAP)

    def test_add_and_override(self) -> None:
        m = parse_map("push=b, swipe_right=page_down")
        self.assertEqual(m["push"], "b")
        self.assertEqual(m["swipe_right"], "page_down")
        self.assertEqual(m["swipe_left"], "left")

    def test_disable(self) -> None:
        self.assertNotIn("swipe_left", parse_map("swipe_left=none"))

    def test_bad_entries(self) -> None:
        with self.assertRaises(SystemExit):
            parse_map("clap=b")
        with self.assertRaises(SystemExit):
            parse_map("push")


if __name__ == "__main__":
    unittest.main()

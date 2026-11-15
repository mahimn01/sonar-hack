"""Checks app/adapter.py against the shared fixtures (the JS twin is checked by adapter.test.js).

Run: python -m unittest discover -s app/tests -v
Read-only; touches nothing outside memory.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from adapter import adapt  # noqa: E402

FIXTURES = json.loads((APP / "tests" / "fixtures" / "engine_events.json").read_text(encoding="utf-8"))


def check(event: dict, expect: dict) -> list[str]:
    """Return a list of mismatches between one event and its expected subset."""
    problems = []
    for key, want in expect.items():
        if key.endswith("_len"):
            field = key[:-4]
            got = len(event.get(field) or [])
        elif key == "rows_shape":
            got = [len(event["rows"]), len(event["rows"][0])]
        elif key == "rows_in_01":
            got = all(0.0 <= v <= 1.0 for row in event["rows"] for v in row)
        else:
            got = event.get(key, "<missing>")
        if isinstance(want, dict) or (isinstance(want, list) and any(not isinstance(w, (int, float)) or isinstance(w, bool) for w in want)):
            if got != want:                       # a nested object or a list of names: compared exactly
                problems.append(f"{key}: got {got!r}, want {want!r}")
        elif isinstance(want, float) and isinstance(got, (int, float)) and not isinstance(got, bool):
            if not math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-6):
                problems.append(f"{key}: got {got!r}, want {want!r}")
        elif isinstance(want, list) and isinstance(got, list):
            if len(got) != len(want) or any(not math.isclose(float(g), float(w)) for g, w in zip(got, want)):
                problems.append(f"{key}: got {got!r}, want {want!r}")
        elif got != want and not (isinstance(want, (int, float)) and isinstance(got, (int, float)) and got == want):
            problems.append(f"{key}: got {got!r}, want {want!r}")
    return problems


class AdapterFixtures(unittest.TestCase):
    def test_all_cases(self) -> None:
        for case in FIXTURES["cases"]:
            with self.subTest(case["name"]):
                raw = case["raw_text"] if "raw_text" in case else case["raw"]
                events = adapt(raw, FIXTURES["now"])
                self.assertEqual(len(events), len(case["expect"]), f"event count for {case['name']}: {events}")
                for got, want in zip(events, case["expect"]):
                    self.assertEqual(check(got, want), [], f"{case['name']}: {got}")

    def test_bytes_input(self) -> None:
        events = adapt(b'{"type":"gesture","label":"push","confidence":0.9}', 1794600000.0)
        self.assertEqual(events[0]["label"], "push")


if __name__ == "__main__":
    unittest.main()

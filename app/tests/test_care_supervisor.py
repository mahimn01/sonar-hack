"""Tests for app/care/supervisor.py (the engine process) and app/care/menubar.py's menu model.

What it checks
    - which command starts the engine in each mode, and that the calm50 model is preferred;
    - an engine already listening on the port is attached to, never started twice, never stopped by us;
    - our own engine: started, seen running once its port answers, stopped cleanly;
    - a crash is noticed, the last error line is kept, and it is restarted; repeated instant crashes give up;
    - the menu shows the right title, lines and items for each situation.

Run: python -m pytest app/tests/test_care_supervisor.py -q        (about 10 s)
Starts only tiny stand-in "engines" (a Python one-liner that listens on a free local port) and stops them.
Writes only inside a temporary folder. No audio, no real engine.
"""

from __future__ import annotations

import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from care import menubar as M  # noqa: E402
from care import supervisor as V  # noqa: E402

# A stand-in engine: listens on the port, prints a line, then (optionally) dies after a delay.
FAKE_ENGINE = r"""
import socket, sys, time
port, lifetime, message = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
if lifetime == 0:
    print(message, flush=True); sys.exit(2)
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", port)); s.listen(5)
print("fake engine listening", flush=True)
end = time.time() + lifetime if lifetime > 0 else None
s.settimeout(0.1)
while end is None or time.time() < end:
    try:
        c, _ = s.accept(); c.close()
    except socket.timeout:
        pass
print(message, flush=True); sys.exit(3)
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Commands(unittest.TestCase):
    def test_live_uses_the_mac_script_with_the_calm50_model(self) -> None:
        cmd, env, model = V.engine_command("live", 8765)
        self.assertEqual(cmd, [str(V.REPO / "scripts" / "run_best_mac.sh"), "--port", "8765"])
        self.assertEqual((env, model), ({"SONAR_MODEL_DIR": "models-candidate-mac-calm50"}, "models-candidate-mac-calm50"))

    def test_sim_adds_the_simulator_flags(self) -> None:
        cmd, env, _ = V.engine_command("sim", 8771, scene="hall", extra=["-v"])
        self.assertEqual(cmd[1:], ["--port", "8771", "--backend", "sim", "--min-peak-db", "3", "--scene", "hall", "-v"])
        self.assertEqual(env["SONAR_MODEL_DIR"], "models-candidate-mac-calm50")

    def test_mock_runs_the_mock_engine(self) -> None:
        cmd, env, model = V.engine_command("mock", 8772, python="py")
        self.assertEqual(cmd, ["py", str(V.REPO / "app" / "mock_engine.py"), "--port", "8772", "--scenario", "care", "--quiet"])
        self.assertEqual(env, {})

    def test_falls_back_to_the_release_model_and_says_when_none_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "scripts").mkdir()
            (repo / "scripts" / "run_best_mac.sh").write_text("#!/bin/sh\n")
            with self.assertRaises(ValueError) as ctx:
                V.engine_command("live", 8765, repo=repo)
            self.assertIn("no gesture model found", str(ctx.exception))
            (repo / "models-release-mac-recall").mkdir()
            (repo / "models-release-mac-recall" / "gesture.onnx").write_bytes(b"x")
            self.assertEqual(V.engine_command("live", 8765, repo=repo)[2], "models-release-mac-recall")
            (repo / "models-candidate-mac-calm50").mkdir()
            (repo / "models-candidate-mac-calm50" / "gesture.json").write_text("{}")     # folder without the weight file
            self.assertEqual(V.pick_model(repo).name, "models-release-mac-recall")
        with self.assertRaises(ValueError):
            V.engine_command("turbo", 1)


class Process(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.port = free_port()
        self.sups: list[V.EngineSupervisor] = []

    def tearDown(self) -> None:
        for sup in self.sups:
            sup.shutdown(timeout=4)
        self.tmp.cleanup()

    def make(self, lifetime: float = -1, message: str = "bye") -> V.EngineSupervisor:
        script = Path(self.tmp.name) / "fake_engine.py"
        script.write_text(FAKE_ENGINE)
        sup = V.EngineSupervisor("live", self.port, Path(self.tmp.name) / "engine.log", repo=Path(self.tmp.name),
                                 command=[sys.executable, str(script), str(self.port), str(lifetime), message])
        sup.backoff_s = (0.1, 0.1, 0.1, 0.1, 0.1)
        self.sups.append(sup)
        return sup

    def until(self, sup: V.EngineSupervisor, state: str, timeout: float = 8.0) -> None:
        end = time.time() + timeout
        seen = []
        while time.time() < end:
            sup.poll()
            if sup.state not in seen:
                seen.append(sup.state)
            if sup.state == state:
                return
            time.sleep(0.05)
        self.fail(f"never reached {state!r}; saw {seen}; last error {sup.last_error!r}")

    def test_start_run_stop(self) -> None:
        sup = self.make()
        self.assertEqual(sup.info()["state"], "stopped")
        sup.start()
        self.assertEqual(sup.state, "starting")
        self.until(sup, "running")
        info = sup.info()
        self.assertTrue(info["managed"] and info["pid"] and info["url"] == f"ws://127.0.0.1:{self.port}")
        self.assertTrue(V.port_open(self.port))
        sup.start()                                        # asking again while it runs changes nothing
        self.assertEqual((sup.state, sup.pid), ("running", info["pid"]))
        sup.stop()
        self.until(sup, "stopped")
        self.assertFalse(V.port_open(self.port))
        self.assertIsNone(sup.pid)
        self.assertIn("starting:", (Path(self.tmp.name) / "engine.log").read_text())

    def test_attaches_to_an_engine_that_is_already_there_and_never_stops_it(self) -> None:
        with socket.socket() as other:                     # somebody else's engine
            other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            other.bind(("127.0.0.1", self.port))
            other.listen(5)
            sup = self.make()
            sup.start()
            self.assertEqual((sup.state, sup.pid, sup.info()["managed"]), ("attached", None, False))
            sup.stop()
            sup.poll()
            self.assertEqual(sup.state, "attached")
            self.assertTrue(V.port_open(self.port), "an engine we did not start must be left alone")
            sup.start()
        # That engine went away while we still want one: start our own.
        sup._checked_at = 0
        self.until(sup, "running")
        self.assertTrue(sup.info()["managed"])

    def test_an_engine_left_behind_by_an_earlier_run_is_adopted_and_can_be_stopped(self) -> None:
        first = self.make()
        first.start()
        self.until(first, "running")
        pid = first.pid
        # The app is killed hard: its engine keeps running and engine.pid still names it. A new app start:
        second = self.make()
        second.start()
        info = second.info()
        self.assertEqual((second.state, info["pid"], info["managed"]), ("attached", pid, True))
        self.assertEqual(info["model"], "left running by an earlier start")
        second.stop()
        # In real life the earlier app is gone and the system collects the dead engine. Here the "earlier app"
        # lives in this test process, so collect it by hand, or it lingers as a zombie that still looks alive.
        proc = first._proc
        end = time.time() + 8
        while time.time() < end and second.state != "stopped":
            proc.poll()
            second.poll()
            time.sleep(0.05)
        self.assertEqual(second.state, "stopped")
        self.assertFalse(V.port_open(self.port), "an engine this app started earlier must be stoppable")
        self.assertFalse((Path(self.tmp.name) / "engine.pid").exists())

    def test_a_stale_pid_file_is_ignored(self) -> None:
        (Path(self.tmp.name) / "engine.pid").write_text(f"999999 {self.port}\n")
        with socket.socket() as other:
            other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            other.bind(("127.0.0.1", self.port))
            other.listen(5)
            sup = self.make()
            sup.start()
            self.assertEqual((sup.state, sup.info()["managed"], sup.pid), ("attached", False, None))

    def test_crash_is_restarted_and_the_last_error_is_kept(self) -> None:
        sup = self.make(lifetime=0.6, message="ERROR: could not open the audio devices")
        sup.start()
        self.until(sup, "running")
        self.until(sup, "restarting")
        self.assertEqual(sup.last_error, "ERROR: could not open the audio devices")
        self.until(sup, "running")
        self.assertGreaterEqual(sup.restarts, 1)

    def test_an_engine_that_dies_at_once_every_time_gives_up(self) -> None:
        sup = self.make(lifetime=0, message="ERROR: models/gesture.onnx is missing")
        sup.start()
        self.until(sup, "failed", timeout=15)
        self.assertEqual(sup.last_error, "ERROR: models/gesture.onnx is missing")
        self.assertEqual(sup.restarts, V.MAX_QUICK_DEATHS - 1)
        before = sup.restarts
        for _ in range(10):
            sup.poll()
            time.sleep(0.02)
        self.assertEqual((sup.state, sup.restarts), ("failed", before), "it must stop trying until someone presses Start")
        sup.start()                                        # Start again: one more round of attempts
        self.assertEqual(sup.state, "starting")

    def test_restart_brings_up_a_new_process(self) -> None:
        sup = self.make()
        sup.start()
        self.until(sup, "running")
        first = sup.pid
        sup.restart()
        self.until(sup, "running")
        self.assertNotEqual(sup.pid, first)

    def test_a_bad_command_fails_with_a_reason(self) -> None:
        sup = V.EngineSupervisor("live", self.port, Path(self.tmp.name) / "engine.log", repo=Path(self.tmp.name),
                                 command=[str(Path(self.tmp.name) / "no-such-program")])
        self.sups.append(sup)
        sup.start()
        self.assertEqual(sup.state, "failed")
        self.assertIn("could not start the engine", sup.last_error)


def state(**over: object) -> dict:
    base = {
        "t": 1000.0, "dry_run": False, "practice": False,
        "engine": {"managed": True, "state": "running", "retry_in_s": None, "last_error": None},
        "link": {"health": "live"},
        "sonar": {"snr_db": 43.4, "signal": "good", "calibrating": False, "source": "live"},
        "presence": {"state": "present", "confirmed": True, "moving": False},
        "breathing": {"bpm": 15.2},
        "stillness": {"still_s": 42.0},
        "control": {"enabled": True, "preset": "slides", "keys": {"mode": "live"}},
        "monitoring": {"enabled": True},
        "alerts": [],
        "last_gesture": {"gesture": "swipe_right", "result": "sent", "action_label": "Right arrow", "t": 988.0},
    }
    for key, value in over.items():
        base[key] = {**base[key], **value} if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return base


class MenuModel(unittest.TestCase):
    def test_everything_fine(self) -> None:
        m = M.menu_model(state())
        self.assertEqual((m["title"], m["level"]), ("◉ Sonar", "live"))
        self.assertEqual(m["lines"], ["Sonar live · signal good (43 dB)", "Presence: someone here · still 42 s",
                                      "Breathing: 15/min (early estimate)", "Last gesture: Swipe right, Right arrow · 12 s ago"])
        self.assertEqual((m["control_on"], m["monitoring_on"], m["preset"]), (True, True, "slides"))
        self.assertEqual((m["engine_item"], m["engine_action"]), ("Stop engine", "stop"))
        self.assertFalse(m["can_ack"] or m["can_cancel_help"] or m["needs_access"])
        self.assertEqual(m["notes"], [])

    def test_states_of_the_sensor(self) -> None:
        self.assertEqual(M.menu_model(state(sonar={"calibrating": True}))["title"], "◌ Calibrating")
        weak = M.menu_model(state(sonar={"snr_db": 18.2, "signal": "weak"}))
        self.assertEqual((weak["title"], weak["lines"][0]), ("◍ Weak signal", "Weak signal · 18 dB"))
        off = M.menu_model(state(link={"health": "offline"}))
        self.assertEqual((off["title"], off["lines"][0], off["can_calibrate"]), ("○ Sonar off", "Sensor offline", False))
        self.assertEqual(len(off["lines"]), 2, "no presence or breathing lines without the sensor")
        sim = M.menu_model(state(sonar={"source": "sim"}))
        self.assertTrue(sim["lines"][0].endswith("· simulated"))
        self.assertEqual(M.menu_model({"starting": True})["title"], "◌ Sonar")

    def test_engine_process_states(self) -> None:
        failed = M.menu_model(state(engine={"state": "failed", "last_error": "ERROR: could not open the audio devices"}, link={"health": "offline"}))
        self.assertEqual(failed["lines"][:2], ["Engine could not start", "ERROR: could not open the audio devices"])
        self.assertEqual((failed["engine_item"], failed["engine_action"]), ("Start engine", "start"))
        again = M.menu_model(state(engine={"state": "restarting", "retry_in_s": 4.2}, link={"health": "offline"}))
        self.assertEqual(again["lines"][0], "Engine restarting in 4 s")
        attached = M.menu_model(state(engine={"managed": False, "state": "attached"}))
        self.assertEqual((attached["engine_item"], attached["engine_action"]), ("Engine: started elsewhere", None))

    def test_presence_and_breathing_wording(self) -> None:
        self.assertIn("Presence: nobody in range", M.menu_model(state(presence={"state": "empty"}))["lines"])
        self.assertIn("Presence: someone arriving", M.menu_model(state(presence={"state": "entering"}))["lines"])
        self.assertIn("Presence: checking (weak sign of someone)", M.menu_model(state(presence={"confirmed": False}))["lines"])
        self.assertIn("Presence: someone here · moving", M.menu_model(state(presence={"moving": True}, stillness={"still_s": 0.4}))["lines"])
        self.assertIn("Breathing: — (no confident reading)", M.menu_model(state(breathing={"bpm": None}))["lines"])
        self.assertIn("Last gesture: none yet", M.menu_model(state(last_gesture=None))["lines"])
        held = M.menu_model(state(last_gesture={"gesture": "push", "result": "held", "action_label": "Space", "t": 999.5}))
        self.assertIn("Last gesture: Push, Space (waiting for a second push) · just now", held["lines"])

    def test_alerts_take_over_the_title(self) -> None:
        pending = {"kind": "help", "level": "alert", "state": "pending", "title": "Help requested", "escalate_at": 1002.4}
        m = M.menu_model(state(alerts=[pending]))
        self.assertEqual((m["title"], m["level"], m["can_cancel_help"], m["can_ack"]), ("▲ Help in 2 s", "alert", True, False))
        active = {"kind": "help", "level": "alert", "state": "active", "title": "Help requested"}
        still = {"kind": "no_movement", "level": "attention", "state": "active", "title": "No movement"}
        m = M.menu_model(state(alerts=[active, still]))
        self.assertEqual((m["title"], m["can_ack"], m["can_cancel_help"]), ("▲ HELP", True, False))
        self.assertEqual(m["lines"][-2:], ["▲ Help requested", "⚠ No movement"])
        self.assertEqual(M.menu_model(state(alerts=[still]))["title"], "▲ No movement")
        resolved = dict(still, state="resolved")
        self.assertEqual(M.menu_model(state(alerts=[resolved]))["title"], "◉ Sonar", "a finished alert does not shout")

    def test_notes(self) -> None:
        self.assertEqual(M.menu_model(state(dry_run=True))["notes"], ["Dry run: no keys are pressed"])
        blocked = M.menu_model(state(control={"keys": {"mode": "blocked"}}))
        self.assertTrue(blocked["needs_access"])
        self.assertIn("Key presses are blocked: allow in Accessibility…", blocked["notes"])
        self.assertIn("Practice mode: nothing is pressed", M.menu_model(state(practice=True))["notes"])


if __name__ == "__main__":
    unittest.main()

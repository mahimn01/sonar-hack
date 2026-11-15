"""End-to-end test of the Sonar Care service against the mock engine.

What it checks (a real engine process, the real service, the real web server; nothing mocked in between)
    - engine events reach the service and the dashboard stream (the relay);
    - a gesture becomes the mapped key press, in dry-run mode: logged, nothing pressed;
    - two pushes raise "help": first a cancellable countdown, then the alert, then acknowledge with time-to-acknowledge;
    - sitting still raises "No movement", and a movement resolves it;
    - the experimental fall alert: off by default; switched on, a one-way burst followed by stillness raises "Possible fall";
    - settings changes are validated, saved to disk and pushed to open pages;
    - the tutorial's practice mode presses nothing;
    - the API and the relay refuse requests from other websites.

Run: python -m pytest app/tests/test_care_service.py -q        (about 25 s; needs: pip install -r app/requirements.txt)
Starts the mock engine and the service on free local ports and stops them afterwards. Writes only inside a
temporary folder. Presses no keys, shows no notification, plays no sound (dry run + quiet).
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

HAVE_WS = importlib.util.find_spec("websockets") is not None
if HAVE_WS:
    import inspect

    from websockets.exceptions import InvalidStatus
    from websockets.sync.client import connect as _connect

    from care import settings as S
    from care.api import make_server, serve_in_thread
    from care.service import Options, Service

    def connect(url: str, **kwargs: Any) -> Any:
        """A blocking WebSocket client that never goes through a system proxy."""
        if "proxy" in inspect.signature(_connect).parameters:
            kwargs["proxy"] = None
        return _connect(url, **kwargs)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, timeout: float = 10.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"nothing listening on {port}")


@unittest.skipUnless(HAVE_WS, "websockets not installed (pip install -r app/requirements.txt)")
class CareService(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine_port = free_port()
        cls.mock = subprocess.Popen(
            [sys.executable, str(APP / "mock_engine.py"), "--port", str(cls.engine_port), "--scenario", "quiet", "--seed", "7", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_for_port(cls.engine_port)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.mock.poll() is None:
            cls.mock.terminate()
            try:
                cls.mock.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.mock.kill()

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.old_home = os.environ.get("SONAR_CARE_HOME")
        os.environ["SONAR_CARE_HOME"] = self.tmp.name
        self.messages: list[dict[str, Any]] = []
        self.service = Service(Options(engine_url=f"ws://127.0.0.1:{self.engine_port}", port=free_port(), dry_run=True, quiet=True))
        self.service.on_message(self.messages.append)
        self.service.start()
        self.server = make_server(self.service, "127.0.0.1", self.service.opts.port)
        self.service.http_port = self.server.server_address[1]
        serve_in_thread(self.server)
        self.base = f"http://127.0.0.1:{self.service.http_port}"
        self.wait(lambda: self.service.snapshot().get("link", {}).get("health") == "live", "the engine link to go live")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.service.stop()
        if self.old_home is None:
            os.environ.pop("SONAR_CARE_HOME", None)
        else:
            os.environ["SONAR_CARE_HOME"] = self.old_home
        self.tmp.cleanup()

    # ---- helpers --------------------------------------------------------------------------------

    def wait(self, condition: Callable[[], Any], what: str, timeout: float = 8.0) -> Any:
        end = time.time() + timeout
        while time.time() < end:
            value = condition()
            if value:
                return value
            time.sleep(0.05)
        self.fail(f"timed out waiting for {what}; care messages so far: "
                  f"{[(m['type'], m.get('change') or m.get('result')) for m in self.messages if m['type'] != 'care_state'][-12:]}")

    def http(self, method: str, path: str, body: Any = None, headers: dict[str, str] | None = None) -> tuple[int, Any]:
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                raw = r.read()
                return r.status, json.loads(raw) if r.headers["Content-Type"].startswith("application/json") else raw.decode()
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read() or b"{}")

    def engine(self, **cmd: Any) -> None:
        """Send a test command to the mock engine through the relay, the way a page would."""
        with connect(f"ws://127.0.0.1:{self.service.relay_port}") as ws:
            ws.send(json.dumps(cmd))
            time.sleep(0.05)

    def care(self, kind: str, **match: Any) -> list[dict[str, Any]]:
        return [m for m in self.messages if m["type"] == kind and all(m.get(k) == v for k, v in match.items())]

    def alerts(self, kind: str, change: str) -> list[dict[str, Any]]:
        return [m["alert"] for m in self.messages if m["type"] == "care_alert" and m["change"] == change and m["alert"]["kind"] == kind]

    # ---- tests ----------------------------------------------------------------------------------

    def test_state_boot_and_pages(self) -> None:
        status, state = self.http("GET", "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual((state["link"]["health"], state["dry_run"], state["mode"]), ("live", True, "attach"))
        self.assertEqual(state["control"]["mapping"]["swipe_right"], {"action": "right", "label": "Right arrow"})
        self.wait(lambda: self.service.snapshot()["presence"]["state"] == "present", "presence from the engine")
        self.wait(lambda: self.service.snapshot()["sonar"]["snr_db"], "a status message")
        self.assertEqual(self.service.snapshot()["sonar"]["source"], "mock")

        status, boot = self.http("GET", "/api/boot.js")
        self.assertEqual(status, 200)
        self.assertIn(f'"relay":"ws://127.0.0.1:{self.service.relay_port}"', boot)
        for page in ("/", "/care.html", "/presenter.html", "/monitor.html", "/js/adapter.js", "/js/care.js", "/js/care-dashboard.js",
                     "/js/care-settings.js", "/js/care-setup.js", "/js/care-api.js", "/css/care.css", "/css/tokens.css"):
            self.assertEqual(self.http("GET", page)[0], 200, page)
        self.assertEqual(self.http("GET", "/api/nope")[0], 404)

    def test_a_page_gets_a_snapshot_then_the_live_stream(self) -> None:
        with connect(f"ws://127.0.0.1:{self.service.relay_port}") as ws:
            first = [json.loads(ws.recv(timeout=3))]
            while first[-1]["type"] != "care_state":
                first.append(json.loads(ws.recv(timeout=3)))
            kinds = [m["type"] for m in first]
            # First who the engine is (it says so once per connection, so the service repeats it), then our own snapshot.
            self.assertEqual(kinds, ["hello", "care_hello", "care_settings", "care_state"])
            self.assertEqual(first[1]["timeline"][0]["text"], "Sonar Care started")
            self.assertEqual(first[2]["settings"], S.defaults())
            seen = set(kinds)
            end = time.time() + 4
            while time.time() < end and not {"presence", "breathing", "status", "spectrogram", "care_state"} <= seen:
                seen.add(json.loads(ws.recv(timeout=3))["type"])
            self.assertTrue({"presence", "breathing", "status", "spectrogram", "care_state"} <= seen, seen)

    def test_gesture_becomes_a_logged_key_press(self) -> None:
        self.engine(cmd="gesture", label="swipe_right", confidence=0.93)
        action = self.wait(lambda: self.care("care_action", gesture="swipe_right"), "the swipe to be handled")[0]
        self.assertEqual((action["result"], action["action"], action["action_label"]), ("dry_run", "right", "Right arrow"))
        self.assertEqual(self.service.actions.log[-1]["detail"], "would press Right arrow")
        entry = self.wait(lambda: [m["entry"] for m in self.care("care_timeline") if m["entry"]["kind"] == "gesture"], "a timeline line")[0]
        self.assertEqual((entry["text"], entry["detail"]), ("Swipe right: Right arrow", "dry run: no key pressed"))
        self.assertEqual(self.service.snapshot()["last_gesture"]["result"], "dry_run")
        # A gesture the user is less sure about than the floor is reported but presses nothing.
        self.engine(cmd="gesture", label="swipe_left", confidence=0.41)
        low = self.wait(lambda: self.care("care_action", gesture="swipe_left"), "the unsure swipe")[0]
        self.assertEqual((low["result"], low["reason"]), ("ignored", "low confidence"))
        self.assertEqual(len(self.service.actions.log), 1)

    def test_double_push_counts_down_then_alerts_then_is_acknowledged(self) -> None:
        self.assertEqual(self.http("POST", "/api/settings", {"monitoring": {"help": {"confirm_s": 1.5}}})[0], 200)
        self.engine(cmd="gesture", label="push", confidence=0.95)
        held = self.wait(lambda: self.care("care_action", gesture="push"), "the first push")[0]
        self.assertEqual((held["result"], held["action"]), ("held", "space"))
        time.sleep(1.2)
        self.engine(cmd="gesture", label="push", confidence=0.95)
        pending = self.wait(lambda: self.alerts("help", "pending"), "the help countdown")[0]
        self.assertEqual(pending["state"], "pending")
        # the alert message goes out a moment before the state that carries it: wait for the state, do not race it
        self.wait(lambda: [x for x in self.service.snapshot()["alerts"] if x["kind"] == "help" and x["state"] == "pending"], "the state to show the countdown", timeout=1.0)
        raised = self.wait(lambda: self.alerts("help", "raised"), "the help alert", timeout=5)[0]
        self.assertEqual((raised["state"], raised["level"], raised["title"]), ("active", "alert", "Help requested"))
        self.assertEqual(self.service.notifier.sent[-1]["kind"], "help")       # the notifier was asked (quiet: nothing shown)
        self.assertEqual(self.service.actions.log, [], "a double push must not press the push key")

        time.sleep(0.6)
        status, reply = self.http("POST", "/api/ack", {"id": raised["id"]})
        self.assertEqual((status, reply), (200, {"acknowledged": 1}))
        acked = self.wait(lambda: self.alerts("help", "acknowledged"), "the acknowledgement")[0]
        self.assertGreaterEqual(acked["ack_s"], 0.5)
        lines = [m["entry"]["text"] for m in self.care("care_timeline")]
        self.assertIn("Help gesture seen", lines)
        self.assertIn("Help requested", lines)
        self.assertTrue(any(t.startswith("Help requested: acknowledged after") for t in lines), lines)

    def test_help_countdown_can_be_cancelled_on_screen(self) -> None:
        self.engine(cmd="gesture", label="push", confidence=0.95)
        self.wait(lambda: self.care("care_action", gesture="push"), "the first push")
        time.sleep(1.2)
        self.engine(cmd="gesture", label="push", confidence=0.95)
        self.wait(lambda: self.alerts("help", "pending"), "the help countdown")
        self.assertEqual(self.http("POST", "/api/help/cancel", {}), (200, {"cancelled": True}))
        gone = self.wait(lambda: self.alerts("help", "dismissed"), "the cancelled request to disappear")[0]
        self.assertEqual((gone["state"], gone["resolution"]), ("cancelled", "cancelled on screen"))
        time.sleep(3.3)
        self.assertEqual(self.alerts("help", "raised"), [])
        self.assertEqual(self.http("POST", "/api/help/cancel", {}), (200, {"cancelled": False}))
        self.assertEqual(self.service.notifier.sent, [])

    def test_stillness_raises_no_movement_and_a_gesture_resolves_it(self) -> None:
        self.http("POST", "/api/settings", {"monitoring": {"no_movement": {"seconds": 5, "breathing_counts": False}}})
        raised = self.wait(lambda: self.alerts("no_movement", "raised"), "the no-movement alert", timeout=12)[0]
        self.assertEqual((raised["level"], raised["title"]), ("attention", "No movement"))
        self.assertEqual(self.service.notifier.sent[-1]["kind"], "no_movement")
        self.engine(cmd="gesture", label="swipe_right", confidence=0.9)
        resolved = self.wait(lambda: self.alerts("no_movement", "resolved"), "the alert to resolve")[0]
        self.assertEqual(resolved["resolution"], "movement seen again")

    def test_fall_alert_is_off_until_switched_on(self) -> None:
        self.engine(cmd="fall", direction="toward", still_s=8)
        time.sleep(2.5)
        self.assertEqual(self.care("fall"), [], "the detector is not even fed while the setting is off")
        self.assertEqual(self.service.snapshot()["fall"], {"enabled": False, "sensitivity": "medium"})
        self.engine(cmd="freeze", on=False)

    def test_fall_alert_end_to_end_when_switched_on(self) -> None:
        code, body = self.http("POST", "/api/settings", {"monitoring": {"fall": {"enabled": True, "stillness_s": 5}, "no_movement": {"seconds": 5, "breathing_counts": False}}})
        self.assertEqual(code, 200, body)
        self.wait(lambda: self.service.snapshot().get("fall", {}).get("has_rows"), "Doppler rows to reach the fall detector")
        self.engine(cmd="moving", seconds=1)                             # so the stillness count starts fresh
        time.sleep(1.2)
        self.engine(cmd="fall", direction="away", still_s=12)
        cand = self.wait(lambda: self.care("fall", stage="candidate"), "the burst to be noticed", timeout=6)[0]
        self.assertEqual((cand["direction"], cand["source"], cand["experimental"]), ("away", "care", True))
        self.assertGreater(cand["burst_level"], 0.25)
        watching = self.wait(lambda: self.service.snapshot()["fall"].get("watching"), "the watching read-out")
        self.assertEqual(watching["needs_s"], 5.0)
        raised = self.wait(lambda: self.alerts("fall", "raised"), "the possible-fall alert", timeout=12)[0]
        self.assertEqual((raised["level"], raised["title"], raised["experimental"]), ("alert", "Possible fall", True))
        self.assertIn("away from the laptop", raised["detail"])
        self.assertEqual(self.service.notifier.sent[-1]["kind"], "fall")
        self.assertGreaterEqual(self.care("fall", stage="confirmed")[0]["stillness_s"], 5.0)
        self.assertIn("Possible fall (experimental)", [e["text"] for e in self.service.timeline()])
        still_open = [a["kind"] for a in self.service.snapshot()["alerts"] if a["state"] in ("active", "pending")]
        self.assertEqual(still_open, ["fall"], "one event, one alert: the fall alert covers the stillness")
        moved = self.wait(lambda: self.care("fall", stage="moved"), "movement to be seen again when the person gets up", timeout=14)[0]
        self.assertGreater(moved["stillness_s"], 10.0)
        alert = [a for a in self.service.snapshot()["alerts"] if a["kind"] == "fall"][0]
        self.assertEqual(alert["state"], "active", "it stays until a person answers it")
        self.assertIn("Movement was seen again", alert["detail"])
        code, body = self.http("POST", "/api/ack", {})
        self.assertEqual(code, 200, body)
        self.wait(lambda: self.alerts("fall", "acknowledged"), "the acknowledgement")

    def test_a_fault_in_the_fall_alert_cannot_take_the_app_down(self) -> None:
        self.http("POST", "/api/settings", {"monitoring": {"fall": {"enabled": True}}})
        self.wait(lambda: self.service.snapshot().get("fall", {}).get("has_rows"), "Doppler rows to reach the fall detector")

        def broken(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            raise ValueError("rows of the wrong shape")
        self.service._fall.on_rows = broken                               # the next Doppler message hits this
        self.wait(lambda: self.service.snapshot()["fall"].get("error"), "the fall alert to switch itself off")
        self.assertFalse(self.service.snapshot()["fall"]["enabled"])
        self.assertIn("Experimental fall alert stopped after an internal error", [e["text"] for e in self.service.timeline()])
        self.engine(cmd="gesture", label="swipe_right", confidence=0.9)   # everything else still works
        self.wait(lambda: self.care("care_action", gesture="swipe_right", result="dry_run"), "a gesture to still press its (dry-run) key")
        self.assertEqual(self.service.snapshot()["link"]["health"], "live")

    def test_settings_are_validated_saved_and_pushed(self) -> None:
        status, body = self.http("GET", "/api/settings")
        self.assertEqual(status, 200)
        self.assertEqual(body["settings"], S.defaults())
        self.assertEqual(body["meta"]["gestures"], ["swipe_left", "swipe_right", "push"])

        status, body = self.http("POST", "/api/settings", {"control": {"custom": {"push": "self_destruct"}}})
        self.assertEqual(status, 400)
        self.assertIn("unknown action", body["error"])
        self.assertEqual(self.http("POST", "/api/settings", b"{not json")[0], 400)
        self.assertFalse(S.settings_path().exists(), "a rejected change must not be saved")

        status, body = self.http("POST", "/api/settings", {"control": {"preset": "media"}, "room": "Ward 3", "onboarded": True})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(S.settings_path().read_text())["control"]["preset"], "media")
        pushed = self.wait(lambda: self.care("care_settings"), "the new settings to be pushed")[-1]
        self.assertEqual((pushed["settings"]["room"], pushed["rev"]), ("Ward 3", 1))
        state = self.service.snapshot()
        self.assertEqual((state["control"]["preset_label"], state["control"]["mapping"]["push"]["action"]), ("Media", "media_play_pause"))
        self.assertIn("Gestures now control: Media", [m["entry"]["text"] for m in self.care("care_timeline")])
        self.engine(cmd="gesture", label="swipe_right", confidence=0.9)
        self.wait(lambda: self.care("care_action", gesture="swipe_right", action="media_next"), "the new mapping to be used")

    def test_practice_mode_presses_nothing(self) -> None:
        self.assertEqual(self.http("POST", "/api/practice", {"active": True}), (200, {"practice": True}))
        self.engine(cmd="gesture", label="swipe_right", confidence=0.9)
        seen = self.wait(lambda: self.care("care_action", gesture="swipe_right"), "the practice swipe")[0]
        self.assertEqual(seen["result"], "practice")
        self.assertEqual(self.service.actions.log, [])
        self.wait(lambda: self.service.snapshot()["practice"] is True, "practice in the state")
        self.assertEqual(self.http("POST", "/api/practice", {"active": False}), (200, {"practice": False}))

    def test_calibrate_reaches_the_engine(self) -> None:
        self.assertEqual(self.http("POST", "/api/calibrate", {}), (200, {"sent": True}))
        self.wait(lambda: self.service.snapshot()["sonar"]["calibrating"], "the engine to report calibration")
        self.wait(lambda: "Calibration complete" in [m["entry"]["text"] for m in self.care("care_timeline")], "calibration to finish", timeout=10)

    def test_api_refuses_other_websites(self) -> None:
        evil = {"Origin": "http://evil.example"}
        self.assertEqual(self.http("POST", "/api/settings", {"control": {"enabled": False}}, evil)[0], 403)
        self.assertEqual(self.http("POST", "/api/ack", {}, evil)[0], 403)
        self.assertEqual(self.http("GET", "/api/state", headers={"Host": "evil.example"})[0], 403)      # DNS rebinding
        self.assertEqual(self.http("OPTIONS", "/api/settings")[0], 403)
        # A form post (what another site can send without asking the browser for permission) is not JSON.
        req = urllib.request.Request(self.base + "/api/settings", data=b'{"control":{"enabled":false}}', method="POST",
                                     headers={"Content-Type": "text/plain"})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 403)
        self.assertTrue(self.service.get_settings()["control"]["enabled"], "nothing may have changed")
        # Our own page is fine.
        own = {"Origin": self.base}
        self.assertEqual(self.http("POST", "/api/ack", {}, own)[0], 200)

    def test_relay_refuses_other_websites(self) -> None:
        url = f"ws://127.0.0.1:{self.service.relay_port}"
        with self.assertRaises(InvalidStatus) as ctx:
            connect(url, additional_headers={"Origin": "http://evil.example"}).close()
        self.assertEqual(ctx.exception.response.status_code, 403)
        with connect(url, additional_headers={"Origin": self.base}) as ws:
            self.assertIn(json.loads(ws.recv(timeout=3))["type"], ("hello", "care_hello"))


if __name__ == "__main__":
    unittest.main()

"""End-to-end smoke test: mock engine + static server + WebSocket client + key bridge (dry run).

What it checks
    - serve.py serves every page and asset with the right content type;
    - the mock engine streams every event kind and adapter.py parses all of it;
    - a command sent over the socket comes back as a gesture event;
    - a simulated breath-hold really flattens the waveform (what the UI's alert relies on);
    - keys_bridge.py (dry run) would press Right for swipe_right and ignores the return stroke.

Run: python -m unittest discover -s app/tests -v      (needs: pip install -r app/requirements.txt)
Starts local processes on free ports and stops them afterwards; presses no keys.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import queue
import socket
import subprocess
import sys
import threading
import time
import unittest
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from adapter import adapt  # noqa: E402

HAVE_WS = importlib.util.find_spec("websockets") is not None


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


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


async def collect(url: str, seconds: float, commands: list[dict] | None = None) -> list[dict]:
    from websockets.asyncio.client import connect
    events: list[dict] = []
    async with connect(url, proxy=None) as ws:
        for cmd in commands or []:
            await ws.send(json.dumps(cmd))
        end = time.time() + seconds
        while time.time() < end:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=max(0.05, end - time.time()))
            except asyncio.TimeoutError:
                break
            events.extend(adapt(msg))
    return events


async def send(url: str, cmd: dict) -> None:
    from websockets.asyncio.client import connect
    async with connect(url, proxy=None) as ws:
        await ws.send(json.dumps(cmd))
        await asyncio.sleep(0.1)


def rms(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


@unittest.skipUnless(HAVE_WS, "websockets not installed (pip install -r app/requirements.txt)")
class Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ws_port, cls.http_port = free_port(), free_port()
        cls.ws_url = f"ws://127.0.0.1:{cls.ws_port}"
        cls.mock = subprocess.Popen(
            [sys.executable, str(APP / "mock_engine.py"), "--port", str(cls.ws_port), "--scenario", "quiet", "--seed", "3", "--quiet"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.web = subprocess.Popen(
            [sys.executable, str(APP / "serve.py"), "--port", str(cls.http_port), "--no-browser"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wait_for_port(cls.ws_port)
        wait_for_port(cls.http_port)

    @classmethod
    def tearDownClass(cls) -> None:
        stop(cls.mock)
        stop(cls.web)

    def test_pages_and_assets_served(self) -> None:
        expected = {
            "": "text/html", "presenter.html": "text/html", "monitor.html": "text/html",
            "js/adapter.js": "text/javascript", "js/store.js": "text/javascript", "js/presenter.js": "text/javascript",
            "css/tokens.css": "text/css", "fonts/atkinson-next-latin.woff2": "font/woff2", "img/favicon.svg": "image/svg+xml",
        }
        for path, mime in expected.items():
            with self.subTest(path):
                with urllib.request.urlopen(f"http://127.0.0.1:{self.http_port}/{path}", timeout=5) as r:
                    self.assertEqual(r.status, 200)
                    self.assertTrue(r.headers["Content-Type"].startswith(mime), r.headers["Content-Type"])
                    self.assertEqual(r.headers["Cache-Control"], "no-store")
        page = urllib.request.urlopen(f"http://127.0.0.1:{self.http_port}/presenter.html", timeout=5).read().decode()
        for script in ("adapter.js", "config.js", "connection.js", "store.js", "ui.js", "widgets.js", "presenter.js"):
            self.assertIn(f"js/{script}", page)

    def test_server_survives_parallel_burst(self) -> None:
        # Regression: the stdlib's default listen backlog (5) reset most of these connections.
        import concurrent.futures as cf
        paths = ["js/adapter.js", "js/store.js", "js/ui.js", "js/widgets.js", "js/presenter.js", "css/base.css", "presenter.html"]

        def get(p: str) -> int | str:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.http_port}/{p}", timeout=5) as r:
                    r.read()
                    return r.status
            except OSError as err:
                return repr(err)
        with cf.ThreadPoolExecutor(30) as ex:
            results = list(ex.map(get, paths * 30))
        self.assertEqual([r for r in results if r != 200], [])

    def test_every_event_kind_arrives_and_parses(self) -> None:
        events = asyncio.run(collect(self.ws_url, 3.0))
        kinds = {e["kind"] for e in events}
        self.assertTrue({"hello", "status", "presence", "breathing", "spectrogram"} <= kinds, kinds)
        self.assertNotIn("unknown", kinds)
        hello = next(e for e in events if e["kind"] == "hello")
        self.assertEqual(hello["source"], "mock")
        status = [e for e in events if e["kind"] == "status"][-1]
        self.assertAlmostEqual(status["snrDb"], 55, delta=5)
        self.assertEqual(status["carriersHz"], [18500, 19000])     # pilot tones (left and right speaker), as in sonar/config.py
        self.assertEqual(status["bandHz"], [19500, 21500])         # chirp band
        self.assertEqual(status["fps"], 50)
        spec = next(e for e in events if e["kind"] == "spectrogram")
        self.assertEqual(len(spec["rows"][0]), 64)
        waves = [e for e in events if e["kind"] == "breathing" and e["waveform"]]
        self.assertTrue(waves)
        self.assertEqual(waves[0]["fs"], 10)
        self.assertEqual(waves[-1]["breathState"], "breathing")      # "tracking" from a settled person
        self.assertAlmostEqual(waves[-1]["bpm"], 14, delta=1.5)
        self.assertLessEqual(max(abs(v) for v in waves[-1]["waveform"]), 1.0)   # window scaled to its peak

    def test_command_comes_back_as_gesture(self) -> None:
        events = asyncio.run(collect(self.ws_url, 2.0, [{"cmd": "gesture", "label": "swipe_right", "confidence": 0.9}]))
        gestures = [e for e in events if e["kind"] == "gesture"]
        self.assertEqual([(g["label"], g["confidence"]) for g in gestures], [("swipe_right", 0.9)])

    def test_breath_hold_flattens_waveform(self) -> None:
        # The engine sends its whole window scaled to the window's peak, so a held breath
        # shows up as a flat tail: compare the last 3 s of the newest window before and during.
        def tail(events: list[dict]) -> list[float]:
            waves = [e for e in events if e["kind"] == "breathing" and e["waveform"]]
            return waves[-1]["waveform"][-int(3 * waves[-1]["fs"]):]
        before = tail(asyncio.run(collect(self.ws_url, 2.5)))
        asyncio.run(send(self.ws_url, {"cmd": "apnea", "on": True}))
        time.sleep(3.0)                                       # amplitude fades over ~1.5 s
        during = tail(asyncio.run(collect(self.ws_url, 2.5)))
        asyncio.run(send(self.ws_url, {"cmd": "apnea", "on": False}))
        self.assertLess(rms(during), 0.3 * rms(before), (rms(during), rms(before)))

    def test_keys_bridge_dry_run(self) -> None:
        bridge = subprocess.Popen(
            [sys.executable, "-u", str(APP / "keys_bridge.py"), "--ws", self.ws_url, "--dry-run"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines: queue.Queue[str] = queue.Queue()
        threading.Thread(target=lambda: [lines.put(line) for line in bridge.stdout], daemon=True).start()

        def wait_for(text: str, timeout: float = 6.0) -> str:
            end = time.time() + timeout
            seen = []
            while time.time() < end:
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    continue
                seen.append(line.strip())
                if text in line:
                    return line
            self.fail(f"bridge never printed {text!r}; saw {seen}")

        try:
            wait_for("connected to")
            asyncio.run(send(self.ws_url, {"cmd": "gesture", "label": "swipe_right", "confidence": 0.92}))
            wait_for("would press: right")
            # The hand coming back (opposite swipe right after) must not press Left.
            # Cooldown or return-stroke may catch it first; the exact rule is unit-tested in test_app_bridge.py.
            asyncio.run(send(self.ws_url, {"cmd": "gesture", "label": "swipe_left", "confidence": 0.92}))
            line = wait_for("swipe_left")
            self.assertIn("ignored", line)
        finally:
            stop(bridge)


if __name__ == "__main__":
    unittest.main()

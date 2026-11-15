"""supervisor.py: starts the sonar engine, watches it, and restarts it if it dies.

What it does
    * start():  if something is already listening on the engine port, ATTACH to it and start
                nothing. Otherwise start the engine as a child process, with its output going
                to a log file.
    * poll():   called a few times a second. Notices when the engine has come up (its port
                answers) or has died. A dead engine is restarted after a pause that grows
                1, 2, 4, 8, 15 s. Six quick deaths in a row = "failed": it stops trying and
                shows the engine's last error line, until someone presses Start.
    * stop():   asks the engine to stop the polite way (Ctrl+C), then insists.

    Which engine it starts:
      live   scripts/run_best_mac.sh                       (this laptop's speakers and microphone)
      sim    scripts/run_best_mac.sh --backend sim         (the simulated visitor, no sound)
      mock   app/mock_engine.py                            (scripted fake events for rehearsing the screens)
    with the calm50 model (models-candidate-mac-calm50), or models-release-mac-recall if the
    calm50 weight file is missing.

Why
    The app should be one thing to start. Nobody should need a second terminal for the engine,
    and a crash at the venue must heal by itself.

    If the app is killed hard (kill -9, a crash), its engine keeps running. The next start finds it
    on the port and recognises it by the pid it wrote down (engine.pid next to the log), so
    Stop and Quit still work on it. An engine somebody else started is used but never stopped.

What it changes
    Starts and stops one child process. Appends to the engine log file and writes engine.pid.
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
MODEL_DIRS = ("models-candidate-mac-calm50", "models-release-mac-recall")     # first one with gesture.onnx wins

BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 15.0)
QUICK_DEATH_S = 10.0          # an engine that lived less than this "died quickly"
MAX_QUICK_DEATHS = 6          # that many in a row = give up until someone presses Start
HEALTHY_S = 30.0              # an engine that lived this long resets the restart pause
START_TIMEOUT_S = 40.0        # the port must answer within this (loading the model takes a few seconds)
STOP_GRACE_S = 3.0            # time for a polite stop before insisting
LOG_MAX_BYTES = 5 * 1024 * 1024


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.25) -> bool:
    """Is something listening on this port?"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def pick_model(repo: Path = REPO) -> Path | None:
    """The best model folder that actually has its weight file."""
    for name in MODEL_DIRS:
        if (repo / name / "gesture.onnx").is_file():
            return repo / name
    return None


def engine_command(mode: str, port: int, repo: Path = REPO, scene: str | None = None, extra: list[str] | None = None,
                   python: str = sys.executable) -> tuple[list[str], dict[str, str], str]:
    """(command, extra environment, model name) for the engine in this mode. Raises ValueError with a readable reason."""
    extra = list(extra or [])
    if mode == "mock":
        cmd = [python, str(repo / "app" / "mock_engine.py"), "--port", str(port), "--scenario", scene or "care", "--quiet"]
        return cmd + extra, {}, "mock engine: scripted data"
    if mode not in ("live", "sim"):
        raise ValueError(f"unknown engine mode {mode!r}")
    model = pick_model(repo)
    if model is None:
        raise ValueError("no gesture model found: expected " + " or ".join(f"{m}/gesture.onnx" for m in MODEL_DIRS))
    script = repo / "scripts" / "run_best_mac.sh"
    if not script.is_file():
        raise ValueError(f"{script} is missing")
    cmd = [str(script), "--port", str(port)]
    if mode == "sim":
        # The simulated hand is far quieter than a real one; without this the model's "too weak" gate drops every gesture.
        cmd += ["--backend", "sim", "--min-peak-db", "3"]
        if scene:
            cmd += ["--scene", scene]
    return cmd + extra, {"SONAR_MODEL_DIR": model.name}, model.name


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _default_sigint() -> None:
    """Runs in the child just before the engine starts: make Ctrl+C work there.

    A program started in the background by a shell inherits "ignore Ctrl+C", and Python then never
    installs its own handler. Our polite stop IS a Ctrl+C, so put the default back.
    """
    signal.signal(signal.SIGINT, signal.SIG_DFL)


class EngineSupervisor:
    """One engine process. Call start(), then poll() a few times a second, then shutdown() at the end."""

    def __init__(self, mode: str, port: int, log_path: Path, repo: Path = REPO, scene: str | None = None,
                 extra: list[str] | None = None, command: list[str] | None = None, env: dict[str, str] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.mode = mode
        self.port = port
        self.log_path = log_path
        self.repo = repo
        self._scene, self._extra = scene, extra
        self._command, self._env = command, env or {}      # a fixed command (tests) instead of engine_command()
        self._now = clock
        self.backoff_s = BACKOFF_S    # pauses before restart attempts (tests shorten them)
        self.state = "stopped"        # stopped | starting | running | attached | restarting | stopping | failed
        self.model = ""
        self.pid: int | None = None
        self.restarts = 0             # times the engine was restarted after dying
        self.last_error: str | None = None
        self.wanted = False           # the user wants an engine running
        self._proc: subprocess.Popen[bytes] | None = None
        self._adopted: int | None = None   # pid of an engine a previous run of this app started and left behind
        self.pid_path = log_path.with_name("engine.pid")
        self._log: Any = None
        self._started_at = 0.0
        self._up_at = 0.0
        self._quick_deaths = 0
        self._retry_at = 0.0
        self._stop_deadline = 0.0
        self._checked_at = 0.0
        atexit.register(self._kill_on_exit)

    # ---- what the UI reads ---------------------------------------------------------------------------

    def info(self) -> dict[str, Any]:
        now = self._now()
        return {
            "managed": self.state != "attached" or self._adopted is not None, "state": self.state, "mode": self.mode, "port": self.port,
            "url": f"ws://127.0.0.1:{self.port}", "pid": self.pid, "model": self.model, "restarts": self.restarts,
            "last_error": self.last_error, "log": str(self.log_path),
            "uptime_s": round(now - self._up_at, 0) if self.state == "running" and self._up_at else None,
            "retry_in_s": round(max(0.0, self._retry_at - now), 1) if self.state == "restarting" else None,
        }

    # ---- commands --------------------------------------------------------------------------------------

    def start(self) -> None:
        """Make sure an engine is running: attach to one that is there, else start ours."""
        self.wanted = True
        self._quick_deaths = 0
        self.last_error = None
        if self._proc is not None and self._proc.poll() is None:
            return
        if port_open(self.port):
            self.state = "attached"
            self._adopted = self._read_pid()
            self.pid = self._adopted
            self.model = "left running by an earlier start" if self._adopted else "started elsewhere"
            return
        self._spawn()

    def stop(self) -> None:
        """Stop our engine (politely first). An engine we only attached to is left alone."""
        self.wanted = False
        if self._adopted is not None:                      # ours from an earlier run: we may stop it
            self._signal(signal.SIGINT)
            self.state = "stopping"
            self._stop_deadline = self._now() + STOP_GRACE_S
            return
        if self._proc is None or self._proc.poll() is not None:
            if self.state != "attached":
                self.state = "stopped"
            return
        self._signal(signal.SIGINT)
        self.state = "stopping"
        self._stop_deadline = self._now() + STOP_GRACE_S

    def restart(self) -> None:
        self.stop()
        self.wanted = True            # poll() starts it again once the old one is gone

    def shutdown(self, timeout: float = 6.0) -> None:
        """Stop our engine and wait for it (app exit)."""
        self.wanted = False
        if self._adopted is not None:
            self._signal(signal.SIGINT)
            end = time.monotonic() + timeout
            while time.monotonic() < end and _alive(self._adopted):
                time.sleep(0.1)
            if _alive(self._adopted):
                self._signal(signal.SIGKILL)
            self._adopted = None
        proc = self._proc
        if proc is not None and proc.poll() is None:
            self._signal(signal.SIGINT)
            try:
                proc.wait(timeout=timeout / 2)
            except subprocess.TimeoutExpired:
                self._signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=timeout / 2)
                except subprocess.TimeoutExpired:
                    self._signal(signal.SIGKILL)
                    proc.wait(timeout=2)
        self._reap()
        if self.state != "attached":
            self.state = "stopped"

    # ---- the watch loop ----------------------------------------------------------------------------------

    def poll(self) -> bool:
        """Notice changes. Returns True if the state changed (so the caller can refresh the screen)."""
        before = (self.state, self.pid, self.last_error)
        now = self._now()
        proc = self._proc

        if self._adopted is not None and self.state == "stopping":
            if not _alive(self._adopted):
                self._adopted, self.pid, self.state = None, None, "stopped"
                self._forget_pid()
            elif now > self._stop_deadline + STOP_GRACE_S:
                self._signal(signal.SIGKILL)
            elif now > self._stop_deadline:
                self._signal(signal.SIGTERM)
        elif self.state == "attached":
            if now - self._checked_at >= 2.0:
                self._checked_at = now
                if not port_open(self.port):               # the engine we attached to is gone
                    self.state, self._adopted, self.pid = "stopped", None, None
                    if self.wanted:
                        self._spawn()
        elif proc is not None and proc.poll() is not None:
            self._on_exit(proc.returncode, now)
        elif self.state == "starting":
            if now - self._checked_at >= 0.5:
                self._checked_at = now
                if port_open(self.port):
                    self.state = "running"
                    self._up_at = now
                elif now - self._started_at > START_TIMEOUT_S:
                    self.last_error = f"the engine did not open port {self.port} within {START_TIMEOUT_S:.0f} s"
                    self._signal(signal.SIGTERM)
        elif self.state == "running":
            if self._quick_deaths and now - self._up_at >= HEALTHY_S:
                self._quick_deaths = 0                     # it has been fine for a while: forget the earlier crashes
        elif self.state == "stopping":
            if now > self._stop_deadline + STOP_GRACE_S:
                self._signal(signal.SIGKILL)
            elif now > self._stop_deadline:
                self._signal(signal.SIGTERM)
        elif self.state == "restarting":
            if now >= self._retry_at:
                self._spawn()
        elif self.state == "stopped" and self.wanted:      # restart(): the old process is gone, start the new one
            self.start()
        return before != (self.state, self.pid, self.last_error)

    # ---- internals -----------------------------------------------------------------------------------------

    def _spawn(self) -> None:
        try:
            if self._command is not None:
                cmd, env, self.model = list(self._command), dict(self._env), "custom"
            else:
                cmd, env, self.model = engine_command(self.mode, self.port, self.repo, self._scene, self._extra)
        except ValueError as err:
            self.state, self.last_error, self.pid = "failed", str(err), None
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists() and self.log_path.stat().st_size > LOG_MAX_BYTES:
                self.log_path.replace(self.log_path.with_suffix(self.log_path.suffix + ".1"))
            self._log = self.log_path.open("ab")
            self._log.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} starting: {' '.join(cmd)}\n".encode())
            self._log.flush()
            self._proc = subprocess.Popen(cmd, cwd=str(self.repo), env={**os.environ, "PYTHONUNBUFFERED": "1", **env},
                                          stdin=subprocess.DEVNULL, stdout=self._log, stderr=subprocess.STDOUT,
                                          start_new_session=True,      # its own group: our Ctrl+C does not reach it twice
                                          preexec_fn=_default_sigint)
        except OSError as err:
            self.state, self.last_error, self.pid = "failed", f"could not start the engine: {err}", None
            return
        self.pid = self._proc.pid
        try:
            self.pid_path.write_text(f"{self.pid} {self.port}\n")
        except OSError:
            pass                                           # only costs the "adopt after a hard kill" nicety
        self.state = "starting"
        self._started_at = self._checked_at = self._now()
        self._up_at = 0.0

    def _on_exit(self, code: int | None, now: float) -> None:
        lived = now - self._started_at
        self._reap()
        if self.state == "stopping" or not self.wanted:
            self.state = "stopped"
            if self.wanted:                                # restart(): come straight back
                self.start()
            return
        how = f"killed by signal {-code}" if code is not None and code < 0 else f"exit code {code}"
        self.last_error = self._last_log_line() or f"the engine stopped unexpectedly ({how})"
        self._quick_deaths = self._quick_deaths + 1 if lived < QUICK_DEATH_S else 0
        if self._quick_deaths >= MAX_QUICK_DEATHS:
            self.state = "failed"
            return
        self.restarts += 1
        self.state = "restarting"
        self._retry_at = now + self.backoff_s[max(0, min(self._quick_deaths, len(self.backoff_s)) - 1)]

    def _read_pid(self) -> int | None:
        """The pid an earlier run of this app wrote down, if that process is still alive and was for this port."""
        try:
            pid_text, port_text = self.pid_path.read_text().split()[:2]
            pid = int(pid_text)
        except (OSError, ValueError):
            return None
        return pid if int(port_text) == self.port and _alive(pid) else None

    def _forget_pid(self) -> None:
        try:
            self.pid_path.unlink()
        except OSError:
            pass

    def _reap(self) -> None:
        self._proc = None
        self.pid = None
        self._forget_pid()
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None

    def _signal(self, sig: int) -> None:
        if self._adopted is not None:
            try:
                os.killpg(self._adopted, sig)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(self._adopted, sig)
                except (ProcessLookupError, PermissionError):
                    pass
            return
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)                       # the engine is its own process group leader
        except (ProcessLookupError, PermissionError):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass

    def _last_log_line(self) -> str | None:
        """The engine's last ERROR line from this run, for the UI (None if it logged none)."""
        try:
            with self.log_path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            return None
        # Only this run's lines (after the last "==== ... starting" header), and only ones that look like an error.
        run = tail.rsplit("\n====", 1)[-1]
        errors = [ln.strip() for ln in run.splitlines() if "ERROR" in ln or "Error" in ln or "error:" in ln]
        return errors[-1][:240] if errors else None

    def _kill_on_exit(self) -> None:
        """Last resort at interpreter exit: never leave our engine playing tones with nobody watching it."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

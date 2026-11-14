#!/usr/bin/env python3
"""train_4090.py — the whole gesture-model training in one command (RTX 4090 box, or any machine).

What it does, in order (every step prints the exact command it runs, so any step can be re-run by hand):
  1. checks the machine: Python, PyTorch, the GPU (CUDA), onnxruntime;
  2. makes the synthetic training set with datagen (reused on the next run if the generator code
     and settings did not change). On a CUDA GPU the clips are computed on the GPU (--gen-device,
     default: the training device when it is CUDA) after scripts/check_gpu_parity.py has shown
     that the GPU makes the same pictures as the CPU code; --gen-device cpu is the old path;
  3. if real recordings exist under --data (default data/): lists every session folder,
     skips broken or simulator ones, cuts each session into training pictures once
     (in parallel, cached per session), measures which way round the laptop's left/right
     channels are, and makes extra synthetic gestures laid over the real "nobody gesturing"
     recordings (the scene and breathing sessions);
  4. pre-trains the model on the synthetic set (once per --widths value; cached);
  5. fine-tunes on the real recordings from those weights, scoring each run on people the
     model has never seen (--val-person; picked automatically if not given), for every
     width x seed of the sweep, and keeps the run with the best held-out accuracy;
  6. trains that winning setting once more on everyone (unless --no-refit) and exports
     gesture.onnx + gesture.json (what the engine loads) + gesture.pt into models/ and
     models-release/, with TRAINING_RUN.json describing the run and its honest score;
  7. checks that the engine loads the model and runs it on the simulated demo visitor.

How to run (from the repo root; scripts/train_4090.sh and .ps1 do exactly this):
    python scripts/train_4090.py                            # full run on the GPU (CUDA)
    python scripts/train_4090.py --sweep                    # + widths 16,24,32 x seeds 0,1,2
    python scripts/train_4090.py --device cpu --size tiny   # few-minute smoke test, no GPU needed
    python scripts/train_4090.py --gen-device cpu           # make the data on the CPU cores as before

What it changes: writes train_out/ (intermediate files, caches and a log of every run),
models/ and models-release/. It only reads data/ and never opens an audio device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TextIO

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
LABELS = "swipe_left,swipe_right,push"          # DATA_STRATEGY.md section 6: three gestures, pull is left out
RELEASE_FILES = ("gesture.onnx", "gesture.json", "gesture.pt", "metrics.json", "confusion.txt")
BACKGROUND_SCRIPTS = ("scene", "breathing")      # recordings with nobody gesturing: real backgrounds for datagen

# Sizes. "full" is ~3x the 650-per-class set of the Mac (12,000 clips, ~20,000 pictures).
PRESETS: dict[str, dict[str, Any]] = {
    "full": {"per_class": 2000, "negatives": 2.0, "bg_per_class": 1000, "pretrain_epochs": 60, "finetune_epochs": 60},
    "small": {"per_class": 300, "negatives": 2.0, "bg_per_class": 150, "pretrain_epochs": 20, "finetune_epochs": 30},
    "tiny": {"per_class": 6, "negatives": 1.0, "bg_per_class": 4, "pretrain_epochs": 2, "finetune_epochs": 2},
}
# Code whose changes make cached synthetic data / session pictures stale.
FEATURE_CODE = ["sonar/config.py", "sonar/tx.py", "sonar/dsp.py", "sonar/frontend.py", "sonar/gestures.py",
                "sonar/features.py", "sonar/pipeline.py", "sonar/presence.py", "sonar/breathing.py", "sonar/dataset.py"]
GENERATOR_CODE = ["datagen/adapter.py", "datagen/background.py", "datagen/generate.py", "datagen/hardware.py",
                  "datagen/kinematics.py", "datagen/physics.py", "datagen/quicklook.py", "datagen/render.py",
                  "datagen/scenes.py"]
# The GPU versions of the two lists above (only part of a cache key when the GPU path is used).
GPU_CODE = ["datagen/plan.py", "datagen/render_torch.py", "datagen/gpu_pipeline.py", "sonar/features_torch.py"]
PARITY_CODE = ["datagen/parity.py", "scripts/check_gpu_parity.py"]
TRAIN_CODE = ["sonar/train.py", "sonar/model.py", "sonar/batch_augment.py"]


class StepFailed(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Small helpers: logging, running commands, caching
# ---------------------------------------------------------------------------
class Log:
    """Prints to the screen and appends the same text to the run's log.txt."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file: TextIO = open(path, "a", encoding="utf-8")

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        self.file.write(text + "\n")
        self.file.flush()

    def raw(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        self.file.write(text)


def show(cmd: list[str]) -> str:
    """A command as you would type it in this OS's shell."""
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


def child_env(single_thread_math: bool = False) -> dict[str, str]:
    # PYTORCH_ENABLE_MPS_FALLBACK: on a Mac, an operation Apple's GPU backend lacks runs on the CPU instead of failing
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTORCH_ENABLE_MPS_FALLBACK="1")
    if single_thread_math:   # one process per core already: stop maths libraries starting threads of their own
        env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    return env


def run(cmd: list[str], log: Log, env: dict[str, str] | None = None) -> float:
    """Run one step, streaming its output to the screen and the log. Returns the seconds it took."""
    log(f"\n$ {show(cmd)}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env or child_env(),
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        log.raw(line)
    code = proc.wait()
    if code != 0:
        raise StepFailed(f"this step failed with exit code {code}: {show(cmd)}")
    return time.time() - t0


class Job:
    """One command running in the background. Its output goes to a file and is copied into the log when it ends,
    so several jobs can run at once without their lines mixing."""

    def __init__(self, cmd: list[str], out_file: Path, env: dict[str, str] | None = None) -> None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        self.cmd, self.out_file, self.t0 = cmd, out_file, time.time()
        self._fh = open(out_file, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(cmd, cwd=REPO, stdout=self._fh, stderr=subprocess.STDOUT, env=env or child_env())

    def finish(self, log: Log) -> float:
        """Wait for it, copy its output into the log, raise StepFailed if it failed. Returns its seconds."""
        code = self.proc.wait()
        seconds = time.time() - self.t0
        self._fh.close()
        log(f"\n$ {show(self.cmd)}")
        log.raw(self.out_file.read_text(encoding="utf-8", errors="replace"))
        if code != 0:
            raise StepFailed(f"this step failed with exit code {code}: {show(self.cmd)}")
        return seconds


def gpu_env(k: int, n_gpus: int) -> dict[str, str] | None:
    """With several GPUs, training run number k sees only GPU k modulo their number."""
    if n_gpus <= 1:
        return None
    return dict(child_env(), CUDA_VISIBLE_DEVICES=str(k % n_gpus))


def run_many(cmds: list[tuple[list[str], Path]], log: Log, at_once: int, n_gpus: int = 1) -> list[float]:
    """Run commands `at_once` at a time (several small models share a GPU; with several GPUs they are dealt
    out in turn). Output appears in order."""
    if at_once <= 1:
        return [run(cmd, log) for cmd, _ in cmds]
    seconds: list[float] = []
    running: list[Job] = []
    failed: StepFailed | None = None
    pending = list(cmds)
    started = 0
    while pending or running:
        while pending and len(running) < at_once:
            cmd, out_file = pending.pop(0)
            running.append(Job(cmd, out_file, env=gpu_env(started, n_gpus)))
            started += 1
        job = running.pop(0)                                  # finish them in the order they were started
        try:
            seconds.append(job.finish(log))
        except StepFailed as exc:
            failed = failed or exc
            pending = []                                      # let the running ones end, start no more
    if failed is not None:
        raise failed
    return seconds


def code_hash(files: list[str]) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode())
        h.update((REPO / name).read_bytes())
    return h.hexdigest()


def key_of(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def cached(folder: Path) -> bool:
    return (folder / "done.json").exists()


def mark_done(folder: Path, info: dict[str, Any]) -> None:
    (folder / "done.json").write_text(json.dumps({**info, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def default_workers() -> int:
    n = os.cpu_count() or 1
    return min(n, 61) if os.name == "nt" else n


# ---------------------------------------------------------------------------
# 1. The machine
# ---------------------------------------------------------------------------
CHECK = r"""
import json, sys, platform
out = {"python": platform.python_version()}
try:
    import torch
    out["torch"] = torch.__version__
    out["torch_cuda"] = torch.version.cuda
    out["cuda"] = torch.cuda.is_available()
    if out["cuda"]:
        p = torch.cuda.get_device_properties(0)
        out.update(gpu=p.name, gpu_gb=round(p.total_memory / 2**30, 1), bf16=torch.cuda.is_bf16_supported(),
                   capability=f"{p.major}.{p.minor}", gpu_count=torch.cuda.device_count())
    out["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
except Exception as exc:
    out["torch_error"] = f"{type(exc).__name__}: {exc}"
for mod in ("onnx", "onnxruntime", "soundfile", "scipy", "numpy"):
    try:
        out[mod] = __import__(mod).__version__
    except Exception as exc:
        out[mod + "_error"] = f"{type(exc).__name__}: {exc}"
print("CHECK " + json.dumps(out))
"""


def check_machine(device: str, log: Log) -> dict[str, Any]:
    res = subprocess.run([PY, "-c", CHECK], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
                         env=child_env())
    line = next((x for x in res.stdout.splitlines() if x.startswith("CHECK ")), None)
    if line is None:
        raise StepFailed(f"could not check the Python environment:\n{res.stdout}\n{res.stderr}")
    info = json.loads(line[6:])
    info["os"] = f"{platform.system()} {platform.release()}"
    info["cpus"] = os.cpu_count()
    log(f"machine: {info['os']}, {info['cpus']} CPU threads, Python {info['python']}")
    missing = [m for m in ("torch", "onnx", "onnxruntime", "soundfile", "scipy", "numpy") if m not in info]
    if missing:
        raise StepFailed(f"missing Python packages {missing}: run the install steps in TRAINING.md (step 2-3)")
    log(f"PyTorch {info['torch']} (CUDA build {info['torch_cuda']}), onnxruntime {info['onnxruntime']}")
    if info.get("cuda"):
        log(f"GPU: {info['gpu']}, {info['gpu_gb']} GB, compute {info['capability']}, bf16 {'yes' if info['bf16'] else 'no'}"
            + (f"; {info['gpu_count']} GPUs" if info.get("gpu_count", 1) > 1 else ""))
    if device == "cuda" and not info.get("cuda"):
        raise StepFailed(
            "--device cuda, but PyTorch cannot see a CUDA GPU (torch.cuda.is_available() is False).\n"
            "  * PyTorch is the CPU build (version without '+cu...'): reinstall it from the CUDA index, TRAINING.md step 3;\n"
            "  * or the NVIDIA driver is missing / too old: `nvidia-smi` must work;\n"
            "  * for a quick test without a GPU add: --device cpu --size tiny")
    if device == "mps" and not info.get("mps"):
        raise StepFailed("--device mps, but Apple MPS is not available here")
    return info


# ---------------------------------------------------------------------------
# 2. Synthetic data
# ---------------------------------------------------------------------------
def generate(out: Path, per_class: int, negatives: float, args: argparse.Namespace, log: Log, swap: tuple[bool, bool],
             mics: int, backgrounds: list[str] | None = None, seed: int = 0, gen_batch: int = 0) -> tuple[Path, float]:
    """datagen.generate into a cached folder named after everything that shapes the data."""
    bg_ids = [(p, os.path.getsize(p), int(os.path.getmtime(p))) for p in backgrounds or []]
    on_gpu = args.gen_device != "cpu"
    how = [code_hash(GPU_CODE), args.gen_device.split(":")[0], args.gen_precision] if on_gpu else []   # CPU keys are as before
    key = key_of(code_hash(GENERATOR_CODE + FEATURE_CODE), per_class, negatives, args.profile, seed, swap, mics, bg_ids, *how)
    folder = out / key
    if cached(folder) and not args.force_synth:
        n = json.loads((folder / "done.json").read_text()).get("examples")
        log(f"\nreusing {rel(folder)} ({n} examples; generator code and settings unchanged). --force-synth makes it again.")
        return folder / "dataset.npz", 0.0
    if folder.exists():
        shutil.rmtree(folder)
    cmd = [PY, "-m", "datagen.generate", "--out", str(folder), "--per-class", str(per_class), "--negatives", str(negatives),
           "--workers", str(args.workers), "--seed", str(seed), "--profile", args.profile]
    if backgrounds:
        cmd += ["--background", ",".join(backgrounds)]
    if swap[0]:
        cmd.append("--swap-mics")
    if swap[1]:
        cmd.append("--swap-speakers")
    if mics != 2:
        cmd += ["--mics", str(mics)]
    if on_gpu:
        cmd += ["--gen-device", args.gen_device, "--gen-precision", args.gen_precision]
        if gen_batch or args.gen_batch:
            cmd += ["--gen-batch", str(gen_batch or args.gen_batch)]
        if args.gen_gpus:
            cmd += ["--gen-gpus", str(args.gen_gpus)]
    seconds = run(cmd, log, env=child_env(single_thread_math=True))
    if not (folder / "dataset.npz").exists():
        raise StepFailed(f"datagen wrote no dataset.npz into {folder}")
    import numpy as np

    with np.load(folder / "dataset.npz") as z:
        n = int(len(z["y"]))
    mark_done(folder, {"examples": n, "seconds": round(seconds, 1)})
    return folder / "dataset.npz", seconds


def parity_gate(args: argparse.Namespace, info: dict[str, Any], work: Path, log: Log) -> dict[str, Any]:
    """Refuse to make training data on a GPU that does not reproduce the CPU code's pictures.

    Runs scripts/check_gpu_parity.py (1,000 fixed clips through both paths; 100 for a tiny smoke test).
    A pass is remembered for this exact code, PyTorch, GPU and number format, so later runs do not repeat it.
    """
    clips = 100 if args.size == "tiny" else 1000
    key = key_of(code_hash(GENERATOR_CODE + FEATURE_CODE + GPU_CODE + PARITY_CODE), info.get("torch"), info.get("gpu"),
                 args.gen_device, args.gen_precision, args.profile, clips)
    marker = work / "parity" / f"passed_{key}.json"
    if marker.exists() and not args.recheck_parity:
        old = json.loads(marker.read_text())
        log(f"\nGPU parity check: passed earlier for this code on this GPU ({old.get('clips')} clips, "
            f"{rel(marker)}); --recheck-parity runs it again")
        return {"status": "passed earlier", "clips": old.get("clips"), "report": rel(marker)}
    log(f"\n=== GPU parity check: {clips} fixed clips through the CPU code and the GPU code (--skip-parity to skip)")
    report = work / "parity" / f"last_{key}.json"
    cmd = [PY, "scripts/check_gpu_parity.py", "--device", args.gen_device, "--precision", args.gen_precision,
           "--clips", str(clips), "--workers", str(args.workers), "--work", str(work / "parity"), "--data", args.data,
           "--json", str(report)]
    if args.profile != "default":
        cmd += ["--profile", args.profile]
    try:
        seconds = run(cmd, log, env=child_env(single_thread_math=True))
    except StepFailed:
        raise StepFailed(
            "the GPU does NOT make the same training pictures as the CPU code (see the table above), so nothing was trained.\n"
            "  * to train anyway with the CPU generator (slow, but the reference): add  --gen-device cpu\n"
            f"  * the full report: {rel(report)}; please send it with the log\n"
            "  * --skip-parity goes on with the GPU data regardless (not recommended)") from None
    report.replace(marker)
    return {"status": "passed", "clips": clips, "seconds": round(seconds, 1), "report": rel(marker)}


# ---------------------------------------------------------------------------
# 3. Real recordings
# ---------------------------------------------------------------------------
def inventory(root: Path, profile: str, allow_sim: bool, log: Log) -> list[dict[str, Any]]:
    """Every session folder under root, with what it holds and whether it can be used."""
    import soundfile as sf

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for meta_path in sorted(root.rglob("session.json")):
        folder = meta_path.parent
        row: dict[str, Any] = {"folder": str(folder), "session": folder.name, "use": False}
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            audio = folder / meta.get("audio", "audio.wav")
            info = sf.info(str(audio))
            row.update(session=meta.get("session_id", folder.name), person=str(meta.get("person", "unknown")),
                       script=str(meta.get("script", "?")), profile=str(meta.get("profile", "default")),
                       synthetic=bool(meta.get("synthetic", False)), seconds=round(info.frames / max(info.samplerate, 1), 1),
                       trials=sum(1 for e in meta.get("events", []) if e.get("kind") == "gesture"), audio=str(audio),
                       size=audio.stat().st_size, mtime=int(audio.stat().st_mtime), mics=int(info.channels),
                       meta_id=[meta_path.stat().st_size, int(meta_path.stat().st_mtime)],
                       device="simulator" if meta.get("synthetic") else str(meta.get("backend", {}).get("input", "unknown")))
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            row["why"] = f"cannot read it ({type(exc).__name__}: {str(exc)[:80]})"
            rows.append(row)
            continue
        if row["session"] in seen:
            row["why"] = "same session id as a folder above (a second copy)"
        elif row["seconds"] < 8:
            row["why"] = f"only {row['seconds']} s of audio (still recording, or copied while it was)"
        elif row["profile"] != profile:
            row["why"] = f"recorded with profile {row['profile']}, this run uses {profile}"
        elif row["synthetic"] and not allow_sim:
            row["why"] = "simulator recording (--allow-sim-sessions to use it)"
        else:
            row["use"] = True
            seen.add(row["session"])
        rows.append(row)
    if rows:
        log(f"\n{len(rows)} session folder(s) under {rel(root)}:")
        log(f"  {'session':<42} {'person':<12} {'script':<10} {'minutes':>7} {'trials':>6} {'mics':>4}  status")
        for r in rows:
            status = "use" if r["use"] else f"SKIP: {r['why']}"
            log(f"  {r['session'][:42]:<42} {r.get('person', '?')[:12]:<12} {r.get('script', '?'):<10} "
                f"{r.get('seconds', 0) / 60:7.1f} {r.get('trials', 0):6d} {r.get('mics', 0):4d}  {status}")
    return rows


def one_device(usable: list[dict[str, Any]], mixed_ok: bool, log: Log) -> int:
    """All recordings must come from one laptop: a model only works on the device it was trained for.

    Returns the number of microphone channels of that device.
    """
    devices: dict[tuple[str, int], int] = {}
    for r in usable:
        devices[(r["device"], r["mics"])] = devices.get((r["device"], r["mics"]), 0) + 1
    listing = "; ".join(f"{d} ({m} mic{'s' if m > 1 else ''}): {n} session(s)" for (d, m), n in devices.items())
    if len({m for _, m in devices}) > 1 or (len(devices) > 1 and not mixed_ok):
        raise StepFailed(
            f"the recordings come from more than one device: {listing}.\n"
            "  A model works only on the laptop whose recordings trained it (HP and MacBook sound different and have\n"
            "  different microphones). Put each laptop's sessions in its own folder and run with --data <that folder>.\n"
            "  Same laptop under two device names (e.g. after a driver update)? Add --mixed-devices.")
    log(f"\nrecorded on: {listing}")
    return next(iter(devices))[1]


def cut_sessions(rows: list[dict[str, Any]], out: Path, workers: int, log: Log, device: str = "cpu") -> list[Path]:
    """One .npz of training pictures per session (python -m sonar.dataset), in parallel, cached per session.

    device "cuda": one process cuts every session with the Doppler columns computed on the GPU.
    """
    out.mkdir(parents=True, exist_ok=True)
    code = code_hash(FEATURE_CODE) if device == "cpu" else key_of(code_hash(FEATURE_CODE + GPU_CODE), device.split(":")[0])
    jobs = []
    for r in rows:
        target = out / f"{r['session']}__{key_of(code, r['size'], r['mtime'], r['meta_id'])}.npz"
        jobs.append((r, target))
    todo = [(r, t) for r, t in jobs if not t.exists()]
    if device != "cpu":
        log(f"\ncutting {len(todo)} session(s) into training pictures on {device} ({len(jobs) - len(todo)} already done)")
        for a in range(0, len(todo), 30):                     # 30 per command keeps the command line short
            part = todo[a:a + 30]
            cmd = [PY, "-m", "sonar.dataset", "--device", device, "--pairs",
                   *[f"{r['folder']}::{t.with_suffix('.tmp.npz')}" for r, t in part]]
            try:
                run(cmd, log)
            except StepFailed:
                pass                                          # every session of this part failed: reported below
            for r, t in part:
                if t.with_suffix(".tmp.npz").exists():
                    t.with_suffix(".tmp.npz").replace(t)
        missing = [r["session"] for r, t in todo if not t.exists()]
        if missing:
            log(f"WARNING: could not cut {missing}; training goes on without them")
        return [t for r, t in jobs if t.exists()]
    log(f"\ncutting {len(todo)} session(s) into training pictures ({len(jobs) - len(todo)} already done), "
        f"{min(workers, max(len(todo), 1))} at a time")

    def one(job: tuple[dict[str, Any], Path]) -> tuple[str, str, int]:
        r, target = job
        tmp = target.with_suffix(".tmp.npz")
        cmd = [PY, "-m", "sonar.dataset", r["folder"], "--out", str(tmp)]
        res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             env=child_env(single_thread_math=True))
        if res.returncode == 0 and tmp.exists():
            tmp.replace(target)
        return show(cmd), res.stdout + res.stderr, res.returncode

    failed = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(todo) or 1))) as pool:
        for (r, _), (cmd, text, code_) in zip(todo, pool.map(one, todo)):
            log(f"$ {cmd}")
            log.raw(text)
            if code_ != 0:
                failed.append(r["session"])
    if failed:
        log(f"WARNING: could not cut {failed}; training goes on without them")
    return [t for r, t in jobs if t.exists()]


def load_real(files: list[Path]) -> dict[str, Any]:
    sys.path.insert(0, str(REPO))
    from sonar.dataset import merge
    from sonar.train import load_npz

    parts = [load_npz(f) for f in files]
    return merge(parts) if len(parts) > 1 else parts[0]


def measure_orientation(real: dict[str, Any], log: Log) -> tuple[tuple[bool, bool], dict[str, Any]]:
    """Which way round are the laptop's microphone and speaker channels? Read off the recorded swipes."""
    import numpy as np

    from sonar.config import SonarConfig
    from sonar.dataset import center_crop
    from sonar.train import fit_heuristic

    labels = [str(s) for s in real["labels"]]
    cfg = SonarConfig.from_dict(json.loads(str(real["config"])))
    sel = np.isin(real["y"], [labels.index("swipe_left"), labels.index("swipe_right")])
    heur = fit_heuristic(center_crop(real["X"][sel].astype(np.float32), cfg, int(real["pad"])), real["y"][sel], labels, cfg)
    if heur["n_swipes"] == 0:
        log("\nleft/right wiring: not measured (fewer than 4 recorded swipes each way); assuming the usual wiring")
        return (False, False), heur
    t_mic, t_spk = heur.get("t_mics", 0.0), heur.get("t_speakers", 0.0)

    def word(t: float) -> str:
        return "as expected" if t > 3 else "REVERSED" if t < -3 else "unclear (cue too weak)"

    log(f"\nleft/right wiring from {heur['n_swipes']} recorded swipes: microphones {word(t_mic)} (t = {t_mic:+.1f}), "
        f"speakers {word(t_spk)} (t = {t_spk:+.1f})")
    return (t_mic < -3, t_spk < -3), heur


def pick_val_people(rows: list[dict[str, Any]], log: Log) -> list[str]:
    """Hold out people for the honest score: never the one with the most gesture trials (likely the presenter)."""
    trials: dict[str, int] = {}
    for r in rows:
        if r["use"] and r.get("trials", 0) > 0:
            trials[r["person"]] = trials.get(r["person"], 0) + r["trials"]
    ranked = sorted(trials, key=lambda p: (-trials[p], p))
    if len(ranked) < 2:
        log("\nonly one person has recorded gestures: the score holds out whole sessions instead of a person "
            "(optimistic for strangers). Record more people.")
        return []
    n = 2 if len(ranked) >= 4 else 1
    chosen = ranked[1:1 + n]
    log(f"\nheld out for the honest score: {', '.join(chosen)} (picked automatically; --val-person NAME,NAME to choose)")
    log("gesture trials per person: " + ", ".join(f"{p} {trials[p]}" for p in ranked))
    return chosen


# ---------------------------------------------------------------------------
# 4-6. Training
# ---------------------------------------------------------------------------
def train_cmd(datasets: list[Path], out: Path, args: argparse.Namespace, width: int, seed: int, epochs: int,
              init: Path | None = None, val_person: str = "", extra: list[str] | None = None) -> list[str]:
    cmd = [PY, "-m", "sonar.train", "--dataset", *[str(d) for d in datasets], "--labels", args.labels,
           "--width", str(width), "--seed", str(seed), "--epochs", str(epochs), "--device", args.device, "--out", str(out)]
    if init is not None:
        cmd += ["--init", str(init)]
    if val_person:
        cmd += ["--val-person", val_person]
    if args.batch:
        cmd += ["--batch", str(args.batch)]
    if args.compile:
        cmd.append("--compile")
    return cmd + (extra or [])


def read_metrics(folder: Path) -> dict[str, Any]:
    m = json.loads((folder / "metrics.json").read_text())
    meta = json.loads((folder / "gesture.json").read_text())
    val = m.get("validation", {})
    return {"folder": str(folder), "accuracy": val.get("accuracy"), "macro_f1": val.get("macro_f1"),
            "recall": m.get("gesture_recall_deployed"), "false_triggers": m.get("false_triggers_deployed"),
            "best_epoch": m.get("best_epoch"), "seconds": m.get("train_seconds"), "temperature": meta.get("temperature", 1.0),
            "split": m.get("split"), "n_train": m.get("n_train"), "n_val": m.get("n_val")}


def table(rows: list[dict[str, Any]], log: Log, title: str) -> None:
    def f(v: Any) -> str:
        return "   -  " if v is None else f"{v:6.3f}"

    log(f"\n{title}")
    log(f"  {'width':>5} {'seed':>4}  {'accuracy':>8} {'macro-F1':>8} {'recall@0.6':>10} {'false@0.6':>9} {'best ep':>7} {'sec':>6}")
    for r in rows:
        log(f"  {r['width']:5d} {r['seed']:4d}  {f(r['accuracy']):>8} {f(r['macro_f1']):>8} {f(r['recall']):>10} "
            f"{f(r['false_triggers']):>9} {r['best_epoch'] or 0:7d} {r['seconds'] or 0:6.0f}")


def num(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def verdict(r: dict[str, Any]) -> str:
    acc, rec, fal = r.get("accuracy") or 0.0, r.get("recall") or 0.0, r.get("false_triggers")
    fal = 1.0 if fal is None else fal
    if acc >= 0.90 and rec >= 0.85 and fal <= 0.10:
        return "GOOD: ship it"
    if acc >= 0.80 and rec >= 0.70 and fal <= 0.20:
        return "OK: usable for the demo; more people in the recordings will lift it (TRAINING.md, 'If the numbers are poor')"
    return "POOR: do not rely on it yet; see TRAINING.md, 'If the numbers are poor'"


# ---------------------------------------------------------------------------
# 7. The engine check
# ---------------------------------------------------------------------------
def verify_engine(model: Path, profile: str, log: Log, seconds: float = 25.0) -> dict[str, Any]:
    """Load the exported model the way the engine does and run the simulated demo visitor through it."""
    sys.path.insert(0, str(REPO))
    from sonar.audio import SimAudio
    from sonar.classifier import OnnxClassifier, load_classifier
    from sonar.config import get_profile
    from sonar.engine import Engine

    cfg = get_profile(profile)
    mics = int(json.loads(model.with_suffix(".json").read_text()).get("config", {}).get("n_mics", cfg.n_mics))
    if mics != cfg.n_mics:                                    # e.g. a MacBook model: one microphone
        cfg = cfg.with_(n_mics=mics)
    clf = load_classifier(model, cfg)
    if not isinstance(clf, OnnxClassifier):
        raise StepFailed(f"the engine did not load {model} as a trained model")
    if clf.mismatch:
        raise StepFailed(f"{model} was trained with different signal settings {clf.mismatch} than profile {profile}")
    calls: list[str] = []                                   # every time the engine asks the model about a burst
    predict = clf.predict

    def counted(window: Any) -> tuple[str, float, dict[str, float]]:
        out = predict(window)
        calls.append(out[0])
        return out

    clf.predict = counted                                     # type: ignore[method-assign]
    backend = SimAudio(cfg, scene="demo", realtime=False)
    engine = Engine(cfg, backend, profile=profile)
    engine.set_classifier(clf)
    # this checks that the model loads and runs, not the demo's filters: let every simulated burst reach it
    engine.min_peak_db, engine.gate_range, engine.gate_entering = cfg.seg_min_peak_db, False, False
    events: list[dict[str, Any]] = []
    for _ in range(int(seconds * cfg.fs / backend.block)):
        events += engine.step(backend.read())
    gestures = [f"{e['label']} {e['confidence']:.2f}" for e in events if e["type"] == "gesture"]
    status = [e for e in events if e["type"] == "status"]
    if not status or status[-1]["classifier"] != "model" or not calls:
        raise StepFailed("the engine did not run the trained model")
    log(f"\nengine check: loaded {rel(model)} (classes {list(clf.labels)}, temperature {clf.temperature:.2f}); "
        f"ran {seconds:.0f} s of the simulated demo visitor through it")
    log(f"  movement bursts the model classified: {len(calls)} ({', '.join(f'{calls.count(c)} {c}' for c in sorted(set(calls)))}); "
        f"gestures the engine reported: {gestures or 'none'}")
    log("  (the simulator's visitor is not real data: this proves the model loads and runs, not how accurate it is)")
    return {"bursts_classified": len(calls), "gestures": gestures}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def git_state() -> dict[str, Any]:
    def g(*a: str) -> str:
        res = subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
        return res.stdout.strip() if res.returncode == 0 else ""

    return {"commit": g("rev-parse", "--short", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "uncommitted_changes": bool(g("status", "--porcelain", "--untracked-files=no"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="train_4090", description="Generate, pre-train, fine-tune, export (see the file header)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"], help="training device (default cuda)")
    ap.add_argument("--size", default="full", choices=sorted(PRESETS), help="full (4090), small, or tiny (smoke test)")
    ap.add_argument("--data", default="data", help="folder with recorded sessions (searched recursively)")
    ap.add_argument("--labels", default=LABELS, help="gesture classes (default swipe_left,swipe_right,push)")
    ap.add_argument("--profile", default="default", help="signal profile the recordings were made with")
    ap.add_argument("--val-person", default="", help="held-out people for the score, e.g. guest3,guest7 (default: automatic)")
    ap.add_argument("--sweep", action="store_true", help="try widths 16,24,32 x seeds 0,1,2 and keep the best")
    ap.add_argument("--widths", help="model widths to try (default 24; with --sweep 16,24,32)")
    ap.add_argument("--seeds", help="seeds to try (default 0; with --sweep 0,1,2)")
    ap.add_argument("--no-refit", action="store_true", help="ship the winning held-out run itself instead of retraining it on everyone")
    ap.add_argument("--per-class", type=int, help="synthetic clips per gesture (default by --size)")
    ap.add_argument("--bg-per-class", type=int, help="synthetic clips per gesture on real backgrounds (default by --size)")
    ap.add_argument("--pretrain-epochs", type=int)
    ap.add_argument("--finetune-epochs", type=int)
    ap.add_argument("--batch", type=int, help="training batch (default: 256 on CUDA, 64 otherwise)")
    ap.add_argument("--compile", action="store_true", help="try torch.compile (Linux + CUDA only; usually no gain here)")
    ap.add_argument("--workers", type=int, default=default_workers(), help="CPU processes for data generation (default all)")
    ap.add_argument("--swap", default="auto", choices=["auto", "none", "mics", "speakers", "both"],
                    help="left/right channel wiring for the synthetic data (auto: measured from recorded swipes)")
    ap.add_argument("--allow-sim-sessions", action="store_true", help="also use simulator recordings (smoke tests)")
    ap.add_argument("--mixed-devices", action="store_true", help="accept recordings made under different input device names")
    ap.add_argument("--mics", type=int, help="microphone channels of the target laptop when there are no recordings yet "
                                             "(default: taken from the recordings, else 2 = the HP)")
    ap.add_argument("--force-synth", action="store_true", help="make the synthetic data again even if a cached copy exists")
    ap.add_argument("--gen-device", default="auto", choices=["auto", "cpu", "cuda", "mps", "torch-cpu"],
                    help="where the synthetic clips and session pictures are computed: auto = cuda when training on cuda, "
                         "else cpu; cpu = the original numpy code on every core (the reference); torch-cpu = the GPU code "
                         "run on the CPU (for tests)")
    ap.add_argument("--gen-precision", default="auto", choices=["auto", "ref", "fast", "single"],
                    help="GPU number format (default ref: float64 where the CPU code uses it; fast: float32 FFTs)")
    ap.add_argument("--gen-batch", type=int, default=0, help="clips per GPU batch (default: fills the GPU's memory)")
    ap.add_argument("--gen-gpus", type=int, default=0, help="GPUs used for making data and for the training runs (default: all)")
    ap.add_argument("--skip-parity", action="store_true", help="do not run the GPU-against-CPU picture check before making data on a GPU")
    ap.add_argument("--recheck-parity", action="store_true", help="run that check again even if it already passed for this code")
    ap.add_argument("--sweep-jobs", type=int, default=0,
                    help="training runs sharing the GPU at the same time (default: 4 on CUDA, 1 otherwise)")
    ap.add_argument("--no-overlap", action="store_true",
                    help="CUDA: do not pre-train while the clips on real backgrounds are still being made")
    ap.add_argument("--out", default="train_out", help="working folder (caches, logs, every run)")
    ap.add_argument("--models", help="where the engine looks for the model (default models/; tiny: train_out/tiny/models)")
    ap.add_argument("--release", help="folder committed to hand the model back (default models-release/; tiny: train_out/tiny/...)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")             # a Windows console must never crash on a character

    preset = dict(PRESETS[args.size])
    for k in ("per_class", "bg_per_class", "pretrain_epochs", "finetune_epochs"):
        if getattr(args, k) is not None:
            preset[k] = getattr(args, k)
    widths = [int(w) for w in (args.widths or ("16,24,32" if args.sweep else "24")).split(",")]
    seeds = [int(s) for s in (args.seeds or ("0,1,2" if args.sweep else "0")).split(",")]
    work = (REPO / args.out).resolve()
    # a smoke-test model must never replace a real one in models/ or end up in a commit of models-release/
    smoke = args.size == "tiny"
    models_dir = (REPO / (args.models or (Path(args.out) / "tiny" / "models" if smoke else "models"))).resolve()
    release_dir = (REPO / (args.release or (Path(args.out) / "tiny" / "models-release" if smoke else "models-release"))).resolve()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = work / "runs" / stamp
    log = Log(run_dir / "log.txt")
    t_start = time.time()
    times: dict[str, float] = {}
    summary: dict[str, Any] = {"started": stamp, "argv": sys.argv[1:], "size": args.size, "preset": preset,
                               "labels": args.labels, "widths": widths, "seeds": seeds, "git": git_state()}
    try:
        log(f"=== gesture training run {stamp} ({args.size}, device {args.device}); log: {rel(run_dir / 'log.txt')}")
        summary["machine"] = info = check_machine(args.device, log)
        if args.gen_device == "auto":
            args.gen_device = "cuda" if args.device == "cuda" else "cpu"
        if args.gen_device == "cuda" and not info.get("cuda"):
            raise StepFailed("--gen-device cuda, but PyTorch cannot see a CUDA GPU; use --gen-device cpu")
        if args.gen_device == "mps" and not info.get("mps"):
            raise StepFailed("--gen-device mps, but Apple MPS is not available here; use --gen-device cpu")
        on_cuda = args.device == "cuda"
        n_gpus = max(1, min(int(info.get("gpu_count", 1)), args.gen_gpus or 10**6)) if on_cuda else 1
        sweep_jobs = args.sweep_jobs or (4 * n_gpus if on_cuda else 1)
        summary["generation"] = {"device": args.gen_device, "precision": args.gen_precision if args.gen_device != "cpu" else None}
        log(f"synthetic clips and session pictures are computed on: {args.gen_device}"
            + ("" if args.gen_device != "cpu" else " (numpy, every core)"))
        if args.gen_device != "cpu":
            if args.skip_parity:
                log("GPU parity check: SKIPPED (--skip-parity)")
                summary["parity"] = {"status": "skipped"}
            else:
                t0 = time.time()
                summary["parity"] = parity_gate(args, info, work, log)
                times["GPU parity check"] = time.time() - t0

        # 3a. real recordings: inventory, cut into pictures, wiring
        data_root = (REPO / args.data).resolve()
        rows = inventory(data_root, args.profile, args.allow_sim_sessions, log) if data_root.exists() else []
        usable = [r for r in rows if r["use"]]
        summary["sessions"] = [{k: r.get(k) for k in ("session", "person", "script", "seconds", "trials", "use", "why")} for r in rows]
        real_files: list[Path] = []
        val_people: list[str] = []
        swap = {"none": (False, False), "mics": (True, False), "speakers": (False, True), "both": (True, True)}.get(args.swap, (False, False))
        mics = args.mics or 2
        if usable:
            mics = one_device(usable, args.mixed_devices, log)
            if args.mics and args.mics != mics:
                raise StepFailed(f"--mics {args.mics}, but the recordings have {mics} microphone channel(s)")
            t0 = time.time()
            real_files = cut_sessions(usable, work / "sessions", args.workers, log, device=args.gen_device)
            times["cut sessions"] = time.time() - t0
        if real_files:
            real = load_real(real_files)
            names = [str(s) for s in real["labels"]]
            log(f"\nreal training pictures: {len(real['y'])} " +
                str({n: int((real["y"] == i).sum()) for i, n in enumerate(names)}))
            measured, heur = measure_orientation(real, log)
            summary["wiring"] = {k: heur.get(k) for k in ("n_swipes", "t_mics", "t_speakers")}
            if args.swap == "auto":
                swap = measured
                if any(swap):
                    log(f"=> synthetic data will be made with the same wiring (swap mics {swap[0]}, swap speakers {swap[1]})")
            del real
            val_people = [p for p in args.val_person.split(",") if p] or pick_val_people(usable, log)
        else:
            log(f"\nno usable recordings under {rel(data_root)}: synthetic pre-training only "
                "(fine-tune later by running this again once data/ holds the recorded sessions)")
        summary["swap"] = {"mics": swap[0], "speakers": swap[1]}
        summary["microphones"] = mics
        if mics == 1:
            log("one microphone channel (a MacBook): no two-microphone direction cue; left/right swipes rely on the two speakers")

        # 2. synthetic data (pure), 3b. synthetic gestures on real backgrounds
        log("\n=== synthetic data")
        synth, sec = generate(work / "synth", preset["per_class"], preset["negatives"], args, log, swap, mics)
        times["generate synthetic"] = sec
        # Real backgrounds: the held-out people's recordings stay out of the clips the scored runs train on
        # (their real movements would otherwise sit in training and in the score); the final retrain uses all.
        bg_all = [r["audio"] for r in usable if r["script"] in BACKGROUND_SCRIPTS]
        bg_scored = [r["audio"] for r in usable if r["script"] in BACKGROUND_SCRIPTS and r["person"] not in val_people]
        refit = bool(val_people) and not args.no_refit
        synth_bg = synth_bg_all = None

        # 4. pre-training, one per width (cached). On CUDA it starts NOW and runs while the GPU also makes the
        #    clips on real backgrounds below: the model is tiny, so both fit side by side.
        pre_code = code_hash(TRAIN_CODE)
        pretrained: dict[int, Path] = {}
        pre_todo: list[tuple[list[str], Path]] = []
        pre_folders: dict[int, Path] = {}
        fresh: set[int] = set()                               # widths pre-trained in this run (the others are cached)
        for w in widths:
            folder = work / "pretrain" / key_of(synth.parent.name, synth.stat().st_size, int(synth.stat().st_mtime), args.labels,
                                                w, preset["pretrain_epochs"], args.device, args.batch, args.compile, pre_code)
            pre_folders[w] = folder
            pretrained[w] = folder / "gesture.pt"
            if not cached(folder):
                fresh.add(w)
                pre_todo.append((train_cmd([synth], folder, args, w, 0, preset["pretrain_epochs"]), run_dir / "pretrain" / f"w{w}.txt"))
        will_make_bg = bool(real_files and bg_scored)
        overlap = on_cuda and not args.no_overlap and bool(pre_todo) and will_make_bg
        pre_jobs: list[Job] = []
        t_pre = time.time()                                   # overlapping: the clock runs from the start of the jobs
        if overlap:
            log(f"\n=== pre-training {len(pre_todo)} model(s) in the background while the next clips are made "
                f"(their output follows below; --no-overlap for one step at a time)")
            pre_jobs = [Job(cmd, out_file, env=gpu_env(k, n_gpus)) for k, (cmd, out_file) in enumerate(pre_todo)]
        # while training shares the GPU, the generator leaves room for it (about 3 GB per training run)
        gen_batch = 0
        if overlap and args.gen_device == "cuda" and not args.gen_batch:
            sys.path.insert(0, str(REPO))
            from datagen.gpu_pipeline import batch_for_memory

            precision = "fast" if args.gen_precision == "fast" else "ref"
            gen_batch = batch_for_memory(float(info.get("gpu_gb", 24.0)) - 1.5 - 3.0 * len(pre_jobs), precision)

        try:
            if will_make_bg:
                log(f"\n=== synthetic gestures on {len(bg_scored)} real background recording(s)"
                    + (f" (not {', '.join(val_people)}'s: they are held out)" if len(bg_scored) < len(bg_all) else ""))
                synth_bg, sec = generate(work / "synth_bg", preset["bg_per_class"], preset["negatives"], args, log, swap, mics,
                                         backgrounds=bg_scored, seed=1, gen_batch=gen_batch)
                times["generate on real backgrounds"] = sec
            synth_bg_all = synth_bg
            if real_files and refit and len(bg_all) > len(bg_scored):
                log(f"\n=== synthetic gestures on all {len(bg_all)} real background recording(s), for the final retrain")
                synth_bg_all, sec = generate(work / "synth_bg", preset["bg_per_class"], preset["negatives"], args, log, swap,
                                             mics, backgrounds=bg_all, seed=1, gen_batch=gen_batch)
                times["generate on all backgrounds"] = sec
        except BaseException:
            for job in pre_jobs:                              # do not leave training running after a failed step
                job.proc.kill()
            raise
        if real_files and not bg_all:
            log("\nno scene / breathing recordings: no synthetic gestures on real backgrounds (record a `--script scene` session)")

        log("\n=== pre-training on synthetic data")
        if pre_jobs:
            failed: StepFailed | None = None
            for job in pre_jobs:
                try:
                    job.finish(log)
                except StepFailed as exc:
                    failed = failed or exc
            if failed is not None:
                raise failed
        elif pre_todo:
            t_pre = time.time()
            run_many(pre_todo, log, sweep_jobs, n_gpus)
        pre_rows = []
        for w in widths:
            if w in fresh:
                mark_done(pre_folders[w], {"width": w})
            else:
                log(f"\nreusing pre-trained width {w}: {rel(pre_folders[w])}")
            pre_rows.append({"width": w, "seed": 0, **read_metrics(pre_folders[w])})
        times["pre-train"] = time.time() - t_pre
        table(pre_rows, log, "pre-training, scored on held-out SYNTHETIC clips:")
        summary["pretrain"] = pre_rows

        # 5. fine-tune sweep on the real recordings
        if real_files:
            val = ",".join(val_people)
            tune_sets = real_files + [synth_bg or synth]
            log(f"\n=== fine-tuning on {len(real_files)} recorded session(s) + {rel(Path(tune_sets[-1]).parent)}")
            sweep_rows = []
            t0 = time.time()
            grid = [(w, s, run_dir / "sweep" / f"w{w}_s{s}") for w in widths for s in seeds]
            if sweep_jobs > 1 and len(grid) > 1:
                log(f"{len(grid)} runs, {min(sweep_jobs, len(grid))} at a time on {n_gpus} GPU(s) (--sweep-jobs 1 for one at a time; "
                    "the seconds per run below include the sharing)")
            run_many([(train_cmd(tune_sets, folder, args, w, s, preset["finetune_epochs"], init=pretrained[w], val_person=val),
                       folder / "output.txt") for w, s, folder in grid], log, sweep_jobs, n_gpus)
            for w, s, folder in grid:
                sweep_rows.append({"width": w, "seed": s, **read_metrics(folder)})
            times["fine-tune sweep"] = time.time() - t0
            score_on = f"held-out {'people' if len(val_people) > 1 else 'person'} {val}" if val_people else "held-out sessions"
            table(sweep_rows, log, f"fine-tuning, scored on {score_on} (never seen in training):")
            best = max(sweep_rows, key=lambda r: (r["accuracy"] or 0.0, r["macro_f1"] or 0.0, -r["width"], -r["seed"]))
            log(f"\nwinner: width {best['width']}, seed {best['seed']} (highest held-out accuracy)")
            if len(sweep_rows) > 1:
                log("  (picking the best of several runs on the same people flatters its score a little)")
            summary.update(val_people=val_people, sweep=sweep_rows, winner=best)
            final = Path(best["folder"])
            if refit:
                log(f"\n=== retraining the winner on everyone, {', '.join(val_people)} included (--no-refit to skip)")
                final = run_dir / "final"
                t0 = time.time()
                run(train_cmd(real_files + [synth_bg_all or synth], final, args, best["width"], best["seed"], preset["finetune_epochs"],
                              init=pretrained[best["width"]], extra=["--val-fraction", "0", "--temperature", f"{best['temperature']:.4f}"]), log)
                times["refit on everyone"] = time.time() - t0
            honest = best
        else:
            best = max(pre_rows, key=lambda r: (r["accuracy"] or 0.0, r["macro_f1"] or 0.0, -r["width"]))
            final, honest = Path(best["folder"]), None
            summary["winner"] = best

        # 6. export
        log("\n=== export")
        for dest in (models_dir, release_dir):
            dest.mkdir(parents=True, exist_ok=True)
            for name in RELEASE_FILES + ("TRAINING_RUN.json",):
                (dest / name).unlink(missing_ok=True)            # nothing stale from an earlier run survives
                if (final / name).exists():
                    shutil.copy2(final / name, dest / name)
            log(f"copied {', '.join(RELEASE_FILES)} from {rel(final)} to {rel(dest)}")

        summary["model_from"] = rel(final)
        if honest is not None:
            summary["verdict"] = verdict(honest)
            summary["honest_score"] = {k: honest[k] for k in ("accuracy", "macro_f1", "recall", "false_triggers", "split")}
        else:
            summary["verdict"] = "SYNTHETIC ONLY: a pre-trained starting point, not the demo model (no recordings yet)"

        def save() -> None:
            for path in (run_dir / "summary.json", release_dir / "TRAINING_RUN.json"):
                path.write_text(json.dumps(summary, indent=1, default=str))

        save()                                                # the score is kept even if the engine check fails

        # 7. engine check
        summary["engine_check"] = verify_engine(models_dir / "gesture.onnx", args.profile, log)
        times["total"] = time.time() - t_start
        summary["minutes"] = {k: round(v / 60, 1) for k, v in times.items()}
        summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save()

        log("\n" + "=" * 78)
        log(f"DONE in {times['total'] / 60:.1f} min. Model: {rel(models_dir)}/gesture.onnx (+ gesture.json, gesture.pt), "
            f"copied to {rel(release_dir)}/")
        if honest is not None:
            log(f"Honest score on {', '.join(summary['val_people']) or 'held-out sessions'}: accuracy {num(honest['accuracy'])}, "
                f"macro-F1 {num(honest['macro_f1'])}; at the engine's 0.6 threshold {num(honest['recall'])} of gestures "
                f"recognised, {num(honest['false_triggers'])} of non-gestures would trigger")
        log(f"Verdict: {summary['verdict']}")
        log("Time per step: " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in times.items()))
        if smoke:
            log("This was a smoke test: its model is junk and stays under train_out/tiny/. Now run the full training.")
        else:
            log(f"Next: commit {rel(release_dir)}/ and push (TRAINING.md, step 8).")
        log(f"Full log: {rel(run_dir / 'log.txt')}")
        return 0
    except StepFailed as exc:
        log(f"\nSTOPPED: {exc}")
        log(f"Full log: {rel(run_dir / 'log.txt')}")
        return 1
    except Exception:
        log(f"\nSTOPPED by an unexpected error (please send this log):\n{traceback.format_exc()}")
        log(f"Full log: {rel(run_dir / 'log.txt')}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

"""suite.py — the whole leave-one-person-out experiment in one command, built for the RTX 4090.

What it does, in order (every step is cached, so a second run only does what is missing):
  1. finds every recorded session under --data and runs each through the engine's own pipeline once;
  2. builds one table of every movement burst with its label, session and person, and checks each session's
     left/right swipes against the others (a session whose swipes go the "wrong" way is reported);
  3. makes the synthetic sets on the GPU with the unchanged generator: the plain set (for pre-training) and one
     set laid over each person's own "nobody gesturing" recordings, so a fold never trains on audio of the
     person it is tested on;
  4. pre-trains once per model width on the plain synthetic set;
  5. trains every recipe ("arm") for every held-out person and seed, several runs sharing the GPU, plus the
     extra runs that give a confidence temperature without looking at the test person;
  6. writes the scores (report.md, results.json) and every model's raw answers (logits) so anything else can be
     recomputed later without the GPU;
  7. trains the best recipes once more on everyone and exports them the way the engine loads them
     (gesture.onnx + gesture.json + gesture.pt). Nothing is copied into models/ or models-release*/.

How to run (from the repo root):
    python -m recall.suite run --device cuda                       # the 4090: about 30 minutes
    python -m recall.suite run --device cpu --smoke                # 2-minute self-test with junk-size data
    python -m recall.suite report --run train_out/recall/suite/<stamp>      # rebuild the report from saved answers

What it changes: writes only under train_out/recall/ (caches) and train_out/recall/suite/<stamp>/ (this run).
It reads the recordings and never opens an audio device.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "train_out" / "recall"
LABELS = ("none", "swipe_left", "swipe_right", "push")
BUDGET = 0.055            # false-trigger share of the released model: the level to stay at or under
SYNTH_LABELS = "swipe_left,swipe_right,push"

# ---------------------------------------------------------------------------
# The recipes. Every arm is scored the same way; only what is listed differs from the first one.
#   real   which recorded rows train the model: "official" (strongest burst of each prompt + other bursts + idle),
#          "aux_none" (also the wind-up / return / tail bursts of each trial, as "none"),
#          "no_lance_swipes" (official without lance's and lance2's swipes: they are faster than the signal can show)
#   synth  None | "old" (plain synthetic set) | "old_bg" (synthetic gestures over the training people's real backgrounds)
#   mix    "class_balanced" (released: all rows pooled, each class equally often)
#          or the share of each batch drawn from the REAL recordings, 0..1
#   init   None (from scratch) | "old" (weights pre-trained on the plain synthetic set)
#   none_share  share of each batch that is "none" (default: every class equally often, i.e. 0.25 with 3 gestures)
#   aug    augmentation options (recall/augment.py), e.g. a wider speed range; width = model width (default 24)
# ---------------------------------------------------------------------------
ARMS: dict[str, dict[str, Any]] = {
    "released":        {"real": "official", "synth": "old_bg", "mix": "class_balanced", "init": "old"},
    "real_only":       {"real": "official", "synth": None, "mix": "class_balanced", "init": None},
    "real_pre":        {"real": "official", "synth": None, "mix": "class_balanced", "init": "old"},
    "real_pre_lr03":   {"real": "official", "synth": None, "mix": "class_balanced", "init": "old", "lr_scale": 0.3},
    "share25":         {"real": "official", "synth": "old_bg", "mix": 0.25, "init": "old"},
    "share50":         {"real": "official", "synth": "old_bg", "mix": 0.50, "init": "old"},
    "share75":         {"real": "official", "synth": "old_bg", "mix": 0.75, "init": "old"},
    "share50_scratch": {"real": "official", "synth": "old_bg", "mix": 0.50, "init": None},
    "share50_pure":    {"real": "official", "synth": "old", "mix": 0.50, "init": "old"},
    "share50_aux":     {"real": "aux_none", "synth": "old_bg", "mix": 0.50, "init": "old"},
    "share50_nolsw":   {"real": "no_lance_swipes", "synth": "old_bg", "mix": 0.50, "init": "old"},
    "share50_speed":   {"real": "official", "synth": "old_bg", "mix": 0.50, "init": "old",
                        "aug": {"speed": [0.5, 2.2], "speed_p": 0.8}},
    "share50_w48":     {"real": "official", "synth": "old_bg", "mix": 0.50, "init": "old", "width": 48},
    "share50_none50":  {"real": "official", "synth": "old_bg", "mix": 0.50, "init": "old", "none_share": 0.5},
    "share50_aux_speed": {"real": "aux_none", "synth": "old_bg", "mix": 0.50, "init": "old",
                          "aug": {"speed": [0.5, 2.2], "speed_p": 0.8}},
    # --- the simulator fitted to this MacBook (train_out/synth_cal/<family>/<fold>/dataset.npz, made by the simulator
    #     session's handoff). "{fold}" = the set that leaves out the held-out human. Skipped if the files are missing.
    "cal50":           {"real": "official", "synth": "macbook_v1_{fold}", "mix": 0.50, "init": "cal:macbook_v1"},
    "cal25":           {"real": "official", "synth": "macbook_v1_{fold}", "mix": 0.25, "init": "cal:macbook_v1"},
    "cal50_aux":       {"real": "aux_none", "synth": "macbook_v1_{fold}", "mix": 0.50, "init": "cal:macbook_v1"},
    "cal50_oldbg":     {"real": "official", "synth": "macbook_v1_{fold}+old_bg", "mix": 0.50, "init": "cal:macbook_v1"},
    "cal_pre_real":    {"real": "official", "synth": None, "mix": "class_balanced", "init": "cal:macbook_v1"},
    "calm50":          {"real": "official", "synth": "macbook_v1_motion_{fold}", "mix": 0.50, "init": "cal:macbook_v1_motion"},
    "calm50_aux":      {"real": "aux_none", "synth": "macbook_v1_motion_{fold}", "mix": 0.50, "init": "cal:macbook_v1_motion"},
}
SIZES = {
    "full": {"per_class": 2000, "bg_per_class": 500, "epochs": 60, "pre_epochs": 60, "seeds": (0, 1, 2), "parity_clips": 1000},
    "smoke": {"per_class": 5, "bg_per_class": 3, "epochs": 1, "pre_epochs": 1, "seeds": (0,), "parity_clips": 100, "steps": 2},
}


def log(text: str = "", fh: Any = None) -> None:
    print(text, flush=True)
    if fh is not None:
        fh.write(text + "\n")
        fh.flush()


def load_train_4090() -> Any:
    """scripts/train_4090.py as a module (its machine check and its GPU-against-CPU picture gate are reused as they are)."""
    spec = importlib.util.spec_from_file_location("train_4090", REPO / "scripts" / "train_4090.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def child_env() -> dict[str, str]:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", PYTORCH_ENABLE_MPS_FALLBACK="1")
    return env


def run_cmd(cmd: list[str], fh: Any, single_thread: bool = False) -> float:
    log(f"\n$ {' '.join(str(c) for c in cmd)}", fh)
    env = child_env()
    if single_thread:
        env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    t0 = time.time()
    proc = subprocess.Popen([str(c) for c in cmd], cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        fh.write(line)
    if proc.wait() != 0:
        raise RuntimeError(f"step failed (exit code {proc.returncode}): {' '.join(str(c) for c in cmd)}")
    return time.time() - t0


# ---------------------------------------------------------------------------
# 1-2. Sessions -> table, label audit
# ---------------------------------------------------------------------------
def build_table(data_root: Path, run_dir: Path, locked: list[str], fh: Any) -> dict[str, np.ndarray]:
    """Every burst of every session as arrays (saved to <run>/table.npz for the workers)."""
    from . import cache_streams
    from .common import HUMAN, load_sessions
    from .data import full_table

    cache_streams.main_for(data_root)
    sessions = load_sessions()
    present = {str(p.parent.name) for p in data_root.rglob("session.json")}
    sessions = [s for s in sessions if s.name in present]
    t = full_table(sessions)
    people = sorted(set(t.person))
    unknown = [p for p in people if p not in HUMAN]
    if unknown:
        log(f"NOTE: new people {unknown}: each is treated as their own human (held out on their own).", fh)
    cols = {
        "X": t.X, "label": t.label, "session": t.session, "person": t.person, "human": t.human, "source": t.source,
        "peak_db": t.peak_db, "trial": t.trial, "seg": t.seg, "forced": t.forced,
        "locked": np.isin(t.session, locked), **{k: v for k, v in t.extra.items()},
    }
    np.savez_compressed(run_dir / "table.npz", **cols)
    log(f"\ntable: {len(t)} rows from {len(sessions)} sessions; humans {sorted(set(t.human))}; "
        f"locked (never trained on, scored separately): {[s for s in locked if s in present] or 'none'}", fh)
    for name in sorted(set(t.session)):
        m = t.session == name
        counts = {k: int(((t.source == k) & m).sum()) for k in ("gesture", "aux", "burst", "idle")}
        log(f"  {name:<40} {t.person[m][0]:<8} {counts}", fh)
    return cols


def label_audit(cols: dict[str, np.ndarray], run_dir: Path, fh: Any) -> dict[str, Any]:
    """Per session: do the recorded swipes go the way their prompts say?

    Uses the physics cue (left-speaker tone Doppler minus right-speaker tone Doppler, strongest half second,
    sonar.gestures.heuristic_classify). For each gesture session: t-statistic of left-prompt minus right-prompt.
    Clearly positive = as the other sessions; clearly negative = the swipes go the opposite way to the prompts.
    """
    sys.path.insert(0, str(REPO))
    from sonar.config import SonarConfig
    from sonar.gestures import heuristic_classify

    cfg = SonarConfig(n_mics=1)
    sw = np.isin(cols["label"], ["swipe_left", "swipe_right"]) & (cols["source"] == "gesture")
    cue = np.full(len(sw), np.nan)
    for i in np.nonzero(sw)[0]:
        cue[i] = heuristic_classify(cols["X"][i].astype(np.float32)[:, :, 8:88], cfg).lateral_hz
    out: dict[str, Any] = {}
    lines = ["label audit: left-speaker tone minus right-speaker tone, Hz (positive = hand moving LEFT)",
             f"  {'session':<40} {'n':>3} {'left prompts':>16} {'right prompts':>16} {'t':>7}  verdict"]
    for name in sorted(set(cols["session"][sw])):
        m = sw & (cols["session"] == name)
        left, right = cue[m & (cols["label"] == "swipe_left")], cue[m & (cols["label"] == "swipe_right")]
        if min(len(left), len(right)) < 4:
            continue
        se = np.sqrt(left.var() / len(left) + right.var() / len(right)) + 1e-9
        t = float((left.mean() - right.mean()) / se)
        verdict = "as prompted" if t > 3 else "OPPOSITE to the prompts" if t < -3 else "unclear (no clean direction cue)"
        out[str(name)] = {"t": t, "left_hz": float(left.mean()), "right_hz": float(right.mean()), "n": int(m.sum()), "verdict": verdict}
        lines.append(f"  {name:<40} {int(m.sum()):3d} {left.mean():+8.1f}±{left.std():5.1f} {right.mean():+8.1f}±{right.std():5.1f} {t:+7.2f}  {verdict}")
    log("\n" + "\n".join(lines), fh)
    (run_dir / "label_audit.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------------------
# 3. Synthetic data (the unchanged generator, on the GPU)
# ---------------------------------------------------------------------------
def generate(folder: Path, per_class: int, seed: int, gen_device: str, workers: int, fh: Any,
             backgrounds: list[str] | None = None, extra: list[str] | None = None) -> Path:
    """datagen.generate into `folder` unless its dataset is already there."""
    target = folder / "dataset.npz"
    if target.exists() and (folder / "done.json").exists():
        log(f"reusing {target.relative_to(REPO)}", fh)
        return target
    cmd = [sys.executable, "-m", "datagen.generate", "--out", str(folder), "--per-class", str(per_class), "--negatives", "2.0",
           "--workers", str(workers), "--seed", str(seed), "--profile", "default", "--mics", "1"]
    if backgrounds:
        cmd += ["--background", ",".join(backgrounds)]
    if gen_device != "cpu":
        cmd += ["--gen-device", gen_device, "--gen-precision", "auto"]
    cmd += extra or []
    seconds = run_cmd(cmd, fh, single_thread=True)
    if not target.exists():
        raise RuntimeError(f"datagen wrote no dataset.npz into {folder}")
    (folder / "done.json").write_text(json.dumps({"seconds": round(seconds, 1), "cmd": cmd}, indent=1))
    return target


def remap(d: dict[str, np.ndarray]) -> np.ndarray:
    """Labels of a datagen/sonar dataset file -> indices into LABELS (pull and anything else -> none)."""
    names = [str(s) for s in d["labels"]]
    lut = np.array([LABELS.index(n) if n in LABELS else 0 for n in names])
    return lut[d["y"]]


# ---------------------------------------------------------------------------
# 5. One training job (runs inside a worker process)
# ---------------------------------------------------------------------------
class Bank:
    """All pictures a worker may need, resident on the training device, with their bookkeeping columns."""

    def __init__(self, run_dir: Path, device: str) -> None:
        import torch

        from .trainlib import pick_device

        self.dev = pick_device(device)
        man = json.loads((run_dir / "manifest.json").read_text())
        self.man = man
        with np.load(run_dir / "table.npz", allow_pickle=False) as z:
            self.real = {k: z[k] for k in z.files}
        lut = {n: i for i, n in enumerate(LABELS)}
        self.y_real = np.array([lut.get(str(v), 0) for v in self.real["label"]], dtype=np.int64)
        parts = [self.real["X"]]
        ys = [self.y_real]
        self.slices: dict[str, slice] = {"real": slice(0, len(self.y_real))}
        at = len(self.y_real)
        for name, path in man["synth"].items():
            with np.load(path, allow_pickle=False) as z:
                d = {k: z[k] for k in ("X", "y", "labels")}
            parts.append(d["X"])
            ys.append(remap(d))
            self.slices[name] = slice(at, at + len(d["y"]))
            at += len(d["y"])
        self.x = torch.from_numpy(np.concatenate(parts).astype(np.float16)).to(self.dev)
        self.y_np = np.concatenate(ys)
        self.y = torch.from_numpy(self.y_np).to(self.dev)
        self.n = at

    def real_mask(self, mode: str, humans: list[str]) -> np.ndarray:
        r = self.real
        ok = np.isin(r["human"], humans) & ~r["locked"]
        base = np.isin(r["source"], ["gesture", "burst", "idle"])
        if mode == "official":
            return ok & base
        if mode == "aux_none":
            return ok & (base | (r["source"] == "aux"))
        if mode == "no_lance_swipes":
            lance_swipe = (r["human"] == "lance") & (r["source"] == "gesture") & np.isin(r["label"], ["swipe_left", "swipe_right"])
            return ok & base & ~lance_swipe
        raise ValueError(f"unknown real mode {mode!r}")

    def weights(self, job: dict[str, Any]) -> np.ndarray:
        """Sampling weight of every row in the bank for this job (0 = not used)."""
        humans = job["train_humans"]
        w = np.zeros(self.n)
        real = np.zeros(self.n, bool)
        real[self.slices["real"]] = self.real_mask(job["real"], humans)
        synth = np.zeros(self.n, bool)
        if job["synth"] == "old":
            synth[self.slices["old"]] = True
        elif job["synth"] == "old_bg":
            for h in humans:
                if f"bg_{h}" in self.slices:
                    synth[self.slices[f"bg_{h}"]] = True
        elif job["synth"]:
            for name in str(job["synth"]).split("+"):
                if name == "old_bg":
                    for h in humans:
                        if f"bg_{h}" in self.slices:
                            synth[self.slices[f"bg_{h}"]] = True
                    continue
                key = name.replace("{fold}", job["synth_fold"])
                if key not in self.slices:
                    raise KeyError(f"synthetic set {key!r} is not loaded (have {sorted(self.slices)})")
                synth[self.slices[key]] = True

        none_share = job.get("none_share")

        def balanced(mask: np.ndarray) -> np.ndarray:
            out = np.zeros(self.n)
            if mask.any():
                counts = np.bincount(self.y_np[mask], minlength=len(LABELS)).astype(float)
                mass = np.ones(len(LABELS))                      # every class equally often ...
                if none_share is not None:                       # ... unless "none" gets its own share
                    mass[:] = (1.0 - float(none_share)) / (len(LABELS) - 1)
                    mass[0] = float(none_share)
                out[mask] = (mass / np.maximum(counts, 1.0))[self.y_np[mask]]
                out /= out.sum()
            return out

        if job["mix"] == "class_balanced" or not synth.any() or not real.any():
            return balanced(real | synth)
        share = float(job["mix"])
        return share * balanced(real) + (1.0 - share) * balanced(synth)


def run_job(bank: Bank, job: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    import torch

    from .trainlib import CFG, TrainSpec, logits_of, train

    t0 = time.time()
    w = bank.weights(job)
    spec = TrainSpec(epochs=job["epochs"], steps=int(job.get("steps", 0)), width=job.get("width", 24), seed=job["seed"],
                     lr_scale=job.get("lr_scale", 1.0), device=str(bank.dev), aug=job.get("aug") or {})
    init = None
    if job.get("init"):
        state = torch.load(job["init"], map_location="cpu")
        init = state["model"] if isinstance(state, dict) and "model" in state else state
    model = train(bank.x, bank.y, torch.from_numpy(w), LABELS, spec, init=init, cfg=CFG)
    logits = logits_of(model, bank.real["X"], device=str(bank.dev))
    np.savez_compressed(out_dir / f"{job['id']}.npz", logits=logits.astype(np.float32))
    info = {"id": job["id"], "seconds": round(time.time() - t0, 1), "n_train": int((w > 0).sum()),
            "real_share": float(w[bank.slices["real"]].sum())}
    if job.get("save_model"):
        torch.save({"model": model.state_dict(), "labels": list(LABELS), "width": spec.width}, out_dir / f"{job['id']}.pt")
    return info


def worker_main(run_dir: Path, jobs_file: Path, device: str) -> int:
    jobs = json.loads(jobs_file.read_text())
    out_dir = run_dir / "jobs"
    out_dir.mkdir(exist_ok=True)
    bank = Bank(run_dir, device)
    failed = 0
    for job in jobs:
        if (out_dir / f"{job['id']}.npz").exists() and (not job.get("save_model") or (out_dir / f"{job['id']}.pt").exists()):
            continue
        try:
            info = run_job(bank, job, out_dir)
            (out_dir / f"{job['id']}.json").write_text(json.dumps(info))
            print(f"  done {job['id']}  {info['seconds']} s  n_train {info['n_train']}  real share {info['real_share']:.2f}", flush=True)
        except Exception:
            failed += 1
            (out_dir / f"{job['id']}.error.txt").write_text(traceback.format_exc())
            print(f"  FAILED {job['id']}:\n{traceback.format_exc()}", flush=True)
    return 1 if failed else 0


def run_jobs(run_dir: Path, jobs: list[dict[str, Any]], device: str, workers: int, fh: Any, tag: str) -> None:
    """Deal the jobs out to `workers` processes that share the GPU; wait; copy their output into the log."""
    todo = [j for j in jobs if not (run_dir / "jobs" / f"{j['id']}.npz").exists()
            or (j.get("save_model") and not (run_dir / "jobs" / f"{j['id']}.pt").exists())]
    log(f"\n=== {tag}: {len(jobs)} training runs, {len(todo)} to do, {min(workers, max(len(todo), 1))} at a time on {device}", fh)
    if not todo:
        return
    t0 = time.time()
    procs = []
    for k in range(min(workers, len(todo))):
        part = todo[k::workers]
        jf = run_dir / f"jobs_{tag}_{k}.json"
        jf.write_text(json.dumps(part))
        out = open(run_dir / f"worker_{tag}_{k}.txt", "w", encoding="utf-8")
        procs.append((subprocess.Popen([sys.executable, "-m", "recall.suite", "worker", "--run", str(run_dir), "--jobs", str(jf),
                                        "--device", device], cwd=REPO, stdout=out, stderr=subprocess.STDOUT, env=child_env()), out, k))
    for proc, out, k in procs:
        proc.wait()
        out.close()
        fh.write((run_dir / f"worker_{tag}_{k}.txt").read_text(encoding="utf-8", errors="replace"))
    errors = sorted((run_dir / "jobs").glob("*.error.txt"))
    log(f"{tag}: finished in {time.time() - t0:.0f} s; {len(errors)} failed run(s)" + (f": {[e.name for e in errors[:5]]}" if errors else ""), fh)


# ---------------------------------------------------------------------------
# 6. Scores
# ---------------------------------------------------------------------------
def load_logits(run_dir: Path, job_id: str) -> np.ndarray | None:
    p = run_dir / "jobs" / f"{job_id}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        return z["logits"]


def make_report(run_dir: Path, fh: Any = None) -> dict[str, Any]:
    """Scores of every arm from the saved answers. Safe to re-run anywhere (needs only numpy)."""
    from .trainlib import fit_temperature, recall_at_false, score, softmax

    man = json.loads((run_dir / "manifest.json").read_text())
    with np.load(run_dir / "table.npz", allow_pickle=False) as z:
        r = {k: z[k] for k in z.files if k != "X"}
    lut = {n: i for i, n in enumerate(LABELS)}
    y = np.array([lut.get(str(v), 0) for v in r["label"]], dtype=np.int64)
    scored = np.isin(r["source"], ["gesture", "burst", "idle"]) & ~r["locked"]       # the released definition of an example
    humans, seeds = man["humans"], man["seeds"]
    results: dict[str, Any] = {}
    lines: list[str] = []
    for arm in man["arms"]:
        res: dict[str, Any] = {"folds": {}}
        pooled: dict[str, list[np.ndarray]] = {"t1": [], "nested": [], "y": [], "human": []}
        missing = 0
        for h in humans:
            test = scored & (r["human"] == h)
            outer = [load_logits(run_dir, f"{arm}__{h}__s{s}") for s in seeds]
            if any(l is None for l in outer):
                missing += 1
                continue
            temp = 1.0                       # nested temperature: fitted on the training humans, each held out in turn
            inner_l, inner_y = [], []
            for h2 in [x for x in humans if x != h]:
                l2 = load_logits(run_dir, f"{arm}__{h}__in_{h2}__s{seeds[0]}")
                if l2 is not None:
                    m2 = scored & (r["human"] == h2)
                    inner_l.append(l2[m2])
                    inner_y.append(y[m2])
            if inner_l:
                temp = fit_temperature(np.concatenate(inner_l), np.concatenate(inner_y))
            p1 = np.mean([softmax(l[test], 1.0) for l in outer], axis=0)             # 3-seed average
            pn = np.mean([softmax(l[test], temp) for l in outer], axis=0)
            s1, sn = score(np.log(p1 + 1e-12), y[test], LABELS), score(np.log(pn + 1e-12), y[test], LABELS)
            per_seed = [score(l[test], y[test], LABELS, temp) for l in outer]
            best_r, best_thr = recall_at_false(np.log(pn + 1e-12), y[test], BUDGET)
            res["folds"][h] = {
                "recall_t1": s1["recall"], "false_t1": s1["false"], "recall": sn["recall"], "false": sn["false"], "temp": temp,
                "hits": sn["hits"], "n_gesture": sn["n_gesture"], "false_n": sn["false_n"], "n_none": sn["n_none"],
                "per_class": sn["per_class"], "wrong_gesture": sn["wrong_gesture"], "silent": sn["silent"],
                "recall_seeds": [p["recall"] for p in per_seed], "false_seeds": [p["false"] for p in per_seed],
                "oracle_recall_at_budget": best_r, "oracle_thr": best_thr, "confusion": sn["confusion"].tolist(),
            }
            pooled["t1"].append(np.log(p1 + 1e-12))
            pooled["nested"].append(np.log(pn + 1e-12))
            pooled["y"].append(y[test])
            pooled["human"].append(r["human"][test])
        if pooled["y"]:
            yy, hh = np.concatenate(pooled["y"]), np.concatenate(pooled["human"])
            for key in ("t1", "nested"):
                lg = np.concatenate(pooled[key])
                s = score(lg, yy, LABELS)
                res[f"pooled_{key}"] = {"recall": s["recall"], "recall_ci": list(s["recall_ci"]), "false": s["false"],
                                        "false_ci": list(s["false_ci"]), "hits": s["hits"], "n_gesture": s["n_gesture"],
                                        "false_n": s["false_n"], "n_none": s["n_none"], "per_class": s["per_class"],
                                        "wrong_gesture": s["wrong_gesture"], "silent": s["silent"],
                                        "oracle_recall_at_budget": recall_at_false(lg, yy, BUDGET)[0]}
                clean = hh != "lance"
                if clean.any() and (~clean).any():
                    sc = score(lg[clean], yy[clean], LABELS)
                    res[f"pooled_{key}"]["no_lance"] = {"recall": sc["recall"], "false": sc["false"], "hits": sc["hits"],
                                                         "n_gesture": sc["n_gesture"]}
        # the locked session(s): never trained on; scored with the models of the fold that holds their human out
        res["locked"] = {}
        for name in sorted(set(r["session"][r["locked"]])):
            m = r["locked"] & (r["session"] == name) & np.isin(r["source"], ["gesture", "burst", "idle"])
            h = str(r["human"][m][0])
            outer = [load_logits(run_dir, f"{arm}__{h}__s{s}") for s in seeds]
            if any(l is None for l in outer) or h not in res["folds"]:
                continue
            temp = res["folds"][h]["temp"]
            lg = np.log(np.mean([softmax(l[m], temp) for l in outer], axis=0) + 1e-12)
            flip = np.array([0, 2, 1, 3])[y[m]]
            a, b = score(lg, y[m], LABELS), score(lg, flip, LABELS)
            res["locked"][str(name)] = {"as_prompted": {"recall": a["recall"], "hits": a["hits"], "n": a["n_gesture"], "per_class": a["per_class"]},
                                        "left_right_exchanged": {"recall": b["recall"], "hits": b["hits"], "n": b["n_gesture"], "per_class": b["per_class"]},
                                        "false": a["false"], "false_n": a["false_n"], "n_none": a["n_none"]}
        res["missing_folds"] = missing
        results[arm] = res

    def f3(v: float) -> str:
        return f"{v:.3f}"

    lines.append("# Leave-one-human-out scores (gestures recognised / non-gestures that trigger, at the engine's 0.6 threshold)")
    lines.append("")
    lines.append(f"Run `{run_dir.name}`. Each cell: 3-seed average of the model's probabilities. Temperature from the "
                 f"training humans only (nested). Budget for false triggers: {BUDGET}.")
    lines.append("")
    head = "| arm | pooled recall [95% CI] | pooled false | " + " | ".join(f"{h} recall / false" for h in humans) + \
           " | without lance | wrong / silent | recall at temperature 1 | best possible at budget |"
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for arm, res in results.items():
        if "pooled_nested" not in res:
            lines.append(f"| {arm} | (no complete fold) |")
            continue
        p, p1 = res["pooled_nested"], res["pooled_t1"]
        cells = []
        for h in humans:
            f = res["folds"].get(h)
            cells.append("-" if f is None else f"{f3(f['recall'])} ({f['hits']}/{f['n_gesture']}) / {f3(f['false'])} ({f['false_n']}/{f['n_none']})")
        nl = p.get("no_lance")
        lines.append(f"| {arm} | **{f3(p['recall'])}** [{p['recall_ci'][0]:.2f},{p['recall_ci'][1]:.2f}] ({p['hits']}/{p['n_gesture']}) | "
                     f"{f3(p['false'])} ({p['false_n']}/{p['n_none']}) | " + " | ".join(cells) +
                     f" | {'-' if nl is None else f3(nl['recall']) + ' / ' + f3(nl['false'])} | {p['wrong_gesture']} / {p['silent']} | "
                     f"{f3(p1['recall'])} / {f3(p1['false'])} | {f3(p['oracle_recall_at_budget'])} |")
    lines += ["", "## Per class (pooled, nested temperature): recognised / total", "",
              "| arm | swipe_left | swipe_right | push |", "|---|---|---|---|"]
    for arm, res in results.items():
        if "pooled_nested" in res:
            pc = res["pooled_nested"]["per_class"]
            lines.append(f"| {arm} | " + " | ".join(f"{pc[c][0]}/{pc[c][1]}" for c in LABELS[1:]) + " |")
    lines += ["", "## Spread over seeds (single models, nested temperature): recall per fold", ""]
    for arm, res in results.items():
        if res["folds"]:
            lines.append(f"- {arm}: " + "; ".join(f"{h} {' '.join(f3(v) for v in f['recall_seeds'])}" for h, f in res["folds"].items()))
    if any(res["locked"] for res in results.values()):
        lines += ["", "## Locked session(s): never trained on, never used to pick anything",
                  "", "| arm | session | recall as prompted | recall with left/right exchanged | false |", "|---|---|---|---|---|"]
        for arm, res in results.items():
            for name, v in res["locked"].items():
                a, b = v["as_prompted"], v["left_right_exchanged"]
                lines.append(f"| {arm} | {name} | {f3(a['recall'])} ({a['hits']}/{a['n']}) | {f3(b['recall'])} ({b['hits']}/{b['n']}) | "
                             f"{f3(v['false'])} ({v['false_n']}/{v['n_none']}) |")
    text = "\n".join(lines) + "\n"
    (run_dir / "report.md").write_text(text, encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(results, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    log("\n" + text, fh)
    return results


def pick_finalists(results: dict[str, Any], n: int = 3) -> list[str]:
    """Pre-declared rule: among arms whose pooled false-trigger share is within the budget, the highest pooled recall.

    If none is within the budget, the arms with the lowest false-trigger share. "released" is always included,
    so the old recipe is exported next to the new ones for a like-for-like check in the engine.
    """
    rows = [(a, r["pooled_nested"]) for a, r in results.items() if "pooled_nested" in r and not r["missing_folds"]]
    ok = sorted([x for x in rows if x[1]["false"] <= BUDGET], key=lambda x: -x[1]["recall"])
    rest = sorted([x for x in rows if x[1]["false"] > BUDGET], key=lambda x: x[1]["false"])
    names = [a for a, _ in (ok + rest)][:n]
    if "released" in results and "released" not in names:
        names.append("released")
    return names


# ---------------------------------------------------------------------------
# 7. Export the finalists the way the engine loads them
# ---------------------------------------------------------------------------
def export_final(run_dir: Path, arm: str, results: dict[str, Any], fh: Any) -> None:
    import torch

    sys.path.insert(0, str(REPO))
    from sonar.config import SonarConfig
    from sonar.model import GestureNet
    from sonar.train import check_onnx, export_onnx, fit_heuristic

    man = json.loads((run_dir / "manifest.json").read_text())
    seed = man["seeds"][0]
    pt = run_dir / "jobs" / f"{arm}__final__s{seed}.pt"
    if not pt.exists():
        log(f"final model of {arm} is missing ({pt.name})", fh)
        return
    state = torch.load(pt, map_location="cpu")
    width = int(state.get("width", 24))
    cfg = SonarConfig(n_mics=1)
    model = GestureNet(cfg.n_feat_channels, len(LABELS), width=width)
    model.load_state_dict(state["model"])
    model.eval()
    out = run_dir / "final" / arm
    out.mkdir(parents=True, exist_ok=True)
    torch.save(state, out / "gesture.pt")
    in_shape = (cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames)
    export_onnx(model, out / "gesture.onnx", in_shape)
    with np.load(run_dir / "table.npz", allow_pickle=False) as z:
        probe = np.ascontiguousarray(z["X"][:64].astype(np.float32)[..., 8:88])
        peak, src, lab, locked = z["peak_db"], z["source"], z["label"], z["locked"]
        sw = (src == "gesture") & np.isin(lab, ["swipe_left", "swipe_right"]) & ~locked
        x_sw = np.ascontiguousarray(z["X"][sw].astype(np.float32)[..., 8:88])
    diff, ms = check_onnx(out / "gesture.onnx", model, probe)
    # which way round this laptop's speakers are, for the engine's physics fallback (same rule as sonar.train)
    heur = fit_heuristic(x_sw, np.array([LABELS.index(str(v)) for v in lab[sw]]), list(LABELS), cfg)
    heur["source"] = "recorded"
    folds = results.get(arm, {}).get("folds", {})
    temps = [f["temp"] for f in folds.values()]
    gest = (src == "gesture") & np.isin(lab, LABELS[1:]) & ~locked
    meta = {
        "labels": list(LABELS),
        "temperature": float(np.exp(np.mean(np.log(temps)))) if temps else 1.0,      # geometric mean of the folds' temperatures
        "input_shape": list(in_shape),
        "config": cfg.to_dict(),
        "heuristic": heur,
        "gesture_peak_db_p10": round(float(np.percentile(peak[gest], 10)), 1) if gest.any() else None,
        "metrics": {"split": "none (trained on everything); honest score = leave-one-human-out, see report.md",
                    "arm": arm, "lopo": results.get(arm, {}).get("pooled_nested"), "onnx_max_logit_diff": diff,
                    "onnx_ms_per_inference": round(ms, 3)},
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
    }
    (out / "gesture.json").write_text(json.dumps(meta, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    log(f"exported {out.relative_to(REPO)}: ONNX vs PyTorch max logit difference {diff:.2e}, temperature {meta['temperature']:.2f}", fh)


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------
def main_run(args: argparse.Namespace) -> int:
    size = SIZES["smoke" if args.smoke else "full"]
    stamp = args.stamp or time.strftime("%Y%m%d-%H%M%S")
    run_dir = WORK / "suite" / (("smoke_" if args.smoke else "") + stamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    fh = open(run_dir / "log.txt", "a", encoding="utf-8")
    t_start = time.time()
    try:
        log(f"=== recall suite {stamp} ({'smoke' if args.smoke else 'full'}), device {args.device}; log: {run_dir / 'log.txt'}", fh)
        t4 = load_train_4090()
        info = t4.check_machine(args.device, t4.Log(run_dir / "machine.txt"))
        gen_device = args.gen_device if args.gen_device != "auto" else ("cuda" if args.device == "cuda" else "cpu")
        workers = args.workers or (4 if args.device == "cuda" else 1)
        cpu_workers = args.cpu_workers or min(os.cpu_count() or 4, 61)
        if gen_device != "cpu" and not args.skip_parity:
            ns = argparse.Namespace(size="tiny" if args.smoke else "full", gen_device=gen_device, gen_precision="auto", profile="default",
                                    recheck_parity=False, workers=cpu_workers, data=args.data)
            t4.parity_gate(ns, info, REPO / "train_out", t4.Log(run_dir / "parity.txt"))
            log("GPU-against-CPU picture check: passed (see parity.txt)", fh)

        from .common import HUMAN

        data_root = (REPO / args.data).resolve()
        locked = [s for s in args.locked.split(",") if s]
        cols = build_table(data_root, run_dir, locked, fh)
        label_audit(cols, run_dir, fh)
        humans = [h for h in ("lance", "aarsh", "mahimn") if h in set(cols["human"][~cols["locked"]])]
        humans += sorted(set(cols["human"][~cols["locked"]]) - set(humans))

        log("\n=== synthetic data", fh)
        tag = "smoke" if args.smoke else "full"
        synth = {"old": str(generate(WORK / "synth" / f"old_{tag}", size["per_class"], 0, gen_device, cpu_workers, fh))}
        sessions = {str(p.parent.name): p.parent for p in data_root.rglob("session.json")}
        for k, h in enumerate(humans):
            bgs = []
            for name, folder in sorted(sessions.items()):
                meta = json.loads((folder / "session.json").read_text())
                if meta.get("script") in ("scene", "breathing") and HUMAN.get(meta.get("person"), meta.get("person")) == h:
                    bgs.append(str(folder / meta.get("audio", "audio.wav")))
            if bgs:
                synth[f"bg_{h}"] = str(generate(WORK / "synth" / f"bg_{h}_{tag}", size["bg_per_class"], 1 + k, gen_device, cpu_workers, fh, backgrounds=bgs))
            else:
                log(f"no scene / breathing recordings of {h}: no background set for them", fh)
        # calibrated sets from the simulator session, if they were generated on this machine (optional)
        for fam in ("macbook_v1", "macbook_v1_motion"):
            for fold in ["all"] + [f"no_{h}" for h in humans]:
                p = REPO / "train_out" / "synth_cal" / fam / fold / "dataset.npz"
                if p.exists():
                    synth[f"{fam}_{fold}"] = str(p)
        arms = {a: ARMS[a] for a in (args.arms.split(",") if args.arms else ARMS) if a in ARMS}
        if args.smoke and not args.arms:
            arms = {a: ARMS[a] for a in ("released", "real_only", "share50_aux_speed", "share50_none50", "share50_w48", "cal50", "cal50_oldbg")}
        folds_all = ["all"] + [f"no_{h}" for h in humans]

        def needs(spec: dict[str, Any]) -> set[str]:
            out: set[str] = set()
            for name in str(spec.get("synth") or "").split("+"):
                if "{fold}" in name:
                    out |= {name.replace("{fold}", f) for f in folds_all}
            if str(spec.get("init") or "").startswith("cal:"):
                out |= {f"{spec['init'][4:]}_{f}" for f in folds_all}
            return out

        for a in list(arms):
            missing = sorted(needs(arms[a]) - set(synth))
            if missing:
                log(f"arm {a}: skipped, its synthetic sets are not on this machine ({', '.join(missing[:2])}{' ...' if len(missing) > 2 else ''})", fh)
                del arms[a]
        seeds = list(size["seeds"])
        man = {"stamp": stamp, "humans": humans, "seeds": seeds, "arms": list(arms), "arm_specs": arms, "synth": synth,
               "labels": list(LABELS), "locked": locked, "size": size, "machine": info, "git": t4.git_state()}
        (run_dir / "manifest.json").write_text(json.dumps(man, indent=1, default=str))

        # pre-training on the plain synthetic set, once per width
        widths = sorted({int(a.get("width", 24)) for a in arms.values() if a.get("init") == "old"})
        pre_jobs = [{"id": f"pretrain_old_w{w}", "real": "official", "synth": "old", "mix": "class_balanced", "train_humans": [],
                     "synth_fold": "all", "seed": 0, "epochs": size["pre_epochs"], "steps": size.get("steps", 0), "width": w,
                     "save_model": True} for w in widths]
        for fam, w in sorted({(a["init"][4:], int(a.get("width", 24))) for a in arms.values() if str(a.get("init") or "").startswith("cal:")}):
            pre_jobs += [{"id": f"pretrain_{fam}_{f}_w{w}", "real": "official", "synth": f"{fam}_{f}", "mix": "class_balanced",
                          "train_humans": [], "synth_fold": f, "seed": 0, "epochs": size["pre_epochs"], "steps": size.get("steps", 0),
                          "width": w, "save_model": True} for f in folds_all]
        run_jobs(run_dir, pre_jobs, args.device, workers, fh, "pretrain")

        def job(arm: str, spec: dict[str, Any], test: str | None, inner: str | None, seed: int, final: bool = False) -> dict[str, Any]:
            train_humans = [h for h in humans if h not in (test, inner)]
            jid = f"{arm}__final__s{seed}" if final else f"{arm}__{test}__s{seed}" if inner is None else f"{arm}__{test}__in_{inner}__s{seed}"
            w = int(spec.get("width", 24))
            fold_name = "all" if final else f"no_{test}"
            init = spec.get("init")
            init_file = (f"pretrain_old_w{w}.pt" if init == "old" else f"pretrain_{init[4:]}_{fold_name}_w{w}.pt" if init else None)
            return {"id": jid, "arm": arm, "real": spec["real"], "synth": spec["synth"], "mix": spec["mix"], "train_humans": train_humans,
                    "synth_fold": fold_name, "seed": seed, "epochs": size["epochs"],
                    "steps": size.get("steps", 0), "width": w,
                    "lr_scale": spec.get("lr_scale", 1.0), "aug": spec.get("aug"), "none_share": spec.get("none_share"),
                    "init": str(run_dir / "jobs" / init_file) if init_file else None, "save_model": final}

        jobs = []
        for arm, spec in arms.items():
            for h in humans:
                jobs += [job(arm, spec, h, None, s) for s in seeds]
                jobs += [job(arm, spec, h, h2, seeds[0]) for h2 in humans if h2 != h]
        run_jobs(run_dir, jobs, args.device, workers, fh, "folds")
        results = make_report(run_dir, fh)

        finalists = pick_finalists(results)
        log(f"\n=== finalists (rule: highest pooled recall with pooled false <= {BUDGET}; 'released' always): {finalists}", fh)
        run_jobs(run_dir, [job(a, arms[a], None, None, s, final=True) for a in finalists for s in seeds], args.device, workers, fh, "final")
        for a in finalists:
            export_final(run_dir, a, results, fh)
        (run_dir / "finalists.json").write_text(json.dumps(finalists))
        log(f"\nDONE in {(time.time() - t_start) / 60:.1f} min. Everything is in {run_dir.relative_to(REPO)}", fh)
        return 0
    except Exception:
        log(f"\nSTOPPED by an error:\n{traceback.format_exc()}", fh)
        return 1
    finally:
        fh.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m recall.suite", description="Leave-one-human-out experiment suite (see the file header)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    r.add_argument("--gen-device", default="auto", choices=["auto", "cpu", "cuda"])
    r.add_argument("--data", default="mac", help="folder with the recorded sessions")
    r.add_argument("--locked", default="20261114-001248_mahimn_gestures",
                   help="sessions that are never trained on and are scored separately (comma-separated)")
    r.add_argument("--arms", default="", help="only these arms (comma-separated; default: all)")
    r.add_argument("--workers", type=int, default=0, help="training runs sharing the GPU (default 4 on CUDA, else 1)")
    r.add_argument("--cpu-workers", type=int, default=0, help="CPU processes for the generator's random draws (default: all cores)")
    r.add_argument("--stamp", default="", help="reuse a run folder (finishes what is missing)")
    r.add_argument("--skip-parity", action="store_true")
    r.add_argument("--smoke", action="store_true", help="tiny sizes: checks that every step runs (its numbers mean nothing)")
    w = sub.add_parser("worker")
    w.add_argument("--run", required=True)
    w.add_argument("--jobs", required=True)
    w.add_argument("--device", default="cuda")
    p = sub.add_parser("report")
    p.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    if args.cmd == "run":
        return main_run(args)
    if args.cmd == "worker":
        return worker_main(Path(args.run), Path(args.jobs), args.device)
    make_report(Path(args.run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

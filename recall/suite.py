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
    python -m recall.suite run --device cuda                       # the 4090: about 3 hours (measured 2026-11-14)
    python -m recall.suite run --device cpu --smoke                # 2-minute self-test with junk-size data
    python -m recall.suite report --run train_out/recall/suite/<stamp>      # rebuild the report from saved answers
    python -m recall.suite finals --run <run folder> --arms real_pre --device cpu
                                                                   # only the models that ship: the named recipes trained
                                                                   # on everyone, in an existing run folder (minutes)

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
BUDGET = 0.0551           # false-trigger share of the released model on mahimn (6/109 = 0.05505): the level to stay at or under
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
#   epochs fine-tuning epochs (default 60, as released); lr_scale multiplies the released learning rate
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
    # --- combinations around the likely-good region (one 4090 run is cheap, a round trip is not)
    "share25_aux":        {"real": "aux_none", "synth": "old_bg", "mix": 0.25, "init": "old"},
    "share75_aux":        {"real": "aux_none", "synth": "old_bg", "mix": 0.75, "init": "old"},
    "real_pre_aux":       {"real": "aux_none", "synth": None, "mix": "class_balanced", "init": "old"},
    "share75_aux_speed":  {"real": "aux_none", "synth": "old_bg", "mix": 0.75, "init": "old", "aug": {"speed": [0.5, 2.2], "speed_p": 0.8}},
    "real_pre_aux_speed": {"real": "aux_none", "synth": None, "mix": "class_balanced", "init": "old", "aug": {"speed": [0.5, 2.2], "speed_p": 0.8}},
    "share50_aux_none50": {"real": "aux_none", "synth": "old_bg", "mix": 0.50, "init": "old", "none_share": 0.5},
    "share50_aux_e20":    {"real": "aux_none", "synth": "old_bg", "mix": 0.50, "init": "old", "epochs": 20},
    "share50_aux_lr03":   {"real": "aux_none", "synth": "old_bg", "mix": 0.50, "init": "old", "lr_scale": 0.3},
    # --- the same recordings through a +-750 Hz Doppler window instead of +-250 Hz (recall/wide.py). Real recordings
    #     only, because the synthetic sets are cut with the +-250 Hz window; each has its +-250 Hz twin to compare with.
    "real_aux":        {"real": "aux_none", "synth": None, "mix": "class_balanced", "init": None},
    "real_only_wide":  {"real": "official", "synth": None, "mix": "class_balanced", "init": None, "features": "wide"},
    "real_aux_wide":   {"real": "aux_none", "synth": None, "mix": "class_balanced", "init": None, "features": "wide"},
    # --- the same recordings with the burst ended 12 dB below its own peak (recall/relcut.py) instead of at the engine's
    #     1.4 s limit, which chops 111 of 218 recorded gestures. "table": "rel12" = pictures and bursts cut that way.
    #     The old synthetic sets keep the released cutting (their faint hands rarely reach the limit anyway).
    "real_only_rel12":      {"real": "official", "synth": None, "mix": "class_balanced", "init": None, "table": "rel12"},
    "real_aux_rel12":       {"real": "aux_none", "synth": None, "mix": "class_balanced", "init": None, "table": "rel12"},
    "real_pre_aux_rel12":   {"real": "aux_none", "synth": None, "mix": "class_balanced", "init": "old", "table": "rel12"},
    "share50_aux_rel12":    {"real": "aux_none", "synth": "old_bg", "mix": 0.50, "init": "old", "table": "rel12"},
    "share75_aux_rel12":    {"real": "aux_none", "synth": "old_bg", "mix": 0.75, "init": "old", "table": "rel12"},
    "real_aux_wide_rel12":  {"real": "aux_none", "synth": None, "mix": "class_balanced", "init": None, "features": "wide", "table": "rel12"},
    # --- the same, and the segmenter's click rejection switched off ("rel12nc"). That test also fires on very fast hands:
    #     with it on, the loudest moment of the trial is outside the labelled burst for 27 of lance's 80 gestures and 8 of
    #     mahimn's 45 fast ones; with it off for 1 and 1, and the stroke's peak sits 11-13 frames before the picture's end.
    "real_aux_rel12nc":     {"real": "aux_none", "synth": None, "mix": "class_balanced", "init": None, "table": "rel12nc"},
    "real_pre_aux_rel12nc": {"real": "aux_none", "synth": None, "mix": "class_balanced", "init": "old", "table": "rel12nc"},
    "share50_aux_rel12nc":  {"real": "aux_none", "synth": "old_bg", "mix": 0.50, "init": "old", "table": "rel12nc"},
    "share75_aux_rel12nc":  {"real": "aux_none", "synth": "old_bg", "mix": 0.75, "init": "old", "table": "rel12nc"},
    # --- the simulator fitted to this MacBook (train_out/synth_cal/<family>/<fold>/dataset.npz, made by the simulator
    #     session's handoff). "CAL" = the newest family there (macbook_v2, else macbook_v1); "{fold}" = the set that
    #     leaves out the held-out human. Skipped automatically while the files are missing.
    "cal50":           {"real": "official", "synth": "CAL_{fold}", "mix": 0.50, "init": "cal:CAL"},
    "cal25":           {"real": "official", "synth": "CAL_{fold}", "mix": 0.25, "init": "cal:CAL"},
    "cal50_aux":       {"real": "aux_none", "synth": "CAL_{fold}", "mix": 0.50, "init": "cal:CAL"},
    "cal_pre_real":    {"real": "official", "synth": None, "mix": "class_balanced", "init": "cal:CAL"},
    "calm50":          {"real": "official", "synth": "CAL_motion_{fold}", "mix": 0.50, "init": "cal:CAL_motion"},
    "calm50_aux":      {"real": "aux_none", "synth": "CAL_motion_{fold}", "mix": 0.50, "init": "cal:CAL_motion"},
    "calbg50":         {"real": "official", "synth": "CAL_bg_{fold}", "mix": 0.50, "init": "cal:CAL_bg"},
    "calbg50_aux":     {"real": "aux_none", "synth": "CAL_bg_{fold}", "mix": 0.50, "init": "cal:CAL_bg"},
    "calbg25_aux":     {"real": "aux_none", "synth": "CAL_bg_{fold}", "mix": 0.25, "init": "cal:CAL_bg"},
    # --- both together: recordings AND calibrated synthetic clips cut with the peak-relative rule (sets generated with
    #     SONAR_SEG_REL_OFF_DB=12 into train_out/synth_cal/<family>_rel12/ and <family>_bg_rel12/)
    "calbg50_aux_rel12": {"real": "aux_none", "synth": "CAL_bg_rel12_{fold}", "mix": 0.50, "init": "cal:CAL_bg_rel12", "table": "rel12"},
    "calbg25_aux_rel12": {"real": "aux_none", "synth": "CAL_bg_rel12_{fold}", "mix": 0.25, "init": "cal:CAL_bg_rel12", "table": "rel12"},
    "calbg75_aux_rel12": {"real": "aux_none", "synth": "CAL_bg_rel12_{fold}", "mix": 0.75, "init": "cal:CAL_bg_rel12", "table": "rel12"},
    "cal50_aux_rel12":   {"real": "aux_none", "synth": "CAL_rel12_{fold}", "mix": 0.50, "init": "cal:CAL_rel12", "table": "rel12"},
}
SIZES = {
    "full": {"per_class": 2000, "bg_per_class": 500, "epochs": 60, "pre_epochs": 60, "seeds": (0, 1, 2), "parity_clips": 1000},
    "smoke": {"per_class": 5, "bg_per_class": 3, "epochs": 1, "pre_epochs": 1, "seeds": (0,), "parity_clips": 100, "steps": 2},
}


def rel(path: Path) -> str:
    """A path as you would type it from the repo folder (or unchanged if it is elsewhere)."""
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


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
def table_file(variant: str) -> str:
    return "table.npz" if variant == "std" else f"table_{variant}.npz"


def build_table(data_root: Path, run_dir: Path, locked: list[str], fh: Any, flip: list[str] | None = None,
                wide: bool = False, variant: str = "std") -> dict[str, np.ndarray]:
    """Every burst of every session as arrays (saved to <run>/table.npz for the workers).

    `flip`: sessions in which the person swept the opposite way to the prompts: their swipe_left / swipe_right
    labels are exchanged here, before anything trains on them. ["auto"] = the sessions the swipe audit finds
    opposite (t below FLIP_T); a session that was corrected at the source then simply is not found any more.
    """
    from . import cache_streams
    from .common import HUMAN, load_sessions
    from .data import full_table

    if variant == "std":                                       # bursts cut exactly as the engine cuts them
        cache_streams.main_for(data_root)
        sessions = load_sessions()
    elif variant.startswith("rel"):                            # bursts ended N dB below their own peak (recall/relcut.py)
        from . import relcut

        rel_db, clicks = relcut.parse(variant)
        relcut.main_for(data_root, rel_db, clicks)
        sessions = load_sessions(root=relcut.out_dir(rel_db, clicks))
    else:
        raise ValueError(f"unknown table variant {variant!r}")
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
    if wide:                                                   # the same rows cut from the +-750 Hz feature stream
        from . import wide as wide_mod
        from .data import cut_feat

        wide_mod.main_for(data_root)
        xw = np.zeros((len(t), 2, wide_mod.WIDE["doppler_bins"], t.X.shape[-1]), np.float16)
        for name in sorted(set(t.session)):
            feat = wide_mod.load(str(name))
            for i in np.nonzero(t.session == name)[0]:
                nb = int(t.extra["not_before"][i])
                xw[i] = cut_feat(feat, int(t.extra["end_frame"][i]), None if nb < 0 else nb)
        cols["X_wide"] = xw
    cols["label_recorded"] = cols["label"].copy()
    if flip == ["auto"]:                                       # the sessions whose swipes measurably go the opposite way
        flip = [name for name, v in swipe_audit(cols).items() if v["t"] < FLIP_T]
    flipped = np.isin(cols["session"], flip or [])
    cols["flipped"] = flipped
    swap = {"swipe_left": "swipe_right", "swipe_right": "swipe_left"}
    for key in ("label", "prompt"):
        cols[key] = np.array([swap.get(str(v), str(v)) if f else str(v) for v, f in zip(cols[key], flipped)])
    done = sorted(set(cols["session"][flipped]))
    if done:
        log(f"\nleft/right labels EXCHANGED in {done} (the person swept the opposite way to the prompts; see the label audit below)", fh)
    target = run_dir / table_file(variant)
    if target.exists():                # a run folder is being continued: its saved answers are indexed by row, so the rows must not move
        with np.load(target, allow_pickle=False) as z:
            same = all(k in z.files and z[k].shape == cols[k].shape and bool(np.all(z[k] == cols[k])) for k in ("session", "seg", "label", "source"))
        if not same:
            raise RuntimeError(f"{target.name} in this run folder was built from other sessions, labels or code than now; "
                               "its saved answers would no longer line up. Start a new run (leave out --stamp).")
    np.savez_compressed(target, **cols)
    log(f"\ntable ({variant}): {len(t)} rows from {len(sessions)} sessions; humans {sorted(set(t.human))}; "
        f"locked (never trained on, scored separately): {[s for s in locked if s in present] or 'none'}", fh)
    for name in sorted(set(t.session)):
        m = t.session == name
        counts = {k: int(((t.source == k) & m).sum()) for k in ("gesture", "aux", "burst", "idle")}
        log(f"  {name:<40} {t.person[m][0]:<8} {counts}", fh)
    return cols


SHIFTS = (-6, -3, 3, 6)   # frames (20 ms each) by which the picture is cut later (-) or earlier (+) for the extra answers
FLIP_T = -4.0          # a session whose left/right statistic is below this has its swipes the opposite way round


def swipe_audit(cols: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Per gesture session: do the recorded swipes go the way their prompts say?

    Uses the physics cue (left-speaker tone Doppler minus right-speaker tone Doppler over the strongest half
    second, sonar.gestures.heuristic_classify) on the labels AS RECORDED. t = (mean over "swipe_left" prompts
    minus mean over "swipe_right" prompts) / its standard error. Clearly positive = as prompted; clearly
    negative = the person swept the opposite way; near 0 = no clean direction cue (lance's very fast swipes).
    """
    sys.path.insert(0, str(REPO))
    from sonar.config import SonarConfig
    from sonar.gestures import heuristic_classify

    cfg = SonarConfig(n_mics=1)
    recorded = cols.get("label_recorded", cols["label"])
    sw = np.isin(recorded, ["swipe_left", "swipe_right"]) & (cols["source"] == "gesture")
    cue = np.full(len(sw), np.nan)
    for i in np.nonzero(sw)[0]:
        cue[i] = heuristic_classify(cols["X"][i].astype(np.float32)[:, :, 8:88], cfg).lateral_hz
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(set(cols["session"][sw])):
        m = sw & (cols["session"] == name)
        left, right = cue[m & (recorded == "swipe_left")], cue[m & (recorded == "swipe_right")]
        if min(len(left), len(right)) < 4:
            continue
        se = np.sqrt(left.var() / len(left) + right.var() / len(right)) + 1e-9
        out[str(name)] = {"t": float((left.mean() - right.mean()) / se), "left_hz": float(left.mean()), "right_hz": float(right.mean()),
                          "left_sd": float(left.std()), "right_sd": float(right.std()), "n": int(m.sum())}
    return out


def label_audit(cols: dict[str, np.ndarray], run_dir: Path, fh: Any) -> dict[str, Any]:
    """Print and save the swipe audit, with what was done about each session."""
    out = swipe_audit(cols)
    lines = ["label audit (labels as recorded): left-speaker tone minus right-speaker tone, Hz (positive = hand moving LEFT)",
             f"  {'session':<40} {'n':>3} {'left prompts':>16} {'right prompts':>16} {'t':>7}  verdict"]
    for name, v in out.items():
        t = v["t"]
        verdict = "as prompted" if t > 3 else "OPPOSITE to the prompts" if t < -3 else "unclear (no clean direction cue)"
        if bool(cols["flipped"][cols["session"] == name].any()):
            verdict += "  -> left/right labels exchanged for this run"
        elif t < -3:
            verdict += "  -> NOT exchanged (--flip none, or t is not below the limit): check this session"
        v["verdict"] = verdict
        lines.append(f"  {name:<40} {v['n']:3d} {v['left_hz']:+8.1f}±{v['left_sd']:5.1f} {v['right_hz']:+8.1f}±{v['right_sd']:5.1f} {t:+7.2f}  {verdict}")
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
        log(f"reusing {rel(target)}", fh)
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
    """The pictures a worker needs. The real recordings stay on the training device; a synthetic set is loaded
    when a job asks for it and at most two stay loaded (the calibrated sets are about 0.7 GB each, there are
    twelve, and four workers share one GPU)."""

    KEEP = 2

    def __init__(self, run_dir: Path, device: str) -> None:
        from .trainlib import pick_device

        self.dev = pick_device(device)
        self.run_dir = run_dir
        self.man = json.loads((run_dir / "manifest.json").read_text())
        self._tables: dict[str, dict[str, Any]] = {}
        self._sets: dict[str, tuple[Any, np.ndarray]] = {}          # name -> (pictures on the device, labels), newest last

    def table(self, variant: str | None) -> dict[str, Any]:
        """The recorded rows of one cutting ("std" = the engine's): columns, labels, pictures on the device."""
        import torch

        variant = variant or "std"
        if variant not in self._tables:
            with np.load(self.run_dir / table_file(variant), allow_pickle=False) as z:
                real = {k: z[k] for k in z.files}
            lut = {n: i for i, n in enumerate(LABELS)}
            self._tables[variant] = {
                "real": real,
                "y": np.array([lut.get(str(v), 0) for v in real["label"]], dtype=np.int64),
                "x": torch.from_numpy(np.ascontiguousarray(real["X"], dtype=np.float16)).to(self.dev),
                "x_wide": torch.from_numpy(real["X_wide"]).to(self.dev) if "X_wide" in real else None,
            }
        return self._tables[variant]

    def synth(self, name: str) -> tuple[Any, np.ndarray]:
        import torch

        if name in self._sets:
            self._sets[name] = self._sets.pop(name)                 # mark as most recently used
            return self._sets[name]
        if name not in self.man["synth"]:
            raise KeyError(f"synthetic set {name!r} is not on this machine (have {sorted(self.man['synth'])})")
        while len(self._sets) >= self.KEEP:
            self._sets.pop(next(iter(self._sets)))
        with np.load(self.man["synth"][name], allow_pickle=False) as z:
            d = {k: z[k] for k in ("X", "y", "labels")}
        self._sets[name] = (torch.from_numpy(np.ascontiguousarray(d["X"], dtype=np.float16)).to(self.dev), remap(d))
        return self._sets[name]

    def real_mask(self, mode: str, humans: list[str], r: dict[str, np.ndarray]) -> np.ndarray:
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

    def synth_names(self, job: dict[str, Any]) -> list[str]:
        """The synthetic sets this job trains on."""
        names: list[str] = []
        for name in str(job.get("synth") or "").split("+"):
            if not name:
                continue
            if name == "old_bg":
                names += [f"bg_{h}" for h in job["train_humans"] if f"bg_{h}" in self.man["synth"]]
            else:
                names.append(name.replace("{fold}", job["synth_fold"]))
        return names

    def assemble(self, job: dict[str, Any]) -> tuple[Any, Any, np.ndarray]:
        """(pictures, labels, sampling weights) for one job: the real rows it may use, then its synthetic sets."""
        import torch

        t = self.table(job.get("table"))
        real = self.real_mask(job["real"], job["train_humans"], t["real"])
        parts_x, parts_y, is_real = [t["x"]], [t["y"]], [real]
        used = [real]
        for name in self.synth_names(job):
            x, y = self.synth(name)
            parts_x.append(x)
            parts_y.append(y)
            is_real.append(np.zeros(len(y), bool))
            used.append(np.ones(len(y), bool))
        y_np = np.concatenate(parts_y)
        real_m = np.concatenate(is_real)
        synth_m = np.concatenate(used) & ~np.concatenate([np.ones(len(t["y"]), bool)] + [np.zeros(len(v), bool) for v in parts_y[1:]])
        none_share = job.get("none_share")

        def balanced(mask: np.ndarray) -> np.ndarray:
            out = np.zeros(len(y_np))
            if mask.any():
                counts = np.bincount(y_np[mask], minlength=len(LABELS)).astype(float)
                mass = np.ones(len(LABELS))                      # every class equally often ...
                if none_share is not None:                       # ... unless "none" gets its own share
                    mass[:] = (1.0 - float(none_share)) / (len(LABELS) - 1)
                    mass[0] = float(none_share)
                out[mask] = (mass / np.maximum(counts, 1.0))[y_np[mask]]
                out /= out.sum()
            return out

        if job["mix"] == "class_balanced" or not synth_m.any() or not real_m.any():
            w = balanced(real_m | synth_m)
        else:
            share = float(job["mix"])
            w = share * balanced(real_m) + (1.0 - share) * balanced(synth_m)
        x = parts_x[0] if len(parts_x) == 1 else torch.cat(parts_x)
        return x, torch.from_numpy(y_np).to(self.dev), w


def run_job(bank: Bank, job: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    import torch

    from .trainlib import CFG, TrainSpec, logits_of, train

    t0 = time.time()
    x, y, w = bank.assemble(job)
    tab = bank.table(job.get("table"))
    n_real = len(tab["y"])
    spec = TrainSpec(epochs=job["epochs"], steps=int(job.get("steps", 0)), width=job.get("width", 24), seed=job["seed"],
                     lr_scale=job.get("lr_scale", 1.0), device=str(bank.dev), aug=job.get("aug") or {})
    init = None
    if job.get("init"):
        state = torch.load(job["init"], map_location="cpu")
        init = state["model"] if isinstance(state, dict) and "model" in state else state
    if job.get("features") == "wide":                         # real recordings only, +-750 Hz pictures
        from .wide import wide_config

        if tab["x_wide"] is None:
            raise RuntimeError("this run's table has no wide pictures")
        model = train(tab["x_wide"], y[:n_real], torch.from_numpy(w[:n_real]), LABELS, spec, init=None, cfg=wide_config(CFG))
        logits = logits_of(model, tab["real"]["X_wide"], device=str(bank.dev))
    else:
        model = train(x, y, torch.from_numpy(w), LABELS, spec, init=init, cfg=CFG)
        logits = logits_of(model, tab["real"]["X"], device=str(bank.dev))
    del x, y
    # the same model's answers when the picture is cut a little earlier or later (the table keeps 8 spare frames on each
    # side): lets the analysis test "average over small time shifts" without the GPU
    key = "X_wide" if job.get("features") == "wide" else "X"
    shifted = np.stack([logits_of(model, tab["real"][key], pad=8 + k, device=str(bank.dev)) for k in SHIFTS])
    # ... and its answers on the SAME recordings cut the other way (engine cutting <-> peak-relative), when this run has
    # both tables: a model trained on one cutting can then be scored as it would run in an engine using the other
    cross = {}
    for variant in bank.man.get("tables", []):
        if variant != (job.get("table") or "std"):
            other = bank.table(variant)["real"]
            if key in other:
                cross[f"logits__{variant}"] = logits_of(model, other[key], device=str(bank.dev)).astype(np.float32)
    np.savez_compressed(out_dir / f"{job['id']}.npz", logits=logits.astype(np.float32), logits_shifted=shifted.astype(np.float16),
                        shifts=np.array(SHIFTS), **cross)
    info = {"id": job["id"], "seconds": round(time.time() - t0, 1), "n_train": int((w > 0).sum()),
            "real_share": float(w[:n_real].sum())}
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
    todo = sorted(todo, key=lambda j: (str(j.get("synth")), str(j.get("synth_fold")), j["id"]))
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
    lost = [j["id"] for j in todo if not (run_dir / "jobs" / f"{j['id']}.npz").exists()]
    if lost:              # a worker process died (out of GPU memory, killed): say so instead of reporting on half the runs
        log(f"WARNING: {len(lost)} of these runs produced no result (first: {lost[:3]}). Run the same command again with "
            f"--stamp {run_dir.name.replace('smoke_', '')}: it only does what is missing.", fh)


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
    lut = {n: i for i, n in enumerate(LABELS)}
    tables: dict[str, tuple] = {}

    def table_of(arm: str) -> tuple:
        """(columns, labels, scored-rows mask) of the cutting this arm was trained and answered on."""
        variant = (man.get("arm_specs", {}).get(arm, {}) or {}).get("table") or "std"
        if variant not in tables:
            with np.load(run_dir / table_file(variant), allow_pickle=False) as z:
                cols = {k: z[k] for k in z.files if k not in ("X", "X_wide")}
            yy = np.array([lut.get(str(v), 0) for v in cols["label"]], dtype=np.int64)
            tables[variant] = (cols, yy, np.isin(cols["source"], ["gesture", "burst", "idle"]) & ~cols["locked"])
        return tables[variant]

    humans, seeds = man["humans"], man["seeds"]
    results: dict[str, Any] = {}
    lines: list[str] = []
    for arm in man["arms"]:
        r, y, scored = table_of(arm)                 # "scored" = the released definition of an example
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
    """One finalist as gesture.onnx + gesture.json + gesture.pt (seed 0; all seeds' weights stay in jobs/ for the
    packaging step). A recipe on the wide Doppler window or the peak-relative cutting is exported for THAT input
    (its gesture.json says so under "needs"): the engine can only load it once it has that profile."""
    import torch

    sys.path.insert(0, str(REPO))
    from sonar.config import SonarConfig
    from sonar.model import GestureNet
    from sonar.train import check_onnx, export_onnx, fit_heuristic

    from .wide import wide_config

    man = json.loads((run_dir / "manifest.json").read_text())
    spec = man.get("arm_specs", {}).get(arm, {}) or {}
    wide, variant = spec.get("features") == "wide", spec.get("table") or "std"
    seed = man["seeds"][0]
    pt = run_dir / "jobs" / f"{arm}__final__s{seed}.pt"
    if not pt.exists():
        log(f"final model of {arm} is missing ({pt.name})", fh)
        return
    state = torch.load(pt, map_location="cpu")
    width = int(state.get("width", 24))
    base = SonarConfig(n_mics=1)
    cfg = wide_config(base) if wide else base
    model = GestureNet(cfg.n_feat_channels, len(LABELS), width=width)
    model.load_state_dict(state["model"])
    model.eval()
    out = run_dir / "final" / arm
    out.mkdir(parents=True, exist_ok=True)
    torch.save(state, out / "gesture.pt")
    in_shape = (cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames)
    export_onnx(model, out / "gesture.onnx", in_shape)
    with np.load(run_dir / table_file(variant), allow_pickle=False) as z:
        probe = np.ascontiguousarray(z["X_wide" if wide else "X"][:64].astype(np.float32)[..., 8:88])
    with np.load(run_dir / "table.npz", allow_pickle=False) as z:      # speaker wiring and gesture strength: the engine's own cutting
        peak, src, lab, locked = z["peak_db"], z["source"], z["label"], z["locked"]
        sw = (src == "gesture") & np.isin(lab, ["swipe_left", "swipe_right"]) & ~locked
        x_sw = np.ascontiguousarray(z["X"][sw].astype(np.float32)[..., 8:88])
    diff, ms = check_onnx(out / "gesture.onnx", model, probe)
    # which way round this laptop's speakers are, for the engine's physics fallback (same rule as sonar.train)
    heur = fit_heuristic(x_sw, np.array([LABELS.index(str(v)) for v in lab[sw]]), list(LABELS), base)
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
    if wide or variant != "std":
        meta["needs"] = {"doppler_window": "+-750 Hz (decim 12, stft_win 256, stft_fft 512, doppler_bins 192, demod_pass_hz 800)" if wide else "engine default",
                         "burst_cutting": (f"ends {variant[3:].replace('nc', '')} dB below the burst's own peak"
                                           + (", click rejection off" if variant.endswith("nc") else "")) if variant != "std" else "engine default",
                         "note": "the engine cannot produce this input yet; see recall/wide.py and recall/relcut.py"}
    (out / "gesture.json").write_text(json.dumps(meta, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    log(f"exported {rel(out)}: input {list(in_shape)}, ONNX vs PyTorch max logit difference {diff:.2e}, "
        f"temperature {meta['temperature']:.2f}" + ("  (needs the wide window / peak-relative cutting in the engine)" if "needs" in meta else ""), fh)


def make_job(run_dir: Path, humans: list[str], size: dict[str, Any], arm: str, spec: dict[str, Any], test: str | None,
             inner: str | None, seed: int, final: bool = False) -> dict[str, Any]:
    """One training run of a recipe: `test` held out (and `inner` too, for choosing the confidence setting), or,
    with final=True, trained on everyone (the model that ships; its weights are saved)."""
    train_humans = [h for h in humans if h not in (test, inner)]
    jid = f"{arm}__final__s{seed}" if final else f"{arm}__{test}__s{seed}" if inner is None else f"{arm}__{test}__in_{inner}__s{seed}"
    w = int(spec.get("width", 24))
    fold_name = "all" if final else f"no_{test}"
    init = spec.get("init")
    init_file = (f"pretrain_old_w{w}.pt" if init == "old" else f"pretrain_{init[4:]}_{fold_name}_w{w}.pt" if init else None)
    return {"id": jid, "arm": arm, "real": spec["real"], "synth": spec["synth"], "mix": spec["mix"], "train_humans": train_humans,
            "synth_fold": fold_name, "seed": seed, "epochs": min(int(spec.get("epochs", size["epochs"])), size["epochs"]),
            "steps": size.get("steps", 0), "width": w,
            "lr_scale": spec.get("lr_scale", 1.0), "aug": spec.get("aug"), "none_share": spec.get("none_share"),
            "features": spec.get("features"), "table": spec.get("table"),
            "init": str(run_dir / "jobs" / init_file) if init_file else None, "save_model": final}


def main_finals(args: argparse.Namespace) -> int:
    """Train and export ONLY the models that ship (each named recipe trained on everyone, every seed) in an
    existing run folder. Nothing else in the folder is redone or changed: the held-out runs, the tables and the
    report stay as they are. For a recipe that trains on the recordings alone this is three small runs."""
    run_dir = Path(args.run).resolve()
    man = json.loads((run_dir / "manifest.json").read_text())
    fh = open(run_dir / "log.txt", "a", encoding="utf-8")
    try:
        arms = [a for a in args.arms.split(",") if a]
        log(f"\n=== finals only: {arms} trained on everyone ({man['humans']}), seeds {man['seeds']}, device {args.device}", fh)
        jobs = []
        for a in arms:
            spec = man.get("arm_specs", {}).get(a)
            if spec is None:
                log(f"ERROR: {a} is not a recipe of this run (see manifest.json)", fh)
                return 1
            for j in [make_job(run_dir, man["humans"], man["size"], a, spec, None, None, s, final=True) for s in man["seeds"]]:
                sets = [n.replace("{fold}", "all") for n in str(j.get("synth") or "").split("+") if n]
                sets = [x for n in sets for x in ([f"bg_{h}" for h in man["humans"]] if n == "old_bg" else [n])]
                gone = [n for n in sets if n not in man["synth"] or not Path(man["synth"][n]).exists()]
                if gone or (j["init"] and not Path(j["init"]).exists()):
                    log(f"ERROR: {a} needs files that are not on this machine: {gone or j['init']}. Run this on the machine that made the run folder.", fh)
                    return 1
                jobs.append(j)
        run_jobs(run_dir, jobs, args.device, args.workers or (4 if args.device == "cuda" else 1), fh, "final")
        results = json.loads((run_dir / "results.json").read_text()) if (run_dir / "results.json").exists() else {}
        import platform

        import torch

        for a in arms:
            export_final(run_dir, a, results, fh)
            # where these final networks were trained (the held-out runs of the same folder may come from another machine)
            (run_dir / "final" / a / "built.json").write_text(json.dumps({
                "device": args.device, "machine": f"{platform.system()} {platform.machine()}", "torch": torch.__version__,
                "built": time.strftime("%Y-%m-%d %H:%M:%S"), "run_made_on": man.get("machine", {}).get("gpu") or "see machine.txt"}, indent=1))
        log(f"finals done: {[rel(run_dir / 'final' / a) for a in arms]}", fh)
        return 0
    except Exception:
        log(f"\nSTOPPED by an error:\n{traceback.format_exc()}", fh)
        return 1
    finally:
        fh.close()


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
        if os.environ.pop("SONAR_SEG_REL_OFF_DB", None) is not None:      # this suite sets the burst cutting per recipe itself
            log("NOTE: SONAR_SEG_REL_OFF_DB was set in this shell; ignored here (the rel12 recipes cut their own pictures)", fh)
        if gen_device != "cpu" and not args.skip_parity:
            ns = argparse.Namespace(size="tiny" if args.smoke else "full", gen_device=gen_device, gen_precision="auto", profile="default",
                                    recheck_parity=False, workers=cpu_workers, data=args.data)
            try:
                t4.parity_gate(ns, info, REPO / "train_out", t4.Log(run_dir / "parity.txt"))
                log("GPU-against-CPU picture check: passed (see parity.txt)", fh)
            except t4.StepFailed as exc:
                # never train on GPU-made data that failed the check; the CPU generator is the reference, only slower
                log(f"GPU-against-CPU picture check did NOT pass ({str(exc).splitlines()[0]}). The synthetic data is made with the "
                    "CPU generator instead (about 15 minutes longer). Please send parity.txt back with the results.", fh)
                gen_device = "cpu"

        from .common import HUMAN

        data_root = (REPO / args.data).resolve()
        locked = [s for s in args.locked.split(",") if s]
        flip = [] if args.flip == "none" else [s for s in args.flip.split(",") if s]
        want = [a for a in (args.arms.split(",") if args.arms else ARMS) if a in ARMS]
        if args.smoke and not args.arms:
            want = [a for a in ARMS if a in ("released", "real_aux_wide", "share50_aux_rel12", "real_aux_wide_rel12")]
        def wants_wide(variant: str | None) -> bool:           # the +-750 Hz pictures are only cut where a recipe uses them
            return args.smoke or any(ARMS[a].get("features") == "wide" and ARMS[a].get("table") == variant for a in want)

        cols = build_table(data_root, run_dir, locked, fh, flip, wide=wants_wide(None))
        flip = sorted({str(v) for v in cols["session"][cols["flipped"]]})            # what "auto" found; the other cuttings use the same
        variants = sorted({ARMS[a]["table"] for a in want if ARMS[a].get("table")})
        for variant in variants:
            build_table(data_root, run_dir, locked, fh, flip, variant=variant, wide=wants_wide(variant))
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
        cal_root = REPO / "train_out" / "synth_cal"
        versions = sorted({int(d.name.split("_")[1][1:]) for d in cal_root.glob("macbook_v*") if d.is_dir() and d.name.split("_")[1][1:].isdigit()})
        cal = f"macbook_v{versions[-1]}" if versions else "macbook_v1"
        for fam in (cal, f"{cal}_motion", f"{cal}_bg", f"{cal}_rel12", f"{cal}_bg_rel12", f"{cal}_motion_rel12"):
            for fold in ["all"] + [f"no_{h}" for h in humans]:
                p = cal_root / fam / fold / "dataset.npz"
                if p.exists():
                    synth[f"{fam}_{fold}"] = str(p)
        if versions:
            log(f"calibrated simulator data: family {cal} ({sum(1 for k in synth if k.startswith(cal))} sets found)", fh)

        def resolve(spec: dict[str, Any]) -> dict[str, Any]:
            return {k: (v.replace("CAL", cal) if isinstance(v, str) else v) for k, v in spec.items()}

        arms = {a: resolve(ARMS[a]) for a in (args.arms.split(",") if args.arms else ARMS) if a in ARMS}
        if args.smoke and not args.arms:
            arms = {a: resolve(ARMS[a]) for a in ("released", "real_only", "share50_aux_speed", "share50_none50", "share50_w48",
                                                  "cal50", "calbg50_aux", "real_aux_wide", "share50_aux_rel12", "real_aux_wide_rel12")}
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
        old_man = run_dir / "manifest.json"
        if old_man.exists():                                  # same run folder again: keep the arms already done there
            prev = json.loads(old_man.read_text())
            if prev.get("humans") == humans and prev.get("seeds") == seeds:
                keep = {a: sp for a, sp in prev.get("arm_specs", {}).items() if a not in arms}
                if keep:
                    log(f"this run folder already holds {sorted(keep)}: kept in the report", fh)
                arms = {**keep, **arms}
        man = {"stamp": stamp, "humans": humans, "seeds": seeds, "arms": list(arms), "arm_specs": arms, "synth": synth,
               "labels": list(LABELS), "locked": locked, "flip": flip, "size": size, "machine": info, "git": t4.git_state(),
               "tables": ["std"] + sorted({sp["table"] for sp in arms.values() if sp.get("table")} | set(variants))}
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
            return make_job(run_dir, humans, size, arm, spec, test, inner, seed, final)

        jobs = []
        for arm, spec in arms.items():
            for h in humans:
                jobs += [job(arm, spec, h, None, s) for s in seeds]
                jobs += [job(arm, spec, h, h2, seeds[0]) for h2 in humans if h2 != h]
        run_jobs(run_dir, jobs, args.device, workers, fh, "folds")
        results = make_report(run_dir, fh)

        # finalists: the best three recipes the engine can load today, plus the best two overall if those need the
        # wide window or the peak-relative cutting (their weights then come back too, so no second round trip)
        ready = pick_finalists({a: r for a, r in results.items() if arms[a].get("features") != "wide" and not arms[a].get("table")})
        finalists = ready + [a for a in pick_finalists(results, n=2) if a not in ready]
        finalists += [a for a in args.finals.split(",") if a in arms and a not in finalists]        # asked for by name
        log(f"\n=== finalists (rule: highest pooled recall with pooled false <= {BUDGET}; 'released' always): {finalists}", fh)
        run_jobs(run_dir, [job(a, arms[a], None, None, s, final=True) for a in finalists for s in seeds], args.device, workers, fh, "final")
        for a in finalists:
            export_final(run_dir, a, results, fh)
        (run_dir / "finalists.json").write_text(json.dumps(finalists))
        log(f"\nDONE in {(time.time() - t_start) / 60:.1f} min. Everything is in {rel(run_dir)}", fh)
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
    r.add_argument("--locked", default="",
                   help="sessions that are never trained on and are scored separately (comma-separated; default none)")
    r.add_argument("--flip", default="auto",
                   help="sessions whose swipes go the opposite way to their prompts get their left/right labels exchanged: "
                        "auto (default) = the sessions the label audit finds opposite, t below -4 (today: mahimn's first "
                        "session, t = -7.1); or session names, comma-separated; or none")
    r.add_argument("--arms", default="", help="only these arms (comma-separated; default: all)")
    r.add_argument("--finals", default="", help="also train and export these arms on everyone (comma-separated), whatever their rank")
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
    f = sub.add_parser("finals", help="train and export only the models that ship, in an existing run folder")
    f.add_argument("--run", required=True, help="the run folder (train_out/recall/suite/<time>)")
    f.add_argument("--arms", required=True, help="recipes to train on everyone (comma-separated)")
    f.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    f.add_argument("--workers", type=int, default=0, help="training runs at a time (default 4 on CUDA, else 1)")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(REPO))
    if args.cmd == "run":
        return main_run(args)
    if args.cmd == "finals":
        return main_finals(args)
    if args.cmd == "worker":
        return worker_main(Path(args.run), Path(args.jobs), args.device)
    make_report(Path(args.run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

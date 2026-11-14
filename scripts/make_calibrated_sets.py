"""make_calibrated_sets.py — synthetic training sets from the laptop fitted to the real recordings, one per scoring fold.

What it does
    The honest score tests the model on one HUMAN at a time, with that human left out of
    training (lance and lance2 are the same human). This script makes, for every such fold,
    a synthetic training set that knows nothing about the held-out human:
      1. it sorts the recorded sessions under --data by human;
      2. it makes sure every fold has its fitted laptop:
             calibration/macbook_v<N>.json            everyone (the newest N) (fold "all": for the final model that ships)
             calibration/macbook_v<N>_no_<human>.json how people move, fitted WITHOUT that human
         Missing fold files are fitted first (python -m datagen.calibration --folds; a few minutes on the GPU);
      3. it generates, per arm and fold,
             train_out/synth_cal/macbook_v<N>/<fold>/dataset.npz           gestures from the formulas
             train_out/synth_cal/macbook_v<N>_motion/<fold>/dataset.npz    + half of the gestures replay real tracked hand paths
             train_out/synth_cal/macbook_v<N>_bg/<fold>/dataset.npz        + 70 % of the clips are laid over the real
                                                                         "nobody gesturing" recordings (never the held-out human's)
         with <fold> = all, no_lance, no_aarsh, no_mahimn (from the people found), each with its
         manifest.jsonl next to it. Each arm adds one thing to the one before, so their scores
         show what that thing is worth;
      4. it writes train_out/synth_cal/INDEX.json: per set the calibration file, the recordings
         used, the number of clips and pictures, the seconds it took.

Why
    The recall session's leave-one-human-out suite reads exactly these folders (a missing one is
    skipped there). Drawing a fold's clips around the held-out human's own habits, or laying
    them over that human's own recordings, would flatter the score.

How to run (on the GPU box, from the repo folder; `mac` is where the recorded sessions are)
    .venv\\Scripts\\python scripts\\make_calibrated_sets.py --data mac
    more: --per-class 2000 --negatives 2 --arms plain,motion,bg --folds all,no_lance --force --dry-run

What it changes
    Writes train_out/synth_cal/** and, when missing, calibration/macbook_v<N>_no_<human>.json plus a
    report next to them. Only reads the recordings. A finished set is skipped on the next run
    (so a stopped run can simply be started again); --force makes everything again.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
MOTION_LIBRARY = "datagen/motion_library/unica_v1.npz"
BACKGROUND_SCRIPTS = ("scene", "breathing")      # recordings with nobody gesturing
# arm -> (what is added to the calibration's name for its folder, share of gestures that replay real hand paths,
#         share of clips laid over real recordings). Each arm adds one thing to the one before.
ARMS = {"plain": ("", 0.0, 0.0), "motion": ("_motion", 0.5, 0.0), "bg": ("_bg", 0.5, 0.7)}

sys.path.insert(0, str(REPO))


def show(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


def run(cmd: list[str]) -> float:
    """Run one step with its output on the screen. Returns the seconds it took; stops the script if the step fails."""
    print(f"\n$ {show(cmd)}", flush=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    # one worker process per core already: the maths libraries must not start threads of their own
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    t0 = time.time()
    code = subprocess.call(cmd, cwd=REPO, env=env)
    if code != 0:
        raise SystemExit(f"this step failed with exit code {code}: {show(cmd)}")
    return time.time() - t0


def rel(path: Path) -> str:
    """A path as it is written in the index: relative to the repo when it is inside it."""
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def sessions(root: Path) -> list[dict]:
    """Every readable recorded session under `root`: who, which script, where the audio is."""
    rows = []
    for meta_path in sorted(root.rglob("session.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            audio = meta_path.parent / meta.get("audio", "audio.wav")
            if meta.get("synthetic") or not audio.exists() or audio.stat().st_size < 100_000:
                continue
            if "excluded" in {part.lower() for part in meta_path.parent.relative_to(root).parts}:
                print(f"  not used: {meta_path.parent.name} (it is in a folder named 'excluded')")
                continue
            rows.append({"session": meta_path.parent.name, "person": str(meta.get("person", "unknown")),
                         "script": str(meta.get("script", "?")), "audio": str(audio)})
        except (OSError, ValueError) as exc:
            print(f"  skipping {meta_path.parent.name}: cannot read it ({type(exc).__name__})")
    return rows


def main(argv: list[str] | None = None) -> int:
    from datagen.calibration import fold_path, human_of, newest

    ap = argparse.ArgumentParser(description="Calibrated synthetic sets, one per leave-one-human-out fold (see the file header)")
    ap.add_argument("--data", default="data", help="folder with the recorded sessions (default data; the MacBook ones are in mac)")
    ap.add_argument("--calibration", default="auto", help="the everyone file; fold files sit next to it "
                                                          "(auto: the newest calibration/macbook_v<N>.json)")
    ap.add_argument("--out", default="train_out/synth_cal", help="where the sets go")
    ap.add_argument("--per-class", type=int, default=2000, help="clips per gesture in every set (default 2000)")
    ap.add_argument("--negatives", type=float, default=2.0, help="no-gesture clips, as a multiple of --per-class (default 2)")
    ap.add_argument("--arms", default="plain,motion,bg", help="plain = formula gestures; motion = half of them replay real hand paths; "
                                                             "bg = motion + most clips laid over real recordings")
    ap.add_argument("--folds", default="", help="only these folds, e.g. all,no_lance (default: all of them)")
    ap.add_argument("--bg-share", type=float, default=ARMS["bg"][2],
                    help="arm bg: share of the clips laid over the real no-gesture recordings of the fold's own humans")
    ap.add_argument("--gen-device", default="auto", choices=["auto", "cpu", "cuda", "mps", "torch-cpu"],
                    help="auto = cuda when there is one, else the numpy code on the CPU cores (slow)")
    ap.add_argument("--profile", default=None, help="signal profile of the recordings, as in datagen.generate (default: the core default)")
    ap.add_argument("--gen-batch", type=int, default=0, help="clips per GPU batch (default: fills the GPU's memory)")
    ap.add_argument("--gen-kernel", default="auto", choices=["auto", "on", "off"], help="as in datagen.generate")
    ap.add_argument("--workers", type=int, default=0, help="CPU processes (default: every core)")
    ap.add_argument("--fit-clips", type=int, default=100, help="trial clips per gesture and setting when fold files must be fitted")
    ap.add_argument("--refit-folds", action="store_true", help="fit the fold files again even if they exist")
    ap.add_argument("--force", action="store_true", help="make every set again")
    ap.add_argument("--dry-run", action="store_true", help="only print what would be made")
    args = ap.parse_args(argv)

    if args.gen_device == "auto":
        import torch

        args.gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    workers = args.workers or min(os.cpu_count() or 1, 61 if sys.platform == "win32" else 10**6)
    base = newest(REPO / "calibration") if args.calibration == "auto" else (REPO / args.calibration).resolve()
    if base is None or not base.exists():
        raise SystemExit(f"no calibration file ({base or 'calibration/macbook_v<N>.json'}); "
                         f"python -m datagen.calibration --real {args.data} --out calibration/macbook_v1.json makes one")
    print(f"fitted laptop: {rel(base)}")
    data = (REPO / args.data).resolve()
    rows = sessions(data) if data.exists() else []
    if not rows:
        raise SystemExit(f"no recorded sessions under {data} (give --data; the MacBook sessions were in `mac`)")
    humans = sorted({human_of(r["person"]) for r in rows if r["script"] == "gestures"})
    print(f"{len(rows)} recorded sessions under {data}; humans with gestures: "
          + ", ".join(f"{h} ({'+'.join(sorted({r['person'] for r in rows if human_of(r['person']) == h}))})" for h in humans))
    folds = [("all", None)] + [(f"no_{h}", h) for h in humans]
    if args.folds:
        wanted = [f.strip() for f in args.folds.split(",") if f.strip()]
        unknown = [f for f in wanted if f not in dict(folds)]
        if unknown:
            raise SystemExit(f"unknown fold(s) {unknown}; there are {[f for f, _ in folds]}")
        folds = [(f, h) for f, h in folds if f in wanted]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if any(a not in ARMS for a in arms):
        raise SystemExit(f"--arms takes {list(ARMS)}")
    motion = (REPO / MOTION_LIBRARY).resolve()
    if any(ARMS[a][1] > 0 for a in arms) and not motion.exists():
        raise SystemExit(f"{motion} is missing (it comes with git pull); or use --arms plain")

    # 2. the fitted laptop of every fold
    cal_of = {name: (base if human is None else fold_path(base, human)) for name, human in folds}
    missing = [name for name, path in cal_of.items() if not path.exists()]
    if (missing or args.refit_folds) and any(h is not None for _, h in folds):
        print(f"\n=== fitting how people move, once without each human ({'again' if not missing else ', '.join(missing) + ' missing'})")
        cmd = [PY, "-m", "datagen.calibration", "--real", str(data), "--folds", str(base), "--device", args.gen_device,
               "--clips", str(args.fit_clips), "--workers", str(workers)]
        if args.dry_run:
            print(f"(dry run) $ {show(cmd)}")
        else:
            run(cmd)
    out_root = (REPO / args.out).resolve()
    index_path = out_root / "INDEX.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"sets": {}}
    index.update(calibration=rel(base), data=str(data),
                 humans={h: sorted({r["person"] for r in rows if human_of(r["person"]) == h}) for h in humans})

    # 3. one set per arm and fold
    total = 0.0
    for a, arm in enumerate(arms):
        for f, (fold, human) in enumerate(folds):
            cal = cal_of[fold]
            suffix, motion_share, bg_share = ARMS[arm]
            bg_share = args.bg_share if arm == "bg" else bg_share
            name = f"{base.stem}{suffix}/{fold}"
            folder = out_root / f"{base.stem}{suffix}" / fold
            if not cal.exists() and not args.dry_run:
                # never fall back to the everyone file here: that would put the held-out human's habits into the fold
                print(f"\n{name}: SKIPPED, {cal.name} does not exist (the fold fit above did not write it)")
                continue
            backgrounds = sorted(r["audio"] for r in rows if r["script"] in BACKGROUND_SCRIPTS
                                 and (human is None or human_of(r["person"]) != human)) if bg_share > 0 else []
            if bg_share > 0 and not backgrounds:
                print(f"\n{name}: SKIPPED, no scene / breathing recordings to lay the clips over")
                continue
            seed = 20 + 10 * list(ARMS).index(arm) + [n for n, _ in [("all", None)] + [(f"no_{h}", h) for h in humans]].index(fold)
            cmd = [PY, "-m", "datagen.generate", "--out", str(folder), "--per-class", str(args.per_class), "--negatives", str(args.negatives),
                   "--workers", str(workers), "--seed", str(seed), "--mics", "1", "--calibration", str(cal)]
            if args.profile:
                cmd += ["--profile", args.profile]
            if motion_share > 0:
                cmd += ["--motion", str(motion), "--motion-share", str(motion_share)]
            if backgrounds:
                cmd += ["--background", ",".join(backgrounds), "--background-share", str(bg_share)]
            if args.gen_device != "cpu":
                cmd += ["--gen-device", args.gen_device, "--gen-kernel", args.gen_kernel]
                if args.gen_batch:
                    cmd += ["--gen-batch", str(args.gen_batch)]
            settings = {"calibration": cal.name, "calibration_sha256": None, "per_class": args.per_class, "negatives": args.negatives,
                        "seed": seed, "motion_share": motion_share, "background_share": bg_share,
                        "backgrounds": [Path(b).parent.name for b in backgrounds], "held_out_human": human,
                        "profile": args.profile}
            if cal.exists():
                import hashlib

                settings["calibration_sha256"] = hashlib.sha256(cal.read_bytes()).hexdigest()[:16]
            done = folder / "done.json"
            if done.exists() and not args.force and json.loads(done.read_text()).get("settings") == settings:
                print(f"\n{name}: already made ({json.loads(done.read_text()).get('examples')} pictures); --force makes it again")
                index["sets"][name] = json.loads(done.read_text())
                continue
            if args.dry_run:
                print(f"\n{name}: (dry run) $ {show(cmd)}")
                continue
            print(f"\n=== {name}: laptop {cal.name}" + (f", not {human}'s movement habits" if human else ", everyone")
                  + (f", {motion_share:.0%} real hand paths" if motion_share else "")
                  + (f", {bg_share:.0%} of the clips over {len(backgrounds)} real recordings" if backgrounds else ""))
            if folder.exists():
                shutil.rmtree(folder)
            seconds = run(cmd)
            import numpy as np

            with np.load(folder / "dataset.npz", allow_pickle=False) as z:
                labels = [str(s) for s in z["labels"]]
                counts = {labels[i]: int((z["y"] == i).sum()) for i in range(len(labels))}
            shutil.rmtree(folder / "backgrounds", ignore_errors=True)       # the generator's working copies of the recordings
            if (folder / "audio").exists() and not any((folder / "audio").iterdir()):
                (folder / "audio").rmdir()
            info = {"dataset": rel(folder / "dataset.npz"), "examples": int(sum(counts.values())), "per_label": counts,
                    "clips": int(args.per_class * (4 + args.negatives)), "seconds": round(seconds, 1), "device": args.gen_device,
                    "made": time.strftime("%Y-%m-%d %H:%M"), "settings": settings}
            done.write_text(json.dumps(info, indent=1))
            index["sets"][name] = info
            total += seconds
            out_root.mkdir(parents=True, exist_ok=True)
            index_path.write_text(json.dumps(index, indent=1))
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, indent=1))
        print(f"\n{'set':<34}{'pictures':>9}{'seconds':>9}  laptop")
        for name, info in sorted(index["sets"].items()):
            print(f"{name:<34}{info['examples']:>9}{info['seconds']:>9.0f}  {info['settings']['calibration']}")
        print(f"\nDONE: {len(index['sets'])} set(s) under {out_root} ({total / 60:.1f} min of generation in this run); index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

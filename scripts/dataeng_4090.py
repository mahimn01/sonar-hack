#!/usr/bin/env python3
"""dataeng_4090.py — one command on the training box: why does the model miss the held-out person, and what helps?

What it does, in order (every step prints the exact command it runs, so any step can be re-run by hand):
  0. makes a corrected VIEW of the recordings (sonar.dataview: labels only, the audio is not copied):
     a session found twice is taken once, and the one session whose swipes went the opposite way
     to their prompts (20261114-001248_mahimn_gestures) gets its left/right labels exchanged.
     Everything below reads that view, so `--data mac` is safe although mac\\excluded\\ lies inside it;
  1. finds the model of the newest training run that left people out, and who they were
     (or takes --model / --people);
  2. replays those people's recordings through the real engine and counts recognised / wrong /
     missed gestures and the reason for every miss, at confidence thresholds 0.6, 0.5 and 0.4,
     with the engine's gates on and off, and with the other burst cutting (sonar.gesture_replay);
  3. checks the labels and the burst cutting of ALL recordings (sonar.labelqa), and draws
     two pictures of one gesture session of a held-out person;
  4. measures what a "each gesture 3 times" table calibration would give those people
     (sonar.adapt eval);
  5. checks that the fast batched code gives what the original code gives (sonar.weaklabel --parity);
  6. ONLY when asked for (the recall session's suite scores training recipes more thoroughly):
       * --train-weak     trains once more with the extra weighted examples (--weak);
       * --train-relcut   trains once more with the peak-relative burst cutting (SONAR_SEG_REL_OFF_DB=12);
     each about 12-20 minutes on the RTX 4090, each in its own folders;
  7. collects every printed block, report file and picture into handoff_results/dataeng/ and
     writes SUMMARY.txt with the handful of numbers that matter. That folder is what goes back.

How to run (from the repo root, on the 4090 box):
    .venv\\Scripts\\python scripts\\dataeng_4090.py --data mac                       # about 5 minutes
    .venv\\Scripts\\python scripts\\dataeng_4090.py --data mac --train-relcut        # + one training run
A step that fails is written down and the others still run.

What it changes: writes handoff_results/dataeng/ and train_out/dataeng/ (reports and the view), and for
the optional training runs train_out/rel12/ and train_out/weak_models, train_out/weak_release plus caches
under train_out/. It never writes to models/ or models-release*/, never changes a recording and never
opens an audio device.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
RESULTS = REPO / "handoff_results" / "dataeng"
REL_DB = "12"


def show(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


def run(name: str, cmd: list[str], env_extra: dict[str, str] | None = None, timeout_min: float = 60.0) -> tuple[int, str]:
    """Run one step, show its output live, keep a copy in handoff_results/dataeng/<name>.txt. Returns (exit code, output)."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", **(env_extra or {}))
    note = "".join(f"{k}={v} " for k, v in (env_extra or {}).items())
    print(f"\n=== {name}\n$ {note}{show(cmd)}", flush=True)
    t0 = time.time()
    lines: list[str] = []
    try:
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lines.append(line)
            if time.time() - t0 > timeout_min * 60:
                proc.kill()
                lines.append(f"\nSTOPPED after {timeout_min:.0f} minutes\n")
                break
        code = proc.wait()
    except OSError as exc:
        code, lines = 127, [f"could not start: {exc}\n"]
    text = "".join(lines)
    (RESULTS / f"{name}.txt").write_text(f"$ {note}{show(cmd)}\n(exit code {code}, {time.time() - t0:.0f} s)\n\n{text}", encoding="utf-8")
    if code != 0:
        print(f"!!! step {name} ended with exit code {code}; going on with the next step", flush=True)
    return code, text


def newest_held_out_run(work: Path) -> tuple[Path | None, list[str]]:
    """(model of the newest run that scored held-out people, those people), from train_out/runs/*/summary.json."""
    for folder in sorted((work / "runs").glob("*"), reverse=True):
        try:
            s = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        people = [str(p) for p in s.get("val_people") or []]
        model = Path(str(s.get("winner", {}).get("folder", ""))) / "gesture.onnx"
        if not model.is_absolute():
            model = REPO / model
        if people and model.exists() and model.with_suffix(".pt").exists():
            return model, people
    return None, []


def copy_if(src: Path, dst_name: str) -> None:
    if src.exists():
        shutil.copy2(src, RESULTS / dst_name)


def grab(text: str, start: str, n: int = 12) -> list[str]:
    """The lines of a printed block that begins with `start`."""
    rows = text.splitlines()
    for i, row in enumerate(rows):
        if row.strip().startswith(start):
            return rows[i:i + n]
    return []


def honest(text: str) -> list[str]:
    return [r for r in text.splitlines() if r.startswith(("Honest score", "Verdict", "DONE in", "STOPPED"))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dataeng_4090", description="Diagnose the recall on held-out people and measure the fixes")
    ap.add_argument("--data", nargs="+", default=["mac"], help="folder(s) with the recorded sessions (default mac)")
    ap.add_argument("--model", default="auto", help="model that did NOT train on --people (default: the sweep model of the newest run)")
    ap.add_argument("--people", default="", help="the held-out people (default: those of that run)")
    ap.add_argument("--out", default="train_out", help="the training working folder (default train_out)")
    ap.add_argument("--train-weak", action="store_true", help="also train once with the extra weighted examples (--weak)")
    ap.add_argument("--train-relcut", action="store_true", help="also train once with the peak-relative burst cutting")
    ap.add_argument("--flip-swipes", default="20261114-001248_mahimn_gestures",
                    help="sessions whose swipe_left / swipe_right labels are exchanged in the view (\"\" for none)")
    ap.add_argument("--drop", default="", help="sessions to leave out of the view")
    ap.add_argument("--train-args", default="", help="extra options for both training runs, e.g. \"--size small\" for a rehearsal")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    t_start = time.time()
    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    RESULTS.mkdir(parents=True)
    work = (REPO / args.out).resolve()
    reports = work / "dataeng"
    reports.mkdir(parents=True, exist_ok=True)
    summary: list[str] = [f"data-engine run {time.strftime('%Y-%m-%d %H:%M')} on {platform.node()} ({platform.system()} {platform.release()})"]

    # 0. the machine and the code
    code, text = run("00_machine", [PY, "-c", (
        "import torch, sys; ok = torch.cuda.is_available(); "
        "print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', ok, '|', torch.cuda.get_device_name(0) if ok else 'no CUDA GPU')")])
    summary.append(text.strip().splitlines()[-1] if text.strip() else "machine: unknown")
    git = subprocess.run(["git", "log", "-1", "--format=%h %s"], cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True, text=True,
                           encoding="utf-8", errors="replace").stdout.strip()
    summary.append(f"code: {git.stdout.strip() or 'unknown commit'}" + (" + uncommitted changes" if dirty else ""))

    # 1. which model, which people
    model = Path(args.model) if args.model != "auto" else None
    people = [p for p in args.people.split(",") if p]
    if model is None:
        model, found = newest_held_out_run(work)
        people = people or found
    if model is None or not model.exists():
        print(f"\nSTOPPED: no model that left people out under {work / 'runs'} (run the training first, or give --model and --people)")
        (RESULTS / "SUMMARY.txt").write_text("\n".join(summary + ["STOPPED: no held-out model found"]), encoding="utf-8")
        return 2
    if not people:
        print("\nSTOPPED: --model given without --people (who did this model NOT train on?)")
        return 2
    who = ",".join(people)
    summary.append(f"model: {model}   held-out: {who}")
    print(f"\nmodel that did not train on {who}: {model}")

    # 1b. a corrected view of the recordings: duplicates once, the known flipped session with its labels exchanged
    view = reports / "view"
    code, text = run("00_view", [PY, "-m", "sonar.dataview", *(str((REPO / d).resolve()) for d in args.data), "--out", str(view),
                                 "--flip-swipes", args.flip_swipes, "--drop", args.drop])
    if code != 0:
        print("\nSTOPPED: no usable session in the view (see 00_view.txt)")
        (RESULTS / "SUMMARY.txt").write_text("\n".join(summary + ["STOPPED: the view of the recordings is empty"]), encoding="utf-8")
        return 2
    summary += ["[00_view]"] + [r for r in text.splitlines() if "EXCHANGED" in r or "skipped" in r or r.startswith("view of")]
    data = str(view)

    # 2. replay through the engine: the shipped settings, then lower confidence thresholds, with and without the gates
    #    The last two cut the bursts the other way (peak-relative) with the SAME model: no retraining, so only a hint.
    rel_env = {"SONAR_SEG_REL_OFF_DB": REL_DB}
    for name, extra, env in (("01_replay", [], None), ("02_replay_conf05", ["--min-confidence", "0.5"], None),
                             ("02_replay_conf04", ["--min-confidence", "0.4"], None), ("02_replay_nogates", ["--no-gates"], None),
                             ("02_replay_nogates_conf05", ["--no-gates", "--min-confidence", "0.5"], None),
                             ("02_replay_relcut", [], rel_env), ("02_replay_relcut_nogates_conf05", ["--no-gates", "--min-confidence", "0.5"], rel_env)):
        code, text = run(name, [PY, "-m", "sonar.gesture_replay", data, "--model", str(model), "--people", who, *extra,
                                "--json", str(reports / f"{name}.json")], env)
        copy_if(reports / f"{name}.json", f"{name}.json")
        summary += [f"[{name}]" + (f"   (bursts cut with SONAR_SEG_REL_OFF_DB={REL_DB}, same model)" if env else "")] \
            + grab(text, "replayed through the engine", 14 if name == "01_replay" else 1 + len(people))

    # 3. labels and burst cutting of every recording; two pictures of a held-out gesture session
    code, text = run("03_labelqa", [PY, "-m", "sonar.labelqa", data, "--model", str(model), "--no-plot", "--out", str(reports / "qa")])
    copy_if(reports / "qa" / "labelqa_summary.json", "03_labelqa_summary.json")
    warned, session_row = [], ""
    for r in text.splitlines():
        if r[:1].isdigit():
            session_row = r.split(":")[0]
        elif r.strip().startswith("!!"):
            warned.append(f"  {session_row}: {r.strip()[3:].split('. ')[0]}")
    summary += ["[03_labelqa]"] + warned[:16] + grab(text, "===", 14)
    long_bursts = False
    try:
        cut = json.loads((reports / "qa" / "labelqa_summary.json").read_text(encoding="utf-8"))["totals"]["all"].get("cutting", {})
        if cut.get("gestures"):
            share = cut["cut_at_limit"] / cut["gestures"]
            loud = sorted(cut["peak_db"])[len(cut["peak_db"]) // 2]
            long_bursts = share >= 0.3 and loud >= 25.0
            summary.append(f"gesture bursts cut off at the length limit: {100 * share:.0f} %, typical strength {loud:.0f} dB -> long loud bursts: {long_bursts}")
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        summary.append("burst cutting: could not be read from the label check")
    session = None
    theirs: list[str] = []                                # every session folder of the held-out people
    for folder in sorted(Path(data).rglob("session.json")):
        try:
            meta = json.loads(folder.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(meta.get("person")) in people:
            theirs.append(str(folder.parent))
            if any(e.get("kind") == "gesture" for e in meta.get("events", [])):
                session = folder.parent                   # the last one found = the newest gesture session of a held-out person
    if session is not None:
        run("04_labelqa_pictures", [PY, "-m", "sonar.labelqa", str(session), "--model", str(model), "--out", str(reports / "qa")], timeout_min=10)
        for k in (1, 2, 3):
            copy_if(reports / "qa" / session.name / f"qa_p{k}.png", f"04_{session.name}_qa_p{k}.png")
        copy_if(reports / "qa" / session.name / "qa.json", f"04_{session.name}_qa.json")

    # 4. the table calibration, measured on the held-out people
    code, text = run("05_adapt_eval", [PY, "-m", "sonar.adapt", "eval", "--model", str(model), "--data", *(theirs or [data]), "--people", who,
                                       "--json", str(reports / "adapt_eval.json")])
    copy_if(reports / "adapt_eval.json", "05_adapt_eval.json")
    table = text.splitlines()
    first = next((i for i, r in enumerate(table) if r.strip().startswith("base") or "bn+proto" in r), None)
    summary += ["[05_adapt_eval]"] + (table[first:first + 3 + len(people) + 5] if first is not None else ["no table (see 05_adapt_eval.txt)"])

    # 5. fast path against the original code
    code, text = run("06_parity", [PY, "-m", "sonar.weaklabel", data, "--model", str(model), "--parity"], timeout_min=15)
    summary += ["[06_parity]"] + grab(text, "parity on", 7)
    parity_ok = "PARITY OK" in text

    # 6. the heavy runs
    extra = shlex.split(args.train_args, posix=os.name != "nt")
    trainer = [PY, "scripts/train_4090.py", "--data", data, "--val-person", who, *extra]
    if not (args.train_weak or args.train_relcut):
        summary.append("training runs: none asked for (--train-weak / --train-relcut)")
    elif not parity_ok:
        summary.append("training runs: NOT started, the parity check did not pass")
    else:
        if args.train_weak:
            code, text = run("07_train_weak", [*trainer, "--weak", "--out", args.out, "--models", f"{args.out}/weak_models",
                                               "--release", f"{args.out}/weak_release"], timeout_min=90)
            copy_if(work / "weak_release" / "TRAINING_RUN.json", "07_weak_TRAINING_RUN.json")
            copy_if(work / "weak" / "weak.json", "07_weak_report.json")
            summary += ["[07_train_weak]"] + [r for r in text.splitlines() if r.startswith(("time:", "wrote ")) and "weak" in r][:3] + honest(text)
        if args.train_relcut:
            has_parity_flag = "--skip-parity" in subprocess.run([PY, "scripts/train_4090.py", "--help"], cwd=REPO, capture_output=True,
                                                               text=True, encoding="utf-8", errors="replace").stdout
            rel_dir = f"{args.out}/rel{REL_DB}"                 # its own working folder: the caches do not know about the switch
            rel = [*trainer, "--out", rel_dir, "--models", f"{rel_dir}/models", "--release", f"{rel_dir}/release"]
            # the GPU-against-CPU check compares with a reference made with the usual cutting: skip it for this one run
            code, text = run("08_train_relcut", rel + (["--skip-parity"] if has_parity_flag else []), rel_env, timeout_min=90)
            rel_work = work / f"rel{REL_DB}"
            copy_if(rel_work / "release" / "TRAINING_RUN.json", "08_relcut_TRAINING_RUN.json")
            summary += [f"[08_train_relcut, SONAR_SEG_REL_OFF_DB={REL_DB}]"] + honest(text)
            new_model = rel_work / "models" / "gesture.onnx"
            if new_model.exists():
                code, text = run("09_labelqa_relcut", [PY, "-m", "sonar.labelqa", data, "--model", str(new_model), "--no-plot",
                                                       "--out", str(rel_work / "qa")], rel_env)
                copy_if(rel_work / "qa" / "labelqa_summary.json", "09_relcut_labelqa_summary.json")
                summary += ["[09_labelqa_relcut]"] + [r for r in text.splitlines() if "burst cutting" in r or "recognised in SOME" in r]

    summary.append(f"total: {(time.time() - t_start) / 60:.1f} minutes")
    (RESULTS / "SUMMARY.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n" + "=" * 78 + "\n" + "\n".join(summary))
    print("=" * 78)
    print(f"Everything to send back is in {RESULTS.relative_to(REPO)} ({sum(1 for _ in RESULTS.iterdir())} files). Push it:")
    print("  git switch -c dataeng-results")
    print("  git add handoff_results/dataeng")
    print('  git commit -m "Data-engine results from the 4090: replay, label check, calibration, burst cutting"')
    print("  git push -u origin dataeng-results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

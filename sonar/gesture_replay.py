"""gesture_replay.py — replay recorded sessions through the REAL engine and say, trial by trial, what happened.

Plain words: the training script scores pictures ("was the loudest burst of this prompt named
correctly?"). The demo is something else: the engine listens to the whole recording, cuts its
own bursts, applies its gates (too faint, somebody walking in, movement too far away), its
confidence threshold and its debounce rules, and either reports the right gesture, a wrong
one, or nothing. This tool plays each recorded session through that same engine, offline and
faster than real time, and counts per prompted (or camera-timed) gesture:

    recognised   the engine reported the right gesture in time
    wrong        it reported another gesture
    missed       it reported nothing, and WHY: no burst at all / blocked as too faint by the
                 engine's strength gate / blocked as "somebody walking" / the model said none /
                 the right gesture but below the confidence threshold / held back by the debounce

plus the gestures it reported where none was asked for (false triggers per minute, and during
what). The "why" column is the point: it says which knob would raise recall on a new person.

How to run:
    python -m sonar.gesture_replay data/ --model models/gesture.onnx
    python -m sonar.gesture_replay data/ --model train_out/runs/<run>/sweep/w24_s0/gesture.onnx --people mahimn
    python -m sonar.gesture_replay data/<session> --model models/gesture.onnx --min-peak-db 8 --min-confidence 0.5
Score people the model was NOT trained on (the sweep model of a training run and its held-out
people) for the honest number.

This is CPU work by nature (it IS the realtime engine); sessions are replayed in parallel
worker processes (--workers). What it changes: nothing, unless --json names a report file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .classifier import load_classifier
from .config import GESTURE_LABELS
from .dataset import find_sessions, load_session
from .engine import Engine
from .weaklabel import zone_of

REASONS = (
    "no movement burst",
    "blocked: too faint for the engine's gate",
    "blocked: taken for somebody walking in or past",
    "right gesture, confidence too low",
    "another gesture, below the threshold",
    "model said none",
    "held back by the debounce rules",
)


class _Recorded:
    """Just enough of an audio backend for the engine to be built around a recording."""

    name = "file"
    block = 1920

    def __init__(self, n_mics: int) -> None:
        self.n_mics = n_mics
        self.info = {"mode": "replay"}


def replay_session(folder: str | Path, model: str | Path | None, min_confidence: float = 0.6, min_peak_db: float | None = None,
                   gates: bool = True, block: int = 1920) -> dict[str, Any]:
    """Play one session through the engine. Returns the trials with their outcome and the false triggers."""
    folder = Path(folder)
    meta, audio, cfg = load_session(folder)
    engine = Engine(cfg, _Recorded(audio.shape[1]), min_confidence=min_confidence, emit_spectrogram=False)
    clf = load_classifier(model, engine.cfg)
    if getattr(clf, "mismatch", []):
        raise ValueError(f"{model} was trained with different signal settings for {clf.mismatch}")
    engine.set_classifier(clf)
    if not gates:
        engine.gate_entering = engine.gate_range = False
        engine.min_peak_db = engine.cfg.seg_min_peak_db
    if min_peak_db is not None:
        engine.min_peak_db = float(min_peak_db)
    known = set(getattr(clf, "labels", GESTURE_LABELS))
    fs = cfg.fs
    reported: list[tuple[float, str, float, float]] = []          # (time s, label, confidence, peak dB)
    decisions: list[tuple[float, str, float, float]] = []         # every burst the engine looked at, also the rejected ones
    last: Any = engine.decider.last_decision
    for i in range(0, len(audio), block):
        t = min(i + block, len(audio)) / fs
        for e in engine.step(audio[i:i + block]):
            if e["type"] == "gesture":
                reported.append((t, e["label"], float(e["confidence"]), float(e["peak_db"])))
        d = engine.decider.last_decision
        if d is not last and d:
            decisions.append((t, str(d.get("label")), float(d.get("confidence", 0.0)), float(d.get("peak_db", 0.0))))
            last = d
    trials = []
    used: set[int] = set()
    spans: list[tuple[float, float]] = []
    for ev in meta.get("events", []):
        t0, t1 = ev["start"] / fs, ev["end"] / fs
        if ev.get("kind") in ("block", "prompt"):
            spans.append((t0, t1 + 1.5))                          # gestures may happen here: nothing in it is a false trigger
        label = ev.get("label", "")
        if label not in GESTURE_LABELS or label == "none":
            continue
        a, b = (t0 - 0.3, t1 + 1.6) if ev.get("tight") else (t0 - 0.2, t1 + 1.3)      # the engine answers up to ~0.5 s after the hand stops
        spans.append((a, b))
        row: dict[str, Any] = {"label": label, "t": round(t0, 2), "source": ev.get("source", "prompt"), "rep": ev.get("rep")}
        if label not in known:
            row["outcome"] = "not a class of this model"
            trials.append(row)
            continue
        mine = [k for k, r in enumerate(reported) if a <= r[0] <= b and k not in used]
        right = [k for k in mine if reported[k][1] == label]
        seen = [d for d in decisions if a <= d[0] <= b]
        if right:
            used.add(right[0])
            row.update(outcome="recognised", confidence=round(reported[right[0]][2], 3), peak_db=round(reported[right[0]][3], 1),
                       delay_s=round(reported[right[0]][0] - t0, 2))
        elif mine:
            used.update(mine)
            row.update(outcome="wrong", said=[reported[k][1] for k in mine])
        else:
            row.update(outcome="missed", why=_why(label, seen, min_confidence))
            if seen:
                row["strongest_burst_db"] = round(max(d[3] for d in seen), 1)
        trials.append(row)
    false = []
    for k, r in enumerate(reported):
        if k in used or any(a <= r[0] <= b for a, b in spans):
            continue
        false.append({"t": round(r[0], 2), "label": r[1], "confidence": round(r[2], 3), "during": zone_of(int(r[0] * fs), meta)})
    quiet_s = max(len(audio) / fs - sum(b - a for a, b in spans), 0.0)
    return {
        "session": str(meta.get("session_id", folder.name)), "person": str(meta.get("person", "unknown")), "script": meta.get("script"),
        "seconds": round(len(audio) / fs, 1), "classifier": clf.source, "gate_db": round(float(engine.min_peak_db), 1),
        "trials": trials, "false_triggers": false, "seconds_without_prompts": round(quiet_s, 1), "bursts_seen": len(decisions),
    }


def _why(label: str, seen: list[tuple[float, str, float, float]], min_confidence: float) -> str:
    """Why the engine reported nothing for this trial, from the verdicts it reached on the bursts in the window.

    A prompt usually produces several bursts (wind-up, stroke, the hand going back). If any of them was read as
    the right gesture, that is the answer (too unsure, or dropped by the debounce). Otherwise the verdict on the
    STRONGEST burst counts: that one is the gesture; a faint side burst being gated says nothing about it.
    """
    if not seen:
        return REASONS[0]
    names = [d[1] for d in seen]
    if any(n == label and d[2] >= min_confidence for n, d in zip(names, seen)):
        return REASONS[6]
    if any(n == label for n in names):
        return REASONS[3]
    top = max(seen, key=lambda d: d[3])[1]
    if top == "blocked:weak":
        return REASONS[1]
    if top in ("blocked:entering", "blocked:far"):
        return REASONS[2]
    if top != "none":
        return REASONS[4]
    return REASONS[5]


def _job(args: tuple[str, str | None, float, float | None, bool]) -> dict[str, Any]:
    folder, model, min_confidence, min_peak_db, gates = args
    try:
        return replay_session(folder, model, min_confidence, min_peak_db, gates)
    except (OSError, RuntimeError, ValueError, KeyError, IndexError) as exc:
        return {"session": Path(folder).name, "error": f"{type(exc).__name__}: {exc}"}


def tally(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Per person and overall: recognised / wrong / missed (by reason), false triggers per minute."""
    def blank() -> dict[str, Any]:
        return {"trials": 0, "recognised": 0, "wrong": 0, "missed": 0, "why": {}, "false_triggers": 0, "minutes_without_prompts": 0.0,
                "by_gesture": {}, "false_during": {}}

    out: dict[str, Any] = {"all": blank(), "by_person": {}}
    for res in results:
        if "error" in res:
            continue
        for acc in (out["all"], out["by_person"].setdefault(res["person"], blank())):
            for t in res["trials"]:
                if t["outcome"] == "not a class of this model":
                    continue
                acc["trials"] += 1
                acc[t["outcome"]] += 1
                g = acc["by_gesture"].setdefault(t["label"], {"trials": 0, "recognised": 0})
                g["trials"] += 1
                g["recognised"] += t["outcome"] == "recognised"
                if t["outcome"] == "missed":
                    acc["why"][t["why"]] = acc["why"].get(t["why"], 0) + 1
            acc["false_triggers"] += len(res["false_triggers"])
            acc["minutes_without_prompts"] += res["seconds_without_prompts"] / 60.0
            for f in res["false_triggers"]:
                acc["false_during"][f["during"]] = acc["false_during"].get(f["during"], 0) + 1
    for acc in [out["all"], *out["by_person"].values()]:
        acc["recall"] = round(acc["recognised"] / acc["trials"], 4) if acc["trials"] else None
        acc["minutes_without_prompts"] = round(acc["minutes_without_prompts"], 2)
        acc["false_per_minute"] = round(acc["false_triggers"] / acc["minutes_without_prompts"], 3) if acc["minutes_without_prompts"] > 0.2 else None
    return out


def print_tally(t: dict[str, Any], gate_db: float | None, min_confidence: float) -> None:
    def line(name: str, a: dict[str, Any]) -> str:
        if not a["trials"]:
            fpm = "n/a" if a["false_per_minute"] is None else f"{a['false_per_minute']:.2f}"
            return f"  {name:<14} no prompted gestures; false triggers {a['false_triggers']} ({fpm} per minute)"
        fpm = "n/a" if a["false_per_minute"] is None else f"{a['false_per_minute']:.2f}/min"
        per = ", ".join(f"{g} {v['recognised']}/{v['trials']}" for g, v in sorted(a["by_gesture"].items()))
        return (f"  {name:<14} recognised {a['recognised']:>3}/{a['trials']:<3} ({100 * a['recall']:5.1f}%)   wrong {a['wrong']:>2}   missed {a['missed']:>3}   "
                f"false triggers {a['false_triggers']} ({fpm})   [{per}]")

    print(f"\nreplayed through the engine (confidence >= {min_confidence}" + (f", strength gate {gate_db} dB" if gate_db is not None else "") + "):")
    for person, a in sorted(t["by_person"].items()):
        print(line(person, a))
    if len(t["by_person"]) > 1:
        print(line("ALL", t["all"]))
    a = t["all"]
    if a["missed"]:
        print("why the missed ones were missed:")
        for reason in REASONS:
            if a["why"].get(reason):
                print(f"  {a['why'][reason]:>4}  {reason}")
        top = max(a["why"], key=lambda k: a["why"][k])
        hint = {
            REASONS[0]: "the gesture never rose above the burst detector: too far or too slow (closer / brisker), or the volume is too low",
            REASONS[1]: "the engine's strength gate is set from the training people's gestures; try a lower one: --min-peak-db 8 here and on the engine",
            REASONS[2]: "the walking-in / range gates fire on hand gestures: try the engine with --no-range-gate",
            REASONS[3]: "the model leans the right way but is unsure: more people in training, the table calibration (sonar.adapt), or a lower --min-confidence",
            REASONS[4]: "the model mixes gestures up: look at left/right first (one microphone leaves only the two-speaker cue)",
            REASONS[5]: "the model does not see a gesture in this person's bursts: check the burst cutting (sonar.labelqa) and add people",
            REASONS[6]: "the debounce drops them: gestures too close together in the recording, or a wind-up read as the opposite swipe",
        }[top]
        print(f"most common: {top} -> {hint}")
    if a["false_during"]:
        print("false triggers happened during: " + ", ".join(f"{k} ({v})" for k, v in sorted(a["false_during"].items(), key=lambda kv: -kv[1])))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sonar.gesture_replay", description="Replay sessions through the engine and score the gestures")
    ap.add_argument("roots", nargs="+", help="a session folder, or a folder of sessions")
    ap.add_argument("--model", default="models/gesture.onnx", help="model to test (the physics rules if the file is missing)")
    ap.add_argument("--people", default="", help="only these people, e.g. the held-out ones of a training run")
    ap.add_argument("--min-confidence", type=float, default=0.6, help="the engine's confidence threshold")
    ap.add_argument("--min-peak-db", type=float, help="strength gate (default: what the model file sets, as in the live engine)")
    ap.add_argument("--no-gates", action="store_true", help="switch the engine's strength, walking-in and range gates off (classifier + debounce only)")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16), help="sessions replayed at the same time")
    ap.add_argument("--json", help="write every trial and the totals to this file")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    people = [p for p in args.people.split(",") if p]
    folders = []
    for root in args.roots:
        for folder in find_sessions(root):
            try:
                person = str(json.loads((folder / "session.json").read_text()).get("person", "unknown"))
            except (OSError, ValueError):
                continue
            if not people or person in people:
                folders.append(folder)
    if not folders:
        print(f"ERROR: no session of {people or 'anybody'} under {args.roots}", file=sys.stderr)
        return 2
    model = args.model if Path(args.model).exists() else None
    if model is None:
        print(f"NOTE: {args.model} not found: replaying with the physics rules")
    jobs = [(str(f), model, args.min_confidence, args.min_peak_db, not args.no_gates) for f in folders]
    t0 = time.perf_counter()
    workers = max(1, min(args.workers, len(jobs)))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    good = [r for r in results if "error" not in r]
    for r in results:
        if "error" in r:
            print(f"WARNING: skipped {r['session']}: {r['error']}", file=sys.stderr)
            continue
        n = sum(1 for t in r["trials"] if t["outcome"] != "not a class of this model")
        hit = sum(1 for t in r["trials"] if t["outcome"] == "recognised")
        print(f"{r['session']}: {hit}/{n} recognised, {sum(1 for t in r['trials'] if t['outcome'] == 'wrong')} wrong, "
              f"{len(r['false_triggers'])} false trigger(s) in {r['seconds']:.0f} s")
    if not good:
        return 1
    totals = tally(good)
    print_tally(totals, good[0]["gate_db"], args.min_confidence)
    minutes = sum(r["seconds"] for r in good) / 60.0
    print(f"time: {minutes:.1f} min of audio replayed in {time.perf_counter() - t0:.1f} s on {workers} worker process(es); "
          f"classifier: {good[0]['classifier']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"model": str(model), "min_confidence": args.min_confidence, "totals": totals, "sessions": results}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""engine_replay.py — score a model the way the DEMO experiences it: which slide changes actually fire.

Plain words: the training score looks at one picture per prompt. The live engine sees more: every movement
burst (the wind-up, the stroke, the hand dropping back), then its own rules decide which ones become a gesture
event (confidence threshold, "too weak" gate, "somebody is walking in" gate, range gate, 0.4 s dead time, ignore
the opposite gesture for 1.2 s, hold a swipe 0.3 s to tell wind-up from stroke). This module replays a recorded
session through exactly those rules and counts, per prompt, whether the right gesture fired first, whether a
wrong one fired, whether extra ones fired, and how many gestures fired where nobody gestured.

How it stays exact: the per-frame facts the rules need (presence state, near/far movement levels, "a burst is
open") are recorded once per session by running the engine's own Pipeline on the audio; the decision rules are
the engine's own GestureDecider class. `python -m recall.engine_replay --check` runs the real Engine on a
session with a real model and compares the two event lists.

How to run:   .venv/bin/python -m recall.engine_replay --model train_out/recall/baseline/models/gesture.onnx
What it changes: writes train_out/recall/traces/<session>.npz once per session. Read-only on the recordings.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .common import REPO, STREAMS, fixed_meta

sys.path.insert(0, str(REPO))

from sonar.config import SonarConfig  # noqa: E402
from sonar.engine import GestureDecider  # noqa: E402
from sonar.gestures import Segment  # noqa: E402

TRACES = REPO / "train_out" / "recall" / "traces"
STATES = ("", "empty", "entering", "present")


def build_trace(name: str, data_root: Path | None = None) -> Path:
    """Run the engine's Pipeline over one session and keep what the engine's gates look at. Cached."""
    out = TRACES / f"{name}.npz"
    if out.exists():
        return out
    from sonar.dataset import load_session
    from sonar.pipeline import Pipeline

    root = data_root or (REPO / "mac" / "data")
    if not (root / name).exists():
        for alt in (REPO / "mac" / "excluded", REPO / "train_out" / "recall" / "excluded", REPO / "train_out" / "recall" / "data13"):
            if (alt / name).exists():
                root = alt
                break
    meta, audio, cfg = load_session(root / name)
    pipe = Pipeline(cfg, n_mics=audio.shape[1])
    state, near, far, active, segs, windows = [], [], [], [], [], []
    for i in range(0, len(audio), 1920):
        for fr in pipe.process(audio[i:i + 1920]):
            p = fr.presence
            state.append(STATES.index(p.state) if p is not None else 0)
            near.append(p.near_db if p is not None else np.nan)
            far.append(p.far_db if p is not None else np.nan)
            if fr.segment is not None and fr.window is not None:
                s = fr.segment
                segs.append([fr.index, s.start, s.end, s.decided, s.peak_db, s.energy_db, int(s.forced)])
                windows.append(fr.window)
            active.append(pipe.segmenter.active)
    TRACES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, state=np.array(state, np.int8), near=np.array(near, np.float32), far=np.array(far, np.float32),
                        active=np.array(active, bool), segs=np.array(segs, np.float64).reshape(-1, 7),
                        windows=np.stack(windows).astype(np.float16) if windows else np.zeros((0, 2, 64, 80), np.float16))
    return out


@dataclass
class Fired:
    frame: int          # frame at which the engine emitted the event
    seg_end: int        # last active frame of the burst it came from
    label: str
    confidence: float
    seg: int            # index of that burst in the trace


class _Lookup:
    """Stands in for the classifier: answers with the prepared prediction of the burst being decided."""

    source = "model"

    def __init__(self) -> None:
        self.answer: tuple[str, float, dict[str, float]] = ("none", 1.0, {})

    def predict(self, window: np.ndarray) -> tuple[str, float, dict[str, float]]:
        return self.answer


def replay(trace: dict[str, np.ndarray], probs: np.ndarray, labels: tuple[str, ...], cfg: SonarConfig,
           min_confidence: float = 0.6, min_peak_db: float | None = None, swipe_hold_s: float = 0.3,
           gate_entering: bool = True, gate_range: bool = True, entering_cap_s: float | None = None) -> tuple[list[Fired], list[str]]:
    """Engine.step's gesture logic for one session. probs [n_bursts, n_classes] are the model's calibrated answers.

    entering_cap_s is NOT in the engine today: a what-if in which the "somebody is walking in" gate only blocks
    for that many seconds after the presence detector left "empty" (None = the engine as it is: as long as
    the state stays "entering").
    Returns (events fired, per-burst verdict: "weak" / "entering" / "far" / "none" / "low" / "seen").
    """
    fps = int(round(cfg.frame_rate))
    min_peak = cfg.seg_min_peak_db if min_peak_db is None else min_peak_db
    clf = _Lookup()
    decider = GestureDecider(cfg, clf, min_confidence, swipe_hold_s)
    segs = trace["segs"]
    by_frame = {int(s[0]): k for k, s in enumerate(segs)}
    state, near, far, active = trace["state"], trace["near"], trace["far"], trace["active"]
    entering_until, far_until = -1, -1
    entering_since = -1                                   # frame at which the current "entering" spell began
    zones: list[tuple[int, float, float]] = []
    fired: list[Fired] = []
    verdict = ["" for _ in range(len(segs))]
    current = -1
    for i in range(len(state)):
        index = i + 1
        has_p = state[i] != 0
        if has_p and STATES[state[i]] == "entering":
            if entering_since < 0:
                entering_since = index
            if entering_cap_s is None or index - entering_since <= entering_cap_s * fps:
                entering_until = index + fps // 2
        elif has_p:
            entering_since = -1
        if has_p:
            zones.append((index, float(near[i]), float(far[i])))
            if len(zones) > 200:
                zones.pop(0)
        events: list[dict[str, Any]] = []
        k = by_frame.get(index)
        if k is not None:
            _, start, end, decided, peak, energy, forced = segs[k]
            blocked = ""
            if peak < min_peak:
                blocked = "weak"
            elif gate_entering and index < entering_until:
                blocked = "entering"
            elif gate_range and index < far_until:
                blocked = "far"
            elif gate_range:
                zone = [(n, f) for j, n, f in zones if start <= j <= end]
                if zone:
                    zn, zf = max(z[0] for z in zone), max(z[1] for z in zone)
                    if zf > cfg.gate_far_db and zf - zn > cfg.gate_far_margin_db:
                        blocked = "far"
                        far_until = index + 2 * fps
            if blocked:
                verdict[k] = blocked
            else:
                p = probs[k]
                best = int(np.argmax(p))
                clf.answer = (labels[best], float(p[best]), {n: float(v) for n, v in zip(labels, p)})
                verdict[k] = "none" if labels[best] == "none" else "low" if p[best] < min_confidence else "seen"
                current = k
                seg = Segment(int(start), int(end), int(decided), float(peak), float(energy), bool(forced))
                events += decider.on_segment(index, seg, None)
        events += decider.tick(index, bool(active[i]))
        for e in events:
            # the burst an event came from: the latest burst with that label that was "seen" and not yet used
            src = next((j for j in range(current, -1, -1) if verdict[j] == "seen" and labels[int(np.argmax(probs[j]))] == e["label"]
                        and not any(f.seg == j for f in fired)), current)
            fired.append(Fired(index, int(segs[src][2]), e["label"], float(e["confidence"]), src))
    return fired, verdict


def trial_table(meta: dict[str, Any], fired: list[Fired], labels: tuple[str, ...], n_frames: int) -> dict[str, Any]:
    """Per prompt: what fired between its GO and the next GO. Plus everything that fired outside any prompt."""
    evs = [e for e in meta.get("events", []) if e.get("kind") == "gesture"]
    starts = [int(e["start"] // 960) + 1 for e in evs]
    rows = []
    owned = set()
    for n, e in enumerate(evs):
        a = starts[n] - 10
        b = starts[n + 1] - 10 if n + 1 < len(evs) else starts[n] + 250
        mine = [f for f in fired if a <= f.seg_end < b]
        owned.update(id(f) for f in mine)
        first = mine[0].label if mine else None
        rows.append({"prompt": e["label"], "rep": e.get("rep"), "fired": [f.label for f in mine],
                     "outcome": "missed" if first is None else "correct" if first == e["label"] else "wrong",
                     "extra": max(0, len(mine) - 1), "extra_other": sum(1 for f in mine[1:] if f.label != e["label"])})
    outside = [f for f in fired if id(f) not in owned]
    trial_frames = sum(min(starts[n + 1] if n + 1 < len(evs) else starts[n] + 250, n_frames) - starts[n] for n in range(len(evs)))
    return {"trials": rows, "outside": outside, "outside_minutes": max(n_frames - trial_frames, 0) / 50.0 / 60.0}


def summarise(tables: list[dict[str, Any]], labels: tuple[str, ...]) -> dict[str, Any]:
    """Pool several sessions: event-level recall per gesture, wrong / extra shares, false events per minute."""
    gest = [r for t in tables for r in t["trials"] if r["prompt"] in labels[1:]]
    other = [r for t in tables for r in t["trials"] if r["prompt"] not in labels[1:]]         # e.g. pull prompts
    n = len(gest)
    out = {
        "n_prompts": n,
        "correct": sum(r["outcome"] == "correct" for r in gest),
        "wrong": sum(r["outcome"] == "wrong" for r in gest),
        "missed": sum(r["outcome"] == "missed" for r in gest),
        "clean": sum(r["outcome"] == "correct" and r["extra"] == 0 for r in gest),      # right gesture fired once, nothing else
        "extra_other": sum(r["outcome"] == "correct" and r["extra_other"] > 0 for r in gest),
        "extra_same": sum(r["outcome"] == "correct" and r["extra"] > 0 and r["extra_other"] == 0 for r in gest),
        "per_class": {c: [sum(r["outcome"] == "correct" for r in gest if r["prompt"] == c), sum(1 for r in gest if r["prompt"] == c)]
                      for c in labels[1:]},
        "non_gesture_prompts": len(other), "non_gesture_prompts_fired": sum(1 for r in other if r["fired"]),
        "outside_fired": sum(len(t["outside"]) for t in tables),
        "outside_minutes": sum(t["outside_minutes"] for t in tables),
    }
    out["recall"] = out["correct"] / max(n, 1)
    out["clean_rate"] = out["clean"] / max(n, 1)
    out["false_per_min"] = out["outside_fired"] / max(out["outside_minutes"], 1e-9)
    return out


def load_trace(name: str) -> dict[str, np.ndarray]:
    with np.load(build_trace(name)) as z:
        return {k: z[k] for k in z.files}


def onnx_probs(model: Path, windows: np.ndarray) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    import onnxruntime as ort

    meta = json.loads(model.with_suffix(".json").read_text())
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {sess.get_inputs()[0].name: windows.astype(np.float32)})[0] if len(windows) else np.zeros((0, len(meta["labels"])))
    z = logits / float(meta.get("temperature", 1.0))
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return p, tuple(meta["labels"]), meta


def check_against_engine(model: Path, name: str) -> None:
    """Run the REAL Engine over the session with the model and compare its gesture events with the replay's."""
    from sonar.audio import FileAudio
    from sonar.classifier import load_classifier
    from sonar.dataset import load_session
    from sonar.engine import Engine

    meta, audio, cfg = load_session(REPO / "mac" / "data" / name)
    backend = FileAudio(cfg, REPO / "mac" / "data" / name / "audio.wav", realtime=False)
    engine = Engine(cfg, backend, emit_spectrogram=False)
    engine.set_classifier(load_classifier(model, engine.cfg))
    real = []
    frame = 0
    while True:
        rx = backend.read(timeout=0.1)
        if rx is None:
            break
        n_before = engine._frames
        for e in engine.step(rx):
            if e["type"] == "gesture":
                real.append((e["label"], round(e["confidence"], 3), round(e["peak_db"], 1)))
    trace = load_trace(name)
    probs, labels, m = onnx_probs(model, trace["windows"])
    fired, _ = replay(trace, probs, labels, cfg, min_peak_db=max(cfg.seg_min_peak_db, float(m["gesture_peak_db_p10"]) - 6.0))
    mine = [(f.label, round(f.confidence, 3), round(float(trace["segs"][f.seg][4]), 1)) for f in fired]
    same = len(real) == len(mine) and all(a[0] == b[0] and abs(a[1] - b[1]) < 0.01 and abs(a[2] - b[2]) < 0.11 for a, b in zip(real, mine))
    print(f"{name}: real engine fired {len(real)} gestures, replay fired {len(mine)}; same labels, confidences within 0.01: {same}")
    if not same:
        for a, b in zip(real, mine):
            print("   ", a, b, "" if a[0] == b[0] else "  <-- differs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="train_out/recall/baseline/models/gesture.onnx")
    ap.add_argument("--check", action="store_true", help="compare the replay with the real Engine on two sessions")
    ap.add_argument("--person", default="", help="only sessions of this person")
    args = ap.parse_args()
    model = (REPO / args.model).resolve()
    names = sorted(p.stem for p in STREAMS.glob("*.npz"))
    if args.person:
        names = [n for n in names if f"_{args.person}_" in n]
    if args.check:
        for n in [x for x in names if "gestures" in x][-1:] + [x for x in names if "scene" in x][-1:]:
            check_against_engine(model, n)
        return
    cfg = SonarConfig(n_mics=1)
    tables = {}
    for n in names:
        trace = load_trace(n)
        probs, labels, m = onnx_probs(model, trace["windows"])
        min_peak = max(cfg.seg_min_peak_db, float(m.get("gesture_peak_db_p10", 9.0)) - 6.0)
        fired, verdict = replay(trace, probs, labels, cfg, min_peak_db=min_peak)
        meta = fixed_meta(json.loads((STREAMS / f"{n}.json").read_text())["meta"], n)
        tables[n] = trial_table(meta, fired, labels, len(trace["state"]))
        s = summarise([tables[n]], labels)
        v = {k: verdict.count(k) for k in sorted(set(verdict))}
        print(f"{n:<40} prompts {s['n_prompts']:3d}: correct {s['correct']:3d} wrong {s['wrong']:2d} missed {s['missed']:2d} | clean {s['clean']:3d} "
              f"extra-other {s['extra_other']:2d} extra-same {s['extra_same']:2d} | outside prompts: {s['outside_fired']} in {s['outside_minutes']:.1f} min | bursts {v}")
    for person in sorted({n.split("_")[1] for n in names}):
        s = summarise([t for n, t in tables.items() if n.split("_")[1] == person], labels)
        print(f"{person:<8} event recall {s['recall']:.3f} ({s['correct']}/{s['n_prompts']}), clean {s['clean_rate']:.3f}, wrong {s['wrong']}, "
              f"false events/min {s['false_per_min']:.2f} ({s['outside_fired']} in {s['outside_minutes']:.1f} min), per class {s['per_class']}, "
              f"pull prompts that fired {s['non_gesture_prompts_fired']}/{s['non_gesture_prompts']}")


if __name__ == "__main__":
    main()

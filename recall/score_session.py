"""score_session.py — score NEW recordings with the shipped model, exactly as the engine would run it.

Why: "recall on an unseen person" is only proven on a person no model has ever seen. Every number so far holds
out one of the same three people. The moment a new person's session exists (sonar/record.py, gestures script),
this gives the honest answer in about 20 seconds per session, with no training and no GPU.

What it does for each session folder (audio.wav + session.json):
  1. runs the audio through the engine's own Pipeline, with the burst cutting the shipped model runs with
     (the model's gesture.json says which under "run_with"; otherwise the start script's: rel12),
  2. label check: do the swipes go the way their prompts say? (left-speaker tone minus right-speaker tone; the
     suite's audit.) A session whose swipes measurably go the OPPOSITE way is scored with left/right exchanged
     and says so loudly (--flip none keeps the labels as recorded),
  3. picture-level score, the released definition: the strongest burst of each prompt named correctly with
     confidence >= 0.6, and the share of non-gesture pictures (other bursts away from prompts, quiet moments)
     that would trigger,
  4. engine-level score: the session replayed through the engine's own gesture rules (recall.engine_replay,
     checked equal to the real engine): per prompt whether the right gesture fired first, and how many
     gestures fired where nobody gestured.

How to run:
    .venv/bin/python -m recall.score_session <session folder> [more folders ...]
    .venv/bin/python -m recall.score_session ~/sonar-hack/data/20261114-*_NAME_* --model models-release-mac-recall
What it changes: nothing. It reads the recordings and the model and prints. No cache files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .common import HUMAN, REPO, Sess, fixed_meta
from .data import LABELS, full
from .engine_replay import STATES, onnx_probs, replay, summarise, trial_table
from .relcut import NO_CLICKS, parse
from .trainlib import score, wilson

sys.path.insert(0, str(REPO))


def run_session(folder: Path, rel_db: float, clicks: bool) -> tuple[Sess, dict[str, np.ndarray], Any]:
    """One recording through the engine's Pipeline: (its bursts and feature stream, the per-frame facts the engine's
    gates look at, the config used). Same bookkeeping as recall.relcut.one and recall.engine_replay.build_trace."""
    from sonar.dataset import load_session
    from sonar.pipeline import Pipeline

    meta, audio, cfg = load_session(folder)
    cfg = cfg.with_(seg_rel_off_db=rel_db, impulse_ratio=cfg.impulse_ratio if clicks else NO_CLICKS)
    pipe = Pipeline(cfg, n_mics=audio.shape[1])
    cfg = pipe.cfg
    zero = np.zeros((cfg.n_feat_channels, cfg.doppler_bins), np.float32)
    feats, ready, act, seg6, nb = [], [], [], [], []
    state, near, far, active, seg7, windows = [], [], [], [], [], []
    prev_end = -1
    for i in range(0, len(audio), 1920):
        for fr in pipe.process(audio[i:i + 1920]):
            feats.append(zero if fr.feat is None else fr.feat)
            ready.append(fr.feat is not None)
            act.append(fr.activity_db)
            p = fr.presence
            state.append(STATES.index(p.state) if p is not None else 0)
            near.append(p.near_db if p is not None else np.nan)
            far.append(p.far_db if p is not None else np.nan)
            if fr.segment is not None:
                s = fr.segment
                seg6.append([s.start, s.end, s.decided, s.peak_db, s.energy_db, int(s.forced)])
                nb.append(prev_end)
                prev_end = s.end + 1
                if fr.window is not None:
                    seg7.append([fr.index, s.start, s.end, s.decided, s.peak_db, s.energy_db, int(s.forced)])
                    windows.append(fr.window)
            active.append(pipe.segmenter.active)
    name = folder.name
    # pictures rounded to 16-bit floats, as every cached table of this project stores them (keeps scores comparable)
    sess = Sess(name, meta["person"], meta["script"], meta, np.stack(feats).astype(np.float16).astype(np.float32), np.array(ready),
                np.array(act, np.float32), np.array(seg6, np.float64).reshape(-1, 6), np.array(nb, np.int64))
    trace = {"state": np.array(state, np.int8), "near": np.array(near, np.float32), "far": np.array(far, np.float32),
             "active": np.array(active, bool), "segs": np.array(seg7, np.float64).reshape(-1, 7),
             "windows": np.stack(windows).astype(np.float16) if windows else np.zeros((0, cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames), np.float16)}
    return sess, trace, cfg


def main() -> int:
    from .suite import FLIP_T, swipe_audit

    ap = argparse.ArgumentParser(description="Score new recordings with the shipped model, as the engine would run it.")
    ap.add_argument("folders", nargs="+", help="session folders (each holds audio.wav and session.json)")
    ap.add_argument("--model", default="models-release-mac-recall", help="folder with gesture.onnx and gesture.json")
    ap.add_argument("--cutting", choices=("std", "rel12", "rel12nc"), help="the engine's burst cutting (default: what the model's gesture.json says, else rel12)")
    ap.add_argument("--flip", default="auto", help='"auto" = exchange left/right in sessions whose swipes measurably go the opposite way; "none" = never')
    ap.add_argument("--range-gate", action="store_true", help="keep the engine's range gate on (the MacBook start script switches it off)")
    ap.add_argument("--entering-gate-s", type=float, default=3.0, help="cap of the walking-in gate in seconds (start script: 3; negative = uncapped)")
    args = ap.parse_args()

    model_dir = Path(args.model) if Path(args.model).is_absolute() else REPO / args.model
    onnx = model_dir / "gesture.onnx"
    meta_m = json.loads((model_dir / "gesture.json").read_text())
    run_with = meta_m.get("run_with", {})
    cutting = args.cutting or ("std" if run_with and not run_with.get("env") else None)
    if cutting is None:
        db = (run_with.get("env") or {}).get("SONAR_SEG_REL_OFF_DB", "12")
        cutting = f"rel{db}" + ("nc" if "--no-click-rejection" in run_with.get("flags", []) else "")
    rel_db, clicks = (0.0, True) if cutting == "std" else parse(cutting)
    cap = None if args.entering_gate_s < 0 else args.entering_gate_s
    p10 = float(meta_m.get("gesture_peak_db_p10", 9.0))
    print(f"model {model_dir.name} ({meta_m.get('metrics', {}).get('arm', 'no recipe name')}), burst cutting {cutting}, range gate {'on' if args.range_gate else 'off'}, "
          f"walking-in gate {'uncapped' if cap is None else f'{cap:g} s'}, weak-burst gate {p10 - 6:.1f} dB, threshold 0.6\n")

    rows: list[dict[str, Any]] = []
    for arg in args.folders:
        folder = Path(arg).expanduser().resolve()
        if not (folder / "session.json").exists():
            print(f"{folder}: no session.json here, skipped")
            continue
        sess, trace, cfg = run_session(folder, rel_db, clicks)
        t = full(sess)
        cols = {"X": t.X, "label": t.label, "source": t.source, "session": t.session}
        audit = swipe_audit(cols).get(sess.name)
        flipped = bool(args.flip == "auto" and audit and audit["t"] < FLIP_T)
        swap = {"swipe_left": "swipe_right", "swipe_right": "swipe_left"}
        label = np.array([swap.get(str(v), str(v)) if flipped else str(v) for v in t.label])
        # picture level (the released definition): gestures + bursts away from prompts + quiet moments
        scored = np.isin(t.source, ["gesture", "burst", "idle"])
        probs, labels, _ = onnx_probs(onnx, np.ascontiguousarray(t.X[scored].astype(np.float32)[..., 8:88]))
        assert tuple(labels) == LABELS, labels
        lut = {n: i for i, n in enumerate(LABELS)}
        y = np.array([lut.get(v, 0) for v in label[scored]], dtype=np.int64)
        sc = score(np.log(probs + 1e-12), y, LABELS)
        # engine level
        pw, _, _ = onnx_probs(onnx, trace["windows"])
        fired, _ = replay(trace, pw, labels, cfg, min_peak_db=max(cfg.seg_min_peak_db, p10 - 6.0), gate_range=args.range_gate, entering_cap_s=cap)
        tab = trial_table(fixed_meta(sess.meta, sess.name, [sess.name] if flipped else []), fired, labels, len(trace["state"]))
        ev = summarise([tab], labels)
        rows.append({"name": sess.name, "person": sess.person, "probs": probs, "y": y, "tab": tab})
        known = "  (IN the training data unless the model held this person out)" if sess.person in HUMAN else ""
        print(f"{sess.name}  person {sess.person}{known}, script {sess.script}, {len(trace['state']) / 50 / 60:.1f} min, {len(sess.segs)} bursts")
        if audit:
            verdict = ("REVERSED: the swipes go the opposite way to the prompts" + (" -> scored with left/right EXCHANGED" if flipped else " (labels kept as recorded)")
                       if audit["t"] < FLIP_T else "as prompted" if audit["t"] > -FLIP_T else "no clean direction (very fast swipes? ask for a half-second pace)")
            print(f"  label check: left prompts {audit['left_hz']:+.0f} Hz, right prompts {audit['right_hz']:+.0f} Hz, t = {audit['t']:+.1f}: {verdict}")
        if sc["n_gesture"]:
            per = ", ".join(f"{k} {a}/{b}" for k, (a, b) in sc["per_class"].items())
            print(f"  pictures: gestures recognised {sc['hits']}/{sc['n_gesture']} = {sc['recall']:.3f} ({per}; wrong gesture {sc['wrong_gesture']}, silent {sc['silent']})")
        print(f"  pictures: non-gestures that trigger {sc['false_n']}/{sc['n_none']}" + (f" = {sc['false']:.3f}" if sc["n_none"] else ""))
        if ev["n_prompts"]:
            print(f"  engine:   right gesture first on {ev['correct']}/{ev['n_prompts']} prompts = {ev['recall']:.3f} (wrong first {ev['wrong']}, nothing fired {ev['missed']}, "
                  f"clean {ev['clean']})")
        print(f"  engine:   gestures fired where nobody gestured: {ev['outside_fired']} in {ev['outside_minutes']:.1f} min\n")

    for person in sorted({r["person"] for r in rows}) + (["everyone above"] if len({r["person"] for r in rows}) > 1 else []):
        mine = [r for r in rows if person in (r["person"], "everyone above")]
        sc = score(np.log(np.concatenate([r["probs"] for r in mine]) + 1e-12), np.concatenate([r["y"] for r in mine]), LABELS)
        ev = summarise([r["tab"] for r in mine], LABELS)
        lo, hi = sc["recall_ci"]
        flo, fhi = wilson(sc["false_n"], sc["n_none"])
        print(f"== {person}: gestures recognised {sc['hits']}/{sc['n_gesture']} = {sc['recall']:.3f} [95% range {lo:.2f}-{hi:.2f}]; non-gestures that trigger "
              f"{sc['false_n']}/{sc['n_none']} = {sc['false']:.3f} [{flo:.2f}-{fhi:.2f}]; engine: right gesture first {ev['correct']}/{ev['n_prompts']}"
              f"{f' = {ev['recall']:.3f}' if ev['n_prompts'] else ''}, false events {ev['outside_fired']} in {ev['outside_minutes']:.1f} min "
              f"({ev['false_per_min']:.2f} per min)   target: 0.70 or more at 0.055 or less")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

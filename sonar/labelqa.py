"""labelqa.py — check the labels of a session against the sonar itself, and say where they disagree.

Plain words: a label is only as good as its witness. For every movement burst of a session this
tool lines up what each witness says:

    the label      from the camera (camrecord.py) or from the prompt window (record.py)
    the physics    the training-free Doppler rules (DATA_STRATEGY 3.6, gestures.heuristic_classify)
    the model      the current trained model, if there is one

and reports (1) how often they agree, (2) a list of "flags": single moments worth a look
(a label both other witnesses contradict, a swipe whose direction looks flipped, a gesture the
camera saw but the sonar did not, a place where the model fires although nothing was asked for),
and (3) a picture per 30 s: the sonar's Doppler picture with the label spans, the per-burst
verdicts and the flags drawn underneath, plus the camera's hand-speed curve over the sonar's
movement curve (if the two do not rise and fall together, the clocks are off).

How to run:
    python -m sonar.labelqa data/20261114-101500_v07_free                 (one session)
    python -m sonar.labelqa data/ --model models/gesture.onnx             (every session + totals)
    python -m sonar.labelqa data/ --no-plot                               (numbers only: all sessions judged in one batched
                                                                           pass, on the GPU if there is one)
What it changes: writes qa.json and qa_p1.png, qa_p2.png ... into each session folder (or --out),
and labelqa_summary.json next to the sessions when a whole folder is checked. Reads everything else.
Needs matplotlib for the pictures (requirements-train.txt).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import GESTURE_LABELS
from .dataset import SESSION_FILE, find_sessions
from .weaklabel import OPPOSITE, Burst, SessionTable, build_tables, fires, table_rows, timing_line

COLORS = {"swipe_left": "tab:red", "swipe_right": "tab:blue", "push": "tab:green", "pull": "tab:orange", "none": "0.6",
          "review": "tab:purple", "return": "0.45", "other": "0.45"}
SWIPES = ("swipe_left", "swipe_right")
# "Too fast" = the bright part of a gesture's picture sits in the outermost rows (the top and bottom 6 of 64:
# tone shifts beyond about 200 of the 250 Hz the picture shows). Set on the MacBook recordings of 2026-11-14:
# sessions with flicked strokes have a typical edge share of 0.16-0.21, sessions at a calm pace 0.05.
EDGE_PART = 6 / 64
EDGE_FLOOR = 0.35           # picture brightness that counts as "stands out of the background"
EDGE_SHARE_FAST = 0.15


# ---------------------------------------------------------------------------
# Numbers and flags
# ---------------------------------------------------------------------------
def _vote(b: Burst, witness: str, threshold: float) -> str:
    """One witness's verdict on one burst: a gesture name or "none"."""
    if witness == "physics":
        return b.physics if b.physics != "none" and b.physics_conf >= 0.5 else "none"
    return b.model if b.model not in ("", "none") and b.model_conf >= threshold else "none"


def edge_shares(power: np.ndarray, doppler_bins: int) -> np.ndarray:
    """For each picture [n, channels, speed rows, frames]: the share of its bright part that lies in the outermost rows.

    Near 0 for a stroke the picture can show; large when the hand was faster than the picture's speed range.
    """
    n_edge = max(1, round(EDGE_PART * doppler_bins))
    bright = np.clip(power - EDGE_FLOOR, 0.0, None)
    at_edge = bright[:, :, :n_edge].sum(axis=(1, 2, 3)) + bright[:, :, -n_edge:].sum(axis=(1, 2, 3))
    return at_edge / (bright.sum(axis=(1, 2, 3)) + 1e-9)


def cutting_report(table: SessionTable, min_confidence: float = 0.6) -> dict[str, Any]:
    """How the session was cut into bursts: the numbers that show whether a gesture gets ONE clean picture.

    A healthy session: bursts well under a second, hardly any cut off at the length limit, one or two
    bursts per prompted gesture, the chosen burst starting a steady time after the prompt, few
    saturated pixels. Long, loud bursts chopped at the limit mean the stroke lands at a different
    place in every picture (or is cut in two), which a model learns badly from few people.
    """
    bs, cfg = table.bset, table.cfg
    use_model = table.classifier == "model"
    rate = cfg.frame_rate
    out: dict[str, Any] = {"bursts": len(bs), "limit_s": round(cfg.seg_max_frames / rate, 2)}
    if not len(bs):
        return out
    lens = (bs.seg[:, 1] - bs.seg[:, 0] + 1) / rate
    forced = bs.seg[:, 3].astype(bool)
    gestures = [b for b in table.bursts if b.role == "gesture"]
    out.update(burst_seconds_median=round(float(np.median(lens)), 2), cut_at_limit_share=round(float(forced.mean()), 3))
    if not gestures:
        return out
    idx = np.array([b.k for b in gestures])
    per_trial: dict[int, int] = {}
    for b in table.bursts:
        if b.trial is not None:
            per_trial[b.trial] = per_trial.get(b.trial, 0) + 1
    power = bs.windows()[idx][:, :cfg.n_power_channels].astype(np.float32)
    events = table.meta.get("events", [])
    delay = [(b.start - bs.frame_of_sample(events[b.event]["start"])) / rate for b in gestures
             if b.event is not None and not events[b.event].get("tight")]
    edge_share = edge_shares(power, cfg.doppler_bins)
    better = trial_hit = 0
    for b in gestures:
        mates = [o for o in table.bursts if o.trial == b.trial]
        votes = [_vote(o, "model" if use_model else "physics", min_confidence) for o in mates]
        mine = _vote(b, "model" if use_model else "physics", min_confidence)
        trial_hit += b.label in votes
        better += mine != b.label and any(v == b.label for o, v in zip(mates, votes) if o is not b)
    out.update(
        gestures=len(gestures), gesture_seconds_median=round(float(np.median(lens[idx])), 2),
        gesture_cut_at_limit_share=round(float(forced[idx].mean()), 3),
        bursts_per_gesture=round(float(np.mean(list(per_trial.values()))), 2) if per_trial else None,
        gesture_peak_db={"p10": round(float(np.percentile(bs.peak_db[idx], 10)), 1), "median": round(float(np.median(bs.peak_db[idx])), 1)},
        saturated_share=round(float((power >= 0.999).mean()), 4),
        edge_share_median=round(float(np.median(edge_share)), 3), off_the_speed_range=int((edge_share > EDGE_SHARE_FAST).sum()),
        another_burst_fits_better=int(better), recognised_in_some_burst=int(trial_hit),
        witness="model" if use_model else "physics",
    )
    if delay:
        out["burst_start_after_prompt_s"] = {"median": round(float(np.median(delay)), 2),
                                             "spread_p10_p90": round(float(np.percentile(delay, 90) - np.percentile(delay, 10)), 2)}
    return out


def review_session(table: SessionTable, min_confidence: float = 0.6) -> dict[str, Any]:
    """Agreement numbers and flags for one session."""
    use_model = table.classifier == "model"
    witnesses = ["physics"] + (["model"] if use_model else [])
    events = table.meta.get("events", [])
    gestures = [b for b in table.bursts if b.role == "gesture"]
    negatives = [b for b in table.bursts if b.role == "none"]
    flags: list[dict[str, Any]] = []

    def flag(kind: str, t0: float, t1: float, text: str, level: str = "check") -> None:
        flags.append({"kind": kind, "level": level, "t0": round(t0, 2), "t1": round(t1, 2), "text": text})

    agree: dict[str, dict[str, Any]] = {}
    for src in ("camera", "prompt"):
        mine = [b for b in gestures if b.label_source == src]
        if not mine:
            continue
        row: dict[str, Any] = {"labelled": len(mine)}
        for w in witnesses:
            votes = [_vote(b, w, min_confidence) for b in mine]
            row[w] = {
                "same": sum(v == b.label for v, b in zip(votes, mine)),
                "saw_no_gesture": sum(v == "none" for v in votes),
                "other_gesture": sum(v not in ("none", b.label) for v, b in zip(votes, mine)),
            }
        agree[src] = row
    both = {"either_fires": 0, "same": 0}
    if use_model:
        for b in table.bursts:
            p, m = _vote(b, "physics", min_confidence), _vote(b, "model", min_confidence)
            if p != "none" or m != "none":
                both["either_fires"] += 1
                both["same"] += p == m

    # swipe direction: does the label point the other way than the sonar's own direction cues?
    flips = {src: [0, 0] for src in ("camera", "prompt")}
    for b in gestures:
        if b.label in SWIPES and b.physics in SWIPES and b.physics_conf >= 0.5:
            flips[b.label_source][1] += 1
            flips[b.label_source][0] += b.physics != b.label

    for b in gestures:
        votes = {w: _vote(b, w, min_confidence) for w in witnesses}
        against = [w for w, v in votes.items() if v not in ("none", b.label)]
        if len(against) == len(witnesses) and (len(witnesses) > 1 or b.physics_conf >= 0.7):
            said = ", ".join(f"{w} says {votes[w]}" for w in against)
            level = "wrong label?" if len(witnesses) > 1 else "check"
            flag("contradicted", b.t0, b.t1, f"labelled {b.label} by the {b.label_source}, but {said}", level)
        elif b.label in SWIPES and any(votes[w] == OPPOSITE[b.label] for w in witnesses):
            who = [w for w in witnesses if votes[w] == OPPOSITE[b.label]]
            flag("direction", b.t0, b.t1, f"labelled {b.label}, {' and '.join(who)} read{'s' if len(who) == 1 else ''} the opposite direction")
        if b.peak_db < 6.0:
            flag("faint", b.t0, b.t1, f"{b.label}: the sonar movement is faint ({b.peak_db:.0f} dB): too far or too slow?", "info")
        if b.label_source == "camera" and b.label_conf < 0.65:
            flag("camera unsure", b.t0, b.t1, f"{b.label}: camera confidence only {b.label_conf:.2f}", "info")
    for ev in table.missed:
        t0, t1 = ev["start"] / table.cfg.fs, ev["end"] / table.cfg.fs
        who = "camera saw" if ev.get("source") == "camera" else "prompt asked for"
        flag("no sonar movement", t0, t1, f"the {who} {ev['label']}, but the sonar saw no movement there (too far / too faint / clock offset?)")
    false_triggers = 0
    for b in negatives:
        fired, conf = fires(b, min_confidence, use_model)
        if fired != "none":
            false_triggers += 1
            flag("fires on a negative", b.t0, b.t1, f"{'model' if use_model else 'physics'} says {fired} ({conf:.2f}) where no gesture was asked for "
                 f"({b.zone})", "hard negative")
    unseen = 0
    for b in table.bursts:
        if b.role == "excluded" and b.cam == "" and use_model and b.model == b.physics and b.model not in ("", "none") \
                and b.model_conf >= 0.9 and b.physics_conf >= 0.5 and (b.hand_seen is None or b.hand_seen < 0.8):
            unseen += 1
            flag("unlabelled gesture?", b.t0, b.t1, f"model and physics both say {b.model}, the camera did not see the hand", "info")
    reviews = [ev for ev in events if ev.get("kind") == "review"]
    for ev in reviews:
        flag("camera review", ev["start"] / table.cfg.fs, ev["end"] / table.cfg.fs,
             f"camera was unsure: maybe {ev.get('candidate')} ({ev.get('confidence', 0):.2f}); {ev.get('note', '')}".strip("; "), "review")

    verdicts = []
    for src, (bad, n) in flips.items():
        if n >= 4 and bad / n >= 0.7:
            verdicts.append(
                f"LEFT/RIGHT LOOKS FLIPPED for the {src} labels ({bad} of {n} swipes): "
                + ("the camera picture is mirrored the wrong way (key M on the consent screen, or --camera-mirrored), "
                   "or the microphone / speaker channels are swapped" if src == "camera"
                   else "if the OTHER sessions of this laptop do not show this, this session's swipes went the opposite way to "
                        f"their prompts: exchange its labels with `python -m sonar.dataview <data> --out <view> --flip-swipes {table.session}` "
                        "and use the view. If every session shows it, the microphone / speaker channels are swapped "
                        "(training measures and handles that)"))
    n_cam = sum(1 for b in gestures if b.label_source == "camera")
    missed_cam = sum(1 for ev in table.missed if ev.get("source") == "camera")
    if n_cam + missed_cam >= 4 and missed_cam >= 0.5 * (n_cam + missed_cam):
        verdicts.append("HALF OR MORE of the camera's gestures have no sonar movement: gestures too far from the laptop, "
                        "the volume too low, or the camera clock far off")
    cutting = cutting_report(table, min_confidence)
    if cutting.get("gestures", 0) >= 6 and cutting["gesture_cut_at_limit_share"] >= 0.3 and cutting["gesture_peak_db"]["median"] >= 25.0:
        verdicts.append(
            f"LONG BURSTS: {100 * cutting['gesture_cut_at_limit_share']:.0f}% of the gesture bursts run into the {cutting['limit_s']} s limit "
            f"(typical strength {cutting['gesture_peak_db']['median']:.0f} dB). The stroke then sits at a different place in every picture "
            "or is cut in two. Try the peak-relative cutting (SONAR_SEG_REL_OFF_DB=12, see DATA_ENGINE.md) and compare")
    if cutting.get("gestures", 0) >= 6 and cutting["off_the_speed_range"] >= 0.3 * cutting["gestures"]:
        cfg = table.cfg
        top_speed = cfg.doppler_bins / 2 * cfg.bb_fs / cfg.stft_fft * 343.0 / (2 * float(np.mean(cfg.tone_hz)))
        verdicts.append(
            f"TOO FAST: {cutting['off_the_speed_range']} of {cutting['gestures']} gestures run off the edge of the picture "
            f"(it shows hand speeds up to about {top_speed:.1f} m/s). A stroke that fast can lose its direction, and a fast push can look like a swipe. "
            "Record again at a calm pace: about half a second across the screen, not a flick")
    if cutting.get("gestures", 0) >= 6 and cutting["another_burst_fits_better"] >= 0.2 * cutting["gestures"]:
        verdicts.append(
            f"WRONG BURST CHOSEN? In {cutting['another_burst_fits_better']} of {cutting['gestures']} prompts the loudest burst is not read as the "
            f"prompted gesture by the {cutting['witness']}, but another burst of the same prompt is (the hand going back was louder than the stroke?)")
    sync = table.meta.get("sync")
    if sync and sync.get("method") != "motion":
        verdicts.append("the camera clock was NOT lined up with the audio in this session (no common movement found): "
                        "camera-timed labels may be off by a tenth of a second or more")
    elif sync and sync.get("residual_ms") is not None and sync["residual_ms"] > 60:
        verdicts.append(f"clock sync is loose (stroke-to-stroke spread {sync['residual_ms']:.0f} ms)")
    return {
        "session": table.session, "person": table.person, "script": table.meta.get("script"), "classifier": table.classifier,
        "bursts": len(table.bursts), "gestures": len(gestures), "from_camera": n_cam, "from_prompt": len(gestures) - n_cam,
        "no_sonar_movement": len(table.missed), "camera_review": len(reviews), "negatives": len(negatives),
        "fires_on_negatives": false_triggers, "possible_unlabelled_gestures": unseen,
        "agreement": agree, "physics_vs_model": both if use_model else None,
        "swipe_direction": {src: {"flipped": bad, "swipes": n} for src, (bad, n) in flips.items() if n},
        "sync": {k: sync.get(k) for k in ("method", "offset_ms", "residual_ms", "peak_corr")} if sync else None,
        "cutting": cutting,
        "verdicts": verdicts, "flags": sorted(flags, key=lambda f: f["t0"]),
    }


def one_line(rep: dict[str, Any]) -> str:
    parts = [f"{rep['gestures']} gestures ({rep['from_camera']} camera, {rep['from_prompt']} prompt)"]
    if rep["no_sonar_movement"]:
        parts.append(f"{rep['no_sonar_movement']} with no sonar movement")
    for src, row in rep["agreement"].items():
        for w in ("physics", "model"):
            if w in row:
                parts.append(f"{w} agrees with the {src} label {row[w]['same']}/{row['labelled']}")
    if rep["negatives"]:
        parts.append(f"fires on {rep['fires_on_negatives']}/{rep['negatives']} negatives")
    if rep["camera_review"]:
        parts.append(f"{rep['camera_review']} for review")
    serious = sum(1 for f in rep["flags"] if f["level"] in ("wrong label?", "check"))
    parts.append(f"{serious} to check")
    return f"{rep['session']}: " + "; ".join(parts)


def totals(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Agreement over many sessions, overall and per person."""
    def add(acc: dict[str, Any], rep: dict[str, Any]) -> None:
        for key in ("gestures", "from_camera", "from_prompt", "no_sonar_movement", "camera_review", "negatives", "fires_on_negatives"):
            acc[key] = acc.get(key, 0) + rep[key]
        for src, row in rep["agreement"].items():
            a = acc.setdefault("agreement", {}).setdefault(src, {"labelled": 0})
            a["labelled"] += row["labelled"]
            for w in ("physics", "model"):
                if w in row:
                    t = a.setdefault(w, {"same": 0, "saw_no_gesture": 0, "other_gesture": 0})
                    for k in t:
                        t[k] += row[w][k]
        for src, d in rep["swipe_direction"].items():
            s = acc.setdefault("swipe_direction", {}).setdefault(src, {"flipped": 0, "swipes": 0})
            s["flipped"] += d["flipped"]
            s["swipes"] += d["swipes"]
        c = rep.get("cutting", {})
        if c.get("gestures"):
            cut = acc.setdefault("cutting", {"gestures": 0, "cut_at_limit": 0.0, "seconds": [], "per_gesture": [], "peak_db": [],
                                             "saturated": [], "fits_better": 0, "in_some_burst": 0, "start_spread": [], "too_fast": 0})
            cut["gestures"] += c["gestures"]
            cut["cut_at_limit"] += c["gesture_cut_at_limit_share"] * c["gestures"]
            cut["seconds"].append(c["gesture_seconds_median"])
            cut["per_gesture"].append(c["bursts_per_gesture"] or 0.0)
            cut["peak_db"].append(c["gesture_peak_db"]["median"])
            cut["saturated"].append(c["saturated_share"])
            cut["fits_better"] += c["another_burst_fits_better"]
            cut["too_fast"] += c["off_the_speed_range"]
            cut["in_some_burst"] += c["recognised_in_some_burst"]
            if "burst_start_after_prompt_s" in c:
                cut["start_spread"].append(c["burst_start_after_prompt_s"]["spread_p10_p90"])
        acc["to_check"] = acc.get("to_check", 0) + sum(1 for f in rep["flags"] if f["level"] in ("wrong label?", "check"))

    out: dict[str, Any] = {"sessions": len(reports), "all": {}, "by_person": {}}
    for rep in reports:
        add(out["all"], rep)
        add(out["by_person"].setdefault(rep["person"], {}), rep)
    return out


def print_totals(tot: dict[str, Any]) -> None:
    def pct(a: int, b: int) -> str:
        return "  n/a" if not b else f"{100.0 * a / b:4.0f}%"

    a = tot["all"]
    print(f"\n=== {tot['sessions']} session(s): {a.get('gestures', 0)} labelled gestures "
          f"({a.get('from_camera', 0)} camera-timed, {a.get('from_prompt', 0)} prompt windows), "
          f"{a.get('no_sonar_movement', 0)} with no sonar movement, {a.get('camera_review', 0)} for review, {a.get('to_check', 0)} to check")
    for src, row in a.get("agreement", {}).items():
        for w in ("physics", "model"):
            if w in row:
                r = row[w]
                print(f"  {w:<7} vs {src:<6} labels: same {pct(r['same'], row['labelled'])}, saw no gesture {pct(r['saw_no_gesture'], row['labelled'])}, "
                      f"other gesture {pct(r['other_gesture'], row['labelled'])}   (n={row['labelled']})")
    for src, d in a.get("swipe_direction", {}).items():
        print(f"  swipe direction vs the physics cues, {src} labels: {d['flipped']} of {d['swipes']} opposite")
    if a.get("negatives"):
        print(f"  fires on known negatives: {a['fires_on_negatives']} of {a['negatives']} ({pct(a['fires_on_negatives'], a['negatives']).strip()})")
    c = a.get("cutting")
    if c and c["gestures"]:
        spread = f", burst start after the prompt varies by {np.median(c['start_spread']):.2f} s" if c["start_spread"] else ""
        print(f"  burst cutting ({c['gestures']} gesture bursts): typical length {np.median(c['seconds']):.2f} s, "
              f"{pct(int(round(c['cut_at_limit'])), c['gestures']).strip()} cut off at the length limit, {np.mean(c['per_gesture']):.1f} bursts per gesture, "
              f"typical strength {np.median(c['peak_db']):.0f} dB, {100 * np.mean(c['saturated']):.1f}% of the picture saturated{spread}")
        print(f"  recognised in SOME burst of its prompt: {c['in_some_burst']} of {c['gestures']}; "
              f"another burst fits the prompt better than the chosen one: {c['fits_better']}")
        print(f"  too fast for the picture (the stroke runs off the edge of the speed range): {c['too_fast']} of {c['gestures']} gestures")
    if len(tot["by_person"]) > 1:
        print("  per person (physics agreement with the labels / to check):")
        for person, p in sorted(tot["by_person"].items()):
            same = sum(row.get("physics", {}).get("same", 0) for row in p.get("agreement", {}).values())
            n = sum(row["labelled"] for row in p.get("agreement", {}).values())
            print(f"    {person:<14} {same:>3}/{n:<3} {pct(same, n)}   {p.get('to_check', 0)} to check")


# ---------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------
def camera_speed(table: SessionTable) -> np.ndarray | None:
    """The camera's hand-speed curve on the sonar frame grid, clock offset removed (None without a landmark file)."""
    path = table.folder / "landmarks.jsonl"
    if not path.exists():
        return None
    from .camlabel import LabelConfig, all_strokes, build_tracks, load_landmarks, speed_curve

    header, frames = load_landmarks(path)
    cfg = LabelConfig(hfov_deg=float(header.get("hfov_deg", 70.0)))
    offset = float(table.meta.get("sync", {}).get("offset_ms", 0.0)) / 1e3
    for f in frames:
        f.sample -= offset * table.cfg.fs
    _, grids = all_strokes(build_tracks(frames, int(header.get("width", 640)), int(header.get("height", 480)), table.cfg.fs, cfg), cfg)
    return speed_curve(grids, table.stream.n_frames + 1, cfg.grid_hz)[1:]


def plot_session(table: SessionTable, rep: dict[str, Any], out_dir: Path, seconds: float = 30.0) -> list[Path]:
    """One PNG per `seconds` of the session. Returns the files written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cfg, stream, meta = table.cfg, table.stream, table.meta
    rate = cfg.frame_rate
    total = stream.n_frames / rate
    speed = camera_speed(table)
    use_model = table.classifier == "model"
    lanes = ["session", "label", "trainer", "physics"] + (["model"] if use_model else [])
    files = []
    for page in range(max(1, int(np.ceil(total / seconds)))):
        t_a, t_b = page * seconds, min((page + 1) * seconds, total)
        a, b = int(t_a * rate), max(int(t_b * rate), int(t_a * rate) + 2)
        t = np.arange(a, b) / rate
        rows = 3 + (1 if cfg.has_lateral else 0)
        fig, axes = plt.subplots(rows, 1, figsize=(17, 2.2 * rows + 1.6), sharex=True,
                                 gridspec_kw={"height_ratios": [1.3] + ([1.0] if cfg.has_lateral else []) + [1.6, 1.0]})
        extent = [t[0], t[-1], cfg.doppler_hz[0], cfg.doppler_hz[-1]]
        ax = axes[0]
        ax.imshow(stream.feat[a:b, :cfg.n_power_channels].mean(axis=1).T, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=0, vmax=1)
        ax.set_ylabel("Doppler Hz")
        ax.set_title("sonar: Doppler picture, mean of all tone/microphone channels (up = hand approaching)", fontsize=9, loc="left")
        if cfg.has_lateral:
            axes[1].imshow(stream.feat[a:b, cfg.n_power_channels].T, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r", vmin=-1, vmax=1)
            axes[1].set_ylabel("Doppler Hz")
            axes[1].set_title("left/right picture from the two microphones (red = moving left, blue = moving right)", fontsize=9, loc="left")
        ax = axes[-2]                                             # the label lanes
        y_of = {name: len(lanes) - 1 - i for i, name in enumerate(lanes)}

        def box(lane: str, t0: float, t1: float, color: str, text: str = "", alpha: float = 0.9, hatch: str | None = None, h: float = 0.8) -> None:
            if t1 < t_a or t0 > t_b:
                return
            ax.add_patch(Rectangle((t0, y_of[lane] + (1 - h) / 2), max(t1 - t0, 0.04), h, color=color, alpha=alpha, hatch=hatch, lw=0))
            if text:
                ax.text(max(t0, t_a) + 0.03, y_of[lane] + 0.5, text, fontsize=6.5, va="center", clip_on=True)

        for ev in meta.get("events", []):
            t0, t1 = ev["start"] / cfg.fs, ev["end"] / cfg.fs
            kind, label = ev.get("kind", ""), ev.get("label", "")
            if kind == "block":
                box("session", t0, t1, COLORS.get(ev.get("intent", ""), "0.8"), f"block: {ev.get('intent')} {ev.get('count')}/{ev.get('target')}", 0.25)
            elif kind == "prompt":
                box("session", t0, t1, COLORS.get(ev.get("intent", ""), "0.8"), f"prompt: {ev.get('intent')}", 0.25)
            elif kind in ("scene", "calibration"):
                box("session", t0, t1, "0.85", ev.get("activity") or label, 0.8)
            elif kind == "review":
                box("label", t0, t1, COLORS["review"], f"? {ev.get('candidate')}", 0.6, hatch="//")
            elif kind == "motion":
                box("label", t0, t1, COLORS["other"], "", 0.35, h=0.35)
            elif label in GESTURE_LABELS and label != "none":
                if not ev.get("tight"):
                    box("session", t0, t1, COLORS.get(label, "0.8"), f"prompt: {label}", 0.25)
                    box("label", t0, t1, COLORS.get(label, "0.5"), label, 0.35)
                else:
                    box("label", t0, t1, COLORS.get(label, "0.5"), f"{label} {ev.get('confidence', 1):.2f}")
        for bst in table.bursts:
            if bst.role == "gesture":
                box("trainer", bst.t0, bst.t1, COLORS.get(bst.label, "0.5"), bst.label)
            elif bst.role == "none":
                box("trainer", bst.t0, bst.t1, COLORS["none"], "none", 0.7)
            else:
                box("trainer", bst.t0, bst.t1, "0.88", "", 1.0, h=0.4)
            box("physics", bst.t0, bst.t1, COLORS.get(bst.physics, "0.6"), "" if bst.physics == "none" else f"{bst.physics} {bst.physics_conf:.2f}",
                0.9 if bst.physics != "none" else 0.35)
            if use_model:
                shown = bst.model if bst.model_conf >= 0.6 else "none"
                box("model", bst.t0, bst.t1, COLORS.get(shown, "0.6"), "" if shown == "none" else f"{bst.model} {bst.model_conf:.2f}",
                    0.9 if shown != "none" else 0.35)
        n_flag = 0
        for f in rep["flags"]:
            if f["t1"] < t_a or f["t0"] > t_b or f["level"] == "info":
                continue
            n_flag += 1
            ax.axvspan(f["t0"] - 0.05, f["t1"] + 0.05, ymin=0, ymax=1, fill=False, edgecolor="red", lw=1.5)
            ax.text(f["t0"], len(lanes) + 0.05, f["kind"], color="red", fontsize=6.5, rotation=20, va="bottom", clip_on=False)
        ax.set_ylim(0, len(lanes) + 0.6)
        ax.set_yticks([y_of[name] + 0.5 for name in lanes])
        ax.set_yticklabels(["what was asked", "label (camera / prompt)", "used by the trainer as", "physics rules", "model"][:len(lanes)], fontsize=8)
        ax.set_title(f"labels and verdicts per movement burst ({n_flag} red box(es) on this page = worth a look)", fontsize=9, loc="left")
        ax = axes[-1]
        ax.plot(t, stream.activity_db[a:b], color="black", lw=0.8, label="sonar movement (dB)")
        ax.axhline(cfg.seg_min_peak_db, color="gray", ls=":", lw=0.8)
        for seg in stream.segments:
            if seg.end / rate >= t_a and seg.start / rate <= t_b:
                ax.axvspan(seg.start / rate, seg.end / rate, color="gray", alpha=0.25)
        ax.set_ylabel("movement dB")
        if speed is not None:
            ax2 = ax.twinx()
            ax2.plot(t, speed[a:b], color="tab:cyan", lw=0.9, label="camera: hand speed (m/s)")
            ax2.set_ylabel("hand speed m/s", color="tab:cyan")
            ax2.set_ylim(0, max(2.5, float(np.max(speed[a:b])) * 1.1 if b > a else 2.5))
            ax.set_title("sonar movement (black) and the camera's hand speed (cyan): they must rise and fall together", fontsize=9, loc="left")
        ax.set_xlabel("seconds")
        ax.set_xlim(t_a, t_a + seconds)
        sync = rep.get("sync") or {}
        head = f"{table.session}   person {table.person}   page {page + 1}"
        if sync:
            head += f"   clock sync: {sync.get('method')}, {sync.get('offset_ms')} ms"
        fig.suptitle(head, fontsize=11)
        fig.tight_layout()
        path = out_dir / f"qa_p{page + 1}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        files.append(path)
    return files


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def write_report(table: SessionTable, min_confidence: float, plot: bool, out: Path | None, seconds: float) -> dict[str, Any]:
    """Numbers, flags, qa.json and (if asked) the pictures of one session."""
    rep = review_session(table, min_confidence)
    target = table.folder if out is None else out / table.folder.name
    target.mkdir(parents=True, exist_ok=True)
    if plot:
        try:
            rep["pictures"] = [p.name for p in plot_session(table, rep, target, seconds)]
        except ImportError:
            print("matplotlib is not installed (pip install matplotlib): numbers only", file=sys.stderr)
    (target / "qa.json").write_text(json.dumps({**rep, "bursts": table_rows(table)}, indent=1))
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sonar.labelqa", description="Check session labels against the physics rules and the model")
    ap.add_argument("roots", nargs="+", help="a session folder, or a folder of sessions")
    ap.add_argument("--model", default="models/gesture.onnx", help="current model (physics rules only if the file is missing)")
    ap.add_argument("--min-confidence", type=float, default=0.6, help="the engine's confidence threshold")
    ap.add_argument("--seconds", type=float, default=30.0, help="seconds per picture")
    ap.add_argument("--no-plot", action="store_true", help="numbers and flags only")
    ap.add_argument("--out", help="write qa.json and the pictures here (one sub-folder per session) instead of into the session folders")
    ap.add_argument("--device", default="auto", help="where the verdicts are computed: auto (CUDA if there is one, else CPU) | cuda | cpu | none")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 16), help="CPU processes for cutting the sessions (with --no-plot)")
    ap.add_argument("--cache", help="folder that keeps the bursts of each session between runs (with --no-plot)")
    args = ap.parse_args(argv)
    folders = [f for root in args.roots for f in find_sessions(root)]
    if not folders:
        print(f"ERROR: no session ({SESSION_FILE}) under {args.roots}", file=sys.stderr)
        return 2
    model = args.model if Path(args.model).exists() else None
    out = Path(args.out) if args.out else None
    reports = []

    def show(rep: dict[str, Any]) -> None:
        reports.append(rep)
        print(one_line(rep))
        for v in rep["verdicts"]:
            print(f"   !! {v}")
        for f in rep["flags"]:
            if f["level"] in ("wrong label?", "check"):
                print(f"   [{f['t0']:6.1f} s] {f['level']}: {f['text']}")

    if args.no_plot:
        # numbers only: every burst of every session is judged in one batched pass on the device
        tables, skipped, info = build_tables(folders, model, args.device, Path(args.cache) if args.cache else None, args.workers)
        for line in skipped:
            print(f"WARNING: skipped {line}", file=sys.stderr)
        for table in tables:
            show(write_report(table, args.min_confidence, False, out, args.seconds))
        if tables:
            print(timing_line({"audio_minutes": round(info["audio_minutes"], 2), "bursts": info["bursts"], "burst_sets": info["how"],
                               "judge": info["judge"], "seconds": {**info["seconds"], "rules": 0.0}}))
    else:
        for folder in folders:                                 # pictures need the whole feature stream: one session at a time
            tables, skipped, _ = build_tables([folder], model, args.device, keep_stream=True)
            for line in skipped:
                print(f"WARNING: skipped {line}", file=sys.stderr)
            for table in tables:
                show(write_report(table, args.min_confidence, True, out, args.seconds))
    if not reports:
        return 1
    if len(reports) > 1 or Path(args.roots[0]) != folders[0]:
        tot = totals(reports)
        print_totals(tot)
        dest = (out or Path(args.roots[0])) / "labelqa_summary.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({"totals": tot, "sessions": [{k: v for k, v in r.items() if k != "flags"} for r in reports]}, indent=1))
        print(f"\nwrote qa.json per session and {dest}")
    else:
        print(f"wrote {('qa.json and ' + ', '.join(reports[0].get('pictures', []))) if reports[0].get('pictures') else 'qa.json'} "
              f"in {folders[0] if out is None else out / folders[0].name}")
    print("classifier used next to the physics rules: " + ("the trained model " + str(model) if model else "none (no model file)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

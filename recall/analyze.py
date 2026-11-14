"""analyze.py — everything that can be learned from a finished suite run WITHOUT the GPU (runs on this Mac in seconds).

Input: a run folder written by recall.suite on the 4090 (table.npz, manifest.json, jobs/*.npz = every model's
raw answers for every burst). Output, printed and saved as analysis.md in that folder:

  1. the operating point, chosen honestly. The engine fires when the top class is a gesture with confidence
     >= 0.6. How sure the model is can be shifted after training with two numbers: a temperature (stored in
     gesture.json) and a handicap on "none" (folded into the model's last layer). For each held-out human both
     numbers are chosen ONLY on the two training humans (each held out from a model trained on the other), to
     get the most gestures while non-gesture triggers stay at or under the budget; then applied to the
     held-out human. Reported next to temperature 1 and to the best-possible ("oracle") point.
  2. the engine-level score: each session replayed through the engine's own gesture rules (recall.engine_replay)
     with the held-out models: right gesture first per prompt, wrong ones, extra ones, false events per minute.
  3. where the misses are: by person, by class, bursts cut by the 1.4 s limit or not, burst strength.

How to run:   .venv/bin/python -m recall.analyze <run folder> [--arms a,b] [--no-events]
What it changes: writes <run folder>/analysis.md and analysis.json. Read-only otherwise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import REPO, STREAMS, fixed_meta
from .trainlib import CFG, score, softmax, wilson

LABELS = ("none", "swipe_left", "swipe_right", "push")
BUDGET = 0.055


class Run:
    def __init__(self, folder: Path) -> None:
        self.dir = Path(folder)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        with np.load(self.dir / "table.npz", allow_pickle=False) as z:
            self.r = {k: z[k] for k in z.files if k not in ("X", "X_wide")}
        lut = {n: i for i, n in enumerate(LABELS)}
        self.y = np.array([lut.get(str(v), 0) for v in self.r["label"]], dtype=np.int64)
        self.humans, self.seeds, self.arms = self.man["humans"], self.man["seeds"], self.man["arms"]
        self.scored = np.isin(self.r["source"], ["gesture", "burst", "idle"]) & ~self.r["locked"]
        self._cache: dict[str, np.ndarray | None] = {}

    def logits(self, job: str) -> np.ndarray | None:
        if job not in self._cache:
            p = self.dir / "jobs" / f"{job}.npz"
            self._cache[job] = np.load(p)["logits"] if p.exists() else None
        return self._cache[job]

    def outer(self, arm: str, h: str) -> list[np.ndarray]:
        out = [self.logits(f"{arm}__{h}__s{s}") for s in self.seeds]
        return [l for l in out if l is not None]

    def inner(self, arm: str, h: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Held-out answers among the TRAINING humans of fold h: (logits, labels), scored rows only."""
        ls, ys = [], []
        for h2 in [x for x in self.humans if x != h]:
            l2 = self.logits(f"{arm}__{h}__in_{h2}__s{self.seeds[0]}")
            if l2 is None:
                return None
            m = self.scored & (self.r["human"] == h2)
            ls.append(l2[m])
            ys.append(self.y[m])
        return np.concatenate(ls), np.concatenate(ys)


def apply(logits: np.ndarray, t: float, b: float) -> np.ndarray:
    """Probabilities after dividing by the temperature and handicapping "none" by b (in logit units)."""
    z = logits.astype(np.float64).copy()
    z[:, 0] -= b
    return softmax(z, t)


def rates(p: np.ndarray, y: np.ndarray, thr: float = 0.6) -> tuple[float, float]:
    arg, conf = p.argmax(axis=1), p.max(axis=1)
    pred = np.where(conf >= thr, arg, 0)
    g = y != 0
    return (float((pred[g] == y[g]).mean()) if g.any() else 0.0, float((pred[~g] != 0).mean()) if (~g).any() else 0.0)


def choose_point(logits: np.ndarray, y: np.ndarray, budget: float = BUDGET, margin: float = 0.0) -> tuple[float, float]:
    """(temperature, none handicap) with the most gestures recognised while false triggers stay <= budget - margin.

    Ties go to the point closest to "do nothing" (temperature 1, handicap 0).
    """
    best = (-1.0, 0.0, 1.0, 0.0)
    for t in (0.5, 0.6, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0):
        for b in np.linspace(-3.0, 4.0, 29):
            r, f = rates(apply(logits, t, b), y)
            if f <= budget - margin:
                key = (r, -(abs(np.log(t)) + 0.1 * abs(b)))
                if key > (best[0], best[1]):
                    best = (r, key[1], t, float(b))
    return best[2], best[3]


def operating_points(run: Run, arm: str) -> dict[str, Any]:
    """Per fold and pooled: recall / false at temperature 1, at the nested point, and at the oracle point."""
    out: dict[str, Any] = {"folds": {}}
    pooled: dict[str, list] = {k: [] for k in ("t1", "nested", "y", "h")}
    for h in run.humans:
        test = run.scored & (run.r["human"] == h)
        outer = run.outer(arm, h)
        if not outer:
            continue
        y = run.y[test]
        inner = run.inner(arm, h)
        t, b = choose_point(*inner) if inner is not None else (1.0, 0.0)
        p1 = np.mean([softmax(l[test], 1.0) for l in outer], axis=0)
        pn = np.mean([apply(l[test], t, b) for l in outer], axis=0)
        to, bo = choose_point(np.log(p1 + 1e-12), y)                       # looks at the test human: an upper bound only
        r1, f1 = rates(p1, y)
        rn, fn = rates(pn, y)
        ro, fo = rates(apply(np.log(p1 + 1e-12), to, bo), y)
        out["folds"][h] = {"t": t, "b": b, "recall_t1": r1, "false_t1": f1, "recall": rn, "false": fn, "recall_oracle": ro, "false_oracle": fo,
                           "n_gesture": int((y != 0).sum()), "n_none": int((y == 0).sum())}
        pooled["t1"].append(p1)
        pooled["nested"].append(pn)
        pooled["y"].append(y)
        pooled["h"].append(run.r["human"][test])
    if pooled["y"]:
        y, hh = np.concatenate(pooled["y"]), np.concatenate(pooled["h"])
        for key in ("t1", "nested"):
            p = np.concatenate(pooled[key])
            s = score(np.log(p + 1e-12), y, LABELS)
            out[key] = {k: s[k] for k in ("recall", "false", "hits", "n_gesture", "false_n", "n_none", "per_class", "wrong_gesture", "silent")}
            out[key]["recall_ci"], out[key]["false_ci"] = s["recall_ci"], s["false_ci"]
            m = hh != "lance"
            s2 = score(np.log(p[m] + 1e-12), y[m], LABELS)
            out[key]["no_lance"] = {k: s2[k] for k in ("recall", "false", "hits", "n_gesture", "false_n", "n_none")}
            out[key]["no_lance"]["recall_ci"] = s2["recall_ci"]
    return out


def event_scores(run: Run, arm: str, points: dict[str, Any], gate_range: bool = True, min_peak: float | None = None,
                 use_point: bool = True) -> dict[str, Any]:
    """Replay every session of each held-out human through the engine rules with that fold's models."""
    from .engine_replay import load_trace, replay, summarise, trial_table

    r = run.r
    out: dict[str, Any] = {}
    all_tabs = []
    gest = (r["source"] == "gesture") & np.isin(r["label"], LABELS[1:]) & ~r["locked"]
    for h in run.humans:
        outer = run.outer(arm, h)
        if not outer or h not in points["folds"]:
            continue
        t, b = (points["folds"][h]["t"], points["folds"][h]["b"]) if use_point else (1.0, 0.0)
        train_h = gest & (r["human"] != h)                                # the weak gate comes from the training humans' gestures
        gate = min_peak if min_peak is not None else max(CFG.seg_min_peak_db, float(np.percentile(r["peak_db"][train_h], 10)) - 6.0)
        tabs = []
        for name in sorted(set(r["session"][(r["human"] == h) & ~r["locked"]])):
            if not (STREAMS / f"{name}.npz").exists():
                continue
            trace = load_trace(str(name))
            rows = np.nonzero((r["session"] == name) & (r["seg"] >= 0))[0]
            by_seg = {int(r["seg"][i]): i for i in rows}
            if len(by_seg) != len(trace["segs"]):
                out.setdefault("warnings", []).append(f"{name}: {len(by_seg)} bursts in the run's table, {len(trace['segs'])} here")
            probs = np.zeros((len(trace["segs"]), len(LABELS)))
            probs[:, 0] = 1.0
            for k in range(len(trace["segs"])):
                i = by_seg.get(k)
                if i is not None and int(r["start"][i]) == int(trace["segs"][k][1]) and int(r["end"][i]) == int(trace["segs"][k][2]):
                    probs[k] = np.mean([apply(l[i:i + 1], t, b)[0] for l in outer], axis=0)
                elif i is not None:
                    out.setdefault("warnings", []).append(f"{name}: burst {k} has other frames in the run's table")
            fired, _ = replay(trace, probs, LABELS, CFG, min_peak_db=gate, gate_range=gate_range)
            meta = fixed_meta(json.loads((STREAMS / f"{name}.json").read_text())["meta"], str(name), run.man.get("flip", []))
            tabs.append(trial_table(meta, fired, LABELS, len(trace["state"])))
        out[h] = summarise(tabs, LABELS)
        out[h]["weak_gate_db"] = gate
        all_tabs += tabs
    if all_tabs:
        out["pooled"] = summarise(all_tabs, LABELS)
    return out


def miss_breakdown(run: Run, arm: str, points: dict[str, Any]) -> list[str]:
    """Where the held-out gestures are lost, pooled over folds (nested operating point)."""
    r = run.r
    rows_hit, rows_meta = [], []
    for h in run.humans:
        outer = run.outer(arm, h)
        if not outer or h not in points["folds"]:
            continue
        t, b = points["folds"][h]["t"], points["folds"][h]["b"]
        m = run.scored & (r["human"] == h) & (run.y != 0)
        p = np.mean([apply(l[m], t, b) for l in outer], axis=0)
        pred = np.where(p.max(axis=1) >= 0.6, p.argmax(axis=1), 0)
        rows_hit.append(pred == run.y[m])
        rows_meta.append({k: r[k][m] for k in ("forced", "peak_db", "start", "end", "order", "n_in_trial", "human", "label")})
    if not rows_hit:
        return []
    hit = np.concatenate(rows_hit)
    meta = {k: np.concatenate([d[k] for d in rows_meta]) for k in rows_meta[0]}
    dur = (meta["end"] - meta["start"] + 1) / 50.0

    def line(name: str, mask: np.ndarray) -> str:
        k, n = int(hit[mask].sum()), int(mask.sum())
        lo, hi = wilson(k, n)
        return f"  {name:<34} {k:3d}/{n:<3d} = {k / max(n, 1):.2f}  [{lo:.2f},{hi:.2f}]"

    out = [line("burst cut by the 1.4 s limit", meta["forced"]), line("burst ended by itself", ~meta["forced"]),
           line("burst shorter than 0.6 s", dur < 0.6), line("burst 0.6-1.2 s", (dur >= 0.6) & (dur < 1.2)), line("burst 1.2 s or longer", dur >= 1.2),
           line("peak below 38 dB", meta["peak_db"] < 38), line("peak 38-44 dB", (meta["peak_db"] >= 38) & (meta["peak_db"] < 44)),
           line("peak 44 dB or more", meta["peak_db"] >= 44),
           line("the only burst of its trial", meta["n_in_trial"] == 1), line("first of several bursts", (meta["n_in_trial"] > 1) & (meta["order"] == 0)),
           line("a later burst of its trial", (meta["n_in_trial"] > 1) & (meta["order"] > 0))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--arms", default="")
    ap.add_argument("--no-events", action="store_true")
    args = ap.parse_args()
    run = Run(Path(args.run))
    arms = [a for a in (args.arms.split(",") if args.arms else run.arms) if a in run.arms]
    lines = [f"# Analysis of run `{run.dir.name}`", "",
             "Gestures recognised / non-gestures that trigger at the engine's 0.6 threshold; 3-seed average; each human held out in turn.",
             f"\"nested\" = temperature and none-handicap chosen on the two TRAINING humans only (budget {BUDGET}).", "",
             "## 1. Picture-level score (the released definition)", "",
             "| arm | pooled nested recall [95% CI] / false | without lance | " + " | ".join(f"{h} (T, handicap)" for h in run.humans) +
             " | at temperature 1 | oracle point (upper bound) |", "|---|---|---|" + "---|" * len(run.humans) + "---|---|"]
    result: dict[str, Any] = {}
    for arm in arms:
        pts = operating_points(run, arm)
        result[arm] = {"points": pts}
        if "nested" not in pts:
            continue
        n, t1 = pts["nested"], pts["t1"]
        cells = [f"{f['recall']:.3f} / {f['false']:.3f} ({f['t']:.2f}, {f['b']:+.2f})" for f in (pts["folds"][h] for h in run.humans if h in pts["folds"])]
        orc = np.average([f["recall_oracle"] for f in pts["folds"].values()], weights=[f["n_gesture"] for f in pts["folds"].values()])
        nl = n["no_lance"]
        lines.append(f"| {arm} | **{n['recall']:.3f}** [{n['recall_ci'][0]:.2f},{n['recall_ci'][1]:.2f}] ({n['hits']}/{n['n_gesture']}) / {n['false']:.3f} ({n['false_n']}/{n['n_none']}) | "
                     f"{nl['recall']:.3f} [{nl['recall_ci'][0]:.2f},{nl['recall_ci'][1]:.2f}] / {nl['false']:.3f} | " + " | ".join(cells) +
                     f" | {t1['recall']:.3f} / {t1['false']:.3f} | {orc:.3f} |")
    if not args.no_events:
        lines += ["", "## 2. Engine-level score (what the demo would do): held-out models replayed through the engine's rules", "",
                  "Per prompt: the right gesture fired first (\"recall\"), fired once and nothing else (\"clean\"), a wrong gesture first; "
                  "false events = gestures fired in the scene / breathing sessions and between prompts.", "",
                  "| arm | engine setting | pooled recall | clean | wrong first | false events per min | " + " | ".join(f"{h} recall" for h in run.humans) + " |",
                  "|---|---|---|---|---|---|" + "---|" * len(run.humans)]
        for arm in arms:
            pts = result[arm]["points"]
            if "nested" not in pts:
                continue
            result[arm]["events"] = {}
            for name, kw in (("as shipped", {}), ("range gate off", {"gate_range": False}), ("range gate off, weak gate 18 dB", {"gate_range": False, "min_peak": 18.0})):
                ev = event_scores(run, arm, pts, **kw)
                result[arm]["events"][name] = ev
                if "pooled" not in ev:
                    continue
                p = ev["pooled"]
                lines.append(f"| {arm} | {name} | {p['recall']:.3f} ({p['correct']}/{p['n_prompts']}) | {p['clean_rate']:.3f} | {p['wrong']} | "
                             f"{p['false_per_min']:.2f} ({p['outside_fired']} in {p['outside_minutes']:.0f} min) | " +
                             " | ".join(f"{ev[h]['recall']:.3f}" if h in ev else "-" for h in run.humans) + " |")
                for w in ev.get("warnings", [])[:3]:
                    lines.append(f"|  | WARNING | {w} |")
    lines += ["", "## 3. Where the held-out gestures are lost (nested point, pooled over the three humans)", ""]
    for arm in arms:
        if "nested" in result[arm]["points"]:
            lines += [f"**{arm}**", "```"] + miss_breakdown(run, arm, result[arm]["points"]) + ["```", ""]
    text = "\n".join(lines) + "\n"
    (run.dir / "analysis.md").write_text(text)
    (run.dir / "analysis.json").write_text(json.dumps(result, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(text)


if __name__ == "__main__":
    main()

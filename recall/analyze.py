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
  3. every recipe scored again as it would run in an engine that cuts movement bursts ANOTHER way than the recipe
     was trained on (std = as shipped, rel12 = ended 12 dB below the burst's own peak, rel12nc = that and click
     rejection off). The operating point is chosen again, on the training humans' pictures cut that same way.
     This table decides which model ships together with which engine cutting.
  4. where the misses are: by person, by class, bursts cut by the 1.4 s limit or not, burst strength.

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
BUDGET = 0.0551      # 6/109 = 0.05505, the released model's false-trigger share on mahimn


class Run:
    """A finished suite run: its manifest, the recorded rows of every cutting it used, and every model's answers."""

    def __init__(self, folder: Path) -> None:
        self.dir = Path(folder)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        self.humans, self.seeds, self.arms = self.man["humans"], self.man["seeds"], self.man["arms"]
        self._tables: dict[str, tuple] = {}
        self._cache: dict[tuple[str, str], np.ndarray | None] = {}

    def variant(self, arm: str) -> str:
        return (self.man.get("arm_specs", {}).get(arm, {}) or {}).get("table") or "std"

    def tab(self, arm: str, cut: str | None = None) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """(columns, labels, scored-rows mask) of one burst cutting: `cut`, or the one this arm was trained on."""
        v = cut or self.variant(arm)
        if v not in self._tables:
            name = "table.npz" if v == "std" else f"table_{v}.npz"
            with np.load(self.dir / name, allow_pickle=False) as z:
                r = {k: z[k] for k in z.files if k not in ("X", "X_wide")}
            lut = {n: i for i, n in enumerate(LABELS)}
            y = np.array([lut.get(str(x), 0) for x in r["label"]], dtype=np.int64)
            self._tables[v] = (r, y, np.isin(r["source"], ["gesture", "burst", "idle"]) & ~r["locked"])
        return self._tables[v]

    def logits(self, job: str, cut: str | None = None) -> np.ndarray | None:
        """One model's raw answers for every row of a cutting's table (`cut`; default: the cutting it was trained on)."""
        key = "logits" if cut in (None, self.variant(job.split("__")[0])) else f"logits__{cut}"
        if (job, key) not in self._cache:
            p = self.dir / "jobs" / f"{job}.npz"
            val = None
            if p.exists():
                with np.load(p) as z:
                    val = z[key] if key in z.files else None
            self._cache[(job, key)] = val
        return self._cache[(job, key)]

    def views(self, job: str) -> list[np.ndarray] | None:
        """The model's answers for the picture as the engine cuts it AND for the copies cut 3 and 6 frames earlier / later."""
        p = self.dir / "jobs" / f"{job}.npz"
        if not p.exists():
            return None
        with np.load(p) as z:
            if "logits_shifted" not in z.files:
                return None
            return [z["logits"].astype(np.float64)] + [v.astype(np.float64) for v in z["logits_shifted"]]

    def outer(self, arm: str, h: str, cut: str | None = None) -> list[np.ndarray]:
        out = [self.logits(f"{arm}__{h}__s{s}", cut) for s in self.seeds]
        return [l for l in out if l is not None]

    def outer_views(self, arm: str, h: str) -> list[list[np.ndarray]]:
        out = [self.views(f"{arm}__{h}__s{s}") for s in self.seeds]
        return [v for v in out if v is not None]

    def inner(self, arm: str, h: str, cut: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Held-out answers among the TRAINING humans of fold h: (logits, labels, whose row it is), scored rows only."""
        r, y, scored = self.tab(arm, cut)
        ls, ys, who = [], [], []
        for h2 in [x for x in self.humans if x != h]:
            l2 = self.logits(f"{arm}__{h}__in_{h2}__s{self.seeds[0]}", cut)
            if l2 is None:
                return None
            m = scored & (r["human"] == h2)
            ls.append(l2[m])
            ys.append(y[m])
            who.append(r["human"][m])
        return np.concatenate(ls), np.concatenate(ys), np.concatenate(who)


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


def choose_point(logits: np.ndarray | list[np.ndarray], y: np.ndarray, budget: float = BUDGET, margin: float = 0.0,
                 person: np.ndarray | None = None) -> tuple[float, float]:
    """(temperature, none handicap) with the most gestures recognised while false triggers stay <= budget - margin.

    `logits`: one model's answers, or a list of models' answers for the same rows; then the point is applied to
    each model and the probabilities are averaged, exactly as the packaged graph does (recall.package).
    `person`: whose row each one is. Then EVERY person's own false-trigger share must stay within the budget,
    not just their average (the "each person" setting; one person with easy non-gestures cannot hide another).
    Ties go to the point closest to "do nothing" (temperature 1, handicap 0).
    """
    members = logits if isinstance(logits, list) else [logits]
    groups = [person == h for h in np.unique(person)] if person is not None else []
    best = (-1.0, 0.0, 1.0, 0.0)
    for t in (0.5, 0.6, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0):
        for b in np.linspace(-3.0, 4.0, 29):
            p = np.mean([apply(l, t, b) for l in members], axis=0)
            r, f = rates(p, y)
            if groups:
                f = max(rates(p[g], y[g])[1] for g in groups)
            if f <= budget - margin:
                key = (r, -(abs(np.log(t)) + 0.1 * abs(b)))
                if key > (best[0], best[1]):
                    best = (r, key[1], t, float(b))
    return best[2], best[3]


def operating_points(run: Run, arm: str, cut: str | None = None) -> dict[str, Any]:
    """Per fold and pooled: recall / false at temperature 1, at the nested point, and at the oracle point.

    `cut`: score the arm's models on the recordings cut this way (the engine's cutting at run time) instead of
    the way the arm was trained on; the nested point is then chosen on the training humans' pictures cut that way.
    """
    out: dict[str, Any] = {"folds": {}}
    pooled: dict[str, list] = {k: [] for k in ("t1", "nested", "each", "y", "h", "members")}
    r_, y_, scored_ = run.tab(arm, cut)
    for h in run.humans:
        test = scored_ & (r_["human"] == h)
        outer = run.outer(arm, h, cut)
        if not outer:
            continue
        y = y_[test]
        inner = run.inner(arm, h, cut)
        t, b = choose_point(inner[0], inner[1]) if inner is not None else (1.0, 0.0)
        te, be = choose_point(inner[0], inner[1], person=inner[2]) if inner is not None else (1.0, 0.0)   # each training person within budget
        p1 = np.mean([softmax(l[test], 1.0) for l in outer], axis=0)
        pn = np.mean([apply(l[test], t, b) for l in outer], axis=0)
        pe = np.mean([apply(l[test], te, be) for l in outer], axis=0)
        to, bo = choose_point(np.log(p1 + 1e-12), y)                       # looks at the test human: an upper bound only
        r1, f1 = rates(p1, y)
        rn, fn = rates(pn, y)
        re_, fe = rates(pe, y)
        ro, fo = rates(apply(np.log(p1 + 1e-12), to, bo), y)
        out["folds"][h] = {"t": t, "b": b, "recall_t1": r1, "false_t1": f1, "recall": rn, "false": fn, "recall_oracle": ro, "false_oracle": fo,
                           "t_each": te, "b_each": be, "recall_each": re_, "false_each": fe,
                           "n_gesture": int((y != 0).sum()), "n_none": int((y == 0).sum())}
        pooled["t1"].append(p1)
        pooled["nested"].append(pn)
        pooled["each"].append(pe)
        pooled["y"].append(y)
        pooled["h"].append(r_["human"][test])
        pooled["members"].append([l[test].astype(np.float64) for l in outer])
    if pooled["y"]:
        y, hh = np.concatenate(pooled["y"]), np.concatenate(pooled["h"])
        # The operating point for the model that SHIPS (trained on everyone): chosen on all three humans' held-out
        # answers at once (each human answered by models that never saw them), applied the way the shipped graph
        # applies it: to each seed's network, then the probabilities are averaged. Not used for any score above.
        members = [np.concatenate([fold[k] for fold in pooled["members"]]) for k in range(min(len(fold) for fold in pooled["members"]))]
        ft, fb = choose_point(members, y)
        rf, ff = rates(np.mean([apply(l, ft, fb) for l in members], axis=0), y)
        out["final_point"] = {"t": ft, "b": fb, "recall_on_the_data_it_was_chosen_on": rf, "false_on_the_data_it_was_chosen_on": ff}
        # ... and the stricter one: every one of the three people within the budget on their own held-out answers
        et, eb = choose_point(members, y, person=hh)
        pe_all = np.mean([apply(l, et, eb) for l in members], axis=0)
        re_all, fe_all = rates(pe_all, y)
        out["final_point_each"] = {"t": et, "b": eb, "recall_on_the_data_it_was_chosen_on": re_all, "false_on_the_data_it_was_chosen_on": fe_all,
                                   "false_per_person": {str(h): rates(pe_all[hh == h], y[hh == h])[1] for h in np.unique(hh)}}
        for key in ("t1", "nested", "each"):
            p = np.concatenate(pooled[key])
            s = score(np.log(p + 1e-12), y, LABELS)
            out[key] = {k: s[k] for k in ("recall", "false", "hits", "n_gesture", "false_n", "n_none", "per_class", "wrong_gesture", "silent")}
            out[key]["recall_ci"], out[key]["false_ci"] = s["recall_ci"], s["false_ci"]
            m = hh != "lance"
            s2 = score(np.log(p[m] + 1e-12), y[m], LABELS)
            out[key]["no_lance"] = {k: s2[k] for k in ("recall", "false", "hits", "n_gesture", "false_n", "n_none")}
            out[key]["no_lance"]["recall_ci"] = s2["recall_ci"]
    return out


def shift_average(run: Run, arm: str) -> dict[str, Any] | None:
    """The same honest score when every model answers five slightly shifted cuts of the picture and the answers are
    averaged (it can be built into the exported model). The operating point is again chosen on the training humans."""
    r_, y_, scored_ = run.tab(arm)
    pooled_p, pooled_y = [], []
    for h in run.humans:
        test = scored_ & (r_["human"] == h)
        views = run.outer_views(arm, h)
        if not views:
            return None
        inner_l, inner_y = [], []
        for h2 in [x for x in run.humans if x != h]:
            v2 = run.views(f"{arm}__{h}__in_{h2}__s{run.seeds[0]}")
            if v2 is None:
                return None
            m2 = scored_ & (r_["human"] == h2)
            inner_l.append(np.log(np.mean([softmax(v[m2], 1.0) for v in v2], axis=0) + 1e-12))
            inner_y.append(y_[m2])
        t, b = choose_point(np.concatenate(inner_l), np.concatenate(inner_y))
        # per model: average over the shifted cuts at temperature 1, then the operating point on that average, then over seeds
        per_model = [np.log(np.mean([softmax(v[test], 1.0) for v in vs], axis=0) + 1e-12) for vs in views]
        pooled_p.append(np.mean([apply(lg, t, b) for lg in per_model], axis=0))
        pooled_y.append(y_[test])
    p, y = np.concatenate(pooled_p), np.concatenate(pooled_y)
    sc = score(np.log(p + 1e-12), y, LABELS)
    return {k: sc[k] for k in ("recall", "false", "hits", "n_gesture", "false_n", "n_none")}


def event_scores(run: Run, arm: str, points: dict[str, Any], gate_range: bool = True, min_peak: float | None = None,
                 use_point: bool = True, entering_cap_s: float | None = None, cut: str | None = None, each: bool = False) -> dict[str, Any]:
    """Replay every session of each held-out human through the engine rules with that fold's models.

    `cut`: the engine's burst cutting at run time (default: the one the arm was trained on); `points` must then
    come from operating_points(run, arm, cut). `each`: use the fold's stricter setting (each training person
    within the budget) instead of the pooled one.
    """
    from .engine_replay import load_trace, replay, summarise, trial_table

    r = run.tab(arm, cut)[0]
    variant = cut or run.variant(arm)
    out: dict[str, Any] = {}
    all_tabs = []
    gest = (r["source"] == "gesture") & np.isin(r["label"], LABELS[1:]) & ~r["locked"]
    for h in run.humans:
        outer = run.outer(arm, h, cut)
        if not outer or h not in points["folds"]:
            continue
        f_ = points["folds"][h]
        t, b = ((f_["t_each"], f_["b_each"]) if each else (f_["t"], f_["b"])) if use_point else (1.0, 0.0)
        train_h = gest & (r["human"] != h)                                # the weak gate comes from the training humans' gestures
        gate = min_peak if min_peak is not None else max(CFG.seg_min_peak_db, float(np.percentile(r["peak_db"][train_h], 10)) - 6.0)
        tabs = []
        for name in sorted(set(r["session"][(r["human"] == h) & ~r["locked"]])):
            if not (STREAMS / f"{name}.npz").exists():
                continue
            trace = load_trace(str(name), variant)
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
            fired, _ = replay(trace, probs, LABELS, CFG, min_peak_db=gate, gate_range=gate_range, entering_cap_s=entering_cap_s)
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
    r, y_all, scored = run.tab(arm)
    rows_hit, rows_meta = [], []
    for h in run.humans:
        outer = run.outer(arm, h)
        if not outer or h not in points["folds"]:
            continue
        t, b = points["folds"][h]["t"], points["folds"][h]["b"]
        m = scored & (r["human"] == h) & (y_all != 0)
        p = np.mean([apply(l[m], t, b) for l in outer], axis=0)
        pred = np.where(p.max(axis=1) >= 0.6, p.argmax(axis=1), 0)
        rows_hit.append(pred == y_all[m])
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


EVENTS_KEY = "range gate off, walking-in gate 3 s"      # the MacBook start script's engine setting


def pick_winner(run: Run, result: dict[str, Any]) -> list[str]:
    """The model + engine cutting to ship, by the rule written down before any result existed
    (.runs/fable_recall.progress.md, 2026-11-14 06:29). Returns report lines; stores the pick in result["_pick"]."""
    cands = []
    for k, arm in enumerate(run.arms):
        pts = result.get(arm, {}).get("points", {})
        if "nested" not in pts or len(pts["folds"]) != len(run.humans):
            continue
        own = run.variant(arm)
        wide = (run.man.get("arm_specs", {}).get(arm, {}) or {}).get("features") == "wide"
        rows = [(own, pts, result[arm].get("events", {}).get(EVENTS_KEY, {}), result[arm].get("events_each", {}))]
        rows += [(cut, c["points"], c.get("events", {}), c.get("events_each", {})) for cut, c in result[arm].get("cross", {}).items()]
        for cut, p_, ev, ev_e in rows:
            n, e_ = p_["nested"], p_.get("each")
            cands.append({"arm": arm, "order": k, "trained_on": own, "cut": cut, "wide": wide, "hits": n["hits"], "n": n["n_gesture"],
                          "recall": n["recall"], "ci": n["recall_ci"], "false": n["false"], "false_n": n["false_n"], "n_none": n["n_none"],
                          "folds": p_["folds"], "events": ev.get("pooled"), "no_lance": n.get("no_lance"),
                          "each": e_, "events_each": ev_e.get("pooled"),
                          "worst": max(f["false"] for f in p_["folds"].values()),
                          "worst_each": max(f["false_each"] for f in p_["folds"].values()),
                          # rule 1: its own cutting, or a recipe trained on today's cutting run with the 12 dB cutting
                          "candidate": cut == own or (own == "std" and cut == "rel12")})

    def best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        ok = [c for c in rows if c["false"] <= BUDGET]
        if not ok:
            return None
        top = max(c["hits"] for c in ok)
        near = [c for c in ok if c["hits"] >= top - 1]                       # rule 3: within one gesture of each other
        return min(near, key=lambda c: (c["false"], c["cut"] != c["trained_on"], c["order"]))

    elig = [c for c in cands if c["candidate"] and not c["wide"]]
    filt_on, filt_off = best([c for c in elig if c["cut"] != "rel12nc"]), best([c for c in elig if c["cut"] == "rel12nc"])
    pick = filt_on
    if filt_off is not None and (filt_on is None or filt_off["hits"] >= filt_on["hits"] + 5):      # rule 4
        pick = filt_off
    lines = ["## 0. The pick, by the rule written before any result existed (progress file, 06:29)", ""]
    if pick is None:
        within = sorted(elig, key=lambda c: c["false"])[:3]
        lines += [f"No eligible model + cutting keeps false triggers at or under {BUDGET}. Lowest: " +
                  "; ".join(f"{c['arm']} with {c['cut']} cutting: {c['recall']:.3f} at {c['false']:.3f}" for c in within), ""]
        return lines

    def show(c: dict[str, Any]) -> str:
        return (f"{c['hits']}/{c['n']} = {c['recall']:.3f} [{c['ci'][0]:.2f},{c['ci'][1]:.2f}] at {c['false_n']}/{c['n_none']} = {c['false']:.3f}")

    bench = pick["folds"].get("mahimn")
    met = bench is not None and bench["recall"] >= 0.70 and bench["false"] <= BUDGET
    finals = len(list((run.dir / "jobs").glob(f"{pick['arm']}__final__s*.pt")))
    result["_pick"] = {k: pick[k] for k in ("arm", "trained_on", "cut", "hits", "n", "recall", "false", "false_n", "n_none")}
    result["_pick"].update(target_met_on_mahimn=bool(met), final_models_in_run=finals)
    lines += [f"**{pick['arm']}** (trained on {pick['trained_on']} cutting), engine cutting **{pick['cut']}**: all three people, each held out in turn: "
              f"gestures recognised {show(pick)} false triggers.", "",
              "| held-out person | gestures recognised | non-gestures that trigger | confidence setting (temperature, none handicap) |", "|---|---|---|---|"]
    for h in run.humans:
        f = pick["folds"][h]
        lines.append(f"| {h} | {round(f['recall'] * f['n_gesture'])}/{f['n_gesture']} = {f['recall']:.3f} | {round(f['false'] * f['n_none'])}/{f['n_none']} = {f['false']:.3f} | {f['t']:.2f}, {f['b']:+.2f} |")
    nl = pick["no_lance"]
    lines += ["", f"- Without Lance (people who gesture at the prompted pace): {nl['hits']}/{nl['n_gesture']} = {nl['recall']:.3f} at {nl['false']:.3f}." if nl else "",
              f"- Target (0.70 or more at 0.055 or less) on the original benchmark person, Mahimn: **{'MET' if met else 'NOT met'}**"
              + (f" ({bench['recall']:.3f} at {bench['false']:.3f})." if bench else " (no Mahimn fold in this run)."),
              (f"- Engine level ({EVENTS_KEY}): right gesture first on {pick['events']['correct']}/{pick['events']['n_prompts']} prompts = {pick['events']['recall']:.3f}, "
               f"{pick['events']['outside_fired']} gestures fired where nobody gestured in {pick['events']['outside_minutes']:.0f} min." if pick["events"] else "- Engine level: not computed (--no-events)."),
              f"- Click filter: best with it on {show(filt_on) if filt_on else 'none within budget'}; best with it off {show(filt_off) if filt_off else 'none within budget'} "
              "(off is taken only if it wins by 5 gestures or more)."]
    if finals >= len(run.seeds):
        lines.append(f"- Its models trained on everyone are in this run folder. Package: `.venv/bin/python -m recall.package {rel_run(run)} {pick['arm']} "
                     f"--out models-release-mac-recall --cutting {pick['cut']} --no-range-gate --entering-gate-s 3`")
    else:
        lines.append(f"- Its models trained on everyone are NOT in this run folder ({finals} of {len(run.seeds)}). On the 4090, about 5 minutes: "
                     f"`powershell -ExecutionPolicy Bypass -File scripts\\recall_suite.ps1 --stamp {run.dir.name} --finals {pick['arm']}`")
    # the stricter setting (each training person within the budget), added after the first results: same pick?
    e = pick["each"]
    if e is not None:
        strict = [dict(c, hits=c["each"]["hits"], false=c["each"]["false"], recall=c["each"]["recall"], ci=c["each"]["recall_ci"],
                       false_n=c["each"]["false_n"], n_none=c["each"]["n_none"]) for c in elig if c["each"] is not None]
        inside = [c for c in strict if c["worst_each"] <= BUDGET]          # every held-out person within the budget
        s_on = best([c for c in inside if c["cut"] != "rel12nc"])
        mb = pick["folds"].get("mahimn")
        met_e = mb is not None and mb["recall_each"] >= 0.70 and mb["false_each"] <= BUDGET
        result["_pick"]["each"] = {"hits": e["hits"], "n": e["n_gesture"], "recall": e["recall"], "false": e["false"], "false_n": e["false_n"],
                                   "n_none": e["n_none"], "worst_person_false": pick["worst_each"], "target_met_on_mahimn": bool(met_e),
                                   "pick_by_this_setting": (s_on or {}).get("arm"), "pick_cut_by_this_setting": (s_on or {}).get("cut")}
        lines += ["", "**The same pick at the stricter confidence setting** (every training person within the budget, not their average; "
                  "added after the first results, see section 2d):", "",
                  f"- All three: {e['hits']}/{e['n_gesture']} = {e['recall']:.3f} [{e['recall_ci'][0]:.2f},{e['recall_ci'][1]:.2f}] at "
                  f"{e['false_n']}/{e['n_none']} = {e['false']:.3f}. Per held-out person: " +
                  "; ".join(f"{h} {round(f['recall_each'] * f['n_gesture'])}/{f['n_gesture']} = {f['recall_each']:.3f} at "
                            f"{round(f['false_each'] * f['n_none'])}/{f['n_none']} = {f['false_each']:.3f}" for h, f in pick["folds"].items()) + ".",
                  f"- Target on Mahimn at this setting: **{'MET' if met_e else 'NOT met'}**" + (f" ({mb['recall_each']:.3f} at {mb['false_each']:.3f})." if mb else "."),
                  (f"- Engine level: right gesture first on {pick['events_each']['correct']}/{pick['events_each']['n_prompts']} prompts = {pick['events_each']['recall']:.3f}, "
                   f"{pick['events_each']['outside_fired']} gestures fired where nobody gestured in {pick['events_each']['outside_minutes']:.0f} min." if pick["events_each"] else "- Engine level: not computed."),
                  f"- Largest false-trigger share on any one held-out person: pooled setting {pick['worst']:.3f}, stricter setting {pick['worst_each']:.3f} (budget {BUDGET}).",
                  f"- Candidates that keep EVERY held-out person within the budget at the stricter setting: {len(inside)} of {len(strict)}"
                  + (f"; the best of them (rules 2-3): **{s_on['arm']}** with {s_on['cut']} cutting, {show(s_on)}." if s_on else "."), ""]
    wide = best([c for c in cands if c["wide"] and c["cut"] == c["trained_on"]])
    if wide is not None and wide["hits"] >= pick["hits"] + 10:
        lines.append(f"- NOTE: {wide['arm']} (wide pictures, the engine cannot make them yet) scores {show(wide)}: 10 or more gestures better. Worth building the engine profile.")
    others = sorted([c for c in cands if c["false"] <= BUDGET and c is not pick], key=lambda c: (-c["hits"], c["false"]))[:8]
    lines += ["", "Next best, any recipe and any cutting (for checking the pick; \"-\" = not a candidate by rule 1, or wide):", "",
              "| arm | trained on | engine cuts | pooled | candidate |", "|---|---|---|---|---|"]
    lines += [f"| {c['arm']} | {c['trained_on']} | {c['cut']} | {show(c)} | {'yes' if c['candidate'] and not c['wide'] else '-'} |" for c in others]
    return lines + [""]


def rel_run(run: Run) -> str:
    try:
        return str(run.dir.resolve().relative_to(REPO))
    except ValueError:
        return str(run.dir)


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
            for name, kw in (("as shipped", {}), ("range gate off", {"gate_range": False}),
                             (EVENTS_KEY, {"gate_range": False, "entering_cap_s": 3.0}),
                             ("the same, weak gate 18 dB", {"gate_range": False, "entering_cap_s": 3.0, "min_peak": 18.0})):
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
    lines += ["", "## 2b. Averaging each model's answer over five slightly shifted cuts of the picture (nested point, pooled)", "",
              "| arm | as the engine cuts it | averaged over shifts of 0, +-3, +-6 frames |", "|---|---|---|"]
    for arm in arms:
        pts = result[arm]["points"]
        sa = shift_average(run, arm) if "nested" in pts else None
        if sa is not None:
            result[arm]["shift_average"] = sa
            n = pts["nested"]
            lines.append(f"| {arm} | {n['recall']:.3f} / {n['false']:.3f} | {sa['recall']:.3f} ({sa['hits']}/{sa['n_gesture']}) / {sa['false']:.3f} ({sa['false_n']}/{sa['n_none']}) |")
    lines += ["", "## 2c. Every recipe in an engine that cuts bursts another way than it was trained on", "",
              "Operating point chosen again on the training humans, on pictures cut the run-time way. Engine level = range gate off, "
              "walking-in gate 3 s (the MacBook start script). The first row of each recipe is its own cutting, for comparison.", "",
              "| arm | trained on | engine cuts | pooled recall [95% CI] / false | " + " | ".join(f"{h} recall / false" for h in run.humans) +
              " | engine level: right gesture first | false events per min |", "|---|---|---|---|" + "---|" * len(run.humans) + "---|---|"]
    each_rows: list[tuple] = []
    for arm in arms:
        if "nested" not in result[arm]["points"]:
            continue
        own = run.variant(arm)
        result[arm]["cross"] = {}
        for cut in [own] + [v for v in run.man.get("tables", ["std"]) if v != own]:
            pts = result[arm]["points"] if cut == own else operating_points(run, arm, cut)
            if "nested" not in pts or len(pts["folds"]) != len(run.humans):
                continue                                           # this run holds no answers of this arm for that cutting
            ev = {} if args.no_events else event_scores(run, arm, pts, gate_range=False, entering_cap_s=3.0, cut=cut)
            ev_each = {} if args.no_events else event_scores(run, arm, pts, gate_range=False, entering_cap_s=3.0, cut=cut, each=True)
            if cut != own:
                result[arm]["cross"][cut] = {"points": pts, "events": ev, "events_each": ev_each}
            else:
                result[arm]["events_each"] = ev_each
            each_rows.append((arm, own, cut, pts, ev_each.get("pooled")))
            n = pts["nested"]
            cells = [f"{f['recall']:.3f} / {f['false']:.3f}" for f in (pts["folds"][h] for h in run.humans)]
            e = ev.get("pooled")
            lines.append(f"| {arm} | {own} | {'**' + cut + '**' if cut != own else cut} | {n['recall']:.3f} [{n['recall_ci'][0]:.2f},{n['recall_ci'][1]:.2f}] "
                         f"({n['hits']}/{n['n_gesture']}) / {n['false']:.3f} ({n['false_n']}/{n['n_none']}) | " + " | ".join(cells) + " | " +
                         (f"{e['recall']:.3f} ({e['correct']}/{e['n_prompts']}) | {e['false_per_min']:.2f} ({e['outside_fired']} in {e['outside_minutes']:.0f} min) |" if e else "- | - |"))
    lines += ["", "## 2d. The stricter confidence setting: EVERY training person within the budget, not their average", "",
              "Added 2026-11-14 10:23, after the first results showed the pooled setting running over the budget on Mahimn "
              "(Lance's non-gestures almost never trigger, so an average over Lance + Aarsh hides a loose setting). "
              "Reported next to the pooled setting above, never instead of it. \"worst\" = the held-out person with the largest "
              f"false-trigger share; * = over the budget of {BUDGET}.", "",
              "| arm | trained on | engine cuts | pooled recall [95% CI] / false | " + " | ".join(f"{h} recall / false" for h in run.humans) +
              " | worst person's false | engine level: right gesture first | false events per min |", "|---|---|---|---|" + "---|" * len(run.humans) + "---|---|---|"]
    for arm, own, cut, pts, e in each_rows:
        n = pts["each"]
        cells = [f"{f['recall_each']:.3f} / {f['false_each']:.3f}" for f in (pts["folds"][h] for h in run.humans)]
        worst = max(f["false_each"] for f in pts["folds"].values())
        lines.append(f"| {arm} | {own} | {'**' + cut + '**' if cut != own else cut} | {n['recall']:.3f} [{n['recall_ci'][0]:.2f},{n['recall_ci'][1]:.2f}] "
                     f"({n['hits']}/{n['n_gesture']}) / {n['false']:.3f} ({n['false_n']}/{n['n_none']}) | " + " | ".join(cells) + f" | {worst:.3f}{'*' if worst > BUDGET else ''} | " +
                     (f"{e['recall']:.3f} ({e['correct']}/{e['n_prompts']}) | {e['false_per_min']:.2f} ({e['outside_fired']} in {e['outside_minutes']:.0f} min) |" if e else "- | - |"))
    at = lines.index("## 1. Picture-level score (the released definition)")
    lines[at:at] = pick_winner(run, result)
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

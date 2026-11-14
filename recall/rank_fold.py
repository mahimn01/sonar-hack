"""rank_fold.py — rank a suite run's recipes on ONE held-out person and compare each with the model shipped now.

Why: "does any retrained recipe clearly beat what is in the engine?" needs two things the main analysis does not
print: the ranking on the benchmark person alone, and a gesture-by-gesture comparison with the shipped model on
the same prompts (two scores 3 gestures apart on 81 prompts can be pure chance). It also works on a run that is
still training: a fold that lacks any of its runs is listed as unfinished instead of being scored from fewer models.

What it prints (markdown):
  A. the ranking with the confidence setting chosen on the training people's AVERAGE false-trigger share
     (the rule written before the results), B. the same with EACH training person inside the budget.
     Candidates: every recipe with the burst cutting it was trained on, and recipes trained on the engine's
     default cutting run with the 12 dB cutting. Order: inside the budget first, then most gestures recognised.
     Last column: gestures this recipe gets that the shipped model misses / the reverse, and the two-sided sign
     test p (how likely a split that uneven is by chance).
  C. the recipes whose fold for this person is not finished, with the single runs that do exist scored at the
     natural setting (temperature 1, no handicap).
  D. every recipe against the model to beat at EQUAL false-trigger counts: the most gestures each recognises when
     at most k non-gestures trigger, its confidence setting swept for that. A recipe that recognises more only
     because it runs at a looser setting shows no gain here. This table looks at the held-out person for every
     model alike: it compares models, it is not an honest score of any of them. It needs no setting runs, so
     recipes that lack one are included (marked with their number of networks).

How to run:
    .venv/bin/python -m recall.rank_fold <run folder> [--person mahimn] [--against models-release-mac-recall] [--min-seeds 2]
--min-seeds 2 also ranks folds that lack one held-out run (marked "2 of 3 networks"); a fold that lacks a setting
run is never ranked, because its confidence setting could not be chosen the honest way.
What it changes: nothing (reads the run folder and the model, prints).
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

from .analyze import BUDGET, Run, apply, choose_point
from .common import REPO
from .engine_replay import onnx_probs


def sign_p(a: int, b: int) -> float:
    """Two-sided sign test: chance of a split at least this uneven if gains and losses were equally likely."""
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


LEVELS = (3, 5, 6, 8, 10)                                   # false-trigger counts compared in section D


def ceiling(members: list[np.ndarray], y: np.ndarray) -> dict[int, tuple[int, np.ndarray]]:
    """For each k in LEVELS: (most gestures recognised with at most k non-gestures triggering, which gestures),
    over a sweep of the temperature and the "none" handicap applied the way the packaged model applies them."""
    g = y != 0
    best = {k: (-1, np.zeros(int(g.sum()), bool)) for k in LEVELS}
    for t in (0.5, 0.6, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0):
        for b in np.linspace(-4.0, 6.0, 81):
            p = np.mean([apply(l, t, float(b)) for l in members], axis=0)
            pred = np.where(p.max(axis=1) >= 0.6, p.argmax(axis=1), 0)
            ok, f = pred[g] == y[g], int((pred[~g] != 0).sum())
            for k in LEVELS:
                if f <= k and int(ok.sum()) > best[k][0]:
                    best[k] = (int(ok.sum()), ok)
    return best


def counts(pred: np.ndarray, y: np.ndarray) -> dict[str, int]:
    g = y != 0
    return {"hits": int((pred[g] == y[g]).sum()), "n": int(g.sum()), "false": int((pred[~g] != 0).sum()), "n_none": int((~g).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank a suite run's recipes on one held-out person against the shipped model.")
    ap.add_argument("run", help="run folder written by recall.suite")
    ap.add_argument("--person", default="mahimn", help="the held-out person to rank on (default: the benchmark person)")
    ap.add_argument("--against", default="models-release-mac-recall", help="folder of the model to beat (gesture.onnx + gesture.json)")
    ap.add_argument("--against-cutting", choices=("std", "rel12", "rel12nc"),
                    help="burst cutting the model to beat runs with (default: what its gesture.json says, else rel12)")
    ap.add_argument("--min-seeds", type=int, default=0,
                    help="rank a fold from this many held-out networks when some of its runs are missing (default: all; such rows are "
                         "marked). A fold whose setting runs are missing is never ranked.")
    args = ap.parse_args()
    run = Run(Path(args.run))
    need = min(args.min_seeds, len(run.seeds)) if args.min_seeds else len(run.seeds)
    h = args.person
    cuts = run.man.get("tables", ["std"])
    shipped = Path(args.against) if Path(args.against).is_absolute() else REPO / args.against
    rw = json.loads((shipped / "gesture.json").read_text()).get("run_with")
    base_cut = args.against_cutting or ("rel12" if rw is None else ("std" if not rw.get("env") else "rel12" + ("nc" if "--no-click-rejection" in rw.get("flags", []) else "")))
    any_arm = run.arms[0]
    if base_cut not in cuts:
        raise SystemExit(f"This run folder has no recordings cut the {base_cut} way (it has {cuts}), and {shipped.name} runs with {base_cut}. "
                         f"Pass --against-cutting {cuts[0]} to compare on a cutting this run has.")

    # the shipped model on this person's rows, as the engine cuts them for it; gestures keyed by prompt
    with np.load(run.dir / ("table.npz" if base_cut == "std" else f"table_{base_cut}.npz"), allow_pickle=False) as z:
        X = z["X"]
    r, y, scored = run.tab(any_arm, base_cut)
    m = scored & (r["human"] == h)
    probs, _, _ = onnx_probs(shipped / "gesture.onnx", np.ascontiguousarray(X[m].astype(np.float32)[..., 8:88]))
    pred = np.where(probs.max(axis=1) >= 0.6, probs.argmax(axis=1), 0)
    g = y[m] != 0
    base = counts(pred, y[m])
    base_keys = [f"{s}#{t}" for s, t in zip(r["session"][m][g], r["trial"][m][g])]
    base_ok = dict(zip(base_keys, pred[g] == y[m][g]))
    base_ceiling = ceiling([np.log(probs + 1e-12)], y[m])
    assert len(base_ok) == base["n"], "a prompt has two labelled bursts"
    print(f"Held-out person: {h}. Model to beat: {shipped.name} with {base_cut} cutting: {base['hits']}/{base['n']} = {base['hits'] / base['n']:.3f} recognised, "
          f"{base['false']}/{base['n_none']} = {base['false'] / base['n_none']:.3f} false triggers (budget {BUDGET}).")

    rows, unfinished, equal = [], [], []
    for k, arm in enumerate(run.arms):
        own = run.variant(arm)
        wide = (run.man.get("arm_specs", {}).get(arm, {}) or {}).get("features") == "wide"
        n_seed = sum(run.logits(f"{arm}__{h}__s{s}") is not None for s in run.seeds)
        n_inner = sum(run.logits(f"{arm}__{h}__in_{h2}__s{run.seeds[0]}") is not None for h2 in run.humans if h2 != h)
        for cut in [own] + (["rel12"] if own == "std" and "rel12" in cuts else []):      # section D: needs held-out networks only
            outer = run.outer(arm, h, cut)
            if outer:
                r, y, scored = run.tab(arm, cut)
                m = scored & (r["human"] == h)
                g = y[m] != 0
                keys = [f"{s}#{t}" for s, t in zip(r["session"][m][g], r["trial"][m][g])]
                equal.append((arm + (" (wide)" if wide else ""), own, cut, len(outer), int((~g).sum()), keys, ceiling([l[m].astype(np.float64) for l in outer], y[m])))
        if n_seed < need or n_inner < len(run.humans) - 1:
            peek = []
            r, y, scored = run.tab(arm, base_cut)
            m = scored & (r["human"] == h)
            for s in run.seeds:
                lg = run.logits(f"{arm}__{h}__s{s}", base_cut)
                if lg is not None:                                  # the natural setting needs no setting runs
                    p = apply(lg[m], 1.0, 0.0)
                    c = counts(np.where(p.max(axis=1) >= 0.6, p.argmax(axis=1), 0), y[m])
                    peek.append(f"seed {s}: {c['hits']}/{c['n']} at {c['false']}/{c['n_none']}")
            unfinished.append((arm, own, n_seed, n_inner, peek))
            continue
        for cut in [own] + (["rel12"] if own == "std" and "rel12" in cuts else []):      # the candidates of the pick rule
            outer = run.outer(arm, h, cut)
            inner = run.inner(arm, h, cut)
            if not outer and inner is None:
                continue                                           # this run holds no answers of this recipe for that cutting
            assert len(outer) == n_seed and inner is not None, f"{arm} with {cut} cutting: answers for the two cuttings do not match"
            r, y, scored = run.tab(arm, cut)
            m = scored & (r["human"] == h)
            g = y[m] != 0
            keys = [f"{s}#{t}" for s, t in zip(r["session"][m][g], r["trial"][m][g])]
            members = [l[m].astype(np.float64) for l in outer]
            row = {"arm": arm + (" (wide pictures: the engine cannot make them yet)" if wide else "")
                          + (f" [{n_seed} of {len(run.seeds)} networks]" if n_seed < len(run.seeds) else ""), "order": k, "own": own, "cut": cut}
            for name, pt in (("average", choose_point(inner[0], inner[1])), ("each", choose_point(inner[0], inner[1], person=inner[2]))):
                p = np.mean([apply(l, *pt) for l in members], axis=0)
                pred = np.where(p.max(axis=1) >= 0.6, p.argmax(axis=1), 0)
                ok = dict(zip(keys, pred[g] == y[m][g]))
                gained = sum(1 for q in ok if ok[q] and not base_ok.get(q, False))
                lost = sum(1 for q in ok if not ok[q] and base_ok.get(q, False))
                row[name] = {**counts(pred, y[m]), "gained": gained, "lost": lost, "p": sign_p(gained, lost)}
            rows.append(row)

    ranked = {c["arm"].split(" ")[0] for c in rows}
    lost = [a for a in run.arms if a not in ranked and a not in {u[0] for u in unfinished}]
    assert not lost, f"recipes neither ranked nor listed as unfinished: {lost}"

    def inside(c: dict) -> bool:
        return c["false"] / c["n_none"] <= BUDGET

    for name, title in (("average", "A. Setting chosen with the training people's AVERAGE false-trigger share inside the budget (the rule written before the results)"),
                        ("each", "B. Stricter setting: EACH training person inside the budget")):
        print(f"\n### {title}\n")
        print("| # | recipe | trained on | engine cuts | gestures recognised | non-gestures that trigger | inside the budget? | against the model to beat: gained / lost (p) |")
        print("|---|---|---|---|---|---|---|---|")
        order = sorted(rows, key=lambda c: (not inside(c[name]), -c[name]["hits"], c[name]["false"] / c[name]["n_none"], c["cut"] != c["own"], c["order"]))
        for i, c in enumerate(order, 1):
            d = c[name]
            print(f"| {i} | {c['arm']} | {c['own']} | {c['cut']} | {d['hits']}/{d['n']} = {d['hits'] / d['n']:.3f} | {d['false']}/{d['n_none']} = {d['false'] / d['n_none']:.3f} | "
                  f"{'yes' if inside(d) else 'NO'} | +{d['gained']} / -{d['lost']} (p = {d['p']:.2f}) |")
    print(f"\n### C. Recipes whose fold for {h} is not finished ({len(unfinished)})\n")
    print(f"| recipe | trained on | held-out runs done (of {len(run.seeds)}) | setting runs done (of {len(run.humans) - 1}) | the runs that exist, natural setting, {base_cut} cutting: recognised at false |")
    print("|---|---|---|---|---|")
    for arm, own, n_seed, n_inner, peek in unfinished:
        print(f"| {arm} | {own} | {n_seed} | {n_inner} | {'; '.join(peek) if peek else '-'} |")
    print(f"\n### D. At EQUAL false-trigger counts: most gestures recognised (of {base['n']}) when at most k non-gestures trigger\n")
    print("Each model's confidence setting is swept for this, looking at the held-out person for every model alike: a comparison of "
          "models, not an honest score. In brackets: gestures gained / lost against the model to beat at the same k, and the sign-test p.\n")
    print("| recipe | trained on | engine cuts | networks | non-gestures | " + " | ".join(f"k <= {k}" for k in LEVELS) + " |")
    print("|---|---|---|---|---|" + "---|" * len(LEVELS))
    print(f"| **model to beat** ({shipped.name}) | | {base_cut} | 1 | {base['n_none']} | " + " | ".join(str(base_ceiling[k][0]) if base_ceiling[k][0] >= 0 else "-" for k in LEVELS) + " |")
    tally = {k: [0, 0, 0] for k in LEVELS}                      # clearly better / within chance / clearly worse than the model to beat
    for arm, own, cut, n_net, n_none, keys, c in equal:
        cells = []
        for k in LEVELS:
            hits, ok = c[k]
            if hits < 0 or base_ceiling[k][0] < 0:
                cells.append("-")
                continue
            a, b_ = dict(zip(keys, ok)), dict(zip(base_keys, base_ceiling[k][1]))
            gained = sum(1 for q in a if a[q] and not b_.get(q, False))
            lost = sum(1 for q in a if not a[q] and b_.get(q, False))
            pv = sign_p(gained, lost)
            tally[k][1 if pv >= 0.05 else 0 if gained > lost else 2] += 1
            cells.append(f"{hits} (+{gained}/-{lost}, p={pv:.2f})")
        print(f"| {arm} | {own} | {cut} | {n_net} | {n_none} | " + " | ".join(cells) + " |")
    print(f"\nOf these {len(equal)} rows, clearly better than the model to beat (p < 0.05) / within chance / clearly worse: " +
          "; ".join(f"k <= {k}: {t[0]} / {t[1]} / {t[2]}" for k, t in tally.items()) + ".")


if __name__ == "__main__":
    main()

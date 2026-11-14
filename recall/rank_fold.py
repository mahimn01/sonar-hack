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

How to run:
    .venv/bin/python -m recall.rank_fold <run folder> [--person mahimn] [--against models-release-mac-recall]
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


def counts(pred: np.ndarray, y: np.ndarray) -> dict[str, int]:
    g = y != 0
    return {"hits": int((pred[g] == y[g]).sum()), "n": int(g.sum()), "false": int((pred[~g] != 0).sum()), "n_none": int((~g).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank a suite run's recipes on one held-out person against the shipped model.")
    ap.add_argument("run", help="run folder written by recall.suite")
    ap.add_argument("--person", default="mahimn", help="the held-out person to rank on (default: the benchmark person)")
    ap.add_argument("--against", default="models-release-mac-recall", help="folder of the model to beat (gesture.onnx + gesture.json)")
    args = ap.parse_args()
    run = Run(Path(args.run))
    h = args.person
    cuts = run.man.get("tables", ["std"])
    shipped = Path(args.against) if Path(args.against).is_absolute() else REPO / args.against
    rw = json.loads((shipped / "gesture.json").read_text()).get("run_with")
    base_cut = "rel12" if rw is None else ("std" if not rw.get("env") else "rel12" + ("nc" if "--no-click-rejection" in rw.get("flags", []) else ""))
    any_arm = run.arms[0]

    # the shipped model on this person's rows, as the engine cuts them for it; gestures keyed by prompt
    with np.load(run.dir / ("table.npz" if base_cut == "std" else f"table_{base_cut}.npz"), allow_pickle=False) as z:
        X = z["X"]
    r, y, scored = run.tab(any_arm, base_cut)
    m = scored & (r["human"] == h)
    probs, _, _ = onnx_probs(shipped / "gesture.onnx", np.ascontiguousarray(X[m].astype(np.float32)[..., 8:88]))
    pred = np.where(probs.max(axis=1) >= 0.6, probs.argmax(axis=1), 0)
    g = y[m] != 0
    base = counts(pred, y[m])
    base_ok = dict(zip((f"{s}#{t}" for s, t in zip(r["session"][m][g], r["trial"][m][g])), pred[g] == y[m][g]))
    assert len(base_ok) == base["n"], "a prompt has two labelled bursts"
    print(f"Held-out person: {h}. Model to beat: {shipped.name} with {base_cut} cutting: {base['hits']}/{base['n']} = {base['hits'] / base['n']:.3f} recognised, "
          f"{base['false']}/{base['n_none']} = {base['false'] / base['n_none']:.3f} false triggers (budget {BUDGET}).")

    rows, unfinished = [], []
    for k, arm in enumerate(run.arms):
        own = run.variant(arm)
        wide = (run.man.get("arm_specs", {}).get(arm, {}) or {}).get("features") == "wide"
        n_seed = sum(run.logits(f"{arm}__{h}__s{s}") is not None for s in run.seeds)
        n_inner = sum(run.logits(f"{arm}__{h}__in_{h2}__s{run.seeds[0]}") is not None for h2 in run.humans if h2 != h)
        if n_seed < len(run.seeds) or n_inner < len(run.humans) - 1:
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
            if run.logits(f"{arm}__{h}__s{run.seeds[0]}", cut) is None:
                continue
            r, y, scored = run.tab(arm, cut)
            m = scored & (r["human"] == h)
            g = y[m] != 0
            keys = [f"{s}#{t}" for s, t in zip(r["session"][m][g], r["trial"][m][g])]
            members = [l[m].astype(np.float64) for l in run.outer(arm, h, cut)]
            inner = run.inner(arm, h, cut)
            row = {"arm": arm + (" (wide pictures: the engine cannot make them yet)" if wide else ""), "order": k, "own": own, "cut": cut}
            for name, pt in (("average", choose_point(inner[0], inner[1])), ("each", choose_point(inner[0], inner[1], person=inner[2]))):
                p = np.mean([apply(l, *pt) for l in members], axis=0)
                pred = np.where(p.max(axis=1) >= 0.6, p.argmax(axis=1), 0)
                ok = dict(zip(keys, pred[g] == y[m][g]))
                gained = sum(1 for q in ok if ok[q] and not base_ok.get(q, False))
                lost = sum(1 for q in ok if not ok[q] and base_ok.get(q, False))
                row[name] = {**counts(pred, y[m]), "gained": gained, "lost": lost, "p": sign_p(gained, lost)}
            rows.append(row)

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


if __name__ == "__main__":
    main()

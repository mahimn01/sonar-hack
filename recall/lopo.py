"""lopo.py — the evaluation used all night: leave one HUMAN out, score on that human, never tune on them.

Plain words: there are three humans in the recordings (lance, who recorded twice as lance and lance2; aarsh;
mahimn). For each one in turn the model is trained on the other two (plus whatever synthetic data an
experiment adds) and scored on every example of the held-out human: their prompted gestures and all their
non-gesture movements (scene and breathing sessions, pulls, idle moments).

The score is the released pipeline's own: at the engine's 0.6 confidence threshold, the share of real gestures
named correctly ("recall") and the share of non-gestures that would fire a gesture ("false triggers").
Differences from the released pipeline, which looked at the test person twice: the model is the LAST one after
a fixed number of steps (not the epoch that scored best on the test person), and the confidence temperature
is 1.0 or comes from the training humans only (--nested).

How to run:   .venv/bin/python -m recall.lopo --arm real_only        (see ARMS at the bottom for the list)
What it changes: appends one line per run to train_out/recall/results.jsonl. Read-only on the data.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

import numpy as np

from .common import REPO
from .data import LABELS, Table
from .trainlib import TrainSpec, fit_temperature, logits_of, recall_at_false, score, softmax, wilson

HUMANS = ("lance", "aarsh", "mahimn")
RESULTS = REPO / "train_out" / "recall" / "results.jsonl"
BUDGET = 0.0551           # false-trigger share of the released model on mahimn (6/109 = 0.05505): the level to stay at or under


def fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.2f},{ci[1]:.2f}]"


def run_lopo(
    name: str,
    table: Table,
    fit: Callable[[Table, int], Any],
    seeds: tuple[int, ...] = (0,),
    nested: bool = False,
    thr: float = 0.6,
    verbose: bool = True,
    humans: tuple[str, ...] = HUMANS,
    predict: Callable[[Any, np.ndarray], np.ndarray] = logits_of,
    note: str = "",
) -> dict:
    """`fit(train_table, seed)` -> model. Returns the result record (also appended to results.jsonl)."""
    t_start = time.time()
    y_all = table.y()
    rec: dict[str, Any] = {"arm": name, "note": note, "seeds": list(seeds), "nested": nested, "folds": {}}
    pooled_logits, pooled_y, pooled_h, pooled_cls = [], [], [], []
    for h in humans:
        test = table.human == h
        train_t = table.take(~test)
        temps, fold_logits = [], []
        for seed in seeds:
            temp = 1.0
            if nested:        # temperature from the training humans only: hold each of them out in turn
                inner_l, inner_y = [], []
                for h2 in [x for x in humans if x != h]:
                    m2 = fit(train_t.take(train_t.human != h2), seed)
                    hold = train_t.take(train_t.human == h2)
                    inner_l.append(predict(m2, hold.X))
                    inner_y.append(hold.y())
                temp = fit_temperature(np.concatenate(inner_l), np.concatenate(inner_y))
            model = fit(train_t, seed)
            fold_logits.append(predict(model, table.X[test]))
            temps.append(temp)
        # several seeds: average the calibrated probabilities (reported per seed as well)
        probs = np.mean([softmax(l, t) for l, t in zip(fold_logits, temps)], axis=0)
        lg = np.log(probs + 1e-12)
        y = y_all[test]
        s = score(lg, y, LABELS, 1.0, thr)
        per_seed = [score(l, y, LABELS, t, thr) for l, t in zip(fold_logits, temps)]
        best_r, best_thr = recall_at_false(lg, y, BUDGET)
        rec["folds"][h] = {
            "recall": s["recall"], "false": s["false"], "n_gesture": s["n_gesture"], "n_none": s["n_none"],
            "hits": s["hits"], "false_n": s["false_n"], "per_class": s["per_class"], "acc_raw": s["acc_raw"],
            "wrong_gesture": s["wrong_gesture"], "silent": s["silent"], "temps": temps,
            "recall_seeds": [p["recall"] for p in per_seed], "false_seeds": [p["false"] for p in per_seed],
            "recall_at_budget_oracle": best_r, "thr_oracle": best_thr, "confusion": s["confusion"].tolist(),
            "confusion_raw": s["confusion_raw"].tolist(),
        }
        pooled_logits.append(lg)
        pooled_y.append(y)
        pooled_h.append(np.array([h] * len(y)))
        if verbose:
            pc = "  ".join(f"{k} {a}/{b}" for k, (a, b) in s["per_class"].items())
            print(f"  [{name}] held out {h:<7} recall {s['recall']:.3f} {fmt_ci(s['recall_ci'])} ({s['hits']}/{s['n_gesture']})  "
                  f"false {s['false']:.3f} {fmt_ci(s['false_ci'])} ({s['false_n']}/{s['n_none']})  | {pc} | "
                  f"wrong-gesture {s['wrong_gesture']} silent {s['silent']} | raw acc {s['acc_raw']:.3f} | "
                  f"best recall at false<={BUDGET}: {best_r:.3f} (thr {best_thr:.2f}, oracle) | T {np.mean(temps):.2f}", flush=True)
            if len(seeds) > 1:
                print("      per seed: recall " + " ".join(f"{p['recall']:.3f}" for p in per_seed)
                      + " | false " + " ".join(f"{p['false']:.3f}" for p in per_seed))
    lg, y = np.concatenate(pooled_logits), np.concatenate(pooled_y)
    s = score(lg, y, LABELS, 1.0, thr)
    best_r, best_thr = recall_at_false(lg, y, BUDGET)
    rec["pooled"] = {"recall": s["recall"], "recall_ci": s["recall_ci"], "false": s["false"], "false_ci": s["false_ci"],
                     "n_gesture": s["n_gesture"], "n_none": s["n_none"], "per_class": s["per_class"],
                     "recall_at_budget_oracle": best_r, "thr_oracle": best_thr, "confusion": s["confusion"].tolist()}
    # the two clean-speed humans only (lance's swipes are faster than the signal design can show; see the diagnosis)
    rec["seconds"] = round(time.time() - t_start, 1)
    if verbose:
        pc = "  ".join(f"{k} {a}/{b}" for k, (a, b) in s["per_class"].items())
        print(f"  [{name}] POOLED          recall {s['recall']:.3f} {fmt_ci(s['recall_ci'])}  false {s['false']:.3f} {fmt_ci(s['false_ci'])}  "
              f"| {pc} | best recall at false<={BUDGET}: {best_r:.3f} (oracle)  ({rec['seconds']:.0f} s)", flush=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    return rec

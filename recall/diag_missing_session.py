"""diag_missing_session.py — what can the two reported "wiring" numbers tell us about the session we do not have?

The 4090 runs printed a left/right "wiring" t-statistic computed from all recorded swipes:
  run 1 (02:26, lance + lance2 + mahimn's FIRST gesture session 001248):            t = -2.52 from 84 swipes
  run 2 (03:18, all 12 sessions in the repo + that same session 001248):            t = +3.51 from 138 swipes
Session 001248 is not in the repo. This script recomputes the same statistic (sonar.train.fit_heuristic) on
the sessions we do have, and asks: what would those two numbers be if 001248's swipes looked like mahimn's
second session (a) as labelled, (b) with left and right exchanged?
How to run:   .venv/bin/python -m recall.diag_missing_session        Read-only.
"""

from __future__ import annotations

import sys

import numpy as np

from .common import REPO
from .data import all_official

sys.path.insert(0, str(REPO))
from sonar.config import SonarConfig  # noqa: E402
from sonar.gestures import heuristic_classify  # noqa: E402

CFG = SonarConfig(n_mics=1)


def tstat(left: np.ndarray, right: np.ndarray) -> float:
    diff = left.mean() - right.mean()
    se = np.sqrt(left.var() / len(left) + right.var() / len(right)) + 1e-9
    return float(diff / se)


def main() -> None:
    t = all_official()
    sw = np.isin(t.label, ["swipe_left", "swipe_right"]) & (t.source == "gesture")
    cue = np.array([heuristic_classify(x.astype(np.float32)[:, :, 8:88], CFG).lateral_hz for x in t.X[sw]])
    lab, person = t.label[sw], t.person[sw]
    for p in ("lance", "lance2", "aarsh", "mahimn"):
        m = person == p
        l, r = cue[m & (lab == "swipe_left")], cue[m & (lab == "swipe_right")]
        print(f"{p:<7} n={m.sum():3d}  left {l.mean():+7.1f}±{l.std():5.1f}  right {r.mean():+7.1f}±{r.std():5.1f}  t = {tstat(l, r):+.2f}")
    left_all, right_all = cue[lab == "swipe_left"], cue[lab == "swipe_right"]
    print(f"all 12 sessions: n={len(cue)}  t = {tstat(left_all, right_all):+.2f}   (4090 log for these 12 on this Mac: +9.4)")
    ml, mr = cue[(person == "mahimn") & (lab == "swipe_left")], cue[(person == "mahimn") & (lab == "swipe_right")]
    rng = np.random.default_rng(0)
    for name, (xl, xr) in {"as labelled": (ml, mr), "left/right exchanged": (mr, ml), "no direction cue (shuffled)": (None, None)}.items():
        r1, r2 = [], []
        for _ in range(400):                      # 15 left + 15 right swipes drawn from mahimn's second session
            if xl is None:
                pool = rng.permutation(np.concatenate([ml, mr]))
                a, b = rng.choice(pool, 15), rng.choice(pool, 15)
            else:
                a, b = rng.choice(xl, 15), rng.choice(xr, 15)
            early = np.isin(person, ["lance", "lance2"])
            r1.append(tstat(np.concatenate([cue[early & (lab == "swipe_left")], a]), np.concatenate([cue[early & (lab == "swipe_right")], b])))
            r2.append(tstat(np.concatenate([left_all, a]), np.concatenate([right_all, b])))
        print(f"if session 001248 were mahimn-like, {name:<28}: run 1 t = {np.mean(r1):+.2f} (reported -2.52), run 2 t = {np.mean(r2):+.2f} (reported +3.51)")


if __name__ == "__main__":
    main()

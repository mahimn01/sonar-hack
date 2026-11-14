"""data.py — turn the cached session streams into labelled example tables for experiments (read-only on mac/data).

Plain words: the official builder (sonar.dataset.session_examples) gives each prompted gesture ONE picture:
the strongest burst in its prompt window, cut where the engine would cut it. This module does the same from
the cached streams and keeps, for every example, where it came from (session, human, trial number, which
burst, its frames, whether the 1.4 s limit cut it), so experiments can change one decision at a time
(which burst gets the label, where the picture is cut) and scores can be broken down by person and trial.

`official()` reproduces sonar.dataset exactly (checked by `python -m recall.data`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .common import HUMAN, Sess, load_sessions

PAD = 8
T = 80
LABELS = ("none", "swipe_left", "swipe_right", "push")          # what the released model recognises; pull -> none


def cut_feat(feat: np.ndarray, end_frame: int, not_before: int | None = None, pad: int = PAD, n: int = T) -> np.ndarray:
    """Picture [C, F, n + 2*pad] from feature columns feat [T, C, F] (row i = frame i + 1), un-padded part ending at
    frame `end_frame`; frames before `not_before` blanked (same rule as features.window_from_stream)."""
    first = end_frame - n + 1 - pad
    last = end_frame + pad
    img = np.zeros((feat.shape[1], feat.shape[2], last - first + 1), np.float32)
    lo, hi = max(first, 1), min(last, len(feat))
    if hi >= lo:
        img[:, :, lo - first:hi - first + 1] = feat[lo - 1:hi].transpose(1, 2, 0)
    if not_before is not None and not_before > first:
        img[:, :, :min(not_before - first, img.shape[-1])] = 0.0
    return img


def cut(s: Sess, end_frame: int, not_before: int | None = None, pad: int = PAD, n: int = T) -> np.ndarray:
    return cut_feat(s.feat, end_frame, not_before, pad, n)


@dataclass
class Table:
    """A set of examples with their bookkeeping columns (all numpy arrays of the same length)."""

    X: np.ndarray                 # [N, C, 64, 96] float16
    label: np.ndarray             # [N] str: none | swipe_left | swipe_right | push | pull
    session: np.ndarray           # [N] str
    person: np.ndarray            # [N] str (as recorded)
    human: np.ndarray             # [N] str (lance2 -> lance)
    source: np.ndarray            # [N] str: gesture | burst | idle | aux | synthetic
    peak_db: np.ndarray           # [N] float
    trial: np.ndarray             # [N] int: prompt number inside the session, -1 if none
    seg: np.ndarray               # [N] int: burst index inside the session, -1 for idle
    forced: np.ndarray            # [N] bool: burst was cut by the 1.4 s limit
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.label)

    def y(self, labels: tuple[str, ...] = LABELS) -> np.ndarray:
        lut = {name: i for i, name in enumerate(labels)}
        return np.array([lut.get(str(v), 0) for v in self.label], dtype=np.int64)

    def take(self, mask: np.ndarray) -> "Table":
        return Table(*(getattr(self, k)[mask] for k in ("X", "label", "session", "person", "human", "source", "peak_db",
                                                         "trial", "seg", "forced")), extra={k: v[mask] for k, v in self.extra.items()})


def concat(tables: list[Table]) -> Table:
    keys = ("X", "label", "session", "person", "human", "source", "peak_db", "trial", "seg", "forced")
    return Table(*(np.concatenate([getattr(t, k) for t in tables]) for k in keys))


def official(s: Sess, n_idle: int = 8, seed: int = 0, aux: bool = False, keep_guard: bool = False) -> Table:
    """The examples sonar.dataset.session_examples makes for one session (same rules, same random idle picks).

    aux=True also returns the other bursts of each trial (source "aux"); keep_guard=True also returns the few bursts
    the official builder drops because they sit within a second or two of a prompt window (source "guard").
    """
    rows: list[tuple] = []
    rng = np.random.default_rng(seed)
    guard = 50
    segs = s.segs
    used = np.zeros(len(segs), bool)
    windows: list[tuple[int, int]] = []

    def cut_seg(k: int) -> np.ndarray:
        nb = None if s.not_before[k] < 0 else int(s.not_before[k])
        return cut(s, int(segs[k, 2]), nb)

    n_trial = 0
    for ev in s.meta.get("events", []):
        if ev.get("kind") != "gesture":
            continue
        n_trial += 1
        f0, f1 = s.frame(ev["start"]), s.frame(ev["end"])
        windows.append((f0 - guard, f1 + 2 * guard))
        inside = [k for k in range(len(segs)) if segs[k, 1] >= f0 - 5 and segs[k, 0] <= f1 + 25]
        if not inside:
            continue
        best = max(inside, key=lambda k: segs[k, 3])
        used[inside] = True
        rows.append((cut_seg(best), ev["label"], "gesture", segs[best, 3], int(ev.get("rep", n_trial)), best, bool(segs[best, 5])))
        if aux:
            for k in inside:
                if k != best:
                    rows.append((cut_seg(k), "none", "aux", segs[k, 3], int(ev.get("rep", n_trial)), k, bool(segs[k, 5])))
    for k in range(len(segs)):
        if used[k]:
            continue
        if any(a <= segs[k, 1] and segs[k, 0] <= b for a, b in windows):
            if keep_guard:
                rows.append((cut_seg(k), "none", "guard", segs[k, 3], -1, k, bool(segs[k, 5])))
            continue
        rows.append((cut_seg(k), "none", "burst", segs[k, 3], -1, k, bool(segs[k, 5])))
    quiet = np.where(s.ready & (s.act < 1.0))[0] + 1
    quiet = quiet[(quiet > T + PAD) & (quiet < len(s.feat) - PAD)]
    quiet = np.array([q for q in quiet if not any(a <= q <= b for a, b in windows)], dtype=int)
    if len(quiet):
        for q in rng.choice(quiet, size=min(n_idle, len(quiet)), replace=False):
            rows.append((cut(s, int(q)), "none", "idle", 0.0, -1, -1, False))
    n = len(rows)
    return Table(
        X=np.stack([r[0] for r in rows]).astype(np.float16) if n else np.zeros((0, 2, 64, T + 2 * PAD), np.float16),
        label=np.array([r[1] for r in rows]), session=np.array([s.name] * n), person=np.array([s.person] * n),
        human=np.array([HUMAN.get(s.person, s.person)] * n), source=np.array([r[2] for r in rows]),
        peak_db=np.array([r[3] for r in rows], np.float32), trial=np.array([r[4] for r in rows], int),
        seg=np.array([r[5] for r in rows], int), forced=np.array([r[6] for r in rows], bool),
    )


def all_official(aux: bool = False) -> Table:
    return concat([official(s, aux=aux) for s in load_sessions()])


# ---------------------------------------------------------------------------
# Every burst of every session, with what it is (the table the 4090 suite trains and scores on)
# ---------------------------------------------------------------------------
KINDS = ("gesture", "aux", "burst", "guard", "idle")


def full(s: Sess, n_idle: int = 8, seed: int = 0) -> Table:
    """ALL movement bursts of one session plus the official idle pictures.

    source column:
      gesture  the strongest burst inside a prompt window (what the official builder labels with the prompt)
      aux      the other bursts inside (or right around) a prompt window: wind-up, hand dropping back, the tail
               of a movement the 1.4 s limit cut in two. The official builder throws these away.
      guard    bursts within a second or two of a prompt window but not inside it. Also thrown away officially;
               kept here only so that every burst has a model answer for the engine-level replay.
      burst    bursts away from any prompt (scene and breathing sessions, pauses): "none"
      idle     8 random quiet moments per session: "none"
    label is the prompt's label for gesture rows and "none" for everything else; extra["prompt"] keeps the
    prompt label for aux rows too, extra["order"] / extra["n_in_trial"] say which burst of its trial a row is.
    """
    base = official(s, n_idle=n_idle, seed=seed, aux=True, keep_guard=True)
    segs = s.segs
    prompt = np.array(["none"] * len(base), dtype=object)
    order = np.full(len(base), -1)
    n_in = np.zeros(len(base), int)
    start = np.full(len(base), -1)
    end = np.full(len(base), -1)
    decided = np.full(len(base), -1)
    trials_ev = [ev for ev in s.meta.get("events", []) if ev.get("kind") == "gesture"]
    by_trial: dict[int, list[int]] = {}
    for i in range(len(base)):
        k = int(base.seg[i])
        if k >= 0:
            start[i], end[i], decided[i] = int(segs[k, 0]), int(segs[k, 1]), int(segs[k, 2])
        if base.source[i] in ("gesture", "aux"):
            by_trial.setdefault(int(base.trial[i]), []).append(i)
    reps = {int(ev.get("rep", n + 1)): ev["label"] for n, ev in enumerate(trials_ev)}
    for rep, rows in by_trial.items():
        rows = sorted(rows, key=lambda i: start[i])
        for j, i in enumerate(rows):
            prompt[i] = reps.get(rep, "none")
            order[i], n_in[i] = j, len(rows)
    # where each picture was cut: its last frame and the frame before which it is blanked (-1 = not blanked)
    end_frame = decided.copy()
    not_before = np.full(len(base), -1)
    for i in range(len(base)):
        k = int(base.seg[i])
        if k >= 0:
            not_before[i] = int(s.not_before[k])
    idle = np.nonzero(base.seg < 0)[0]
    if len(idle):                                   # the idle pictures: the same random quiet frames official() drew
        rng = np.random.default_rng(seed)
        guard = 50
        windows = [(s.frame(ev["start"]) - guard, s.frame(ev["end"]) + 2 * guard) for ev in trials_ev]
        quiet = np.where(s.ready & (s.act < 1.0))[0] + 1
        quiet = quiet[(quiet > T + PAD) & (quiet < len(s.feat) - PAD)]
        quiet = np.array([q for q in quiet if not any(a <= q <= b for a, b in windows)], dtype=int)
        picks = rng.choice(quiet, size=min(n_idle, len(quiet)), replace=False)
        assert len(picks) == len(idle), (s.name, len(picks), len(idle))
        end_frame[idle] = picks
    base.extra = {"prompt": prompt.astype(str), "order": order, "n_in_trial": n_in, "start": start, "end": end, "decided": decided,
                  "end_frame": end_frame, "not_before": not_before}
    return base


def full_table(sessions: list[Sess] | None = None) -> Table:
    tabs = [full(s) for s in (sessions if sessions is not None else load_sessions())]
    out = concat(tabs)
    out.extra = {k: np.concatenate([t.extra[k] for t in tabs]) for k in tabs[0].extra}
    return out


def _check() -> None:
    """Compare with the official builder on every session (pictures, labels, order)."""
    import sys

    from .common import REPO

    sys.path.insert(0, str(REPO))
    from sonar.dataset import session_examples

    worst = 0.0
    for s in load_sessions():
        ex, rep = session_examples(REPO / "mac" / "data" / s.name)
        mine = official(s)
        assert len(ex) == len(mine), (s.name, len(ex), len(mine))
        assert [e.label for e in ex] == list(mine.label), s.name
        assert [e.source for e in ex] == list(mine.source), s.name
        d = float(np.max(np.abs(np.stack([e.x for e in ex]).astype(np.float16).astype(np.float32) - mine.X.astype(np.float32))))
        worst = max(worst, d)
        print(f"{s.name}: {len(ex)} examples identical in label/source/order, max picture difference {d:.2e}  {rep['gestures']} gestures, {rep['missed']} missed")
    print("worst picture difference:", worst)


if __name__ == "__main__":
    _check()

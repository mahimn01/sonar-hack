"""relcut.py — the same recordings cut with a PEAK-RELATIVE burst end (an experiment; sonar/ is not changed).

Why: on this MacBook a hand echo is 40-50 dB above the noise floor, so the movement never falls back to "quiet"
between raising the hand, the stroke and dropping the hand. The engine's burst then runs into its 1.4 s limit and
is chopped wherever that falls: 111 of 218 recorded gestures (aarsh 36/45, mahimn 27/48 + 19/45). The model's
picture is cut at that arbitrary point.

The rule tried here (the data-engine session's `seg_rel_off_db`, one changed line): a burst also ends once the
movement has been more than N dB below the burst's OWN peak for 0.2 s. With N = 12 only 10 of 218 gestures hit
the limit, the labelled burst is 0.3-0.8 s long and ends right after the stroke; the hand dropping back becomes a
burst of its own.

How: the engine's own Pipeline with its segmenter replaced by `RelSegmenter` (the engine's segmenter with that one
line changed). Cached per session in the same format as recall.cache_streams.

How to run:   .venv/bin/python -m recall.relcut [folder] [rel12 | rel12nc]     (default: every session under mac/data, rel12)
What it changes: writes train_out/recall/streams_rel12/<session>.npz (+ .json). Read-only on the recordings.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sonar.gestures import GestureSegmenter, Segment  # noqa: E402


NO_CLICKS = 1e9      # impulse_ratio out of reach: the segmenter's click test can then never be true


def parse(variant: str) -> tuple[float, bool]:
    """"rel12" -> (12.0, click rejection on); "rel12nc" -> (12.0, click rejection off)."""
    clicks = not variant.endswith("nc")
    return float(variant[3:-2] if not clicks else variant[3:]), clicks


def out_dir(rel_db: float, clicks: bool = True) -> Path:
    return REPO / "train_out" / "recall" / f"streams_rel{int(rel_db)}{'' if clicks else 'nc'}"


class RelSegmenter(GestureSegmenter):
    """sonar.gestures.GestureSegmenter with one change: "still moving" also needs to be within `rel_db` of the
    burst's own peak. Everything else (thresholds, click rejection, 0.2 s hang, 1.4 s limit) is the engine's."""

    def __init__(self, cfg, rel_db: float) -> None:
        super().__init__(cfg)
        self.rel_db = float(rel_db)

    def push(self, index: int, feat: np.ndarray, excess_db: np.ndarray) -> Segment | None:
        cfg = self.cfg
        self._hist.append(feat)
        self._hist_index.append(index)
        act, impulse = self.frame_activity(excess_db)
        self.activity_db, self.impulse = act, impulse
        if impulse:
            self._impulse_hold = 10
        held = self._impulse_hold > 0
        if held:
            self._impulse_hold -= 1
            act = -10.0

        if not self._active:
            self._long_active = 0
            if act < self.off_thr and not held:
                self._update_idle_stats(act)
            self._onset = self._onset + 1 if act >= self.on_thr else 0
            if self._onset >= 2:
                self._active = True
                self._start = index - 1
                self._last_loud = index
                self._peak = act
                self._energy = act
                self._onset = 0
            return None

        self._long_active += 1
        self._peak = max(self._peak, act)
        self._energy += max(act, 0.0)
        off = max(self.off_thr, self._peak - self.rel_db)          # <- the one changed line
        if act >= off:
            self._last_loud = index
        quiet = index - self._last_loud
        too_long = index - self._start + 1 >= cfg.seg_max_frames
        if quiet < cfg.seg_hang_frames and not too_long:
            return None
        self._active = False
        seg = Segment(self._start, self._last_loud, index, self._peak, self._energy, forced=too_long)
        if seg.n_frames < cfg.seg_min_frames or seg.peak_db < cfg.seg_min_peak_db:
            return None
        return seg


def one(folder: str, rel_db: float = 12.0, clicks: bool = True) -> str:
    """One session through the engine's Pipeline with the peak-relative segmenter; cached like cache_streams.

    clicks=False also switches the segmenter's click rejection off. That test (energy flat across Doppler and
    equal on both signs) fires on very fast hands too: with it on, the loudest moment of the trial is NOT inside
    the labelled burst for 10/35 and 17/45 of lance's gestures and 8/45 of mahimn's fast session; with it off,
    for 0, 1 and 1, and the stroke's peak sits 11-13 frames before the end of the picture for everybody.
    """
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    from sonar.dataset import load_session
    from sonar.pipeline import Pipeline

    meta, audio, cfg = load_session(folder)
    name = meta.get("session_id", Path(folder).name)
    if not clicks:
        cfg = cfg.with_(impulse_ratio=NO_CLICKS)
    pipe = Pipeline(cfg, n_mics=audio.shape[1])
    pipe.segmenter = RelSegmenter(pipe.cfg, rel_db)
    cfg = pipe.cfg
    zero = np.zeros((cfg.n_feat_channels, cfg.doppler_bins), np.float32)
    feats, ready, act, segs, nb = [], [], [], [], []
    prev_end = -1
    for i in range(0, len(audio), 9600):
        for fr in pipe.process(audio[i:i + 9600]):
            feats.append(zero if fr.feat is None else fr.feat)
            ready.append(fr.feat is not None)
            act.append(fr.activity_db)
            if fr.segment is not None:
                s = fr.segment
                segs.append([s.start, s.end, s.decided, s.peak_db, s.energy_db, int(s.forced)])
                nb.append(prev_end)
                prev_end = s.end + 1
    out = out_dir(rel_db, clicks)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", feat=np.stack(feats).astype(np.float16), ready=np.array(ready),
                        activity_db=np.array(act, np.float32), segments=np.array(segs, np.float64).reshape(-1, 6),
                        not_before=np.array(nb, np.int64), presence=np.array([""]))
    (out / f"{name}.json").write_text(json.dumps({"meta": meta, "config": cfg.to_dict(), "seg_rel_off_db": rel_db, "click_rejection": clicks}, indent=1))
    forced = int(sum(s[5] for s in segs))
    return f"{name}: peak-relative cutting {rel_db:.0f} dB{'' if clicks else ', click rejection off'}: {len(segs)} bursts, {forced} cut by the 1.4 s limit"


def main_for(root: Path, rel_db: float = 12.0, clicks: bool = True, workers: int = 4) -> None:
    out = out_dir(rel_db, clicks)
    out.mkdir(parents=True, exist_ok=True)
    folders = sorted(str(p.parent) for p in Path(root).rglob("session.json"))
    todo = [f for f in folders if not (out / f"{Path(f).name}.npz").exists()]
    print(f"peak-relative cutting ({rel_db:.0f} dB{'' if clicks else ', click rejection off'}): {len(folders)} sessions under {root}, "
          f"{len(todo)} to cut", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as pool:
            for line in pool.map(partial(one, rel_db=rel_db, clicks=clicks), todo):
                print(line, flush=True)


if __name__ == "__main__":
    # python -m recall.relcut [folder with sessions] [variant: rel12 | rel12nc]
    _db, _clicks = parse(sys.argv[2]) if len(sys.argv) > 2 else (12.0, True)
    main_for(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "mac" / "data", _db, _clicks)

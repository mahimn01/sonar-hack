"""cache_streams.py — run every recorded session through the engine's own pipeline once and keep the result.

What it does: for each session folder under mac/data it runs the same CPU code the live engine
uses (sonar.features.stream_features) and saves, per session, the feature columns for every
20 ms frame, the movement-energy trace, every movement burst the segmenter cut, and the prompts.
Why: every later diagnosis (did a burst get cut for each prompt? when? how strong?) and every
training experiment reads these files instead of re-running 36 minutes of audio.

How to run:   cd ~/Documents/Dev/sonar-hack-recall && .venv/bin/python -m recall.cache_streams
What it changes: writes train_out/recall/streams/<session>.npz (+ .json). Read-only on mac/data.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "train_out" / "recall" / "streams"


def one(folder: str) -> str:
    """One session -> its cached stream file. Returns a one-line report."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    sys.path.insert(0, str(REPO))
    from sonar.dataset import load_session
    from sonar.features import stream_features

    t0 = time.time()
    meta, audio, cfg = load_session(folder)
    stream = stream_features(audio, cfg)
    name = meta.get("session_id", Path(folder).name)
    segs = np.array([[s.start, s.end, s.decided, s.peak_db, s.energy_db, int(s.forced)] for s in stream.segments], dtype=np.float64)
    nb = np.array([-1 if v is None else v for v in stream.not_before], dtype=np.int64)
    np.savez_compressed(
        OUT / f"{name}.npz",
        feat=stream.feat.astype(np.float16), ready=stream.ready, activity_db=stream.activity_db.astype(np.float32),
        segments=segs.reshape(-1, 6), not_before=nb, presence=np.array(stream.presence),
    )
    (OUT / f"{name}.json").write_text(json.dumps({"meta": meta, "config": cfg.to_dict()}, indent=1))
    return f"{name}: {len(audio) / cfg.fs:.0f} s audio, {stream.n_frames} frames, {len(stream.segments)} bursts, {time.time() - t0:.0f} s"


def main_for(root: Path, workers: int = 4) -> int:
    """Cache every session folder under `root` that is not cached yet. Returns how many sessions exist."""
    OUT.mkdir(parents=True, exist_ok=True)
    folders = sorted(str(p.parent) for p in Path(root).rglob("session.json"))
    todo = [f for f in folders if not (OUT / f"{Path(f).name}.npz").exists()]
    print(f"{len(folders)} sessions under {root}, {len(todo)} to cut", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as pool:
            for line in pool.map(one, todo):
                print(line, flush=True)
    return len(folders)


def main() -> int:
    main_for(REPO / "mac" / "data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

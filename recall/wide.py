"""wide.py — the same recordings seen through a WIDER Doppler window (an experiment; sonar/ is not changed).

Why: the engine keeps +-250 Hz of Doppler around each pilot tone. Lance's swipes and Mahimn's fast pushes shift
the tones by up to +-700 Hz. The part beyond +-250 Hz is either lost or lands in the other tone's picture, where
it looks like movement in the opposite direction (a fast push towards the screen shows up as "moving away" on
the right-speaker tone). A +-750 Hz window shows the whole trace.

How: the engine's own receive chain (sonar.frontend.Frontend, the noise-floor normaliser and the segmenter,
exactly as sonar.features.stream_from_columns runs them) with five receive settings changed:
    baseband rate 2 kHz -> 4 kHz (decim 12), window 64 ms as before (256 samples), FFT 512 (7.8 Hz bins as
    before), 192 bins kept (+-750 Hz), tone band-pass +-800 Hz.
The transmitted signal and the recordings are untouched. The config check that forbids overlapping tone windows
is switched off only inside `no_validate()`.

How to run:   .venv/bin/python -m recall.wide            (caches every session under mac/data)
What it changes: writes train_out/recall/streams_wide/<session>.npz. Read-only on the recordings.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "train_out" / "recall" / "streams_wide"
WIDE = {"decim": 12, "stft_win": 256, "stft_fft": 512, "doppler_bins": 192, "demod_pass_hz": 800.0}


@contextlib.contextmanager
def no_validate() -> Iterator[None]:
    """Inside this block SonarConfig.validate() accepts overlapping tone windows (nothing else changes)."""
    sys.path.insert(0, str(REPO))
    from sonar.config import SonarConfig

    real = SonarConfig.validate
    SonarConfig.validate = lambda self: None
    try:
        yield
    finally:
        SonarConfig.validate = real


def wide_config(cfg: "object") -> "object":
    return dataclasses.replace(cfg, **WIDE)


def one(folder: str) -> str:
    """One session -> its wide feature columns [T, 2, 192] (float16), cached."""
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    sys.path.insert(0, str(REPO))
    from sonar.dataset import load_session
    from sonar.features import stream_from_columns
    from sonar.frontend import Frontend

    meta, audio, cfg = load_session(folder)
    name = meta.get("session_id", Path(folder).name)
    with no_validate():
        wcfg = wide_config(cfg)
        fe = Frontend(wcfg, n_mics=audio.shape[1])
        cols = []
        for i in range(0, len(audio), 9600):
            cols += [fr.doppler_db for fr in fe.process(audio[i:i + 9600])]
        stream = stream_from_columns(wcfg, np.stack(cols), None)       # the engine's normaliser + segmenter, frame by frame
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", feat=stream.feat.astype(np.float16), n_segments=np.array(len(stream.segments)))
    return f"{name}: wide features {stream.feat.shape}, {len(stream.segments)} bursts with the wide window"


def main_for(root: Path, workers: int = 4) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    folders = sorted(str(p.parent) for p in Path(root).rglob("session.json"))
    todo = [f for f in folders if not (OUT / f"{Path(f).name}.npz").exists()]
    print(f"wide window: {len(folders)} sessions under {root}, {len(todo)} to cut", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as pool:
            for line in pool.map(one, todo):
                print(line, flush=True)


def load(name: str) -> np.ndarray:
    with np.load(OUT / f"{name}.npz") as z:
        return z["feat"].astype(np.float32)


if __name__ == "__main__":
    main_for(REPO / "mac" / "data")

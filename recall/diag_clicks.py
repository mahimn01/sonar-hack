"""diag_clicks.py — does the engine's CLICK rejection swallow fast gestures on the real recordings?

The segmenter treats a frame as a click (keyboard, clap) when the movement energy is spread flat across Doppler
and equal on the "approaching" and "leaving" sides; such a frame and the 0.2 s after it can neither open nor
sustain a burst. A very fast hand also lights up both sides and the outer rows (its trace leaves the +-250 Hz
picture and the other tone's trace comes in), so it may be taken for a click.
For every prompted gesture this prints, per session: how many have click-flagged frames at the loudest moment of
the trial, how many frames, and whether the LOUDEST frame of the trial itself is flagged (then the stroke's peak
cannot be inside an open burst).
How to run:   .venv/bin/python -m recall.diag_clicks        Read-only (runs the engine's Pipeline on each gesture session).
"""

from __future__ import annotations

import sys

import numpy as np

from .common import REPO, fixed_meta

sys.path.insert(0, str(REPO))
from sonar.dataset import load_session  # noqa: E402
from sonar.pipeline import Pipeline  # noqa: E402


def main() -> None:
    folders = sorted(p.parent for p in (REPO / "mac" / "data").rglob("session.json") if "gestures" in p.parent.name)
    extra = REPO / "train_out" / "recall" / "excluded" / "20261114-001248_mahimn_gestures"
    if extra.exists():
        folders.append(extra)
    print(f"{'session':<34} {'class':<12} {'n':>3} {'click frames near the peak':>27} {'loudest frame is a click':>25} {'click frames per trial (median)':>32}")
    for folder in sorted(folders, key=lambda p: p.name):
        meta, audio, cfg = load_session(folder)
        meta = fixed_meta(meta, folder.name)
        pipe = Pipeline(cfg, n_mics=audio.shape[1])
        act, imp = [], []
        for i in range(0, len(audio), 9600):
            for fr in pipe.process(audio[i:i + 9600]):
                act.append(fr.activity_db)
                imp.append(fr.impulse)
        act, imp = np.array(act), np.array(imp)
        for g in ("swipe_left", "swipe_right", "push"):
            near = peak_is = 0
            counts = []
            n = 0
            for ev in meta["events"]:
                if ev.get("kind") != "gesture" or ev["label"] != g:
                    continue
                n += 1
                a, b = ev["start"] // 960 - 10, ev["end"] // 960 + 50
                k = a + int(np.argmax(act[a:b]))
                w = imp[max(k - 12, 0):k + 13]
                near += int(w.any())
                peak_is += int(imp[k])
                counts.append(int(w.sum()))
            if n:
                print(f"{folder.name:<34} {g:<12} {n:3d} {near:19d}/{n:<7d} {peak_is:17d}/{n:<7d} {np.median(counts):32.0f}")


if __name__ == "__main__":
    main()

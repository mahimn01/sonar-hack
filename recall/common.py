"""common.py — shared loaders for the recall experiments (read-only helpers).

Loads the cached per-session streams written by recall.cache_streams and gives each prompt
("trial") its list of movement bursts, using the same matching rule as sonar.dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
STREAMS = REPO / "train_out" / "recall" / "streams"
GESTURES = ("swipe_left", "swipe_right", "push", "pull")
# lance and lance2 are the same human on the same MacBook, 24 minutes apart (session.json: same device,
# lance2 notes "25cm"). For "unseen person" scores they must be held out together. lance3 (04:50, pushed 11:42 as
# a held-out session) is the same human again: a fresh session of a KNOWN person, never an unseen-person test.
HUMAN = {"lance": "lance", "lance2": "lance", "lance3": "lance", "aarsh": "aarsh", "mahimn": "mahimn"}
# Sessions in which the person swept the OPPOSITE way to the prompts (measured on the audio: left-speaker tone minus
# right-speaker tone is -63 Hz on "swipe_left" prompts and +91 Hz on "swipe_right" prompts; label-audit t = -7.1).
FLIPPED = ("20261114-001248_mahimn_gestures",)
_SWAP = {"swipe_left": "swipe_right", "swipe_right": "swipe_left"}


def fixed_meta(meta: dict, name: str, flip: tuple[str, ...] | list[str] = FLIPPED) -> dict:
    """session.json contents with the swipe prompts exchanged if the session is one of the flipped ones."""
    if name not in flip:
        return meta
    out = dict(meta)
    out["events"] = [{**e, "label": _SWAP.get(e.get("label"), e.get("label"))} for e in meta.get("events", [])]
    return out


@dataclass
class Sess:
    name: str
    person: str
    script: str
    meta: dict
    feat: np.ndarray          # [T, C, F] float32, row i = frame i + 1
    ready: np.ndarray
    act: np.ndarray           # [T] activity dB
    segs: np.ndarray          # [n, 6] start, end, decided, peak_db, energy_db, forced
    not_before: np.ndarray    # [n] (-1 = None)

    @property
    def human(self) -> str:
        return HUMAN.get(self.person, self.person)

    def frame(self, sample: int) -> int:
        return int(sample // 960) + 1


def load_sessions(script: str | None = None, root: Path | None = None) -> list[Sess]:
    """The cached streams (recall.cache_streams), or another cutting of the same sessions when `root` is given."""
    out = []
    for f in sorted((root or STREAMS).glob("*.npz")):
        meta = json.loads(f.with_suffix(".json").read_text())["meta"]
        if script and meta["script"] != script:
            continue
        z = np.load(f)
        out.append(Sess(f.stem, meta["person"], meta["script"], meta, z["feat"].astype(np.float32), z["ready"],
                        z["activity_db"], z["segments"], z["not_before"]))
    return out


def trials(s: Sess) -> list[dict]:
    """Every prompted gesture of a session with the bursts the dataset builder would consider for it."""
    out = []
    for ev in s.meta["events"]:
        if ev["kind"] != "gesture":
            continue
        f0, f1 = s.frame(ev["start"]), s.frame(ev["end"])
        inside = [k for k in range(len(s.segs)) if s.segs[k, 1] >= f0 - 5 and s.segs[k, 0] <= f1 + 25]
        out.append({"label": ev["label"], "rep": ev.get("rep"), "f0": f0, "f1": f1, "inside": inside})
    return out

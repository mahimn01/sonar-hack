"""dataview.py — a corrected VIEW of recorded sessions, without touching the recordings.

Plain words: sometimes a recording is fine but its labels are not (a whole session whose
swipes went the opposite way to the prompts), or a folder holds a session twice, or holds one
that must stay out. Editing or moving recordings is risky. This makes a small second folder
instead: one sub-folder per session with only a `session.json` in it, whose `audio` entry
points at the original WAV by its full path. Every tool that reads sessions (the dataset
builder, train_4090.py, label QA, the engine replay, the calibration) reads such a view exactly
like the real thing. Nothing is copied except the small label files.

What it can do while making the view:
    --flip-swipes ID[,ID]   exchange "swipe_left" and "swipe_right" in those sessions' labels
    --drop ID[,ID]          leave those sessions out
    duplicates              a session found twice (same session_id) is taken once; the copy
                            outside a folder called "excluded" wins
The default for --flip-swipes is the one session known to be recorded that way:
20261114-001248_mahimn_gestures (27 of its 30 swipes go the opposite way to their prompts,
measured on the audio by the recall session and again by sonar.labelqa).

How to run:
    python -m sonar.dataview mac --out mac_fixed                       (then use --data mac_fixed everywhere)
    python -m sonar.dataview mac --out mac_fixed --flip-swipes ""      (a plain de-duplicated view, labels untouched)
What it changes: writes the --out folder (label files only). The recordings are only read.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from .dataset import AUDIO_FILE, SESSION_FILE

KNOWN_FLIPPED = ("20261114-001248_mahimn_gestures",)
SWAP = {"swipe_left": "swipe_right", "swipe_right": "swipe_left"}


def exchange_swipes(meta: dict[str, Any]) -> int:
    """Exchange swipe_left and swipe_right in a session's labels, in place. Returns how many labels changed."""
    n = 0
    for ev in meta.get("events", []):
        for key in ("label", "intent", "candidate"):
            if ev.get(key) in SWAP:
                ev[key] = SWAP[ev[key]]
                n += key == "label"
    for row in meta.get("prompts", []):
        if row.get("intent") in SWAP:
            row["intent"] = SWAP[row["intent"]]
    return n


def make_view(roots: list[str | Path], out: str | Path, flip_swipes: tuple[str, ...] = KNOWN_FLIPPED, drop: tuple[str, ...] = (),
              verbose: bool = True) -> list[dict[str, Any]]:
    """Write the view folder. Returns one row per session found: what it is and what was done with it."""
    out = Path(out)
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(Path(root).rglob(SESSION_FILE)):
            folder = path.parent
            if out.resolve() in folder.resolve().parents or folder.resolve() == out.resolve():
                continue                                        # never read our own output
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                rows.append({"session": folder.name, "folder": str(folder), "action": f"skipped: cannot read it ({type(exc).__name__})"})
                continue
            sid = str(meta.get("session_id", folder.name))
            if sid in found:
                old = found[sid][0]
                # the same session twice: the copy that is NOT under a folder called "excluded" wins
                if "excluded" in old.parts and "excluded" not in folder.parts:
                    rows.append({"session": sid, "folder": str(old), "action": "skipped: a second copy of the same session"})
                    found[sid] = (folder, meta)
                else:
                    rows.append({"session": sid, "folder": str(folder), "action": "skipped: a second copy of the same session"})
                continue
            found[sid] = (folder, meta)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for sid, (folder, meta) in sorted(found.items()):
        row: dict[str, Any] = {"session": sid, "folder": str(folder), "person": str(meta.get("person", "unknown")), "script": meta.get("script")}
        if sid in drop:
            rows.append({**row, "action": "left out (--drop)"})
            continue
        audio = (folder / meta.get("audio", AUDIO_FILE)).resolve()
        if not audio.exists():
            rows.append({**row, "action": f"skipped: no audio at {audio}"})
            continue
        view = dict(meta)
        view["audio"] = str(audio)                             # a full path: the view holds no audio of its own
        view["view_of"] = str(folder.resolve())
        action = "as recorded"
        if sid in flip_swipes:
            n = exchange_swipes(view)
            view["label_fix"] = "swipe_left and swipe_right exchanged: the swipes of this session go the opposite way to their prompts"
            action = f"LEFT/RIGHT EXCHANGED in {n} swipe labels"
        target = out / sid
        target.mkdir()
        (target / SESSION_FILE).write_text(json.dumps(view, indent=1), encoding="utf-8")
        if (folder / "landmarks.jsonl").exists():
            shutil.copy2(folder / "landmarks.jsonl", target / "landmarks.jsonl")
        rows.append({**row, "action": action})
    if verbose:
        for row in sorted(rows, key=lambda r: r["session"]):
            print(f"  {row['session']:<40} {row.get('person', ''):<10} {row['action']}")
        used = sum(1 for r in rows if r["action"] == "as recorded" or r["action"].startswith("LEFT/RIGHT"))
        print(f"view of {used} session(s) -> {out}   (labels only; the audio stays where it is)")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sonar.dataview", description="A corrected view of recorded sessions (labels only, audio by reference)")
    ap.add_argument("roots", nargs="+", help="folders with recorded sessions")
    ap.add_argument("--out", required=True, help="folder for the view (it is emptied first)")
    ap.add_argument("--flip-swipes", default=",".join(KNOWN_FLIPPED),
                    help="sessions whose swipe_left / swipe_right labels are exchanged (default: the one known session; \"\" for none)")
    ap.add_argument("--drop", default="", help="sessions to leave out")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    rows = make_view(args.roots, args.out, tuple(s for s in args.flip_swipes.split(",") if s), tuple(s for s in args.drop.split(",") if s))
    return 0 if any(r["action"] == "as recorded" or r["action"].startswith("LEFT/RIGHT") for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())

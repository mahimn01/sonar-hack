#!/usr/bin/env python3
"""check_gpu_parity.py — prove that the GPU clip maker gives the same training pictures as the CPU one.

What it does
    Makes a fixed, seeded set of 1,000 synthetic clips twice: with the CPU reference
    (datagen/generate.py, numpy) and with the GPU path (datagen/gpu_pipeline.py, PyTorch).
    It then compares them clip by clip: the audio before rounding, the final audio, the
    feature stream, where the movement bursts are, the labels, and the pictures the
    network sees. The fixture covers every gesture, every kind of non-gesture, clips laid
    over a recording, one-microphone clips, swapped channels and hard cases (fastest
    strokes, clipping, large clock mismatch ...). If --data has recorded sessions, their
    audio goes through both feature codes as well.
    It also compares today's CPU results with tests/fixtures/gpu_parity_reference.npz,
    so a later change to either path is noticed.

Why
    The model must be trained on exactly what the live engine computes. A second
    implementation that drifts would change the model's input without anyone noticing.
    scripts/train_4090.py runs this check before it makes data on a GPU and refuses to
    train if it fails.

How to run (from the repo root)
    python scripts/check_gpu_parity.py                         # the full check on the GPU (about 1-2 min on the 4090 box)
    python scripts/check_gpu_parity.py --clips 60 --device torch-cpu    # a quick look without a GPU
    python scripts/check_gpu_parity.py --device torch-cpu --write-reference   # after an INTENDED change of the CPU path

What it changes
    Read-only, except: a stand-in background recording and (with --json) a report under
    --work (default train_out/parity/), and tests/fixtures/gpu_parity_reference.npz with
    --write-reference. Exit code 0 = same pictures, 1 = different, 2 = could not run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def pick_device(name: str) -> str:
    """auto = the best PyTorch device here; torch-cpu = the GPU code on the CPU."""
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"                       # Apple MPS is opt-in (--device mps): it has no float64, so it is less exact
    return "cpu" if name == "torch-cpu" else name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare the GPU clip maker with the CPU reference (see the file header)")
    ap.add_argument("--device", default="auto", help="auto (cuda if there is one, else the GPU code on the CPU) | cuda | mps | torch-cpu")
    ap.add_argument("--precision", default="auto", choices=["auto", "fast", "ref", "single"], help="GPU number format to check (default: the one training uses)")
    ap.add_argument("--clips", type=int, default=1000, help="fixture size (default 1000; smaller = an even sample of it)")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 61), help="CPU processes (default: every core)")
    ap.add_argument("--batch", type=int, default=0, help="clips per GPU batch (default: fills the GPU)")
    ap.add_argument("--profile", default=None, help="signal profile (default: the core default)")
    ap.add_argument("--data", default="data", help="folder with recorded sessions; their audio is checked too if it exists")
    ap.add_argument("--work", default="train_out/parity", help="folder for the stand-in background and reports")
    ap.add_argument("--json", help="also write the full report to this file")
    ap.add_argument("--motion", default="auto", help="motion library whose clips are checked too (auto: datagen/motion_library/unica_v1.npz "
                                                     "if it exists; none: skip)")
    ap.add_argument("--calibration", default="auto", help="fitted laptop whose clips are checked too (auto: the newest "
                                                          "calibration/macbook_v<N>.json if there is one; none: skip)")
    ap.add_argument("--noise", default="cpu", choices=["cpu", "gpu"], help="hall noise and gain wander: cpu = shaped in the worker processes, as render.py does (default); gpu = the workers only draw the random numbers and the GPU shapes them (about half the CPU work per clip; same audio to 1e-12)")
    ap.add_argument("--kernel", default="auto", choices=["auto", "on", "off", "host"],
                    help="echo loop: auto = the fused CUDA kernel if its self-check passes (as training will use it), "
                         "off = PyTorch operations, host = the kernel's C code on the CPU (tests)")
    ap.add_argument("--write-reference", action="store_true", help="store today's CPU results as the reference file (full fixture only)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    try:
        import torch  # noqa: F401
        from datagen import parity
    except Exception as exc:                 # no PyTorch, broken install
        print(f"could not start the parity check: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    device = pick_device(args.device)
    default_motion = REPO / "datagen" / "motion_library" / "unica_v1.npz"
    motion = None if args.motion == "none" else (str(default_motion) if args.motion == "auto" and default_motion.exists()
                                                 else (None if args.motion == "auto" else args.motion))
    from datagen.calibration import newest

    default_cal = newest(REPO / "calibration")
    cal = None if args.calibration == "none" else (str(default_cal) if args.calibration == "auto" and default_cal is not None
                                                   else (None if args.calibration == "auto" else args.calibration))
    data = (REPO / args.data)
    try:
        report = parity.run(args.clips, device, precision=args.precision, workers=args.workers, batch=args.batch,
                            profile=args.profile, work=(REPO / args.work), data=data if data.exists() else None,
                            kernel=args.kernel, motion=motion, calibration=cal, noise=args.noise)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"\nPARITY CHECK COULD NOT RUN: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    parity.print_report(report)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(parity.to_json(report))
    if args.write_reference:
        if args.clips < 1000 or args.profile:
            print("--write-reference needs the full fixture of the default profile", file=sys.stderr)
            return 2
        if not report["passed"] and any("reference" not in f for f in report["failures"]):
            print("not writing the reference: the two paths disagree (fix that first)", file=sys.stderr)
            return 1
        parity.write_reference(report["packed"])
        print(f"wrote {parity.REFERENCE_FILE.relative_to(REPO)} (sha256 {str(report['packed']['sha256'])[:16]}); commit it")
        return 0
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

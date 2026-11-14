#!/usr/bin/env python3
"""bench_gpu.py — how fast is each step of the data pipeline on this machine, and is the GPU kept busy?

What it does
    Times, one after the other and each in its own process (so nothing shares the GPU or
    blurs the memory figures):
      1. the CPU reference clip maker: milliseconds per clip per core for each stage, and
         clips per second with every core working;
      2. the GPU clip maker, for each number format (ref, fast): clips per second for the
         random draws (CPU), each rendering stage (GPU), the Doppler columns (GPU) and the
         labelling (CPU); then the whole thing end to end with the GPU's load and memory
         sampled from nvidia-smi;
      3. the training step (augmentation + model) for widths 16 / 24 / 32.
    It prints one table and saves the numbers as JSON to send back.

Why
    To see where the time goes on the 4090, to choose --gen-precision, and to check that
    the batch size uses the card's memory without overflowing it.

How to run (from the repo root; close other GPU programs first)
    python scripts/bench_gpu.py                                # on the 4090 box, about 3-5 minutes
    python scripts/bench_gpu.py --device torch-cpu --clips 24  # the GPU code on a CPU (slow; for comparison)
    python scripts/bench_gpu.py --only cpu                     # just the CPU reference

What it changes
    Writes train_out/bench/bench_<time>.json. Nothing else: no data, no models, no audio devices.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
SEED0 = 7_000_000


def bench_jobs(n: int, profile: str | None) -> list[tuple]:
    """The mix datagen.generate makes: 4 gestures in equal parts, half as many negative-only clips again."""
    import numpy as np

    from datagen import generate as g

    per = max(1, n // 6)
    kinds = [k for k in g.GESTURE4 for _ in range(per)]
    names, p = list(g.NEGATIVE_MIX), np.array(list(g.NEGATIVE_MIX.values()))
    kinds += list(np.random.default_rng(0).choice(names, size=max(1, n - len(kinds)), p=p / p.sum()))
    return [(SEED0 + i, str(k), profile, 6.0, False, (), 0.0, False, False, None) for i, k in enumerate(kinds[:n])]


# --------------------------------------------------------------------------
# The timed runs (each executed in its own process, see main)
# --------------------------------------------------------------------------
def run_cpu(args: argparse.Namespace) -> dict:
    """CPU reference: stage costs on one core, then throughput on every core."""
    import collections
    from concurrent.futures import ProcessPoolExecutor

    from datagen import adapter as ad
    from datagen import generate as g
    from datagen import render as rmod
    from sonar import dsp, gestures, presence

    spent: dict = collections.defaultdict(float)

    def wrap(obj: object, name: str, key: str) -> None:
        fn = getattr(obj, name)

        def timed(*a, **k):
            t = time.process_time()
            try:
                return fn(*a, **k)
            finally:
                spent[key] += time.process_time() - t

        setattr(obj, name, timed)

    wrap(g, "sample_scene", "scene (random story)")
    wrap(g, "sample_hardware", "virtual laptop")
    wrap(g, "render", "render: total")
    wrap(rmod, "_to_grid", "render: trajectories to the baseband grid")
    wrap(rmod, "_noise", "render: hall noise")
    wrap(rmod, "_gain_track", "render: gain wander")
    wrap(rmod, "_clock_warp", "render: clock mismatch")
    wrap(ad, "stream_features", "features: total")
    for cls, name, key in ((dsp.Prefilter, "process", "features: high-pass"), (dsp.ToneDemod, "process", "features: tone demodulation"),
                           (dsp.DopplerSTFT, "process", "features: Doppler spectra"),
                           (dsp.ChirpReceiver, "process", "features: chirp receiver (not needed for gestures)"),
                           (presence.PresenceDetector, "update", "features: presence (not needed for gestures)"),
                           (gestures.DopplerNormalizer, "features", "features: noise floor"),
                           (gestures.GestureSegmenter, "push", "features: burst detector")):
        wrap(cls, name, key)
    jobs = bench_jobs(args.cpu_clips, args.profile)
    g._setup(args.profile)
    g.make_clip(jobs[0])
    spent.clear()
    c0, w0 = time.process_time(), time.perf_counter()
    for job in jobs:
        g.make_clip(job)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    n = len(jobs)
    out = {"clips": n, "ms_per_clip_cpu": 1e3 * cpu / n, "ms_per_clip_wall": 1e3 * wall / n,
           "stages_ms": {k: 1e3 * v / n for k, v in spent.items()}}
    many = bench_jobs(max(args.workers * 3, 12), args.profile)
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(g.make_clip, many[:args.workers]))                 # start the workers
        t0 = time.perf_counter()
        list(pool.map(g.make_clip, many, chunksize=1))
        out["all_cores_clips_per_s"] = len(many) / (time.perf_counter() - t0)
    out["workers"] = args.workers
    return out


class GpuWatch(threading.Thread):
    """Samples the GPU's load and memory with nvidia-smi while a stage runs (CUDA only)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.samples: list[tuple[float, float]] = []
        self._stop_flag = threading.Event()

    def run(self) -> None:
        while not self._stop_flag.is_set():
            try:
                res = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5)
                util, mem = res.stdout.strip().splitlines()[0].split(",")
                self.samples.append((float(util), float(mem)))
            except Exception:                                   # no nvidia-smi: nothing to sample
                return
            self._stop_flag.wait(0.25)

    def stop(self) -> dict:
        self._stop_flag.set()
        self.join(timeout=6)
        if not self.samples:
            return {}
        util = sorted(u for u, _ in self.samples)
        return {"gpu_util_mean": sum(util) / len(util), "gpu_util_median": util[len(util) // 2],
                "gpu_mem_peak_mb": max(m for _, m in self.samples), "samples": len(util)}


def run_gpu(args: argparse.Namespace) -> dict:
    """GPU clip maker: every stage on its own, then end to end."""
    from concurrent.futures import ProcessPoolExecutor

    import torch

    from datagen import generate as g
    from datagen.gpu_pipeline import GpuRenderer, default_batch, finish_clip, generate_gpu, plan_clip

    device = "cpu" if args.device == "torch-cpu" else args.device
    gpu = GpuRenderer(args.profile, device, args.precision, kernel=args.kernel, noise=args.noise)
    dev = gpu.device
    cuda = dev.type == "cuda"
    batch = args.batch or default_batch(device, gpu.precision)
    jobs = bench_jobs(args.clips, args.profile)
    out: dict = {"device": str(dev), "precision": gpu.precision, "clips": len(jobs), "batch": batch, "torch": torch.__version__,
                 "echo_loop": gpu.echo_impl, "kernel_note": gpu.kernel_note,
                 "noise_shaping": "on the GPU" if gpu.shape_on_device else "in the CPU workers"}
    if cuda:
        p = torch.cuda.get_device_properties(dev)
        out["gpu"] = f"{p.name}, {p.total_memory / 2**30:.0f} GB"

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize(dev)
        elif dev.type == "mps":
            torch.mps.synchronize()

    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(plan_clip, [(j, 0, None, True) for j in jobs[:args.workers]]))          # start the workers
        t0 = time.perf_counter()
        planned = list(pool.map(plan_clip, [(j, 0, None, dev.type != "mps", None, gpu.shape_on_device) for j in jobs], chunksize=1))
        out["plan_clips_per_s"] = len(jobs) / (time.perf_counter() - t0)
        plans = [p["plan"] for p in planned]
        # warm up (FFT plans, memory pool), then time the stages
        gpu.render(plans[:min(8, len(plans))])
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        stages: dict = {}
        t0 = time.perf_counter()
        rx_all = []
        for a in range(0, len(plans), batch):
            res = gpu.render(plans[a:a + batch], timings=stages)
            rx_all.append(res["rx"])
            del res
        sync()
        t_render = time.perf_counter() - t0
        out["render_clips_per_s"] = len(plans) / t_render
        out["render_stage_ms_per_clip"] = {k: 1e3 * v / len(plans) for k, v in stages.items()}
        if cuda:
            out["render_peak_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 2**30
            out["render_peak_mem_mb_per_clip"] = 1024 * out["render_peak_mem_gb"] / min(batch, len(plans))
        out["oom_splits"] = gpu.oom_splits
        t0 = time.perf_counter()
        cols = [gpu.columns(rx) for rx in rx_all]
        sync()
        out["features_clips_per_s"] = len(plans) / (time.perf_counter() - t0)
        import numpy as np

        db = np.concatenate([c[0] for c in cols])
        lat = np.concatenate([c[1] for c in cols]) if cols[0][1] is not None else None
        del rx_all, cols
        tasks = [(jobs[i], planned[i]["state"], 20.0, db[i], None if lat is None else lat[i], None) for i in range(len(jobs))]
        t0 = time.perf_counter()
        list(pool.map(finish_clip, tasks, chunksize=1))
        out["finish_clips_per_s"] = len(jobs) / (time.perf_counter() - t0)
        del planned, plans, tasks
        # end to end, as datagen.generate runs it
        watch = GpuWatch() if cuda else None
        if watch:
            watch.start()
        stats: dict = {}
        t0 = time.perf_counter()
        n_done = sum(1 for _ in generate_gpu(jobs, args.profile, device, args.workers, precision=gpu.precision, batch=batch,
                                             stats=stats, gpu=gpu, pool=pool))
        sync()
        out["end_to_end_clips_per_s"] = n_done / (time.perf_counter() - t0)
        out["end_to_end"] = {k: stats.get(k) for k in ("render_s", "features_s", "redraws", "batches", "oom_splits", "plan_transport")}
        if watch:
            out["end_to_end"].update(watch.stop())
    out["workers"] = args.workers
    return out


def run_train(args: argparse.Namespace) -> dict:
    """Training step: batch drawn, augmented and trained on the device, as sonar.train does it."""
    import contextlib

    import torch
    from torch import nn

    from sonar.batch_augment import BatchAugmenter
    from sonar.config import default_config
    from sonar.dataset import PAD
    from sonar.model import GestureNet
    from sonar.train import default_batch, scaled_lr

    device = torch.device("cpu" if args.device == "torch-cpu" else args.device)
    cuda = device.type == "cuda"
    cfg = default_config()
    labels = ["none", "swipe_left", "swipe_right", "push"]
    batch = default_batch(device.type)
    n = 4096 if cuda else 512
    x = torch.rand((n, cfg.n_feat_channels, cfg.doppler_bins, cfg.gesture_frames + 2 * PAD), device=device).half()
    y = torch.randint(0, len(labels), (n,), device=device)
    aug = BatchAugmenter(cfg, labels, device=device)
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = cuda and torch.cuda.is_bf16_supported()
    autocast = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if amp else contextlib.nullcontext
    out: dict = {"device": str(device), "batch": batch, "bf16": bool(amp), "widths": {}}
    steps = 200 if cuda else 12
    for width in (16, 24, 32):
        model = GestureNet(cfg.n_feat_channels, len(labels), width=width).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=scaled_lr(batch), **({"fused": True} if cuda else {}))
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
        for i in range(steps + 10):
            if i == 10:
                if cuda:
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
            pick = torch.randint(0, n, (batch,), device=device)
            xb, yb = aug(x[pick], y[pick])
            opt.zero_grad()
            with autocast():
                loss = loss_fn(model(xb).float(), yb)
            loss.backward()
            opt.step()
        if cuda:
            torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) / steps
        out["widths"][width] = {"ms_per_step": 1e3 * dt, "examples_per_s": batch / dt}
    return out


# --------------------------------------------------------------------------
# Parent: one process per timed run, then the table
# --------------------------------------------------------------------------
def child(args: argparse.Namespace, what: str, extra: list[str]) -> dict | None:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", what, "--clips", str(args.clips), "--cpu-clips", str(args.cpu_clips),
           "--workers", str(args.workers), "--device", args.device, "--batch", str(args.batch), *extra]
    if args.profile:
        cmd += ["--profile", args.profile]
    env = dict(os.environ, PYTHONUTF8="1", PYTORCH_ENABLE_MPS_FALLBACK="1")
    if what != "train" and args.device != "torch-cpu":          # worker processes: one maths thread each
        env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    line = next((x for x in res.stdout.splitlines() if x.startswith("BENCH ")), None)
    if line is None:
        print(f"  ({what} {' '.join(extra)} did not finish: {(res.stderr or res.stdout).strip().splitlines()[-1:] or 'no output'})")
        return None
    return json.loads(line[6:])


def other_gpu_programs() -> list[str]:
    try:
        res = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return [x.strip() for x in res.stdout.splitlines() if x.strip()]
    except Exception:
        return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Time every step of the data pipeline (see the file header)")
    ap.add_argument("--device", default="cuda", help="cuda (default) | mps | torch-cpu (the GPU code on the CPU)")
    ap.add_argument("--clips", type=int, default=384, help="clips for the GPU timings (default 384)")
    ap.add_argument("--cpu-clips", type=int, default=16, help="clips for the single-core CPU reference breakdown (default 16)")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 61), help="CPU processes (default: every core)")
    ap.add_argument("--batch", type=int, default=0, help="clips per GPU batch (default: fills the GPU)")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--only", choices=["cpu", "gpu", "train"], help="run just one part")
    ap.add_argument("--precisions", default="ref,fast", help="GPU number formats to time (default ref,fast)")
    ap.add_argument("--kernel", default="auto", help=argparse.SUPPRESS)
    ap.add_argument("--noise", default="cpu", help=argparse.SUPPRESS)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--precision", default="auto", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:                                               # one timed run, in this process
        result = {"cpu": run_cpu, "gpu": run_gpu, "train": run_train}[args.child](args)
        print("BENCH " + json.dumps(result))
        return 0

    report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv[1:], "cpus": os.cpu_count()}
    others = other_gpu_programs()
    if others:
        print("NOTE: other programs are using the GPU; the timings below include them:\n  " + "\n  ".join(others))
        report["other_gpu_programs"] = others
    if args.only in (None, "cpu"):
        print(f"\n1. CPU reference (numpy): {args.cpu_clips} clips on one core, then every core ...")
        report["cpu"] = r = child(args, "cpu", [])
        if r:
            print(f"   {r['ms_per_clip_cpu']:.0f} ms of CPU per clip on one core; with {r['workers']} processes: "
                  f"{r['all_cores_clips_per_s']:.1f} clips/s")
            for k, v in sorted(r["stages_ms"].items(), key=lambda kv: -kv[1]):
                print(f"     {k:<52} {v:8.1f} ms/clip  {100 * v / r['ms_per_clip_cpu']:5.1f} %")
    if args.only in (None, "gpu"):
        runs = [(p, "auto", "cpu") for p in args.precisions.split(",") if p]
        if args.device.startswith("cuda"):                       # the same once more without the fused kernel, to see what it buys
            runs.insert(1, (runs[0][0], "off", "cpu"))
        if not args.device.startswith("mps"):                    # ... and with the hall noise shaped on the GPU (--gen-noise gpu)
            runs.insert(1, (runs[0][0], "auto", "gpu"))
        for prec, kernel, noise in runs:
            print(f"\n2. GPU clip maker on {args.device}, number format '{prec}'"
                  + (", fused kernel OFF" if kernel == "off" else "")
                  + (", hall noise shaped on the GPU (--gen-noise gpu)" if noise == "gpu" else "") + f", {args.clips} clips ...")
            report[f"gpu_{prec}" + ("_nokernel" if kernel == "off" else "") + ("_noisegpu" if noise == "gpu" else "")] = r = \
                child(args, "gpu", ["--precision", prec, "--kernel", kernel, "--noise", noise])
            if not r:
                continue
            print(f"   {r.get('gpu', r['device'])}; batch {r['batch']}; PyTorch {r['torch']}")
            print(f"   echo loop: {r['echo_loop']}" + (f" ({r['kernel_note']})" if r.get("kernel_note") else ""))
            print(f"     random draws (CPU, {r['workers']} processes)          {r['plan_clips_per_s']:9.1f} clips/s   (results pickled, the old way; "
                  f"hall noise shaped {r.get('noise_shaping', 'in the CPU workers')})")
            print(f"     rendering (GPU)                                 {r['render_clips_per_s']:9.1f} clips/s")
            for k, v in r["render_stage_ms_per_clip"].items():
                print(f"        {k:<44} {v:7.2f} ms/clip")
            print(f"     Doppler columns (GPU)                           {r['features_clips_per_s']:9.1f} clips/s")
            print(f"     bursts + labels (CPU, {r['workers']} processes)       {r['finish_clips_per_s']:9.1f} clips/s")
            e = r["end_to_end"]
            print(f"     END TO END (plans via {e.get('plan_transport', 'pickle')})".ljust(53) + f"{r['end_to_end_clips_per_s']:9.1f} clips/s"
                  + (f"   GPU load: mean {e['gpu_util_mean']:.0f} %, median {e['gpu_util_median']:.0f} %, memory peak "
                     f"{e['gpu_mem_peak_mb'] / 1024:.1f} GB" if "gpu_util_mean" in e else ""))
            if "render_peak_mem_gb" in r:
                print(f"     memory at the largest batch: {r['render_peak_mem_gb']:.1f} GB ({r['render_peak_mem_mb_per_clip']:.0f} MB per clip)"
                      + (f"; {r['oom_splits']} batch(es) did not fit and were split: use a smaller --gen-batch" if r.get("oom_splits") else ""))
            c = report.get("cpu")
            if c:
                print(f"     against the CPU reference on every core ({c['all_cores_clips_per_s']:.1f} clips/s): "
                      f"{r['end_to_end_clips_per_s'] / c['all_cores_clips_per_s']:.1f}x")
    if args.only in (None, "train"):
        print(f"\n3. training step on {args.device} (augmentation + model, as sonar.train) ...")
        report["train"] = r = child(args, "train", [])
        if r:
            for w, v in r["widths"].items():
                print(f"     width {w}: {v['ms_per_step']:.2f} ms per step of {r['batch']} = {v['examples_per_s']:,.0f} examples/s"
                      + (" (bf16)" if r["bf16"] else ""))
    out = REPO / "train_out" / "bench" / f"bench_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"\nsaved {out.relative_to(REPO)} (send this file back with the output above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

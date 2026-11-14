"""gpu_pipeline.py — datagen.generate with the heavy maths on the GPU.

Plain words
-----------
generate.py makes one clip per CPU core at a time, and about 99 % of that time is
arithmetic on long arrays (the echo of every body part, filters, spectrograms).
Here the same clips are made in three steps:

    1. plan    CPU workers. The random part of a clip: the story, the virtual laptop,
               the hall noise. The same code and the same random draws as the CPU
               path (generate.draw_world, render_torch.make_plan), about 20 ms.
    2. render  the GPU, hundreds of clips at a time: echoes, room, microphone, 16-bit
               rounding (render_torch), then the Doppler columns (sonar.features_torch).
    3. finish  CPU workers. The engine's own noise-floor and movement-burst code
               decides where the bursts are, and the labels are attached, exactly as
               generate.make_clip does, about 10 ms.

A gesture too faint to count is redrawn (up to 4 attempts) just like on the CPU: the
GPU measures how audible it is and the clip goes back to step 1 with its random
generator where it stopped. So for the same seed this path makes the same clips as
the CPU path; scripts/check_gpu_parity.py measures how closely.

Read-only helper module: generate.py writes the files.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import Callable, Iterator

import numpy as np

from . import adapter as ad
from . import generate as g
from .hardware import Profile

_TX_OVERLAY: dict = {}       # per worker: the speaker signal for overlay clips (shorter pre-roll)


def overlay_tx(core: ad.Core, n: int) -> tuple[np.ndarray, int]:
    """The speaker buffer background.inject renders against, and its pre-roll."""
    key = (id(core), n)
    if key not in _TX_OVERLAY:
        pre = ad.pre_roll(core, 2500 + core.period)
        _TX_OVERLAY[key] = (core.tx(pre, n), pre)
    return _TX_OVERLAY[key]


_BIG = ("noise", "gain", "pos", "rho", "nrm", "real")     # the arrays that make a plan several megabytes
SLOT_BYTES = 12 * 2**20                                   # one shared-memory slot (a plan that does not fit is pickled as before)


def _to_slot(plan, slot: str | None) -> None:
    """Worker: move the plan's big arrays into the main process's shared-memory slot, leaving only where they are.

    Why: on Windows every result of a worker is pickled through one pipe that a single thread reads (about
    700 MB/s on Lance's PC = 100 plans a second, the limit of the whole pipeline). Shared memory has no such limit.
    """
    plan.shm = None
    if not slot:
        return
    from multiprocessing import shared_memory

    arrays = [(name, np.ascontiguousarray(getattr(plan, name))) for name in _BIG if getattr(plan, name) is not None]
    if sum(a.nbytes for _, a in arrays) > SLOT_BYTES:
        return
    try:
        shm = shared_memory.SharedMemory(name=slot)
    except (OSError, ValueError):
        return
    try:                                                  # (the block stays the main process's: it creates and removes it)
        where, offset = [], 0
        for name, a in arrays:
            np.ndarray(a.shape, a.dtype, buffer=shm.buf, offset=offset)[...] = a
            where.append((name, a.dtype.str, a.shape, offset))
            offset += a.nbytes
            setattr(plan, name, None)
        plan.shm = where
    finally:
        shm.close()


def _from_slot(plan, shm) -> None:
    """Main process: the plan's big arrays as views into its slot (valid until the slot is handed out again)."""
    for name, dtype, shape, offset in plan.shm or []:
        setattr(plan, name, np.ndarray(shape, np.dtype(dtype), buffer=shm.buf, offset=offset))


def plan_clip(task: tuple) -> dict:
    """Worker: the random half of one attempt at a clip (see the file header)."""
    from .plan import make_plan

    job, attempt, carry, noise64 = task[:4]
    slot = task[4] if len(task) > 4 else None
    j = g.unpack_job(job)
    core, (tx, pre, n) = g._setup(j["profile"])
    hw_profile = Profile(band_hz=core.band, swap_mics=j["swap_mics"], swap_speakers=j["swap_speakers"])
    if carry is None:
        rng = np.random.default_rng(j["seed"])
        use_real = bool(j["backgrounds"]) and rng.random() < j["bg_share"]
    else:                                   # a redraw: carry on with the clip's random generator where it stopped
        rng = np.random.default_rng()
        rng.bit_generator.state = carry["rng"]
        use_real = carry["use_real"]
    scene, hw, excerpt, _ = g.draw_world(rng, core, j, hw_profile, use_real, n)
    dtype = np.float64 if noise64 else np.float32
    if use_real:                            # background.inject: only the moving scene, no noise, no rounding
        tx_o, pre_o = overlay_tx(core, n)
        plan = make_plan(tx_o, hw, scene, n, rng, pre_o, core.band, add_noise=False, include_static=False, noise_dtype=dtype)
        plan.real = np.ascontiguousarray(excerpt, dtype=np.float32)
    else:
        plan = make_plan(tx, hw, scene, n, rng, pre, core.band, noise_dtype=dtype)
    meta = g.clip_meta(j, scene, hw, float("nan"), use_real, 0)       # echo_snr_db and n_mics are filled in after the GPU
    _to_slot(plan, slot)
    return {"plan": plan, "state": {"rng": rng.bit_generator.state, "use_real": use_real, "meta": meta}}


def finish_clip(task: tuple) -> dict:
    """Worker: Doppler columns of one clip -> the labelled examples, as generate.make_clip returns them."""
    from sonar.features import stream_from_columns

    job, state, snr, doppler_db, lateral, audio = task[:6]
    want_stream = len(task) > 6 and task[6]
    j = g.unpack_job(job)
    core, _ = g._setup(j["profile"])
    n_mics = doppler_db.shape[1] // core.cfg.n_tones
    cfg = core.cfg if n_mics == core.cfg.n_mics else core.cfg.with_(n_mics=n_mics)
    stream = ad.wrap_stream(core, stream_from_columns(cfg, doppler_db, lateral))
    meta = state["meta"]
    meta["echo_snr_db"] = None if not np.isfinite(snr) else round(float(snr), 1)
    meta["n_mics"] = int(n_mics)
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng"]
    events = meta["events"]
    result = {"meta": meta, "examples": g.label_segments(stream, events, core), "audio": audio, "error": None}
    if rng.random() < 0.2:
        idle = g.idle_example(rng, stream, core, events)
        if idle is not None:
            result["examples"].append(idle)
    for ex in result["examples"]:           # the padded picture is what gets stored; do not ship both
        if "padded" in ex:
            ex["window"] = None
    if want_stream:
        result["stream"] = stream.core
    return result


def batch_for_memory(free_gb: float, precision: str) -> int:
    """Clips rendered at once for this much free GPU memory (peak working memory per clip is an estimate;
    a batch that still does not fit is split in two automatically, see GpuRenderer.render)."""
    per_clip_gb = {"ref": 0.085, "fast": 0.045}.get(precision, 0.045)
    return int(max(16, min(512, (free_gb - 4.0) / per_clip_gb)))


def default_batch(device, precision: str) -> int:
    """The 4090 is dedicated to this job, so a batch fills most of the memory that is free right now."""
    import torch

    dev = torch.device(device)
    if dev.type == "cuda":
        return batch_for_memory(torch.cuda.mem_get_info(dev)[0] / 2**30, precision)
    return 16 if dev.type == "mps" else 8                  # Apple GPUs share the computer's memory: stay modest


class GpuRenderer:
    """Holds the per-run tables on the device; turns plans into audio and Doppler columns."""

    def __init__(self, profile: str | None, device: str, precision: str = "auto", synth: Callable | None = None,
                 kernel: str = "auto") -> None:
        """kernel: "auto" = use the fused CUDA kernel for the echo loop if its probe passes on this GPU (else the
        PyTorch operations), "off" = PyTorch operations, "on" = the kernel without asking the probe."""
        import torch

        from .render_torch import pick_precision

        self.torch = torch
        self.core, (self.tx, self.pre, self.n) = g._setup(profile)
        if not self.core.available:
            raise RuntimeError("the GPU path needs the core sonar/ package (probe signal and feature code)")
        self.device = torch.device(device)
        self.precision = pick_precision(self.device, precision).name
        self.synth = synth
        self.echo_impl = "PyTorch operations" if synth is None else getattr(synth, "__name__", "custom")
        self.kernel_note = ""
        if synth is None and kernel == "host":               # tests: the kernel's C code compiled for the CPU
            from . import echo_kernel

            self.synth, self.echo_impl = echo_kernel.synth_paths_host, "the kernel's C code on the CPU (test build)"
        elif synth is None and kernel != "off" and self.device.type == "cuda" and self.precision != "single":
            from . import echo_kernel

            ok, why = (True, "not probed (--gen-kernel on)") if kernel == "on" else echo_kernel.usable(self.device)
            if ok:
                self.synth, self.echo_impl = echo_kernel.synth_paths_cuda, "fused CUDA kernel"
            self.kernel_note = why
        self.oom_splits = 0                  # batches that did not fit in memory and were split
        self._ctx: dict = {}
        self._fe: dict = {}

    def context(self, plan):
        from .render_torch import RenderContext

        key = (plan.pre, plan.n_total)
        if key not in self._ctx:
            tx = self.tx if key == (self.pre, len(self.tx)) else overlay_tx(self.core, self.n)[0]
            self._ctx[key] = RenderContext(tx, self.core.fs, self.n, plan.pre, self.core.band, self.device,
                                           precision=self.precision, period=self.core.period)
        return self._ctx[key]

    def frontend(self, n_mics: int):
        from sonar.features_torch import TorchFrontend

        if n_mics not in self._fe:
            self._fe[n_mics] = TorchFrontend(self.core.cfg, n_mics, self.device,
                                             precision="single" if self.precision == "single" else "auto")
        return self._fe[n_mics]

    def render(self, plans: list, want_parts: bool = False, timings: dict | None = None) -> dict:
        """Audio for a list of plans of one clip type. A batch that does not fit in memory is split in two."""
        from .render_torch import render_batch

        torch = self.torch
        try:
            with torch.no_grad():
                return render_batch(self.context(plans[0]), plans, want_parts=want_parts, synth=self.synth, timings=timings)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                if self.echo_impl != "fused CUDA kernel":
                    raise
                # the kernel failed while running: carry on with the PyTorch operations (same numbers, slower)
                self.synth, self.echo_impl = None, "PyTorch operations"
                self.kernel_note = f"the fused kernel failed at run time and was switched off: {str(exc)[:200]}"
                print(f"WARNING: {self.kernel_note}", flush=True)
                return self.render(plans, want_parts, timings)
            if len(plans) < 2:
                raise
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.oom_splits += 1
        half = len(plans) // 2
        a, b = self.render(plans[:half], want_parts, timings), self.render(plans[half:], want_parts, timings)
        return {k: torch.cat([a[k], b[k]]) for k in a}

    def columns(self, rx, chunk: int = 256) -> tuple[np.ndarray, np.ndarray | None]:
        """rx [B, n, n_mics] on the device -> (doppler_db, lateral) as numpy, in pieces that fit in memory."""
        torch = self.torch
        fe = self.frontend(rx.shape[2])
        dbs, lats = [], []
        for a in range(0, rx.shape[0], chunk):
            db, lat = fe.columns(rx[a:a + chunk])
            dbs.append(db.cpu().numpy())
            if lat is not None:
                lats.append(lat.cpu().numpy())
        return np.concatenate(dbs), (np.concatenate(lats) if lats else None)


def gpu_devices(device: str, gpus: int = 0) -> list[str]:
    """The devices one run uses: "cuda" means every visible GPU (or the first `gpus`), anything else just itself."""
    if device != "cuda":
        return [device]
    import torch

    n = torch.cuda.device_count()
    n = min(n, gpus) if gpus > 0 else n
    return [f"cuda:{i}" for i in range(max(n, 1))]


def _render_group(gpu: "GpuRenderer", group: list, want_capture: bool) -> dict:
    """One batch on one GPU: audio, which clips are loud enough to keep, and their Doppler columns."""
    torch = gpu.torch

    def sync() -> None:
        if gpu.device.type == "cuda":
            torch.cuda.synchronize(gpu.device)

    t0 = time.perf_counter()
    out = gpu.render([item[3] for item in group], want_parts=want_capture)
    sync()
    t_render = time.perf_counter() - t0
    snr = out["snr_db"].cpu().double().numpy()                # (MPS has no float64: convert on the host)
    keep, redo = [], []
    for k, (i, job, attempt, _, state) in enumerate(group):
        j = g.unpack_job(job)
        (keep if g.accept(j["scene"], snr[k], j["min_snr"]) or attempt + 1 >= g.MAX_ATTEMPTS else redo).append(k)
    by_mics: dict = defaultdict(list)                        # normally one group: every clip of a run has the same microphones
    for k in keep:
        by_mics[g.unpack_job(group[k][1])["mics"]].append(k)
    t0 = time.perf_counter()
    rows = []                                                # (k, doppler_db, lateral, audio, captured parts)
    for mics, ks in by_mics.items():
        rx = out["rx"][torch.tensor(ks, device=gpu.device)]
        if mics:
            rx = rx[:, :, :mics]
        db, lat = gpu.columns(rx)
        for row, k in enumerate(ks):
            job = group[k][1]
            want_audio = g.unpack_job(job)["keep_audio"] or want_capture
            audio = rx[row].cpu().numpy() if want_audio else None
            cap = None
            if want_capture:
                cap = {"rx": audio, "snr": float(snr[k]), "attempts": group[k][2] + 1,
                       "dynamic": out["dynamic"][k].to(torch.float32).cpu().numpy(),
                       "static": out["static"][k].to(torch.float32).cpu().numpy(),
                       "converter_in": out["converter_in"][k].cpu().numpy()}
            rows.append((k, db[row], None if lat is None else lat[row], audio if g.unpack_job(job)["keep_audio"] else None, cap))
    sync()
    return {"rows": rows, "redo": redo, "snr": snr, "render_s": t_render, "features_s": time.perf_counter() - t0}


def generate_gpu(jobs: list, profile: str | None, device: str, workers: int, precision: str = "auto",
                 batch: int = 0, synth: Callable | None = None, stats: dict | None = None,
                 capture: Callable | None = None, gpu: "GpuRenderer | list | None" = None,
                 pool: ProcessPoolExecutor | None = None, gpus: int = 0, kernel: str = "auto",
                 shared: bool = True) -> Iterator[dict]:
    """Yield one result per job, in job order: the same dicts generate.make_clip returns.

    device "cuda" uses every visible GPU (`gpus` limits how many): each GPU renders its own batches.
    `stats`, if given, collects seconds per stage and the number of redraws.
    `capture(index, dict)`, if given, receives each accepted clip's audio and its parts (the parity
    check compares them with the CPU path); results then also carry the feature stream.
    `gpu` / `pool` let a caller reuse the device tables and the worker processes between calls.
    `shared` = hand the plans' big arrays from the workers to this process through shared memory (see _to_slot).
    """
    from concurrent.futures import ThreadPoolExecutor

    if gpu is None:
        fleet = [GpuRenderer(profile, d, precision, synth, kernel) for d in gpu_devices(device, gpus)]
    else:
        fleet = list(gpu) if isinstance(gpu, (list, tuple)) else [gpu]
    batch = batch or default_batch(str(fleet[0].device), fleet[0].precision)
    noise64 = fleet[0].device.type != "mps"                  # the hall noise goes to the GPU as drawn (float64)
    ahead = max(int(1.5 * batch) * len(fleet), 4 * workers)   # plans made ahead of the GPUs (they are a few MB each)
    stats = stats if stats is not None else {}
    stats.update({"batch": batch, "precision": fleet[0].precision, "device": ", ".join(str(f.device) for f in fleet),
                  "render_s": 0.0, "features_s": 0.0, "redraws": 0, "batches": 0, "oom_splits": 0,
                  "echo_loop": fleet[0].echo_impl, "kernel_note": fleet[0].kernel_note})
    todo: deque = deque((i, job, 0, None) for i, job in enumerate(jobs))
    planning: dict[Future, tuple] = {}
    ready: dict[tuple, list] = defaultdict(list)             # clip type -> planned clips waiting for a GPU
    rendering: dict[Future, tuple] = {}                      # batch on a GPU -> (GPU number, its clips)
    finishing: dict[Future, int] = {}
    done: dict[int, dict] = {}
    idle = list(range(len(fleet)))
    next_out = 0
    # shared-memory slots for the plans' big arrays (see _to_slot); none if the system refuses
    slots: list = []
    if shared and workers > 0:
        from multiprocessing import shared_memory

        try:
            for _ in range(ahead + batch * len(fleet) + 8):
                slots.append(shared_memory.SharedMemory(create=True, size=SLOT_BYTES))
        except (OSError, ValueError):
            for shm in slots:
                shm.close()
                shm.unlink()
            slots = []
    free = list(range(len(slots)))
    stats["plan_transport"] = "shared memory" if slots else "pickled"
    own_pool = pool is None
    pool = pool or ProcessPoolExecutor(max(1, workers))
    threads = ThreadPoolExecutor(len(fleet))                 # one thread per GPU; the main thread keeps the workers fed
    try:
        while next_out < len(jobs):
            n_ready = sum(len(v) for v in ready.values())
            while todo and len(planning) + n_ready < ahead:
                i, job, attempt, carry = todo.popleft()
                k = free.pop() if free else None
                planning[pool.submit(plan_clip, (job, attempt, carry, noise64, slots[k].name if k is not None else None))] = (i, job, attempt, k)
            # hand full batches (or, when nothing else is coming, the rest) to idle GPUs
            while idle:
                more_coming = bool(planning) or bool(todo)
                kinds = [k for k, v in ready.items() if len(v) >= batch] or \
                        ([k for k, v in ready.items() if v] if not more_coming and not rendering else [])
                if not kinds:
                    break
                kind = max(kinds, key=lambda k: len(ready[k]))
                group, ready[kind] = ready[kind][:batch], ready[kind][batch:]
                n = idle.pop()
                rendering[threads.submit(_render_group, fleet[n], group, capture is not None)] = (n, group)
            futures = list(planning) + list(finishing) + list(rendering)
            if futures:
                finished, _ = wait(futures, return_when=FIRST_COMPLETED)
                for f in finished:
                    if f in planning:
                        i, job, attempt, k = planning.pop(f)
                        res = f.result()
                        if k is not None and res["plan"].shm is None:      # the plan did not use its slot (too big)
                            free.append(k)
                            k = None
                        if k is not None:
                            _from_slot(res["plan"], slots[k])
                        res["state"]["slot"] = k
                        ready[(res["plan"].pre, res["plan"].n_total)].append((i, job, attempt, res["plan"], res["state"]))
                    elif f in finishing:
                        done[finishing.pop(f)] = f.result()
                    else:
                        n, group = rendering.pop(f)
                        idle.append(n)
                        res = f.result()
                        for item in group:                    # the audio is on the GPU now: the slots can be reused
                            k = item[4].pop("slot", None)
                            if k is not None:
                                for name in _BIG:
                                    setattr(item[3], name, None)
                                free.append(k)
                        stats["render_s"] += res["render_s"]
                        stats["features_s"] += res["features_s"]
                        stats["batches"] += 1
                        for k in res["redo"]:                 # too faint: redraw the room for this clip
                            i, job, attempt, _, state = group[k]
                            stats["redraws"] += 1
                            todo.appendleft((i, job, attempt + 1, {"rng": state["rng"], "use_real": state["use_real"]}))
                        for k, db, lat, audio, cap in res["rows"]:
                            i, job, _, _, state = group[k]
                            if capture is not None:
                                capture(i, cap)
                            task = (job, state, float(res["snr"][k]), db, lat, audio, capture is not None)
                            finishing[pool.submit(finish_clip, task)] = i
            while next_out in done:
                yield done.pop(next_out)
                next_out += 1
        stats["oom_splits"] = sum(f.oom_splits for f in fleet)
        stats.update(echo_loop=fleet[0].echo_impl, kernel_note=fleet[0].kernel_note)
    finally:
        threads.shutdown(wait=True)
        if own_pool:
            pool.shutdown()
        for shm in slots:
            try:
                shm.close()
                shm.unlink()
            except (OSError, BufferError):
                pass

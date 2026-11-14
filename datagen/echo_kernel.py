"""echo_kernel.py — run the fused echo-path kernel (echo_paths.c) instead of 300 PyTorch operations.

Plain words
-----------
The echo-path loop is 86 % of the rendering time. render_torch.synth_paths_torch does it with
ordinary PyTorch operations, each a separate pass over hundreds of megabytes. echo_paths.c does
the same arithmetic in one pass on the GPU. This file
  * packs a batch into the flat arrays the kernel reads (`pack`),
  * compiles the kernel for the GPU at run time and starts it (`synth_paths_cuda`): with NVRTC, the
    compiler library that ships with PyTorch's CUDA build, through the CUDA driver; or with CuPy
    if that is installed. No C compiler and no extra package is needed on the training machine;
  * compiles the same file as plain C for the CPU (`synth_paths_host`), which is how the kernel's
    arithmetic was checked against the PyTorch path on a computer without CUDA;
  * `probe` (also: `python -m datagen.echo_kernel`) renders a few clips with the PyTorch path and
    with the kernel and compares them. gpu_pipeline uses the kernel only if this passes; a failure
    of any kind falls back to the PyTorch path and says why.

What is NOT verified where this was written: the two GPU launch routes themselves (that Mac has
no CUDA). The arithmetic is verified; the probe and scripts/check_gpu_parity.py verify the rest
on the GPU that runs it.

How to run:   python -m datagen.echo_kernel            (cuda if there is one, else the plain-C build)
What it changes: writes a small verdict file in the system's temp folder so the next run need not
repeat the probe. Reads everything else.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .render_torch import RenderContext, spline_coefs, synth_paths_torch, wobble_paths

SOURCE = Path(__file__).with_name("echo_paths.c")
_ARGS = ("cd", "cf", "ci", "pos_c", "rho_c", "nrm_c", "sf", "flags", "piece", "dt", "kdec", "env", "dyn", "key")
_BLOCK = 256


def source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def source_hash() -> str:
    return hashlib.sha256((source_text() + Path(__file__).read_text(encoding="utf-8")).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# The kernel's input
# --------------------------------------------------------------------------
def pack(ctx: RenderContext, sub: dict, env_ri: torch.Tensor) -> dict:
    """One sub-batch (render_torch._gather_sub) as the flat arrays echo_paths.c reads. All on ctx.device."""
    dev, f32, f64 = ctx.device, torch.float32, torch.float64
    b, grid, sp = sub["b"], sub["grid"], sub["spline"]
    counts = np.asarray(sub["counts"], np.int64)
    ends = np.cumsum(counts)
    k = {"n_clip": b, "nb": grid["nb"], "n_mic": sub["n_mic"], "n_piece": int(sp["dx"].shape[0]), "n_total": ctx.n_total,
         "fs": float(ctx.fs), "f0": float(ctx.f0)}
    mic = sub["mic"].to(f64)
    if mic.shape[1] < 2:                                               # the kernel has room for two microphones
        mic = torch.cat([mic, mic], dim=1)
    k["cd"] = torch.cat([sub["spk"].to(f64).reshape(b, 6), mic[:, :2].reshape(b, 6), sub["c"].to(f64)[:, None],
                         sub["pre_eff"].to(f64)[:, None]], dim=1).contiguous()
    kappa = sub["alpha"] * (math.log(10.0) / 20.0)
    k["cf"] = torch.cat([sub["spk_axis"].reshape(b, 6), sub["mic_axis"].reshape(b, 3)]
                        + [sub[n][:, None] for n in ("spk_n", "spk_floor", "mic_n", "mic_floor", "table_refl", "k_amp", "gamma")]
                        + [kappa[:, None], sub["spk_on"]], dim=1).to(f32).contiguous()
    k["ci"] = torch.stack([sub["env_row"].to(torch.int32), torch.from_numpy((ends - counts).astype(np.int32)).to(dev),
                           torch.from_numpy(ends.astype(np.int32)).to(dev)], dim=1).contiguous()
    k["pos_c"] = spline_coefs(sub["pos"], sp).to(f64).contiguous()                      # [N, P, 4, 3]
    k["rho_c"] = spline_coefs(sub["rho"][..., None], sp)[..., 0].to(f32).contiguous()   # [N, P, 4]
    k["nrm_c"] = spline_coefs(sub["nrm"], sp).to(f32).contiguous()                      # [N, P, 4, 3]
    k["sf"] = torch.stack([sub["spec_q"], sub["spec_floor"]], dim=1).to(f32).contiguous()
    k["flags"] = (sub["use_spec"].to(torch.int32) + 2 * sub["key"].to(torch.int32)).contiguous()
    k["piece"] = grid["piece"].to(torch.int32).contiguous()
    k["dt"] = grid["dt"].to(f64).contiguous()
    k["kdec"] = grid["kdec"].to(f64).contiguous()
    k["env"] = env_ri.contiguous()
    k["dyn"] = torch.zeros((b, sub["n_mic"], grid["nb"], 2), dtype=f32, device=dev)
    k["key"] = torch.zeros((b, sub["n_mic"], grid["nb"], 2), dtype=f32, device=dev)
    return k


# --------------------------------------------------------------------------
# Plain C on the CPU (tests only)
# --------------------------------------------------------------------------
_HOST = None


def _host_library() -> ctypes.CDLL:
    """echo_paths.c compiled as an ordinary shared library with the system's C compiler (cached by content)."""
    global _HOST
    if _HOST is None:
        ext = ".dll" if os.name == "nt" else (".dylib" if sys.platform == "darwin" else ".so")
        out = Path(tempfile.gettempdir()) / f"sonar_echo_paths_{source_hash()}{ext}"
        if not out.exists():
            cc = os.environ.get("CC", "cc")
            subprocess.run([cc, "-O2", "-shared", "-fPIC", "-o", str(out), str(SOURCE), "-lm"], check=True, capture_output=True)
        lib = ctypes.CDLL(str(out))
        lib.echo_paths.restype = None
        lib.echo_paths.argtypes = [ctypes.c_int] * 4 + [ctypes.c_longlong, ctypes.c_double, ctypes.c_double] + [ctypes.c_void_p] * len(_ARGS)
        _HOST = lib
    return _HOST


def synth_paths_host(ctx: RenderContext, sub: dict, env_ri: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's arithmetic on the CPU: same result as synth_paths_torch up to float32 rounding."""
    k = pack(ctx, sub, env_ri)
    _host_library().echo_paths(k["n_clip"], k["nb"], k["n_mic"], k["n_piece"], k["n_total"], k["fs"], k["f0"],
                               *[ctypes.c_void_p(k[name].data_ptr()) for name in _ARGS])
    wobble_paths(ctx, sub, env_ri, k["dyn"])
    return k["dyn"], k["key"]


# --------------------------------------------------------------------------
# CUDA: compile with NVRTC, launch through the driver (or let CuPy do both)
# --------------------------------------------------------------------------
def _candidates(patterns: list[str]) -> list[Path]:
    roots = [Path(torch.__file__).parent / "lib", Path(torch.__file__).parent.parent / "nvidia"]
    for var in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(var):
            roots.append(Path(os.environ[var]))
    found: list[Path] = []
    for root in roots:
        if root.exists():
            for pat in patterns:
                found += sorted(root.rglob(pat), reverse=True)
    return [p for p in found if "builtins" not in p.name and "alt" not in p.name]


class _Nvrtc:
    """The run-time CUDA compiler that ships with PyTorch's CUDA build: C source in, GPU code (PTX) out."""

    def __init__(self) -> None:
        names = _candidates(["nvrtc64_*.dll"] if os.name == "nt" else ["libnvrtc.so*"])
        last: Exception | None = None
        self.lib = None
        for path in [*names, Path("nvrtc64_120_0.dll" if os.name == "nt" else "libnvrtc.so")]:
            try:
                if os.name == "nt" and path.parent.exists() and str(path.parent) != ".":
                    os.add_dll_directory(str(path.parent))             # NVRTC loads its "builtins" library from here
                self.lib = ctypes.CDLL(str(path))
                self.path = str(path)
                break
            except OSError as exc:
                last = exc
        if self.lib is None:
            raise RuntimeError(f"NVRTC (the CUDA run-time compiler) was not found next to PyTorch: {last}")
        L, P = self.lib, ctypes.c_void_p
        L.nvrtcCreateProgram.argtypes = [ctypes.POINTER(P), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, P, P]
        L.nvrtcCompileProgram.argtypes = [P, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        L.nvrtcGetPTXSize.argtypes = [P, ctypes.POINTER(ctypes.c_size_t)]
        L.nvrtcGetPTX.argtypes = [P, ctypes.c_char_p]
        L.nvrtcGetProgramLogSize.argtypes = [P, ctypes.POINTER(ctypes.c_size_t)]
        L.nvrtcGetProgramLog.argtypes = [P, ctypes.c_char_p]
        L.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(P)]

    def compile(self, src: str, options: list[str]) -> bytes:
        L = self.lib
        prog = ctypes.c_void_p()
        if L.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"echo_paths.cu", 0, None, None) != 0:
            raise RuntimeError("nvrtcCreateProgram failed")
        try:
            opts = (ctypes.c_char_p * len(options))(*[o.encode() for o in options])
            code = L.nvrtcCompileProgram(prog, len(options), opts)
            if code != 0:
                size = ctypes.c_size_t()
                L.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
                log = ctypes.create_string_buffer(size.value + 1)
                L.nvrtcGetProgramLog(prog, log)
                raise RuntimeError(f"NVRTC could not compile the kernel (code {code}): {log.value.decode(errors='replace')[-600:]}")
            size = ctypes.c_size_t()
            L.nvrtcGetPTXSize(prog, ctypes.byref(size))
            ptx = ctypes.create_string_buffer(size.value)
            L.nvrtcGetPTX(prog, ptx)
            return ptx.raw
        finally:
            L.nvrtcDestroyProgram(ctypes.byref(prog))


class _DriverKernel:
    """echo_paths loaded into PyTorch's own CUDA context and launched with cuLaunchKernel."""

    name = "CUDA kernel (NVRTC)"

    def __init__(self, device: torch.device) -> None:
        self.device = device
        major, minor = torch.cuda.get_device_capability(device)
        nvrtc = _Nvrtc()
        ptx, errors = None, []
        for arch in dict.fromkeys([f"compute_{major}{minor}", "compute_89", "compute_86", "compute_80", "compute_75"]):
            try:
                ptx = nvrtc.compile(source_text(), ["-DSONAR_CUDA", f"--gpu-architecture={arch}", "--std=c++14"])
                break
            except RuntimeError as exc:
                errors.append(f"{arch}: {exc}")
        if ptx is None:
            raise RuntimeError("; ".join(errors)[-900:])
        self.drv = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
        D, P = self.drv, ctypes.c_void_p
        D.cuModuleLoadData.argtypes = [ctypes.POINTER(P), ctypes.c_char_p]
        D.cuModuleGetFunction.argtypes = [ctypes.POINTER(P), P, ctypes.c_char_p]
        D.cuLaunchKernel.argtypes = [P] + [ctypes.c_uint] * 7 + [P, ctypes.POINTER(P), P]
        D.cuCtxGetCurrent.argtypes = [ctypes.POINTER(P)]
        with torch.cuda.device(device):
            torch.zeros(1, device=device)                              # makes PyTorch's context current on this thread
            ctxp = P()
            if D.cuCtxGetCurrent(ctypes.byref(ctxp)) != 0 or not ctxp.value:
                raise RuntimeError("no current CUDA context after a PyTorch operation")
            self.module, self.func = P(), P()
            code = D.cuModuleLoadData(ctypes.byref(self.module), ptx)
            if code != 0:
                raise RuntimeError(f"cuModuleLoadData failed with code {code}")
            code = D.cuModuleGetFunction(ctypes.byref(self.func), self.module, b"echo_paths")
            if code != 0:
                raise RuntimeError(f"cuModuleGetFunction failed with code {code}")

    def __call__(self, k: dict) -> None:
        values = [ctypes.c_int(k["n_clip"]), ctypes.c_int(k["nb"]), ctypes.c_int(k["n_mic"]), ctypes.c_int(k["n_piece"]),
                  ctypes.c_longlong(k["n_total"]), ctypes.c_double(k["fs"]), ctypes.c_double(k["f0"])]
        values += [ctypes.c_void_p(k[name].data_ptr()) for name in _ARGS]
        params = (ctypes.c_void_p * len(values))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in values])
        threads = k["n_clip"] * k["nb"]
        with torch.cuda.device(self.device):
            stream = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
            code = self.drv.cuLaunchKernel(self.func, (threads + _BLOCK - 1) // _BLOCK, 1, 1, _BLOCK, 1, 1, 0, stream, params, None)
            if code != 0:
                raise RuntimeError(f"cuLaunchKernel failed with code {code}")
            # wait for it: the input arrays must stay alive until the kernel has read them, and a fault inside the
            # kernel should surface here (where the fallback can catch it), not in some later PyTorch call
            torch.cuda.synchronize(self.device)


class _CupyKernel:
    """The same kernel through CuPy's RawKernel (used if NVRTC cannot be reached directly and CuPy is installed)."""

    name = "CUDA kernel (CuPy)"

    def __init__(self, device: torch.device) -> None:
        import cupy

        self.cupy, self.device = cupy, device
        self.kernel = cupy.RawKernel(source_text(), "echo_paths", options=("-DSONAR_CUDA",))

    def __call__(self, k: dict) -> None:
        cp = self.cupy
        with cp.cuda.Device(self.device.index or 0):
            args = (np.int32(k["n_clip"]), np.int32(k["nb"]), np.int32(k["n_mic"]), np.int32(k["n_piece"]), np.int64(k["n_total"]),
                    np.float64(k["fs"]), np.float64(k["f0"]), *[cp.asarray(k[name]) for name in _ARGS])
            threads = k["n_clip"] * k["nb"]
            self.kernel(((threads + _BLOCK - 1) // _BLOCK,), (_BLOCK,), args)
            cp.cuda.get_current_stream().synchronize()


_KERNELS: dict = {}


def cuda_kernel(device: torch.device):
    """Compile (once per GPU) and return the launcher. Raises if neither route works."""
    key = str(device)
    if key not in _KERNELS:
        errors = []
        for cls in (_DriverKernel, _CupyKernel):
            try:
                _KERNELS[key] = cls(device)
                break
            except Exception as exc:                                    # try the next route
                errors.append(f"{cls.name}: {type(exc).__name__}: {exc}")
        else:
            raise RuntimeError(" | ".join(errors))
    return _KERNELS[key]


def synth_paths_cuda(ctx: RenderContext, sub: dict, env_ri: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """synth_paths_torch's result from the fused kernel. Same arguments, same return value."""
    k = pack(ctx, sub, env_ri)
    if k["pos_c"].shape[0]:                                             # clips without any reflector: nothing to add
        cuda_kernel(ctx.device)(k)
    wobble_paths(ctx, sub, env_ri, k["dyn"])
    return k["dyn"], k["key"]


# --------------------------------------------------------------------------
# The probe: does the kernel give the PyTorch path's numbers on this machine?
# --------------------------------------------------------------------------
def probe(device: str = "auto", clips: int = 6, log=print) -> dict:
    """Render a few fixture clips with both echo loops and compare. Returns {"ok", "why", "rel", ...}."""
    from . import parity
    from .gpu_pipeline import GpuRenderer, plan_clip

    host = device == "host" or (device == "auto" and not torch.cuda.is_available())
    dev = "cpu" if host else ("cuda" if device == "auto" else device)
    jobs = [job for _, job in parity.fixture_jobs(1000, None)]
    picks = [jobs[i] for i in (3, 110, 215, 320, 405, 545, 760, 765)[:clips]]            # gestures, none, walkers, fastest
    plans = [plan_clip((job, 0, None, True))["plan"] for job in picks]
    out: dict = {"device": dev, "host": host, "clips": len(plans)}
    try:
        ref_gpu = GpuRenderer(None, dev, kernel="off")
        ref = ref_gpu.render(plans, want_parts=True)
        synth = synth_paths_host if host else synth_paths_cuda
        got_gpu = GpuRenderer(None, dev, synth=synth, kernel="off")
        got = got_gpu.render(plans, want_parts=True)                    # includes compiling the kernel
        if not host:
            torch.cuda.synchronize()
        a, b = got["dynamic"].double(), ref["dynamic"].double()
        rel = float(torch.sqrt(((a - b) ** 2).mean((1, 2)) / (b ** 2).mean((1, 2)).clamp_min(1e-300)).max())
        snr = float((got["snr_db"] - ref["snr_db"]).abs().nan_to_num(0.0).max())
        times = {}
        for name, g in (("torch", ref_gpu), ("kernel", got_gpu)):
            t: dict = {}
            g.render(plans, timings=t)
            times[name] = 1e3 * t.get("echo paths", 0.0) / len(plans)
        out.update(rel=rel, snr_db=snr, ms_per_clip=times, ok=bool(rel < 2e-5 and snr < 1e-3),
                   route="plain C on the CPU" if host else cuda_kernel(torch.device(dev)).name)
        out["why"] = "" if out["ok"] else f"the kernel's echoes differ from the PyTorch path by {rel:.2e} (limit 2e-5)"
    except Exception as exc:
        out.update(ok=False, why=f"{type(exc).__name__}: {exc}")
    return out


def verdict_file(device: torch.device) -> Path:
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    key = hashlib.sha256(f"{source_hash()}|{torch.__version__}|{name}".encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"sonar_echo_kernel_{key}.json"


def usable(device: torch.device, log=print) -> tuple[bool, str]:
    """True if the kernel may be used on this GPU. The first call runs the probe in its OWN process (so a
    crash inside untested GPU code cannot take the training run with it); the answer is remembered."""
    if device.type != "cuda":
        return False, "not a CUDA device"
    vf = verdict_file(device)
    if not vf.exists():
        try:
            res = subprocess.run([sys.executable, "-m", "datagen.echo_kernel", "--device", str(device), "--write", str(vf)],
                                 cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, timeout=600)
            if not vf.exists():
                tail = (res.stderr or res.stdout).strip().splitlines()[-1:] or ["no output"]
                vf.write_text(json.dumps({"ok": False, "why": f"the probe process ended with code {res.returncode}: {tail[0][:300]}"}))
        except Exception as exc:
            vf.write_text(json.dumps({"ok": False, "why": f"the probe could not run: {type(exc).__name__}: {exc}"}))
    v = json.loads(vf.read_text())
    if v.get("ok"):
        return True, f"{v.get('route')}: echoes within {v.get('rel', 0):.1e} of the PyTorch path; " \
                     f"{v['ms_per_clip']['kernel']:.2f} ms per clip against {v['ms_per_clip']['torch']:.2f}"
    return False, str(v.get("why"))


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Check the fused echo kernel against the PyTorch path (see the file header)")
    ap.add_argument("--device", default="auto", help="auto (cuda if there is one, else the plain-C build) | cuda | cuda:0 | host")
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--write", help="also write the verdict to this file")
    args = ap.parse_args(argv)
    t0 = time.time()
    res = probe(args.device, args.clips)
    res["seconds"] = round(time.time() - t0, 1)
    if args.write:
        Path(args.write).write_text(json.dumps(res))
    if res["ok"]:
        print(f"KERNEL OK ({res['route']}, {res['clips']} clips): echoes within {res['rel']:.2e} of the PyTorch path; "
              f"echo loop {res['ms_per_clip']['kernel']:.2f} ms per clip against {res['ms_per_clip']['torch']:.2f} ms")
        return 0
    print(f"KERNEL NOT USED: {res['why']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

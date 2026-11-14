"""render_torch.py — the renderer of render.py, for many clips at once on a GPU (PyTorch).

Plain words
-----------
render.py computes one clip at a time with numpy on one CPU core (about 0.9 s per clip,
almost all of it in the loop over echo paths). This file computes exactly the same thing
for a whole batch of clips with PyTorch tensors, so an RTX 4090 does the arithmetic and
the CPU only draws the random numbers.

The work is split in two:

    make_plan(...)     (plan.py) CPU, cheap (about 20 ms): everything random about one clip. It runs
                       render.py's own random draws in render.py's own order (hum phase,
                       hall noise, gain wander), so a plan holds the same numbers the CPU
                       renderer would have used for that seed.
    render_batch(...)  GPU: turns a list of plans into microphone audio.

Because the random numbers are shared, the GPU audio equals the CPU audio for the same
seed up to rounding. scripts/check_gpu_parity.py measures that on a 1,000-clip fixture.

What is NOT bit-identical to render.py, and why it does not matter
------------------------------------------------------------------
* The sum over echo paths is added up in a different order (float32, as in render.py):
  differences of about 1e-6 of the echo.
* The default precision "ref" uses float64 for the FFTs that carry the loud direct signal,
  like render.py. With precision "fast" those run in float32: differences of about 1e-7 of
  the direct signal, and after 16-bit rounding a few samples in ten thousand land one step
  apart (see pick_precision for what that does to the pictures).
* Path lengths and carrier phases are float64 on CPU and CUDA in both modes (they carry
  the Doppler information). Apple MPS has no float64: there everything is float32, which
  is less exact, so MPS has looser limits in the parity check.

Read-only helper module: it returns tensors, it never touches files or audio devices.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy import fft as sfft
from scipy.interpolate import CubicSpline
from scipy.signal import resample_poly

from .plan import MARGIN, RESP_FIELDS as _RESP_FIELDS, ClipPlan, make_plan  # noqa: F401  (re-exported)
from .render import SNR_REF_HZ


# --------------------------------------------------------------------------
# Precision: which number format each stage uses on this device
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Precision:
    name: str                     # "ref" | "fast" | "single"
    geom: torch.dtype             # path lengths, carrier phase, envelope read position
    sig: torch.dtype              # the big FFTs (direct path, microphone colouration, clock warp)

    @property
    def csig(self) -> torch.dtype:
        return torch.complex128 if self.sig == torch.float64 else torch.complex64


def pick_precision(device: torch.device, name: str = "auto") -> Precision:
    """Which number format each stage uses.

    ref (the default on CUDA and CPU): float64 for the three FFT steps that carry the loud direct signal
        (room, microphone colouration, clock warp) and for path lengths; float32 for the echo sum, as render.py.
        On a 48-clip sample every clip had the same bursts and labels as the CPU path.
    fast: those FFTs in float32. A few samples in ten thousand round the other way, and about 1 clip in 25
        gets a burst boundary moved by a frame. Use it only if it is much faster on your GPU (scripts/bench_gpu.py).
    single: everything float32. The only choice on Apple MPS (no float64 there); least exact.
    """
    if device.type == "mps" or name == "single":
        return Precision("single", torch.float32, torch.float32)
    if name in ("auto", "ref"):
        return Precision("ref", torch.float64, torch.float64)
    if name == "fast":
        return Precision("fast", torch.float64, torch.float32)
    raise ValueError(f"unknown precision {name!r}; choose ref, fast or single")


# --------------------------------------------------------------------------
# Tables shared by every clip of a run
# --------------------------------------------------------------------------
class RenderContext:
    """Per (speaker signal, clip length, pre-roll): spectra, carrier table, band grids, spline matrices."""

    def __init__(self, tx: np.ndarray, fs: int, n_out: int, pre: int, band: tuple, device: torch.device | str,
                 precision: str = "auto", period: int | None = None, chunk_elems: int | None = None) -> None:
        self.device = torch.device(device)
        self.prec = pick_precision(self.device, precision)
        dev, gd, sd, cd = self.device, self.prec.geom, self.prec.sig, self.prec.csig
        tx = np.asarray(tx, np.float64)
        self.fs, self.n_out, self.pre, self.band = int(fs), int(n_out), int(pre), (float(band[0]), float(band[1]))
        self.n_total = n_total = tx.shape[0]
        self.period = period
        self.f0 = f0 = 0.5 * (band[0] + band[1])
        # paths x baseband samples handled per step of the echo loop (memory against speed)
        self.chunk_elems = chunk_elems or {"cuda": 12_000_000, "mps": 8_000_000}.get(dev.type, 1_500_000)
        self.hd = torch.float32 if dev.type == "mps" else torch.float64        # per-clip sums (MPS has no float64)
        self.smooth_tables: dict = {}                                          # _device_noise: spline and babble tables

        freqs = sfft.rfftfreq(n_total, 1.0 / fs)
        self.n_freq = len(freqs)
        self.freqs = torch.from_numpy(freqs).to(dev, gd)
        self.txf = torch.from_numpy(np.stack([sfft.rfft(tx[:, c]) for c in range(tx.shape[1])])).to(dev, cd)
        # exp(+j 2 pi f0 n / fs): render.py's demodulator is its conjugate, its carrier a shifted window of it
        cyc = np.mod(np.arange(n_total, dtype=np.float64) * (f0 / fs), 1.0)
        base = np.exp(2j * np.pi * cyc)
        self.base_cyc = torch.from_numpy(cyc).to(dev, sd)
        self.base_re = torch.from_numpy(base.real.copy()).to(dev, sd)
        self.base_im = torch.from_numpy(base.imag.copy()).to(dev, sd)
        self.base32_re, self.base32_im = self.base_re.to(torch.float32), self.base_im.to(torch.float32)

        # static part: band of the probe signal, and the 2 Hz grid the room response is evaluated on
        band_mask = (freqs > band[0] - 1500.0) & (freqs < band[1] + 1500.0)
        fb = freqs[band_mask]
        step = max(1, int(round(2.0 / (fs / n_total))))
        fc = np.append(fb[::step], fb[-1] + (fb[1] - fb[0]) * step)
        j = np.arange(len(fb)) // step
        self.band_idx = torch.from_numpy(np.nonzero(band_mask)[0]).to(dev)
        self.fb = torch.from_numpy(fb).to(dev, gd)
        self.fc = torch.from_numpy(fc).to(dev, gd)
        self.fc_j = torch.from_numpy(j).to(dev)
        self.fc_dx = torch.from_numpy(fb - fc[j]).to(dev, gd)                 # x - xp[j]
        self.fc_w = torch.from_numpy(fc[j + 1] - fc[j]).to(dev, gd)           # xp[j+1] - xp[j]
        self._fb_np = fb

        f_out = sfft.rfftfreq(n_out, 1.0 / fs)
        self.f_out = torch.from_numpy(f_out).to(dev, gd)
        self.in_band = torch.from_numpy(np.nonzero((f_out > band[0] - 500.0) & (f_out < band[1] + 500.0))[0]).to(dev)
        self.k_out = torch.arange(n_out, device=dev, dtype=gd)
        if period:                                           # background overlay: bins between the probe lines
            quiet = np.ones(len(f_out), bool)
            for line in np.arange(0.0, fs / 2, fs / period):
                quiet &= np.abs(f_out - line) > 6.0
            quiet &= (f_out > band[0] - 400.0) & (f_out < band[1] + 400.0)
            self.quiet = torch.from_numpy(np.nonzero(quiet)[0]).to(dev)
        self._splines: dict = {}
        self._grids: dict = {}
        self._taps: dict = {}

    # ---- lazily built tables -------------------------------------------------
    def spline(self, t_ctrl: np.ndarray) -> dict:
        """Knot-slope matrix of scipy's CubicSpline on this time axis (so the GPU spline IS scipy's spline)."""
        key = (len(t_ctrl), float(t_ctrl[0]), float(t_ctrl[-1]))
        if key not in self._splines:
            n = len(t_ctrl)
            cs = CubicSpline(t_ctrl, np.eye(n), axis=0)
            m = np.concatenate([cs.c[2], cs(t_ctrl[-1:], 1)], axis=0)          # [n, n]: slopes at the knots = m @ values
            gd = self.prec.geom
            self._splines[key] = {"t": t_ctrl, "m": torch.from_numpy(m).to(self.device, gd),
                                  "dx": torch.from_numpy(np.diff(t_ctrl)).to(self.device, gd)}
        return self._splines[key]

    def grid(self, t_ctrl: np.ndarray, dec: int) -> dict:
        """Baseband time grid for one decimation factor: which spline piece each sample falls in."""
        key = (len(t_ctrl), float(t_ctrl[0]), float(t_ctrl[-1]), dec)
        if key not in self._grids:
            nb = int(np.ceil(self.n_out / dec)) + 2 * MARGIN
            kb = np.arange(nb, dtype=np.float64) - MARGIN
            tb = kb * dec / self.fs
            t_clip = np.clip(tb, t_ctrl[0], t_ctrl[-1])
            piece = np.clip(np.searchsorted(t_ctrl, t_clip, side="right") - 1, 0, len(t_ctrl) - 2)
            gd = self.prec.geom
            self._grids[key] = {
                "nb": nb, "tb": tb,
                "piece": torch.from_numpy(piece).to(self.device),
                "dt": torch.from_numpy(t_clip - t_ctrl[piece]).to(self.device, gd),
                "kdec": torch.from_numpy(kb * dec).to(self.device, gd),
                "kdec_int": torch.from_numpy((kb * dec).astype(np.int64)).to(self.device),
            }
        return self._grids[key]

    def taps(self, dec: int) -> torch.Tensor:
        """scipy.signal.resample_poly's interpolation filter for this factor, as a conv1d weight [dec, 1, 21]."""
        if dec not in self._taps:
            probe = np.zeros(41, np.complex64)
            probe[20] = 1.0
            y = resample_poly(probe, dec, 1)                                  # whatever taps this scipy version uses
            half = 10 * dec
            h = np.real(y[20 * dec - half:20 * dec + half + 1]).astype(np.float64)   # h[half + j] multiplies x[q] at output q*dec + j
            bank = np.zeros((21, dec))
            for j in range(21):                                               # window position j holds x[q + j - 10]
                for r in range(dec):
                    i = half + r + (10 - j) * dec
                    if 0 <= i < len(h):
                        bank[j, r] = h[i]
            self._taps[dec] = torch.from_numpy(bank.T.copy()[:, None, :]).to(self.device, torch.float32)   # [dec, 1, 21]
        return self._taps[dec]


# --------------------------------------------------------------------------
# Small batched helpers
# --------------------------------------------------------------------------
def _eval_responses(packs: list, f: torch.Tensor, gd: torch.dtype, cd: torch.dtype, chunk: int = 96) -> torch.Tensor:
    """hardware.Response.__call__ / background.MeasuredResponse.__call__ for many curves: [R, F] complex."""
    dev = f.device
    out = torch.empty((len(packs), f.shape[0]), dtype=cd, device=dev)
    plain = [i for i, p in enumerate(packs) if p[0] == "resp"]
    for a in range(0, len(plain), chunk):
        idx = plain[a:a + chunk]
        p = torch.from_numpy(np.stack([packs[i][1] for i in idx])).to(dev, gd)          # [r, 13]
        col = {k: p[:, j:j + 1] for j, k in enumerate(_RESP_FIELDS)}

        def mag_db(x: torch.Tensor) -> torch.Tensor:
            d = col["tilt_db_per_khz"] * (x - col["f_ref"]) / 1000.0
            d = d + col["shelf_db"] / (1.0 + torch.exp(-(x - col["shelf_f"]) / col["shelf_width"].clamp_min(1.0)))
            d = d + col["ripple_db"] * torch.cos(2 * math.pi * x / col["ripple_period"] + col["ripple_phase"])
            return d - 10.0 * torch.log10(1.0 + (x.clamp_min(0.0) / col["lp_f"]) ** (2 * col["lp_order"]))

        db = mag_db(f[None, :]) - mag_db(col["f_ref"]) + col["gain_db"]
        mag = 10.0 ** (db.clamp(-120.0, 40.0) / 20.0)
        ang = -2 * math.pi * f[None, :] * col["delay_s"] + col["phase_rad"]
        out[idx] = torch.complex(mag * torch.cos(ang), mag * torch.sin(ang)).to(cd)
    for i, p in enumerate(packs):
        if p[0] != "table":
            continue
        xp = torch.from_numpy(p[1]).to(dev, gd)
        fp = torch.from_numpy(p[2]).to(dev, gd)
        j = (torch.searchsorted(xp, f, right=True) - 1).clamp(0, len(xp) - 2)             # np.interp, clamped ends
        g = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]) * (f - xp[j]) + fp[j]
        g = torch.where(f <= xp[0], fp[0], torch.where(f >= xp[-1], fp[-1], g)) + p[3]
        out[i] = (10.0 ** (g / 20.0)).to(cd)
    return out


def _directivity(cos_t: torch.Tensor, n: torch.Tensor, floor: torch.Tensor) -> torch.Tensor:
    return torch.where(n > 0, floor + (1.0 - floor) * cos_t.clamp_min(0.0) ** n, torch.ones_like(cos_t))


def _frac_cycles(x: torch.Tensor) -> torch.Tensor:
    return x - torch.floor(x)


# --------------------------------------------------------------------------
# The echo paths (the hot loop of render.py), batched over scatterers
# --------------------------------------------------------------------------
def spline_coefs(values: torch.Tensor, sp: dict) -> torch.Tensor:
    """scipy CubicSpline pieces for values [N, Tc, D]: [N, Tc-1, 4, D], highest power first (as PPoly.c)."""
    m, dx = sp["m"], sp["dx"][None, :, None]
    v = values.to(m.dtype)
    s = torch.einsum("ts,nsd->ntd", m, v)                                      # slopes at the knots
    slope = (v[:, 1:] - v[:, :-1]) / dx
    t_ = (s[:, :-1] + s[:, 1:] - 2.0 * slope) / dx
    return torch.stack([t_ / dx, (slope - s[:, :-1]) / dx - t_, s[:, :-1], v[:, :-1]], dim=2)


def _spline_eval(values: torch.Tensor, sp: dict, grid: dict, out_dtype: torch.dtype) -> torch.Tensor:
    """scipy CubicSpline(t, values, axis=1)(tb) for values [N, Tc, D] -> [N, K, D]."""
    c = spline_coefs(values, sp)
    piece, dt = grid["piece"], grid["dt"][None, :, None]
    if out_dtype != c.dtype:                                                   # amplitudes: float32 is plenty
        c, dt = c.to(out_dtype), dt.to(out_dtype)
    out = c[:, :, 0].index_select(1, piece)
    for q in (1, 2, 3):
        out = out * dt + c[:, :, q].index_select(1, piece)
    return out


def add_path(ctx: RenderContext, sub: dict, env_flat: torch.Tensor, acc: torch.Tensor, row: torch.Tensor, spk: int,
             tau: torch.Tensor, amp: torch.Tensor) -> None:
    """render.add_path: exact carrier phase, linearly interpolated envelope. acc [n, nb, 2] += one path's echo."""
    n_total = ctx.n_total
    if ctx.prec.geom == torch.float64:
        idx = sub["base"][row] - tau * float(ctx.fs)                           # read position in the speaker buffer
        i0f = torch.floor(idx)
        frac = (idx - i0f).to(torch.float32)[..., None]
        i0 = i0f.long()
    else:                        # float32 cannot hold position + fraction: keep the whole samples as integers
        off = -(tau * float(ctx.fs))
        i0f = torch.floor(off)
        frac = (off - i0f)[..., None]
        i0 = sub["base_int"][row] + i0f.long()
    flat = (sub["env_row"][row][:, None] * 2 + spk) * n_total + i0.clamp_(0, n_total - 2)
    e = env_flat[flat] * (1.0 - frac) + env_flat[flat + 1] * frac
    ph = _frac_cycles(ctx.f0 * tau).to(torch.float32) * (-2.0 * math.pi)
    cr, si = amp * torch.cos(ph), amp * torch.sin(ph)
    acc[..., 0] += cr * e[..., 0] - si * e[..., 1]
    acc[..., 1] += cr * e[..., 1] + si * e[..., 0]


def wobble_paths(ctx: RenderContext, sub: dict, env_ri: torch.Tensor, dyn: torch.Tensor) -> None:
    """Lid wobble (desk bumps): the direct speaker->microphone path itself moves a little. Adds to dyn in place."""
    if sub["wob_clip"] is None:
        return
    row, nb, env_flat = sub["wob_clip"], sub["grid"]["nb"], env_ri.reshape(-1, 2)
    w = _spline_eval(sub["wobble"][..., None], sub["spline"], sub["grid"], ctx.prec.geom)[..., 0]       # [n_w, nb]
    for m in range(sub["n_mic"]):
        acc = torch.zeros((row.shape[0], nb, 2), dtype=torch.float32, device=ctx.device)
        for s in range(2):
            tau = sub["wob_tau0"][:, m, s][:, None] + w / sub["c"][row][:, None]
            amp = (sub["wob_amp"][:, m, s] * sub["spk_on"][row, s]).to(torch.float32)[:, None].expand(-1, nb)
            add_path(ctx, sub, env_flat, acc, row, s, tau, amp)
        dyn[:, m].index_add_(0, row, acc)                                      # one wobble per clip: no order to keep


def synth_paths_torch(ctx: RenderContext, sub: dict, env_ri: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of all moving echo paths at the baseband rate, for the clips of one decimation factor.

    sub: tensors of those clips (see _gather_sub). env_ri: [B_all, 2, n_total, 2] float32 envelopes.
    Returns (dyn_bb, key_bb), each [b, n_mic, nb, 2] float32 (real, imaginary).
    """
    dev, gd = ctx.device, ctx.prec.geom
    grid, sp = sub["grid"], sub["spline"]
    nb, n_mic, b = grid["nb"], sub["n_mic"], sub["b"]
    dyn = torch.zeros((b, n_mic, nb, 2), dtype=torch.float32, device=dev)
    key = torch.zeros((b, n_mic, nb, 2), dtype=torch.float32, device=dev)
    env_flat = env_ri.reshape(-1, 2)
    n_scat = sub["pos"].shape[0]
    per = max(1, ctx.chunk_elems // nb)

    def accumulate(total: torch.Tensor, a: int, acc: torch.Tensor) -> None:
        """total[clip] += acc[scatterer], one reflector of each clip at a time and in the scene's order.

        A plain index_add_ would add several reflectors of one clip in whatever order the GPU schedules them;
        float32 sums then differ in the last bit from run to run. This way a clip comes out the same whatever
        else is in the batch.
        """
        rank, clip = sub["scat_rank_np"][a:a + acc.shape[0]], sub["scat_clip_np"][a:a + acc.shape[0]]
        for r in np.unique(rank):
            pick = np.nonzero(rank == r)[0]
            if len(pick) == len(rank):
                total[torch.from_numpy(clip).to(dev)] += acc
            else:
                total[torch.from_numpy(clip[pick]).to(dev)] += acc[torch.from_numpy(pick).to(dev)]

    for a in range(0, n_scat, per):
        sl = slice(a, min(a + per, n_scat))
        row = sub["scat_clip"][sl]                                             # which clip each scatterer belongs to
        cl = row[:, None]
        p = _spline_eval(sub["pos"][sl], sp, grid, gd)                         # [n, nb, 3]
        rho = _spline_eval(sub["rho"][sl][..., None], sp, grid, torch.float32)[..., 0]
        spec_on = sub["use_spec"][sl]
        any_spec = bool(sub["use_spec_np"][sl].any())
        nrm = _spline_eval(sub["nrm"][sl], sp, grid, torch.float32) if any_spec else None
        kappa = (sub["alpha"][cl] * (math.log(10.0) / 20.0))
        gamma = sub["gamma"][cl]

        def leg(point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Distance (exact), unit vector and range loss from each reflector position to one fixed point."""
            d = point[:, None, :] - p
            r = torch.sqrt((d * d).sum(-1)).clamp_min(0.03)
            r32 = r.to(torch.float32)
            return r, d.to(torch.float32) / r32[..., None], r32 ** (-gamma) * torch.exp(-kappa * r32)

        tx_side = []
        for s in range(2):
            r, u, loss = leg(sub["spk"][row, s])
            g = _directivity(-(u * sub["spk_axis"][row, s][:, None, :]).sum(-1), sub["spk_n"][cl], sub["spk_floor"][cl])
            tx_side.append((r, u, g * loss * sub["spk_on"][row, s][:, None]))
        base_amp = sub["k_amp"][cl] * rho
        key_w = sub["key"][sl].to(torch.float32)[:, None, None]
        for m in range(n_mic):
            mic = sub["mic"][row, m]
            r_d, u_d, loss_d = leg(mic)
            g_rx = _directivity(-(u_d * sub["mic_axis"][row][:, None, :]).sum(-1), sub["mic_n"][cl], sub["mic_floor"][cl])
            r_i, u_i, loss_i = leg(mic * sub["mirror"])                        # the same microphone mirrored in the desk
            acc = torch.zeros((row.shape[0], nb, 2), dtype=torch.float32, device=dev)
            for r_rx, u_rx, rx_gain in ((r_d, u_d, g_rx * loss_d), (r_i, u_i, g_rx * loss_i * sub["table_refl"][cl])):
                for s in range(2):
                    r_tx, u_tx, tx_gain = tx_side[s]
                    amp = base_amp * tx_gain * rx_gain
                    if any_spec:                                               # mirror-like parts: palm, chest
                        bis = u_tx + u_rx
                        bis = bis / torch.sqrt((bis * bis).sum(-1)).clamp_min(1e-9)[..., None]
                        cosb = (nrm * bis).sum(-1).abs()
                        shine = sub["spec_floor"][sl][:, None] + (1.0 - sub["spec_floor"][sl][:, None]) * cosb ** sub["spec_q"][sl][:, None]
                        amp = amp * torch.where(spec_on[:, None], shine, torch.ones_like(shine))
                    add_path(ctx, sub, env_flat, acc, row, s, (r_tx + r_rx) / sub["c"][cl], amp)
            accumulate(dyn[:, m], a, acc)
            accumulate(key[:, m], a, acc * key_w)

    wobble_paths(ctx, sub, env_ri, dyn)
    return dyn, key


# --------------------------------------------------------------------------
# The whole renderer for one batch
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Hall noise and gain wander finished on the device (the other half of plan._noise_split / _gain_split)
# --------------------------------------------------------------------------
def _smooth_full(ctx: "RenderContext", ctrl: torch.Tensor, rate: float) -> torch.Tensor:
    """The last steps of physics.smooth_noise for a batch: low-rate samples [B, m] -> [B, n_out], zero mean, unit variance.

    The spline is scipy's CubicSpline (not-a-knot) through samples at k / rate, evaluated at every audio sample.
    """
    n, fs = ctx.n_out, float(ctx.fs)
    if rate < fs:
        m = ctrl.shape[1]
        key = (m, rate, ctrl.dtype)
        if key not in ctx.smooth_tables:
            from scipy.interpolate import CubicSpline

            x = np.arange(m) / rate
            slopes = CubicSpline(x, np.eye(m)).derivative()(x)      # row k: the slope at knot k as a mix of the samples
            t = np.arange(n) / fs
            piece = np.clip(np.floor(t * rate).astype(np.int64), 0, m - 2)
            ctx.smooth_tables[key] = (torch.from_numpy(slopes).to(ctx.device, ctrl.dtype), torch.from_numpy(piece).to(ctx.device),
                                      torch.from_numpy(t - x[piece]).to(ctx.device, ctrl.dtype))
        slopes, piece, dt = ctx.smooth_tables[key]
        dx = 1.0 / rate
        s = ctrl @ slopes.T
        v0, s0 = ctrl[:, :-1], s[:, :-1]
        slope = (ctrl[:, 1:] - v0) / dx
        tt = (s0 + s[:, 1:] - 2.0 * slope) / dx
        y = (tt / dx).index_select(1, piece)                         # scipy's pieces, highest power first
        for c in ((slope - s0) / dx - tt, s0, v0):
            y = y * dt + c.index_select(1, piece)
    else:
        y = ctrl
    y = y - y.mean(dim=1, keepdim=True)
    return y / (y.std(dim=1, keepdim=True, unbiased=False) + 1e-12)


def _device_noise(ctx: "RenderContext", plans: list, noise: torch.Tensor) -> tuple:
    """For plans made with shape_on_device: add the shaped hall babble to `noise` [B, n_mic, n_out] and build the gain [B, n_out].

    The arithmetic is render._noise's and render._gain_track's, on the numbers the workers drew (float64 on CUDA:
    it agrees with the CPU functions to about 1e-14).
    """
    dev, dtype, n, fs = ctx.device, noise.dtype, ctx.n_out, float(ctx.fs)
    b, n_mic = len(plans), noise.shape[1]
    gain = torch.empty((b, n), device=dev, dtype=dtype)
    if ("babble", dtype) not in ctx.smooth_tables:
        f = torch.fft.rfftfreq(n, 1.0 / fs, device=dev, dtype=dtype)
        shape = 1.0 / torch.sqrt(1.0 + (f / 400.0) ** 2) / (1.0 + (f / 3500.0) ** 2)
        shape[0] = 0.0
        ctx.smooth_tables[("babble", dtype)] = (shape, torch.arange(n, device=dev, dtype=dtype))
    shape, k = ctx.smooth_tables[("babble", dtype)]

    def rows(part: list, name: str, width: int) -> torch.Tensor:      # clips without that piece get zeros (their level is 0)
        got = [getattr(p, name) for p in part]
        return torch.from_numpy(np.stack([np.zeros(width) if a is None else np.asarray(a) for a in got])).to(dev, dtype)

    for a in range(0, b, 48):                                         # in pieces: every temporary here is [clips, n_out] float64
        part = plans[a:a + 48]
        with_babble = [p for p in part if p.babble is not None]
        if with_babble:
            w = torch.fft.irfft(torch.fft.rfft(rows(part, "babble", n), dim=1) * shape, n=n, dim=1)
            env = _smooth_full(ctx, rows(part, "babble_env", len(with_babble[0].babble_env)), with_babble[0].babble_par[0])
            w = w * (0.35 + env.abs())
            del env
            level = torch.tensor([p.babble_par[1] for p in part], device=dev, dtype=dtype)
            w = w * (level / (w.std(dim=1, unbiased=False) + 1e-12))[:, None]
            fac = torch.tensor([list(p.babble_par[2]) or [0.0] * n_mic for p in part], device=dev, dtype=dtype)
            noise[a:a + len(part)] += w[:, None, :] * fac[:, :, None]
            del w
        g_db = torch.zeros((len(part), n), device=dev, dtype=dtype)
        with_wander = [p for p in part if p.gain_ctrl is not None]
        if with_wander:
            drift = torch.tensor([p.gain_par[1] if p.gain_ctrl is not None else 0.0 for p in part], device=dev, dtype=dtype)
            g_db += drift[:, None] * _smooth_full(ctx, rows(part, "gain_ctrl", len(with_wander[0].gain_ctrl)), with_wander[0].gain_par[0])
        if any(p.gain_par[2] >= 0 for p in part):
            k0 = torch.tensor([p.gain_par[2] for p in part], device=dev)
            step = torch.tensor([p.gain_par[3] if p.gain_par[2] >= 0 else 0.0 for p in part], device=dev, dtype=dtype)
            g_db += step[:, None] * ((k[None, :] - k0[:, None].to(dtype)) / (0.02 * fs)).clamp_(0.0, 1.0)
        gain[a:a + len(part)] = torch.pow(10.0, g_db / 20.0)
        del g_db
    return noise, gain


def _stack(plans: list, name: str, dev: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.from_numpy(np.stack([np.asarray(getattr(p, name)) for p in plans])).to(dev, dtype)


def _vec(plans: list, name: str, dev: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([getattr(p, name) for p in plans], device=dev, dtype=dtype)


def render_batch(ctx: RenderContext, plans: list, want_parts: bool = False, synth=None, timings: dict | None = None) -> dict:
    """Microphone audio for a list of ClipPlan (all with ctx's clip length, pre-roll and static/noise setting).

    Returns a dict of tensors on ctx.device:
        rx       [B, n_out, n_mic] float32: what render() returns as .rx (mixed with the real excerpt for overlays)
        snr_db   [B] float64: echo_snr_db (NaN where render() gives NaN)
        echo     [B, n_out, n_mic] float32, overlays only: the synthetic echo alone (bg.inject's out.rx)
        static / dynamic / converter_in [B, n_mic, n_out], only with want_parts: render()'s parts
    `synth` replaces synth_paths_torch (e.g. the CUDA kernel); same arguments and result.
    `timings`, if given, collects seconds per stage (the device is made to finish each stage first,
    which costs a little speed: scripts/bench_gpu.py uses it, normal runs do not).
    """
    if not plans:
        raise ValueError("no plans")
    dev, prec = ctx.device, ctx.prec
    gd, sd, cd = prec.geom, prec.sig, prec.csig
    p0 = plans[0]
    b, n_out, n_total, pre, fs = len(plans), ctx.n_out, ctx.n_total, ctx.pre, ctx.fs
    n_mic = p0.mic.shape[0]
    for p in plans:
        if (p.n_out, p.n_total, p.pre, p.include_static, p.add_noise, p.mic.shape[0], p.shaped) != \
                (n_out, n_total, pre, p0.include_static, p0.add_noise, n_mic, p0.shaped):
            raise ValueError("plans of different clip types in one batch")
    include_static, add_noise = p0.include_static, p0.add_noise
    synth = synth or synth_paths_torch
    t_last = [time.perf_counter()]

    def tick(stage: str) -> None:
        if timings is None:
            return
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elif dev.type == "mps":
            torch.mps.synchronize()
        now = time.perf_counter()
        timings[stage] = timings.get(stage, 0.0) + now - t_last[0]
        t_last[0] = now

    # ---- 1. speaker colouration + complex envelope of each speaker signal ----
    spk_col = torch.tensor([p.spk_col for p in plans], device=dev)                          # [B, 2]
    spk_on = spk_col >= 0
    h_spk = _eval_responses([r for p in plans for r in p.spk_resp], ctx.freqs, gd, cd).view(b, 2, -1)
    spec = ctx.txf[spk_col.clamp_min(0)] * h_spk * spk_on[..., None].to(sd)                 # [B, 2, n_freq]
    del h_spk
    spec_band = spec.index_select(2, ctx.band_idx) if include_static else None
    spec = spec.to(torch.complex64)                          # render.py stores the envelope as complex64: a float32 FFT is enough
    full = torch.zeros((b, 2, n_total), dtype=torch.complex64, device=dev)
    full[..., :ctx.n_freq] = 2.0 * spec
    full[..., 0] = spec[..., 0]
    if n_total % 2 == 0:
        full[..., ctx.n_freq - 1] = spec[..., -1]
    del spec
    env = torch.fft.ifft(full, dim=-1)
    del full
    env_ri = torch.empty((b, 2, n_total, 2), dtype=torch.float32, device=dev)               # env * conj(base)
    env_ri[..., 0] = env.real * ctx.base32_re + env.imag * ctx.base32_im
    env_ri[..., 1] = env.imag * ctx.base32_re - env.real * ctx.base32_im
    del env
    tick("speaker envelope (FFT)")

    # ---- 2. moving reflectors at the baseband rate, per decimation factor ----
    pre_eff = torch.tensor([p.pre - p.latency for p in plans], device=dev)                  # [B] int
    hd = ctx.hd
    scale = torch.tensor([10.0 ** (p.direct_gain_db / 20.0) for p in plans], device=dev, dtype=hd)
    dyn = torch.zeros((b, n_mic, n_out), dtype=sd, device=dev)
    p_key = torch.zeros(b, dtype=hd, device=dev)
    has_key = torch.zeros(b, dtype=torch.bool, device=dev)
    k_out = torch.arange(n_out, device=dev)
    for dec in sorted({p.dec for p in plans}):
        rows = [i for i, p in enumerate(plans) if p.dec == dec]
        sub = _gather_sub(ctx, plans, rows, dec, pre_eff, spk_on)
        dyn_bb, key_bb = synth(ctx, sub, env_ri)                                            # [b, n_mic, nb, 2]
        tick("echo paths")
        nb = sub["grid"]["nb"]
        ridx = torch.tensor(rows, device=dev)
        # how audible is the labelled motion: mean power of its echo during the event (render()'s p_key)
        tb = sub["grid"]["tb"]
        sel_np = np.zeros((len(rows), nb), bool)
        for j, i in enumerate(rows):
            for t0, t1 in (plans[i].events or [(0.0, n_out / fs)]):
                sel_np[j] |= (tb >= t0) & (tb <= t1)
        sel = torch.from_numpy(sel_np).to(dev)
        pw = (key_bb.to(hd) ** 2).sum(-1)                                                   # [b, n_mic, nb]
        cnt = sel.sum(-1).clamp_min(1).to(hd) * n_mic
        p_key[ridx] = 0.5 * (pw * sel[:, None, :]).sum((1, 2)) / cnt * scale[ridx] ** 2
        has_key[ridx] = sel.any(-1) & (key_bb != 0).flatten(1).any(-1)
        del key_bb, pw
        # upsample to the audio rate (scipy.signal.resample_poly's filter) and put it back on the carrier
        if dec > 1:
            x = dyn_bb.permute(0, 1, 3, 2).reshape(-1, 1, nb)                               # one row per (clip, mic, re/im)
            up = torch.nn.functional.conv1d(x, ctx.taps(dec), padding=10)                   # [rows, dec, nb]: output phase r
            up = up.transpose(1, 2).reshape(len(rows), n_mic, 2, nb * dec)
        else:
            up = dyn_bb.permute(0, 1, 3, 2)
        up = up[..., MARGIN * dec:MARGIN * dec + n_out].to(sd)
        ci = pre_eff[ridx][:, None] + k_out[None, :]                                        # carrier sample of each output
        dyn[ridx] = up[:, :, 0] * ctx.base_re[ci][:, None, :] - up[:, :, 1] * ctx.base_im[ci][:, None, :]
        del dyn_bb, up, sub
        tick("upsample + carrier")
    del env_ri

    # ---- 3. direct leakage + static reflections (frequency domain, as render.py) ----
    static = None
    if include_static:
        static = _static(ctx, plans, spec_band)
        clean = static + dyn
        tick("direct sound + room (FFT)")
    else:
        clean = dyn if not want_parts else dyn.clone()
    out: dict = {}
    if want_parts:
        out["dynamic"] = dyn
        out["static"] = static if static is not None else torch.zeros_like(dyn)

    # ---- 4. mains hum on the speaker amplifier, microphone colouration ----
    depth_host = [2.0 * 10.0 ** (p.hum_db / 20.0) if p.hum_hz > 0 else 0.0 for p in plans]
    if any(d > 0 for d in depth_host):
        depth = torch.tensor(depth_host, device=dev, dtype=sd)[:, None]
        ph = _stack(plans, "hum_ph", dev, sd)
        cyc = _frac_cycles(ctx.k_out[None, :] * (_vec(plans, "hum_hz", dev, gd)[:, None] / fs))   # hum cycles, reduced exactly
        a1, a2 = (2 * math.pi * cyc).to(sd), (4 * math.pi * cyc).to(sd)
        clean = clean * (1.0 + depth * torch.cos(a1 + ph[:, :1]) + 0.4 * depth * torch.cos(a2 + ph[:, 1:]))[:, None, :]
    h_mic = _eval_responses([r for p in plans for r in p.mic_resp], ctx.f_out, gd, cd).view(b, n_mic, -1)
    clean = torch.fft.irfft(torch.fft.rfft(clean, dim=-1) * h_mic, n=n_out, dim=-1)
    del h_mic

    # ---- 5. clock mismatch between the speaker and microphone devices ----
    warp = [i for i, p in enumerate(plans) if p.clock_ppm != 0.0]
    if warp:
        widx = torch.tensor(warp, device=dev)
        clean[widx] = _clock_warp(ctx, clean[widx], torch.tensor([plans[i].clock_ppm for i in warp], device=dev, dtype=gd))
    tick("microphone + clock (FFT)")

    # ---- 6. converter units, noise, gain wander, clipping, rounding ----
    # from here on float64 where the device has it (cheap, and it keeps the rounding step exact)
    rx = clean.to(hd) * scale[:, None, None]
    del clean
    snr = torch.full((b,), float("nan"), dtype=hd, device=dev)
    if add_noise:
        noise = _stack(plans, "noise", dev, hd)                                             # [B, n_mic, n_out]
        if p0.shaped:
            gain = _stack(plans, "gain", dev, hd)
        else:                                                 # the workers only drew the numbers: shape them here
            noise, gain = _device_noise(ctx, plans, noise)
        n_spec = torch.fft.rfft(noise.to(torch.float32), dim=-1).index_select(2, ctx.in_band)   # only a level: float32 is plenty
        n0 = 2.0 * (n_spec.real.to(hd) ** 2 + n_spec.imag.to(hd) ** 2).mean((1, 2)) / (n_out * fs)
        del n_spec
        snr = torch.where(has_key, 10.0 * torch.log10(p_key / (n0 * SNR_REF_HZ) + 1e-30), snr)
        rx = (rx + noise) * gain[:, None, :]
        del noise, gain
    if want_parts:
        out["converter_in"] = rx.clone()
    rx = rx.clamp_(-1.0, 1.0)
    if any(p.adc_bits > 0 for p in plans):
        bits = torch.tensor([p.adc_bits for p in plans], device=dev)
        q = (2.0 ** (bits.clamp_min(1) - 1).to(hd))[:, None, None]
        rx = torch.where((bits > 0)[:, None, None], torch.round(rx * q) / q, rx)
    if any(p.swap_mics for p in plans):
        swap = torch.tensor([p.swap_mics for p in plans], device=dev)
        rx = torch.where(swap[:, None, None], rx.flip(1), rx)
    rx = rx.transpose(1, 2).to(torch.float32)                                               # [B, n_out, n_mic]

    if p0.real is not None:                                   # overlay on a real recording (background.inject)
        real = torch.from_numpy(np.stack([p.real for p in plans])).to(dev)                  # [B, n_out, real_mics] float32
        out["echo"] = rx
        snr = _overlay_snr(ctx, plans, real, rx)
        rx = (real + rx[:, :, :real.shape[2]]).clamp_(-1.0, 1.0)
    out["rx"] = rx
    out["snr_db"] = snr
    tick("noise, gain, rounding")
    return out


def _gather_sub(ctx: RenderContext, plans: list, rows: list, dec: int, pre_eff: torch.Tensor, spk_on: torch.Tensor) -> dict:
    """Tensors for the clips `rows` (all with decimation `dec`), flattened over their scatterers."""
    dev, gd = ctx.device, ctx.prec.geom
    f32 = torch.float32
    sel = [plans[i] for i in rows]
    t_ctrl = sel[0].t_ctrl
    for p in sel:
        if len(p.t_ctrl) != len(t_ctrl) or p.t_ctrl[0] != t_ctrl[0] or p.t_ctrl[-1] != t_ctrl[-1]:
            raise ValueError("clips with different scene time axes in one batch")
    grid = ctx.grid(t_ctrl, dec)
    counts = [p.pos.shape[0] for p in sel]
    scat_clip = np.repeat(np.arange(len(sel)), counts)
    ridx = torch.tensor(rows, device=dev)

    def cat(name: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(np.concatenate([getattr(p, name) for p in sel], axis=0)).to(dev, dtype)

    sub = {
        "b": len(sel), "n_mic": sel[0].mic.shape[0], "grid": grid, "spline": ctx.spline(t_ctrl),
        "scat_clip": torch.from_numpy(scat_clip).to(dev), "scat_clip_np": scat_clip,
        "scat_rank_np": np.concatenate([np.arange(c) for c in counts]) if counts else np.zeros(0, np.int64),
        "env_row": ridx,                                                       # row of each clip in the full batch
        "base": grid["kdec"][None, :] + pre_eff[ridx].to(gd)[:, None],          # [b, nb] read position before the delay
        "base_int": grid["kdec_int"][None, :] + pre_eff[ridx][:, None],
        "pre_eff": pre_eff[ridx], "counts": counts,
        "pos": cat("pos", gd), "rho": cat("rho", f32), "nrm": cat("nrm", f32),
        "use_spec": cat("use_spec", torch.bool), "spec_q": cat("spec_q", f32), "spec_floor": cat("spec_floor", f32),
        "use_spec_np": np.concatenate([p.use_spec for p in sel]) if sel else np.zeros(0, bool),
        "key": cat("key", torch.bool),
        "spk": _stack(sel, "spk", dev, gd), "spk_axis": _stack(sel, "spk_axis", dev, f32),
        "mic": _stack(sel, "mic", dev, gd), "mic_axis": _stack(sel, "mic_axis", dev, f32),
        "mirror": torch.tensor([1.0, 1.0, -1.0], device=dev, dtype=gd),
        "spk_on": spk_on[ridx].to(f32),
        "c": _vec(sel, "c", dev, gd),
    }
    for name in ("alpha", "gamma", "k_amp", "spk_n", "spk_floor", "mic_n", "mic_floor", "table_refl"):
        sub[name] = _vec(sel, name, dev, f32)
    wob = [j for j, p in enumerate(sel) if p.wobble is not None]
    sub["wob_clip"] = None
    if wob:
        sub["wob_clip"] = torch.tensor(wob, device=dev)
        sub["wobble"] = torch.from_numpy(np.stack([sel[j].wobble for j in wob])).to(dev, gd)
        sub["wob_tau0"] = torch.from_numpy(np.stack([sel[j].wobble_tau0 for j in wob])).to(dev, gd)
        sub["wob_amp"] = torch.from_numpy(np.stack([sel[j].wobble_amp for j in wob])).to(dev, f32)
    return sub


def _static(ctx: RenderContext, plans: list, spec_band: torch.Tensor, chunk: int = 24) -> torch.Tensor:
    """Direct leakage + room reflections: [B, n_mic, n_out]. One fixed linear room per clip, as render.py."""
    dev, prec = ctx.device, ctx.prec
    gd, sd, cd = prec.geom, prec.sig, prec.csig
    b, n_mic = len(plans), plans[0].mic.shape[0]
    out = torch.empty((b, n_mic, ctx.n_out), dtype=sd, device=dev)
    n_tap = max(p.static_delays.shape[-1] for p in plans)
    delays = np.zeros((b, n_mic, 2, n_tap))
    amps = np.zeros((b, n_mic, 2, n_tap))
    for i, p in enumerate(plans):
        t = p.static_delays.shape[-1]
        delays[i, ..., :t] = p.static_delays
        delays[i, ..., t:] = p.static_delays[..., :1]
        amps[i, ..., :t] = p.static_amps
    d_ref = delays[..., 0]                                                                  # [B, n_mic, 2]
    rel_all = torch.from_numpy(delays - d_ref[..., None]).to(dev, gd)
    amps_all = torch.from_numpy(amps).to(dev, gd)
    any_all = torch.from_numpy(np.stack([p.static_any for p in plans])).to(dev)
    # exp(-j 2 pi fb d_ref): the bulk delay, applied exactly (phase reduced in float64 on the CPU for float32 devices)
    if gd == torch.float64:
        bulk = _frac_cycles(ctx.fb[None, None, None, :] * torch.from_numpy(d_ref).to(dev, gd)[..., None])
    else:
        bulk = torch.from_numpy(np.mod(ctx._fb_np[None, None, None, :] * d_ref[..., None], 1.0)).to(dev, gd)
    for a in range(0, b, chunk):
        sl = slice(a, min(a + chunk, b))
        theta = -2 * math.pi * ctx.fc[None, None, None, :, None] * rel_all[sl][:, :, :, None, :]   # [c, n_mic, 2, n_fc, T]
        w = amps_all[sl][:, :, :, None, :]
        h_re, h_im = (torch.cos(theta) * w).sum(-1), (torch.sin(theta) * w).sum(-1)
        del theta

        def interp(h: torch.Tensor) -> torch.Tensor:                                        # np.interp(fb, fc, h)
            lo, hi = h.index_select(-1, ctx.fc_j), h.index_select(-1, ctx.fc_j + 1)
            return (hi - lo) / ctx.fc_w * ctx.fc_dx + lo

        h = torch.complex(interp(h_re), interp(h_im))
        ang = -2 * math.pi * bulk[sl]
        h = h * torch.complex(torch.cos(ang), torch.sin(ang)) * any_all[sl][..., None]
        band = (spec_band[sl][:, None, :, :] * h.to(cd)).sum(2)                             # sum over speakers
        spec = torch.zeros((band.shape[0], n_mic, ctx.n_freq), dtype=cd, device=dev)
        spec[..., ctx.band_idx] = band
        out[sl] = torch.fft.irfft(spec, n=ctx.n_total, dim=-1)[..., ctx.pre:ctx.pre + ctx.n_out]
    return out


def _clock_warp(ctx: RenderContext, x: torch.Tensor, ppm: torch.Tensor) -> torch.Tensor:
    """render._clock_warp for x [b, n_mic, n]: resample the envelope, keep the carrier phase exact."""
    n = x.shape[-1]
    gd, sd = ctx.prec.geom, ctx.prec.sig
    eps = (ppm * 1e-6)[:, None]
    k = ctx.k_out[None, :]
    if gd == torch.float64:
        pos = (k * (1.0 + eps)).clamp(0, n - 1.001)
        i0f = torch.floor(pos)
        frac = (pos - i0f).to(sd)
        i0 = i0f.long()
        up_cyc = _frac_cycles(pos * (ctx.f0 / ctx.fs)).to(sd)
    else:                                   # float32 cannot hold sample positions: work with the small offset pos - k
        delta = torch.minimum(k * eps, (n - 1.001) - k).clamp_min(-k)
        d0 = torch.floor(delta)
        frac = (delta - d0).to(sd)
        i0 = (k + d0).long()
        up_cyc = _frac_cycles(ctx.base_cyc[:n][None, :] + delta * (ctx.f0 / ctx.fs)).to(sd)
    spec = torch.fft.fft(x.to(ctx.prec.csig), dim=-1)
    spec[..., n // 2 + 1:] = 0.0
    spec[..., 1:(n + 1) // 2] *= 2.0
    e = torch.fft.ifft(spec, dim=-1)
    del spec
    e_re = e.real * ctx.base_re[:n] + e.imag * ctx.base_im[:n]                              # e * exp(-j 2 pi f0 k / fs)
    e_im = e.imag * ctx.base_re[:n] - e.real * ctx.base_im[:n]
    del e
    i0x = i0[:, None, :].expand(-1, x.shape[1], -1)
    i1x = (i0 + 1).clamp_max(n - 1)[:, None, :].expand(-1, x.shape[1], -1)
    f = frac[:, None, :]
    w_re = e_re.gather(2, i0x) * (1.0 - f) + e_re.gather(2, i1x) * f
    w_im = e_im.gather(2, i0x) * (1.0 - f) + e_im.gather(2, i1x) * f
    ang = 2 * math.pi * up_cyc[:, None, :]
    return w_re * torch.cos(ang) - w_im * torch.sin(ang)


def _overlay_snr(ctx: RenderContext, plans: list, real: torch.Tensor, echo: torch.Tensor) -> torch.Tensor:
    """background.inject's yardstick: the added echo against the REAL noise between the probe lines, in 25 Hz."""
    dev, n, fs = ctx.device, ctx.n_out, ctx.fs
    f64 = ctx.hd
    spec = torch.fft.rfft(real.to(f64), dim=1).index_select(1, ctx.quiet)                   # [B, n_quiet, mics]
    pw = (spec.real ** 2 + spec.imag ** 2).flatten(1).sort(dim=1).values
    cnt = pw.shape[1]
    med = pw[:, cnt // 2] if cnt % 2 else 0.5 * (pw[:, cnt // 2 - 1] + pw[:, cnt // 2])     # np.median
    n0 = 2.0 * med / (n * fs) / 0.69
    k = torch.arange(n, device=dev)[None, :]
    sel = torch.zeros((len(plans), n), dtype=torch.bool, device=dev)
    for i, p in enumerate(plans):
        for t0, t1 in (p.events or [(0.0, n / fs)]):
            sel[i] |= (k[0] >= max(0, int(t0 * fs))) & (k[0] < max(0, int(t1 * fs)))
    cnt_s = sel.sum(1)
    p_echo = ((echo.to(f64) ** 2).sum(-1) * sel).sum(1) / (cnt_s.clamp_min(1) * echo.shape[2])
    snr = 10.0 * torch.log10(p_echo / (n0 * 25.0) + 1e-30)
    return torch.where((cnt_s > 0) & (p_echo > 0), snr, torch.full_like(snr, float("nan")))

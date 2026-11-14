"""features_torch.py — the gesture feature chain for many clips at once on a GPU (PyTorch).

Plain words
-----------
The live engine turns microphone audio into Doppler pictures with `Frontend`
(dsp.py) one block at a time on the CPU. Training needs the same pictures for
thousands of clips, so this file computes the gesture part of that chain for a
whole batch on the GPU:

    high-pass  ->  shift each pilot tone to 0 Hz and keep +-300 Hz  ->  remove the
    steady carrier  ->  one Doppler spectrum column per 20 ms (+ the left/right
    signal of the microphone pair)

and hands the columns to the SAME noise-floor normaliser and movement segmenter
classes the engine uses (gestures.py), so the decisions "a burst started / ended"
are made by the engine's own code.

It leaves out the chirp receiver, presence and breathing: they cost a quarter of
the CPU time of the feature step and the gesture pictures do not depend on them.

How it stays equal to dsp.py
----------------------------
* The high-pass and the tone band-pass are two FIR filters in a row; here they
  are one FIR (their convolution), evaluated only at the samples that are kept.
* The 5 Hz clutter high-pass is a recursive filter started in steady state. That
  equals filtering (signal - first sample) from rest, which is done exactly in
  the frequency domain with enough zero padding for its ringing to die out.
* Everything else is the same arithmetic in float64 (float32 on Apple MPS).
scripts/check_gpu_parity.py and tests/test_gpu_parity.py compare the two.

Needs PyTorch (training machine only). Read-only: no files, no audio devices.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy import fft as sfft
from scipy import signal as sps

from .config import SonarConfig
from .dsp import DopplerSTFT, ToneDemod, design_prefilter
from .features import StreamFeatures, stream_from_columns

_HP_SECONDS = 1.7         # zero padding after which the 5 Hz high-pass has rung out (< 1e-16): 3400 samples at 2 kHz


class TorchFrontend:
    """Audio [B, n, n_mics] -> Doppler columns, as dsp.Prefilter -> ToneDemod -> DopplerSTFT produce them."""

    def __init__(self, cfg: SonarConfig, n_mics: int | None = None, device: torch.device | str = "cpu",
                 precision: str = "auto", fir: str = "conv") -> None:
        """fir: how the long filter in front (prefilter and tone band-pass in one, then every 24th sample) is computed.
        "conv" = as a strided convolution (the default); "fft" = the same sum through FFTs (_decimate_fft):
        much less arithmetic on a GPU, same numbers to rounding. Apple GPUs always use "conv"."""
        n_mics = cfg.n_mics if n_mics is None else n_mics
        self.cfg = cfg if n_mics == cfg.n_mics else cfg.with_(n_mics=n_mics)
        cfg = self.cfg
        self.n_mics = n_mics
        self.device = dev = torch.device(device)
        single = dev.type == "mps" or precision == "single"
        self.rd = torch.float32 if single else torch.float64
        self.cd = torch.complex64 if single else torch.complex128

        # one FIR = prefilter (*) tone band-pass, per tone; as a strided cross-correlation weight
        demod = ToneDemod(cfg, n_mics)
        band = demod._g_rev[:, ::-1]                                    # [tones, L] lowpass * exp(+j 2 pi f i / fs)
        pre = design_prefilter(cfg)
        w = np.stack([np.convolve(pre, band[t]) for t in range(cfg.n_tones)])          # [tones, Lw]
        self.lw = w.shape[1]
        rev = w[:, ::-1]
        weight = np.stack([part for t in range(cfg.n_tones) for part in (rev[t].real, rev[t].imag)])
        self.weight = torch.from_numpy(weight.copy()[:, None, :]).to(dev, self.rd)      # [2*tones, 1, Lw]
        self.fir = "conv" if (single or fir != "fft") else "fft"
        self._w = w                                                     # [tones, Lw] the filter itself, for the FFT route
        self._wf: dict = {}                                             # FFT length -> its spectrum on the device
        self.tones = np.asarray(cfg.tone_hz, np.float64)
        self.sos = demod._sos
        stft = DopplerSTFT(cfg, n_mics)
        self.win = torch.from_numpy(stft._win).to(dev, self.rd)
        self.scale = float(stft._scale)
        self.sel = torch.from_numpy(np.asarray(stft._sel)).to(dev)
        self._rot: dict = {}
        self._hp: dict = {}

    def _tables(self, m: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Per clip length: the exact down-mixing phase of every baseband sample, and the high-pass response."""
        if m not in self._rot:
            cfg = self.cfg
            n_abs = np.arange(m, dtype=np.float64) * cfg.decim
            phase = np.fmod(self.tones[:, None] * n_abs[None, :], cfg.fs) / cfg.fs       # exact, as ToneDemod
            self._rot[m] = torch.from_numpy(np.exp(-2j * np.pi * phase)).to(self.device, self.cd)   # [tones, m]
            n_fft = sfft.next_fast_len(m + int(math.ceil(_HP_SECONDS * cfg.bb_fs)))   # (follows a config with another baseband rate)
            _, h = sps.sosfreqz(self.sos, worN=n_fft, whole=True)
            self._hp[m] = (torch.from_numpy(h).to(self.device, self.cd), n_fft)
        return self._rot[m], self._hp[m][0], self._hp[m][1]

    def _decimate_fft(self, x: torch.Tensor) -> torch.Tensor:
        """x [R, n] real -> [R, tones, ceil(n / decim)] complex: sample 0, decim, 2*decim, ... of (filter * x), from silence.

        The strided convolution computes every output sample as a sum over the whole filter (several hundred
        taps). Here: one FFT of the input, a multiplication with the filter's spectrum, and "keep every decim-th
        sample" done in the frequency domain, where it means adding up the decim stretches of the spectrum
        (all of them, so the filter's small stop-band leakage is kept exactly as the convolution has it).
        """
        cfg = self.cfg
        r, n = x.shape
        d, m = cfg.decim, math.ceil(n / cfg.decim)
        n_fft = -(-(n + self.lw - 1) // d) * d                         # long enough that nothing wraps around; a multiple of decim
        while True:                                                    # ... and a length FFTs are fast for
            rest = n_fft
            for p in (2, 3, 5, 7):
                while rest % p == 0:
                    rest //= p
            if rest == 1:
                break
            n_fft += d
        if n_fft not in self._wf:
            self._wf = {n_fft: torch.from_numpy(np.fft.fft(self._w, n_fft, axis=1)).to(self.device, self.cd)}
        wf = self._wf[n_fft]
        out = torch.empty((r, cfg.n_tones, m), device=self.device, dtype=self.cd)
        rows = max(1, int(0.8e8 // n_fft))                              # about 1.3 GB of complex128 at a time, two or three such arrays alive
        for a in range(0, r, rows):
            spec = torch.fft.fft(x[a:a + rows], n=n_fft, dim=-1)
            for t in range(cfg.n_tones):
                folded = (spec * wf[t]).reshape(-1, d, n_fft // d).sum(1) / d
                out[a:a + rows, t] = torch.fft.ifft(folded, dim=-1)[:, :m]
            del spec
        return out

    @torch.no_grad()
    def baseband(self, rx: torch.Tensor) -> torch.Tensor:
        """rx [B, n, n_mics] -> high-passed complex baseband [B, n_mics, tones, m] (ToneDemod.process output)."""
        cfg = self.cfg
        b, n, n_mics = rx.shape
        if n_mics != self.n_mics:
            raise ValueError(f"expected {self.n_mics} microphone channels, got {n_mics}")
        x = rx.to(self.device, self.rd).transpose(1, 2).reshape(b * n_mics, 1, n)
        if self.fir == "fft":
            z = self._decimate_fft(x[:, 0])                                             # [B*mics, tones, m]
            m = z.shape[-1]
            rot, hp, n_fft = self._tables(m)
            z = z * rot[None]
        else:
            x = torch.nn.functional.pad(x, (self.lw - 1, 0))                            # filters start from silence
            y = torch.nn.functional.conv1d(x, self.weight, stride=cfg.decim)            # [B*mics, 2*tones, m]
            m = y.shape[-1]
            rot, hp, n_fft = self._tables(m)
            z = torch.complex(y[:, 0::2], y[:, 1::2]) * rot[None]                       # [B*mics, tones, m]
            del y
        del x
        # 5 Hz high-pass started in steady state = zero-state filtering of (z - first sample)
        d = z - z[..., :1]
        out = torch.fft.ifft(torch.fft.fft(d, n=n_fft, dim=-1) * hp, dim=-1)[..., :m]
        return out.reshape(b, n_mics, cfg.n_tones, m)

    @torch.no_grad()
    def columns(self, rx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """rx [B, n, n_mics] float -> (doppler_db [B, T, tones*mics, F] float32, lateral [B, T, F] complex64 or None).

        Exactly the (power_db, lateral) pair DopplerSTFT emits per frame, for every frame of every clip.
        """
        cfg = self.cfg
        bb = self.baseband(rx)                                                          # [B, mics, tones, m]
        bb = torch.nn.functional.pad(bb, (cfg.stft_win - cfg.stft_hop, 0))              # the STFT buffer starts with zeros
        frames = bb.unfold(-1, cfg.stft_win, cfg.stft_hop) * self.win                   # [B, mics, tones, T, win]
        spec = torch.fft.fft(frames, n=cfg.stft_fft, dim=-1).index_select(-1, self.sel) * self.scale
        del frames, bb
        power = spec.real ** 2 + spec.imag ** 2                                         # [B, mics, tones, T, F]
        db = 10.0 * torch.log10(power.clamp_min(1e-30))
        b, _, _, t, f = db.shape
        doppler_db = db.permute(0, 3, 2, 1, 4).reshape(b, t, cfg.n_tones * self.n_mics, f).to(torch.float32)
        lateral = None
        if self.n_mics >= 2:
            cross = spec[:, 0] * spec[:, 1].conj()                                      # [B, tones, T, F]
            lag = (cross[:, :, 1:] * cross[:, :, :-1].conj()).sum(1)                    # [B, T-1, F]
            lag = torch.nn.functional.pad(lag, (1, 1))                                  # zero neighbours at both ends
            third = 1.0 / 3.0
            smooth = lag[..., :-2] * third + lag[..., 1:-1] * third + lag[..., 2:] * third
            lateral = torch.cat([torch.zeros_like(smooth[:, :1]), smooth], dim=1).to(torch.complex64)
        return doppler_db, lateral


@torch.no_grad()
def stream_features_torch(rx: np.ndarray, cfg: SonarConfig, device: torch.device | str = "cpu",
                          frontend: TorchFrontend | None = None) -> StreamFeatures:
    """features.stream_features for one recording [n, n_mics], with the heavy part on `device`."""
    rx = np.asarray(rx)
    if rx.ndim == 1:
        rx = rx[:, None]
    fe = frontend or TorchFrontend(cfg, rx.shape[1], device)
    db, lat = fe.columns(torch.from_numpy(np.ascontiguousarray(rx))[None])
    return stream_from_columns(fe.cfg, db[0].cpu().numpy(), None if lat is None else lat[0].cpu().numpy())


def frames_for(cfg: SonarConfig, n_samples: int) -> int:
    """How many feature columns `n_samples` of audio give."""
    m = math.ceil(n_samples / cfg.decim)
    return (m - cfg.stft_hop) // cfg.stft_hop + 1 if m >= cfg.stft_hop else 0

"""selftest.py — a 1-minute check that this laptop can do sonar, before you record or demo.

Plain words: it plays test sounds above hearing through each speaker channel,
records the microphones, and prints

  1. the microphone noise floor,
  2. for every test frequency from 16 to 22 kHz and each speaker channel: how
     strong the tone arrives and its signal-to-noise ratio (SNR),
  3. whether the left and right speaker channels really come out of different
     places (needed for the two-speaker swipe-direction cue),
  4. with the real sonar signal: does the receiver lock, how much the speaker
     and microphone clocks drift, how much headroom echoes have, whether the
     microphone clips, and how much AUDIBLE sound the signal leaks (speaker
     distortion),
  5. a verdict and which signal profile to use.

How to run (Windows demo laptop; close Teams/Discord/browser audio first):
    py -3.12 -m sonar.selftest
    py -3.12 -m sonar.selftest --profile strong --report selftest.json
Without hardware (checks the script itself):
    python -m sonar.selftest --backend sim

What it changes: plays about 35 s of near-inaudible test tones at the current
system volume and holds the audio devices exclusively meanwhile. Writes the
report file only if --report is given.

Reading the numbers: "SNR 1 Hz" is tone power over the noise in a 1 Hz band.
The measurement that started this project was ~60 dB at 17-18 kHz and ~55 dB
at 19-20 kHz at 36 % volume. Below ~45 dB, raise the volume or use a lower profile.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np
from scipy import signal as sps

from .audio import AudioError, LiveAudio
from .config import SonarConfig, get_profile, profile_names
from .pipeline import Pipeline
from .tx import fade_ramp, make_tx_frame

TEST_FREQS = np.arange(16000.0, 22001.0, 500.0)
TONE_S = 0.4
TONE_AMP = 0.5
GAP_S = 0.05
MARKER_S = 0.05


# ---------------------------------------------------------------------------
# Probe devices: play a signal, return what the microphones heard
# ---------------------------------------------------------------------------
class LiveProbe:
    """Plays through the real speakers (same stream set-up as the engine)."""

    def __init__(self, cfg: SonarConfig, **audio_kwargs: Any) -> None:
        self.cfg = cfg
        self.audio = LiveAudio(cfg, **audio_kwargs)
        self.audio.idle_silent = True
        self.n_mics = self.audio.n_mics
        self.audio.start()
        self.info = self.audio.info
        time.sleep(0.3)
        self.audio.read(timeout=1.0)                    # throw away the start-up samples

    def run(self, program: np.ndarray) -> np.ndarray:
        """Play `program` [n, n_out]; returns the recording [n + ~0.6 s, n_mics] (unknown small latency)."""
        need = len(program) + int(0.6 * self.cfg.fs)
        self.audio.read(timeout=0.05)
        done = self.audio.play_program(program)
        got: list[np.ndarray] = []
        total = 0
        while total < need:
            rx = self.audio.read(timeout=1.0)
            if rx is None:
                continue
            got.append(rx)
            total += len(rx)
        done.wait(timeout=1.0)
        return np.concatenate(got)[:need]

    def close(self) -> None:
        self.audio.stop()


class SimProbe:
    """A stand-in laptop: delay, gain, weaker response above 18.8 kHz, noise. For testing this script."""

    def __init__(self, cfg: SonarConfig, cn0_db: float = 60.0, separate_speakers: bool = True, seed: int = 0) -> None:
        self.cfg = cfg
        self.n_mics = cfg.n_mics
        self.info = {"mode": "sim"}
        self._rng = np.random.default_rng(seed)
        self._gain = 10.0 ** (-73.0 / 20.0)
        carrier = (0.5 * self._gain) ** 2 / 2.0
        self._sigma = float(np.sqrt(carrier / 10.0 ** (cn0_db / 10.0) * cfg.fs / 2.0))
        # speaker -> mic distances in samples (left speaker is nearer the left mic, and vice versa)
        near, far = 31.0, 37.5
        self._delay = np.array([[near, far], [far, near]]) if separate_speakers else np.array([[34.0, 34.0], [34.0, 34.0]])

    def run(self, program: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        latency = 3301
        n = len(program) + int(0.6 * cfg.fs)
        nfft = int(2 ** np.ceil(np.log2(n + 4096)))
        f = np.fft.rfftfreq(nfft, 1.0 / cfg.fs)
        resp = 10.0 ** (-10.0 / (1.0 + np.exp(-(f - 18800.0) / 60.0)) / 20.0) * self._gain
        spec = np.fft.rfft(program.astype(np.float64), nfft, axis=0)
        out = np.zeros((n, self.n_mics))
        for m in range(self.n_mics):
            acc = np.zeros(len(f), dtype=np.complex128)
            for s in range(program.shape[1]):
                acc += spec[:, s] * resp * np.exp(-2j * np.pi * f * (latency + self._delay[min(s, 1), min(m, 1)]) / cfg.fs)
            out[:, m] = np.fft.irfft(acc, nfft)[:n]
        return (out + self._sigma * self._rng.standard_normal(out.shape)).astype(np.float32)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Test signals and measurements
# ---------------------------------------------------------------------------
def _marker(cfg: SonarConfig) -> np.ndarray:
    """A 50 ms windowed chirp: lets us find where the played signal starts inside the recording."""
    n = int(MARKER_S * cfg.fs)
    t = np.arange(n) / cfg.fs
    return 0.5 * sps.chirp(t, 17500.0, MARKER_S, 21000.0) * np.hanning(n)


def tone_program(cfg: SonarConfig, channel: int) -> tuple[np.ndarray, list[tuple[float, int, int]]]:
    """Marker + one faded tone per test frequency on one speaker channel. Returns (signal, [(f, start, end)])."""
    fs = cfg.fs
    n_tone, n_gap, n_fade = int(TONE_S * fs), int(GAP_S * fs), int(0.02 * fs)
    env = np.ones(n_tone)
    env[:n_fade] = fade_ramp(n_fade)
    env[-n_fade:] = fade_ramp(n_fade)[::-1]
    parts = [_marker(cfg), np.zeros(int(0.1 * fs))]
    spans = []
    pos = sum(len(p) for p in parts)
    for f in TEST_FREQS:
        parts += [TONE_AMP * env * np.cos(2 * np.pi * f * np.arange(n_tone) / fs), np.zeros(n_gap)]
        spans.append((float(f), pos + n_fade + int(0.03 * fs), pos + n_tone - n_fade - int(0.01 * fs)))
        pos += n_tone + n_gap
    mono = np.concatenate(parts)
    out = np.zeros((len(mono), cfg.n_out), dtype=np.float32)
    out[:, channel] = mono
    return out, spans


def find_offset(cfg: SonarConfig, rx: np.ndarray) -> int:
    """Sample index in `rx` where the program (its marker) starts."""
    m = _marker(cfg)
    head = rx[:int(1.0 * cfg.fs) + len(m)].sum(axis=1)
    corr = np.abs(sps.hilbert(sps.fftconvolve(head, m[::-1], mode="valid")))
    return int(np.argmax(corr))


def tone_phasor(x: np.ndarray, f: float, fs: int) -> np.ndarray:
    """Complex amplitude of a tone at `f` in each column of x (amplitude in full-scale units)."""
    n = np.arange(len(x))
    w = np.hanning(len(x))
    return 2.0 * (x * (w * np.exp(-2j * np.pi * f * n / fs))[:, None]).sum(axis=0) / w.sum()


def noise_density(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch noise spectrum: (frequencies, dB re full-scale-sine power per Hz) per mic."""
    f, p = sps.welch(x, fs=fs, nperseg=8192, axis=0)
    return f, 10 * np.log10(p / 0.5 + 1e-30)            # a full-scale sine has power 0.5


def run_selftest(probe: Any, cfg: SonarConfig, profile: str = "default", verbose: bool = True) -> dict[str, Any]:
    fs = cfg.fs
    say = print if verbose else (lambda *a, **k: None)
    report: dict[str, Any] = {"profile": profile, "device": probe.info, "fs": fs, "checks": {}}

    # 1. noise floor -------------------------------------------------------------------------
    say("1/4 listening to the room (2 s of silence) ...")
    quiet = probe.run(np.zeros((int(2.0 * fs), cfg.n_out), dtype=np.float32))[int(0.3 * fs):]
    nf, ndb = noise_density(quiet, fs)
    noise_at = lambda f: float(np.mean(ndb[np.argmin(np.abs(nf - f))]))        # noqa: E731  (mean over mics)
    report["noise_dbfs_hz"] = {f"{int(f)}": round(noise_at(f), 1) for f in TEST_FREQS}
    report["audible_noise_dbfs"] = round(float(10 * np.log10(np.mean(sps.sosfilt(
        sps.butter(4, [200, 8000], "bandpass", fs=fs, output="sos"), quiet, axis=0) ** 2) + 1e-30)), 1)

    # 2. tones per speaker channel -------------------------------------------------------------
    table: dict[str, dict[str, Any]] = {}
    phasors: dict[int, dict[float, np.ndarray]] = {}
    for ch in range(cfg.n_out):
        say(f"2/4 test tones 16-22 kHz on speaker channel {ch} ({'left' if ch == 0 else 'right'}) ...")
        prog, spans = tone_program(cfg, ch)
        rx = probe.run(prog)
        off = find_offset(cfg, rx)
        phasors[ch] = {}
        for f, a, b in spans:
            ph = tone_phasor(rx[off + a:off + b], f, fs)
            phasors[ch][f] = ph
            level = 20 * np.log10(np.abs(ph) + 1e-12)                           # dBFS per mic
            power_db = level - 3.01                                             # tone power re full-scale sine power
            table.setdefault(f"{int(f)}", {})[f"ch{ch}"] = {
                "level_dbfs": [round(float(v), 1) for v in level],
                "snr_1hz_db": round(float(np.mean(power_db) + 3.01 - noise_at(f)), 1),
            }
    report["tones"] = table
    if verbose:
        say("\n   freq    " + "".join(f"  ch{ch} level dBFS (mic0/mic1)   SNR 1 Hz" for ch in range(cfg.n_out)))
        for f in TEST_FREQS:
            row = f"  {f / 1000:5.1f} k "
            for ch in range(cfg.n_out):
                e = table[f"{int(f)}"][f"ch{ch}"]
                row += f"     {e['level_dbfs'][0]:7.1f} /{e['level_dbfs'][-1]:7.1f}        {e['snr_1hz_db']:5.1f} dB"
            say(row)

    # 3. are the two speaker channels in different places? ------------------------------------
    sep_us = None
    if cfg.n_out >= 2 and probe.n_mics >= 2:
        tdoa = []
        usable = [f for f in TEST_FREQS if 17500 <= f <= 21000]
        for ch in range(2):
            # phase difference between the mics vs frequency -> arrival-time difference between the mics
            dphi = np.unwrap([np.angle(phasors[ch][f][0] * np.conj(phasors[ch][f][1])) for f in usable])
            tdoa.append(-np.polyfit(usable, dphi, 1)[0] / (2 * np.pi))
        sep_us = float(abs(tdoa[0] - tdoa[1]) * 1e6)
        report["speaker_separation_us"] = round(sep_us, 1)
        report["mic_tdoa_us"] = [round(float(t * 1e6), 1) for t in tdoa]

    # 4. the real sonar signal ---------------------------------------------------------------------
    say("\n3/4 the real sonar signal for 8 s (keep hands away) ...")
    frame = make_tx_frame(cfg)
    reps = int(8.0 * fs / cfg.frame_len)
    ramp = np.ones(reps * cfg.frame_len)
    k = int(cfg.fade_ms * 1e-3 * fs)
    ramp[:k], ramp[-k:] = fade_ramp(k), fade_ramp(k)[::-1]
    rx = probe.run((np.tile(frame, (reps, 1)) * ramp[:, None]).astype(np.float32))
    body = rx[int(0.5 * fs):int(7.5 * fs)]
    pipe = Pipeline(cfg, n_mics=probe.n_mics)
    frames = pipe.process(body)
    locked = [fr for fr in frames if fr.cir is not None]
    peak = float(np.max(np.abs(rx)))
    sonar: dict[str, Any] = {
        "locked": bool(pipe.frontend.locked),
        "resyncs": int(pipe.frontend.n_resync),
        "drift_ppm": round(float(locked[-1].drift_ppm), 2) if locked else None,
        "coherence": round(float(np.median([fr.coherence for fr in locked])), 3) if locked else 0.0,
        "tone_headroom_db": round(pipe.snr_db(), 1),
        "peak_dbfs": round(20 * np.log10(peak + 1e-12), 1),
        "clipped": peak >= 0.99,
    }
    if len(locked) > 50:
        cir = np.stack([fr.cir for fr in locked[-200:]])                        # [frames, mics, taps]
        direct = np.abs(cir.mean(axis=0))[:, cfg.direct_tap - 2:cfg.direct_tap + 3].max()
        jitter = np.std(cir[:, :, 40:], axis=0).mean()                          # far taps: only noise
        sonar["chirp_snr_db"] = round(float(20 * np.log10(direct / (jitter + 1e-20))), 1)
    # audible leak: the signal repeats every 20 ms, so anything the speaker adds below 10 kHz sits on
    # multiples of 50 Hz. Average the frames (random room noise averages out), look at those lines.
    n_fr = len(body) // cfg.frame_len
    avg = body[:n_fr * cfg.frame_len].reshape(n_fr, cfg.frame_len, -1).mean(axis=0)
    spec = np.abs(np.fft.rfft(avg, axis=0)) ** 2
    bins = np.fft.rfftfreq(cfg.frame_len, 1.0 / fs)
    aud = spec[(bins >= 200) & (bins <= 10000)].sum(axis=0)
    ult = spec[bins >= 17000].sum(axis=0)
    q_fr = len(quiet) // cfg.frame_len
    q_avg = quiet[:q_fr * cfg.frame_len].reshape(q_fr, cfg.frame_len, -1).mean(axis=0)
    q_aud = (np.abs(np.fft.rfft(q_avg, axis=0)) ** 2)[(bins >= 200) & (bins <= 10000)].sum(axis=0) * q_fr / max(n_fr, 1)
    sonar["audible_leak_db"] = round(float(10 * np.log10(np.mean(aud) / np.mean(ult) + 1e-30)), 1)
    sonar["audible_leak_above_room"] = bool(np.mean(aud) > 4.0 * np.mean(q_aud))
    report["sonar"] = sonar

    # 5. verdict ----------------------------------------------------------------------------------
    def tone_snr(f: float) -> float:
        key = f"{int(TEST_FREQS[np.argmin(np.abs(TEST_FREQS - f))])}"
        return float(np.mean([table[key][f"ch{ch}"]["snr_1hz_db"] for ch in range(cfg.n_out)]))

    scores = {}
    for name in profile_names():
        c = get_profile(name)
        if c.n_out > cfg.n_out:
            continue
        chirp = np.mean([tone_snr(f) for f in np.arange(c.chirp_lo_hz, c.chirp_hi_hz + 1, 500.0)])
        scores[name] = {"tone_snr_db": round(min(tone_snr(f) for f in c.tone_hz), 1), "chirp_snr_db": round(float(chirp), 1)}
    report["profiles"] = scores
    checks = report["checks"]
    mine = scores.get(profile, {"tone_snr_db": 0.0, "chirp_snr_db": 0.0})
    checks["tone_snr"] = "PASS" if mine["tone_snr_db"] >= 50 else ("WARN" if mine["tone_snr_db"] >= 40 else "FAIL")
    checks["chirp_snr"] = "PASS" if mine["chirp_snr_db"] >= 45 else ("WARN" if mine["chirp_snr_db"] >= 35 else "FAIL")
    checks["lock"] = "PASS" if sonar["locked"] and sonar["resyncs"] == 0 and sonar["coherence"] > 0.8 else "FAIL"
    checks["clock_drift"] = "PASS" if sonar["drift_ppm"] is not None and abs(sonar["drift_ppm"]) < 5 else "WARN"
    checks["clipping"] = "FAIL" if sonar["clipped"] else "PASS"
    checks["audible_leak"] = "WARN" if sonar["audible_leak_above_room"] and sonar["audible_leak_db"] > -45 else "PASS"
    if sep_us is not None:
        checks["speaker_separation"] = "PASS" if sep_us >= 60 else "WARN"
    recommend = profile
    if checks["tone_snr"] != "PASS":
        better = max(scores, key=lambda n: scores[n]["tone_snr_db"] if n in ("default", "strong", "low") else -1e9)
        if scores[better]["tone_snr_db"] >= mine["tone_snr_db"] + 5:
            recommend = better
    if checks.get("speaker_separation") == "WARN" and checks["audible_leak"] == "WARN" and "clean" in scores:
        recommend = "clean"
    report["recommended_profile"] = recommend

    if verbose:
        say("\n4/4 results")
        say(f"   noise floor at 19 kHz        {report['noise_dbfs_hz']['19000']} dBFS/Hz, audible room noise {report['audible_noise_dbfs']} dBFS")
        say(f"   profile '{profile}': weakest pilot tone SNR {mine['tone_snr_db']} dB, chirp band SNR {mine['chirp_snr_db']} dB (in 1 Hz)   [{checks['tone_snr']}/{checks['chirp_snr']}]")
        say(f"   receiver lock                {sonar['locked']}, coherence {sonar['coherence']}, re-locks {sonar['resyncs']}   [{checks['lock']}]")
        say(f"   speaker/mic clock drift      {sonar['drift_ppm']} ppm   [{checks['clock_drift']}]")
        say(f"   echo headroom (tones)        {sonar['tone_headroom_db']} dB above the Doppler noise floor; chirp direct path {sonar.get('chirp_snr_db', 'n/a')} dB per frame")
        say(f"   microphone peak              {sonar['peak_dbfs']} dBFS   [{checks['clipping']}]")
        say(f"   audible leak of the signal   {sonar['audible_leak_db']} dB relative to the ultrasound"
            f"{' (above room noise)' if sonar['audible_leak_above_room'] else ' (not above room noise)'}   [{checks['audible_leak']}]")
        if sep_us is not None:
            say(f"   left/right speaker channels  arrive {sep_us:.0f} microseconds apart at the mic pair   [{checks['speaker_separation']}]"
                + ("" if sep_us >= 60 else "  -> both channels seem to come from the same place: only the two-MIC direction cue will work"))
        say("   profile comparison (weakest tone SNR / chirp SNR): "
            + ", ".join(f"{n} {v['tone_snr_db']:.0f}/{v['chirp_snr_db']:.0f}" for n, v in scores.items()))
        bad = [k for k, v in checks.items() if v == "FAIL"]
        warn = [k for k, v in checks.items() if v == "WARN"]
        say(f"\nVERDICT: {'FAIL ' + str(bad) if bad else ('OK with warnings ' + str(warn) if warn else 'PASS')}"
            f"   recommended profile: {recommend}")
        if checks["tone_snr"] != "PASS" or checks["chirp_snr"] != "PASS":
            say("   -> raise the Windows speaker volume (then re-run), or use the recommended profile.")
        if checks["clipping"] == "FAIL":
            say("   -> lower the speaker volume or the microphone level: the input is clipping.")
        if checks["lock"] == "FAIL":
            say("   -> the chirp was not received cleanly: check exclusive mode, close other audio apps, try --blocksize 3840.")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sonar.selftest", description="1-minute sonar hardware self-test")
    ap.add_argument("--profile", choices=profile_names(), default="default")
    ap.add_argument("--backend", choices=["live", "sim"], default="live")
    ap.add_argument("--in-device")
    ap.add_argument("--out-device")
    ap.add_argument("--blocksize", type=int, default=1920)
    ap.add_argument("--shared", action="store_true", help="Windows: use shared mode (expected to FAIL: the mic is low-passed)")
    ap.add_argument("--duplex", action="store_true", help="one duplex stream instead of separate streams")
    ap.add_argument("--report", help="write the full report as JSON to this file")
    args = ap.parse_args(argv)
    cfg = get_profile(args.profile)
    try:
        if args.backend == "live":
            probe: Any = LiveProbe(cfg, in_device=args.in_device, out_device=args.out_device, exclusive=not args.shared,
                                   blocksize=args.blocksize, mode="duplex" if args.duplex else "auto")
        else:
            probe = SimProbe(cfg)
    except AudioError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"sonar self-test, profile '{args.profile}', devices: {probe.info}")
    try:
        report = run_selftest(probe, cfg.with_(n_mics=probe.n_mics) if probe.n_mics != cfg.n_mics else cfg, args.profile)
    except AudioError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        probe.close()
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"report written to {args.report}")
    return 1 if "FAIL" in report["checks"].values() else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""external_real_check.py — test a synthetic-trained model on REAL public recordings.

What it does
------------
Takes the push / pull / slide clips of the public UltrasonicGS dataset (real
people, a 20 kHz tone, ONE microphone, a Raspberry Pi instead of our laptop),
re-shapes each clip so the core feature chain accepts it, and asks:
  * does the training-free rule (which side of the tone lights up) get push vs
    pull right?
  * does a CNN trained ONLY on synthetic data recognise real pushes and pulls?

How a foreign clip is made to fit (plain words)
    1. find the recorded tone precisely and keep +-400 Hz around it;
    2. play it back on our two tone frequencies (18.0 and 18.5 kHz), identically
       on both and on both "microphones";
    3. add two quiet seconds in front (copied from the quiet end of the clip) so
       the core can learn its noise floor, as it does on a live stream.
Because all four model channels are identical, the clip carries NO left/right
information: only push / pull / "some swipe" / none can be judged. Left-right
has to be tested on our own laptop.

With --fewshot it also answers the question that matters for tonight's plan:
"if we record only K real clips per gesture, how much does synthetic
pre-training help?" For K = 1, 3, 5, 10 it fine-tunes the synthetic-trained
model on K real clips per class and compares it with the same network trained
from scratch on those K clips, both tested on all the remaining real clips.
(Classes there: push, pull, slide (either direction), none.)

Why
---
It is the only real data we can test on before our own recordings exist. It is
different hardware, so treat the result as a sanity check, never as demo accuracy.

Getting the data (354 MB, no licence stated: evaluation only, do not redistribute)
    https://github.com/yuejiaowang/database   ->  SingleGesture/wavs/<gesture>/*.wav
    paper: Wang et al., Sensors 23(4):1790, 2023, https://www.mdpi.com/1424-8220/23/4/1790

How to run (from the repo root; -I because the wav files are downloaded data)
    cd ~/Documents/Dev/sonar-hack
    datagen/.venv/bin/python -I -c "import sys; sys.path.insert(0, '.'); from datagen.external_real_check import main; main()" \\
        --wavs /path/to/SingleGesture/wavs --model datagen/out/synth_v2/pretrained.pt

What it changes: nothing (prints a table). Read-only.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from scipy.signal import hilbert, resample_poly

from . import adapter as ad

CLASSES = {"push": "push", "pull": "pull", "slide_left": "swipe_left", "slide_right": "swipe_right"}
LEAD_S, TAIL_S = 2.0, 1.0


def to_core_audio(x: np.ndarray, fs: int, core: ad.Core, rng: np.random.Generator) -> tuple:
    """Foreign single-tone mono clip -> 2-mic audio on the core's tone frequencies. Returns (audio, t_clip_start)."""
    if fs != core.fs:
        g = np.gcd(int(fs), int(core.fs))
        x = resample_poly(x, core.fs // g, fs // g)
    fs = core.fs
    z = hilbert(x)
    n = np.arange(len(z), dtype=np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    band = (f > 16500.0) & (f < 22500.0)
    f_c = float(f[band][np.argmax(spec[band])])
    dec = fs // 2000
    bb = resample_poly(z * np.exp(-2j * np.pi * np.mod(n * (f_c / fs), 1.0)), 1, dec)  # 2 kHz, tone near 0 Hz
    rate = fs / dec
    # refine the tone frequency from the steady phase creep, so the carrier lands exactly on 0 Hz
    k = int(0.1 * rate)
    blocks = bb[: len(bb) // k * k].reshape(-1, k).mean(axis=1)
    df = float(np.angle((blocks[1:] * np.conj(blocks[:-1])).sum()) / (2 * np.pi * k / rate))
    bb = bb * np.exp(-2j * np.pi * df * np.arange(len(bb)) / rate)
    # quiet lead-in and tail: tile the quietest 0.4 s of the clip, cross-faded
    w = int(0.4 * rate)
    mov = np.abs(bb - np.convolve(bb, np.ones(w) / w, mode="same"))
    energy = np.convolve(mov**2, np.ones(w), mode="valid")
    i0 = int(np.argmin(energy[w // 2 : len(energy) - w // 2])) + w // 2
    idle = bb[i0 : i0 + w]
    fade = int(0.02 * rate)
    ramp = np.linspace(0.0, 1.0, fade)

    def tile(seconds: float) -> np.ndarray:
        out = idle.copy()
        while len(out) < seconds * rate:
            out = np.concatenate([out[:-fade], out[-fade:] * (1 - ramp) + idle[:fade] * ramp, idle[fade:]])
        return out

    lead, tail = tile(LEAD_S), tile(TAIL_S)
    ext = np.concatenate([lead[:-fade], lead[-fade:] * (1 - ramp) + bb[:fade] * ramp, bb[fade:-fade],
                          bb[-fade:] * (1 - ramp) + tail[:fade] * ramp, tail[fade:]])
    up = resample_poly(ext, dec, 1)
    m = np.arange(len(up), dtype=np.float64)
    carrier = sum(np.exp(2j * np.pi * np.mod(m * (f_t / fs), 1.0)) for f_t in core.tones)
    y = np.real(up * carrier)
    y = y / (np.abs(y).max() + 1e-12) * 0.05
    audio = np.stack([y, y], axis=1) + 1e-5 * rng.standard_normal((len(y), 2))
    return audio.astype(np.float32), len(lead) / rate


def main() -> None:
    ap = argparse.ArgumentParser(description="Test on real public recordings (see the file header).")
    ap.add_argument("--wavs", required=True, help="folder with push/ pull/ slide_left/ slide_right subfolders")
    ap.add_argument("--model", default="", help="pretrained.pt from datagen.train_check (optional)")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N clips per class")
    ap.add_argument("--fewshot", action="store_true", help="also run the K-real-clips fine-tuning comparison")
    args = ap.parse_args()

    import soundfile as sf

    core = ad.load_core()
    if not core.available or core.module("gestures") is None:
        raise SystemExit("the core sonar/ package is needed for this check")
    g = core.module("gestures")
    labels = list(core.gesture_labels)
    model = None
    if args.model:
        import torch

        from .train_check import SmallCNN

        ckpt = torch.load(args.model, map_location="cpu")
        model = SmallCNN(ckpt["n_channels"], len(ckpt["labels"]))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

    rng = np.random.default_rng(0)
    rows = {c: {"n": 0, "found": 0, "cnn": [], "rule": []} for c in CLASSES}
    real_x, real_y = [], []  # model inputs of the real clips, for the few-shot comparison
    for folder, truth in CLASSES.items():
        files = sorted(glob.glob(os.path.join(args.wavs, folder, "*.wav")))
        for path in files[: args.limit or None]:
            x, fs = sf.read(path, dtype="float64", always_2d=True)
            audio, t0 = to_core_audio(x[:, 0], fs, core, rng)
            stream = ad.stream_features(core, audio)
            clip = (t0, t0 + len(x) / fs)
            segs = [(s, w) for s, w in zip(stream.segments, stream.windows)
                    if w is not None and s.start / stream.frame_rate < clip[1] and (s.end + 1) / stream.frame_rate > clip[0]]
            rows[folder]["n"] += 1
            if not segs:
                rows[folder]["cnn"].append("none")
                rows[folder]["rule"].append("none")
                continue
            rows[folder]["found"] += 1
            quiet_end = int(t0 * stream.frame_rate) - 5  # a window that ends inside the quiet lead-in = a real "none"
            if quiet_end > stream.ready_from + 10:  # window_at pads the not-yet-calibrated start with zeros
                real_x.append(ad.window_at(core, stream, quiet_end))
                real_y.append("none")
            # the strongest burst inside the clip (measured on the features themselves, not on core internals)
            seg, win = max(segs, key=lambda sw: float(stream.feat[sw[0].start : sw[0].end + 1].sum()))
            rows[folder]["rule"].append(g.heuristic_classify(win, core.cfg).label)
            real_x.append(win)
            real_y.append({"push": "push", "pull": "pull"}.get(folder, "slide"))
            if model is not None:
                import torch

                with torch.no_grad():
                    pred = int(model(torch.from_numpy(win[None]).float()).argmax(1))
                rows[folder]["cnn"].append(labels[pred])

    print("\nREAL clips (UltrasonicGS: other hardware, one mic, no left/right information)")
    for name, key in (("core physics rules", "rule"), ("CNN trained only on synthetic data", "cnn")):
        if key == "cnn" and model is None:
            continue
        print(f"\n{name}   rows = truth, columns = prediction")
        print(" " * 13 + "".join(f"{lab[:10]:>11s}" for lab in labels) + "   movement found")
        for folder, truth in CLASSES.items():
            r = rows[folder]
            counts = [r[key].count(lab) for lab in labels]
            print(f"{folder:13s}" + "".join(f"{c:11d}" for c in counts) + f"   {r['found']}/{r['n']}")
        pp = sum(rows[c][key].count(CLASSES[c]) for c in ("push", "pull"))
        nn = rows["push"]["n"] + rows["pull"]["n"]
        sw = sum(sum(rows[c][key].count(lab) for lab in ("swipe_left", "swipe_right")) for c in ("slide_left", "slide_right"))
        ns = rows["slide_left"]["n"] + rows["slide_right"]["n"]
        print(f"  push/pull correct: {pp}/{nn} = {100 * pp / max(1, nn):.0f}%   slides called 'a swipe' (either way): {sw}/{ns}")
    if args.fewshot:
        fewshot(np.stack(real_x), real_y, args.model)


def fewshot(x: np.ndarray, names: list, model_path: str, shots: tuple = (1, 3, 5, 10), repeats: int = 5) -> None:
    """K real clips per class: synthetic pre-training + fine-tune versus training from scratch."""
    import torch
    from torch import nn

    from .train_check import SmallCNN, augment

    classes = ["none", "slide", "push", "pull"]
    y = np.array([classes.index(n) for n in names])
    xt = torch.from_numpy(x).float()
    ckpt = torch.load(model_path, map_location="cpu") if model_path else None
    # the synthetic model's own classes, mapped onto these four ("slide" starts from swipe_left's weights)
    src = None if ckpt is None else [ckpt["labels"].index(n) for n in ("none", "swipe_left", "push", "pull")]
    aug_labels = ["none", "swipe_left", "swipe_right", "push", "pull"]  # only used to drive the augmenter

    def run(pretrained: bool, k: int, seed: int) -> float:
        gen = torch.Generator().manual_seed(seed)
        rng = np.random.default_rng(seed)
        train_idx = np.concatenate([rng.choice(np.nonzero(y == c)[0], size=k, replace=False) for c in range(len(classes))])
        test = np.ones(len(y), dtype=bool)
        test[train_idx] = False
        torch.manual_seed(seed)
        net = SmallCNN(x.shape[1], len(classes))
        if pretrained:
            state = {key: val for key, val in ckpt["state_dict"].items() if not key.startswith("head.1")}  # all but the last layer
            net.load_state_dict(state, strict=False)
            with torch.no_grad():
                net.head[1].weight.copy_(ckpt["state_dict"]["head.1.weight"][src])
                net.head[1].bias.copy_(ckpt["state_dict"]["head.1.bias"][src])
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3 if pretrained else 3e-3, weight_decay=1e-3)
        xb0, yb0 = xt[train_idx], torch.from_numpy(y[train_idx])
        for _ in range(60 if pretrained else 150):
            net.train()
            # time shift / brightness / noise only: mirroring is meaningless here (all channels identical)
            dummy = torch.zeros(len(yb0), dtype=torch.long)
            xb, _ = augment(xb0.clone(), dummy, aug_labels, gen)
            loss = nn.functional.cross_entropy(net(xb), yb0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(xt[test]).argmax(1).numpy()
        return float((pred == y[test]).mean())

    print(f"\nFew-shot on REAL clips ({len(y)} windows; classes {classes}; mean of {repeats} random draws)")
    print("  real clips per class | from scratch | synthetic pre-training + fine-tune")
    for k in shots:
        scratch = np.mean([run(False, k, s) for s in range(repeats)])
        line = f"  {k:20d} | {100 * scratch:11.1f}% |"
        if ckpt is not None:
            line += f" {100 * np.mean([run(True, k, s) for s in range(repeats)]):6.1f}%"
        print(line)


if __name__ == "__main__":
    main()

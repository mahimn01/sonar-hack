# TRAINING — train the gesture model on the RTX 4090 box

For Lance: everything needed to train the gesture model on the 4090 machine without help, and hand
the model back to the demo laptop. Every command is given for **Windows (PowerShell)** and
**Linux**; run them from the repo folder.

**Short version:** set up once (steps 1-3), run the 1-minute smoke test (step 4), put the
recordings in `data/` (step 5), run one command (step 6), read the verdict (step 7), commit
`models-release/` (step 8).

**One rule above all:** a model works only on the laptop whose recordings trained it. The demo
laptop is the Windows HP; if the demo moves to a MacBook, the MacBook must record its own data
(section 9). The script refuses to mix recordings from two devices.

## What the one command does

`scripts/train_4090.ps1` (Windows) / `scripts/train_4090.sh` (Linux, macOS) run `scripts/train_4090.py`,
which prints every command before it runs it and keeps a full log in `train_out/runs/<time>/log.txt`.

1. Checks Python, PyTorch, the GPU and onnxruntime.
2. **Synthetic data.** datagen invents 12,000 short clips of virtual hands swiping and pushing in
   front of random virtual laptops (about 20,000 training pictures, 3x the Mac's set). With
   `--device cuda` the clips are computed **on the GPU** (section 11); before the first clip the
   script checks that the GPU makes the same pictures as the CPU code and stops if it does not.
   `--gen-device cpu` makes them on every CPU core as before. The result is cached: later runs
   reuse it as long as the generator code is unchanged.
3. **Real recordings**, if `data/` has any. Lists every session folder (a recording still in
   progress, half copied or from another device is skipped or stopped with a reason), cuts each
   session into training pictures (in parallel; cached per session, so new recordings only add
   their own time), measures which way round the laptop's left/right channels are, and makes about
   6,000 more synthetic clips laid over the real "nobody gesturing" recordings (`scene` and
   `breathing` sessions). The held-out people's recordings are left out of the clips the scored
   runs train on, so their score stays honest.
4. **Pre-training** on the synthetic data. On CUDA it runs at the same time as the clips of step 3
   are made (`--no-overlap` for one step at a time).
5. **Fine-tuning** on the real recordings, starting from the pre-trained weights. It is scored on
   people who are left out of training (`--val-person`, chosen automatically if you do not give
   it), so the score answers "does it work on a stranger?". `--sweep` tries 3 model widths x 3
   seeds and keeps the run with the best held-out accuracy; on CUDA four of those runs share the
   GPU at a time (`--sweep-jobs`).
6. Trains that winning setting once more on everyone (held-out people included), then exports
   `models/gesture.onnx` + `gesture.json` + `gesture.pt` and copies them to `models-release/`
   with `TRAINING_RUN.json` (the honest score, the sweep table, the data used).
7. Loads the model exactly as the engine does and runs the simulated demo visitor through it.

Without recordings it stops after step 4 and exports a synthetic-only model: a starting point,
not the demo model. Run it again once the recordings are in `data/`. The synthetic data and the
pre-training are reused, so the second run is mostly fine-tuning.

## 1. Get the code

The repo is private. Accept the invitation first: https://github.com/mahimn01/sonar-hack/invitations

```
git clone https://github.com/mahimn01/sonar-hack.git
cd sonar-hack
```

## 2. Python 3.12 environment

Windows (if `py` is missing, install Python 3.12 from python.org and tick "Add python.exe to PATH"):
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
```
Linux (Ubuntu: `sudo apt install python3.12 python3.12-venv` if `python3.12` is missing):
```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

## 3. PyTorch with CUDA, then the rest (the order matters)

**3.1** Run `nvidia-smi`. Read **"CUDA Version: X.Y"** at the top right: the newest CUDA your
driver can run. If `nvidia-smi` is not found, install the NVIDIA driver first.

**3.2** Install PyTorch **first**, from PyTorch's CUDA index:

| `nvidia-smi` says | index | PyTorch |
|---|---|---|
| CUDA Version 13.2 or higher | `https://download.pytorch.org/whl/cu132` | 2.15.0 (what this repo was tested with) |
| anything lower (12.x, 13.0, 13.1) | `https://download.pytorch.org/whl/cu126` | 2.14.1 |

Windows:
```powershell
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu132
.venv\Scripts\python -m pip install -r requirements-train.txt
```
Linux:
```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu132
.venv/bin/python -m pip install -r requirements-train.txt
```
Why this order: a plain `pip install torch` on Windows gives the CPU-only build. If
`requirements-train.txt` goes first, pip installs that CPU build, and the CUDA install afterwards
is skipped as "already satisfied".

**3.3** Check that PyTorch sees the GPU:
```powershell
.venv\Scripts\python -c "import torch; ok = torch.cuda.is_available(); print(torch.__version__, ok, torch.cuda.get_device_name(0) if ok else '')"
```
(Linux: `.venv/bin/python -c ...` with the same quoted part.) You want something like
`2.15.0+cu132 True NVIDIA GeForce RTX 4090`.

If it says `False` (or the version ends in `+cpu`):
- version ends in `+cpu`: `pip uninstall -y torch`, then repeat 3.2;
- you used `cu132` and the driver is older: uninstall, install from `cu126`;
- still `False`: update the NVIDIA driver, reboot.

## 4. Smoke test (about a minute, no GPU and no recordings needed)

Windows:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --device cpu --size tiny
```
Linux:
```bash
scripts/train_4090.sh --device cpu --size tiny
```
It runs every step on a few dozen clips and ends with `engine check: loaded ...` and `DONE`
(27 seconds on the Mac Studio). Its model is junk and stays under `train_out/tiny/`, so it never
replaces a real model. Then the same on the GPU: leave out `--device cpu` (cuda is the default).

`-ExecutionPolicy Bypass` is needed because Windows blocks unsigned scripts by default. Typing
it each time is fine; `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` lifts it for good.

## 5. Get the recordings from the demo laptop

The recordings are made **on the demo laptop itself**: on the HP with `sonar/record.py`
(README, section 2), on a MacBook with `scripts/mac_record_all.sh` (section 9). For example on the HP:
```powershell
.venv\Scripts\python -m sonar.record --person alice --script gestures --reps 12
.venv\Scripts\python -m sonar.record --person alice --script scene
```
Each run makes one session folder. The layout the training expects on the 4090 box:
```
sonar-hack/
  data/
    20261114-101500_alice_gestures/     <date>-<time>_<person>_<script>
      audio.wav                         raw microphones, 48 kHz, 24-bit (2 channels on the HP, 1 on a MacBook)
      session.json                      who, when, signal settings, every prompt with its sample range
    20261114-102300_alice_scene/
      audio.wav
      session.json
    ...
```
Every folder with a `session.json` anywhere under `data/` is picked up (sub-folders are fine), so
extracting a zip into `data/` twice, or nesting it one level deeper, does no harm: a second copy
of the same session is skipped. Each person must have their own `--person` name; that name is
how people are held out for the honest score. Keep each laptop's sessions in its own folder
(`data/` for the HP, `mac/` for a MacBook) and point the script at one with `--data`.

**Copy them over: zip (recommended).** On the HP:
```powershell
Compress-Archive -Path data -DestinationPath sonar-data.zip -Force
```
Move `sonar-data.zip` to the 4090 box (Google Drive, USB stick) and extract it **in the repo
folder** so that `data` sits next to `sonar`:
- Windows: `Expand-Archive sonar-data.zip -DestinationPath . -Force`
- Linux: `unzip -o sonar-data.zip`

Check: `dir data` (Windows) / `ls data` (Linux) lists the session folders.

Do not send recordings through git: GitHub refuses files over 100 MB (a 4.5-minute gesture
session is about 80 MB), and committing `data/` on a branch and switching back would take the
recordings out of the HP's `data/` folder while it is still recording.

Recording is still going on? Train with what is there, then copy the new sessions in and run
step 6 again. Sessions already cut are reused, so a re-run only works on the new recordings.

## 6. The full run

Windows:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --sweep        # best of 3 widths x 3 seeds
```
Linux:
```bash
scripts/train_4090.sh
scripts/train_4090.sh --sweep
```
Useful options (all: `--help`):

| option | what it does |
|---|---|
| `--val-person guest3,guest7` | hold out these people for the score (default: automatic; never the person with the most gestures, i.e. the presenter) |
| `--sweep` | widths 16, 24, 32 x seeds 0, 1, 2, keep the best held-out accuracy (`--widths`, `--seeds` to choose) |
| `--no-refit` | ship the winning held-out run itself instead of retraining it on everyone |
| `--data mac --release models-release-mac` | train a MacBook's recordings (section 9) without touching the HP model |
| `--labels swipe_left,swipe_right,push,pull` | train pull as well (default: the three demo gestures; pull recordings become "none") |
| `--size small` | a ~15-minute version of everything (fine on a CPU or a Mac: `--device cpu` / `mps`) |
| `--force-synth` | make the synthetic data again even if a cached copy exists |
| `--gen-device cpu` | make the clips and session pictures with the original CPU code instead of the GPU (section 11) |
| `--sweep-jobs 1`, `--no-overlap` | one training run at a time / no pre-training while clips are made (default on CUDA: 4 at a time, overlapping) |

**Run times** (16-core CPU + RTX 4090; GPU times are estimates, nothing on this list was run on a 4090.
`python scripts/bench_gpu.py` measures them on your machine, section 11):

| step | time | notes |
|---|---|---|
| GPU-against-CPU picture check | 1-2 min, once | 1,000 clips through both paths; remembered until the code, PyTorch or the GPU changes |
| synthetic data, 12,000 clips, on the GPU | 1-2 min (estimate, not measured) | the CPU only draws the random numbers and attaches the labels (about 45 ms of one core per clip, measured on the Mac) |
| the same with `--gen-device cpu` | 10-13 min (16 cores), ~25 min (8 cores) | each clip costs about 1 CPU-core-second (Mac Studio: 3,900 clips in 7-9 min on 8 cores); cached afterwards |
| synthetic clips on real backgrounds, 6,000 | about half of the above | made twice if a held-out person recorded a scene / breathing session (without them for the honest score, with everyone for the final model); cached |
| cutting recordings into pictures | ~2 s per minute of audio, one session per core | Mac Studio, with other jobs running: 6 minutes of audio in 12 s on one core; cached per session |
| pre-training, 60 epochs | about 1 min (estimate) | cached per width |
| fine-tuning, per run | under 1 min (estimate) | `--sweep` = 9 runs, plus 1 retrain |
| first full run | about 20-30 min with `--gen-device cpu`; a few minutes on the GPU (estimate) | `--sweep` adds 9 runs, 4 at a time on CUDA |
| re-run after new recordings | 3-8 min with `--gen-device cpu`; 1-3 min on the GPU (estimate) | synthetic data and pre-training reused; the clips on real backgrounds are made again whenever a new scene / breathing session arrives |

## 7. Read the result

The run ends with a block like this:
```
DONE in 23.4 min. Model: models/gesture.onnx (+ gesture.json, gesture.pt), copied to models-release/
Honest score on guest3, guest7: accuracy 0.874, macro-F1 0.842; at the engine's 0.6 threshold 0.830 of gestures recognised, 0.060 of non-gestures would trigger
Verdict: OK: usable for the demo; more people in the recordings will lift it
```

What the numbers mean:
- **accuracy**: share of the held-out people's examples (gestures and non-gesture movements) the
  model names correctly. **macro-F1**: the same idea, but every class counts equally.
- **gestures recognised (recall@0.6)**: share of their real gestures that would reach the app at
  the engine's 0.6 confidence threshold. **non-gestures that would trigger (false@0.6)**: share of
  their other movements (reaching, typing, the hand dropping back) that would fire a gesture.

What to look for:

| table / line | good | worrying |
|---|---|---|
| pre-training, scored on held-out synthetic clips | accuracy 0.80-0.88 | below 0.70: the synthetic data or the code is off (see below) |
| fine-tuning, scored on the held-out people | accuracy 0.85+, recall@0.6 0.85+, false@0.6 under 0.10 | accuracy under 0.80 or false@0.6 over 0.20 |
| left/right wiring | "as expected" | "REVERSED" is handled automatically. "unclear" for the microphones is normal on a MacBook (one microphone); "unclear" for both cues means swipes depend on the model alone |

The verdict line: **GOOD** = accuracy 0.90+, recall 0.85+, false triggers 0.10 or less. **OK** =
accuracy 0.80+, recall 0.70+, false triggers 0.20 or less. **POOR** = anything below.
DATA_STRATEGY.md expects 80-90 % for strangers with this recipe. With one or two held-out people
the score moves by several points between runs, so read it as a range, not a single number.

**If the numbers are poor:**

| what you see | what to do |
|---|---|
| the session table shows few people | the biggest lever: record more **people** (not more repetitions), 5 of each gesture each; re-run |
| a session's cut line says "N prompts with no movement found" | that person gestured too far or too faintly; ask for a re-record, closer (20-25 cm) and brisker |
| left and right swipes confused (confusion matrix in `models-release/confusion.txt`) | check the wiring line; ask whether the swipes were done "with the hand that starts on that side" (DATA_STRATEGY.md 6.3) and re-record a few |
| false@0.6 high | record another `scene` session (typing, reaching, people walking past); on the demo laptop start the engine with `--min-confidence 0.7` |
| one held-out person much worse than the rest | look at their sessions (`python -m sonar.plot data/<session>`); try `--val-person` with someone else to see if it is the person or the model |
| pre-training accuracy below 0.70 | `--force-synth` (the cache may be stale), check `git status` for local changes in `sonar/` or `datagen/` |
| still poor after more data | `--sweep`; and the engine's physics rules (`--physics`) are the fallback |

## 8. Hand the model back

`models-release/` (and `models-release-mac/` for a MacBook model) are the only output folders
that are committed; `.gitignore` lets them through. They hold:

| file | used by |
|---|---|
| `gesture.onnx` | the engine: the network |
| `gesture.json` | the engine: class names, confidence temperature, signal settings, left/right cue weights for the physics fallback, typical gesture strength |
| `gesture.pt` | further fine-tuning (`--init`), e.g. on a session recorded at the demo table |
| `metrics.json`, `confusion.txt` | the scores of the final training |
| `TRAINING_RUN.json` | the honest held-out score, the sweep table, sessions used, timings, git commit |

On the 4090 box:
```
git switch -c model-sat-am
git add models-release
git commit -m "Gesture model trained on the 4090"
git push -u origin model-sat-am
```
Then tell the team the branch name and paste the "Honest score" and "Verdict" lines.

On the HP (Windows):
```powershell
git fetch origin model-sat-am
git restore --source origin/model-sat-am --worktree -- models-release
New-Item -ItemType Directory -Force models | Out-Null
Copy-Item models-release\gesture.onnx, models-release\gesture.json models\ -Force
.venv\Scripts\python -m sonar.engine
```
The engine must print `classifier: model (gesture.onnx)`. If it prints `WARNING: the model was
trained with different settings`, it was trained for another `--profile`.

## 9. Demo on a MacBook instead of the HP

The HP's recordings cannot train a MacBook model, nor the other way round: the speakers and the
room echo differ, and a MacBook microphone records **one** channel where the HP records two
(Apple-silicon MacBooks report a single built-in microphone channel). With one channel the
two-microphone direction cue is gone; left/right swipes then come from the two speakers alone.
Everything below runs on the MacBook, from the repo folder.

**9.1 Set up (once).** PyTorch from PyPI already includes Apple's GPU backend (MPS):
```bash
uv venv --python 3.12 .venv              # uv fetches Python 3.12 if needed (brew install uv)
uv pip install --python .venv/bin/python -r requirements-train.txt
.venv/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"   # want True
```
Without uv: install Python 3.12 (python.org or `brew install python@3.12`), then
`python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements-train.txt`.

In **Audio MIDI Setup** (Applications > Utilities) set "MacBook Pro Microphone" (or "MacBook Air
Microphone") and the MacBook's speakers to **48,000 Hz**, so macOS does not resample the
ultrasound away. If the microphone offers a format with more than 1 channel, choose it: the
recorder then uses two channels and gets the second direction cue back. Disconnect AirPods and
other Bluetooth audio.

**9.2 Self-test** (1 minute, once per day or after moving the laptop):
```bash
.venv/bin/python -m sonar.selftest --in-device MacBook --out-device MacBook
```
Read its verdict and recommended profile as in README section 1. If it recommends another
profile, use it everywhere: `SONAR_PROFILE=<name>` for recording, `--profile <name>` for training
and for the engine.

**9.3 Record everyone**, one run per person (asks for the name and the repetitions, then records
scene, gestures and breathing, pausing before each one; about 10 minutes per person at 12 reps):
```bash
scripts/mac_record_all.sh
```
The same rules as the HP recordings: everyone their own name, fixed volume, charger in, the lid
at the demo angle; more people beats more repetitions. `SONAR_BACKEND=sim scripts/mac_record_all.sh`
rehearses the whole flow with the simulator (no sound, nothing real recorded). Ctrl+C during a
session ends it early and keeps what was recorded; Ctrl+C at a prompt quits.

**9.4 Train on the MacBook** (MPS), or send the data to the 4090 (9.5):
```bash
scripts/train_4090.sh --device mps --size small --release models-release-mac    # about 15 min (estimate)
scripts/train_4090.sh --device mps --release models-release-mac                 # full size: about an hour, mostly synthetic data on the CPU
```
It detects the single microphone channel and makes the synthetic data to match. The model lands
in `models/`, where the engine looks. If `--device mps` fails, `--device cpu` gives the same
result, more slowly. On a MacBook with 8 GB of memory add `--workers 4`.

**9.5 Or train on the 4090.** On the MacBook:
```bash
ditto -c -k --keepParent data sonar-data-mac.zip
```
Move the zip to the 4090 box and extract it into its **own** folder (never into the HP's `data/`):
- Windows: `Expand-Archive sonar-data-mac.zip -DestinationPath mac -Force`
- Linux: `unzip -o sonar-data-mac.zip -d mac` (or straight from the MacBook: `rsync -a data/ <user>@<4090-box>:sonar-hack/mac/data/`)

Then step 6 with `--data mac --release models-release-mac`, and step 8 with
`git add models-release-mac`. Back on the MacBook:
```bash
git fetch origin model-sat-am
git restore --source origin/model-sat-am --worktree -- models-release-mac
mkdir -p models && cp models-release-mac/gesture.onnx models-release-mac/gesture.json models/
```

**9.6 Run the demo** on the MacBook:
```bash
.venv/bin/python -m sonar.engine --in-device MacBook --out-device MacBook
```
It must print `classifier: model (gesture.onnx)`. An HP model refuses to load here (it expects
two microphone channels), which is the point.

## 10. The same steps by hand

This is what the script runs (Linux/macOS paths; on Windows use `.venv\Scripts\python`). Useful
to repeat one step. The script's log shows the exact commands with its cache folder names.

```bash
# 2. synthetic data (every core; ~3x the Mac's 650 clips per gesture). MacBook recordings: add --mics 1
.venv/bin/python -m datagen.generate --out train_out/synth/manual --per-class 2000 --negatives 2
# 3. one recorded session -> training pictures
.venv/bin/python -m sonar.dataset data/20261114-101500_alice_gestures --out train_out/sessions/alice1.npz
#    synthetic gestures over the real "nobody gesturing" recordings (comma-separated)
.venv/bin/python -m datagen.generate --out train_out/synth_bg/manual --per-class 1000 --seed 1 \
    --background data/20261114-102300_alice_scene/audio.wav,data/20261114-103000_bob_scene/audio.wav
# 4. pre-training (the --labels and --width must match the fine-tuning)
.venv/bin/python -m sonar.train --dataset train_out/synth/manual/dataset.npz \
    --labels swipe_left,swipe_right,push --width 24 --device cuda --out train_out/pretrain/manual
# 5. fine-tuning, scored on people it never saw
.venv/bin/python -m sonar.train --data data --dataset train_out/synth_bg/manual/dataset.npz \
    --init train_out/pretrain/manual/gesture.pt --labels swipe_left,swipe_right,push --width 24 \
    --val-person guest3,guest7 --device cuda --out models
```
`sonar.train --data data` also finds every session folder under `data/` on its own (unreadable
ones are skipped with a warning). If the laptop's left/right wiring is reversed, add
`--swap-mics` and/or `--swap-speakers` to both `datagen.generate` commands.

## 11. GPU pipeline: the data is made on the GPU too

**What changed.** Until now only the model step ran on the GPU. Making the synthetic clips took
about one CPU-core-second per clip, 93 % of it in one loop of `datagen/render.py` (the echo of
every body part, added up path by path), and every new background recording meant another 5-12
minutes of all cores at 100 %. Now, with `--device cuda`:

| step | before | now |
|---|---|---|
| synthetic clips (`datagen`) | numpy, one clip per core | the echo paths, the room, the microphone, the rounding: batches of a few hundred clips on the GPU (`datagen/render_torch.py`); the CPU draws the random numbers (the same ones, in the same order) and attaches the labels |
| the echo loop inside that (86 % of the rendering) | numpy | one hand-written CUDA kernel (`datagen/echo_paths.c`, compiled at run time by the compiler library inside PyTorch; no Visual Studio needed). It checks itself against the plain PyTorch code first and is switched off automatically if anything is wrong (`--gen-kernel off` does it by hand) |
| Doppler pictures of the clips and of the recordings | the engine's code, one clip per core, including the chirp receiver, presence and breathing that gesture pictures never use (a quarter of that step) | the gesture chain only, batched on the GPU (`sonar/features_torch.py`); where a movement burst starts and ends is still decided by the engine's own code |
| pre-training | after all the data was made | runs while the clips on real backgrounds are being made |
| sweep (`--sweep`) | 9 runs one after the other | 4 at a time on the one GPU (`--sweep-jobs`) |
| the training step itself | already on the GPU (section 13) | unchanged |
| the gestures themselves | formula strokes | half of them replay real tracked hand paths (section 12; `--motion none` for formulas only) |

The files are the same as before: same `dataset.npz` layout, shapes, number formats, labels,
manifest, cache folders, `gesture.onnx / .json / .pt`, `TRAINING_RUN.json` (which now also says
where the data was made and what the picture check found). For the same seed the GPU makes the
same clips as the CPU code, so a model trained on either sees the same data.

**The picture check (`scripts/check_gpu_parity.py`).** A second implementation can drift from
the first without anyone noticing. So before any data is made on a GPU, `train_4090.py` makes a
fixed set of 1,000 clips with both codes and compares them: every gesture, every kind of
non-gesture, clips on a recording, one-microphone clips, swapped channels, and hard cases (fastest
strokes, clipping, +-120 ppm clock mismatch, loudest and quietest rooms). If `data/` has
recordings, their audio goes through both feature codes too. It refuses to train if they differ.
A pass is remembered for this exact code, PyTorch and GPU (in `train_out/parity/`, whatever `--out` says); `--recheck-parity` repeats it,
`--skip-parity` skips it (not recommended). What it measured here (CPU reference against the same
GPU code run on a CPU; the 4090 repeats this on its own hardware in its first run):

| compared | limit | worst of 1,000 clips |
|---|---|---|
| moving echoes, before rounding (difference / signal) | 0.00002 | 0.00000066 |
| direct sound and room, before rounding | 0.00000001 | 0.0000000004 |
| audio right before the 16/24-bit converter | 0.0001 | 0.0000024 |
| final audio | never more than one converter step apart, and no more samples one step apart than rounding predicts | held on every clip (quiet 16-bit clips: about 2 samples in a million flip; a deliberately clipped clip: 2.7 %) |
| feature code, fed identical audio: largest picture difference (pictures are 0..1) | 0.0001 | 0.00000024 |
| ... and the bursts it finds | identical | identical on all 1,000 |
| end to end: clips with a burst boundary or label that differs | 2 % | 0.2 % (2 clips) |
| end to end: pictures, average difference | 0.004 | 0.0010 (typical clip: 0.000001) |
| end to end: pictures, largest single pixel | 0.12 | 0.026 |

Why the end-to-end pictures are not bit-identical: the echo sum is added up in float32 (in the
CPU code too) and the order of the additions differs, so the audio differs by a few parts in ten
million before the converter. The converter then rounds to 16 or 24 bits; a sample that sits
almost exactly between two steps can round the other way. Such a sample is a click 100 dB below
full scale: it moves a few noise-floor pixels by up to 0.03 and nothing else, and in about 2
clips in 1,000 it moves a burst boundary by one frame. The limits sit far above that and far
below what a real mistake (wrong delay, channel, filter) produces. The limits and the reasoning
are at the top of `datagen/parity.py`. `tests/fixtures/gpu_parity_reference.npz` stores a small
summary of the CPU results, so a later change to either code is noticed.

**Commands on the 4090** (Windows shown; Linux: `.venv/bin/python`). `HANDOFF_LANCE_GPU.md` has the same
with the expected output of each step and what to send back:
```powershell
.venv\Scripts\python -m datagen.echo_kernel                # the CUDA kernel's self-check (1 min): "KERNEL OK ..." or the reason
.venv\Scripts\python scripts\check_gpu_parity.py          # the picture check alone (1-3 min); exit code 0 = same pictures
.venv\Scripts\python scripts\bench_gpu.py                 # clips per second for every stage, GPU load and memory (3-5 min)
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1            # the full run; data on the GPU, check included
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --sweep    # + 9 fine-tuning runs, 4 at a time
```

**Speed, measured on the RTX 4090** (Lance's runs of Nov 14; the bench output is in `round2/` on branch
`model-sat-am`). "Clips a second" is end to end: random draws, rendering, pictures, labels.

| | clips a second | where the time goes |
|---|---|---|
| old CPU code, 24 processes | 5.5 | 1.07 s of one core per clip; that machine's 24 threads together do only 6x one core |
| GPU path, echo loop as plain PyTorch operations | 32 | the echo loop: 15.3 of 19.7 ms of GPU per clip |
| GPU path, echo loop as the fused CUDA kernel (the default) | 57 in the 384-clip bench, 79 over a 12,000-clip run | echo loop 1.96 ms per clip (7.8x less); rendering alone 162 clips/s, Doppler pictures 327 clips/s |

- A full training run went from 15.9 to 10.6 minutes, and making its 24,000 clips from about 10 to 4.9 minutes.
- What sets the pace now is the CPU: the random draws reach about 100 clips a second with 24 processes on that
  machine, and the GPU waits for them (load 49 % on average in the bench). Two thirds of that CPU time is not
  drawing but shaping the hall noise and the gain wander (two long FFTs, two envelopes, a power per sample).
  `--gen-noise gpu` leaves only the drawing to the workers and shapes on the GPU: the same audio to 1e-12 and,
  on the Mac, bit-identical data sets, with the workers' stage 1.7x faster (145 -> 247 plans a second with 4
  processes).
- The Doppler pictures cost 3.1 ms of GPU per clip there, a third of all GPU time, nearly all of it in one long
  filter computed as a strided convolution in float64. `--gen-features fft` computes the same sum through FFTs
  (same numbers to 2e-15, bit-identical data sets on the Mac); on a GPU that is far less arithmetic.
- Both options are **off by default until they have run on the 4090**. `bench_gpu.py` times them together in a
  block headed `NEW OPTIONS`, next to the default, and `check_gpu_parity.py --noise gpu --features fft` checks
  them. If the estimates hold (workers about 170 plans a second, GPU about 7 ms per clip), the pipeline goes
  from 57-79 to roughly 130 clips a second with the GPU busy nearly all the time. That is an estimate.
- `--gen-precision fast` (float32 FFTs) buys nothing there (52 clips/s against 57) and moves a burst in about 1
  clip in 25, so `ref` stays the default.
- The kernel's self-check on that card: echoes within 2.6e-07 of the PyTorch path. The picture check passed on
  1,040 clips with the kernel in use.
- On the Mac Studio (for comparison): the old code needs 0.92 s of one core per clip; the GPU path on the Apple
  GPU (M1 Max, all float32, no kernel) makes 13.7 clips a second against 6.8 for the old code on 6 cores.

**Options.**

| option | what it does |
|---|---|
| `--gen-device cpu` | the original CPU code for clips and session pictures (the reference; no picture check needed) |
| `--gen-precision fast` | float32 for the three big FFT steps instead of float64. Possibly faster (see `bench_gpu.py`); about 1 clip in 25 then gets a burst boundary moved by a frame, so the default is the exact `ref` |
| `--gen-kernel off` | the echo loop as plain PyTorch operations instead of the fused CUDA kernel (same pictures, slower). Automatic when the kernel's self-check fails |
| `--gen-noise gpu` | the worker processes only draw the random numbers of the hall noise and the gain wander; the GPU shapes them (about half the CPU work per clip, same audio). Default `cpu` until measured on the 4090 |
| `--gen-features fft` | the feature code's long filter through FFTs instead of a strided convolution (same pictures, much less GPU arithmetic). Default `conv` until measured on the 4090 |
| `--gen-batch N` | clips per GPU batch (default: fills the card's free memory; a batch that does not fit is split in two automatically and the log says so) |
| `--gen-gpus N` | on a machine with several GPUs: how many to use (default all; clips are rendered on each, training runs are dealt out in turn) |
| `--skip-parity`, `--recheck-parity` | skip / repeat the picture check |
| `--calibration auto\|none\|FILE` | the virtual laptop the clips are drawn around: fitted to the real recordings (section 12b) or the generic one |
| `--sweep-jobs N`, `--no-overlap` | training runs sharing the GPU (default 4) / make all data first, then train |
| `--gen-device mps` | Apple GPU, opt-in. Checked on an M1 Max: it passes the picture check with its own, looser limits (Apple GPUs have no float64, so about 1 clip in 13 gets a burst boundary moved by a frame) and is about 2x faster than 6 CPU cores. `--gen-batch 32` on a Mac with 32 GB or more |

**Falling back.** If the picture check fails, or the GPU path misbehaves in any way: add
`--gen-device cpu`. That is the old pipeline, byte for byte (checked: the same `dataset.npz` and
`manifest.jsonl` as before this change, for three kinds of data set). Send the report the check
names (`train_out/parity/last_*.json`) and the log.

## 12. Real hand movements in the synthetic data

Until now every synthetic gesture was a formula: a smooth stroke, fast in the middle. Half of them
now replay the path of a **real hand**, tracked from webcam videos of 15 people (the other half and
everything around the hand, the arm, the body, the room, are as before).

- The paths are in `datagen/motion_library/unica_v1.npz` (132 kB, committed): 123 movements, left,
  right, towards and away, from the University of Cagliari gesture videos (CC BY 4.0; credit and
  details in `datagen/motion_library/README.md`). Nothing needs to be downloaded to train.
- What is real: how a stroke speeds up and slows down, how the path curves, how the palm turns,
  the wind-up before it and the movement after it. What is set by us: where in front of the screen
  it happens, its size and tempo (the demo's gesture range), and left/right by mirroring.
- The label of a clip comes from the direction the tracked hand really travels, not from the
  video's file name.
- `--motion none` gives the old formula-only data; `--motion-share 0.3` changes the share
  (default 0.5); `--motion <file>` uses another library. The choice is part of the cache key and is
  written into `TRAINING_RUN.json`.
- The picture check also runs 40 real-path clips through both the CPU and the GPU code.
- To add your own: film each gesture a few times with the demo laptop's webcam and run
  `datagen/mocap_extract.py` on the clips (its header and the library's README have the commands).
  That gives paths with exactly the demo's geometry.

**Does it help? Not known yet.** An earlier version of this section showed an experiment on synthetic test
clips (a model trained on formula gestures recognised 0.26 of "real-path" gestures, one trained on the mix
0.51). That experiment is void, and so is the "motion library" arm of the 4090's round-2 run (0.32 of the
held-out person's gestures against 0.31 without it): until Nov 14 a bug replayed every library path at the
fastest tempo the code allows, a 0.07 s stroke, so neither tested real movements. (A loaded library clip
carried no peak speed, and the tempo was computed from it.) It is fixed, with a test
(`tests/test_datagen_calibration.py`). The real test is the held-out person's score with and without
`--motion none`; round 3 (`HANDOFF_LANCE_CALIBRATION.md`) measures it.

## 12b. The virtual laptop fitted to the real recordings

`datagen/hardware.py` draws a different imaginary laptop for every clip, around a guess of a 15-inch HP. For the
MacBook the recordings were made on, that guess is wrong where it matters. Read off the recordings:

| | the generic laptop assumes | the MacBook recordings show |
|---|---|---|
| microphone | two, on top of the lid | one, on the deck next to the LEFT speaker |
| speaker tones at the microphone | 40-78 dB below what is played | 15-21 dB below (left tone 6.4 dB louder than the right) |
| which tone tells left from right | mostly the two microphones | the LEFT speaker's tone: a left swipe moves it up about 50-100 Hz, a right swipe down; the right tone hardly moves |
| hand height during a gesture | 3-10 cm | 10-25 cm above the keys |
| fastest moment of a stroke | 0.5-2.2 m/s | 0.9-5.5 m/s |

`datagen/calibration.py` fits the simulator to this: it cuts the recorded gestures into pictures with the
engine's own code, reduces every picture to six numbers (how strong, how wide in Doppler, how saturated, how
filled, and where each tone's energy sits), does the same for simulated clips, and changes one simulator
setting at a time, keeping a change only when the simulated numbers come closer to the recorded ones. The
result is a small JSON file; `calibration/macbook_v1.json` was fitted on the Mac (20 trial clips per gesture
and setting), and the 4090 refits it with 150. From that refit on, the trial clips are laid over the real
"nobody gesturing" recordings, as most training clips are, and only the echo level of clips without a recording
underneath is fitted on pure simulation. Reason, measured on the recordings: next to the tones the real hall is
10-25 dB louder at low Doppler than pure simulation (people moving about), and the engine scales every picture by
that floor. With simple picture numbers a plain classifier tells pure simulation from recordings almost always
(0.98, where 0.5 would mean "cannot"), clips laid over recordings less easily (0.92).

How far simulated gesture pictures are from recorded ones (0 = same averages, 1 = one "scale" apart on average;
scales: 6 dB, 60 Hz, 0.05, 0.15, 40 Hz, 40 Hz): **1.72 with the generic laptop, 0.41 fitted.** The same in plain
numbers, left swipe / right swipe, synthetic movement laid over real recordings (48 clips each, so the Hz
figures carry about +-20):

| | left tone moves by | right tone moves by | picture peak |
|---|---|---|---|
| recorded gestures | +53 / -42 Hz | -4 / +13 Hz | 43 dB |
| generic laptop (what rounds 1 and 2 trained on) | +2 / -17 Hz | -25 / +34 Hz | 26-31 dB |
| fitted laptop | +67 / -50 Hz | +24 / -10 Hz | 38-42 dB |

So the generic synthetic gestures were faint and carried the left/right cue on the wrong tone, with the opposite
sign. A model trained mostly on them is taught the wrong thing about this laptop.

What you need to know to use it:
- `train_4090.py --calibration auto` (the default) uses the newest `calibration/macbook_v<N>.json` when the
  recordings have one microphone, `--calibration none` gives the generic laptop (the old behaviour, byte for
  byte), `--calibration <file>` any other file. `datagen.generate --calibration <file>` is the same switch there.
- **Honest scores need one file per held-out person.** How the laptop is built does not depend on who sits in
  front of it, so that part is fitted on every recording. How people move (hand height, speed) does, so
  `python -m datagen.calibration --real <recordings> --folds calibration/macbook_v<N>.json` writes
  `macbook_v<N>_no_<person>.json`, fitted without that person (lance and lance2 count as one). `--calibration auto`
  picks that file by itself when one person is held out, and says so in the log.
- `scripts/make_calibrated_sets.py` makes one synthetic set per held-out person and per recipe
  (`train_out/synth_cal/...`; its header has the list) for the leave-one-person-out scoring. With
  `--rel-off-db 12` it makes the same clips again with the bursts ended 12 dB below their own peak (the engine's
  `seg_rel_off_db`, off by default; `SONAR_SEG_REL_OFF_DB=12` switches it on everywhere) into folders named
  `..._rel12`. Without that option the sets are cut the usual way whatever the shell's variable says.
- A recording that somebody set aside in a folder named `excluded` (e.g. `mac/excluded/<session>`: left and
  right turned out flipped) is skipped by the training script, the fit and the set maker.
- With a fitted laptop, three more things change in the synthetic clips, because the recordings showed them:
  a stroke is never faster than a hand can be accelerated (30-150 m/s^2, so a short stroke cannot be as fast as
  a long sweep); the hand reaches the start of the stroke within 0.7 s and strokes almost at once; and a burst
  that holds less than half of the stroke is left out instead of being labelled "none". (The engine cuts a burst
  after 1.4 s. On the loud fitted laptop the way to the start already opens the burst, so that cut can fall
  inside the stroke. In a 96-clip test 29 % of the fitted gesture clips ended up without a labelled stroke and
  added stroke fragments as "none"; with these changes 11 % have no labelled stroke (13 of 120) and no fragment
  is called "none".)
- The picture check includes clips drawn around the fitted laptop. In those, about 2 clips in 100 get a burst
  moved or dropped by rounding (10 times the generic rate): the fitted clips are loud, so there is more movement
  right at the burst detector's threshold. Every stage before the rounding agrees as closely as for generic clips.

Limits, plainly:
- It is a fit of summary numbers by trial and error, not gradient descent through the renderer: for a recorded
  gesture the true path of the hand is unknown, so there is nothing to line a single simulated clip up with.
- What is measured straight off the recordings rather than searched: the tone levels at the microphone, and (from
  the 4090's refit on) the noise floor around the tones, -81 dBFS as white noise over the whole band on these
  recordings. Speaker and microphone colouration, the reflectivity of hand against arm, and the room's reflections
  were not fitted one by one; clips laid over real recordings bring the real room and noise with them. In
  `macbook_v1.json` their echo level (`"overlay"`) was set from a 48-clip comparison; the refit searches it.
- Still different from the recordings: the real right-tone pictures are brighter than the left-tone ones by
  about 0.10 (simulated: about equal), and lance's swipes show as broadband stripes in both tones that no
  setting reproduces well.
- Whether it raises the held-out score is what round 3 measures. It has not been measured yet.

## 13. Speed: the training step on the GPU

`sonar.train` keeps the whole dataset on the GPU (float16, about 1.2 GB for 20,000 pictures), and
draws every batch, applies all the augmentation (time shift, mirror, faster/slower, louder/quieter,
noise, clicks, masks) and trains there, without going back to the CPU inside an epoch. On CUDA it
also uses bf16 mixed precision, cuDNN autotuning, TF32, a fused optimizer, a batch of 256 with the
learning rate scaled by the square root of the batch growth, and the channels_last layout when a
short timing test at the start shows it is faster on this GPU. CPU and Apple MPS keep full
float32 and the original batch of 64.

`--compile` (torch.compile) is off by default: it needs Triton (Linux only) and its compile time
is longer than the whole training of this small network, so it does not pay off here.

Measured on the Mac Studio's CPU while another long job was running (its GPU was not usable, so
there are no GPU timings yet):
- the old loop augmented every example in Python on one CPU core, taking turns with the GPU step, so
  it could never train faster than one core could augment: at most about 2,350 examples per second
  on this Mac, and less in practice. On the 4090 PC that limit depends on its CPU. The new path
  removes this per-example CPU work; its speed on a GPU is not measured yet;
- on the CPU alone the new batched augmentation does 4,300-5,200 examples per second using 8 threads;
- a whole training run on the CPU is only ~5 % faster with the new path (22.5-22.9 s vs 24.0-24.1 s
  for 4 epochs, two runs each): there the network itself is most of the time;
- `--augment numpy` reproduces the old training exactly (byte-identical model files on CPU). Each
  new augmentation step gives the old step's numbers to within 0.00002 for the same random choices
  (`tests/test_training.py`); the random draws differ, so runs are not identical between the two;
- batch 256 with the scaled learning rate scored about like batch 64 in one CPU run (accuracy 0.81
  vs 0.82, macro-F1 0.73 vs 0.71, more gestures caught but more false triggers: 0.12 vs 0.09). It is
  one run on a small subset, so CUDA's 256 is a reasonable default to confirm on the 4090, not a
  proven match; `--batch 64` gives the exact recipe the Mac's numbers were made with.

## 14. Troubleshooting

| problem | fix |
|---|---|
| `--device cuda, but PyTorch cannot see a CUDA GPU` | step 3.3 |
| `running scripts is disabled on this system` | use `powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 ...` |
| `No Python environment at .venv` | steps 2 and 3, in the repo folder |
| `the recordings come from more than one device` | HP and MacBook sessions are mixed: give each laptop its own folder and use `--data` (section 9.5) |
| `CUDA out of memory` | while training: `--batch 128` or `--sweep-jobs 2`; while making data: `--gen-batch 128` (normally the generator halves its batch by itself and says so) |
| `the GPU does NOT make the same training pictures as the CPU code` | add `--gen-device cpu` to train with the CPU generator, and send `train_out/parity/last_*.json` with the log (section 11, `HANDOFF_LANCE_GPU.md`) |
| the computer becomes sluggish while generating | `--workers 8` (default: every core; on the GPU path the workers only draw random numbers and attach labels) |
| `no usable recordings` although data/ has folders | read the session table: each skipped folder says why (profile, too short, unreadable, simulator) |
| a session "only N s of audio" | it was copied while still recording: copy it again once the recorder has finished |
| `--init ... was trained for the classes ...` | pre-training and fine-tuning need the same `--labels` and `--width` (the script does this for you) |
| MacBook: `no input device matching 'MacBook'` | the device names differ: `.venv/bin/python -m sonar.engine --list-devices`, then `SONAR_IN_DEVICE=<part of the name>` (and `SONAR_OUT_DEVICE`) |
| MacBook: recordings are silent / self-test SNR near 0 | allow the Terminal to use the microphone: System Settings > Privacy & Security > Microphone |
| need a CPU-only fallback | `--device cpu --size small` (about 15 minutes on a laptop CPU, estimate) |

# Handoff, round 3: training with the laptop fitted to the real MacBook recordings (for Lance's RTX 4090)

Everything here runs on the 4090 box, from the repo folder, in **Windows PowerShell**, on branch `gpu-train`
after `git pull`. The new code was smoke-tested on a Mac with the GPU code running on the CPU; it has not run
on a CUDA card yet, so the short steps come first. If one fails, the last section says what to do.

## What round 2 told us (thank you), and what changed because of it

| round 2 on your machine | what it means / what changed |
|---|---|
| `KERNEL OK`: echo loop 4.1 ms per clip against 24.0 | the hand-written CUDA kernel works on your card; rendering went from 52 to 162 clips/s |
| end to end 57 clips/s (10.4x the 24 CPU threads), GPU load 49 % | the CPU side (the random draws) is now the slow part, at about 100 clips/s; two new options go after it, off by default and timed in the optional part D |
| `PARITY OK` on 1,040 clips | the GPU makes the same pictures as the CPU code |
| the parity check ran again in the second training run (4.6 min) | fixed: a pass is now remembered in `train_out\parity\` whatever `--out` says |
| held-out mahimn: 0.32 of gestures recognised with the motion library, 0.31 without | **two things were wrong with that test, see below** |

Two findings that change how to read the scores of rounds 1 and 2:

1. **The score included the flipped session.** `--data mac` also picked up `mac\excluded\20261114-001248_mahimn_gestures`
   (its README says left and right are flipped). It sat in the held-out person's test set, and the result
   table shows exactly that: of mahimn's 27 left swipes 14 were called left and 12 right, and the same for
   the right swipes, while pushes and "no gesture" were mostly correct. A model that is right on the good
   session and "wrong" on the flipped one produces that table. From now on any session inside a folder named
   `excluded` is skipped automatically (the log lists it as `SKIP: it is in a folder named 'excluded'`).
   (The recall session re-scored the old recipe with that session's labels exchanged: 0.65 of the held-out
   person's gestures recognised at 0.055 false triggers. That, not 0.30, is the number to beat.)
2. **The real hand paths were replayed far too fast.** A bug in my round-2 code played every tracked hand path
   at the fastest tempo it allows (a 0.07 s stroke), so the "motion library" run did not test real movements.
   Fixed; there is a test for it now.

## What this round does, in plain words

The first two rounds trained on synthetic clips of an imaginary laptop that is nothing like the MacBook the
recordings come from: the gestures came out faint and slow, with the left/right cue on the wrong speaker tone.
The simulator has now been **fitted to the real recordings** (`calibration\macbook_v1.json`): the microphone
sits on the deck next to the left speaker, the tones arrive much louder, hands pass 10-25 cm above the keys.

| part | what | time on the 4090 | why |
|---|---|---|---|
| A | fit the laptop again on the GPU with more clips, then once without each person | about 12 min | a better fit than the Mac could do; and a score on a held-out person is only honest if the synthetic clips were not drawn around that person's habits |
| B | two training runs: with the fitted laptop, and the old way | about 12 + 7 min | the headline: what is the honest score without the flipped session, and what does the fitted laptop add? |
| C | 12 synthetic sets, one per scoring fold | 45-70 min, unattended | the recall session's scoring suite reads these |
| D | optional: check and time the next speed step | about 8 min | decides whether it becomes the default |
| E | after C: the bg and plain sets again, cut the new way | about 30 min, unattended | the recall session's newest recipes train on pictures cut that way |

If time is short: A and B matter most. C can run afterwards and can be stopped and started again.

`mac` is the folder with the recorded MacBook sessions, as before. Close the other GPU programs if you can.

## Step 0. Get the code and a folder for the results

```powershell
git checkout gpu-train
git pull
New-Item -ItemType Directory -Force round3 | Out-Null
```

## Part A. The fits (about 12 minutes)

```powershell
.venv\Scripts\python -m datagen.calibration --real mac --out calibration\macbook_v2.json --start calibration\macbook_v1.json --device cuda --clips 150 2>&1 | Tee-Object round3\r3_fit.log
.venv\Scripts\python -m datagen.calibration --real mac --folds calibration\macbook_v2.json --device cuda --clips 150 2>&1 | Tee-Object round3\r3_folds.log
```
Line 1 starts from the Mac's fit and tries about 45 settings (where the microphone sits, how strong the echoes
are, how fast and how high people move), 450 trial clips each, keeping a change only when it brings the simulated
gesture pictures closer to the recorded ones. New since the Mac's fit: the trial clips are laid over your real
"nobody gesturing" recordings, as most training clips are (the real hall is much louder at low Doppler than pure
simulation, and the pictures are scaled by that). It must NOT list `20261114-001248_mahimn_gestures` among the
recordings it reads. Expected (numbers will differ):
```
  trial clips are laid over 8 recording(s) with nobody gesturing
  generic laptop: gap 1.xx   (trial clips laid over the real no-gesture recordings)
  starting point: gap 0.xx
    hardware.mic_pos = [...]: gap 0.xx
    overlay.echo_ref_db = [...]: gap 0.xx  <- better
    ...
  clips without a recording underneath (pure simulation): their echo level
    hardware.echo_ref_db = [...]: gap 0.xx
swipe_left    real         43.14  215.78  ...
swipe_left    sim before   2x.xx  1xx.xx  ...
swipe_left    sim after    4x.xx  2xx.xx  ...
gap (0 = same means, 1 = one scale apart on average; trial clips laid over the real no-gesture recordings): generic laptop 1.xx, starting point 0.xx, fitted 0.xx; pure simulation with its own echo level: 0.xx
wrote calibration\macbook_v2.json in ... s
```
It writes working copies of the recordings (about 300 MB) into `calibration\work\`; that folder is not committed.
Line 2 chooses "how people move" three more times, each time without one person:
```
  without aarsh (aarsh): fitted on lance, lance2, mahimn -> peak speed [..] m/s, hand height [..] m (gap 0.xx)
  without lance (lance, lance2): fitted on aarsh, mahimn -> ...
  without mahimn (mahimn): fitted on aarsh, lance, lance2 -> ...
wrote calibration\macbook_v2_no_aarsh.json  (and _no_lance, _no_mahimn, and macbook_v2_folds.report.json)
```
If line 1 fails: go on with line 2 using `--folds calibration\macbook_v1.json`; everything below then uses the
Mac's fit by itself (it always takes the newest `calibration\macbook_v<N>.json` it finds).
If line 2 fails too: go on with part B; it prints `NOTE: there is no file fitted without mahimn ...` and still runs.

## Part B. Two training runs (about 12 + 7 minutes)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --data mac --val-person mahimn --out train_out_cal --models train_out_cal\models --release models-release-cal 2>&1 | Tee-Object round3\r3_train_cal.log
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --data mac --val-person mahimn --calibration none --motion none --out train_out_base --models train_out_base\models --release models-release-base 2>&1 | Tee-Object round3\r3_train_base.log
```
Run 1 = the fitted laptop + real hand paths (the new default). Run 2 = the old imaginary laptop, formulas only.
Both now leave the flipped session out, so run 2 is the like-for-like baseline. Each writes to its own folders;
nothing of the earlier runs is replaced. Near the top of run 1 look for:
```
  20261114-001248_mahimn_gestures   mahimn   gestures  ...  SKIP: it is in a folder named 'excluded'
virtual laptop: fitted to real recordings (calibration\macbook_v2_no_mahimn.json; ...); how people move was fitted without mahimn, who is held out for the score
GPU parity check: ...        (runs ONCE for this new code, about 5 minutes; run 2 then says "passed earlier")
```
and at the end of each run `Honest score on mahimn: ...` and `Verdict: ...`.
**The two `Honest score` lines, and the two small tables above them (rows = truth), are the headline of this round.**

## Part C. The 12 synthetic sets for the scoring suite (45-70 minutes, unattended)

```powershell
.venv\Scripts\python scripts\make_calibrated_sets.py --data mac --arms bg,motion,plain 2>&1 | Tee-Object round3\r3_sets.log
```
What it makes (each folder holds `dataset.npz`, `manifest.jsonl`, `done.json`; about 16,000 clips each; `v2`
becomes `v1` if part A's first line failed):
```
train_out\synth_cal\macbook_v2_bg\{all,no_aarsh,no_lance,no_mahimn}\       70 % of the clips laid over the real "nobody gesturing" recordings
train_out\synth_cal\macbook_v2_motion\{all,no_aarsh,no_lance,no_mahimn}\   half of the gestures replay real hand paths
train_out\synth_cal\macbook_v2\{all,no_aarsh,no_lance,no_mahimn}\          gestures from the formulas only
train_out\synth_cal\INDEX.json                                              what is in each
```
`no_lance` = the set for the fold that tests lance (drawn without lance's habits, never laid over lance's or
lance2's own recordings). `all` = for the final model. The sets stay on this machine (2-3 GB together, too big
for git); the recall session's scoring suite reads them from these folders.

Expected at the end: a table of the 12 sets with picture counts and seconds, then
`DONE: 12 set(s) under ...train_out\synth_cal (.. min of generation in this run)`.
- It can be stopped (Ctrl+C) and started again with the same command: finished sets are skipped.
- Short on time: add `--per-class 1000` (half the size, half the time) or use `--arms bg,motion` (8 sets).
- `... batch(es) did not fit in GPU memory and were split`: harmless (it halved the batch itself).

## Part D, only if there is time: the next speed steps (about 8 minutes)

```powershell
.venv\Scripts\python scripts\check_gpu_parity.py --clips 200 --noise gpu --features fft --json round3\r3_parity_new.json 2>&1 | Tee-Object round3\r3_parity_new.log
.venv\Scripts\python scripts\bench_gpu.py --only gpu --precisions ref 2>&1 | Tee-Object round3\r3_bench.log
```
Your round-2 bench showed two things holding the pipeline back: the CPU's random draws (100 clips a second),
and the Doppler pictures (3.1 ms of GPU per clip, a third of the GPU's work). Two new options go after them:
`--gen-noise gpu` moves the heavy half of the CPU's work to the GPU, `--gen-features fft` computes the
pictures' long filter a cheaper way. Both are off by default and nothing above uses them.
Line 1 checks that they make the same pictures (must end with `PARITY OK`, and say
`hall noise and gain wander shaped on the device; feature filter through FFTs` near the top). Line 2 times the
pipeline three ways; the one to look at is the block headed `... NEW OPTIONS: --gen-noise gpu --gen-features fft`:
its `random draws`, `Doppler columns`, `END TO END` and `GPU load` lines against the first block's. If both
are fine they become the default next round.

## Part E, after part C: the same sets with the new burst cutting (about 30 minutes, unattended)

```powershell
git pull
.venv\Scripts\python scripts\make_calibrated_sets.py --data mac --arms bg,plain --rel-off-db 12 2>&1 | Tee-Object round3\r3_sets_rel12.log
```
Why: the data-engine session found a better rule for where a movement burst ends (12 dB below the burst's own
peak, instead of waiting for the movement level to fall back to the floor). On this loud laptop the old rule
glued hand raise, stroke and hand drop into one long burst that was chopped after 1.4 s. With the new rule, and
no retraining, the held-out person's score went from 0.65 to 0.73 of gestures recognised (the recall session's
measurement). If the engine switches to it, the synthetic clips must be cut the same way.
What it makes: the same clips as part C (same random seeds; only where the bursts end differs), in
```
train_out\synth_cal\macbook_v2_bg_rel12\{all,no_aarsh,no_lance,no_mahimn}\
train_out\synth_cal\macbook_v2_rel12\{all,no_aarsh,no_lance,no_mahimn}\
```
The `git pull` matters: this option is newer than the commit parts A-D were written for. Parts A-D do not need
to be repeated. `git pull` changes two engine files (`sonar\config.py`, `sonar\gestures.py`; the new rule is
off unless asked for), so a training run started after it checks the pictures once more (about 5 minutes).
Expected at the end: `DONE: 20 set(s) under ...` (the 12 from part C and these 8). It can be stopped and
started again like part C.

## What to send back

The same way as round 2: push to branch `model-sat-am`, in a folder `round3\`:
```powershell
New-Item -ItemType Directory -Force round3\calibration, round3\cal, round3\base | Out-Null
Copy-Item calibration\macbook_v2*.json round3\calibration\ -ErrorAction SilentlyContinue
Copy-Item train_out\synth_cal\INDEX.json round3\ -ErrorAction SilentlyContinue
Copy-Item models-release-cal\TRAINING_RUN.json, models-release-cal\metrics.json, models-release-cal\confusion.txt round3\cal\ -ErrorAction SilentlyContinue
Copy-Item models-release-base\TRAINING_RUN.json, models-release-base\metrics.json, models-release-base\confusion.txt round3\base\ -ErrorAction SilentlyContinue
Get-ChildItem train_out_cal\runs | Sort-Object Name | Select-Object -Last 1 | ForEach-Object { Copy-Item "$($_.FullName)\log.txt" round3\cal\log.txt }
Get-ChildItem train_out_base\runs | Sort-Object Name | Select-Object -Last 1 | ForEach-Object { Copy-Item "$($_.FullName)\log.txt" round3\base\log.txt }
```
then commit `round3\` and `models-release-cal\` on `model-sat-am` and push, as you did for round 2.
What matters most, in this order: `round3\r3_train_cal.log` and `r3_train_base.log` (the two honest scores),
`round3\calibration\` (the fitted files; the Mac needs them back), `r3_fit.log`, `r3_folds.log`, `r3_sets.log`, `INDEX.json`,
and from part D `r3_parity_new.log`, `r3_bench.log` and the `train_out\bench\bench_<time>.json` it names; from part E
`r3_sets_rel12.log` and the `INDEX.json` again.

## If something goes wrong

| what you see | do this | and send |
|---|---|---|
| part A line 1 fails | run line 2 with `--folds calibration\macbook_v1.json`, then go on | `r3_fit.log` |
| part A line 2 fails | go on with part B (it says `NOTE: there is no file fitted without mahimn`); run part C with `--folds all` | `r3_folds.log` |
| the log of part B does NOT show `20261114-001248_mahimn_gestures ... SKIP` and lists it with `use` | the session is not inside a folder named `excluded` on your disk: move it into `mac\excluded\` and start the run again | the first 30 lines |
| `PARITY FAILED` inside part B | add `--gen-kernel off` and run it again; if it fails again add `--gen-device cpu` (the old CPU generator, slow but the reference) | the table it printed |
| an error mentioning `echo_kernel`, `nvrtc` or `cuLaunchKernel` | add `--gen-kernel off` to that command | the error text |
| part C says `SKIPPED, macbook_v2_no_<name>.json does not exist` | part A line 2 did not write that file | `r3_folds.log`, `r3_sets.log` |
| `CUDA out of memory` in part B | add `--sweep-jobs 2 --no-overlap` | the last 30 lines |

## Not in this round (noted from your message)

- The model made with one microphone (`models-release-mac`, input 2 x 64 x 80) only loads when the engine runs
  the one-microphone profile; with the default two-microphone profile it fails with (5,64,80) against (2,64,80).
  The model's own `gesture.json` says which it needs (`config.n_mics`, `input_shape`); making the engine read
  that is with the engine's owner, not part of the training handoff.

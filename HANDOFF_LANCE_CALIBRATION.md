# Handoff, round 3: training with the laptop fitted to the real MacBook recordings (for Lance's RTX 4090)

Everything here runs on the 4090 box, from the repo folder, in **Windows PowerShell**, on branch `gpu-train`
after `git pull`. The new code was smoke-tested on a Mac with the GPU code running on the CPU; it has not run
on a CUDA card yet, so the short steps come first. If one fails, the last section says what to do.

## Update, 14 Nov, 2:30 pm: parts C and E are done. Only the optional part F is left

**Both worked, thank you.** All 20 sets are made: the 12 of part C in 28 minutes (13:15 to 13:40) and the 8
`_rel12` sets of part E in 18 minutes (13:43 to 13:59). The eight sets that mix two kinds of clip went
through as expected (24 to 28 of their 64 to 65 batches part-filled). `models-release-cal\` and
`models-release-base\` arrived as well. The code has not changed since `a57e584`: this update is text only.

**Please do not run the set maker again** (`scripts\make_calibrated_sets.py`, the commands of parts C and E).
The 1 pm update below still lists them as steps 2 and 3; both are done. The recall session's training reads
the 20 sets from `train_out\synth_cal` for one and a half to two hours. Run again, either command normally
skips its sets in a few seconds, but it deletes and rebuilds a finished set if something that set depends
on has changed since: a recording added under `mac\`, or a file in `calibration\`.

**What is left of this handoff:** only part F, and it is optional (the same comparison for every person,
about 35 minutes). It wants the GPU to itself, and so does the recall session's training: one at a time,
in the order the coordinator gives.

## Update, 14 Nov, 1 pm: the hang in part E is fixed. What to run now

*(Added at 2:30 pm: everything this section asks for is done: steps 1, 2 and 3 of its list, and the two model
folders have arrived. Do not repeat steps 2 and 3; see the update above.)*

**What happened.** Part E stopped twice on its first set, with the log frozen after `12000 clips on cuda`, the
GPU idle and one Python thread spinning. Your dumps pointed at the right place, thank you. The cause was my
bug in the loop that feeds the GPU, and it has nothing to do with the new burst cutting:

- The `_bg` sets lay 70 % of their clips over the real recordings and make the other 30 % by pure simulation.
  Those two kinds of clip cannot go into the same GPU batch.
- The loop prepares 325 clips ahead and sends a batch when one kind has 217 ready. After the first batch the
  325 were split about 165 / 160: neither kind had a full batch, the queue was full so nothing more was
  prepared, and the loop waited for ever. (Your manifest held 27 lines: exactly the first 8 clips of that set,
  up to the first pure-simulation clip. The same set made here on the CPU gives the same 27.)
- **So every `_bg` set is affected, also the four of part C** (`macbook_v2_bg\...`), with the usual cutting
  too: there the manifest should stop at 19 lines. The `_motion` and plain sets hold one kind of clip and are fine,
  and so is everything `train_4090.ps1` makes (parts B and F).

**The fix** (in `gpu-train` after `git pull`): when the queue is full and no kind has a full batch, a
part-filled batch goes out, the one holding the earliest clip. If the loop ever again has nothing to wait
for, it now stops with a message instead of spinning. Sets with one kind of clip come out byte for byte as
before, so nothing you already made needs to be made again.

**What to run now, in this order** (each can be stopped and started again; finished sets are skipped):

```powershell
# 1. if a set maker is still running and stuck on a _bg set (log frozen, GPU idle): stop it with Ctrl+C
git pull
Select-String -Path datagen\gpu_pipeline.py -Pattern "held_up" -Quiet      # must print True: the fix is here
# 2. DONE at 13:40, do not run it again (see the 2:30 pm update): part C, the 12 sets
#    .venv\Scripts\python scripts\make_calibrated_sets.py --data mac --arms bg,motion,plain 2>&1 | Tee-Object round3\r3_sets.log
# 3. DONE at 13:59, do not run it again: part E, the bg and plain sets cut the new way
#    .venv\Scripts\python scripts\make_calibrated_sets.py --data mac --arms bg,plain --rel-off-db 12 2>&1 | Tee-Object round3\r3_sets_rel12.log
```
How to tell it works: within a minute of `12000 clips on cuda` the first progress line appears
(`  200/12000 clips, ...`), and each `_bg` set ends with a line like
`GPU path on cuda:0: batch 217, ... 0 redraws; 27 of the batches went out part-filled (clips over real recordings and pure simulation cannot share a batch)`.
A `_bg` set should take about as long as the others, about 3 minutes (worked out, not yet measured on your card).

**Two more things, whenever convenient:**

- `models-release-cal\` is not on `model-sat-am` yet (only the logs of part B are). Please add it:
  `gesture.onnx`, `gesture.json`, `gesture.pt`, `metrics.json`, `confusion.txt`, `TRAINING_RUN.json`. It is the
  model from this round that we would try on the laptop. `models-release-base\` too if it is no trouble (the
  old-way model from the same round, to try the two side by side).
- Part F (optional, further down) no longer waits for anything: it needs only parts A and B and can run
  whenever the GPU is free.

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
| E | the bg and plain sets again, cut the new way (needs the fix, see the update above) | about 30 min, unattended | the recall session's newest recipes train on pictures cut that way |
| F | optional, needs only A and B: part B again for each person in turn, three seeds | about 20 min (35 after the `git pull` of the fix) | part B looked good for the fitted laptop but was one person and one run each; this says whether it holds |

If time is short: A and B matter most. C and E can run afterwards, in either order, and can be stopped and
started again. F does not depend on C, D or E. Only one of them at a time on the GPU.

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

## Part C. The 12 synthetic sets for the scoring suite (done on 14 Nov in 28 minutes; do not run it again, see the 2:30 pm update)

```powershell
.venv\Scripts\python scripts\make_calibrated_sets.py --data mac --arms bg,motion,plain 2>&1 | Tee-Object round3\r3_sets.log
```
What it makes (each folder holds `dataset.npz`, `manifest.jsonl`, `done.json`; 12,000 clips each; `v2`
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
- It can be stopped (Ctrl+C) and started again with the same command: finished sets are skipped, and a set
  that was interrupted is made again from its start.
- The four `_bg` sets need the fix described in the update at the top (`git pull` first). Before it they
  stop after a few clips with the GPU idle.
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

## Part E: the same sets with the new burst cutting (done on 14 Nov in 18 minutes; do not run it again, see the 2:30 pm update)

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
The `git pull` matters: it brings the fix for the hang (see the update at the top); without it the first set
stops after 8 clips. Parts A-D do not need to be repeated, and part E does not need part C: the two make
different folders and can run in either order, one after the other.
Expected at the end: a table of the sets made so far, then `DONE: 20 set(s) under ...` (the 12 of part C and
these 8; fewer if part C has not run yet). It can be stopped and started again like part C.

## Part F, optional, needs only parts A and B: the same comparison for every person (about 20 minutes)

What part B showed (thank you): with the fitted laptop the model recognised 32 of mahimn's 36 gestures, against
26 the old way, with 5 false triggers out of 101 non-gestures against 2. That is one person, 36 gestures and
one training run each, so it can still be chance. This part repeats both runs with three seeds, holding out
each person in turn (lance and lance2 are one person). It does not depend on parts C, D or E and does not need
the fix for the hang (these runs only make sets with one kind of clip), so it can run before or after them,
whenever the GPU is free. Nothing else should be using the GPU while it runs.

```powershell
Remove-Item Env:SONAR_SEG_REL_OFF_DB -ErrorAction SilentlyContinue
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --data mac --val-person mahimn --seeds 0,1,2 --out train_out_cal --models train_out_cal\fold_mahimn\models --release train_out_cal\fold_mahimn\release 2>&1 | Tee-Object round3\r3f_cal_mahimn.log
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --data mac --val-person aarsh --seeds 0,1,2 --out train_out_cal --models train_out_cal\fold_aarsh\models --release train_out_cal\fold_aarsh\release 2>&1 | Tee-Object round3\r3f_cal_aarsh.log
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --data mac --val-person lance,lance2 --seeds 0,1,2 --out train_out_cal --models train_out_cal\fold_lance\models --release train_out_cal\fold_lance\release 2>&1 | Tee-Object round3\r3f_cal_lance.log
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --data mac --val-person mahimn --seeds 0,1,2 --calibration none --motion none --out train_out_base --models train_out_base\fold_mahimn\models --release train_out_base\fold_mahimn\release 2>&1 | Tee-Object round3\r3f_base_mahimn.log
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --data mac --val-person aarsh --seeds 0,1,2 --calibration none --motion none --out train_out_base --models train_out_base\fold_aarsh\models --release train_out_base\fold_aarsh\release 2>&1 | Tee-Object round3\r3f_base_aarsh.log
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --data mac --val-person lance,lance2 --seeds 0,1,2 --calibration none --motion none --out train_out_base --models train_out_base\fold_lance\models --release train_out_base\fold_lance\release 2>&1 | Tee-Object round3\r3f_base_lance.log
```
Line 1 makes sure the bursts are cut the same way as in part B (it does nothing if that switch was never set
in this window). The six runs reuse part B's folders. How long they take depends on the code they run on:
- on the commit part B ran on (`ac32fb7`, before the `git pull` of the fix): most of the data is already
  made. The two mahimn runs take about 2 minutes each, the other base runs about 3, the other fitted runs
  about 6: about 20 minutes together.
- after that `git pull`: the generator's code has changed, so the first run checks the pictures once more
  (about 5 minutes) and the synthetic clips are made again (the same clips as before): about 35 minutes.
They do not touch `models-release-cal\` or `models-release-base\`. Expected in each log:
```
GPU parity check: passed earlier for this code on this GPU ...          (after the git pull: once, PARITY OK, then this line)
virtual laptop: fitted to real recordings (calibration\macbook_v2_no_aarsh.json; ...); how people move was fitted without aarsh ...   (the three fitted runs; _no_lance, _no_mahimn for the others)
fine-tuning, scored on held-out person aarsh (never seen in training):   (a table with three rows: seeds 0, 1, 2)
```
If a `lance3` session has been put under `mac\` by then (it was in `mac_hold\` on 14 Nov, which these runs do
not read), hold it out with the other two: `--val-person lance,lance2,lance3` in both lance lines.

Then collect the models that were scored on a person they had not seen (small files, about 11 MB together;
they let the Mac compare the two kinds of model clip by clip, which the tables cannot show, for instance how
many of the false triggers are "pull" movements, for which the model has no class):
```powershell
New-Item -ItemType Directory -Force round3\heldout | Out-Null
foreach ($arm in "cal", "base") {
  Get-ChildItem "train_out_$arm\runs" -Directory | ForEach-Object {
    if (Test-Path "$($_.FullName)\sweep") { Copy-Item "$($_.FullName)\sweep" "round3\heldout\${arm}_$($_.Name)" -Recurse -Force }
  }
}
Get-ChildItem round3\heldout -Recurse -Include *.pt | Remove-Item
```
Git ignores `*.onnx` outside the `models-release*` folders, so when you commit, add this folder with
`git add -f round3\heldout` (a plain `git add round3` would leave the models out without saying so).
If there is no time for the six runs, the last block alone (one minute, no GPU) is still worth sending: it
picks up the two models of part B.

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
Received on 14 Nov, 2 pm (19dedf0, 8929e1f): part E's log and the index of all 20 sets, `models-release-cal\` and
`models-release-base\` (six files each). Thank you; nothing from parts A, B, C and E is missing any more.
What matters most, in this order: `round3\r3_train_cal.log` and `r3_train_base.log` (the two honest scores),
`round3\calibration\` (the fitted files; the Mac needs them back), `r3_fit.log`, `r3_folds.log`, `r3_sets.log`, `INDEX.json`,
and from part D `r3_parity_new.log`, `r3_bench.log` and the `train_out\bench\bench_<time>.json` it names; from part E
`r3_sets_rel12.log` and the `INDEX.json` again; from part F the six `round3\r3f_*.log` and the folder `round3\heldout\`.

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
| a `_bg` set: the log stops after `12000 clips on cuda`, GPU idle, one Python thread busy | you are on the code before the fix: stop it, `git pull`, check with the `Select-String` line in the update at the top, start again. If you cannot pull: add `--gen-batch 48` to the set maker (not tried on your card) | `git log -1 --oneline` |
| `RuntimeError: clip generation stopped making progress at clip ...` | the new safety stop: the feeding loop had nothing left to wait for, which should not happen any more. To keep going, start the same command again with `--gen-batch 48` | the whole error line (it lists what was waiting) |

## Not in this round (noted from your message)

- The model made with one microphone (`models-release-mac`, input 2 x 64 x 80) only loads when the engine runs
  the one-microphone profile; with the default two-microphone profile it fails with (5,64,80) against (2,64,80).
  The model's own `gesture.json` says which it needs (`config.n_mics`, `input_shape`); making the engine read
  that is with the engine's owner, not part of the training handoff.

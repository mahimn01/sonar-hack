# Handoff for Lance: data-engine run on the RTX 4090

Written 2026-11-14 by the data-engine session. One command. It finds out why the model recognises only
30 % of Mahimn's gestures (he was the held-out person in your 03:02 run), measures three possible fixes,
and puts everything worth sending back into one folder that you push to the branch `dataeng-results`.

Nothing here has run on a CUDA GPU or on the real recordings before: your output is the first real check.
The script writes down a failed step and carries on, so let it finish and push whatever it produced.
Background: `DATA_ENGINE.md`.

## 0. Get the code (1 minute)

Run in PowerShell, in the repo folder on the 4090 box (where the MacBook recordings are in `mac\`):

```powershell
git fetch origin
git switch main
git pull
.venv\Scripts\python -c "import sonar.gesture_replay, sonar.labelqa, sonar.adapt, sonar.weaklabel; print('data engine ok')"
```
Expected: `data engine ok`. (If Mahimn names another branch, switch to that one instead of `main`.)
No new packages are needed on this box.

## 1. The run

```powershell
.venv\Scripts\python scripts\dataeng_4090.py --data mac
```

Nothing else may use the GPU meanwhile (close games, Medal, Discord overlays, as for the benchmark).

| what it does | runs on | expected time |
|---|---|---|
| picks the model of your newest run that left somebody out (`train_out\runs\20261114-030255\sweep\w24_s0`) and that person (mahimn) | | seconds |
| replays Mahimn's recordings through the real engine, twice (the model's own strength gate, then a low gate) | CPU, parallel | under 1 min |
| checks labels and burst cutting of all 13 sessions; draws 3 pictures of one of Mahimn's gesture sessions | verdicts on the GPU | 1-2 min |
| measures what an "each gesture 3 times" table calibration would give Mahimn | GPU | about 1 min |
| checks the fast batched code against the original code (`PARITY OK`) | GPU + CPU | under 1 min |
| trains once more with the extra weighted examples (`--weak`). The code changed since your last run, so this run also repeats the GPU-against-CPU picture check and remakes the synthetic data | GPU | about 16-20 min |
| ONLY if the label check found long, loud bursts: trains once more with the other burst cutting, in its own folder | GPU | about 12-16 min |

**Total: about 25 minutes, or about 40 if the last row runs.** `--skip-heavy` does only the first five rows
(about 5 minutes) if the box is needed for something else; the two training runs can follow later with the
same command.

It never writes to `models\` or `models-release*\`, and it does not change the caches of your earlier runs
(the second training run uses `train_out\rel12\`).

### What you should see at the end

```
==============================================================================
data-engine run 2026-11-14 HH:MM on <computer> (Windows 11)
python 3.12.10 | torch 2.15.0+cu132 | cuda True | NVIDIA GeForce RTX 4090
code: <commit>
model: C:\Users\lance\sonar-hack\train_out\runs\20261114-030255\sweep\w24_s0\gesture.onnx   held-out: mahimn
[01_replay]
replayed through the engine (confidence >= 0.6, strength gate NN.N dB):
  mahimn         recognised  NN/93  ( NN.N%)   wrong  N   missed  NN   false triggers N (N.NN/min)   [push .., swipe_left .., swipe_right ..]
why the missed ones were missed:
    NN  blocked: too faint for the engine's gate
    NN  model said none
    ...
[02_replay_gate8]
  ... the same with strength gate 8.0 dB ...
[03_labelqa]
=== 13 session(s): ... labelled gestures ...
  physics vs prompt labels: same  NN%, ...
  model   vs prompt labels: same  NN%, ...
  burst cutting (... gesture bursts): typical length N.NN s, NN% cut off at the length limit, ...
gesture bursts cut off at the length limit: NN %, typical strength NN dB -> long loud bursts: True/False
[05_adapt_eval]
                                            base                        bn                     proto                  bn+proto
mahimn                       0.NNN  0.NNN  0.NNN       0.NNN  0.NNN  0.NNN  ...
[06_parity]
parity on 6 session(s), ... step 1 by: gpu front end; judge: physics rules batched on cuda; model: PyTorch on cuda
PARITY OK
[07_train_weak]
Honest score on mahimn: accuracy 0.NNN, macro-F1 0.NNN; at the engine's 0.6 threshold 0.NNN of gestures recognised, 0.NNN of non-gestures would trigger
[08_train_relcut, SONAR_SEG_REL_OFF_DB=12]          (only if long loud bursts were found)
Honest score on mahimn: ...
total: NN.N minutes
==============================================================================
Everything to send back is in handoff_results\dataeng (about 20 files). Push it:
```

If `[06_parity]` says `PARITY FAILED`, the script does not start the training runs. Push the folder anyway.

## 2. Send the results back (1 minute)

```powershell
git switch -c dataeng-results
git add handoff_results\dataeng
git commit -m "Data-engine results from the 4090: replay, label check, calibration, burst cutting"
git push -u origin dataeng-results
git switch main
```

The folder holds only text, JSON and 3 PNG pictures (about 1-2 MB): the printed output of every step,
`SUMMARY.txt`, the replay and label-check reports, the calibration table, and the `TRAINING_RUN.json` of
each training run. No audio and no model files. Commit as yourself, no AI attribution lines.

Optional and useful: one real recording for Mahimn to look at on the Mac Studio (34 MB, by Drive or AirDrop,
not through git):
```powershell
Compress-Archive -Path mac\20261114-025724_mahimn_gestures -DestinationPath mahimn_gestures.zip -Force
```

## What the numbers will decide

| in `SUMMARY.txt` | then |
|---|---|
| `[01_replay]`: most misses "blocked: too faint", and `[02_replay_gate8]` recognises many more | the engine's strength gate is the problem: run the demo engine with `--min-peak-db 8`. No retraining needed |
| `[01_replay]`: most misses "model said none" or "confidence too low" | the model itself: the next three rows matter |
| `long loud bursts: True`, and `[08_train_relcut]` recognises clearly more than 0.296 | the burst cutting was part of it: Mahimn switches the new cutting on for good and you retrain once |
| `[05_adapt_eval]`: `bn` recall clearly above `base` | "each gesture 3 times" + `sonar.adapt fit` goes into the demo set-up |
| `[03_labelqa]`: `WRONG BURST CHOSEN?` or low agreement with the prompt labels | the labels are the problem: record the next sessions with the camera (below) |
| `[07_train_weak]`: fewer non-gestures trigger than 0.055 at the same recall | `--weak` stays on for the final training |

For comparison, your 03:02 run: accuracy 0.726, 0.296 of gestures recognised, 0.055 of non-gestures trigger.

## If something goes wrong

| what you see | do |
|---|---|
| `STOPPED: no model that left people out` | give them by hand: `--model train_out\runs\<run>\sweep\w24_s0\gesture.onnx --people mahimn` |
| a step ends with `!!! step ... ended with exit code` | let it finish; the error text is in `handoff_results\dataeng\<step>.txt`, push it |
| `CUDA out of memory` in a training step | another program holds the GPU; close it and run the command again (finished steps are quick the second time) |
| it must stop early | Ctrl+C; run again later, `--skip-heavy` first if time is short |

## The same steps by hand (only if you need one of them alone)

```powershell
$M = "train_out\runs\20261114-030255\sweep\w24_s0\gesture.onnx"
.venv\Scripts\python -m sonar.gesture_replay mac --model $M --people mahimn
.venv\Scripts\python -m sonar.gesture_replay mac --model $M --people mahimn --min-peak-db 8
.venv\Scripts\python -m sonar.labelqa mac --model $M --no-plot --out train_out\dataeng\qa
.venv\Scripts\python -m sonar.labelqa mac\20261114-025724_mahimn_gestures --model $M --out train_out\dataeng\qa
.venv\Scripts\python -m sonar.adapt eval --model $M --data mac --people mahimn
.venv\Scripts\python -m sonar.weaklabel mac --model $M --parity
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --data mac --val-person mahimn --weak --models train_out\weak_models --release train_out\weak_release
$env:SONAR_SEG_REL_OFF_DB = "12"
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --data mac --val-person mahimn --skip-parity --out train_out\rel12 --models train_out\rel12\models --release train_out\rel12\release
Remove-Item Env:SONAR_SEG_REL_OFF_DB
```
`--skip-parity` belongs to the last training command only: the picture check compares with a stored
reference made with the usual burst cutting, so with the variable set it would report a difference that is
intended. A model trained that way must not be copied into `models\`: it only works where the same variable
is set.

## On the MacBook: recording with the camera timing the gestures (you and Mahimn only, optional)

Never tried on a real webcam yet. It records like `scripts/mac_record_all.sh`, but the webcam works out when
each gesture really happened, so the labels are exact instead of "somewhere in 2.5 seconds".

```bash
cd ~/sonar-hack && uv pip install --python .venv/bin/python -r requirements-camera.txt
cd ~/sonar-hack && .venv/bin/python -m sonar.camrecord --person lance --in-device MacBook --out-device MacBook
```
- macOS asks to let the Terminal use the camera: allow it, then run the command again.
- On the first screen raise your **right** hand. It must show on the **right**. If not, press **M**. Then **Y**.
- Gesture at the height of the top half of the screen and push at the camera. The counter goes up when the
  camera sees a gesture. SPACE skips a block, Q ends early.
- Expected at the end: `labels: N push, N swipe_left, N swipe_right (... timed by the camera ...)`,
  `clock sync: camera runs +NN ms against the audio (...); corrected`, `sonar check: N/N camera gestures have
  a sonar movement burst`.
- Send back those three lines, and whether the counter followed your gestures. The session lands in `data/`
  like every other and trains with no extra step.

Calibration at the demo table (once the run above says it helps). It needs `gesture.pt` next to
`gesture.onnx`; the model branch does not carry it yet, so add `models-release-mac\gesture.pt` to the next
model commit on the 4090 box.
```bash
cd ~/sonar-hack && .venv/bin/python -m sonar.camrecord --person lance --script calibrate --in-device MacBook --out-device MacBook --consent yes
cd ~/sonar-hack && .venv/bin/python -m sonar.adapt fit --model models/gesture.onnx --calib data/<the calibrate session> --out models/room
cd ~/sonar-hack && .venv/bin/python -m sonar.engine --model models/room/gesture.onnx --in-device MacBook --out-device MacBook
```

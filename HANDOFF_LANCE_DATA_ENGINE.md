# Handoff for Lance: data-engine run on the RTX 4090

**Version 2, written 2026-11-14 about 05:45 EST by the data-engine session. It replaces version 1
(commit 48e2518).**

**Short form: do not run the data-engine step. The 4090 is free for the recall session's Step B
(`HANDOFF_LANCE_RECALL.md`).**

## 1. What changed since version 1

Version 1 asked for `.venv\Scripts\python scripts\dataeng_4090.py --data mac` (about 40 minutes). Please
skip it:

- **It is already measured.** The 13 MacBook recordings and the held-out model of your 03:02 run came back
  on the branches `model-sat-am` and `recall-data-1`, so the same script ran on the Mac Studio (1.5 minutes,
  CPU). The results are in `handoff_results/dataeng/` and summed up in `DATA_ENGINE.md`, section 1.
- **Its two training runs are covered better elsewhere.** The recall session's suite trains the other burst
  cutting as six recipes and scores every recipe with each person held out in turn, three seeds each.
- **Version 1 would have scored on wrong labels.** `--data mac` also reads
  `mac\excluded\20261114-001248_mahimn_gestures` as it was recorded. In that session Mahimn's swipes go the
  opposite way to the prompts (27 of 30). So 30 of his 81 test gestures carried the wrong label, and every
  score for him in that run would have been too low. The new script exchanges those labels first.

## 2. If version 1 already ran, or is running

| situation | do |
|---|---|
| not started | skip it |
| running | Ctrl+C is fine |
| finished | push `handoff_results\dataeng` anyway (commands below). Still useful: `06_parity.txt`, the first check of the fast code on a CUDA GPU, and the timing lines. Not usable: every score for Mahimn (`01_replay`, `02_replay_gate8`, `05_adapt_eval`, "Honest score on mahimn"), because of the labels above |

It wrote only under `train_out\` and `handoff_results\dataeng`, never to `models\` or `models-release*\`.
If its training run finished, `train_out\runs\` holds one more run folder (the newest). It was scored on the
labels as recorded: take no number from it.

```powershell
git switch -c dataeng-results
git add handoff_results\dataeng
git commit -m "Data-engine results from the 4090 (version 1 of the handoff: scores for mahimn are on unfixed labels)"
git push -u origin dataeng-results
git switch recall
```

## 3. Optional, only when the GPU has nothing else to do (about 5 minutes)

Not needed for any decision. It would confirm two things on the 4090: the fast batched code gives what the
original code gives on CUDA (`PARITY OK`), and the box reproduces the Mac Studio's numbers.

Needs the data-engine branch with Mahimn's next push (the script is newer than 48e2518).

```powershell
git fetch origin
git switch data-engine
git pull
git restore --source origin/model-sat-am --worktree -- mac/data mac/excluded
dir mac\data, mac\excluded        # expect 12 session folders + 1 (20261114-001248_mahimn_gestures)
.venv\Scripts\python scripts\dataeng_4090.py --data mac
```

It trains nothing. It first writes a corrected copy of the label files (the audio is not copied, no
recording is changed), then reads only that copy. If it stops with `STOPPED: no model that left people out`,
add `--model train_out\runs\20261114-030255\sweep\w24_s0\gesture.onnx --people mahimn`.

Expected at the end (these are the Mac Studio's numbers for the same recordings and the same model; the
4090 should print the same counts):

```
[00_view]
  20261114-001248_mahimn_gestures          mahimn     LEFT/RIGHT EXCHANGED in 30 swipe labels
view of 13 session(s) -> ...
[01_replay]
replayed through the engine (confidence >= 0.6, strength gate 30.6 dB):
  mahimn         recognised  36/81  ( 44.4%)   wrong  5   missed  40   false triggers 1 (0.15/min)   [push 7/27, swipe_left 18/27, swipe_right 11/27]
why the missed ones were missed:
     2  blocked: too faint for the engine's gate
    11  blocked: taken for somebody walking in or past
    16  right gesture, confidence too low
    10  another gesture, below the threshold
     1  model said none
[02_replay_conf05]                  recognised  47/81   wrong  8   false triggers 1
[02_replay_conf04]                  recognised  50/81   wrong 11   false triggers 1
[02_replay_nogates]                 recognised  43/81   wrong  8   false triggers 5
[02_replay_nogates_conf05]          recognised  58/81   wrong  9   false triggers 7
[02_replay_relcut]                  recognised  44/81   wrong  6   false triggers 0
[02_replay_relcut_nogates_conf05]   recognised  61/81   wrong  8   false triggers 7
[03_labelqa]
  20261113-233122_lance_gestures: TOO FAST: 28 of 35 gestures run off the edge of the picture (it shows hand speeds up to about 2.3 m/s)
  ... (5 LONG BURSTS lines, 3 TOO FAST lines)
=== 13 session(s): 218 labelled gestures (0 camera-timed, 218 prompt windows), 1 with no sonar movement, 0 for review, 11 to check
  model   vs prompt labels: same   76%, saw no gesture   19%, other gesture    5%   (n=218)
  burst cutting (218 gesture bursts): typical length 1.10 s, 51% cut off at the length limit, 2.1 bursts per gesture, ...
  too fast for the picture (the stroke runs off the edge of the speed range): 91 of 218 gestures
[05_adapt_eval]
mahimn                       0.837  0.556  0.050       0.855  0.514  0.010       0.831  0.542  0.060       0.855  0.556  0.010
[06_parity]
parity on 6 session(s), 375 bursts; step 1 by: cpu; judge: physics rules batched on cuda; model: PyTorch on cuda
PARITY OK
training runs: none asked for (--train-weak / --train-relcut)
total: about 2 minutes
```

Send back (branch `dataeng-results`, text, JSON and 3 pictures, about 1 MB, no audio and no model files):

```powershell
git switch -c dataeng-results
git add handoff_results\dataeng
git commit -m "Data-engine diagnostics from the 4090: replay, label check, calibration, parity on CUDA"
git push -u origin dataeng-results
git switch recall
```

Commit as yourself, no AI attribution lines. If `[06_parity]` says `PARITY FAILED`, push the folder all the
same: that file is the one thing this step is for.

## 4. On the MacBook: one check after every new recording (10 seconds, no GPU)

This is the useful part for today. One session was recorded with nearly every swipe going the wrong way,
and it was only found hours later, from a training score that made no sense. This line notices while the
person is still at the table:

```bash
cd ~/sonar-hack && .venv/bin/python -m sonar.labelqa data/<the new session folder> --no-plot
```

Needs the data-engine code on the MacBook with Mahimn's next push (`git fetch origin && git switch
data-engine && git pull`, or `main` once it is merged). No new packages, no model file.

| it prints | meaning | do |
|---|---|---|
| `!! LEFT/RIGHT LOOKS FLIPPED for the prompt labels (N of M swipes)` | the swipes went the opposite way to the prompts | record that person again: "swipe_left = the hand travels to YOUR left" |
| `!! TOO FAST: N of M gestures run off the edge of the picture` | the hand was faster than about 2.3 m/s, the most the sonar picture can show | record again at a calm pace: about half a second across the screen, not a flick |
| `N with no sonar movement` for more than 2 or 3 prompts | gestures too far from the laptop, or the volume changed | check the volume, gesture above the keyboard |
| none of these | the session is fine | carry on |

Measured on the 5 real gesture sessions with the labels as recorded:

| session | `FLIPPED` (swipes against the direction cue) | `TOO FAST` (gestures off the edge) |
|---|---|---|
| lance | no (5 of 19) | **yes** (28 of 35) |
| lance2 | no (7 of 17) | **yes** (37 of 45) |
| mahimn, first (the reversed one) | **yes** (27 of 30) | **yes** (25 of 45) |
| aarsh | no (0 of 29) | no (1 of 45) |
| mahimn, second | no (2 of 23) | no (0 of 48) |

So your own next session is the one to watch: both of yours were too fast. The limit for "too fast" was
set on these 5 sessions, so read it as a first guess. `--model <a gesture.onnx>` adds a line "model agrees
with the prompt label N/M".

## 5. Recording with the camera timing the gestures (you and Mahimn only, optional)

Never tried on a real webcam yet. It records like `sonar.record`, but the webcam works out when each
gesture really happened and which way it went, so a reversed session cannot happen.

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
- If the camera part fails, the same command still records the plain prompted session.

Try it yourself once before using it with another person. For the new people today, `sonar.record` plus the
check in section 4 is the safe choice.

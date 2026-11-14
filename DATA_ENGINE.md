# DATA_ENGINE — better labels, checked labels, and a model that fits the room

Written Saturday 2026-11-14. Everything here is run from the repo root. `py` means the project's Python
(`.venv\Scripts\python` on Windows, `.venv/bin/python` on macOS).

Words used: a **burst** is a stretch of audio in which the sonar saw movement; the engine cuts one and
shows the model its picture. A **prompt window** is the 2.5 s after "GO" in `sonar.record`. **Recall** =
share of real gestures that are recognised. **Held-out person** = somebody the model never trained on.

## 1. Why this exists, and where we stand

The first real training run (MacBook recordings of 4 people, 2026-11-14 03:18) scored on the held-out
person: accuracy 0.73, but only **30 % of gestures recognised** at the engine's 0.6 threshold (5.5 % of
non-gestures trigger). That 30 % is the number to raise. Three things can be wrong, and each has a tool here:

| question | tool | what it tells you |
|---|---|---|
| Are the labels right, and are gestures cut into clean pictures? | `sonar.labelqa` | agreement of the labels with the physics rules and the model, a list of moments to look at, and how the bursts were cut |
| What does the real engine do with those recordings, and why does it miss? | `sonar.gesture_replay` | recognised / wrong / missed per person, and the reason for every miss |
| Does a 30-second calibration at the table fix a new person or room? | `sonar.adapt` | the model retuned to the room; `eval` measures the gain on held-out people |
| Can recording be more exact? | `sonar.camrecord` | the webcam times every stroke, instead of "somewhere in 2.5 s" |
| Can existing recordings give more training examples? | `sonar.weaklabel`, `train_4090.py --weak` | hard negatives, confirmed camera readings, pseudo-labels, each with a weight |

**Not yet done on real audio.** No real recording is on the Mac Studio, so every number below marked
"simulation" comes from the simulator. One command on the 4090 box gives the real numbers and, if they
call for it, trains the two candidate fixes (`scripts/dataeng_4090.py`, see `HANDOFF_LANCE_DATA_ENGINE.md`).

### One finding that needs a real-data check first

The released MacBook model says its gestures are very loud: 10 % of them are below **36.6 dB** of movement
level, where the simulator's hand gives 13-25 dB. At that loudness the movement never falls back to the
"quiet" level between the wind-up, the stroke and the hand going back, so the engine sees one long
movement and chops it at its 1.4 s limit. In simulation at that loudness **about 60 % of all bursts and 95 %
of the gesture bursts are chopped at the limit** (37 % of all bursts at normal loudness), so the stroke
lands at a different place in every picture or is cut in two. Whether this happens in the real recordings is unknown until `sonar.labelqa` has run on
them (section 4 prints it as "cut off at the length limit"). A switchable fix is in section 8.

A second consequence of the loudness: the engine ignores bursts more than 6 dB weaker than the weakest
10 % of the training gestures. For this model that gate sits at **30.6 dB**. A new person who gestures a
little further away falls under it and nothing is reported, whatever the model thinks.
`sonar.gesture_replay` counts exactly these ("blocked: too faint for the engine's gate").

## 2. Record with the camera timing the gestures (you and Lance)

`sonar.camrecord` records exactly like `sonar.record` (same `audio.wav` + `session.json`, same
`--profile`, `--blocksize`, `--in-device`, `--out-device`) while the webcam tracks the hand. It writes when
each stroke really happened, which way it went, how fast and how far. No picture is saved, only hand joint
positions.

Install once per recording laptop (adds MediaPipe and OpenCV; the first run downloads a 7.8 MB model):

```powershell
.venv\Scripts\python -m pip install -r requirements-camera.txt
```
```bash
cd ~/sonar-hack && uv pip install --python .venv/bin/python -r requirements-camera.txt
```

Run:

```powershell
# Windows HP
.venv\Scripts\python -m sonar.camrecord --person mahimn --profile clean --blocksize 3840 --consent yes
.venv\Scripts\python -m sonar.camrecord --person mahimn --profile clean --blocksize 3840 --consent yes --script negatives
```
```bash
# MacBook
cd ~/sonar-hack && .venv/bin/python -m sonar.camrecord --person lance --in-device MacBook --out-device MacBook --consent yes
cd ~/sonar-hack && .venv/bin/python -m sonar.camrecord --person lance --in-device MacBook --out-device MacBook --consent yes --script negatives
# no hardware (simulator + a fake camera), on any machine
cd ~/Documents/Dev/sonar-hack-data && .venv/bin/python -m sonar.camrecord --backend sim --fast --no-window
```

| `--script` | what you do | time |
|---|---|---|
| `free` (default) | 2 pushes at the camera, then swipe left / swipe right / push in small blocks at your own pace; a counter goes up as the camera sees each one | about 60 s |
| `negatives` | type, reach for a cup, fidget, lean, walk off and back. No gesture at all (the session also serves as a real background recording) | about 70 s |
| `gestures` | timed "get ready ... GO" trials as in `sonar.record`, with the camera refining the timing | 4.7 s per trial |
| `calibrate` | each gesture 3 times, for the table calibration of section 7 | about 40 s |

Leave `--consent yes` out the first time on a laptop: the start screen shows the mirrored picture.
**Raise your right hand; it must appear on the right. If not, press M.** A wrong mirror setting swaps every
swipe label. Keys later: SPACE skips a block, Q ends early and keeps what was recorded.

Three things to know:
- **The webcam looks straight out from the top of the screen.** A hand 25 cm in front of the middle of the
  screen is at the bottom edge of its view. Gesture at the height of the top half of the screen and push
  at the camera. The status line shows "NO HAND IN VIEW" when it has lost you.
- **Read the three lines at the end.** "labels" (how many of each), "clock sync" (a number of ms and
  "corrected"), "sonar check" (how many camera gestures the sonar also saw move).
- **No camera, no MediaPipe, or a camera that never sees the hand:** it says why and uses timed prompts.

Files in `data/<date-time>_<person>_<script>/`:

| file | content |
|---|---|
| `audio.wav` | as `sonar.record`. With `--ultrasound-only` everything below about 12 kHz (all speech) is removed; the sonar features do not change (tested) |
| `session.json` | as `sonar.record`, plus `camera`, `sync` (clock offset and its spread), `labels` (counts), `prompts` (what was asked when) |
| `landmarks.jsonl` | one line per camera picture: time, audio sample, 21 hand points per hand. No picture |

`py -m sonar.camrecord --relabel data/<session>` labels a session again from its saved hand points.

Status: works end to end in simulation (clock offsets of 0-250 ms recovered within 3-7 ms; all scripted
gestures named). **Never run on a real webcam or laptop.** MediaPipe itself was run on a sample photo.

## 3. How these recordings reach the trainer

Nothing to do: `sonar.dataset`, `sonar.train` and `scripts/train_4090.py` read camera sessions like any
other. Two additive keys in a `session.json` event change how it is cut (sessions without them are cut
exactly as before):

| key | written for | effect in `sonar.dataset` |
|---|---|---|
| `"tight": true` | a camera-timed gesture: `start`/`end` bound the stroke itself | the burst that **covers the stroke** becomes the example, not the loudest burst nearby (which is often the hand going back) |
| `"exclude": true` | an unsure camera reading (`label: "review"`), a free-form block | never used as a "none" example. In a block you may have gestured where the camera could not see |

`negatives` sessions are also used as real backgrounds for synthetic gestures (like `scene` and `breathing`).

## 4. Check the labels: `sonar.labelqa`

```powershell
.venv\Scripts\python -m sonar.labelqa mac --model models\gesture.onnx --no-plot --out train_out\dataeng\qa
.venv\Scripts\python -m sonar.labelqa mac\20261114-025724_mahimn_gestures --model models\gesture.onnx --out train_out\dataeng\qa
```
```bash
cd ~/sonar-hack && .venv/bin/python -m sonar.labelqa data --model models/gesture.onnx --no-plot
cd ~/sonar-hack && .venv/bin/python -m sonar.labelqa data/<session> --model models/gesture.onnx
```

The first form judges every burst of every session in one batched pass and prints numbers. The second
also draws `qa_p1.png`, `qa_p2.png`, ... (30 s each): the Doppler picture, what was asked, the label, what
the trainer uses each burst as, the verdict of the physics rules and of the model, red boxes around
moments worth a look, and (camera sessions) the camera's hand-speed curve over the sonar's movement curve.

How to read the totals:

| line | good | worrying |
|---|---|---|
| `physics vs prompt labels: same` | 85 %+ | under 60 %: wrong labels, or (one microphone) the direction cue is too weak to judge swipes |
| `swipe direction ... opposite` | a few | most of them: `LEFT/RIGHT LOOKS FLIPPED` (camera mirror, or swapped channels) |
| `burst cutting: ... cut off at the length limit` | under 20 % | 30 %+ with strength above 25 dB: `LONG BURSTS`, see section 8 |
| `bursts per gesture` | 1-2 | 3+: the loudest one may not be the stroke |
| `another burst fits the prompt better` | 0 | a fifth of the gestures or more: `WRONG BURST CHOSEN?` |
| `recognised in SOME burst of its prompt` | close to all | far below: the model fails on this person whichever burst is taken |
| `fires on known negatives` | under 10 % | above: these are the hard negatives `--weak` adds |

## 5. Replay through the real engine: `sonar.gesture_replay`

The training score judges pictures. The demo is the engine: its own burst cutting, its gates (too faint,
somebody walking in, movement too far away), the 0.6 threshold and the debounce. This plays recordings
through that engine and counts per prompted gesture.

```powershell
.venv\Scripts\python -m sonar.gesture_replay mac --model train_out\runs\20261114-030255\sweep\w24_s0\gesture.onnx --people mahimn
.venv\Scripts\python -m sonar.gesture_replay mac --model train_out\runs\20261114-030255\sweep\w24_s0\gesture.onnx --people mahimn --min-peak-db 8
```

Use a model that did **not** train on those people (the `sweep` model of a run and its held-out people);
`models/gesture.onnx` is retrained on everyone and flatters itself. The output ends with why the misses
were missed, and the most common reason comes with the knob that addresses it:

| reason | what to try |
|---|---|
| no movement burst | gesture closer or brisker, or more volume |
| blocked: too faint for the engine's gate | a lower gate: `--min-peak-db 8` here and on `sonar.engine` |
| blocked: taken for somebody walking in or past | `sonar.engine --no-range-gate` |
| right gesture, confidence too low | more people in training, the table calibration (section 7), or a lower `--min-confidence` |
| another gesture, below the threshold | left/right first: one microphone leaves only the two-speaker cue |
| model said none | the burst cutting (sections 4 and 8), more people |
| held back by the debounce rules | gestures too close together, or a wind-up read as the opposite swipe |

In simulation, a model taken to "another room" lost 119 of 127 gestures to the strength gate alone.

## 6. More training examples from what is already recorded: `--weak`

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --data mac --val-person mahimn --weak
```

`scripts/train_4090.py --weak` adds a confidence-weighted set made by `sonar/weaklabel.py`. It never
replaces or repeats the usual examples, and the default run is unchanged.

| kind | what | weight |
|---|---|---|
| `hardneg` | the current model fires in a part recorded as gesture-free (typing, reaching, walking past) | 2.0 |
| `aux` | a hand movement the camera saw that was clearly not a gesture (slow, short, diagonal, the hand dropping back) | 0.6 |
| `rescued` | a camera "review" reading that the model or the physics rules confirm | 0.5-0.7 |
| `pseudo` | a burst nobody labelled where the model (0.90+) and the physics rules name the same gesture | 0.2-0.6 |
| `retimed` | a prompt window whose loudest burst is not read as the prompted gesture while another burst of the same window is (the stroke, where the loudest was the hand going back) | 0.6 |

- The labelling model is `models/gesture.onnx` from the last run, or this run's pre-trained model the first time.
- **Recordings with no labels that may contain gestures** go in their own folder and are passed with
  `--weak-data that_folder`. Never put them under `--data`: the plain builder uses all of such a
  recording as "none".
- **The score stays comparable.** Weak examples are never part of the validation set, and those of
  held-out people are not used at all (`sonar/train.py`).
- `py -m sonar.weaklabel mac --model models\gesture.onnx --out weak.npz --hard-negatives hard.npz --report weak.json`
  runs it by hand; the report lists every burst with every opinion.

Expect it to lower false triggers more than to raise recall. Verified on simulated sessions only.

## 7. Table calibration: `sonar.adapt`

"Do each gesture 3 times" at the demo table, then retune the model to that room. It re-measures the
normalisation statistics inside the network (BatchNorm) on the room's own pictures and blends them with
the trained ones, and lowers the engine's strength gate if gestures here are fainter. No weights change.

```bash
cd ~/sonar-hack && .venv/bin/python -m sonar.camrecord --person lance --script calibrate --in-device MacBook --out-device MacBook --consent yes
cd ~/sonar-hack && .venv/bin/python -m sonar.adapt fit --model models/gesture.onnx --calib data/<that session> --out models/room
cd ~/sonar-hack && .venv/bin/python -m sonar.engine --model models/room/gesture.onnx --in-device MacBook --out-device MacBook
```

(`sonar.record --script gestures --reps 3` makes the calibration session too.) The result is a normal
model folder; leave `--model` out to go back to the original. `--prototypes` adds a bonus for bursts that
look like the 3 examples; off by default because it also raised false triggers. Needs PyTorch and the
model's `gesture.pt` next to `gesture.onnx` (the model branch does not carry the `.pt` yet: add
`models-release-mac/gesture.pt` to the next model commit).

Measure it on people the model never saw (the first 3 of each of their gestures calibrate, the rest is scored):

```powershell
.venv\Scripts\python -m sonar.adapt eval --model train_out\runs\20261114-030255\sweep\w24_s0\gesture.onnx --data mac --people mahimn
```

Simulation, 9 simulated people in three "rooms" (same room, weaker echo, another laptop with the hand
further away), a small model trained in the first room:

| | accuracy | gestures recognised | non-gestures triggering |
|---|---|---|---|
| as trained | 0.70 | 0.52 | 0.04 |
| + BatchNorm statistics (default) | 0.87 | 0.89 | 0.09 |
| + prototypes only | 0.72 | 0.59 | 0.08 |
| + both | 0.88 | 0.92 | 0.10 |

Better for 6 of 9, worse for none; in the unchanged room it did no harm. The test sets are small (about
20 pictures per person) and this is a simulation: the real gain is whatever `eval` prints on the 4090 box.

## 8. Burst cutting when the echoes are very loud

`seg_rel_off_db` (in `sonar/config.py`, default 0 = off, nothing changes) lets a burst also end once the
movement is that many dB below the burst's own peak. It is switched on for every tool at once by an
environment variable, so existing recordings can be cut again without re-recording:

```powershell
$env:SONAR_SEG_REL_OFF_DB = "12"
.venv\Scripts\python -m sonar.labelqa mac --no-plot --out train_out\dataeng\qa_rel12
Remove-Item Env:SONAR_SEG_REL_OFF_DB
```

Simulation at MacBook loudness: gesture bursts chopped at the limit fell from 95 % to 14 % in one set of
sessions and from 100 % to 50 % in another; typical burst length from 1.4 s to 0.7 s; the physics rules
agreed with the prompt in 100 % instead of 94 %. A setting of 18-24 changes nothing at ordinary loudness.

**It did not change recognition in simulation** (a held-out simulated person was recognised 100 % with
both cuttings: simulated people are too alike to show a difference). So this is a candidate, not a proven
fix. Only try it if section 4 reports `LONG BURSTS` on the real recordings, and then as a full A/B
training run in its own working folder. `scripts/dataeng_4090.py` does exactly that by itself (it trains
the second model under `train_out/rel12/` when the label check finds long loud bursts). A model trained
with it needs the same variable set wherever the engine runs, until the default is changed.

## 9. Speed on the training box

| step | where it runs |
|---|---|
| audio to bursts, per session | the GPU front end (`sonar/features_torch.py`) on CUDA when the checkout has it, with worker processes finishing the bursts; otherwise one session per CPU core. Cached per session |
| verdict of the physics rules and of the model | every burst of every session moved to the GPU once and judged in one batched pass (`sonar/batch_judge.py`); no per-burst Python |
| `train_4090.py --weak` | the burst step runs in the background while the synthetic data is made and the model pre-trains |
| `sonar.adapt` | pictures moved to the device once; all people and variants scored in batches |
| `sonar.gesture_replay` | CPU by nature (it is the realtime engine); sessions in parallel worker processes |

`py -m sonar.weaklabel mac --model models\gesture.onnx --parity` compares the fast path with the
original one-burst-at-a-time code on your sessions and prints `PARITY OK` or `PARITY FAILED`. Measured
here on a CPU: bursts identical, pictures identical, physics rules within 7e-8, model within 4e-7.
`torch.compile` is offered (`--compile`) but off: the model pass over a few thousand bursts takes a
fraction of a second, less than compiling would. **Nothing here was timed on a CUDA GPU.**

## 10. What is verified and what is not

Verified in simulation (51 new tests; the 89 older ones still pass): the camera labeller and clock sync;
the recorder end to end with a fake camera, and its fallbacks; the two dataset rules; label QA numbers,
flags and pictures; weak labels and their use in training; the batched verdicts against the originals;
BatchNorm adaptation, export and use in the engine; the engine replay and its miss reasons; the cutting
audit and the cutting switch. The working tree merges with today's main (which holds the GPU pipeline)
without conflicts, and the fast tests pass on the merged tree (checked in a throwaway clone);
`train_4090.py --weak` and `scripts/dataeng_4090.py` ran end to end on a CPU with tiny training runs.

Not verified: anything on real recordings, a real webcam, or a CUDA GPU.

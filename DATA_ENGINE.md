# DATA_ENGINE — better labels, checked labels, and a model that fits the room

Written Saturday 2026-11-14, updated about 05:45 with the numbers from the real recordings. Everything here is run from the repo root. `py` means the project's Python
(`.venv\Scripts\python` on Windows, `.venv/bin/python` on macOS).

Words used: a **burst** is a stretch of audio in which the sonar saw movement; the engine cuts one and
shows the model its picture. A **prompt window** is the 2.5 s after "GO" in `sonar.record`. **Recall** =
share of real gestures that are recognised. **Held-out person** = somebody the model never trained on.

## 1. Where we stand (real recordings, measured 2026-11-14 about 05:30)

Recordings: 13 sessions from Lance's MacBook, 3 humans (lance, who recorded twice as `lance` and `lance2`;
aarsh; mahimn), 218 prompted gestures, about 40 minutes of audio. Test: Mahimn's 81 gestures, with the
model of the 03:02 training run, which never saw him. Everything in this section was run on the Mac Studio
(1.5 minutes of CPU, `scripts/dataeng_4090.py`); the output is in `handoff_results/dataeng/`.

**The reported "30 % of gestures recognised" was mostly a labelling accident.** In Mahimn's first session
(`20261114-001248_mahimn_gestures`) 27 of 30 swipes go the opposite way to their prompts, so 30 of the 81
test gestures could never count. With those labels exchanged the same model recognises 53 % of the pictures
(the recall session measured the same), and their best model so far reaches 65 %. `sonar.labelqa` flags that
session by itself, and `sonar.dataview` exchanges the labels without touching the recording (section 4).

What the real engine does with Mahimn's 81 gestures (`sonar.gesture_replay`, labels fixed):

| engine setting | recognised | wrong gesture | false triggers in his 6.9 min without a gesture |
|---|---|---|---|
| as shipped (threshold 0.6, all gates on) | 36 of 81 (44 %) | 5 | 1 |
| threshold 0.5 | 47 (58 %) | 8 | 1 |
| gates off | 43 (53 %) | 8 | 5 |
| gates off, threshold 0.5 | 58 (72 %) | 9 | 7 (about one a minute) |
| as shipped, **other burst cutting** (section 8), same model | 44 (54 %) | 6 | 0 |
| gates off, threshold 0.5, other burst cutting | 61 (75 %) | 8 | 7 |

Why the 40 misses at the shipped setting: 16 times the model named the right gesture but was less than 0.6
sure, 10 times it leaned to another gesture, 11 were blocked by the "somebody is walking in" and range
gates, 2 were under the strength gate, 1 was read as nothing. These settings were tried on the test person
himself, so the gains are a little flattering; the recall session picks them on the other people.

What the real recordings say about the data side:

| question | answer | where |
|---|---|---|
| Are the labels right? | One session of 5 was reversed (27 of 30 swipes). Aarsh's and Mahimn's second are clean (0 of 29 and 2 of 23 swipes against the direction cue). Lance's two cannot be judged that way: his strokes are too fast for the cue | section 4 |
| Are gestures cut into clean pictures? | No: 51 % of the gesture bursts run into the 1.4 s limit and are chopped there. With the other cutting 5 % | section 8 |
| Does the other cutting help a new person? | It looks like it: 8 more of Mahimn's 81 gestures with the same model, before any retraining, and the recall session sees the same with a second model. The proof is a retrained model; their suite trains it | section 8 |
| Is the wrong burst labelled? | No. For the held-out person the loudest burst of a prompt is as good as the best possible choice (43 of 81 against 43 with the usual cutting, 53 against 55 with the other) | section 4 |
| Are people gesturing too fast? | Yes in 3 of 5 sessions (both of Lance's, Mahimn's first): 91 of the 218 gestures run off the edge of the picture's speed range | section 4, `TOO FAST` |
| Does the "each gesture 3 times" calibration help? | Not the recall (0.556 to 0.514). It cuts false triggers (0.050 to 0.010) | section 7 |
| Is the strength gate the problem? | No: 2 misses of 40. The confidence threshold and the two other gates are | section 5 |
| Do weak labels add anything now? | No: 3 hard negatives in all 13 sessions | section 6 |

The biggest lever is not in this table: **3 humans are too few.** Every new person recorded at a calm pace
and checked with the one line in section 4 is worth more than any setting here.

Two things the first version of this file said were wrong and are withdrawn: that the strength gate was a
likely cause of the low recall, and that the calibration's gain in simulation (0.52 to 0.89) would carry
over to real recordings.

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

**After every new recording (10 seconds, no model needed), while the person is still at the table:**

```bash
cd ~/sonar-hack && .venv/bin/python -m sonar.labelqa data/<the new session> --no-plot
```

| it prints | meaning | do |
|---|---|---|
| `!! LEFT/RIGHT LOOKS FLIPPED for the prompt labels (N of M swipes)` | the swipes went the opposite way to the prompts | record again ("swipe_left = the hand travels to YOUR left"), or exchange the labels with `sonar.dataview` (below) |
| `!! TOO FAST: N of M gestures run off the edge of the picture` | the hand was faster than about 2.3 m/s, the most the picture can show | record again at a calm pace: about half a second across the screen, not a flick |
| `N with no sonar movement` for more than 2 or 3 prompts | too far from the laptop, or the volume changed | check the volume, gesture above the keyboard |
| none of these | the session is fine | carry on |

On the 5 real gesture sessions with the labels as recorded: `FLIPPED` appears for the one reversed session
(27 of 30 swipes) and for no other (5 of 19, 7 of 17, 0 of 29, 2 of 23). `TOO FAST` appears for both of
Lance's sessions (28 of 35, 37 of 45) and Mahimn's first (25 of 45), not for Aarsh's (1 of 45) or Mahimn's
second (0 of 48). The limit for "too fast" was set on these 5 sessions, so treat it as a first guess. The
simulator cannot check it: its too-fast strokes leave the picture altogether instead of piling up at the edge.

The first form above judges every burst of every session in one batched pass and prints numbers. The second
also draws `qa_p1.png`, `qa_p2.png`, ... (30 s each): the Doppler picture, what was asked, the label, what
the trainer uses each burst as, the verdict of the physics rules and of the model, red boxes around
moments worth a look, and (camera sessions) the camera's hand-speed curve over the sonar's movement curve.

How to read the totals:

| line | good | worrying |
|---|---|---|
| `physics vs prompt labels: same` | 85 %+ in simulation; 55-75 % on the MacBook's one microphone at a calm pace | under 40 %: strokes too fast (`TOO FAST`), or wrong labels |
| `swipe direction ... opposite` | a few | most of them in one session: `LEFT/RIGHT LOOKS FLIPPED` (the swipes went the other way, or the camera picture is mirrored). In every session: swapped channels |
| `too fast for the picture` | a few | a third of a session's gestures or more: `TOO FAST` |
| `burst cutting: ... cut off at the length limit` | under 20 % | 30 %+ with strength above 25 dB: `LONG BURSTS`, see section 8 |
| `bursts per gesture` | 1-2 | 3+: the loudest one may not be the stroke |
| `another burst fits the prompt better` | 0 | a fifth of the gestures or more: `WRONG BURST CHOSEN?` |
| `recognised in SOME burst of its prompt` | close to all | far below: the model fails on this person whichever burst is taken |
| `fires on known negatives` | under 10 % | above: these are the hard negatives `--weak` adds |

### Fixing labels without touching a recording: `sonar.dataview`

```powershell
.venv\Scripts\python -m sonar.dataview mac --out train_out\mac_fixed
```

writes a folder with one small `session.json` per session that points at the original audio. Use that
folder as `--data` everywhere: trainer, label check, gesture replay, calibration. While writing it, the tool

- exchanges swipe_left and swipe_right in the sessions named with `--flip-swipes` (default: the one known
  reversed session, `20261114-001248_mahimn_gestures`; `--flip-swipes ""` for none);
- takes a session that is found twice only once;
- leaves out the sessions named with `--drop`.

Why it matters for `--data mac`: `mac\excluded\` lies inside `mac\`, so every tool given `mac` also reads the
reversed session, as recorded. Given the view, they read it with the labels exchanged.

The recordings are only read. The recall session's suite makes the same exchange with its own `--flip`.
Checked here with the dataset builder, the label check, the gesture replay and the calibration; for
`scripts/train_4090.py` on the other branches only by reading the code (it takes the audio path from
`session.json`). `sonar.replay`, the presence replay tool, reads `audio.wav` from the session folder itself
and does not work on a view.

## 5. Replay through the real engine: `sonar.gesture_replay`

The training score judges pictures. The demo is the engine: its own burst cutting, its gates (too faint,
somebody walking in, movement too far away), the 0.6 threshold and the debounce. This plays recordings
through that engine and counts per prompted gesture.

```powershell
.venv\Scripts\python -m sonar.gesture_replay train_out\mac_fixed --model train_out\runs\20261114-030255\sweep\w24_s0\gesture.onnx --people mahimn
.venv\Scripts\python -m sonar.gesture_replay train_out\mac_fixed --model train_out\runs\20261114-030255\sweep\w24_s0\gesture.onnx --people mahimn --min-confidence 0.5 --no-gates
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

On the real recordings (section 1) the strength gate cost 2 gestures of 81. 16 were the right gesture held
back by the 0.6 threshold, and 11 were blocked by the walking-in and range gates. `--no-gates` switches all
three gates off at once, to see what they cost. The recall session's engine options (`--no-range-gate`, `--entering-gate-s 3`) are the finer tools.

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

Expect it to lower false triggers more than to raise recall. On the 13 real recordings of 2026-11-14 it
finds 3 hard negatives and nothing else, so it is not worth a training run yet. It starts to pay with
camera sessions, `negatives` sessions or recordings nobody labelled.

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

Better for 6 of 9, worse for none; in the unchanged room it did no harm.

**On the real recordings that gain did not show up.** Mahimn held out, the first 3 of each of his gestures
used for calibration, the rest scored (190 pictures):

| | accuracy | gestures recognised | non-gestures triggering |
|---|---|---|---|
| as trained | 0.837 | 0.556 | 0.050 |
| + BatchNorm statistics (default) | 0.855 | 0.514 | 0.010 |
| + prototypes only | 0.831 | 0.542 | 0.060 |
| + both | 0.855 | 0.556 | 0.010 |

So on a real new person it makes the model more careful, not better at recognising: use it when false
triggers are the problem at the demo table, not to raise recall. (Session by session the recall moved both
ways: 0.583 to 0.556 and 0.519 to 0.444 with the default.)

## 8. Burst cutting when the echoes are very loud

On this MacBook a hand is 40-48 dB above the background. The movement then never drops back to "quiet"
between raising the hand, the stroke and lowering it, so the engine sees one long movement and chops it at
its 1.4 s limit: **111 of the 218 real gestures (51 %)**, for Aarsh 80 %. The stroke lands at a different
place in every picture, or is cut in two.

`seg_rel_off_db` (in `sonar/config.py`, default 0 = off, nothing changes) lets a burst also end once the
movement is that many dB below the burst's own peak. An environment variable switches it on in every tool
at once, so existing recordings can be cut again without re-recording:

```powershell
$env:SONAR_SEG_REL_OFF_DB = "12"
.venv\Scripts\python -m sonar.labelqa train_out\mac_fixed --no-plot --out train_out\dataeng\qa_rel12
Remove-Item Env:SONAR_SEG_REL_OFF_DB
```

Measured on the 13 real recordings at 12 dB:

| | usual cutting | ends 12 dB below its peak |
|---|---|---|
| gesture bursts chopped at the 1.4 s limit | 51 % | 5 % |
| typical gesture burst | 1.10 s | 0.52 s |
| bursts per prompt | 2.1 | 2.7 |
| Mahimn's gestures recognised by the engine, same model, shipped settings | 36 of 81 | 44 of 81 |
| the same with the gates off and threshold 0.5 | 58 of 81 | 61 of 81 |
| false triggers in those two rows | 1 and 7 | 0 and 7 |

The model in that table was trained on pictures cut the usual way, and it still does better on a person it
never saw when the bursts are cut the other way. The recall session checked it independently with two
models on the same 81 gestures (pictures recognised 53 to 59 and 51 to 59; engine with their flags 54 to
60, false events 2 to 1). That is a strong hint from one person, not a proof. The proof is a model trained
on pictures cut that way: the recall session's suite has six such recipes (`HANDOFF_LANCE_RECALL.md`,
names ending in `_rel12`), each next to a twin with the usual cutting.

12 dB is not a lucky pick. The same replay at other settings (same model, Mahimn's 81 gestures):

| burst ends this far below its peak | off (usual) | 6 dB | 9 dB | 12 dB | 15 dB | 18 dB | 24 dB |
|---|---|---|---|---|---|---|---|
| recognised, shipped settings | 36 | 40 | 46 | 44 | 43 | 43 | 45 |
| false triggers there | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| recognised, gates off and threshold 0.5 | 58 | 56 | 61 | 61 | 62 | 63 | 64 |
| wrong gesture / false triggers there | 9 / 7 | 16 / 6 | 11 / 5 | 8 / 7 | 9 / 6 | 12 / 4 | 8 / 4 |

Anything from 9 to 24 dB gives about the same gain; 6 dB cuts too eagerly (more wrong gestures). Over all 13
sessions (40 minutes of audio) the false triggers at the shipped settings are 3 with either cutting
(aarsh 1 to 2, lance 1 to 1, lance2 0 to 0, mahimn 1 to 0).

The price is more small bursts for the model to reject (the hand going back becomes a burst of its own).
The loudest burst of a prompt stays the right one to label (section 1).

**A model trained with it needs the same switch wherever the engine runs**, because the engine cuts by its
own settings, not by the model's file. Put `SONAR_SEG_REL_OFF_DB=12` in front of the engine command (the
recall branch's `scripts/run_best_mac.sh` does), or change the default in `sonar/config.py` once it has won.

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
on the real recordings on a CPU (375 bursts of 6 sessions): bursts and pictures identical, physics rules
within 4.4e-7, model within 3.5e-6. All 13 sessions (40 minutes of audio, 836 bursts): 6 s to cut, 0.6 s to judge.
`torch.compile` is offered (`--compile`) but off: the model pass over a few thousand bursts takes a
fraction of a second, less than compiling would. **Nothing here was timed on a CUDA GPU.**

## 10. What is verified and what is not

Verified on the 13 real MacBook recordings (2026-11-14, Mac Studio CPU): the label check, including that
it flags the one reversed session and no other; the view tool; the engine replay and its miss reasons; the
burst-cutting numbers and the cutting switch; the calibration measurement; the weak-label pass; the fast
batched verdicts against the original code.

Verified in simulation only (145 tests pass: 89 older, 56 from the data engine): the camera labeller and clock sync; the recorder end to end with
a fake camera, and its fallbacks; the two dataset rules; weak labels and their use in training; BatchNorm
adaptation, export and use in the engine. `train_4090.py --weak` ran end to end on a CPU with tiny runs.

Not verified: a model retrained with the other burst cutting (the recall session's suite does that on the
4090), the camera recorder on a real webcam, and anything of this on a CUDA GPU.

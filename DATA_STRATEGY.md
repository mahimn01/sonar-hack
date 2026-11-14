# DATA_STRATEGY — a reliable demo model with the least hand-collected data

Written Friday 2026-11-13, evening. Covers the gesture model (the only part that needs training data).
Presence and breathing are signal processing in the core package: they need a few check recordings, not a dataset.

Words used below: **Doppler** = the pitch shift of the echo when the hand moves (up = approaching).
**SNR** = how far the echo stands above the noise. **Fine-tune** = continue training a ready model on a few new examples.
**None** = everything the model must ignore (fidgeting, typing, people walking past).

---

## 1. The answer

Do not hand-record hours. Do these four things, in this order.

1. **Fix the gesture definitions and the hardware level first (15 min, Section 6 and 7).** This removes more errors than any dataset.
2. **Record about 27 minutes of real audio from many different people, a few repetitions each (Section 5).** The judge is a stranger, so variety of people beats volume.
3. **Pre-train on synthetic gestures laid over your own real background recordings, then fine-tune on the real clips.** The generator is in `datagen/` and runs unattended (3,900 clips in 7-9 minutes on the Mac Studio, two runs, with other jobs running).
4. **Keep two safety nets:** the physics rules that need no training, and a 30-second "do each gesture 3 times" calibration at the demo table.

Skip tonight: public-dataset pre-training, self-supervised pre-training, camera auto-labelling (reasons in Section 3).

### Why this order: what was measured tonight

| Question | Result | Meaning |
|---|---|---|
| Do simulated gestures look like real ones? | Yes. On 191 real public clips (other hardware), a push is one upward hump, a pull one downward hump, a slide an up-then-down S: the same shapes the simulator draws. Picture: `datagen/examples/real_vs_sim.png`. | The physics transfers. |
| Does the simplest physics rule work on real audio? | Push vs pull by the sign of the pitch shift: **96 of 96 real clips correct.** | Direction of motion is safe to rely on. |
| Is a model trained only on synthetic data ready? | **Not on its own, but closer than expected.** The first attempt recognised 8 of 96 real pushes and pulls. After recalibrating the simulator against the real pictures (slower strokes, thicker arm echo, louder echoes) and fixing a flaw in my test network, the final one recognises **83 of 96 (86%)**, never confusing push with pull; the misses are called "none". On held-out synthetic clips it scores 85% and almost never mixes up two gestures. | Shapes transfer well once the simulator is checked against real pictures. Where "a gesture" ends and "just moving" begins still has to come from real data. |
| How much do a few real clips help? | On the real clips (none / slide / push / pull), fine-tuning the synthetic-trained model beat training from scratch at every size: **94 vs 76%** with 1 real clip per class, **94 vs 83%** with 3, **95.5 vs 87%** with 5, **97 vs 92%** with 10. One real clip per class with pre-training beat ten without. | A handful of real clips per gesture is what closes the gap. |
| Are rules alone enough? | In a harsh simulated hall (wind-ups, fidgeting, bystanders): the core's rules got swipes right 80% of the time, but called 49% of non-gesture movement a gesture and missed most one-way pushes and pulls. The trained model on the same clips: 85% overall, 11% of non-gesture movement called a gesture. | Fine as a fallback, not as the demo. |
| Does left/right survive a change of simulator? | A model trained only on my simulator was tested on the core package's separately written simulator: when it called a swipe, the direction was right 14 of 14 times. Push and pull did not carry over, because the two simulators define them differently (tap-and-return vs approach-and-hold). | The two direction cues are robust; wiring (which channel is left) must be checked once on the laptop. |

---

## 2. What was built (`datagen/`)

| File | What it does |
|---|---|
| `generate.py` | Makes labelled training examples: random virtual laptop + room + person, computes the 2-mic audio, runs it through the core's own feature pipeline (`sonar.features`), labels every movement burst, and writes `dataset.npz` in the core's dataset format. `--background` builds on real recordings. |
| `render.py`, `hardware.py`, `kinematics.py`, `scenes.py`, `physics.py` | The simulator: moving hand/arm/torso/walkers, two speakers, two mics, desk bounce, speaker/mic colouration, direct leakage, room reflections, hall noise, clicks, hum, gain steps, clock mismatch, 16-bit rounding. |
| `background.py` | Lays synthetic gestures over **real** no-gesture recordings: real laptop, room and noise; only the hand is computed. |
| `adapter.py` | The only file that touches `sonar/`. Reads the probe signal and calls the core pipeline, so synthetic and live features come from the same code. It followed four core API changes tonight without touching the simulator. |
| `render_examples.py` | Sanity pictures per class (`datagen/examples/*.png`). |
| `train_check.py` | Trains a small CNN on the synthetic set (1 minute), tests it on held-out synthetic clips and on the core's separately written simulator, runs the core's physics rules on the same clips for comparison, saves `pretrained.pt`. |
| `external_real_check.py` | Tests rules and model on the real public clips, and runs the "K real clips" comparison. |
| `heuristics.py`, `quicklook.py` | Training-free cues from raw audio (pitch-shift sign, two-speaker and two-microphone direction), used for checks and for auto-QA of recordings. |
| `tests/test_datagen.py` | 8 physics and format tests (pass). |

Run everything from the repo root with `datagen/.venv/bin/python` (Section 8).

Already on the Mac Studio (git-ignored, under `datagen/out/`):
- `synth_v5/dataset.npz`: 3,900 synthetic clips = 6,378 training pictures in the core's dataset format (2,299 gesture strokes, 1,885 hand-raise / hand-return bursts, 1,537 other movement, 657 idle). The core segmenter did not pick up 301 of the 2,600 scripted strokes; those are not labelled as gestures.
- `pretrain_core_3g/gesture.pt` and `gesture.onnx` (swipe_left, swipe_right, push) and `pretrain_core/` (the same plus pull): **the core's own model, trained by the core's own `train.py` on that set.** On held-out synthetic clips the 3-gesture model recognises 76-92% of each gesture and never confuses left with right; at the engine's 0.6 confidence threshold it recognises 71% of gestures and 9% of non-gesture pictures would trigger. Use it as the `--init` starting point for tonight's real fine-tuning. The engine can load the ONNX file now, but treat that as a smoke test, not as the demo model.
- The whole chain was run once with the documented commands on stand-in sessions made by the core recorder's simulator mode: record, generate on that background, fine-tune from the pre-trained weights with a held-out person, export ONNX. It ran clean. That proves the plumbing, not the accuracy.

---

## 3. The options, evaluated honestly

### 3.1 Physics simulator with random virtual laptops ("domain randomisation") — USE, as pre-training
- **What transfers:** the pitch-shift picture of each gesture, push/pull sign, the left/right cues, timing. All follow from geometry and the speed of sound.
- **What does not:** this laptop's loudness and colouration, how people here actually perform gestures, and above all where "a gesture" ends and "just moving" begins.
- **Evidence from others:** radar models trained only on synthetic data reached 81% on 12 activities and 96% after one session of the user's own data ([Vid2Doppler, CHI 2021](http://chrisharrison.net/projects/vid2doppler/vid2doppler.pdf)); 85-94% on new gestures from 200 synthetic samples ([RF Genesis, SenSys 2023](https://par.nsf.gov/servlets/purl/10552780)); 94% zero-shot on 8 activities ([SynMotion, SenSys 2022](https://www.cs.cityu.edu.hk/~zhenjili/2022-SenSys-SynMotion.pdf)). Older simulators fell to 23-39% on fine hand gestures they were not built for ([G3R, 2024](https://arxiv.org/html/2404.14934v1)). For sound specifically, a 19 kHz handwriting system trained on data synthesised from one volunteer reached 99% word accuracy on 9 unseen users, and its character error rose by 17% when the adaptation step was removed ([UltraWrite, PerCom 2024](https://yongpanzou.github.io/documents/UltraWritePercom.pdf)). No published paper pre-trains an acoustic gesture model on a physics echo simulator and reports real accuracy, so tonight's own numbers are the evidence.
- **Verdict:** gives the model its "eyes" for free and unlimited negatives. Never ship it without real fine-tuning.

### 3.2 Synthetic gestures on REAL backgrounds — USE, this is the upgrade
- Sound adds up linearly. Record a few minutes with the probe signal playing and nobody gesturing, then add computed hand echoes on top. Laptop colouration, leakage, room, hall noise and converter behaviour are then real by construction.
- `background.py` reads off the real recording: leakage level per tone and mic, where the signal period starts, and the clock mismatch. Checked on a stand-in recording: clock mismatch recovered exactly (18.0 ppm), period start within 7 samples.
- **Verdict:** highest value per minute of human time. Needs sessions S0-S2 of Section 5 (7 minutes).

### 3.3 Public datasets — DO NOT pre-train; use one as an outside sanity test
| Dataset | Signal | Usable? |
|---|---|---|
| [UltrasonicGS](https://github.com/yuejiaowang/database) ([paper](https://www.mdpi.com/1424-8220/23/4/1790)) | 20 kHz tone, 44.1 kHz **raw WAV**, push/pull/slide left/right, 1 mic, Raspberry Pi | Yes, as a real test only. No licence in the repo: evaluation only, do not redistribute. No "none" class, one mic, no user labels. |
| Ibrahim et al. 2021 ([arXiv 2108.02148](https://arxiv.org/abs/2108.02148)) | 20 kHz, stereo WAV, phone | Download link is dead. |
| [Dop-NET](https://doi.org/10.5522/04/25486597.v1) (24 GHz radar, similar Doppler scale) | wave/pinch/click/swipe | No push/pull, 6 GB, licence text conflicts. |
| [deep-soli](https://github.com/simonwsw/deep-soli), [TinyRadarNN](https://tinyradar.ethz.ch), [MCD-Gesture](https://github.com/leeyadong/cross_domain_gesture_dataset), [Widar3.0](https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset) | 60/77 GHz radar, WiFi | Dead link / non-commercial / signed agreement / 200+ GB. Different signal format. |
| SoundWave, AudioGest, UltraGesture, RobuCIR, SonicASL, Multiwave | — | No data released. |
- **Verdict:** nothing matches a laptop with two speakers and two mics at 18-21 kHz. Hours of conversion for a doubtful gain.
- Breathing: no public acoustic-sonar breathing set exists. Real breathing waveforms to drive the simulator's chest are available ([BIDMC, PhysioNet](https://physionet.org/content/bidmc/1.0.0/), ODC-By) if the breathing estimator needs a harder test.

### 3.4 Self-supervised pre-training on unlabelled recordings — SKIP
- Published gains come from 10,000 to 2,000,000 unlabelled clips, large batches and long training ([SimCLR](https://arxiv.org/abs/2002.05709), [BYOL-A](https://ar5iv.labs.arxiv.org/html/2103.06695), [COLA](https://ar5iv.labs.arxiv.org/html/2010.10915): +1.2 points with all of AudioSet).
- Under an hour of one laptop in one room is mostly the same background. Record continuously anyway: the same audio feeds 3.2.

### 3.5 Few-shot calibration at the table ("do each gesture 3 times") — USE as insurance
- Evidence: WiFi gestures 84 / 94 / 96 / 99% with 1 / 3 / 5 / 7 examples ([OneFi, SenSys 2021](https://ruixiao24.github.io/files/rui_onefi.pdf)); phone sonar 93.7% with 2 examples on unseen gestures ([EchoGest, IoT-J 2024](https://yongpanzou.github.io/documents/EchoGestIoTJ.pdf)); the method is [Prototypical Networks](https://arxiv.org/pdf/1703.05175).
- Recipe: run the trained network up to its last layer, average the 3 examples of each gesture into one "prototype" vector, classify by nearest prototype, reject as none when the nearest is too far. No training at the table. `SmallCNN.embed()` in `train_check.py` returns that vector.
- **Verdict:** 30 seconds per venue; protects against a different room or table on demo day.

### 3.6 Rules that need no training — USE as fallback and as auto-QA
- Push vs pull = sign of the pitch shift. Left vs right = (a) the right speaker's tone is shifted up more than the left speaker's while the hand travels right, (b) the phase difference between the two mics keeps turning one way. The core already has both.
- Evidence: SoundWave ran unchanged on 11 computers with 87-100% per gesture, 0 false gestures in an hour once 4 consecutive frames were required, but typing caused false events ([CHI 2012](https://modernmobile.cs.washington.edu/docs/soundwave.pdf)). One tone per laptop speaker gave 93.9% same-user accuracy on 14 gestures ([Multiwave, GI 2017](https://www.cs.ucf.edu/~jjl/pubs/multiwave_gi2017_final.pdf)).
- In my simulation, once the stroke itself is isolated, the two-speaker cue gave the right direction in 74 of 77 swipes (96%) and the two-microphone cue in 70 of 77 (91%). The hard part is isolating the stroke: a simple "loudest fast burst" finder picked the wind-up or the hand coming back about half the time. That is the job the trained model and the segmenter's lock-outs must do.
- **Verdict:** keeps the demo alive if the model misbehaves; also flags recordings where the performer swiped the wrong way.

### 3.7 Camera or IMU auto-labelling — DEFER to Saturday, only if negatives run short
- Idea: the laptop's own webcam runs hand tracking only while collecting, so 10 minutes of natural motion label themselves. Same machine, same clock, so syncing is easy.
- Facts: `mediapipe` 1.2.3 (Apache-2.0) has a Windows wheel for Python 3.12; the old `mp.solutions` API is gone, use `mp.tasks.vision.HandLandmarker` ([docs](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)); it gives no swipe labels, those must come from the wrist path. Precedents: [EchoWrist](https://arxiv.org/html/2401.17409) and [Ring-a-Pose](https://arxiv.org/html/2404.12980v2) used MediaPipe as ground truth for sonar wearables.
- Cost: 4-6 hours including labelling rules and sync. For five coarse gestures the prompted recorder plus the core segmenter already cut and label strokes almost for free.
- Phone IMU on the wrist ([phyphox](https://phyphox.org/docs/remote-interface/)) is more work than the camera for the same result. Skip.

### 3.8 Augmenting the real clips — USE, cheap
- Mirror: swap the two speaker tones and the two mics, flip the sign of the lateral channel, swap swipe_left and swipe_right. Doubles the swipe data.
- Small time shifts (±0.1 s), brightness ±20%, mixing in real background, speed change 0.8-1.25× (stretch time and scale Doppler together).
- Evidence: gains are real but single digits ([Salamon & Bello](https://ar5iv.labs.arxiv.org/html/1608.04363): 73 → 79%). RobuCIR multiplied 3,600 real samples 100× by speed and distance changes and reached 98.4% with held-out users ([INFOCOM 2020](https://wangyanwen01.github.io/papers/infocom2020.pdf)). In a phone-sonar study, frequency/time masking did not help, while feeding both mics as one input did (93.6 vs 87.2% on unseen users, [Ibrahim et al.](https://arxiv.org/abs/2108.02148)).

---

## 4. Ranked plan for tonight

| # | Step | People time | Compute | Expected result | 
|---|---|---|---|---|
| 1 | Hardware checks and gesture definitions (Sections 6, 7) | 15 min | – | Removes the biggest unknowns: level, channel order, hum, what "push" means. |
| 2 | Record sessions S0-S5 (Section 5) | 40 min, 3 people | – | About 27 min of audio: ~165 gesture clips from 9 people plus real negatives. |
| 3 | `generate.py --background` on S0-S2, plus a pure synthetic set | 0 | 20 min | ~7,000 labelled examples on the real laptop's acoustics. |
| 4 | Pre-train on step 3, fine-tune on step 2 with the augmentations of 3.8 | 0 | 15 min | The demo model. |
| 5 | Test on the 2 held-out people. If below target, record more people, not more repetitions. | 10 min | 5 min | An honest number. |
| 6 | Wire the 3-shot calibration and keep the physics rules as fallback | – | core work | Insurance against a new room. |

Expected accuracy, stated as estimates:
- **Rules only:** fine for the presenter's own deliberate gestures in a quiet moment; unreliable with a fidgeting stranger. In the simulated hall test the rules called about half of all non-gesture movement a gesture.
- **Synthetic only:** 86% of real pushes and pulls on other hardware, and left/right is untested on real audio. Not demo-ready on its own.
- **Real only, about 165 clips:** likely 85-93% for people in the set, lower for strangers. Reference: 93.6% on 3 unseen users from 1,920 clips of 4 users ([Ibrahim et al.](https://arxiv.org/abs/2108.02148)).
- **Synthetic pre-training + real fine-tune:** the best option. On real public clips, 5 real examples per class gave 95.5% with synthetic pre-training against 87% without. Expect roughly 90-95% for enrolled people and 80-90% for strangers on swipe-left / swipe-right / push. These are estimates until step 5 measures them.
- **Plus calibration at the table:** protects that number when the venue changes.

Robustness trade-off: every step that raises recall also raises false triggers. Pick the operating point on real negatives (session S1), not on the gesture clips.

---

## 5. The real data still needed

Three people: **Operator** (laptop, recorder, watches the level read-out), **Performer**, **Wrangler** (brings in other hackers, says the prompts, keeps a tally sheet).
Record at the demo spot, laptop on the demo table, lid at the demo angle, charger as it will be during the demo.
Record continuous sessions with prompts, not isolated clips: the moments between gestures (hand raise, hand dropping back) are the negatives the model needs most.

| Session | What | Who | Audio | Used for |
|---|---|---|---|---|
| S0 | Empty: nobody within 2 m, hall as it is | – | 2 min | Real background for synthetic gestures; "empty" check |
| S1 | Life at the laptop: typing, reaching for a cup, scratching the head, talking with hands, leaning. **No deliberate gestures.** | Performer | 3 min | Real negatives; sets the false-trigger rate |
| S2 | People walking past and behind, someone sitting down and leaving | Wrangler + 2 | 2 min | Real negatives; presence check |
| S3 | Presenter: swipe_left, swipe_right, push × 15 each (5 close and brisk, 5 normal, 5 farther and slower). Pull × 15 only if it stays in the demo. | Presenter | 4 min | Fine-tuning |
| S4 | Crowd: 8 other people × 3 gestures × 5 each. Show the gesture once, do not coach. | Wrangler recruits | 10 min | Fine-tuning and the stranger test |
| S5 | 6 walk-ins from different directions; 2 people standing still for 60 s each while someone counts their breaths (also once at a paced 12 and 20 per minute) | anyone | 6 min | Presence and breathing checks only |

Totals: about 27 minutes of audio, about 40 minutes of wall-clock time, about 165 gesture clips without pull (45 presenter + 120 crowd).
Bare minimum if the night runs out: S0 + S1 + S3 (9 minutes of audio).

The core recorder already does this (it prompts on screen, saves one continuous recording per session, and shows after each trial what the physics rules made of it):

```bash
# on the Windows demo laptop
py -3.12 -m sonar.record --person presenter --script scene                       # S0-S2 and the walk-ins of S5, about 4 min
py -3.12 -m sonar.record --person presenter --script gestures --reps 15 --gestures swipe_left,swipe_right,push   # S3
py -3.12 -m sonar.record --person guest1 --script gestures --reps 5 --gestures swipe_left,swipe_right,push       # S4, once per guest
py -3.12 -m sonar.record --person presenter --script breathing                   # S5 breathing, about 2.5 min
```

How this differs from DESIGN.md section 7: that plan assumes training on real data only and targets 4 people × 3 gesture sessions = 576 gestures, about 2 hours with changeovers. With synthetic pre-training, tonight's evidence says far fewer repetitions are needed, so this plan trades repetitions for people: 9 people, about 165 gestures, about 40 minutes. Everything else in DESIGN.md section 7 still applies (vary distance and speed, let the hand drop instead of swiping back, fixed Windows volume, one session at the demo table on demo morning). If the stranger test in step 5 comes out weak, grow towards the DESIGN.md target, adding people first.

Give every guest their own `--person` name: the trainer holds out whole people with `--val-person`.
The `scene` sessions (no prompted gestures) are the real backgrounds for `datagen.generate --background`.

Rules that make the data worth having:
- **Hold out two crowd people entirely** as the test set. Never report accuracy on people who are in the training set.
- More people beats more repetitions. If the stranger test is weak, add people.
- Right after each session, run the physics QA on it and re-record clips where the measured direction disagrees with the prompt.
- Re-run `datagen.generate` whenever the core signal or feature code changes: `dataset.npz` is only a cache of that code.

---

## 6. Demo design choices that cut the data need

1. **Three gestures, not five.** Swipe left, swipe right, push. Pull is the troublemaker: to pull, the hand must first approach, which looks like a push. The app does not use pull.
2. **Define push as "move the hand at the screen and hold it there for half a second".** In the real public data a push is exactly that single approach. A push defined as tap-and-return is half a pull.
3. **Swipe with the hand that starts on that side.** Swiping left with the right hand needs no wind-up. Swiping right with the right hand first carries the hand left, which is a real leftward motion. Tell judges: "sweep once, from where your hand is".
4. **Map "next slide" to the direction that tests better on real data,** and treat an uncertain direction as "next". A wrong "next" is invisible; a wrong "back" is embarrassing.
5. **Brisk and close.** In simulation, recognition fell from 82% for loud echoes (30 dB or more above the noise) to 47% for faint ones (12-20 dB), and from 87% for strokes faster than 1.6 m/s to 55% for strokes slower than 0.6 m/s. Say "about a hand-length from the screen, like wiping a window".
6. **Presenter demonstrates first.** People copy what they just saw, so the judge's gesture lands inside the training set.

---

## 7. Things the simulator and the real clips surfaced for the hardware and core work

1. **The signal is only a few steps of a 16-bit converter.** With the measured leakage (−73 dB) and a 0.35 tone, the received tone peaks at about −82 dBFS, which is 2.6 steps of 16-bit audio; hand echoes are a fraction of one step. Check in 10 seconds: record silence and count distinct sample values. If only a handful appear, ask the driver for 24-bit or float capture and raise the speaker volume. This is an inference from the reported levels, so verify it.
2. **Mains hum.** The real public clips show steady lines 50 Hz either side of the tone. On North American power expect 60 and 120 Hz. Compare charger in and out; the core's per-bin noise floor should absorb steady lines, but check.
3. **Which channel is left.** Both direction cues flip sign if mic or speaker channels are swapped. One slow swipe to the right settles it; the core has `heur_w_mics` and `heur_w_speakers`, datagen has `swap_mics` and `swap_speakers`.
4. **Real echoes can be much stronger and thicker than a point model.** On the public hardware the moving echo was 6-16 dB above the leakage, and the trace covers every speed from zero to the hand's speed (the arm). The simulator now models both; re-check against the first real HP recordings with `render_examples.py`.
5. **Real strokes are slower than intuition says:** about 0.5-0.7 s long, peaking near 80-150 Hz of shift (0.7-1.3 m/s).
6. **One-way pushes.** On the real public clips the core's physics rules called 44 of 48 real pushes and all 48 real pulls "none", and called 94 of 95 slides "push" (those clips carry no left/right information, so the swipe answer was not available). A real push there is a single approach with no return; the rule seems to expect an approach followed by a retreat. Check this before relying on the fallback.

---

## 8. How to run

```bash
cd ~/Documents/Dev/sonar-hack

# sanity pictures per class (random virtual laptops, and --clean for pure physics)
datagen/.venv/bin/python -m datagen.render_examples --out datagen/examples
datagen/.venv/bin/python -m datagen.render_examples --out datagen/examples --clean --classes swipe_left,swipe_right,push,pull

# tests (8, about 5 s)
datagen/.venv/bin/python -m pytest datagen/tests -q

# synthetic training set (3,900 clips -> about 6,400 training pictures in 7-9 min on 8 cores)
datagen/.venv/bin/python -m datagen.generate --out datagen/out/synth_v5 --per-class 650 --negatives 2

# the same, built on REAL background recordings from the demo laptop (sessions S0-S2)
datagen/.venv/bin/python -m datagen.generate --out datagen/out/synth_real_bg --per-class 650 \
    --background "data/<date-time>_presenter_scene/audio.wav"     # several files: comma-separated

# 1) pre-train the CORE model on synthetic data with the core's own training script.
#    --labels must be the same list you will fine-tune with (the weights only load if the classes match).
.venv/bin/python -m sonar.train --dataset datagen/out/synth_v5/dataset.npz --labels swipe_left,swipe_right,push \
    --out datagen/out/pretrain_core_3g --epochs 30

# 2) tonight's real run: fine-tune from those weights on the recorded sessions, test on a held-out person
.venv/bin/python -m sonar.train --data data/ --dataset datagen/out/synth_real_bg/dataset.npz \
    --init datagen/out/pretrain_core_3g/gesture.pt --labels swipe_left,swipe_right,push \
    --val-person <name of a held-out person> --out models/

# checks used for the numbers in this document
datagen/.venv/bin/python -m datagen.train_check --data datagen/out/synth_v5 --core-sim 40
datagen/.venv/bin/python -I -c "import sys; sys.path.insert(0, '.'); from datagen.external_real_check import main; main()" \
    --wavs /path/to/SingleGesture/wavs --model datagen/out/synth_v5/pretrained.pt --fewshot
```

What `generate.py` writes into `--out`:
- `dataset.npz` in exactly the format of `sonar/dataset.py` (`X [N, C, F, T + 2*pad]` float16, `y`, `labels`, `group`, `person`, `source="synthetic"`, `peak_db`, `pad`, `config`). The core's `train.py` loads it with `--dataset`, keeps synthetic examples out of validation, and can start from pre-trained weights with `--init`. Checked end to end: the core script trained and exported a model from this file.
- `manifest.jsonl`: one line per example (label, kind, timing, virtual hardware, scene details, echo loudness).
- `audio/*.flac` if `--save-audio K` is given: raw 2-mic audio of the first K clips of each class.

Labels: a movement burst is the gesture if it covers the scripted stroke; the hand raise before it, the hand dropping back after it, and every other movement are "none"; a few pictures of nothing are added as "none".
The feature pictures are a cache of the core feature code. Re-run `generate.py` after any change to `sonar/` signal or feature settings (the file stores the config it was made with, and `train.py` refuses to mix profiles).
`datagen/out/` and `datagen/.venv/` are git-ignored. `datagen/` also runs in the repo's main `.venv`.

---

## 9. Sources

Acoustic gestures on ordinary speakers and mics
- SoundWave, Gupta et al., CHI 2012: https://modernmobile.cs.washington.edu/docs/soundwave.pdf (numbers checked in the PDF)
- Multiwave, Pittman & LaViola, GI 2017: https://www.cs.ucf.edu/~jjl/pubs/multiwave_gi2017_final.pdf
- AudioGest, Ruan et al., UbiComp 2016: https://research.polyu.edu.hk/en/publications/audiogest-enabling-fine-grained-hand-gesture-detection-by-decodin/
- RobuCIR, Wang et al., INFOCOM 2020: https://wangyanwen01.github.io/papers/infocom2020.pdf
- UltrasonicGS, Wang et al., Sensors 2023: https://www.mdpi.com/1424-8220/23/4/1790 and https://github.com/yuejiaowang/database
- Ibrahim et al., 2021: https://arxiv.org/abs/2108.02148
- EchoGest, IEEE IoT-J 2024: https://yongpanzou.github.io/documents/EchoGestIoTJ.pdf
- UltraWrite, PerCom 2024: https://yongpanzou.github.io/documents/UltraWritePercom.pdf
- LLAP, MobiCom 2016: https://samsonsjarkal.github.io/KeSun/files/mobicom16_ultrasound.pdf
- Beyond-Voice, IPSN 2024: https://arxiv.org/html/2306.17477v2

Synthetic data and transfer
- Vid2Doppler, CHI 2021: http://chrisharrison.net/projects/vid2doppler/vid2doppler.pdf, code https://github.com/FIGLAB/Vid2Doppler
- RF Genesis, SenSys 2023: https://par.nsf.gov/servlets/purl/10552780
- SynMotion, SenSys 2022: https://www.cs.cityu.edu.hk/~zhenjili/2022-SenSys-SynMotion.pdf
- G3R, 2024: https://arxiv.org/html/2404.14934v1
- IMU2Doppler, IMWUT 2021: https://mynkgoel.github.io/pdfs/imu2dop.pdf
- OneFi, SenSys 2021: https://ruixiao24.github.io/files/rui_onefi.pdf
- RF-Diffusion, MobiCom 2024: https://arxiv.org/html/2404.09140v1
- SimHumalator, 2022: https://discovery.ucl.ac.uk/10140989/1/SimHumalator_AESS_Magazine_accepted.pdf
- Domain randomisation, Tobin et al., IROS 2017: https://arxiv.org/abs/1703.06907
- pyroomacoustics 0.10.1 (MIT; no moving sources): https://pypi.org/project/pyroomacoustics/

Few-shot, self-supervised, augmentation
- Prototypical Networks: https://arxiv.org/pdf/1703.05175
- RF-Net, SenSys 2020: https://arxiv.org/pdf/2111.04566
- SimCLR: https://arxiv.org/abs/2002.05709 · BYOL-A: https://ar5iv.labs.arxiv.org/html/2103.06695 · COLA: https://ar5iv.labs.arxiv.org/html/2010.10915 · TS-TCC: https://ar5iv.labs.arxiv.org/html/2106.14112
- SpecAugment: https://arxiv.org/abs/1904.08779 · mixup: https://arxiv.org/abs/1710.09412 · Salamon & Bello: https://ar5iv.labs.arxiv.org/html/1608.04363

Auto-labelling
- MediaPipe Hand Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python · https://pypi.org/project/mediapipe/
- EchoWrist, CHI 2024: https://arxiv.org/html/2401.17409 · Ring-a-Pose, IMWUT 2024: https://arxiv.org/html/2404.12980v2
- phyphox remote interface: https://phyphox.org/docs/remote-interface/

Breathing by sonar (for the core's estimator; no public sonar breathing dataset exists)
- ApneaApp, MobiSys 2015 (18-20 kHz chirps, up to 1 m, under 0.11 breaths/min error): https://homes.cs.washington.edu/~gshyam/Papers/apneaapp.pdf
- BreathJunior, MobiCom 2019: https://homes.cs.washington.edu/~gshyam/Papers/whitenoise.pdf
- SonarBeat (median error 0.2 breaths/min): https://eng.auburn.edu/~szm0001/papers/ACMTCH2021.pdf
- Opioid-overdose sonar, Sci. Transl. Med. 2019: https://par.nsf.gov/servlets/purl/10095248
- Smart-speaker heart rhythm, Commun. Biol. 2021: https://pmc.ncbi.nlm.nih.gov/articles/PMC7943557
- Radar vital signs with references, Schellenberger et al. 2020 (CC BY 4.0): https://figshare.com/articles/dataset/12186516
- BIDMC respiration waveforms (ODC-By): https://physionet.org/content/bidmc/1.0.0/

Physical constants used in the simulator
- Air absorption (ISO 9613-1 formula; 0.52 dB/m at 20 kHz, 20 °C, 50% humidity; my implementation matches the ISO table within 1.4%): https://sengpielaudio.com/AirdampingFormula.htm
- Chest motion about 2 mm RMS in quiet breathing: https://pmc.ncbi.nlm.nih.gov/articles/PMC8070581
- Laptop near-ultrasound output about 63 dB SPL at 20 cm, uneven response: https://arxiv.org/pdf/2103.11261
- Audibility of 18-20 kHz to young listeners: https://physicstoday.aip.org/features/ultrasound-in-air

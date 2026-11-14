# Laptop sonar — gestures, presence and breathing without a camera

The laptop speakers play a quiet signal just above hearing (18.5-21.5 kHz), the
built-in microphones record the echoes, and this package turns them into

- **hand gestures** (swipe left / swipe right / push / pull) to control slides,
- **presence** (empty / someone entering / someone present),
- the **breathing rate** of a person sitting still in front of the laptop.

No camera, nothing leaves the machine. Design and reasoning: `DESIGN.md`. Event
schema for the UI: `INTERFACES.md`. Everything below is run from the repo root.

## Quick start on the Windows demo laptop

### 0. Install (once)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Windows sound settings (once):
- Close Teams, Discord, Zoom and browser tabs that play sound. They block exclusive mode.
- Settings > System > Sound > the speaker > Advanced: tick **"Allow applications to take exclusive control"**. Same for the microphone array.
- Set the speaker volume (start at 60 %) and **do not touch it again**: recordings, training and the demo must use the same volume.
- Plug in the charger. Fix the screen angle; moving the lid changes the echo picture (re-run calibration if you do).

In the commands below `py` means `.venv\Scripts\python`.

### 1. Self-test (1 minute)

```powershell
py -m sonar.selftest
```

It plays test tones through each speaker and prints the signal-to-noise ratio
(SNR) per frequency, whether the receiver locks, clock drift, clipping, audible
leakage, whether the left and right speaker channels come from different
places, and a verdict with a recommended profile. What you want to see:

| line | good | if not |
|---|---|---|
| weakest pilot tone SNR / chirp band SNR | 50+ / 45+ dB | raise the volume and re-run; or use the recommended profile |
| echo headroom (tones) | 40+ dB | same |
| receiver lock | True, re-locks 0 | close other audio apps; `--blocksize 3840`; `--duplex` |
| clock drift | about 0 ppm | nothing to do, it is compensated |
| microphone peak | below -6 dBFS | lower the volume |
| audible leak | PASS | lower the volume or use `--profile clean` |
| left/right speaker channels | 60+ microseconds apart | fine either way: the two-microphone cue still gives swipe direction |

Listen once as well: you should hear nothing, or a faint hiss at most. If you
change profile, use the same `--profile NAME` for every later command.

### 2. Record (about 10 minutes per person)

```powershell
py -m sonar.record --person alice --script gestures --reps 12   # 48 prompted trials, ~4.5 min
py -m sonar.record --person alice --script scene                # empty / walk in / sit / type / walk past, ~3.5 min
py -m sonar.record --person alice --script breathing            # paced breathing 12/15/20 per minute, ~2.5 min
```

Follow the prompts. After each gesture the recorder says whether it saw a
movement and what the training-free physics rules think it was. The last line
(`physics guess agreed with the prompt in X/Y`) is the health check: high
agreement means the hardware does what the design assumes.

How to gesture: hand open, about 25 cm in front of the screen, start on "GO",
then let the hand **drop** back to your lap (do not swipe back).
- swipe left / right: sweep across the screen in about half a second.
- push: palm quickly towards the screen and straight back.
- pull: reach out slowly to ~15 cm from the screen, then pull back quickly.

Collect at least 2 people × 2 gesture sessions to start (20 min); 4 people × 3
gesture sessions plus one scene and one breathing session each for a solid model
(about 75 minutes of recording). Vary seat distance, speed and hand between
sessions. Keep one person out of training for the honest score.

Each session is a folder `data/<date-time>_<person>_<script>/` with the raw
audio and the prompt list. `py -m sonar.plot data\<session>` draws it (needs
matplotlib) so you can see whether swipes and pushes look different.

### 3. Train (a few minutes; on this laptop's CPU, the Mac Studio or the 4090)

```powershell
.venv\Scripts\python -m pip install -r requirements-train.txt      # adds PyTorch; once
py -m sonar.train --data data --out models                         # holds out whole sessions for the score
py -m sonar.train --data data --out models --val-person bob        # score on a person the model never saw
```

It prints the confusion matrix, the score "as deployed" (how many gestures are
recognised and how many non-gestures would trigger at the engine's confidence
threshold) and writes `models/gesture.onnx` + `gesture.json`. To train
elsewhere, copy the `data/` folder there and bring `models/` back. Extra
datasets (for example synthetic data from `datagen/`) are added with
`--dataset file.npz`. Drop a weak class with `--labels swipe_left,swipe_right,push`.

### 4. Run the engine

```powershell
py -m sonar.engine                 # uses models\gesture.onnx if present, otherwise the physics rules
py -m sonar.engine --physics       # physics rules only: works with no training at all
py -m sonar.engine -v              # also print signal status once per second
```

Keep hands away for the first 5 seconds (calibration). The engine prints every
gesture, presence change and breathing rate, and serves the same as JSON events
on `ws://127.0.0.1:8765` for the UI (`app/`). Stop with Ctrl+C.

For the breathing demo: **sit** (or lean on the table) 0.5-0.8 m from the
laptop, face it, keep your hands still. The waveform appears after ~4 s, the
rate after 10-15 s. A person standing free sways in the same rhythm as
breathing, so a standing reading is less reliable.

### No hardware? (Mac, or the laptop is busy)

```bash
python -m sonar.engine --backend sim                              # scripted simulated visitor, real pipeline
python -m sonar.engine --backend file --file data/<session>/audio.wav   # replay a recording as if live
python -m sonar.selftest --backend sim
python -m sonar.record --person test --backend sim --fast --reps 4
```

## Troubleshooting

| symptom | fix |
|---|---|
| "could not open the audio devices" | another app holds the speaker or mic; exclusive control not ticked; try `--list-devices`, `--in-device "Microphone Array"`, `--out-device "Speaker"`, `--duplex`, `--blocksize 3840` |
| self-test SNR near 0, no lock | you are in shared mode or the wrong devices were picked: check the device names it printed |
| recorder says "NO movement detected" | closer (20-25 cm), bigger, faster; raise the volume and re-run the self-test |
| gestures fire when nobody gestures | record a `scene` session with those distractions and retrain; raise `--min-confidence 0.7` or `--min-peak-db 8` |
| swipes mixed up left/right | the model needs more swipe data, or the mic channels are swapped relative to training: retrain on this laptop's own recordings |
| a swipe triggers the opposite gesture | let the hand drop instead of swiping back; `--swipe-hold 0.4` |
| breathing stays "acquiring" | sit, 0.5 m, hands on lap, wait 15 s; raise the volume; check `status.snr_db` |
| someone hears a whine | lower the volume; `--profile clean` |
| engine prints `status ... locked False` repeatedly | audio is dropping out: close CPU-heavy apps, `--blocksize 3840` |

## Development

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-train.txt
.venv/bin/python -m pytest            # 84 tests, ~2 minutes, no audio hardware needed
.venv/bin/python -m pytest -m "not slow"    # the fast ones (~30 s)
```

The tests run a physics simulator (`sonar/sim.py`) through the real pipeline: a
reflector moving at a known speed must give the right Doppler shift, a chest
breathing at 15 per minute must be measured within 1, gestures must be detected
and named, and the whole record → train → export → classify chain must work.

| file | what it does |
|---|---|
| `sonar/config.py` | every signal and processing parameter; named profiles |
| `sonar/tx.py` | the transmitted waveform (two pilot tones + one chirp, 20 ms period) |
| `sonar/dsp.py` | prefilter, tone demodulation, Doppler spectrogram, chirp receiver (sync, drift, echo profile) |
| `sonar/frontend.py`, `sonar/pipeline.py` | the streaming chain used everywhere |
| `sonar/gestures.py` | noise floor, burst segmenter, physics-only classifier |
| `sonar/presence.py`, `sonar/breathing.py` | presence state machine, breathing estimator |
| `sonar/audio.py` | live (WASAPI exclusive), simulator and file backends |
| `sonar/engine.py`, `sonar/server.py` | realtime engine, WebSocket event stream |
| `sonar/selftest.py`, `sonar/record.py`, `sonar/plot.py` | hardware check, data collection, session plots |
| `sonar/features.py`, `sonar/dataset.py`, `sonar/model.py`, `sonar/train.py`, `sonar/classifier.py` | features, dataset + augmentation, CNN, training + ONNX export, inference |
| `sonar/sim.py` | physics echo simulator |
| `tests/` | pytest suite |

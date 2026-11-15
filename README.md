# Laptop sonar and Sonar Care: gestures, presence and breathing without a camera

A MacBook's two speakers play a quiet signal just above hearing (18.5 to 21.5 kHz), its one microphone records the
echoes, and this repo turns them into

- **hand gestures** (swipe left, swipe right, push) that press real keys in any app,
- **presence** (empty, someone arriving, someone present) with a rough distance,
- an early **breathing-rate** estimate for a person sitting still,

and **Sonar Care**, the app built on top: touch-free control, a caregiver's room monitor, a help gesture (two
pushes) and an experimental fall alert.

No camera, nothing leaves the machine. Everything below is run from the repo root.

| Read this | For |
|---|---|
| `docs/SONAR_CARE.md` | the app: running it, permissions, the 90-second demo, troubleshooting, how it is built |
| `INTERFACES.md` | the engine's events and commands (what a UI consumes) |
| `DESIGN.md` | how the sonar works and why |
| `app/README.md` | the pitch slides, the mock engine, the page code |
| `HANDOFF_LANCE.md`, `TRAINING.md` | recording new people on the MacBook and training a model |

## Quick start on the MacBook

```bash
git pull
scripts/sonar_care.sh --dry-run     # first look: shows which key each gesture WOULD press, presses nothing
scripts/sonar_care.sh               # the real thing
```

That one command creates `.venv` if it is missing, installs only the packages that are missing, starts the sonar
engine with the `calm50` model, puts a Sonar Care item in the menu bar and opens the dashboard at
`http://127.0.0.1:8800/care.html` (the first time: a guided setup). No Xcode, no code signing, no build step. Stop
with Ctrl+C or **Quit Sonar Care** in the menu; the engine it started stops too.

Before you start:

- Speaker volume where it was when the model's recordings were made, and do not change it afterwards.
- Nothing else playing sound, headphones and Bluetooth audio disconnected, charger in.
- In **Audio MIDI Setup** the MacBook microphone and speakers are at 48,000 Hz.
- Keep the lid angle fixed. Moving it changes the echo picture: press Calibrate if you do.
- macOS asks once for the **Microphone** (everything needs it) and for **Accessibility** (only for pressing keys)
  for the terminal app you start it from. `docs/SONAR_CARE.md`, section 2.

How to gesture: open hand, a hand-length (20 to 25 cm) from the screen. *Swipe*: sweep across in about half a
second, then let the hand drop to your lap; do not swipe back. *Push*: palm quickly toward the screen and straight
back. *Help*: push, bring the hand back, push again about a second later.

### Only the engine

```bash
SONAR_MODEL_DIR=models-candidate-mac-calm50 scripts/run_best_mac.sh        # live: the MacBook's speakers and microphone
SONAR_MODEL_DIR=models-candidate-mac-calm50 scripts/run_best_mac.sh -v     # also print the signal status once a second
```

The script picks the MacBook's built-in microphone and speakers and passes the flags the model was packaged with
(`--mics 1 --no-range-gate --entering-gate-s 3`, and how a movement burst is cut). The second line it prints must
read `classifier: model (gesture.onnx)`. Keep hands away for the first 5 seconds (calibration). The engine prints
every gesture, presence change and breathing rate, and serves the same as JSON events on `ws://127.0.0.1:8765`.
Sonar Care uses an engine that is already running on that port instead of starting a second one.

### No sonar at hand

```bash
scripts/sonar_care.sh --mock                 # scripted fake data on any Mac, marked "Simulated data"
scripts/sonar_care.sh --backend sim          # the real engine and model with a simulated visitor, no sound
SONAR_MODEL_DIR=models-candidate-mac-calm50 scripts/run_best_mac.sh --backend sim --min-peak-db 3      # the engine alone, simulated
.venv/bin/python -m sonar.engine --backend file --file data/<session>/audio.wav --mics 1               # replay a recording as if live
```

`--min-peak-db 3` is needed with the simulator because `calm50` ignores movements weaker than real hands at this
laptop, and the simulated hand is weaker. Sonar Care adds it by itself in `--backend sim`.

## The model in the repo

`models-candidate-mac-calm50/` (weights included, so a `git pull` delivers a runnable model). It knows three
gestures and was trained on recordings of three people made on one MacBook Pro. A model only works on the laptop
whose recordings trained it.

Honest score, each person held out of training in turn: 133 of 206 gestures recognised (65 %), with 22 of 499 other
movements triggering (4.4 %). On the best-matched held-out person: 70 of 81 (86 %). Breathing on one microphone is
experimental. Not a medical device.

## Recording more people and training (on the MacBook)

```bash
.venv/bin/python -m sonar.selftest --in-device MacBook --out-device MacBook     # 1 minute: signal strength, lock, audible leak
scripts/mac_record_all.sh                                                       # one person: scene, gestures, breathing (about 10 min)
scripts/train_4090.sh --device mps --size small --release models-release-mac    # train on the MacBook (needs requirements-train.txt)
```

The steps, what good output looks like and how to train on the RTX 4090 box instead are in `HANDOFF_LANCE.md` and
`TRAINING.md` (section 9, "Demo on a MacBook"). More different people beats more repetitions. Keep one person out
of training for the honest score.

## Troubleshooting

For the app (menu bar, dashboard, keys blocked, alerts) see `docs/SONAR_CARE.md`, section 5. For the engine:

| Symptom | Fix |
|---|---|
| "could not open the audio devices" | Another app holds the microphone, or the terminal has no Microphone permission (System Settings > Privacy & Security > Microphone). List devices with `.venv/bin/python -m sonar.engine --list-devices`; pick them with `SONAR_IN_DEVICE=... SONAR_OUT_DEVICE=...` |
| The second line says `classifier: physics`, not `model` | The weight file is missing: `git pull` |
| Status shows a low SNR (under 25 dB) or `locked False` | Volume too low or changed, headphones or Bluetooth audio connected, the devices are not at 48,000 Hz, or a CPU-heavy app is causing dropouts. Then Calibrate |
| Gestures fire when nobody gestures | Raise the confidence floor (Sonar Care > Settings > Fine tuning, or `--min-confidence 0.7`) |
| A swipe triggers the opposite gesture | Let the hand drop instead of swiping back |
| Breathing stays on a dash | Sit 0.5 to 0.8 m away, hands on your lap, wait 15 s. It is an early estimate and often has no confident reading |
| Someone hears a whine | Lower the volume (then re-record or expect weaker recognition), or record and run with `--profile clean` |

## Development

```bash
.venv/bin/python -m pytest -m "not slow" -q     # the engine, the app and Sonar Care (about 2 minutes, no audio hardware)
node --test app/tests/                          # the page code: adapters, store, live client (Node 22+)
```

The engine tests run a physics simulator (`sonar/sim.py`) through the real pipeline: a reflector moving at a known
speed must give the right Doppler shift, a chest breathing 15 times a minute must be measured within 1, gestures
must be detected and named, and the record, train, export, classify chain must work.

| File | What it does |
|---|---|
| `sonar/config.py` | every signal and processing parameter; named profiles |
| `sonar/tx.py` | the transmitted waveform (two pilot tones, one per speaker, plus one chirp; 20 ms period) |
| `sonar/dsp.py` | prefilter, tone demodulation, Doppler spectrogram, chirp receiver (sync, drift, echo profile) |
| `sonar/frontend.py`, `sonar/pipeline.py` | the streaming chain used everywhere |
| `sonar/gestures.py` | noise floor, burst segmenter, physics-only classifier |
| `sonar/presence.py`, `sonar/breathing.py` | presence state machine, breathing estimator |
| `sonar/audio.py` | live audio, simulator and file backends |
| `sonar/engine.py`, `sonar/server.py` | realtime engine, WebSocket event stream |
| `sonar/selftest.py`, `sonar/record.py`, `sonar/plot.py`, `sonar/replay.py` | hardware check, data collection, session plots, scoring a recording |
| `sonar/features.py`, `sonar/dataset.py`, `sonar/model.py`, `sonar/train.py`, `sonar/classifier.py` | features, dataset and augmentation, the network, training and ONNX export, inference |
| `sonar/sim.py` | physics echo simulator |
| `app/care/`, `app/sonar_care.py`, `scripts/sonar_care.sh` | Sonar Care: the service, its entry point, the one-command launcher |
| `app/web/` | the dashboard, settings and setup pages, the pitch slides |
| `scripts/run_best_mac.sh`, `scripts/mac_record_all.sh` | start the engine with a packaged model; record one person |
| `tests/`, `app/tests/` | the test suites |

---

*Legacy.* The project started on a Windows laptop with two microphones (an HP ENVY x360, WASAPI exclusive mode).
That path is still in the code (`py -m sonar.engine`, `--shared`, the two-microphone profiles) but it is not the
demo and is no longer documented here. Its instructions are in this file's git history before 15 November 2026.

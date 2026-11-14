# Handoff: Lance's MacBook (for Lance's Grok Bot)

You are helping Lance Li run the HackPrinceton "laptop sonar" project on his MacBook. Hacking ends about
8 AM Sun Nov 15, 2026 (Toronto time). The speaker plays an inaudible 17-21 kHz signal, the mic hears the
echoes, and the software reads hand gestures (to change slides), presence and breathing rate.
The demo, the data recording and the gesture training now all happen on this MacBook.

Full detail lives in `TRAINING.md` section 9 (MacBook), `TRAINING.md` (training), `README.md`,
`DATA_STRATEGY.md` (why and how to record) and `app/README.md` (slides + Care Monitor UI). Read them.

## Ground rules
- Commits: Lance commits as himself. Never add AI attribution or Co-Authored-By trailers.
  Do not commit `data/`, `.venv/` or `train_out/` (already gitignored). Push work to a branch, not `main`,
  unless Lance says otherwise.
- Recordings must be made on THIS MacBook. Data from the HP laptop or any other machine does not transfer.
- Keep Lance in the loop: every step that plays sound or needs people in front of the laptop needs him there.

## Step 1. Set up (once, about 5 min)
```bash
git clone https://github.com/mahimn01/sonar-hack.git && cd sonar-hack
brew install uv            # if uv is missing
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-train.txt
.venv/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"   # want True
```
Then have Lance do these by hand:
- Audio MIDI Setup (Applications > Utilities): built-in MacBook microphone and speakers at 48,000 Hz. If the
  mic offers more than 1 channel, pick it.
- Disconnect AirPods / Bluetooth audio. Set a fixed volume (fairly high) and keep it for every recording and the demo.
- Allow Terminal (or whatever app runs the commands) to use the microphone when macOS asks.

## Step 2. Self-test (1 min): the go/no-go gate
```bash
.venv/bin/python -m sonar.selftest --in-device MacBook --out-device MacBook --report selftest_mac.json
```
- PASS: note the "recommended profile" and use it everywhere (`SONAR_PROFILE=<name>` for recording,
  `--profile <name>` for training and the engine).
- FAIL: try `--blocksize 3840`, raise the volume, close other audio apps. On the HP laptop it only passed
  with `--profile clean --blocksize 3840`. This macOS audio path has NOT been tested on a real MacBook yet,
  so if it still fails, read `sonar/audio.py`, fix the macOS path, and report what you changed.
- Watch the "audible leak" line: if the signal is audible, lower the volume or use `--profile clean`.

## Step 3. Record (about 10 min per person)
```bash
SONAR_PROFILE=<recommended> scripts/mac_record_all.sh
```
It asks for the person's name and repetitions, then records scene (empty room, walk-ins), gestures (on-screen
prompts) and breathing (seated 0.5-0.8 m away, 2.5 min). Use 12-15 reps for the presenter and 5 for guests.
- More different people beats more repetitions (the judge is a stranger). Aim for 6+ people.
- Record at the demo spot: lid at the demo angle, charger as it will be, normal background noise is fine,
  but nobody else moving within about 1 m during gesture trials.
- After each trial the recorder prints the physics guess. The MacBook has one mic channel, so left/right
  swipes depend on the two speakers: if swipe guesses keep disagreeing with the prompt, stop and investigate
  before recording more.
- Recordings land in `data/<date>_<person>_<script>/`.

## Step 4. Train
On the MacBook (MPS):
```bash
scripts/train_4090.sh --device mps --size small --release models-release-mac --profile <recommended>
```
`--device cpu` if MPS fails. Or zip `data/` and train on the RTX 4090 (TRAINING.md 9.5, `--data mac
--release models-release-mac`). Read the "Honest score" and "Verdict" lines; if POOR, record more people.

## Step 5. Demo
```bash
.venv/bin/python -m sonar.engine --in-device MacBook --out-device MacBook --profile <recommended>
.venv/bin/python app/serve.py              # slides; --view monitor for the Care Monitor; F = fullscreen
```
Rehearse the UI without sound: `.venv/bin/python app/serve.py --mock`.
Breathing and presence are pure signal processing and work without a trained model: demo those first.

## Report back to Lance
After each step: self-test verdict and profile, number of sessions/people recorded, honest score and
verdict from training, and anything you had to change in the code.

# INTERFACES — how to talk to the sonar engine

This is the contract between the core package (`sonar/`) and everything built on
top of it (the UI in `app/`, the data tools in `datagen/`). Schema version: **1**.

- [1. Running the engine](#1-running-the-engine)
- [2. WebSocket event stream](#2-websocket-event-stream) (what the UI consumes)
- [3. Commands a client can send](#3-commands-a-client-can-send)
- [4. Python callback API](#4-python-callback-api)
- [5. Files on disk](#5-files-on-disk) (recorded sessions, datasets, models)
- [6. Python API for data tooling](#6-python-api-for-data-tooling)

---

## 1. Running the engine

```powershell
# Windows demo laptop, from the repo root
py -3.12 -m sonar.engine                      # live audio; uses models\gesture.onnx if it exists, else physics rules
py -3.12 -m sonar.engine --physics            # live audio, physics rules only (no trained model needed)
py -3.12 -m sonar.engine --backend sim        # NO hardware: a scripted simulated visitor, real pipeline
py -3.12 -m sonar.engine --backend file --file data\<session>\audio.wav     # replay a recording
```

The engine prints gestures, presence changes and breathing to the console and
serves events on **`ws://127.0.0.1:8765`** (`--host`, `--port` to change; any
number of clients may connect). Useful flags: `--profile`, `--min-confidence 0.6`,
`--swipe-hold 0.3`, `--no-spectrogram`, `--in-device`, `--out-device`,
`--list-devices`, `--min-peak-db`, `--no-range-gate`, `-v` (print status lines).

Start-up: the first ~5 s are calibration (`status.calibrating = true`). Hands
away from the laptop and nobody walking past during those seconds.

`--backend sim` loops a 62 s story: a visitor walks in (4-7 s), swipes right
(~11 s), pushes (~14 s), swipes left (~18 s), sits still breathing at 15/min
(19-50 s), leaves (50-53 s). Use it to rehearse the UI against the real event
stream. `--backend sim --scene hall` is the messy version: a seated person
breathing 15/min who fidgets, people walking past 1.5-3 m behind, a neighbour
2 m away and bursts of noise, at the noise level measured on the demo laptop
(`--scene hall_empty`: the same hall with nobody at the laptop).

To score a recording (or those simulated scenes) instead of watching it live:
`python -m sonar.replay <session folder>` prints presence and breathing second
by second and, per labelled part of the session, how often "moving" fired, how
often breathing restarted and the measured rate against the paced rate.

---

## 2. WebSocket event stream

Every message is one JSON object (text frame). Common fields:

| field | type | meaning |
|---|---|---|
| `type` | string | `hello`, `gesture`, `presence`, `breathing`, `status`, `spectrogram`, `error`, `ack`, `pong` |
| `t` | number | wall-clock time of the event, Unix seconds (float) |
| `seq` | integer | counter that increases by 1 per event (gaps mean lost messages); not on `hello`/`ack`/`pong` |

Unknown fields may be added later; ignore what you do not know.

### `hello` — sent once to each client when it connects

```json
{"type":"hello","schema":1,"version":"0.1.0","t":1794620725.2,
 "labels":["swipe_left","swipe_right","push","pull"],
 "presence_states":["empty","entering","present"],
 "breathing_states":["tracking","acquiring","moving","no_signal"],
 "frame_rate":50.0,"tones_hz":[18500.0,19000.0],"chirp_band_hz":[19500.0,21500.0],
 "doppler_range_hz":[-250.0,242.1875],"doppler_bins":64,
 "profile":"default","backend":"live","source":"live","classifier":"model"}
```

### `gesture` — one per recognised gesture

```json
{"type":"gesture","label":"swipe_right","confidence":0.93,"source":"model",
 "probs":{"none":0.01,"swipe_left":0.02,"swipe_right":0.93,"push":0.03,"pull":0.01},
 "duration_s":0.44,"peak_db":16.4,"t":1794620729.34,"seq":135}
```

| field | meaning |
|---|---|
| `label` | `swipe_left` \| `swipe_right` \| `push` \| `pull`. Never `none`. `swipe_right` = the hand moved to the user's right. |
| `confidence` | 0..1. Already at least `--min-confidence` (0.6 by default). |
| `source` | `model` (trained CNN) or `physics` (training-free rules). |
| `probs` | score of every class. |
| `duration_s` | length of the movement burst. |
| `peak_db` | how far the movement stood above the noise floor (dB). Bigger = clearer. |

The engine has **already** de-duplicated: confidence threshold, 0.4 s dead time,
the opposite gesture is ignored for 1.2 s (the hand coming back), swipes are
held 0.3 s to tell the wind-up from the real stroke, bursts caused by somebody
walking in or walking past are dropped. A UI can act on every `gesture` event
directly. Latency: a swipe is reported about 0.5 s after the hand stops
(0.2 s to be sure the movement ended + 0.3 s hold); push/pull about 0.2 s.

### `presence` — on every change, and once per second

```json
{"type":"presence","state":"present","confidence":0.86,"range_m":0.71,
 "moving":false,"motionless_s":0.0,"changed":true,"t":1794620728.2,"seq":87}
```

| field | meaning |
|---|---|
| `state` | `empty` (nobody within ~1.3 m in front of the laptop) \| `entering` (somebody is walking up) \| `present` (somebody is there). People further back in the room do not count. |
| `confidence` | 0..1, smoothed over ~1.5 s. |
| `range_m` | distance of the person from the laptop in metres, `null` when empty. Good to about ±0.1 m; a 2 s median, so a hand crossing in front does not move it. |
| `moving` | gross body movement of THE PERSON IN FRONT right now (leaning, reaching, gesturing, walking up). Other people walking past further back, and bursts of noise, do not set it. Slow movement (under ~5 cm/s) does not either. |
| `motionless_s` | while `present`: seconds since ANY movement was seen, breathing included. Normally near 0. If it keeps counting, the person is still there (they did not walk away) but nothing moves at all. A pulse or a little sway keeps it at 0, so for "breathing stopped" use `breathing.apnea` / `breathing.stopped_s`. After 40 s the state falls back to `empty`. |
| `changed` | `true` only on the message where `state` changed. |

Typical timing: `entering` within ~0.5 s of someone walking into range,
`present` ~1 s after they stop, `empty` ~5 s after they have walked away. A hand
reaching in from the side while the state is `empty` does not make anybody
"enter" (so it cannot veto its own gesture).

### `breathing` — about once per second (not while calibrating)

```json
{"type":"breathing","bpm":15.2,"confidence":0.97,"state":"tracking",
 "waveform":[-0.42,-0.31, "... up to 300 values ..."],"fs":10.0,
 "range_m":0.71,"amplitude_mm":5.3,"window_s":30.0,
 "apnea":false,"motionless_s":0.0,"stopped_s":0.0,"held":false,"t":1794620760.1,"seq":1410}
```

| field | meaning |
|---|---|
| `bpm` | breaths per minute, or `null` when there is no trustworthy rate (show "--"). |
| `confidence` | 0..1. `bpm` is only non-null from 0.45 up and after 3 consecutive estimates agree. |
| `state` | `tracking` (rate is valid) \| `acquiring` (a chest-like movement is visible, still measuring: the first ~10-15 s, a couple of seconds after the person moved, or while the breathing is irregular, e.g. talking) \| `moving` (the person has been moving for 1.5 s or more; no rate while it lasts) \| `no_signal` (nothing breathing-like in range, or the chest that was being followed has stopped moving). |
| `waveform` | chest displacement, newest sample last, normalised to -1..1, positive = chest towards the laptop (breathing in). Up to 30 s. May be empty. Present from ~4 s of stillness, before `bpm` is. Also sent while a breath is held (it then goes flat). |
| `fs` | sample rate of `waveform` in Hz (10). |
| `range_m` | distance of the chest. |
| `amplitude_mm` | peak-to-peak chest movement in millimetres (approximate for shallow breathing). |
| `window_s` | seconds of data behind this estimate. |
| `held` | `true` when `bpm` repeats the last good rate while a brief disturbance passes (at most 3 s). A UI may dim the number. |
| `stopped_s` | seconds since a breathing rhythm that WAS being tracked stopped and the chest has been still since; 0 while breathing. A pulse or a little sway does not reset it; talking or moving does. |
| `apnea` | `true` when somebody is `present` and either nothing at all has moved for 5 s, or `stopped_s` has reached 6 s (about 10 s after the last breath). |
| `motionless_s` | same counter as in `presence`. |

What a fidget, a walker or a burst of noise does: a brief disturbance at the
chest's distance (under 1.5 s: a hand moves, a head turns, a click) is skipped
and the rate carries on; people walking past further back are ignored; a longer
movement shows `moving`, and the rate is back 2-3 s after it ends (about 10 s if
the person moved for more than 3 s, because the measurement then starts again).

Accuracy to expect: within 1 breath/min for a **seated** person at 0.5-0.8 m.
A free-standing person sways in the same frequency band; expect `acquiring` or a
less reliable rate.

### `status` — once per second

```json
{"type":"status","snr_db":43.5,"carrier_hz":18500.0,"fps":50.0,"calibrating":false,
 "locked":true,"drift_ppm":0.0,"clipped":false,"dropped":0,"resyncs":0,
 "activity_db":0.6,"classifier":"model","backend":"live","source":"live",
 "device":"Microphone Array (Intel...)","latency_ms":42.3,"min_peak_db":3.0,
 "profile":"default","tones_hz":[18500.0,19000.0],"chirp_band_hz":[19500.0,21500.0],
 "last_decision":{"label":"none","confidence":0.71,"peak_db":4.2},
 "version":"0.1.0","t":1794620731.0,"seq":201}
```

| field | meaning |
|---|---|
| `snr_db` | level of the weakest pilot tone above the Doppler noise floor: the headroom echoes live in. Healthy: 40+. Below ~25 gestures get unreliable. |
| `carrier_hz` | first pilot tone. `tones_hz`, `chirp_band_hz` give the whole signal. |
| `fps` | frames processed per second (50 when keeping up). |
| `calibrating` | `true` during start-up and after a `calibrate` command. |
| `locked` | the chirp receiver is aligned to the speaker signal (needed for presence and breathing). |
| `drift_ppm` | measured clock offset between speaker and microphone (parts per million; 0 expected on the demo laptop). |
| `clipped` | the microphone hit full scale in the last second (lower the volume). |
| `dropped`, `resyncs` | audio dropouts noticed / times the receiver had to re-align. |
| `activity_db` | current hand-movement energy above the noise floor (a live "is it seeing me" meter; a gesture needs 3+). Not range-gated: in a busy room it also shows other people. |
| `noise_db` | how much louder the room's ultrasonic noise has been lately than during calibration (0 = unchanged). If it stays above ~6, send `calibrate`. |
| `last_decision` | the classifier's verdict on the latest movement burst, including rejected ones (`none`, `blocked:entering`, `blocked:far`, `blocked:weak`). Useful for a debug HUD. |
| `source` | `live`, `sim` or `file`. Show a "simulated" badge unless it is `live`. |

### `spectrogram` — 10 per second (optional, `--no-spectrogram` turns it off)

```json
{"type":"spectrogram","rows":[[0.0,0.02, "... 64 values ..."], "... 5 rows ..."],
 "n_bins":64,"bins_hz":[-250.0,-242.19, "...", 242.19],"frame_s":0.02,"t":...,"seq":...}
```

`rows` holds the 5 newest time slices, oldest first (20 ms each). Each row has 64
values in 0..1: movement energy per Doppler bin, from `bins_hz[0]` = -250 Hz
(moving away at ~2.3 m/s) through 0 Hz (index 32) to +242 Hz (approaching).
0 = at the noise floor, 1 = 39 dB above it. Draw time on one axis and append rows
as they arrive.

### `error`

`{"type":"error","message":"..."}` — the audio device failed or a bad command was sent.

---

## 3. Commands a client can send

Send JSON text. Replies go only to the sender.

| send | reply | effect |
|---|---|---|
| `{"cmd":"ping"}` | `{"type":"pong"}` | connection check |
| `{"cmd":"hello"}` | the `hello` message | re-read static info |
| `{"cmd":"calibrate"}` | `{"type":"ack","cmd":"calibrate"}` | learn the noise floors again (takes ~5 s; hands away, nobody walking past) |
| `{"cmd":"set_min_confidence","value":0.7}` | `{"type":"ack",...}` | change the gesture confidence threshold |

---

## 4. Python callback API

```python
from sonar.audio import make_backend
from sonar.classifier import load_classifier
from sonar.config import get_profile
from sonar.engine import Engine

cfg = get_profile("default")
engine = Engine(cfg, make_backend("live", cfg), profile="default")     # or make_backend("sim", cfg)
engine.set_classifier(load_classifier("models/gesture.onnx", engine.cfg))   # physics rules if the file is missing

def on_event(event: dict) -> None:          # same dicts as the WebSocket messages
    if event["type"] == "gesture":
        print(event["label"], event["confidence"])

engine.on_event(on_event)                   # called from the engine thread: return quickly
engine.start()                              # opens audio, processes in a background thread
...
engine.stop()
```

`engine.step(samples)` processes one block synchronously and returns the events
(used by the tests and for offline replay). `engine.recalibrate()` is thread-safe.

---

## 5. Files on disk

### Recorded session: `data/<YYYYmmdd-HHMMSS>_<person>_<script>/`

- `audio.wav` — raw microphones, 48 kHz, 24-bit PCM, one channel per mic,
  continuous for the whole session.
- `session.json`:

```json
{"format":1,"session_id":"20261114-101500_alice_gestures","person":"alice","script":"gestures",
 "notes":"seat 60 cm, volume 70 %","created":"2026-11-14T10:15:00","profile":"default",
 "synthetic":false,"audio":"audio.wav","fs":48000,"n_mics":2,"n_samples":10752000,
 "config":{"...":"every SonarConfig field"},"backend":{"...":"device names"},
 "dropouts":[],"peak":0.0021,
 "events":[
   {"label":"calibration","kind":"calibration","start":0,"end":288000},
   {"label":"swipe_left","kind":"gesture","start":432000,"end":552000,"rep":1},
   {"label":"breathing","kind":"scene","start":900000,"end":2820000,"bpm":15.0}
 ]}
```

`start`/`end` are **sample indices** into `audio.wav`. For a gesture they bound
the prompt window ("GO" until 2.5 s later); the movement is somewhere inside.
Labels: gestures `swipe_left`, `swipe_right`, `push`, `pull`; scenes `empty`,
`present_still`, `walk_in`, `walk_out`, `distractor`, `breathing` (with `bpm`).
The first ~2 s of every session must be free of hand movement (noise-floor
calibration). Synthetic sessions written in this format (set `"synthetic": true`)
go through `python -m sonar.dataset` / `sonar.train --data` unchanged.

### Dataset: `.npz` (`python -m sonar.dataset data/ --out dataset.npz`)

| key | shape / type | meaning |
|---|---|---|
| `X` | `[N, C, 64, 96]` float16 | feature pictures: `C` channels, 64 Doppler bins, 80 frames + 8 frames of slack on each side |
| `y` | `[N]` int64 | index into `labels` |
| `labels` | `[5]` str | `none, swipe_left, swipe_right, push, pull` |
| `group` | `[N]` str | session id (validation splits hold out whole groups) |
| `person` | `[N]` str | |
| `source` | `[N]` str | `gesture`, `burst`, `idle`, `aux`, or `synthetic` (**synthetic examples are never used for validation**) |
| `peak_db` | `[N]` float32 | strength of the burst |
| `pad` | int | 8 |
| `config` | str | JSON of the `SonarConfig` |

Channel layout of a picture (default profile, `C` = 5):
`0` tone L/mic 0, `1` tone L/mic 1, `2` tone R/mic 0, `3` tone R/mic 1 — Doppler
power, 0..1 = 3..39 dB above the noise floor; `4` "lateral": -1..1, phase-turn
rate between the two microphones, negative = moving right, shown only where
there is echo energy. `python -m sonar.train --dataset a.npz b.npz` accepts any
number of such files next to `--data`.

### Model: `models/gesture.onnx` + `models/gesture.json`

ONNX input `input` `[batch, C, 64, 80]` float32, output `logits` `[batch, K]`.
`gesture.json`: `labels`, `temperature` (divide logits by it before softmax),
`input_shape`, `config`, `heuristic` (cue weights for the physics rules),
`gesture_peak_db_p10`, `metrics`.

---

## 6. Python API for data tooling

Stable entry points (everything else may still move):

```python
from sonar.config import SonarConfig, get_profile, GESTURE_LABELS, SCENE_LABELS
from sonar.tx import make_tx_frame, make_reference      # one 20 ms period [960, n_out]; loop it
from sonar.features import stream_features, window_from_stream, clip_to_window
from sonar.dataset import pack, Example, PAD, Augmenter
from sonar.pipeline import Pipeline                     # the streaming chain the engine runs

cfg = get_profile("default")
tx = make_tx_frame(cfg)                                 # what the speakers play (float32, peak < 0.9)

# microphone audio [n, n_mics] -> model pictures, exactly as the live engine cuts them
stream = stream_features(rx, cfg)                       # needs ~2 s without hand movement at the start
for seg, nb in zip(stream.segments, stream.not_before):
    picture = window_from_stream(stream, seg.decided, nb, pad=PAD)     # [C, 64, 96]

picture, seg = clip_to_window(rx_clip, cfg, pad=PAD)    # one clip, one gesture -> its picture (or None)
```

Signal facts a simulator must match: 48 kHz; everything repeats every 960
samples; pilot tones at `cfg.tone_hz` on speaker channels `cfg.tone_channels`
(18.5 kHz left, 19.0 kHz right); chirp lines every 50 Hz from `cfg.chirp_lo_hz`
to `cfg.chirp_hi_hz` on `cfg.chirp_channel`. Channel 0 = the user's LEFT speaker
/ microphone. Coordinates used by `sonar/sim.py`: x to the user's right, y from
the screen towards the user, z up, origin at the middle of the hinge.

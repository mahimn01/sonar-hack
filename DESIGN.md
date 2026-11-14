# DESIGN — camera-free laptop sonar (gestures, presence, breathing)

Phase 1 design, HackPrinceton (Healthcare track). Written 2026-11-13.
Companion documents: `README.md` (how to run), `INTERFACES.md` (event schema and
file formats), `DATA_STRATEGY.md` (synthetic data, by the data engineer).

**One-paragraph summary.** The speakers play a signal just above hearing that
repeats every 20 ms: one steady **pilot tone per speaker channel** (18.5 kHz
left, 19.0 kHz right) and one **chirp** (a sweep, 19.5-21.5 kHz). The two
microphones record the echoes. Hand movement shifts the pitch of the tones
(Doppler effect), which gives a small picture a CNN classifies into swipe
left/right, push, pull. The chirp tells us *how far away* each echo comes from,
which lets us look only at the person in front of the laptop: slow changes at
their distance mean "present", and the tiny back-and-forth of the chest shows up
as a turning echo phase from which we read the breathing rate. Presence and
breathing are plain signal processing with no training.

Contents: [1 What the hardware gives us](#1-what-the-hardware-gives-us) ·
[2 Prior work](#2-prior-work) · [3 Signal choice](#3-signal-choice) ·
[4 Transmit signal](#4-transmit-signal) · [5 Processing chain](#5-processing-chain) ·
[6 Features and model](#6-features-and-model) · [7 Data collection](#7-data-collection-protocol) ·
[8 Demo robustness](#8-making-the-demo-robust) · [9 Risks and fallbacks](#9-risks-and-fallbacks) ·
[10 Stretch goals](#10-stretch-goals) · [11 What is verified](#11-what-is-verified-and-what-is-not)

---

## 1. What the hardware gives us

Measured tonight on the demo laptop (HP ENVY x360 15-ed1xxx, Windows 11):

| fact | consequence |
|---|---|
| Shared-mode WASAPI low-passes the microphone: 15-22 kHz arrives at the digital noise floor | **WASAPI exclusive mode on both devices is mandatory** (`sd.WasapiSettings(exclusive=True)`) |
| Exclusive mode, 48 kHz, 36 % volume, 0.5-amplitude tone: -79 dB at 17/18 kHz (60+ dB SNR), -89 dB at 19/20 kHz (~55 dB) | Plenty of signal; the response drops ~10 dB between 18 and 19 kHz; volume can still go up |
| 96 kHz rejected | 48 kHz, so everything must fit below 24 kHz |
| Speaker = Realtek codec, microphones = Intel Smart Sound digital mics: two different Windows devices | Playback and recording are two streams with an unknown, different-every-start offset |

Found during the research for this document (not yet measured on the device):

- **The laptop has three speakers**, not two: two bottom-firing "front" speakers
  and one "rear" speaker at the hinge (HP service guide L84426-001). Which driver
  plays which channel at 19 kHz is unknown. If the hinge speaker plays a mix of
  both channels, the left and right channels come out of the same place and the
  two-speaker direction cue vanishes. The self-test measures this, and the design
  carries a second, independent direction cue from the two microphones.
- **PortAudio never reports dropped buffers in WASAPI mode** (the status flags
  are always 0 in v19.7.0, the build inside the `sounddevice` wheel). A drop
  silently shifts playback against recording. The receiver therefore notices
  misalignment by itself and re-locks.
- A single full-duplex PortAudio stream across two WASAPI endpoints is forced
  into a 2 ms polling loop; **two separate event-driven streams** are the better
  path. We open them separately and fall back to duplex.
- Both devices take their sample clock from the same Intel audio controller, so
  no clock drift is expected. We still measure and correct drift, because being
  wrong here would silently ruin the breathing measurement.

## 2. Prior work

Verified against the papers by a literature pass (links checked 2026-11-13).

| System | Signal | What it showed | What we take from it |
|---|---|---|---|
| **SoundWave** (CHI 2012) [1] | one 18-22 kHz tone, FFT, Doppler | laptop speaker+mic gestures, 87-100 %, no ML; the hand's Doppler echo is often >10 % of the tone | Doppler on a steady tone is enough for gestures and survives noise |
| **Tarzia et al.** (UbiComp 2009) [2] | 20 kHz tone, echo variance | presence detection on stock laptops | presence = "the echo changes"; prior art for our story |
| **LLAP** (MobiCom 2016) [3] | 16 tones, I/Q phase, static-vector removal | 3.5 mm hand tracking | phase carries millimetres; the static echo must be removed first |
| **FingerIO** (CHI 2016) [4], **Strata** (MobiSys 2017) [5] | OFDM / coded frames -> channel impulse response (CIR) | sub-cm finger tracking by watching one delay tap | estimate the CIR every frame, track the phase of the tap that moves |
| **ApneaApp** (MobiSys 2015) [6], **opioid overdose** (Sci. Transl. Med. 2019) [7] | FMCW chirps 18-20/22 kHz, power in the range bin | breathing and apnea at <=1 m on phones, clinically validated | chirps give range gating, which is what makes breathing robust; they used amplitude, we use phase |
| **C-FMCW** (IMWUT 2018) [8] | correlation-based FMCW | <0.35 breaths/min; corrects speaker/mic clock offset | correlation receiver; clock offset must be handled |
| **SonarBeat** (2017) [9], **pure-tone breathing** (Sci. Rep. 2026) [10] | one tone, I/Q phase | 0.2-0.3 breaths/min while people sleep | phase read-out works; a quality gate that discards bad epochs is essential (they keep 65 %) |
| **BreathJunior** (MobiCom 2019) [11], **smart-speaker heart rhythm** (Commun. Biol. 2021) [12] | FMCW, multi-mic combining | infant breathing, heartbeats | combine microphones/taps before reading the phase |
| **UltraGesture** (2018) [13], **RobuCIR** (INFOCOM 2020) [14] | CIR images -> CNN (+LSTM), augmentation | 12-15 gestures at 97-98 % | a small CNN on a time-frequency picture, heavy augmentation |
| **AudioGest** (UbiComp 2016) [15] | Doppler, training-free rules | 6 gestures up to 96 % | physics rules are a viable fallback without any training |
| **"Experience" paper** (MobiCom 2022) [16], **BackDoor** [17], NSDI 2018 [18] | hardware study | amplifier nonlinearity makes multi-tone signals leak audible sound; chirps leak least; splitting frequencies across speakers avoids it; 100 % volume is worse | keep one component per speaker where possible, stay below full volume, measure the leak |
| **WindDancer** (IMWUT 2025) [19] | | airflow makes the speed of sound flicker and degrades sensing | fans/HVAC are a real noise source for the slow detectors |

Commercial proof that this is not a toy: Elliptic Labs ships ultrasonic presence
detection using the built-in laptop speaker and microphone on Lenovo ThinkPad
and Yoga models [20]; Google Nest [21] and Amazon Echo [22] use ultrasound for
presence. None of them does breathing or gestures on a laptop.

Human factors used below: hearing thresholds for young adults are about 64-83 dB
SPL at 18 kHz, ~94 dB at 19 kHz, ~90-95 dB at 20 kHz (Ashihara 2007 [23], Lee
2012 [24]); 18 kHz is audible to some under-25s at laptop volume, 19 kHz and up
to almost nobody. Quiet breathing moves the chest wall 3-5 mm (up to ~12 mm when
deep) at 0.2-0.34 Hz [25]. A free-standing adult sways several millimetres with
most of its power at 0.11-0.24 Hz [26]: the same band as breathing.

## 3. Signal choice

| Option | Gestures | Breathing / presence | Verdict |
|---|---|---|---|
| **A. Pilot tones only** (SoundWave/LLAP) | Excellent: unlimited Doppler range, trivial processing, no timing needed | Weak: no distance information. The direct speaker->mic path is 40-60 dB stronger than a chest echo and sits in the *same* measurement, so any wobble of the laptop (fan, typing, hinge) swamps the breathing. Everybody in the hall who moves is in the measurement. | Not enough alone |
| **B. Chirps only** (FMCW) | Poor fit: a periodic waveform cannot measure speed and distance unambiguously at once. Unambiguous range x speed span = c·λ/8 ≈ 0.77 m²/s. Range to 3.4 m leaves ±0.22 m/s; hands move at 0.5-2 m/s. | Excellent: range gating isolates the chest from the direct path and from other people | Not enough alone |
| **C. Hybrid: tones for Doppler + one chirp for range** | as A | as B | **Chosen** |

Why the hybrid is safe to build in a weekend: both parts are lines of one
periodic waveform, so they share one transmit buffer and one frame clock, yet
the two receive paths are independent. The tone path needs **no timing
alignment at all**, so the headline gesture demo keeps working even if the
chirp path has trouble.

Two things the hybrid costs, and how they are handled:

1. *Audible intermodulation.* A tone and the chirp in the same speaker channel
   can beat against each other in a nonlinear amplifier and leak sound at their
   difference frequency (0.5-3 kHz) [16-18]. The chirp alone and a tone alone do
   not. With two channels and three components one channel must carry two. The
   self-test measures the leak; if it is audible, profile `clean` (left = tone,
   right = chirp, zero intermodulation) is the fallback.
2. *Bandwidth.* Tones need guard bands, leaving 2 kHz for the chirp: range
   resolution c/2B = 8.4 cm. Enough to separate "hand zone", "person" and "the
   hall behind".

## 4. Transmit signal

Everything is an exact multiple of 50 Hz, so one 20 ms period (960 samples at
48 kHz) loops forever with no seam, no clicks and exactly zero energy below the
band (verified: < -110 dB, i.e. float32 rounding).

| Component | Frequency | Speaker channel | Digital peak | Purpose |
|---|---|---|---|---|
| pilot tone L | 18.5 kHz | 0 (left) | 0.35 | gesture Doppler + two-speaker direction cue |
| pilot tone R | 19.0 kHz | 1 (right) | 0.35 | same |
| chirp | 41 lines, 19.5-21.5 kHz, every 50 Hz | 0 (left) | 0.50 | range profile: presence, breathing, range gate |

- **Tones 500 Hz apart**: each tone owns ±250 Hz of Doppler (±2.3 m/s), more
  than any gesture, and their echoes never overlap.
- **Chirp built in the frequency domain**: equal-strength lines with quadratic
  ("Schroeder") phases, `phase_m = -π m² / K`. In time that is a smooth upward
  sweep with an almost constant envelope (crest factor 1.86; a sine is 1.41),
  with a perfectly flat spectrum. No window is needed at transmit, so no
  amplitude modulation and no clicks.
- **Start/stop**: 60 ms raised-cosine fades.
- **Why 18.5 kHz and up**: inaudible to almost everyone (section 2) while
  staying where the speaker still has output. Profile `strong` moves everything
  500 Hz down (+~5 dB on this laptop, audible to some young listeners);
  `low` moves it 1.5 kHz down as an emergency fallback.

Derived numbers: 50 frames/s; Doppler of a hand at speed v is about
`2 v f / c` = 109 Hz per m/s; chirp wavelength λ = 16.7 mm, so 1 mm of chest
movement turns the echo phase by `4π/λ` = 0.75 rad (43°); unambiguous range
3.4 m.

## 5. Processing chain

```
mic audio ──► high-pass FIR ─┬─► per tone: shift to 0 Hz, decimate to 2 kHz ─► 5 Hz high-pass ─► STFT ─► Doppler column
 (2 ch)        (kills speech) │                                                              └─► mic-pair phase turn ("lateral")
                              └─► frame sync ─► FFT ─► ÷ chirp ─► drift fix ─► IFFT ─► echo profile h[mic, tap] (50 per s)
Doppler columns ─► noise-floor normalise ─► burst segmenter ─► 1.6 s picture ─► CNN / physics rules ─► gesture
echo profile    ─► presence state machine          echo profile ─► breathing estimator
```

All blocks are streaming and accept any block size (`sonar/dsp.py`,
`sonar/frontend.py`); the same `Pipeline` class runs live, on recordings, in the
dataset builder and in the tests.

### 5.1 Prefilter
125-tap linear-phase FIR high-pass (Kaiser, 100 dB), passband from 17.65 kHz.
Hall noise (speech, music) is 30-40 dB louder than our own signal at the
microphone; without this it would leak into every later FFT.

### 5.2 Tone path -> Doppler column (`ToneDemod`, `DopplerSTFT`)
For tone frequency f, output sample m is taken at input index n = 24 m:

`y[m] = e^{-j2πfn/fs} · Σ_i x[n-i] · h[i] · e^{+j2πfi/fs}`

(band-pass around the tone, shift to 0 Hz, decimate 48 kHz -> 2 kHz; 197-tap
Kaiser low-pass, 90 dB). The direct path and every static echo land exactly on
0 Hz and are removed by a 5 Hz Butterworth high-pass; a reflector moving at
radial speed v lands on +2vf/c (towards) or -2vf/c (away). A 64 ms Hann window
with a 20 ms hop and a 256-point FFT gives one column of 64 bins per frame, from
-250 Hz to +242 Hz (7.8 Hz per bin, 23 Hz resolution), for each of the 4
(tone, mic) pairs, in dB re full scale.

### 5.3 Two-microphone direction signal ("lateral")
With X0, X1 the complex spectra of the two microphones for the same tone,
`R = X0·conj(X1)` holds their phase difference and
`L = Σ_tones R[now]·conj(R[previous frame])` holds how fast it turns:

`angle(L) = 2π · 20 ms · (Doppler at mic 0 − Doppler at mic 1)`

A reflector moving sideways approaches one microphone and recedes from the
other, so the sign of `angle(L)` is the swipe direction (positive = towards
mic 0 = the user's left). The speaker path cancels, so this works whatever the
speakers do. Expected size for mics 8 cm apart and a hand 30 cm away: ~13 Hz
per m/s, far below the STFT resolution but easy for a phase measurement.

### 5.4 Chirp path -> echo profile (`ChirpReceiver`)
The chirp is a comb of K = 41 lines at FFT bins k with unit phasors U[k] and
amplitude a. For a received frame r[n] of N = 960 samples:

```
H[k]  = FFT(r)[k] · conj(U[k]) / (a·N/2)                  channel at line k
Hc[k] = H[k] · e^{+j2πkδ/N}                               undo clock slip δ (samples)
h[n]  = (1/ΣW) · Σ_m W[m] · Hc[k_lo+m] · e^{+j2πmn/M}     M = 128, Hann window W
```

`|h[n]|` is the amplitude gain of the echo with travel time n·7.5 samples (one
tap = 5.4 cm of path = 2.7 cm of range); its phase turns by `-4π d/λ` when the
reflector moves d. Because the transmit signal is exactly periodic, the plain
FFT of one period has no leakage from our own signal, and the Hann window keeps
the sidelobes of the huge direct path below the echoes at chest range.

- **Frame sync (no reliance on driver timing).** Average 16 frames, compute the
  full-rate correlation envelope, take the earliest strong peak as the direct
  speaker->microphone sound and shift the frame grid so it sits on tap 4. Works
  for any latency and any block size (tested with 12 combinations).
- **Clock drift.** `ψ = angle(Σ_k Hc[k]·conj(Href[k]))` is the phase of the
  direct path against a slowly updated reference. A drift of ε ppm makes it
  ramp. The initial rate comes from a line fit over the first 50 frames; a
  second-order loop (0.05 Hz bandwidth) then tracks it; whole-sample slips move
  the frame grid and are absorbed by δ. Tested at ±25/40 ppm: rate recovered to
  <1 ppm, direct-path phase steady to <3°.
- **Lost lock.** `coherence = |Σ Hc·conj(Href)| / (|Hc||Href|)` under 0.5 for
  10 frames (a dropped buffer, a lid slammed) -> sync again. Tested by deleting
  437 samples mid-stream.

### 5.5 Gestures: normalise, segment, classify (`gestures.py`)
1. **Noise floor.** Per channel and Doppler bin, the median over the first
   ~1.2 s, then slow tracking while nothing moves. Feature =
   `clip((dB − floor − 3) / 36, 0, 1)`: "how far above the floor", independent
   of volume.
2. **Burst segmenter.** Activity = 10·log10 of the mean in-band (20-235 Hz)
   power over the floor. Open at +1.5 dB (or idle mean + 4σ) for 2 frames, close
   after 0.2 s below +0.7 dB or at 1.4 s, discard bursts under 0.12 s or under
   +3 dB.
3. **Click rejection.** A click is flat across Doppler and equal on both signs;
   a hand is a narrow one-sided ridge. Click frames and the 0.2 s after them
   cannot open or sustain a burst.
4. **Picture.** When a burst closes, the last 80 frames (1.6 s) of all channels
   form the model input. Frames before the previous burst ended are blanked, so
   no stroke is classified twice.
5. **Classifier.** The CNN (section 6), or the physics rules:
   - two-speaker cue = Doppler centroid of tone L minus tone R (a hand moving
     right moves away from the left speaker and towards the right one);
   - two-microphone cue = mean `angle(L)`;
   - direction = w_s·cue_s/15 Hz + w_m·cue_m/5 Hz, measured on the strongest
     0.5 s of the burst; sideways if clear and not small next to the
     towards/away Doppler -> swipe left/right;
   - otherwise push (fast approach, then away) or pull (slow approach, fast
     retreat) from the order and speed of the positive and negative Doppler.
6. **Decision rules** (`engine.GestureDecider`): confidence >= 0.6; 0.4 s dead
   time; opposite gesture ignored for 1.2 s (the hand coming back); swipes held
   0.3 s so that "wind-up then stroke" reports only the stroke; bursts while
   somebody walks in, bursts whose movement is beyond 0.6 m (range gate from
   the chirp) and bursts far weaker than the recorded gestures are dropped.

### 5.6 Presence (`presence.py`)
Per range tap: **gross movement** = power of the echo profile after a 7 Hz
high-pass along time (breathing changes it at <3 Hz and is rejected by >25 dB
however strong the chest echo); **micro movement** = power of (0.25 s average −
4 s average). Both are compared with floors learned at start-up; only taps
within 2 m count. Pilot-tone Doppler energy also counts as gross movement.

```
empty ──gross movement 0.3 s──► entering ──still 1 s──► present
present ──walked, then nothing for 5 s──► empty
present ──nothing moves, no walking before──► stays present, motionless_s counts (alarm signal), empty after 40 s
```

A confident breathing detection counts as micro movement, so a quiet, still
person stays "present".

### 5.7 Breathing (`breathing.py`)
1. Average 5 frames (50 -> 10 per second), keep the last 30 s of the taps
   between 0.25 and 1.6 m. Any gross movement restarts the buffer.
2. Zero-phase low-pass the complex taps (1.3 Hz and 3.0 Hz versions; the better
   result wins), subtract each tap's time average.
3. Find where something still moves: excess variance over sensor noise per tap;
   up to 3 candidate ranges.
4. **Combine before reading the phase.** The chest appears in ~5 neighbouring
   taps on both microphones, all turning together. The first component of an
   SVD of that 10-column block is their best-weights sum: one complex signal
   g(t) with ~7-10 dB better SNR than any single tap.
5. **Phase.** g(t) = centre + arc. Long arc (>~150°, deep breathing): Taubin
   circle fit, subtract the centre, unwrap the angle. Short arc (shallow
   breathing or weak echo, where a circle fit is unreliable): project onto the
   arc's direction. Displacement `d = phase · λ/(4π)`.
6. **Rate.** Band-pass 0.08-0.8 Hz; Hann-windowed spectrum tilted by f to
   flatten sway and drift; only true local maxima in 0.1-0.6 Hz;
   score(f) = P(f) + 0.4·P(2f) (the 2nd harmonic protects against rate
   doubling); parabolic peak refinement.
7. **Confidence** = peak share of the band power × window length (>=3 breaths)
   × SNR × plausible amplitude; a rate is reported only at >= 0.45 **and** when
   3 consecutive estimates agree within 2 breaths/min. Otherwise `bpm` is null.
   The estimator also checks that the echo is still moving *now* (last 6 s), so
   a held breath or a person who left is not "measured" from old data.

## 6. Features and model

**Input** `[C, 64, 80]` float32: C = 5 for the default profile (4 Doppler power
pictures: tone L/R × mic 0/1, values 0..1; plus the lateral picture, -1..1,
shown only where there is echo energy); 64 Doppler bins; 80 frames = 1.6 s.

**Why this representation.** The classes differ in ways physics makes obvious:

| gesture | tone L | tone R | lateral |
|---|---|---|---|
| swipe right | Doppler down (leaving the left speaker) | up | negative |
| swipe left | up | down | positive |
| push | up, then down (both tones alike) | same | ~0 |
| pull | slow up, then fast down | same | ~0 |

So a small network, little data and even hand-written rules can separate them.

**Model** (`model.py`): 4 × [3×3 conv, batch-norm, ReLU, 2×2 max-pool] with 24,
48, 96, 96 channels, global average pool, dropout 0.3, linear layer: 137 k
weights, 1.4 ms per inference on one CPU core (ONNX Runtime). Small on purpose:
a few hundred real examples, a CPU-only laptop.

**Training** (`train.py`): AdamW, one-cycle learning rate, label smoothing 0.05,
class-balanced batches, 60 epochs (seconds on MPS/CUDA). Validation holds out
**whole sessions** (or a whole person with `--val-person`); synthetic examples
are never used for validation. Reports accuracy, per-class precision/recall, the
confusion matrix, and the same "as deployed" (after the 0.6 confidence
threshold: how many gestures are recognised, how many non-gestures would
trigger). Temperature scaling (never below 1) makes the confidence honest.
Export to ONNX with a parity check against PyTorch.

**Augmentation** (`dataset.Augmenter`), each a real-world variation: time shift
(±0.16 s); **left/right mirror** (swap tones, swap microphones, negate lateral,
swap the swipe labels: doubles the swipe data); **played backwards** (swipes
only); faster/slower ±18 % with the matching Doppler scaling; echo level −9..+6
dB; fresh noise-floor speckle; a faint broadband click; a small time or
frequency mask.

**Labels come from the audio itself.** Sessions are one continuous recording
plus prompt windows; the dataset builder runs the engine's own pipeline over
the recording and takes the strongest burst inside each prompt window as the
example, cut exactly as the engine will cut it live. Every other burst
(reaching, typing, walking) becomes a "none" example.

## 7. Data collection protocol

Per person, three short sessions (run from the repo root; details in README):

| script | command | what happens | time |
|---|---|---|---|
| gestures | `py -3.12 -m sonar.record --person NAME --script gestures --reps 12` | 48 prompted trials (4 gestures × 12, shuffled); live "movement detected / physics guess" after each | ~4.5 min |
| scene | `... --script scene` | empty room, walk in, sit still, type/reach/drink, stand still, someone walks past behind, walk out | ~3.5 min |
| breathing | `... --script breathing` | paced breathing at 12, 15, 20 per min, 40 s each (ground truth for the estimator) | ~2.5 min |

**Target for a solid model: 4 people × 3 gesture sessions = 12 sessions = 576
gestures (144 per class), plus one scene and one breathing session per person:
about 75 minutes of recording, ~2 hours with changeovers.** Minimum to start:
2 people × 2 gesture sessions (192 gestures, ~20 min) plus one scene session.

Rules that make the data worth having:
- Vary between sessions: seat distance (45/60/75 cm), hand height, speed
  (normal/fast/lazy), left and right hand, laptop position on the table.
- Record in the hall, with its noise and people. Record one session at the
  actual demo table on demo morning.
- Start each gesture on "GO", then let the hand **drop** back (no swipe back).
- Keep the Windows volume fixed for all sessions (write it in `--notes`).
- At least one person who is **not** in the training data records a session for
  the honest score (`--val-person`).
- After each session read the last line: "physics guess agreed with the prompt
  in X/Y". If that is high, the hardware is doing what the design assumes.

## 8. Making the demo robust

- **Self-test first** (`python -m sonar.selftest`, 1 minute): SNR per frequency
  and speaker, lock, drift, clipping, audible leak, whether the two speaker
  channels are physically separate, recommended profile.
- **Start-up calibration** (~5 s, hands away): alignment, reference channel,
  noise floors. `{"cmd":"calibrate"}` repeats it.
- **A "none" class** fed by real distractors, plus the gates in 5.5 (click,
  walking-in, range, strength).
- **Debounce**: dead time, opposite-gesture lock-out, wind-up/stroke hold.
- **Confidence**: threshold 0.6 on calibrated probabilities; breathing needs
  confidence and three agreeing estimates.
- **Self-healing**: lost lock re-syncs within 1.3 s; the noise floor adapts if
  the room gets louder; a stalled stream raises an error event.
- **Physics fallback**: `--physics` runs the whole demo with no trained model.
- **No-hardware fallback**: `--backend file` replays a good recording through
  the real engine; `--backend sim` runs a scripted visitor. Both are labelled as
  such in `status.source`.
- **Breathing demo posture**: seated or leaning on the table, 0.5-0.8 m, facing
  the laptop, hands still. Standing free works only for deep breathing.

## 9. Risks and fallbacks

| Risk | How we would notice | Fallback |
|---|---|---|
| Hand echo much weaker than assumed | self-test "echo headroom" low; recorder says "NO movement detected" | raise volume; profile `strong`, then `low`; gesture closer (20 cm) |
| Both speaker channels come from one driver (3-speaker laptop) | self-test "left/right speaker channels arrive N µs apart" under 60 | nothing breaks: the two-microphone cue carries direction; consider profile `clean` |
| Two-microphone cue too weak (mics very close) | train.py prints "microphones +0.00" | rely on the speaker cue; if both are weak, use push/pull for next/back instead of swipes |
| Audible whine or buzz | ears; self-test "audible leak" | lower volume; profile `clean`; move up is not possible (response falls), so trade volume for level |
| Breathing unreliable in the hall (airflow, standing sway, weak chest echo) | `breathing.state` stays `acquiring` | seat the judge, 0.5 m, hands on lap; raise volume; it reports "--" rather than a wrong number |
| Model does not generalise to judges | `--val-person` score well below the within-team score | more people in the data; mirror augmentation is already on; `--physics`; raise `--min-confidence` |
| False triggers from people walking past | console shows gestures nobody made; `status.last_decision` | range gate and strength gate are on; add scene sessions (distractors); raise `--min-peak-db` |
| Exclusive mode cannot be opened | engine exits with the reason | close Teams/Discord/browser audio; tick "allow exclusive control" for both devices; `--duplex`; `--blocksize 3840` |
| Windows steals the device (notification sound, update) | `error` event, stream stalls | restart the engine (5 s); disable system sounds |
| Hall acoustics differ from the room we trained in | accuracy drop at the venue | features are floor-relative; record a session at the demo table and retrain (2 min) |
| Drop to 17 kHz needed | weak SNR above 18.5 kHz | `--profile low` everywhere (record, train, engine); warn that young listeners may hear it |

## 10. Stretch goals

- **Fall detection** (the natural next healthcare step). Signature: a burst of
  fast, large gross movement whose range changes by >0.5 m in <1 s, followed by
  `present` with `motionless_s` counting or movement only at a new, larger
  range. All ingredients exist (`presence.far_db`, range of the strongest
  movement, `motionless_s`); it needs a rule, a mattress and a volunteer.
- **Apnea alert**: `breathing.apnea` is already emitted (experimental).
- **Sway rejection for standing people**: hips, chest and head sit at different
  ranges for a standing person; subtract the phase of a non-breathing range from
  the chest range.
- **Heart rate**: 0.2-0.5 mm chest movement at 1-2 Hz is visible in principle at
  0.75 rad/mm; needs a seated person and a quiet room.
- **A "wave" gesture**, more gestures, per-user few-shot calibration (see
  `DATA_STRATEGY.md`).
- **Range-time picture as a model input** (hand distance over time).

## 11. What is verified and what is not

Verified in simulation (pytest, 84 tests, no hardware): signal purity and
seamless looping; Doppler sign and speed against geometry; frame sync for any
latency/block size; range to 8 cm; clock drift ±40 ppm; re-lock after dropped
audio; breathing within 1 breath/min at 0.5-1 m under measured-level noise, a
direct path 40-50 dB above the echo, hall noise and sway; no rate invented for an
empty room or a held breath; all four gestures detected and named by the
physics rules; either direction cue alone suffices; clicks rejected; presence
sequence; engine event stream; record -> train -> ONNX -> classify.

**Not verified: anything on the real laptop.** Echo strengths, the speaker
layout, the microphone spacing, audibility and every threshold marked "tune"
are assumptions until the self-test and the first recording session have run.
The simulator's echo levels were chosen conservatively (hand −22 dB, chest −40
dB relative to the direct path, noise at the level measured at 36 % volume).

## References

1. Gupta, Morris, Patel, Tan. SoundWave. CHI 2012. https://ubicomplab.cs.washington.edu/pdfs/soundwave.pdf
2. Tarzia, Dick, Dinda, Memik. Sonar-based measurement of user presence and attention. UbiComp 2009. https://stevetarzia.com/papers/sonar_ubicomp.pdf
3. Wang, Zhao, Liu et al. LLAP: Device-free gesture tracking using acoustic signals. MobiCom 2016. https://samsonsjarkal.github.io/KeSun/files/mobicom16_ultrasound.pdf
4. Nandakumar, Iyer, Tan, Gollakota. FingerIO. CHI 2016. https://fingerio.cs.washington.edu/fingerio.pdf
5. Yun, Chen, Zheng, Mao, Qiu. Strata. MobiSys 2017. https://yichao0319.github.io/assets/publications/mobisys17_yun.pdf
6. Nandakumar, Gollakota, Watson. Contactless sleep apnea detection on smartphones (ApneaApp). MobiSys 2015. https://homes.cs.washington.edu/~gshyam/Papers/apneaapp.pdf
7. Nandakumar, Gollakota, Sunshine. Opioid overdose detection using smartphones. Sci. Transl. Med. 2019. https://par.nsf.gov/servlets/purl/10095248
8. Wang et al. C-FMCW based contactless respiration detection using acoustic signal. IMWUT 1(4), 2018. https://doi.org/10.1145/3161188
9. Wang, Huang, Mao. SonarBeat. ICCCN 2017. https://researchr.org/publication/WangHM17-0
10. Smartphone pure-tone sonar breathing monitoring during sleep. Sci. Rep. 2026. https://doi.org/10.1038/s41598-026-69919-z
11. Wang, Sunshine, Gollakota. Contactless infant monitoring using white noise (BreathJunior). MobiCom 2019. https://homes.cs.washington.edu/~gshyam/Papers/whitenoise.pdf
12. Wang et al. Using smart speakers to contactlessly monitor heart rhythms. Commun. Biol. 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC7943557
13. Ling et al. UltraGesture. SECON 2018 / IEEE TMC 2022. https://doi.org/10.1109/SAHCN.2018.8397099
14. Wang et al. Push the limit of acoustic gesture recognition (RobuCIR). INFOCOM 2020. https://shenjiaxing.win/pdf/wang_push_2020.pdf
15. Ruan et al. AudioGest. UbiComp 2016. https://doi.org/10.1145/2971648.2971736
16. Experience: practical problems for acoustic sensing. MobiCom 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC12798654
17. Roy, Hassanieh, Roy Choudhury. BackDoor. MobiSys 2017. https://synrg-web.web.engr.illinois.edu/papers/backdoor_mobisys17.pdf
18. Roy et al. Inaudible voice commands: the long-range attack and defense. NSDI 2018. https://www.usenix.org/system/files/conference/nsdi18/nsdi18-roy.pdf
19. WindDancer. IMWUT 2025. https://doi.org/10.1145/3729469
20. Elliptic Labs AI Virtual Human Presence Sensor on Lenovo laptops. https://ellipticlabs.com/mfn_news/elliptic-labs-shipping-ai-virtual-human-presence-sensor-on-thinkpad-x1-carbon-gen-12-laptop/
21. Google Nest ultrasound sensing. https://support.google.com/googlenest/answer/9509981
22. Amazon Echo ultrasonic motion sensing. https://www.amazon.science/blog/the-science-behind-ultrasonic-motion-sensing-for-echo
23. Ashihara. Hearing thresholds for pure tones above 16 kHz. JASA 2007. https://pubs.aip.org/asa/jasa/article/122/3/EL52/852964
24. Lee et al. Extended high-frequency hearing thresholds by age, 2012. https://pmc.ncbi.nlm.nih.gov/articles/PMC3606020
25. Chest wall motion during quiet breathing. J. Appl. Physiol. 1997, https://doi.org/10.1152/jappl.1997.83.5.1531 ; review https://pmc.ncbi.nlm.nih.gov/articles/PMC4035586
26. Postural sway, centre-of-pressure statistics. https://stacks.cdc.gov/view/cdc/193681/cdc_193681_DS1.pdf
27. PortAudio WASAPI host API, v19.7.0. https://github.com/PortAudio/portaudio/blob/v19.7.0/src/hostapi/wasapi/pa_win_wasapi.c
28. Cai, Zheng, Luo et al. Ubiquitous acoustic sensing on commodity IoT devices: a survey. IEEE COMST 2022. https://arxiv.org/abs/1901.03450

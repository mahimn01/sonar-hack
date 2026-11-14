# Devpost submission: Laptop Sonar

> **Before you submit**, replace every bracket below, then delete this box.
> - `[HONEST SCORE]`: the "Honest score" line from training, word for word (held-out people, never the presenter). Give accuracy, the share of gestures recognised at the 0.6 threshold, the share of other movements that would trigger, and how many people were held out.
> - `[N PEOPLE]`: how many different people are in the gesture recordings.
> - `[LIVE BREATHING CHECK]`: one line from rehearsal on the HP: what the screen showed against a counted or paced rate, seated at 0.6 m. If it was never checked, delete the sentence that contains it.
> - `[VIDEO LINK]`, `[REPO LINK]`.
> - The optional line at the end of Inspiration: fill it only if it is true, otherwise delete it.
> - The presence/breathing retuning was in progress in another session at 00:41 on Nov 14 (uncommitted). If it lands, presence and breathing look at 0.30-1.3 m instead of 2 m / 0.25-1.6 m, and short disturbances (up to 3 s) no longer restart breathing. Then update the 2 m and 0.25-1.6 m figures here.
> - "about 136,000 weights" assumes the default width (24). If `--sweep` picked width 16 or 32, write 61,000 or 242,000 (check `TRAINING_RUN.json`).

## Name and tagline

**Name:** Laptop Sonar. It matches the slides and the app, and it says what the thing is. Other options: *Stillwatch* (what it watches for), *Room Sonar* (once it runs on more than laptops).

**Tagline** (Devpost shows one line under the name). Recommended first:

1. Camera-free care from the laptop already in the room: breathing, presence and gestures from echoes you can't hear.
2. A stock laptop's speakers and microphones, turned into a sonar that measures breathing and sounds a no-breathing alarm. No camera, nothing to wear.
3. Inaudible sonar on an ordinary Windows laptop. It counts breaths, notices people and reads hand gestures, all on the CPU.

---

## Inspiration

In a year-long study of people over 90, 82% of falls happened while the person was alone. Call alarms were widely available, yet in most of the falls that ended with a long time on the floor, nobody used one (Fleming and Brayne, BMJ 2008). A camera would have seen those falls. But people fall, and sleep, in bedrooms and bathrooms, and those are the rooms where nobody wants a camera.

Breathing has the same blind spot. Respiratory rate is the vital sign hospitals record least often, even though a raised rate is a strong predictor of events such as cardiac arrest, and an adult breathing more than 24 times a minute is likely critically ill (Cretikos et al., Med J Aust 2008).

Most of those rooms already contain a speaker and a microphone. Researchers have measured breathing with phone sonar since 2015 (ApneaApp, University of Washington), and Elliptic Labs ships ultrasonic presence sensing on Lenovo laptops. None of those products reads breathing on a laptop. We wanted to know whether an ordinary laptop, with nothing added, could watch a person breathe in real time without a camera, and we had one weekend to find out.

[Optional, only if true: one sentence on why this matters to one of us.]

## What it does

Laptop Sonar plays a quiet signal at 18.5-21.5 kHz through the laptop's own speakers. Most adults can't hear that band, and it is the first range hearing loses with age. The built-in microphones record the echoes, and the laptop's CPU turns them into four live readings:

- **Breathing rate** of a still, seated person 0.5-0.8 m from the screen, with a waveform of the chest moving. When the estimate is not trustworthy, the screen shows a dash instead of a number.
- **A no-breathing alarm.** After a steady rate has been on screen for 5 seconds, the Care Monitor turns red when breathing movement stops for 10 seconds (the clinical length of an apnea), and counts the seconds until it starts again.
- **Presence** within about 2 m: empty, someone entering, someone present. The engine also reports the person's distance and how long nothing has moved.
- **Hand gestures** a hand-length from the screen: swipe left, swipe right and push. Our pitch deck changes slides when you swipe at it.

The camera stays taped over. No image is ever captured, and the running system stores no audio. The engine's only output is small JSON messages (rates, states, gesture names, the chest waveform and a coarse Doppler strip) on a WebSocket that, by default, only the laptop itself can reach.

There are two screens. The pitch deck follows your hand. The Care Monitor is laid out like a nurse-station view. If you choose to open it to the local network, a phone on the same Wi-Fi can show it too.

## How we built it

**The signal.** One 20 ms waveform loops forever: a steady 18.5 kHz pilot tone on one speaker channel and a 19.5-21.5 kHz chirp on the other. Every component is an exact multiple of 50 Hz, so the loop has no seam, no click and no energy below the band. We built the chirp in the frequency domain (41 lines with Schroeder phases), which gives a flat spectrum and an almost constant envelope. The tone measures speed: a hand moving at 1 m/s shifts it by about 108 Hz. The chirp measures distance, in 2.7 cm steps out to 2.6 m.

**Getting the signal through Windows.** In Windows' normal (shared) audio mode, the microphone recording held nothing from 15 to 22 kHz: Windows low-passes it. So we open both devices in WASAPI exclusive mode at 48 kHz. Two more problems hide there: PortAudio never reports a dropped buffer in that mode, and playback and recording start with a different offset every time. So the receiver finds its own timing from the direct speaker-to-microphone sound, tracks clock drift with a second-order loop, and re-locks by itself when audio goes missing.

**Breathing.** We read the echo's phase, not its loudness. The wavelength is about 1.7 cm, so a 1 mm movement of the chest turns the echo's phase by 43 degrees. The estimator keeps 30 s of echoes from 0.25 to 1.6 m, finds the distance where something is still moving, and adds up about five neighbouring distance taps from both microphones with an SVD, which buys 7-10 dB over any single tap. It fits a circle to that complex signal, reads the angle as millimetres of chest movement, and picks the strongest rhythm between 6 and 36 breaths per minute, checking the second harmonic so it doesn't report double the rate. A number only appears once a confidence gate passes and three estimates in a row agree within 2 breaths per minute. The estimator also checks that the chest moved in the last 6 seconds, so a held breath is never "measured" from old data.

**Presence.** A 7 Hz high-pass on the echo at each distance separates walking and reaching (fast changes) from breathing and sway (slow changes, rejected by more than 25 dB). A small state machine turns that into empty, entering and present, and counts how long nothing has moved. Presence and breathing need no training at all.

**Gestures.** The tone path gives a Doppler picture per microphone, 50 columns a second. A segmenter cuts each movement into a 1.6 s picture, and a four-block CNN of about 136,000 weights, run by ONNX Runtime on the CPU, names it. Rules around the network stop double triggers: a 0.6 confidence threshold, 0.4 s of dead time, ignoring the hand coming back for 1.2 s, and a range gate from the chirp that ignores movement beyond 0.6 m. The self-test told us the two speaker channels reach the microphones within 4 microseconds of each other, so for our purposes they come from one place. Left and right therefore come from the two microphones: as a hand crosses the screen, the phase difference between them turns one way or the other.

**Training data.** A gesture model normally needs hours of recordings, and we had a weekend. We wrote a physics echo simulator (a moving hand, arm and torso; random virtual laptops, rooms and people; noise, clicks and clock mismatch) and checked it against 191 real public recordings made on other hardware. The simplest physics rule, push versus pull from the sign of the pitch shift, got 96 of 96 real clips right. A model trained only on synthetic data recognised 83 of 96 real pushes and pulls and never mixed the two up. With synthetic pre-training, one real clip per class gave 94% accuracy on those recordings, against 76% when training from scratch. For the demo model we laid synthetic gestures over real recordings from our own laptop (sound adds linearly), fine-tuned on recordings of [N PEOPLE] people, and scored it on people it never saw: [HONEST SCORE].

**The app.** Plain HTML, CSS and JavaScript with no build step. One adapter file is the only code that knows the engine's message format. A mock engine lets us rehearse without the hardware, and everything it sends is labelled "Simulated data" on screen.

**Tests.** 89 core tests and 8 simulator tests push simulated physics through the real pipeline: Doppler sign against geometry, sync at any latency and block size, ±40 ppm of clock drift, re-locking after deleted audio, breathing within 1 per minute in 7 simulated cases (10-20 per minute, 0.5-1 m, shallow and deep), no rate at all for an empty room or a held breath, and the whole record, train, export and classify chain. The web app has its own Python and Node tests.

## Challenges we ran into

- **Windows deleted our signal.** Shared mode filters 15-22 kHz out of the microphone. We found it only by measuring, so the project now starts with a one-minute self-test that prints the SNR at every test frequency.
- **The stereo speakers weren't stereo up here.** Both channels arrived within 4 microseconds, which killed the two-speaker direction cue we had planned. The two-microphone phase cue, built as a backup, became the main one.
- **Timing we couldn't trust.** Two separate audio devices, a new offset on every start, and dropped buffers that are never reported. The chirp receiver has to synchronise itself and notice when it slips.
- **Everything slow looks like breathing.** A standing person sways a few millimetres in the same 0.1-0.25 Hz band, and any movement within 2 m restarts the measurement. We built the demo around a seated person and made the estimator show a dash rather than guess.
- **Our first synthetic model recognised 8 of 96 real gestures.** Putting simulated and real pictures side by side showed that real strokes are slower, the arm adds a thick echo, and echoes are louder than we had assumed. After we recalibrated the simulator: 83 of 96.
- **Audible leaks.** A tone and a chirp sharing one speaker can mix into a sound you can hear. The "clean" profile gives each speaker one component, and the self-test measures what leaks into the audible band.

## Accomplishments that we're proud of

- **It works on a stock laptop.** On an HP ENVY x360 we measured: self-test passed, weakest pilot tone 63 dB above the noise, chirp band 73 dB (both in a 1 Hz band), receiver locked (coherence 0.81), clock drift -0.3 ppm, nothing audible above room noise. The live engine kept up in real time with the 50 frames a second the signal produces, at about 85 ms of processing latency as the engine reports it (audio block plus compute). Its weakest pilot tone stood about 53 dB above the Doppler noise floor, and it detected a person at about 0.8 m.
- **Breathing from the echo phase**, within 1 breath per minute in all 7 simulated cases. On the real laptop: [LIVE BREATHING CHECK].
- **96 of 96** real public clips sorted into push and pull by physics alone, and a simulator good enough that one real example per gesture goes a long way.
- **An honest score.** The model is tested on people it never saw, never on the presenter: [HONEST SCORE].
- **Privacy built into the signal chain.** The first filter cuts everything below about 15 kHz by 100 dB, speech included. There is no camera, and the running system stores no audio and uses no cloud.

## What we learned

- Measure the hardware before designing for it. Our two biggest decisions, exclusive mode and the two-microphone direction cue, came from measurements and not from the plan.
- Phase is the right tool for millimetres. A chest that moves 4 mm barely changes how loud the echo is, but swings its phase through almost half a cycle.
- A health tool that says "I don't know" beats one that always shows a number. Most of the breathing code decides when to stay quiet.
- For gestures, more people beats more repetitions. A simulator teaches the shapes. Only real recordings teach where a gesture ends and ordinary movement begins.
- Physics rules make a good safety net and a poor product. In our harshest simulated hall they called 49% of ordinary movement a gesture, against 11% for the trained model.

## What's next for Laptop Sonar

- **Check breathing against a reference**: a respiration belt and counted breaths, across ages, body sizes, chairs and beds. Report the error and how often it shows a dash.
- **Overnight at the bedside**: lying down, under a blanket, and rejecting sway for people who are not sitting.
- **Fall detection.** A fall is a fast movement whose distance changes by more than half a metre in under a second, followed by stillness. Presence, distance and the stillness timer already exist. The rule and the testing don't yet.
- **Share the speaker.** Exclusive mode means no other app can play or record sound while the sonar runs. A product has to live inside the audio driver or DSP.
- **Alerts on a caregiver's phone**, and trends over days instead of a live screen.
- **Other devices** with a speaker and a microphone: tablets, smart speakers, TVs. A MacBook path exists in the code but has not run on a MacBook yet.

## Built with

`python` `numpy` `scipy` `sounddevice` `portaudio` `wasapi` `soundfile` `pytorch` `onnx` `onnxruntime` `websockets` `javascript` `html5` `css3` `pytest` `node.js` `windows`

## Links

- Demo video: [VIDEO LINK]
- Code: [REPO LINK]

---

### Sources for the numbers above

- Falls among people over 90, alone, call alarms: Fleming J, Brayne C. *Inability to get up after falling, subsequent time on floor, and summoning help: prospective cohort study in people over 90.* BMJ 2008;337:a2227.
- Respiratory rate: Cretikos MA et al. *Respiratory rate: the neglected vital sign.* Med J Aust 2008;188(11):657-659.
- Apnea length (10 s): the American Academy of Sleep Medicine scoring definition, used as `apneaSec` in `app/web/js/config.js`.
- Prior work and the Elliptic Labs, Google Nest and Amazon Echo references: `DESIGN.md` section 2.
- Every measured number from the laptop: the self-test and engine runs on the HP ENVY x360 on Nov 13-14, 2026. Simulator and public-clip results: `DATA_STRATEGY.md` section 1 and `DESIGN.md` section 11.

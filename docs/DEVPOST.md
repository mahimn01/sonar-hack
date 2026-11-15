# Devpost submission: Sonar Care

> **Before you submit**, replace every bracket below, then delete this box.
> - `[VIDEO LINK]`, `[REPO LINK]`.
> - The optional line at the end of Inspiration: fill it only if it is true, otherwise delete it.
> - When this was written, these parts had not yet run on the real MacBook (`SONAR_CARE.md`, section 9): a key press reaching another app, the media keys, the alert's notification with chime and voice, the menu bar item, and a deliberate double push. Try each one in rehearsal (`DEMO_RUNBOOK.md`, section 1). If one does not work on the MacBook, change the text here before you submit.
> - "We have not measured how often a deliberate double push is caught" appears under What it does and under Challenges. If you count it in rehearsal, replace that sentence with your count in both places.
> - The gesture score belongs to the model in `models-candidate-mac-calm50`. If a different model ships, the four numbers (133 of 206, 22 of 499, 70 of 81, 5 of 114) change with it.

## Name and tagline

**Name:** Sonar Care. It is the name on the app, the menu bar item and the dashboard.

**Tagline** (Devpost shows one line under the name). Recommended first:

1. Camera-free care from the laptop already in the room: touch-free control, a room monitor and a help gesture, from echoes you can't hear.
2. A MacBook's own speakers and one microphone become a sonar. Swipe to turn a page, push twice to call for help. No camera, nothing to wear.
3. Inaudible sonar on a stock MacBook: gestures that press real keys, a help gesture, and an alert when someone stops moving. All on the CPU.

---

## Inspiration

In a year-long study of people over 90, 82% of falls happened while the person was alone. Call alarms were widely available, yet in most of the falls that ended with a long time on the floor, nobody used one (Fleming and Brayne, BMJ 2008). A camera would have seen those falls. But people fall, and sleep, in bedrooms and bathrooms, and those are the rooms where nobody wants a camera.

A keyboard has the same weakness as a pendant: it only helps if you can reach it and press it. Someone with their hands full, with gloves on, or who cannot reach the laptop is stuck.

Breathing has a blind spot too. Respiratory rate is the vital sign hospitals record least often, even though a raised rate is a strong predictor of events such as cardiac arrest, and an adult breathing more than 24 times a minute is likely critically ill (Cretikos et al., Med J Aust 2008).

Most of those rooms already contain a speaker and a microphone. Researchers have measured breathing with phone sonar since 2015 (ApneaApp, University of Washington), and Elliptic Labs ships ultrasonic presence sensing on Lenovo laptops. None of those products reads gestures or breathing on a laptop. We wanted to know whether an ordinary laptop, with nothing added, could be controlled without touch, take a call for help and keep an eye on a room without a camera. We had one weekend to find out.

[Optional, only if true: one sentence on why this matters to one of us.]

## What it does

Sonar Care is a menu bar app and a dashboard that turn a MacBook into a touch-free, camera-free sensor for patients and caregivers.

The MacBook's speakers play sound too high to hear: a steady 18.5 kHz tone from the left speaker, a steady 19.0 kHz tone from the right, and a 19.5-21.5 kHz chirp (a short sweep that repeats 50 times a second). Most adults can't hear that band, and it is the first range hearing loses with age. The laptop's one microphone records the echoes, and its CPU turns them into four things:

- **Touch-free control.** A hand gesture presses a real key in whatever app is in front. There are three gestures, made with an open hand a hand-length from the screen: swipe left, swipe right and push. Presets: Slides / PDF (swipe right = next, swipe left = previous, push = space), Media (next track, previous track, play/pause), Scroll (page down, page up) and Custom (your own key for each gesture).
- **Room monitor.** A caregiver dashboard shows whether someone is within about 1.3 m and how far away, a timeline of what happened, and an early breathing estimate with its waveform (a line that follows the chest). It raises a "No movement" alert when someone is present but has not moved for a set time: 60 seconds by default.
- **Help gesture.** Two pushes in a row mean "Help requested". The request first shows on screen for 3 seconds with a Cancel button. Then the alert goes out: a full-screen alert with a large Acknowledge button, a macOS notification, a chime, and the words "Help requested" spoken aloud. The time until someone acknowledges it is logged. A hand gesture can never cancel a help request.
- **Guided setup.** A first-run walk-through checks the signal, calibrates ("keep still for 5 seconds"), asks for the macOS Accessibility permission that key presses need, and gives a practice round for each gesture with live feedback.

What it cannot do yet:

- The gesture model fits one laptop. Left and right depend on where this MacBook's two speakers and its microphone sit.
- One push is sometimes read as two, because the hand coming back also looks like a push. After our filter, about 1 single push in 12 can still look like a help gesture. That is why a help request can be cancelled on screen for 3 seconds. We have not measured how often a deliberate double push is caught.
- Breathing from one microphone is experimental. The dashboard labels it "Early estimate" and shows a dash when the engine is not confident. The app also has an experimental "No breathing movement" alert, with the same limits.
- The "No movement" alert is a prompt to check on someone. It is not fall detection.

The camera stays taped over. No image is ever captured, and no audio is stored or sent anywhere. The engine's only output is small JSON messages (states, gesture names, rates, the chest waveform and a coarse Doppler strip) on a connection that only the laptop itself can reach. Sonar Care keeps a settings file and a text log of events on the laptop. Nothing goes to a cloud.

## How we built it

**The signal.** One 20 ms waveform loops forever at 48 kHz: the 18.5 kHz tone on the left speaker, the 19.0 kHz tone on the right, and the chirp, which shares the left speaker. Every component is an exact multiple of 50 Hz, so the loop has no seam, no click and no energy below the band. We built the chirp in the frequency domain (41 lines with Schroeder phases), which gives a flat spectrum and an almost constant envelope. The tones measure speed through the Doppler effect (a moving hand shifts the pitch of its echo): a hand moving at 1 m/s shifts the 18.5 kHz tone by about 108 Hz. The chirp measures distance, in 2.7 cm steps out to 2.6 m.

**Left and right from one microphone.** A MacBook gives us one microphone channel, so direction cannot come from comparing microphones. It comes from the speakers. Each speaker plays its own tone and the two speakers sit apart, so a hand crossing the screen moves away from one speaker while it approaches the other. The two tones change in opposite ways and at different moments, and that pattern tells left from right. The microphone hears both tones at once, and the software separates them by frequency.

**Timing.** Playback and recording are two separate audio streams, and their offset is different on every start. The receiver never trusts it. It finds its own timing from the direct speaker-to-microphone sound, tracks clock drift with a second-order loop, and re-locks by itself when audio goes missing. On macOS the only setup is 48 kHz for the speakers and the microphone in Audio MIDI Setup, plus the Microphone permission.

**Gestures.** Each tone gives a Doppler picture (pitch shift over time), 50 columns a second. A segmenter cuts each movement into a 1.6 s picture, and a small CNN (a neural network for pictures) names it: swipe left, swipe right, push, or nothing. The packaged model averages three copies of a four-block network of about 136,000 weights each, run by ONNX Runtime on the CPU. The engine stops double triggers with a 0.6 confidence threshold, 0.4 s of dead time, and by ignoring the opposite swipe for 1.2 s (the hand coming back). Sonar Care adds its own checks before a key is pressed: 0.7 s between actions, the opposite swipe ignored for 1.5 s, and the same gesture ignored for 1.3 s.

**Training data.** A gesture model normally needs hours of recordings, and we had a weekend and three people. We wrote a physics echo simulator (a moving hand, arm and torso; virtual laptops, rooms and people; noise, clicks and clock mismatch) and checked it against 191 real public recordings made on other hardware. The simplest physics rule, whether the hand is coming toward the microphone or going away (the sign of the pitch shift), got 96 of 96 real clips right. A model trained only on simulated data recognised 83 of those 96 clips and never mixed up the two directions. With simulated pre-training, one real clip per class gave 94% accuracy on those recordings, against 76% when training from scratch. For the demo model we fitted the simulator to this MacBook, made its simulated gestures replay real tracked hand paths, and trained on those clips together with recordings of 3 people made on this MacBook.

**The honest score.** We scored the model on people it had not trained on. With each of the three people held out of training in turn, it recognised 133 of 206 gestures (65%), and 22 of 499 other movements set off a gesture (4.4%). On the best-matched held-out person it recognised 70 of 81 gestures (86%), with 5 of 114 other movements triggering. That is three people on one laptop.

**Presence.** A 7 Hz high-pass on the echo at each distance separates walking and reaching (fast changes) from breathing and sway (slow changes, rejected by more than 25 dB). A small state machine turns that into empty, entering and present. It looks only between 0.3 and 1.3 m: nearer is the laptop's own direct sound, farther is the rest of the room. Presence needs no training.

**Breathing (an early estimate).** We read the echo's phase (its timing within one wavelength), not its loudness. The wavelength is about 1.7 cm, so a 1 mm movement of the chest turns the echo's phase by about 43 degrees. The estimator keeps up to 30 s of echoes from 0.3 to 1.3 m, finds the distance where something is still moving, and combines about five neighbouring distance taps with an SVD (a standard way to add up several noisy copies of one signal). It fits a circle to that signal, reads the angle as millimetres of chest movement, and picks the strongest rhythm between 6 and 36 breaths per minute, checking the second harmonic so it doesn't report double the rate. A number only appears once a confidence gate passes and three estimates in a row agree within 2 breaths per minute. One microphone gives the estimator half the looks at the chest it was designed around, so on this MacBook we treat the reading as experimental. Breathing needs no training either.

**The app.** Sonar Care is one Python process. A supervisor starts the engine and restarts it if it crashes. Every decision (which gesture presses which key, when two pushes mean help, when to raise "No movement", how an alert is acknowledged or resolves) lives in one file of plain logic with no clock and no input or output. So each rule is tested with a fake clock, and alerts fire the same way whether or not a browser is open. Key presses go through macOS CoreGraphics, loaded with ctypes. The menu bar item uses rumps. The notification, the chime and the spoken alert use tools that ship with macOS. The dashboard is plain HTML, CSS and JavaScript with no build step. Its web server listens on this computer only and refuses requests from other websites, so a web page cannot change settings or make the app press keys. A mock engine lets us rehearse without the hardware, and everything it sends is labelled "Simulated data" on screen.

**Tests.** The core tests push simulated physics through the real pipeline: Doppler sign against geometry, sync at any latency and block size, clock drift up to 40 ppm, re-locking after deleted audio, breathing within 1 breath per minute in 7 simulated cases (10-20 per minute, 0.5-1 m, shallow and deep), no rate at all for an empty room or a held breath, and the whole record, train, export and classify chain. Sonar Care's rules are tested with a fake clock, and the whole service is tested against the mock engine. The browser code has its own Node tests.

## Challenges we ran into

- **One push is reported twice.** We replayed our recordings and counted. Of 69 single pushes, 28 (41%) produced two push events, because the hand coming back also looks like a push. So a second push sooner than 1.0 s counts as the same push. That still leaves 6 of 69, about 1 in 12, that look like a double push: the hand came back after a pause. Timing alone cannot remove those.
- **A help call must never be dropped by accident, and a key must never be pressed by accident.** A push can mean "press the push key" or the first half of "help". A single push therefore presses its key only after 2.5 s with no second push, so one movement never does both. A help request shows for 3 s with a Cancel button, because of that 1 in 12. Only a person at the screen can cancel it. A hand gesture cannot, and another push sends it at once. We have not measured how often a deliberate double push is caught.
- **A noisy empty room can look faintly "present".** On another Mac we saw an empty room reported as "present" at 5% confidence. The room monitor only trusts a person after the engine reports "present" at 60% confidence or more twice in a row, so a faint reading never arms the "No movement" alert.
- **The alarm can fool its own sensor.** The alert chime and voice come out of the same speakers the sonar uses, and a loud sound there can look like movement. That would let the "No movement" alert cancel itself. Movement seen while the alert sound plays is ignored.
- **Our first honest score was 24 of 81.** That was one held-out person. The biggest cause was a labelling mistake: one recording session had left and right swapped (27 of its 30 swipes went the opposite way to the prompt). Next, the real recordings were only 3.8% of the training pictures and were drowned by simulated ones. Last, the engine cut loud movements at a fixed 1.4 s, which chopped 111 of 218 recorded gestures. Fixing those took the same person to 59 of 81, and the simulator with real hand paths took it to 70 of 81.
- **Fast hands leave the picture.** The Doppler picture covers hand speeds up to about 2.3 m/s. One of us swipes faster than that, and no version we trained recognised more than half of his gestures. A half-second sweep works. A flick does not.
- **Everything slow looks like breathing.** A standing person sways a few millimetres at the same rhythm as breathing. We made the estimator show a dash rather than guess, and built the demo so that nothing depends on the breathing number.
- **Our first synthetic model recognised 8 of 96 real gestures.** Putting simulated and real pictures side by side showed that real strokes are slower, the arm adds a thick echo, and echoes are louder than we had assumed. After we recalibrated the simulator: 83 of 96.

## Accomplishments that we're proud of

- **One command starts everything.** `scripts/sonar_care.sh` sets up Python, starts the engine, keeps it alive, adds the menu bar item and opens the dashboard. No camera, no extra hardware, no cloud.
- **A gesture that presses a real key in any app**, with left and right told apart by one microphone.
- **A help gesture designed around its own mistakes.** We measured how a single push gets misread, then built the timing rule and the 3-second cancel time from those counts.
- **An honest score.** With each of three people held out of training in turn: 133 of 206 gestures recognised (65%), and 22 of 499 other movements triggering (4.4%). That is the number we lead with. The 86% of the best-matched person is never quoted alone.
- **96 of 96** real public clips sorted by physics alone into toward and away, and a simulator good enough that one real example per gesture goes a long way.
- **Privacy built into the signal chain.** The first filter cuts everything below about 15 kHz by 100 dB, speech included. There is no camera, and the running system stores no audio and uses no cloud.
- **Breathing from the echo phase**, within 1 breath per minute in all 7 simulated cases. On the real MacBook it is still an early estimate, and the screen says so.

## What we learned

- Count before you write a rule. The help gesture's timing (a 1.0 s minimum gap, a 2.5 s limit) came from replaying our recordings and counting what a single push really produces.
- An alert needs an answer for its own false alarms and its own misses. The cancel time covers the push that looks like two. "A gesture can never cancel" covers the real call.
- Check the labels before blaming the model. One session with left and right swapped made 30 of 81 test gestures impossible to get right.
- For gestures, more people beats more recipes. We trained 49 recipes on the same three people, and the best ones are within chance of each other.
- What lifted the simulator was real motion. Replaying real tracked hand paths moved the score. Fitting the simulator to the MacBook's acoustics, by itself, did not.
- Phase is the right tool for millimetres. A chest that moves 4 mm barely changes how loud the echo is, but swings its phase through almost half a cycle.
- A health tool that says "I don't know" beats one that always shows a number. Most of the breathing code decides when to stay quiet.
- Physics rules make a good safety net and a poor product. In our harshest simulated hall they called 49% of ordinary movement a gesture, against 11% for the trained model.

## What's next for Sonar Care

- **More people.** Three people on one laptop is a first measurement. The score needs dozens of people, including people who swipe fast.
- **Teach the model the hand coming back**, so one push is no longer read as two. Then measure the help gesture properly: deliberate double pushes across many people.
- **A model per device, or one that works across devices.** Today's gesture model only fits this MacBook's speaker and microphone layout.
- **Check breathing against a reference**: a respiration belt and counted breaths, across ages, body sizes, chairs and beds. Report the error and how often it shows a dash.
- **Fall detection, tested on real falls.** A fall is a fast movement whose distance changes by more than half a metre in under a second, followed by stillness. Presence, distance and the stillness timer already exist. The testing doesn't.
- **Alerts that reach a caregiver's phone.** Today the alert stays on the laptop (another device on the same network can open the dashboard and acknowledge it, if the app is started with `--lan`).
- **Share the speakers.** Loud audio through the same speakers can weaken the signal. A product has to live inside the audio driver.
- **Other devices** with a speaker and a microphone: tablets, smart speakers, TVs.

## Built with

`python` `numpy` `scipy` `sounddevice` `portaudio` `soundfile` `pytorch` `onnx` `onnxruntime` `websockets` `rumps` `coregraphics` `macos` `javascript` `html5` `css3` `pytest` `node.js`

## Links

- Demo video: [VIDEO LINK]
- Code: [REPO LINK]

---

### Sources for the numbers above

- Falls among people over 90, alone, call alarms: Fleming J, Brayne C. *Inability to get up after falling, subsequent time on floor, and summoning help: prospective cohort study in people over 90.* BMJ 2008;337:a2227.
- Respiratory rate: Cretikos MA et al. *Respiratory rate: the neglected vital sign.* Med J Aust 2008;188(11):657-659.
- Prior work and the Elliptic Labs reference: `DESIGN.md` section 2.
- The gesture score, the first score of 24 of 81 and what fixed it, the 49 recipes: the held-out runs behind `models-candidate-mac-calm50` (each of three people left out of training in turn), written up in the recall session's summary of Nov 14, 2026.
- One push reported twice, the 1 in 12, the help gesture's timing, the faint "present" reading, the alert sound: `docs/SONAR_CARE.md` section 7.
- The signal, the ranges and the gesture rules: `sonar/config.py`, `DESIGN.md` sections 4 and 5.
- Simulator and public-clip results: `DATA_STRATEGY.md` section 1.

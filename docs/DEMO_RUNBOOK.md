# Demo runbook for judging

For the HP ENVY x360 demo laptop on Windows 11. Note: presence/breathing retuning was in progress in another session at 00:41 on Nov 14 (uncommitted). If it lands, presence and breathing look at 0.30-1.3 m instead of 2 m / 0.25-1.6 m, and short disturbances (up to 3 s) no longer restart breathing. Then update the 2 m and 0.25-1.6 m figures here. The demo rules below stay safe either way. Commands run in PowerShell from the `sonar-hack` folder (the one with `README.md`). They call `.venv\Scripts\python` in full on purpose: plain `py` starts the system Python, which has none of the packages.

## On one screen

1. Charger in. Speaker volume at `[VOLUME %]` (the level the self-test passed at), and nobody touches it again.
2. Start in this order: self-test, engine, slides (section 2).
3. Show breathing and presence first. They need no trained model. Gestures come last.
4. **While breathing is measured, nothing moves within 2 m.** That includes the presenter's hands and the judges. Big movements farther away can also restart it.
5. **Ask for the breath-hold only after the rate has been on screen for 5 seconds.** Before that the alarm isn't armed.
6. If the same thing fails twice, switch to the backup and say so out loud (section 5). Never present a recording or the mock as live.

---

## 1. Tonight: rehearse and measure

Every blank here becomes a number in the pitch or the Devpost. Fill them on the HP, at the demo table if you can.

| Check | How | Result |
|---|---|---|
| Speaker volume used | Windows volume slider | `[VOLUME %]` |
| Time to first breathing rate | S sits at the 0.6 m mark, still; stopwatch from sitting down until the number appears. Three tries. | ___ s, ___ s, ___ s |
| Breath-hold to alarm | Wait 5 s after the rate appears, then hold. Stopwatch from the hold to the red orb. Three tries. | `[ALARM TIME]`: ___ s, ___ s, ___ s |
| Breathing accuracy | (a) S breathes in time with a metronome app at 12 and at 15 per minute, 60 s each; write down what the screen settles on. (b) Or replay last night's paced `breathing` session (the replay command is in section 5) and note what the engine prints for each 40 s part (12, 15, 20 per minute). Call that a replayed recording whenever you quote it. | `[LIVE BREATHING CHECK]`: 12 → ___, 15 → ___, 20 → ___ |
| Movement that restarts breathing | While the rate shows: a person stands still at 1.2 m; then talks without moving their hands at 2 m; then waves at 2 m. Does the orb drop back to "Hold still"? | ___ / ___ / ___ |
| Gestures, the trained person | 10 swipes right (left hand), 10 swipes left (right hand), 5 pushes | right ___/10, left ___/10, push ___/5, wrong-way ___ |
| Gestures, someone new | Show them once, then 10 swipes right | ___/10 |
| False gestures | 2 minutes of the presenter talking normally at 1-2 m, nobody gesturing | ___ |
| The 3-minute pitch with a timer | `PITCH.md`, twice | ___ : ___ |
| The 60-second pitch | Five times, with someone playing the judge | done |

Also tonight:

- Record the backup video (`BACKUP_VIDEO.md`). Copy it to both phones and the Mac **as a local file**. Venue Wi-Fi is not a plan.
- Pick the replay session: a `breathing` or `scene` recording from this laptop where someone sits still. Write its folder name here: `[REPLAY SESSION]`. Then run the replay once (command in section 5) to be sure it shows a rate.
- Check the engine prints `classifier: model (gesture.onnx)` with **no** line starting `WARNING: the model was trained with different settings`. If that warning appears, the model was trained for another profile: use `--physics` (and test it) or retrain.

---

## 2. Setup checklist (T-60 to T-30 min)

### Windows (once, 10 minutes)

- [ ] Close Teams, Discord, Zoom, Spotify and every browser tab that plays sound. Any of them can hold the speaker or the microphone.
- [ ] Classic Sound panel (Win+R, `mmsys.cpl`): Playback tab > the Realtek speaker > Properties > Advanced: tick **Allow applications to take exclusive control of this device**. Recording tab > the Intel microphone array: the same.
- [ ] Control Panel > Sound > Sounds: Sound Scheme **No Sounds**. A notification chime can steal the device in the middle of a demo.
- [ ] Do not disturb on. Windows Update paused for a week.
- [ ] Power: plugged in, screen and sleep set to **Never** while plugged in, lid-close action **Do nothing**.
- [ ] Bluetooth off. No headphones, USB microphones or USB audio plugged in.
- [ ] HDMI to a projector is fine: the engine picks the built-in Realtek speaker and Intel microphone array by name. Plug it in **before** starting the engine, then check the device names the engine prints.
- [ ] Volume at `[VOLUME %]`. Not muted (the slider sets how loud the sonar plays).

### The table (5 minutes)

- [ ] Laptop near the front edge of the table, screen facing the chair, lid at the angle used in rehearsal. If anyone moves the lid later, restart the engine (it re-learns the room in 5 s).
- [ ] Tape a mark on the table so that the subject's chest sits **0.6 m** from the screen (the engine wants 0.5-0.8 m).
- [ ] A solid chair that doesn't swivel or roll.
- [ ] Nothing that moves in front of the laptop: no fan, no air vent blowing across the table, no flapping sign. Static objects are fine; the engine removes still echoes.
- [ ] Mark the presenter's spot at least 2 m from the laptop.
- [ ] Tape over the webcam. It is part of the story.
- [ ] The subject takes off a bulky coat. Thick soft fabric soaks up high frequencies and moves less than the chest under it.

### Software (5 minutes)

0. **Pull the latest commit on the HP.** The corrected slides and the tag that marks a replay as not live are in git, not on the HP until you pull.

1. **Self-test** (about 35 s of near-inaudible tones; hands away, nobody walking past):

   ```powershell
   .venv\Scripts\python -m sonar.selftest --profile clean --blocksize 3840
   ```

   Compare with last night:

   | Line | Pass at | Last night |
   |---|---|---|
   | weakest pilot tone SNR / chirp band SNR (in 1 Hz) | 50 / 45 dB | 63 / 73 dB |
   | receiver lock | True, coherence above 0.8, re-locks 0 | True, 0.81 |
   | speaker/mic clock drift | under 5 ppm | -0.3 ppm |
   | microphone peak | not clipped | |
   | audible leak | not above room noise | not above room noise |
   | left/right speaker channels | (a warning here is expected on this laptop) | 4 µs apart |
   | verdict | PASS, or "OK with warnings" where the only warning is `speaker_separation` | passed |

   Coherence was 0.81 against a pass line of 0.8, so it has little margin. If the lock line fails: close other apps, check the charger, run it again.

2. **Engine** (leave this window open; Ctrl+C stops it):

   ```powershell
   .venv\Scripts\python -m sonar.engine --profile clean --blocksize 3840
   ```

   It should print the backend and device names, `classifier: model (gesture.onnx)` (or `physics` if there is no model), `events on ws://127.0.0.1:8765`, then `calibrating for about 5 s: keep hands away from the laptop`. For those 5 seconds nobody moves near the laptop.

3. **Slides** (a second PowerShell window):

   ```powershell
   .venv\Scripts\python app\serve.py --app
   ```

   It opens a clean Chrome or Edge window. Press **F** for full screen. Bottom-left HUD: a green dot and "Sonar live", and **no** "Simulated data" tag. Press **7** and check the live panel: echo SNR around 50 dB, frame rate about 50, latency under 100 ms. Press **5** to park on the live slide. **M** opens the Care Monitor in a new tab if a judge wants the nurse-station view.

### Slide keys

| Key | Does |
|---|---|
| → / ← (also Space, PgDn / PgUp) | next / previous slide |
| 1-9 | jump to a slide |
| O | live panel on or off (same as a push gesture) |
| L | pause gesture navigation: gestures still show in the HUD, slides stop moving |
| D | raw event inspector (debugging) |
| M | Care Monitor in a new tab |
| F, T, H | full screen, light/dark, hide HUD |

---

## 3. What to say while it calibrates or measures

- **Engine start (5 s):** "It's listening to the room for five seconds, learning what nothing moving sounds like, the way you'd zero a scale. If everyone keeps still for a moment, we're ready."
- **Breathing, before the number (10-15 s):** "It waits for ten seconds of stillness and three readings in a row that agree. Until then it shows a dash, because a wrong breathing rate is worse than none."
- **After the number, before the hold (5 s):** "It has to see a steady rate for five seconds before the no-breathing alarm is armed."
- **During the hold:** "Ten seconds without breathing is the clinical length of an apnea. That's what it waits for."
- **Restarting the engine:** "Restarting the sensor takes five seconds while it re-learns the room."

---

## 4. Order of the demo, minute by minute

### At the table (one judge visit, 3-5 minutes)

| Time | What happens | Who |
|---|---|---|
| Before they arrive | Slide 5 on screen. S sits at the mark, still, so a rate is on screen. P stands beside the table. | S, P |
| 0:00-0:10 | Judges walk up (this restarts breathing). P asks them to stand still for 20 s and gives the hook (`PITCH.md`, 60-second version). The presence panel says "Person present". It only shows "Someone entering" when someone walks into an empty room, so don't promise a walk-in moment at a busy table. | P |
| 0:10-0:25 | "Person present" on screen; the breathing line returns, then the number. P explains the 43 degrees. | P |
| 0:25-0:40 | 5 s after the number: "Lance, hold your breath." The orb turns red and counts. "Breathe." | P, S |
| 0:40-0:50 | Gesture last: the trained person swipes right (left hand, left to right). The next slide slides in. | trained person |
| 0:50-1:00 | Close: on-device, numbers only, the honest score. | P |
| 1:00-4:00 | Questions (`JUDGE_QA.md`). Offer the judge the chair: "Sit here for 20 seconds and watch your own breathing." Offer, don't push, the breath-hold. For gestures, show the swipe first, then let them try: people copy what they just saw. | P |
| End | S stands up and walks 2 m away; P presses 5 to return to the live slide; glance at the HUD (green, no "Simulated data"). | P, S |

### On stage (3 minutes, if you are a finalist)

| Time | What happens |
|---|---|
| Before you're called | Laptop on a table on stage, chair at the 0.6 m mark, projector set to Duplicate (Win+P). Start the engine **on stage** (the room changed, so it must re-learn it), then the slides. Slide 1. |
| 0:00-1:12 | Slides 1-4, keyboard (`PITCH.md`). S stands 2 m away. |
| 1:12-2:05 | Slide 5. P steps back to 2 m; S sits. "Person present", the breathing line, the rate, 5 s, the breath-hold, the alarm, breathe. Then the swipe to slide 6. |
| 2:05-3:00 | Slides 6-9 by swipe if the first one worked, otherwise keyboard. |

---

## 5. Fallbacks for each failure

Rule for all of them: **one fix attempt in front of judges, then the backup.** Say what you are switching to.

### A. Before judging

| What you see | Likely cause | Fix, in order |
|---|---|---|
| A1. Engine or self-test says it **could not open the audio devices** (device busy) | Another app holds the speaker or microphone; exclusive control not allowed; a Bluetooth or USB device took over | Close the apps in section 2. Check "Allow exclusive control" on **both** devices. Unplug USB and Bluetooth audio. Run again. Then add `--duplex`. Then list the devices with `.venv\Scripts\python -m sonar.engine --list-devices` and name them: `--in-device "Microphone Array" --out-device "Speaker"`. Last resort before judging: reboot (about 3 minutes). |
| A2. Self-test lock FAIL, or the engine's status shows `locked False` / re-syncs climbing (status lines print only when the engine runs with `-v`; otherwise press D on the slides and read `locked` and `resyncs` in the status messages) | Audio dropouts: CPU busy, or on battery | Charger in, close heavy apps (browsers with many tabs, Docker, OneDrive sync). `--blocksize 3840` is already on. Run again. |
| A3. SNR far below last night (under 50 dB in the self-test, or "Weak echo signal" on screen) | Volume changed, wrong device, shared mode | Volume back to `[VOLUME %]`. Check the device names printed. Never use `--shared` for the demo. |
| A4. Someone hears a whine | Volume too high for young ears | Lower the volume one step, then re-run the self-test and keep the new level for everything. |
| A5. `WARNING: the model was trained with different settings` | Model and `--profile` don't match | Use the profile the model was trained with, or `--physics` (only if you tested it tonight). |
| A6. Page says "Sensor offline" | Engine not running, or it crashed | Look at the engine window for `ERROR`. Start it again. The page reconnects by itself. |

### B. During a judge visit

| What you see | Do | Say |
|---|---|---|
| **B1. No breathing number.** The orb keeps showing "Hold still · N of 10 s" and N falls back to 0, or "Measuring" for more than 30 s. | Something within 2 m is moving: judges, the presenter's hands, a passer-by. Ask everyone to freeze. S: hands in lap, back against the chair, steady breaths a little deeper than normal, chest 0.5-0.6 m from the screen. Wait 20 s more. At 45 s with no number, twice in a row: go to the backup video's breathing shot, or replay the recording (below). | "Anything moving nearby restarts the measurement, which is the presence detector doing its job. Let's all hold still for twenty seconds." Then, if needed: "Here is the same thing recorded on this laptop last night." |
| B2. The rate is clearly wrong (for example double what S is breathing) | Ask S to breathe slower and deeper and keep still. The number only settles after three readings agree. | "Count with me, watch his shoulders. It corrects itself once three readings agree." |
| B3. No alarm 25 s into the breath-hold | The hold probably started before the 5-second arming, or S moved (a swallow, a shift). Let S breathe, wait for the number plus 5 s, try once more. | "It needs five steady seconds before it arms. Let's do that again." |
| B4. The alarm fires while S is breathing | Shallow breathing looks flat. S takes two deeper breaths. | "That's a false alarm, and it's why this is a prototype and not a medical device." |
| B5. A swipe does nothing | Too far, too slow, or too small. A miss usually shows nothing, because the engine drops weak or unsure movements before sending them. Press → and carry on. Later, press D: the newest status message's `last_decision` says what the engine made of the movement (`none`, `blocked:far`, `blocked:weak`). | "It didn't catch that one. I'll use the keyboard." |
| B6. A swipe goes the wrong way, or gestures fire on their own | Press the right arrow key. If it happens again, press **L** to pause gesture navigation for the rest of the visit. | "Wrong way. This is why we score the model on people it has never seen." |
| B7. Gestures keep firing while people walk around the table | Restart the engine with `--min-confidence 0.7` between visits. | |
| B8. Presence never says "Room empty" in a busy hall | Normal: movement within 2 m counts, and so does big movement farther out. Skip the walk-in moment and just point at "Person present". | |
| B9. Mid-demo "Sensor offline", or the engine window shows `ERROR` | Restart the engine (Up arrow, Enter, in its window). Everyone still for 5 s. | The calibration line from section 3. |

### C. Bigger failures

| What happens | Do |
|---|---|
| C1. Audio can't be opened after two tries (about a minute) | Backup video. Then offer the replay or the mock UI walkthrough if the judge wants more. |
| C2. The laptop dies, blue-screens or starts an update | Backup video on a phone or the Mac straight away. Then the mock UI on the Mac (below), labelled as simulated. If Lance's MacBook has passed its own self-test and has its own model, it is a **live** backup (`TRAINING.md` section 9): `.venv/bin/python -m sonar.engine --in-device MacBook --out-device MacBook`. It has one microphone, so swipe direction relies on its two speakers. |
| C3. Projector problems on stage | Turn the laptop toward the judges and keep going. The laptop screen is the demo. |

### The three backups, and when to use each

| Backup | Use it when | Command | What the screen shows | What you must say |
|---|---|---|---|---|
| **Backup video** | The laptop or its audio is down; breathing won't lock twice in a row; you're out of time | Play the file from the phone or Mac | the video | "This is a recording from last night on this laptop." |
| **Replay of a real recording** through the real engine | The laptop works but the hall is too busy for a live breathing lock | Stop the live engine (Ctrl+C), then:<br>`.venv\Scripts\python -m sonar.engine --profile clean --backend file --file data\[REPLAY SESSION]\audio.wav`<br>It plays once, in real time, and the engine stops at the end. Use the profile the session was recorded with (its `session.json` says which). | the slides, with the "Simulated data" tag (any source other than live is tagged) | "This is a recording from this laptop last night, played through the same engine. Live is what you saw a moment ago." |
| **Mock engine** | Only to walk through the UI (alarm flow, monitor) when nothing real is possible | Stop the live engine first (both use port 8765), then:<br>`.venv\Scripts\python app\serve.py --mock` (on the Mac: `.venv/bin/python app/serve.py --mock`) | the slides with "Simulated data" | "This is our interface on simulated data, so you can see the alarm flow. It is not the sensor." |

The mock's status panel reports older signal settings (18.0/18.5 kHz tones). If you want it to match the demo laptop, start it on its own with `.venv\Scripts\python app\mock_engine.py --tones 18500 --chirp 19500,21500` and run `.venv\Scripts\python app\serve.py --app` in a second window.

---

## 6. Honesty rules

- A replay or the mock is never called live. Both are tagged on screen, and you say it as well.
- Quote only `[HONEST SCORE]` for gesture accuracy: the score on held-out people, never the presenter, and never a number from people in the training set.
- If a judge asks something you haven't measured, say "we haven't measured that" and what you would measure.
- The Care Monitor says "Prototype · not a medical device". Say the same if asked.

## 7. After judging

Ctrl+C the engine (it fades the signal out without a click), close `serve.py`, and switch exclusive control back off if this laptop goes back to normal use.

# Demo runbook for judging

For Lance's MacBook Pro (Apple Silicon): its own speakers and its one microphone. The gesture model was trained on this MacBook and works only on it. Commands run in Terminal from the repo folder, the one with `README.md` (`cd ~/sonar-hack`, or wherever the repo is). The full guide to the app is `SONAR_CARE.md`. The words to say are in `PITCH.md`.

## On one screen

1. Charger in. Speaker volume at `[VOLUME %]` (the level the gesture recordings were made at), and nobody touches it again.
2. One command starts everything: `scripts/sonar_care.sh`. The dashboard must say "Sonar live", with no "Simulated data" tag and no "Dry run" tag.
3. Demo settings: the preset "Slides / PDF" and the no-movement choice "Demo · 15 s".
4. The demo, in order: someone is here, swipe the slides, push twice for help, sit still for the "No movement" alert. Breathing is on screen as an early estimate. Nothing depends on it.
5. One driver for the slides. At the table: a PDF or Keynote deck, and Sonar Care presses the keys. On stage: the web deck, with Sonar Care's gesture control switched off. With both on, one swipe can move two slides.
6. If the same thing fails twice, switch to the backup and say so out loud (section 5). Never present a recording or simulated data as live.

---

## 1. Tonight: rehearse and measure

Do this on the MacBook, at the demo table if you can. Four rows are marked **first time**: when the docs were written, those parts had passed their tests but had never run on the real MacBook.

| Check | How | Result |
|---|---|---|
| Speaker volume used | The volume slider in Control Centre | `[VOLUME %]` |
| Signal strength | Dashboard > Sensor card. 35 dB or more reads "good signal". Under 25 dB reads "weak signal". | `[SNR ON THE MACBOOK]` dB |
| A key press reaches another app (**first time**) | Settings > Touch-free control > "Try in 3 s" next to Swipe right, then click into the deck. The slide must change. | yes / no |
| Swipes, the presenter | Deck in front. 10 swipes right, 10 swipes left. | right ___/10, left ___/10, wrong way ___ |
| Swipes, someone new | Show them once, then 10 swipes right. | ___/10 |
| A single push | Deck in front. 10 single pushes. Each should press Space about 2.5 s later. Count the ones that showed "Help gesture seen" by mistake (expect about 1 in 12). | pressed ___/10, false help ___/10 |
| A deliberate double push (**first time**, never measured) | Dashboard in front. 10 tries, about a second apart. Count the ones that show "Help gesture seen". The Setup's practice step shows the gap it measured. | ___/10, usual gap ___ s |
| The help alert (**first time**) | After one double push: 3 s countdown, then the red screen, the banner, the chime, and the laptop says "Help requested". Press Acknowledge. The timeline shows "Help requested: acknowledged after N s". | banner yes/no, chime yes/no, voice yes/no |
| "No movement" | "Demo · 15 s" set. Sit still at 0.6 m, hands down, talking. Stopwatch from your last movement to the banner. Then move a hand: it should resolve. Three tries. | ___ s, ___ s, ___ s |
| What restarts the 15 seconds | While the Room card counts "Still for N s": T shifts their feet a step back; someone leans over the table; P talks with hands down. Does the count restart? | ___ / ___ / ___ |
| False gestures | 2 minutes of the presenter talking normally at 0.6 m, hands moving, nobody gesturing. Count the key presses. | ___ |
| Breathing (optional) | Sit still and silent at 0.6 m for a minute. Does a number appear, and does it hold? This decides whether anyone reads a number aloud, and whether the breath-hold extra in `PITCH.md` is allowed. | number yes/no, steady yes/no |
| The menu bar item (**first time**) | It shows "◉ Sonar". Gesture control and Room monitoring switch on and off from it. | yes / no |
| The 90-second demo with a timer | `PITCH.md`, five times, with someone playing the judge. | done |
| The 3-minute pitch with a timer | `PITCH.md`, twice, with the web deck and gesture control off. | ___ : ___ |

Also tonight:

- Do the first run of `scripts/sonar_care.sh` where the internet works. It installs packages (about a minute). Venue Wi-Fi is not a plan.
- Run the Setup once from start to end (menu bar > Run Setup…), practice round included.
- Record the backup video (`BACKUP_VIDEO.md`). Copy it to two phones **as a local file**.
- If a **first time** row fails, tell the team before the Devpost text is submitted. `DEVPOST.md` describes those parts as working.

---

## 2. Setup checklist (T-60 to T-30 min)

### macOS (once, 10 minutes)

- [ ] Quit Zoom, Teams, FaceTime, Discord, Spotify, Music and every browser tab that plays sound. Any of them can hold the microphone or the speakers, and loud audio through the same speakers can weaken the signal.
- [ ] Audio MIDI Setup (Applications > Utilities): the MacBook Pro microphone and the MacBook Pro speakers both at **48,000 Hz**.
- [ ] Sound output and input: the MacBook's own speakers and microphone. No AirPods, no Bluetooth speaker, no headphones, no USB audio.
- [ ] Volume at `[VOLUME %]`, not muted. The slider sets how loud the sonar plays. If the alert chime is too loud or too quiet, change Settings > How alerts are announced > Alert volume, never the system volume.
- [ ] Quit Messages, Mail, Slack and anything else that can pop up or make a sound. Leave Focus (Do Not Disturb) **off**: with Focus on, the "Help requested" banner is hidden, and only the chime, the voice and the red screen get through.
- [ ] System Settings > Notifications > Script Editor: allowed. That is the name macOS files Sonar Care's banner under. It may only appear in that list after the first test alert.
- [ ] Power: plugged in. Display set to never turn off on the power adapter (System Settings > Lock Screen). Do not close the lid.
- [ ] Chrome installed. It is the tested browser, and the dashboard opens in a Chrome app window.
- [ ] A projector over HDMI is fine: the engine picks the MacBook's own speakers and microphone by name. Plug it in **before** starting, mirror the display, then check the dashboard still says "Sonar live" and the signal strength has not dropped.

### The table (5 minutes)

- [ ] Laptop near the front edge of the table, screen facing the chair, lid at the angle used in the recordings. If anyone moves the laptop or the lid later, press Calibrate (Sensor card, or the menu) and keep still for 5 s.
- [ ] Tape a mark on the table so that the presenter's chest sits about **0.6 m** from the screen.
- [ ] A solid chair that doesn't swivel or roll.
- [ ] Nothing that moves in front of the laptop: no fan, no air vent blowing across the table, no flapping sign. Still objects are fine; the engine removes still echoes.
- [ ] A mark for T, a step back, more than a metre from the screen. T steps in only to press Acknowledge.
- [ ] Tape over the webcam. It is part of the story.

### Software (5 minutes)

0. **Get the latest version and start it.**

   ```bash
   git pull
   scripts/sonar_care.sh
   ```

   The first run creates `.venv` and installs what is missing. macOS asks once for the Microphone permission for the terminal app: allow it. Leave the Terminal open. Ctrl+C there, or Quit Sonar Care in the menu, stops everything.

1. **Read what the terminal printed.** It lists the dashboard (`http://127.0.0.1:8800/care.html`), the slides, the engine and the log files. It must not print `DRY RUN` or `KEYS BLOCKED`. If port 8800 was busy it took the next free one: use the address it printed.

2. **First run only: the Setup opens.** Go through it: the signal check, calibration (keep still for 5 s, hands away, nobody walking past), the Accessibility permission (System Settings > Privacy & Security > Accessibility: switch on the terminal app) and the practice round for each gesture. On the last page choose "Slides / PDF" and "15 seconds".

3. **Check the dashboard.**
   - Top bar: a green "Sonar live". No "Simulated data" tag, no "Dry run" tag.
   - Room card: "Someone is here" with a distance when you sit at the mark, and "Monitoring on".
   - Touch-free control card: the chip says "On" (not "Keys blocked", "Dry run", "Practice" or "Off"), and the preset is "Slides / PDF".
   - Sensor card: "This laptop · its own speakers and one microphone", and a signal near `[SNR ON THE MACBOOK]` dB.
   - Settings > This device: "Engine running", model `models-candidate-mac-calm50`.

4. **Check the demo settings** (Settings tab; every change saves at once).
   - Touch-free control: "Gestures press keys" on, "Slides / PDF".
   - Room monitor: Monitoring on, the "No movement" alert on, the quick choice "Demo · 15 s".
   - Help gesture: on. Second push within 2.5 s, but not sooner than 1.0 s. Time to cancel on screen 3 s. A single push "Waits for the window".
   - How alerts are announced: notification, sound and "Say it aloud" all on. Press "Play a test alert": you should see the banner and hear the chime and the voice.

5. **Open the deck for touch-free control.** A PDF or Keynote deck in its own window, not full screen, so you can click between it and the dashboard. Press the Right and Left arrow keys yourself first: the page must change. If the keyboard does not change the page, Sonar Care's key press will not either. Then press "Try in 3 s" (Settings > Touch-free control) and click into the deck.

### Which deck, where

| Where | Slides | Sonar Care's gesture control | Why |
|---|---|---|---|
| At the table: the 90-second demo | A PDF or Keynote deck in its own window. The web deck stays closed. | **On**, preset "Slides / PDF" | It shows the real product: the gesture presses a real key in another app. |
| On stage: the 3-minute pitch | The web deck: `http://127.0.0.1:8800/presenter.html?ws=ws://127.0.0.1:8765` (the terminal prints the address) | **Off** (menu bar > Gesture control) | The web deck changes slides on a swipe by itself. With gesture control also on, the deck reacts to the gesture and to the arrow key Sonar Care presses, so one swipe can move two slides. |

The help gesture and the "No movement" alert belong to Room monitoring. They work in both rows.

If you open the web deck at the table (a judge wants the numbers on slide 8), drive it with the keyboard, or switch gesture control off first.

### Web deck keys

| Key | Does |
|---|---|
| → / ← (also Space for next) | next / previous slide |
| 1-9, 0 | jump to slide 1-10 |
| O | live panel on or off (the same as a push at the deck) |
| C | open the Sonar Care dashboard in a new tab |
| F, T | full screen, light/dark |

The ten slides: 1 Title · 2 The problem · 3 The insight · 4 How it works · 5 Live demo · 6 Sonar Care · 7 Healthcare impact · 8 Under the hood (with the honest score) · 9 What's next · 10 Team.

On the dashboard, **T** switches light and dark and **F** is full screen.

---

## 3. What to say while it calibrates or waits

- **Engine start, or Calibrate (5 s):** "It's listening to the room for five seconds, learning what nothing moving sounds like, the way you'd zero a scale. If everyone keeps still for a moment, we're ready."
- **The 3-second help countdown:** "It waits three seconds so a mistake can be cancelled on the screen. A hand can never cancel it."
- **A single push that acts late (2.5 s):** "A single push waits two and a half seconds, in case a second push follows and it is a call for help."
- **The 15 seconds of stillness:** "Everything that happened is in this timeline. This card is an early breathing estimate: it shows a dash unless it is sure, because a wrong number is worse than none."
- **The engine restarting:** "The sensor restarted itself. It needs a few seconds, then five more to re-learn the room."

---

## 4. Order of the demo, minute by minute

### At the table (one judge visit, 3-5 minutes)

| Time | What happens | Who |
|---|---|---|
| Before they arrive | Dashboard in front. The deck open in its own window. P seated at the mark, T a step back. With "Demo · 15 s" the "No movement" alert also comes up whenever P sits still while waiting. That is normal: press Acknowledge, or move. When judges walk up, move, and press Dismiss if a finished alert is still showing (it clears by itself after 20 s). | P, T |
| 0:00-0:15 | P gives the hook (`PITCH.md`, 90-second version) and points at the Room card: "Someone is here" and about how far. | P |
| 0:15-0:35 | P clicks into the deck. Swipe right: next slide. Swipe left: back. | P |
| 0:35-0:55 | P clicks an empty part of the dashboard and pushes twice, about a second apart. "Help gesture seen", 3 s to cancel, then the chime, "Help requested" spoken, the red screen. T steps in, presses Acknowledge, steps back. The timeline shows how long it took. | P, T |
| 0:55-1:20 | P sits still, hands down, and talks about the timeline and the early breathing estimate. After 15 s the "No movement" banner and chime arrive. P moves a hand: it resolves by itself. | P |
| 1:20-1:30 | Close: no camera, nothing to wear, no new hardware. It runs on the laptop's CPU. Nothing leaves the laptop. | P |
| 1:30-4:00 | Questions (`JUDGE_QA.md`). Offer the judge the chair: "Swipe the slide, then push twice." Show the gesture first, then let them try: people copy what they just saw. Tell them first that the model was trained on three of us, so a new hand is recognised less often. | P |
| End | Acknowledge or dismiss anything on screen. Deck back to its first page. Glance at the top bar: green "Sonar live", no "Simulated data". | P, T |

### On stage (3 minutes, if you are a finalist)

| Time | What happens |
|---|---|
| Before you're called | Laptop on a table on stage, chair at the 0.6 m mark, projector mirrored. Press Calibrate once the laptop is in place (the room changed), or start Sonar Care there if it was stopped. Menu bar: Gesture control **off**, Room monitoring on. Open the web deck on slide 1 and press F. |
| 0:00-1:07 | Slides 1-4 by keyboard. P stands back, more than about 1.3 m from the laptop, and steps in to press →. Sitting still in front of the laptop for 15 s would raise "No movement" in the middle of the slides. |
| 1:07-1:35 | Slide 5: P sits, points at the presence reading, then one swipe right to slide 6. One line on slide 6, then **C** opens the dashboard. |
| 1:35-2:25 | Dashboard: two pushes, 3 s, the alarm, T acknowledges. Then P sits still for 15 s: "No movement". P stands up: it resolves. |
| 2:25-3:00 | Back to the deck's tab (F again if it left full screen, O if the live panel is open). Slides 7-10 by keyboard, standing. |

---

## 5. Fallbacks for each failure

Rule for all of them: **one fix attempt in front of judges, then the backup.** Say what you are switching to.

### A. Before judging

| What you see | Likely cause | Fix, in order |
|---|---|---|
| A1. After `git pull`, the shell says there is no `scripts/sonar_care.sh` | The checkout is on a branch without Sonar Care | `git branch --show-current` shows the branch. Sonar Care was built on `sonar-care`: switch to it unless the team has merged it elsewhere. |
| A2. Terminal says `no gesture model found` | The model's weight file did not arrive | `git pull` again. To rehearse the screens meanwhile: `scripts/sonar_care.sh --mock`. |
| A3. Dashboard says "The sensor could not start", with an audio error | Another app holds the microphone or the speakers, or the Microphone permission was refused | Quit Zoom, Teams, FaceTime and browser tabs that play sound. System Settings > Privacy & Security > Microphone: the terminal app on. Then menu bar > Start engine. |
| A4. "Weak signal" (under 25 dB), or far below `[SNR ON THE MACBOOK]` | Volume changed, headphones or Bluetooth audio connected, a sound enhancer switched on | Volume back to `[VOLUME %]`. Headphones out, Bluetooth audio off. Then Calibrate. |
| A5. Someone hears a whine | The lowest tone is 18.5 kHz, and a few young people can hear that high | Lower the volume one step only if you must. Then Calibrate and repeat ten swipes: the model was recorded at the old level. Keep the new level for everything. |
| A6. The Touch-free control chip says "Keys blocked" | The Accessibility permission is missing | System Settings > Privacy & Security > Accessibility: switch on the terminal app (Settings has an "Open Accessibility settings" button). The page notices by itself. |
| A7. The chip says "Dry run" | Started with `--dry-run` | Ctrl+C, then start again without it. |
| A8. The top bar shows "Simulated data" | Started with `--mock` or `--backend sim` | Ctrl+C, then start again with plain `scripts/sonar_care.sh`. |
| A9. The page says "Sonar Care is not running" | The page was opened without the app behind it | Start `scripts/sonar_care.sh`, then reload. |
| A10. No menu bar item | The menu bar library did not install | The app runs without it. The dashboard's Settings do everything the menu does. |
| A11. No banner on "Play a test alert" | Focus is on, or notifications for Script Editor are off | Turn Focus off. System Settings > Notifications > Script Editor. The chime and the voice work without the banner. |

### B. During a judge visit

| What you see | Do | Say |
|---|---|---|
| B1. A swipe does nothing | Press the arrow key and carry on. Never retry twice. Afterwards, the "Last:" line in the Touch-free control card says what the app made of it (pressed, or ignored with the reason). If it shows nothing new, the engine saw no gesture: closer, a hand-length from the screen, half a second across. | "It didn't catch that one. I'll use the keyboard." |
| B2. The row lights up but the slide doesn't move | The deck is not the app in front: click into it. Or the chip says "Keys blocked" (A6). | |
| B3. A swipe goes the wrong way | Press the right arrow key. | "Wrong way. Misses like that are in the score we report." |
| B4. One swipe moves two slides | Press ← once. Between visits: if the web deck was in front, switch gesture control off. Otherwise the same swipe was reported twice: Settings > Touch-free control > Fine tuning > raise "Same gesture again". | |
| B5. Help does not trigger | Push twice again, slower: a full second between the two. There is no "Send now" to press: that button only appears once the gesture has been seen. | "Two pushes, a full second apart. Once more." |
| B6. "Help gesture seen" after a single push | Press Cancel during the 3 seconds. | "About one single push in twelve looks like two. That is what the three seconds are for." |
| B7. The help chime keeps coming back | Nobody acknowledged: it repeats every 20 s. Press Acknowledge. | |
| B8. No "No movement" after 20 s of stillness | Something near the laptop is moving, or the Room card says "Checking…" or "Room is empty" instead of "Someone is here", or "Demo · 15 s" is not set. Move on. Check between visits. | "It starts its count again whenever something moves near the laptop." |
| B9. "No movement" comes up early | You sat still for 15 s. Move a hand: it resolves. | "That is the no-movement alert. I sat still for fifteen seconds." |
| B10. Gestures fire on their own while people move around the table | Between visits: Settings > Touch-free control > Fine tuning > raise "How sure the model must be" (the default is 60 %). | |
| B11. "Sensor offline" or "Restarting the sensor" | Keep talking. The app restarts the engine by itself: after 1 s the first time, with longer pauses (2, 4, 8, 15 s) if it keeps crashing. Then it calibrates for about 5 s: everyone still. If it says "The sensor could not start", menu bar > Start engine. | The restart line from section 3. |

### C. Bigger failures

| What happens | Do |
|---|---|
| C1. The engine cannot start after two tries (about a minute) | Backup video. Then offer the mock walk-through if the judge wants more. |
| C2. The MacBook dies or restarts | Backup video on a phone straight away. Then the mock on any other Mac that has the repo, labelled as simulated. There is no second live machine: the gesture model only fits this MacBook. |
| C3. Projector problems on stage | Turn the laptop toward the judges and keep going. The laptop screen is the demo. |

### The backups, and when to use each

Stop the live app first (Ctrl+C). If it is still running, a second start only attaches to its engine.

| Backup | Use it when | Command | What the screen shows | What you must say |
|---|---|---|---|---|
| **Backup video** | The MacBook or its audio is down; the same part failed twice; you're out of time | Play the file from a phone | the video | "This is a recording from last night on this MacBook." |
| **Mock engine** | Only to walk through the screens (help alert, "No movement", timeline) when nothing real is possible. It runs on any Mac. | `scripts/sonar_care.sh --mock` | The dashboard marked "Simulated data", with an 80-second scripted story on a loop. On the dashboard the arrow keys, P, H, M and E fire a swipe, a push, a double push, a movement and arrive/leave. | "This is our interface on simulated data, so you can see the alert flow. It is not the sensor." |
| **Simulated visitor** | To show the real engine running with no sound | `scripts/sonar_care.sh --backend sim` | The Sensor card says "Simulated visitor · the real engine, no sound". | "This is the real engine fed by a simulated person. It is not live." |

`scripts/sonar_care.sh --dry-run` is for a first look. The sensor is live and the dashboard shows which key each gesture would press, but nothing is pressed.

---

## 6. Honesty rules

- Simulated data is never called live. It is tagged on screen, and you say it as well.
- Quote only the held-out gesture score: 133 of 206 gestures recognised (65%) and 22 of 499 other movements triggering (4.4%), with each of three people held out of training in turn. The best-matched person was 70 of 81 (86%). Never quote that one alone.
- Never promise a breathing number. Say "early estimate".
- Say "No movement alert". Do not say "fall detection".
- If a judge asks something you haven't measured, say "we haven't measured that" and what you would measure. The deliberate double push is one of those until you fill in section 1.
- The dashboard says "Prototype · not a medical device". Say the same if asked.

## 7. After judging

Ctrl+C in the Terminal, or Quit Sonar Care in the menu. The engine it started stops too, and it fades the signal out without a click.

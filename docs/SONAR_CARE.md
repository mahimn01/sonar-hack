# Sonar Care

Sonar Care turns a MacBook's own speakers and microphone into a touch-free, camera-free sensor for patients and
caregivers. The speakers play sound too high to hear (18.5 kHz from the left speaker, 19.0 kHz from the right, and
a 19.5-21.5 kHz sweep), one microphone listens to the echoes, and the app turns them into four things:

| | What it gives you |
|---|---|
| **Touch-free control** | A hand gesture presses a real key in whatever app is in front: swipe right = next, swipe left = previous, push = space (or media keys, or scrolling, or your own choice). |
| **Room monitor** | A caregiver dashboard: is someone there and how far, an early breathing-rate estimate, a timeline, and a "No movement" alert when someone is present but still for too long. |
| **Help gesture** | Two pushes in a row = "Help requested": three seconds to cancel on screen, then a full-screen alert, a notification, a chime and the words spoken aloud. The time it took to acknowledge is recorded. |
| **Guided setup** | A first-run check (signal, calibration, permission) and a practice round for each gesture with live feedback. |

No camera. No sound is recorded and nothing leaves the laptop: the engine's only output is small messages such as
"swipe right, 93 % sure" on a connection only this computer can reach.

The three gestures are **swipe left, swipe right and push**. The trained model has no other classes.

- [1. Run it](#1-run-it-on-the-macbook-from-a-fresh-pull)
- [2. Permissions](#2-permissions-macos-asks-once)
- [3. Using it](#3-using-it)
- [4. The 90-second demo](#4-the-90-second-demo-for-judges)
- [5. Troubleshooting](#5-troubleshooting)
- [6. How it is built](#6-how-it-is-built)
- [7. The decisions, and why](#7-the-decisions-and-why)
- [8. For developers](#8-for-developers)
- [9. Known limits](#9-known-limits)

---

## 1. Run it on the MacBook, from a fresh pull

```bash
cd ~/sonar-hack            # wherever the repo is
git pull
scripts/sonar_care.sh
```

That one command:

1. uses the repo's `.venv` (creates it if it is missing) and installs only the packages that are missing;
2. starts the sonar engine with the `calm50` model (`models-candidate-mac-calm50`; it falls back to
   `models-release-mac-recall` if the weight file is missing) and restarts it if it ever crashes;
3. puts a Sonar Care item in the menu bar;
4. opens the dashboard at `http://127.0.0.1:8800/care.html`. The first time, it opens the Setup.

No Xcode, no code signing, no build step. Stop with Ctrl+C in the terminal or **Quit Sonar Care** in the menu;
the engine it started stops too.

Before you start: speaker volume where it was when the model's recordings were made (do not change it afterwards),
nothing else playing sound, headphones unplugged, charger in. In **Audio MIDI Setup** the MacBook microphone and
speakers should be at 48,000 Hz.

Other ways to start it:

| Command | What it does |
|---|---|
| `scripts/sonar_care.sh --dry-run` | Shows which key each gesture WOULD press. Presses nothing. Good for a first look. |
| `scripts/sonar_care.sh --mock` | No sonar at all: scripted fake data (an 80-second story on a loop), marked "Simulated data". Rehearse every screen on any Mac. In the dashboard the arrow keys, `P`, `H`, `M` and `E` fire a swipe, a push, a double push, a movement and arrive/leave. |
| `scripts/sonar_care.sh --backend sim` | The real engine with its simulated visitor and no sound. |
| `scripts/sonar_care.sh --attach ws://127.0.0.1:8765` | Use an engine that is already running. Sonar Care starts and stops none. |
| `scripts/sonar_care.sh --headless` | No menu bar item (also what happens by itself if the menu bar library is missing). |
| `scripts/sonar_care.sh --quiet` | No notification, chime or voice for alerts. |
| `scripts/sonar_care.sh --lan` | Other devices on the same network may open the dashboard and acknowledge alerts (nothing else). |
| `scripts/sonar_care.sh --help` | Every option (`--engine-port`, `--port`, `--scene`, `--engine-args`, `--settings`, `--browser`, `--no-browser`, `-v`). |

If an engine is already listening on the engine port (8765), Sonar Care uses it and starts nothing. Settings are
saved in `~/Library/Application Support/SonarCare/settings.json`; logs go to `~/Library/Logs/SonarCare/`
(`engine.log`, `events.jsonl`).

The pitch slides keep working while it runs: `http://127.0.0.1:8800/presenter.html?ws=ws://127.0.0.1:8765`
(the terminal prints the exact address; `C` on the slides opens the dashboard).

## 2. Permissions (macOS asks once)

| Permission | Who gets it | Needed for | Where |
|---|---|---|---|
| **Microphone** | the terminal app you start it from | everything (the sonar listens) | macOS asks on the first start. Later: System Settings > Privacy & Security > Microphone |
| **Accessibility** | the same terminal app | pressing keys in other apps (touch-free control only) | System Settings > Privacy & Security > Accessibility. The Setup, the Settings page and the menu each have an "Open Accessibility settings" button |
| **Notifications** | "Script Editor" (that is how macOS files notifications sent this way) | the alert banner; the chime and voice work without it | System Settings > Notifications |

Without Accessibility the room monitor and the help gesture work fully; gestures are recognised and shown, and
the app says "Keys blocked" instead of pretending. The page notices by itself when the permission is granted.

## 3. Using it

**Menu bar.** The title shows the state at a glance (`◉ Sonar`, `◌ Calibrating`, `◍ Weak signal`, `○ Sonar off`,
`▲ HELP`, `▲ No movement`). The menu shows the sensor, presence, breathing and the last gesture, and has: Gesture
control on/off, Room monitoring on/off, what gestures control, Open Dashboard / Settings / Setup, Calibrate,
Acknowledge, Cancel help request, Start / Stop engine, Quit.

**Dashboard** (`#/`). Room (is someone there, how far, how long still, with the countdown to the no-movement
alert), Breathing (an early estimate; a dash when the engine is not confident; the line under it is the raw chest
movement), Touch-free control (what each gesture does right now; rows light up as gestures arrive), Timeline,
Sensor (signal strength, the live Doppler strip, Calibrate). `T` switches light and dark, `F` full screen.

**Settings** (`#/settings`). Every change is saved and applied at once. Presets and a key per gesture with
"Try in 3 s", the alerts and their thresholds, the help gesture's timing, how alerts are announced, the room name
and the engine. The experimental fall alert is switched on here too (section 10); it is off by default.

**Setup** (`#/setup`). Run it again any time from the menu or from Settings.

**How to gesture.** Open hand, a hand-length (20 to 25 cm) from the screen.

- *Swipe left / right*: sweep across the screen in about half a second, then let the hand drop to your lap. Do not
  swipe back.
- *Push*: palm quickly toward the screen and straight back, in one motion.
- *Help*: push, bring the hand back, push again about a second later.

## 4. The 90-second demo for judges

Before: `scripts/sonar_care.sh` running, the Setup done once, Settings: **Slides / PDF** and the no-movement
choice **Demo · 15 s**, a PDF or Keynote deck open in another window, you seated about 0.6 m from the laptop,
camera taped over, a teammate at the trackpad as the "caregiver".

| Time | Do | Say |
|---|---|---|
| 0:00 | Point at the dashboard: "Someone is here · about 0.6 m away". | "This MacBook's camera is taped over. Its speakers are playing sound too high to hear, and its one microphone is listening to the echoes. That makes it a sensor." |
| 0:15 | Click into the slides. Swipe right, swipe right, swipe left. | "Hands full, gloves on, or cannot reach the keyboard: a swipe in the air presses the key." |
| 0:35 | Back to the dashboard. Push twice, about a second apart. Wait for the 3-second cancel time. The laptop chimes and says "Help requested"; the screen turns red. The teammate presses Acknowledge. | "If I need help: two pushes. It gives three seconds to cancel a mistake, then it calls. It logs how long the answer took." |
| 0:55 | Sit still, hands down, and talk. After 15 s the "No movement" banner and chime arrive. Then move a hand: it resolves by itself. | "It also knows when someone is here but has stopped moving. And this is an early breathing estimate; when it is not sure it shows a dash, because a wrong number is worse than none." |
| 1:20 | | "No camera, nothing to wear, no new hardware. It runs on this laptop's processor and nothing leaves the laptop." |

If something misses:

| What happens | Do |
|---|---|
| A swipe is not seen | Press the arrow key and keep talking. Never retry twice. |
| Help does not trigger | Push twice again, slower: a full second between the two. |
| A single push raises help by mistake | Press Cancel during the three seconds. Say why the cancel time exists (about 1 single push in 12 looks like two). |
| No "No movement" after 20 s | Something near you is moving, or the 15 s choice is not set. Move on. |
| "Sensor offline" | The app restarts the engine by itself within seconds. Keep talking. |

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| The page says "Sonar Care is not running" | It was opened without the service. Start `scripts/sonar_care.sh` and reload. |
| "The sensor could not start" with an audio error | Another app holds the microphone or speakers. Close Zoom, Teams, FaceTime and browser tabs that play sound, then **Start engine** in the menu. Check the Microphone permission (section 2). |
| "no gesture model found" | The model's weight file did not arrive. `git pull`, or rehearse with `--mock`. |
| "Weak signal" | Volume to about 70 %, headphones out, any "sound enhancer" off, then **Calibrate**. The sound stays inaudible. |
| Gestures are recognised but nothing happens | Is the chip in the Touch-free control card "Keys blocked" (grant Accessibility), "Dry run" (start without `--dry-run`), "Off" (switch it on) or "Paused" (a help alert is open)? Is the right app in front? |
| A push acts late | By design: with the help gesture on, a single push waits 2.5 s in case a second push follows. Settings > Help gesture > "Acts at once" changes it. |
| One swipe moves two slides | The same swipe was reported twice. Settings > Fine tuning > raise "Same gesture again". |
| Help triggers on a single push | Bring the hand straight back in one motion. Settings > Help gesture > raise "But not sooner than". |
| Help never triggers | Leave about a second between the two pushes. Practise in Setup: it shows the gap it measured. |
| "No movement" never fires | Presence must say "Someone is here" (not "Checking…"). With "Detected breathing counts as movement" on, a tracked breathing rate prevents it: that is intended. |
| No notification banner | Notifications for "Script Editor" are off, or a Focus mode is on. The chime and voice still play. Settings > "Play a test alert" checks it. |
| No menu bar item | The menu bar library is missing: the app runs without it (same as `--headless`). The dashboard does everything the menu does. |
| The port is busy | The app takes the next free port and prints the address it used. |
| Music is playing through the laptop | Loud sound through the same speakers can weaken the sonar's signal. Watch the signal strength in the Sensor card; use headphones for the music. |

Logs: `~/Library/Logs/SonarCare/engine.log` (the engine's own output) and `events.jsonl` (every timeline line).
Start with `-v` to print every decision.

## 6. How it is built

```
 speakers + mic                     this app (one Python process)                         you
┌──────────────┐  events   ┌───────────────────────────────────────────────┐
│ sonar engine │──────────▶│ client.py ─▶ rules.py ─▶ service.py           │──▶ key presses (actions.py)
│ (sonar/)     │ WebSocket │  reads        decides     carries out         │──▶ notification, chime, voice (notify.py)
└──────────────┘           │                                               │──▶ timeline (memory + events.jsonl)
       ▲                   │ relay.py: engine events + care events, one    │──▶ dashboard in the browser
       │ starts, watches,  │           stream for the page                 │◀── settings / acknowledge (api.py)
       └── restarts ───────│ supervisor.py                    menubar.py   │──▶ menu bar item
                           └───────────────────────────────────────────────┘
```

| File (in `app/care/`) | What it does |
|---|---|
| `settings.py` | Every user choice in one validated JSON file. |
| `rules.py` | Every decision, as pure logic with no clock and no I/O: gesture filters, the help gesture, "No movement", "No breathing movement", alert life cycle. |
| `client.py` | The connection to the engine. Reconnects by itself (0.25 s doubling to 2 s). |
| `relay.py` | The one WebSocket the page uses: engine messages passed through, plus Sonar Care's own. |
| `api.py` | Local web server: the pages and a small JSON API. |
| `service.py` | Wires it together on one background loop. |
| `actions.py` | Presses keys through macOS CoreGraphics (loaded with `ctypes`: nothing to install or sign). Media keys through AppKit when pyobjc is there, otherwise by asking Spotify or Music. A dry run logs only. |
| `notify.py` | Notification banner, chime, then the alert's title spoken. Rate-limited. |
| `supervisor.py` | Starts the engine, attaches if one is already running, restarts it after a crash (pauses of 1, 2, 4, 8, 15 s; after 6 instant deaths it stops and shows the engine's last error), stops it politely. |
| `menubar.py` | The menu bar item (rumps). Without rumps the app runs headless. |
| `main.py`, `app/sonar_care.py`, `scripts/sonar_care.sh` | The command line and the one-command launcher. |

The page is `app/web/care.html` with `css/care.css` and `js/care*.js`: one page, three views, so an alert shows
wherever you are. It uses the same design tokens as the slides (`css/tokens.css`).

Two rules of the house:

- **Alerts are decided in one place** (`rules.py`), so they fire the same way whether or not a browser is open.
  The older Care Monitor page (`monitor.html`) keeps its own built-in no-breathing alarm and is unchanged.
- **Only the adapter knows the wire format** (`app/adapter.py` in Python, `app/web/js/adapter.js` in the
  browser), for the engine's messages and for Sonar Care's own. Both are checked against the same examples in
  `app/tests/fixtures/engine_events.json`.

## 7. The decisions, and why

### Gestures that press keys

A gesture acts only if it passes four checks, on top of the engine's own:

| Check | Default | Why |
|---|---|---|
| sure enough | 0.6 | the model's confidence, 0 to 1 |
| not too soon | 0.7 s after the last gesture that acted | one movement, one action |
| not the hand coming back | opposite swipe ignored for 1.5 s | in the recordings the return stroke arrived as early as 1.24 s |
| not the same movement twice | same gesture ignored for 1.3 s | see below |

Presets: **Slides / PDF** (right arrow, left arrow, space), **Media** (next, previous, play/pause), **Scroll**
(page down, page up), **Custom** (any key per gesture, or off). There is no volume or mute action on purpose: the
sonar plays through the same speakers at a fixed volume, and changing it would change the sensor's own signal.

### One push is often reported twice (measured)

On 2026-11-15 the 13 recorded sessions (three people, the calm50 model, the engine's own decision logic) were
replayed and every prompt's events counted. The per-person totals matched `models-candidate-mac-calm50/metrics.json`
exactly, so the replay is the same thing the packaged numbers came from.

- Of 69 prompted single pushes, **28 (41 %) produced two `push` events**. The model has no pull class, so the hand
  coming back also looks like a push. The gaps were 0.40-0.98 s in 27 of 33 pairs (median 0.68 s).
- With "a second push event sooner than 1.0 s is the same push", **6 of 69 (about 1 in 12)** single pushes still
  look like a double push: the hand was pulled back after a pause, which is the same movement as a second push.
  Timing alone cannot remove these.
- Swipes almost never produce a push (1 of 138), and the same swipe repeats on 19 of 138 prompts.
- Nothing fired in 21.8 minutes of typing, reaching, walking and sitting.
- Not measured: a deliberate double push. Nobody has recorded one, so how reliably it is caught is unknown until it
  is tried on the demo MacBook. The Setup's practice step shows the gap it measured, and `min_gap_s` and `window_s`
  are settings for exactly that reason.

### The help gesture and the push key

A push can mean "press the push key" or the first half of "help". The decision:

1. **A single push presses its key only after the help window (2.5 s) has passed** with no second push. One
   physical action therefore never does both. (Settings > "Acts at once" presses immediately instead; a double
   push then presses once and also raises help.) Swipes always act immediately.
2. **Two pushes = help.** The second push must arrive between 1.0 s and 2.5 s after the first.
3. **A 3-second cancel time.** Because about 1 single push in 12 can look like two, a help request first shows on
   screen for 3 s with Cancel and Send now, then goes out (alert, notification, chime, voice).
4. **A hand can never cancel a help request.** Only a person at the screen can. Another push sends it at once.
   A real call must never be dropped by accident.
5. While a help alert is open, gestures press nothing, and another push repeats the request.

### "No movement"

Raised when someone is present and no sign of life has been seen for the set time (60 s by default; the demo
choice is 15 s). Signs of life are: body movement the engine reports, any gesture, walking, and by default a
confidently tracked breathing rate.

- **It needs a confirmed person.** The engine must report "present" with at least 60 % confidence twice in a row.
  A noisy empty room can report a faint "present" (5 % confidence was seen on a desktop Mac); that never arms it,
  and the dashboard says "Checking…" for it.
- **The demo choice does not let breathing cancel the alert**, because breathing detection is experimental and the
  alert has to be showable on stage. The alert then says "Breathing is still detected" when it is.
- It resolves by itself when movement returns or the person leaves, and stays visible for 20 s so a caregiver
  still sees that it happened. Acknowledging it while the person is still keeps it quiet until they move.
- An engine outage or a calibration is never counted as stillness.
- **The alert's own sound cannot cancel it.** A loud sound through the same speakers can look like movement to the
  sonar, so movement seen while the chime and voice play is ignored.

### "No breathing movement"

The same logic as the Care Monitor page, moved into the service: after a steady rate has been seen for 5 s on a
still person, the alert fires when breathing movement stops for 10 s (the clinical length of an apnea). It never
fires within 6 s of a gesture or body movement. Breathing is experimental on one microphone.

### "Possible fall" (experimental, off by default)

One sudden one-way movement followed by stillness. It is a prototype rule checked on recordings of ordinary
movement, never on a real fall. What it looks for, what the recordings showed, what it gets wrong and how to tune
it are in section 10.

### Alert life cycle

`pending` (help only, cancellable) → `active` (notification, chime, voice; red alerts repeat the chime every 20 s)
→ `acknowledged` and/or `resolved` → put away. Every step is a line in the timeline; a help request records the
seconds from going out to being acknowledged.

### Running it

- **One process.** The service loop owns the rules and the settings, so nothing is decided twice. The menu bar
  runs on the main thread of the same process; without it the app is identical minus the menu.
- **Ctrl+C, Quit and closing the terminal window all stop the engine too.** If the app is killed hard, the engine
  keeps running; the next start recognises it (by the pid it wrote down) so Stop and Quit work on it again.
- **The local API only answers the app's own pages.** A POST must be JSON from the same origin, so another
  website open in the browser cannot change settings or make the app press keys.

## 8. For developers

### Messages Sonar Care adds to the stream

The relay passes engine messages through unchanged (`INTERFACES.md`) and adds these. All have `type` and `t`.

| `type` | When | Main fields |
|---|---|---|
| `care_hello` | once, when a page connects | `version`, `dry_run`, `mode`, `gestures`, `timeline` (recent entries) |
| `care_settings` | on connect and on every change | `rev`, `settings` |
| `care_state` | once a second and after every decision | `engine`, `link`, `sonar`, `presence`, `breathing`, `stillness`, `help`, `control`, `monitoring`, `alerts`, `last_gesture`, `practice` |
| `care_action` | for every gesture | `gesture`, `confidence`, `result`, `reason`, `action`, `action_label`, `delayed_s`, `help_until` |
| `care_alert` | on every alert change | `change` (`pending`, `raised`, `updated`, `reminder`, `acknowledged`, `resolved`, `dismissed`), `alert` |
| `care_timeline` | for every timeline line | `entry` (`id`, `t`, `kind`, `level`, `icon`, `text`, `detail`) |
| `fall` | only while the experimental fall alert is on: at each step of a possible fall | `stage` (`candidate`, `confirmed`, `dismissed`, `moved`), `burst_db`, `burst_level`, `direction`, `stillness_s`, `confidence`, `reason`, `seq` (full list in `INTERFACES.md`) |

`care_state` also carries `fall`: `enabled`, `sensitivity` and, while it is on, `level` (movement right now),
`slow_level`, `watching` (the stillness count), `last_burst` (what the last big movement looked like and why it did
or did not count) and the `thresholds` in use.

`care_action.result` is one of: `sent`, `dry_run`, `blocked`, `failed` (a key press was attempted); `held` (a push
waiting out the help window); `armed` (a first push with nothing to press); `help`; `paused` (a help alert is open);
`ignored` (with `reason`: low confidence, too soon, return stroke, same movement, same push, alert sound playing,
unsupported gesture); `off`, `unmapped`; `practice`.

### Local API

| | |
|---|---|
| `GET /api/state` | the latest `care_state` |
| `GET /api/settings` | settings, plus the lists the settings page draws from |
| `POST /api/settings` | change settings (a partial JSON object); a bad value is refused with the reason |
| `POST /api/ack` | acknowledge alerts: `{"id": "help-3"}`, or `{}` for all |
| `POST /api/help/cancel`, `/api/help/send` | cancel, or send at once, a help request that is counting down |
| `POST /api/calibrate` | recalibrate (about 5 s, keep still) |
| `POST /api/practice` | `{"active": true}`: tutorial mode, nothing is pressed or raised |
| `POST /api/engine` | `{"action": "start" \| "stop" \| "restart"}` |
| `POST /api/accessibility` | open System Settings on the Accessibility page |
| `POST /api/test-action` | `{"action": "right", "delay_s": 3}`: press one key after a delay |
| `POST /api/test-alert` | announce a harmless test alert |
| `GET /api/timeline` | recent timeline entries |
| `GET /api/boot.js` | tells a page where the relay is |

### Tests

```bash
.venv/bin/python -m pytest -m "not slow" -q     # core engine + app + Sonar Care (about 2 minutes)
node --test app/tests/                          # the browser code: adapters, store, live client
```

| File | What it checks |
|---|---|
| `app/tests/test_care_rules.py` | Every decision above with a fake clock, including the same no-breathing scenarios the web monitor's own tests use. |
| `app/tests/test_care_settings.py` | Defaults, validation, saving. |
| `app/tests/test_care_actions.py` | Key codes, dry run, the blocked state, notification commands and their rate limits. Presses nothing. |
| `app/tests/test_care_supervisor.py` | Engine commands per mode, attach, start, crash and restart, giving up, adopting a left-over engine; the menu's contents. |
| `app/tests/test_care_service.py` | End to end: the real service against the mock engine: gesture to (dry-run) key press, the help countdown, cancel, acknowledge, no movement, settings, practice mode, refusing other websites. |
| `app/tests/test_care_fall.py` | The experimental fall alert: synthetic bursts (fires on a one-way burst plus stillness; not on a fidget, a swipe, a push and its return, a recognised gesture, movement afterwards, walking past, sitting down, a jump, an empty room), six windows of real recorded rows (`fixtures/fall_real_rows.json`), and the alert it raises. |
| `app/tests/store.test.js`, `adapter.test.js` | The page's side: the service's alerts drive the page; both adapters agree on 56 shared examples. |

## 9. Known limits

- **One laptop, three people.** The model was trained on Lance's MacBook Pro with recordings of three people. On
  other speakers and microphones it is not expected to work. With each person held out of training in turn it
  recognised 133 of 206 gestures (65 %) with 22 of 499 other movements triggering (4.4 %); on the best-matched
  held-out person 70 of 81 (source: the recall session's summary, 2026-11-14).
- **Never tried on the MacBook yet:** a real key press reaching another app, the media keys, the notification
  banner with chime and voice, the menu's look, and a deliberate double push. All of them are covered by tests of
  the logic and of the commands; none has run against real hardware at the time of writing.
- **Breathing is experimental** on one microphone, and the no-breathing alert inherits that.
- **The fall alert is a prototype rule** (section 10): off by default, never tried on a real fall or on the
  MacBook live, and it fires when someone steps away from the laptop and nothing else moves.
- **About 1 single push in 12 looks like a double push.** The cancel time is the answer; it is not zero.
- **Chrome is the tested browser** (the dashboard opens in a Chrome app window when Chrome is installed). The
  page needs a 2023 or newer browser (container queries, `color-mix()`).
- **Not a medical device.** A prototype built in a weekend.

## 10. The experimental fall alert

**Off by default. A prototype rule, not a tested fall detector: no real fall has been recorded with it.** Switch it
on in Settings > Fall alert (experimental). When it fires, the caregiver sees a red "Possible fall" banner tagged
"Experimental fall alert", with the same notification, chime, voice and timeline line as the other alerts. It stays
until someone acknowledges it. If movement comes back first, the alert says so and stays up anyway.

### What it looks for

A fall, seen by sonar, is one sudden movement that goes one way and does not come back, followed by stillness.
The rule reads the engine's Doppler strip (the moving picture on the Sensor card: 64 speed bins, 50 rows a second;
111 Hz of Doppler shift is 1 m/s) and needs all of this:

1. a burst of movement lasting 0.4 to 2.5 s;
2. big: it lights up a quarter of the strip or more (size 0.25 on a 0 to 1 scale);
3. with a fast part: energy beyond 100 Hz, about 0.9 m/s;
4. one way, toward or away, counted from 1.5 s before the burst to 1.2 s after it. A hand comes back, sitting down
   goes down and back, walking past goes toward and then away: those score low;
5. the gesture model did not call it a gesture;
6. no train of smaller movements right after it (a jump lands and recovers in steps);
7. then stillness for 15 s (settable 5 to 60 s), slow shifting included;
8. and the engine does not report the room empty when the time is up (that is someone leaving).

A change in the person's measured distance across the burst raises the confidence number and is never required.
Stillness alone never raises this alert; that is what "No movement" is for. When "Possible fall" goes out, an open
"No movement" alert is folded into it, so one event gives one alert.

### What real recordings showed (and how they changed the rule)

The rule was checked against the 12 recorded MacBook sessions of three people (35.8 minutes: prompted gestures,
walking in and out, sitting still, breathing, moving about). Nobody fell, so every alert there is a false alarm.

| Finding | Number |
|---|---|
| A still room on the strip | size 0.001 (median) |
| Fidgeting while sitting | 0.11 to 0.12 (median burst) |
| A person walking in or out | 0.27 to 0.38 |
| A hand gesture at the laptop | 0.45 to 0.51 |

- **A hand 30 cm from the laptop is bigger on the strip than a body 1.5 m away.** "Bigger than a gesture" does not
  exist on this sensor, so size only removes fidgeting. What tells a gesture from a fall is direction: swipes go
  both ways (one-way score 0.1 to 0.2) and a push comes back within a second.
- **The engine's distance zones and its "moving" flag are not usable here.** Near-zone and far-zone movement read
  the same for a push and for walking, and the "moving" flag was on for 61 % of the time while people sat still. So
  the rule takes its stillness from the strip itself: the slowest bins (8 to 23 Hz, 0.07 to 0.2 m/s) read 0.000
  when someone sits still and 0.14 when they shift about.
- **False alarms in the 35.8 minutes** (484 movement bursts, 37 of them big, fast and one-way):

  | Stillness setting | Low | Medium | High |
  |---|---|---|---|
  | 15 s (default) | 0 | 0 | 0 |
  | 10 s | 0 | 1 | 1 |
  | 5 s | not run | 2 | not run |

  Every one of them is a person stepping away from the laptop and then standing still.
- **Gesture, then freeze.** Each of the 174 recorded gesture prompts (54 pushes, 108 swipes, and 12 pull-backs
  from an early recording script) was replayed
  followed by 20 s in which nothing moves, without help from the gesture model: 0 alerts. The same for 8 walk-ins:
  0. For 8 walk-outs: 2 alerts.

Those recordings keep people moving every 5 to 10 s, so "0 at 15 s" is weaker than it looks: nobody in them walked
away and stayed away for 15 s.

### What it gets wrong

- **Fires on** someone stepping quickly away from the laptop (or up to it) who then stays out of view or stands
  still for the whole stillness time. To this sensor that is a one-way burst followed by stillness. Also: a bag
  dropped in front of the laptop that then lies still; a hand pushed toward the laptop and held there when the
  model did not recognise the push.
- **Misses** a fall side-on to the laptop or straight down (little speed toward or away from it), a fall behind
  furniture or far from the laptop (the echo gets too weak; how far it reaches has not been measured, and presence
  itself is tuned for about 1.3 m), a slow slide to the floor (no fast burst), a fall after
  which the person keeps moving, and a fall in a busy room (other people's movement breaks the stillness).

### Why it runs in the Sonar Care service and not in the engine

Everything it needs is already in the engine's event stream at full frame rate. So the engine is not touched:
gestures, presence and breathing cannot be disturbed, it works with an engine that is already running, and every
threshold changes from the Settings page while it runs. An engine-side detector would see the same Doppler picture
plus the distance zones, and the recordings showed those do not help on this hardware.

### Trying it and tuning it on the MacBook (10 minutes, nobody falls)

1. Start Sonar Care (`scripts/sonar_care.sh`, or with `--dry-run`). Open Settings > **Fall alert (experimental)**,
   switch it on and set **Stillness before the alert** to 10 s while you try things.
2. Read the box under the switch. **Movement level now** should sit at 0.00 to 0.02 in a still room. If it stays
   above 0.05, something is moving in view (people nearby, a fan): the stillness can never complete there.
3. Stand 1 to 1.5 m in front of the laptop. Drop a pillow or a bag from chest height in front of you, or take one
   fast step toward the laptop. Then keep still.
4. The box now shows **Last big movement** with its size, fast part, one-way score and duration, and why it did or
   did not count:

   | The box says | What to change |
   |---|---|
   | not big enough (level 0.18, needs 0.25) | Sensitivity to High, or "Size needed" to just under the level shown |
   | not fast enough | lower "Fast part needed" |
   | too short (0.3 s; needs 0.4 s) | lower "Shortest movement", or drop from higher up |
   | too long | it lasted over 2.5 s: that is walking; raise "Longest movement" only if a real test needs it |
   | moved both ways | the thing bounced back or you stepped back. Leave "One-way score needed" at 0.35 or above: lower lets gestures through |
   | counted … dropped: movement was seen again | something moved during the stillness time; keep still |
   | counted … dropped: slow movement was seen again | someone is shifting about in view |
   | possible fall: still for 10 s | the alert went out |

   On the dashboard the Room card counts the stillness ("Sudden movement, then still for 6 s").
5. Check that it stays quiet: swipe and push at the laptop, walk past, sit down. The box should say "moved both
   ways", "too long" or "the model recognised it as a hand gesture". Then step away from the laptop and stand
   still: it will fire. That is the known weak spot, and the reason for the longer default.
6. For a demo use 10 to 15 s of stillness and tell the audience to keep still. Switch it off afterwards.

The live numbers are also in the stream: `care_state.fall` once a second and a `fall` event at each step (`D` on
the dashboard shows both). The six thresholds above are saved in `settings.json` under `monitoring.fall`; the
finer ones (where a burst starts and ends, what counts as quiet) are the `DEFAULTS` table in `app/care/fall.py`.
Without the sonar, `scripts/sonar_care.sh --mock` and the `G` key on the dashboard play a scripted fall.

### Sources and next step

Starting values came from: Lian et al., "Fall Detection via Inaudible Acoustic Sensing", IMWUT 2021,
<https://doi.org/10.1145/3478094>; Erol et al., SPIE 2018,
<https://homepage.villanova.edu/moeness.amin/paper_pdf/Baris_SPIE_2018.pdf>; Erol and Amin, EUSIPCO 2016,
<https://www.eurasip.org/Proceedings/Eusipco/Eusipco2016/papers/1570252332.pdf> (one-sided Doppler, extreme
Doppler as a first cut, the distance change as a supporting cue, a 10 to 20 s stillness timer, rejecting a burst
followed by smaller recovery peaks). The next step is real validation: staged falls onto a mattress with this
hardware, and the University of Glasgow radar dataset (<https://researchdata.gla.ac.uk/848/>) for the shape of
falls against sitting, bending and walking.

# Sonar Care

> **Draft (phase 1 of 5).** The service core described here runs and is tested. The menu bar app, the real key
> presses, the dashboard, settings and onboarding pages are being built in phases 2 to 4; the sections marked
> *(planned)* describe the agreed design, not finished code. Setup steps, troubleshooting and the 90-second demo
> script are written in phase 5, once they can be checked on the real pages.

Sonar Care turns a MacBook's own speakers and microphone into a touch-free, camera-free sensor for patients and
caregivers. The speakers play sound too high to hear (18.5 kHz from the left speaker, 19.0 kHz from the right, and
a 19.5-21.5 kHz sweep), one microphone listens to the echoes, and the app turns them into four things:

| | What it gives you |
|---|---|
| **Touch-free control** | A hand gesture presses a real key in whatever app is in front: swipe right = next, swipe left = previous, push = space (or media keys, or scrolling, or your own choice). |
| **Room monitor** | A caregiver dashboard: is someone there, an early breathing-rate estimate, and a "No movement" alert when someone is present but still for too long. |
| **Help gesture** | Two pushes in a row = "Help requested": a full-screen alert, a notification and a sound, with the time it took to acknowledge. |
| **Guided setup** | A first-run check (signal, calibration, permission) and a short practice round for each gesture. *(planned)* |

No camera. No audio is stored or leaves the laptop: the engine's only output is small messages such as
"swipe right, 93 % sure" on a connection only this computer can reach.

The three gestures are **swipe left, swipe right and push**. There is no pull and no wave: the trained model does
not have those classes.

---

## How it is built

```
 speakers + mic                     this app (one Python process)                         you
┌──────────────┐  events   ┌───────────────────────────────────────────────┐
│ sonar engine │──────────▶│ client.py ─▶ rules.py ─▶ service.py           │──▶ key presses (actions.py)
│ (sonar/)     │ WebSocket │  reads        decides     carries out         │──▶ notification + sound (notify.py)
└──────────────┘           │                                               │──▶ timeline (memory + events.jsonl)
       ▲                   │ relay.py: engine events + care events, one    │──▶ dashboard in the browser
       │ starts, watches   │           stream for the pages                │◀── settings / acknowledge (api.py)
       └───────────────────│ supervisor.py (planned)   menubar.py (planned)│
                           └───────────────────────────────────────────────┘
```

| File (in `app/care/`) | What it does | State |
|---|---|---|
| `settings.py` | Every user choice in one validated JSON file. | done, tested |
| `rules.py` | Every decision, as pure logic with no clock and no I/O: gesture filters, the help gesture, "No movement", "No breathing movement", alert life cycle. | done, tested |
| `client.py` | The connection to the engine. Reconnects by itself (0.25 s doubling to 2 s). | done |
| `relay.py` | The one WebSocket the dashboard uses: engine messages passed through, plus Sonar Care's own. | done, tested |
| `api.py` | Local web server: the pages and a small JSON API. | done, tested |
| `service.py` | Wires it together on one background loop. | done, tested |
| `actions.py` | Presses keys. Phase 1 has the dry run only (logs "would press Right arrow"). | dry run done |
| `notify.py` | macOS notification + sound for an alert. Phase 1 records what it would show. | stub |
| `supervisor.py` | Starts the engine, restarts it if it crashes, attaches if one is already running. | *(planned, phase 2)* |
| `menubar.py` | The menu bar item. | *(planned, phase 2)* |
| `main.py` + `app/sonar_care.py` | The command line. | done |

Two rules of the house:

- **Alerts are decided in one place** (`rules.py`), so they fire the same way whether or not a browser is open.
  The older Care Monitor page (`monitor.html`) keeps its own built-in no-breathing alarm and is unchanged.
- **Only the adapter knows the engine's message format** (`app/adapter.py` in Python, `app/web/js/adapter.js` in
  the browser). Both are checked against the same examples in `app/tests/fixtures/engine_events.json`.

### Run the phase 1 service

It attaches to an engine that is already running (starting the engine itself comes with the supervisor).

```bash
cd ~/Documents/Dev/sonar-hack-care
# 1. an engine: the simulated visitor, with the calm50 model
#    (--min-peak-db 3 because the simulator's hand is much quieter than a real one; never use it live)
SONAR_MODEL_DIR=models-candidate-mac-calm50 scripts/run_best_mac.sh --backend sim --port 8771 --min-peak-db 3
# 2. Sonar Care, pressing nothing
.venv/bin/python app/sonar_care.py --headless --engine-port 8771 --dry-run --quiet --no-browser --port 8801 -v
# 3. look at it
curl -s http://127.0.0.1:8801/api/state | python3 -m json.tool | head -40
```

`--dry-run` logs the key each gesture would press and presses nothing. `--quiet` means no notification and no
sound. Without `--settings PATH`, settings live in `~/Library/Application Support/SonarCare/settings.json` and the
event log in `~/Library/Logs/SonarCare/events.jsonl` (set `SONAR_CARE_HOME` to move both).

---

## The decisions, and why

### Gestures that press keys

A gesture acts only if it passes four checks, on top of the engine's own:

| Check | Default | Why |
|---|---|---|
| sure enough | 0.6 | the model's confidence, 0 to 1 |
| not too soon | 0.7 s after the last gesture that acted | one movement, one action |
| not the hand coming back | opposite swipe ignored for 1.5 s | in the recordings the return stroke arrived as early as 1.24 s |
| not the same movement twice | same gesture ignored for 1.3 s | see below |

Presets: **Slides / PDF** (right arrow, left arrow, space), **Media** (next, previous, play/pause), **Scroll**
(page down, page up), **Custom** (any key per gesture, or off).

### One push is often reported twice (measured)

On 2026-11-15 the 13 recorded sessions (4 people, the calm50 model, the engine's own decision logic) were replayed
and every prompt's events counted. The per-person totals matched `models-candidate-mac-calm50/metrics.json`
exactly, so the replay is the same thing the packaged numbers came from.

- Of 69 prompted single pushes, **28 (41 %) produced two `push` events**. The model has no pull class, so the hand
  coming back also looks like a push. The gaps were 0.40-0.98 s in 27 of 33 pairs (median 0.68 s).
- With "a second push event sooner than 1.0 s is the same push", **6 of 69 (about 1 in 12)** single pushes still
  look like a double push: the hand was pulled back after a pause, which is the same movement as a second push.
  Timing alone cannot remove these.
- Swipes almost never produce a push (1 of 138) and the same swipe repeats on 19 of 138 prompts.
- Nothing fired in 21.8 minutes of typing, reaching, walking and sitting.
- Not measured: a deliberate double push. Nobody has recorded one, so how reliably it is caught is unknown until it
  is tried on the demo MacBook. `min_gap_s` and `window_s` are settings for exactly that reason.

### The help gesture and the push key

A push can mean "press the push key" or the first half of "help". The decision:

1. **A single push presses its key only after the help window (2.5 s) has passed** with no second push. One
   physical action therefore never does both. (Setting `push_timing: "instant"` presses at once instead; a double
   push then presses once and also raises help.) Swipes always act immediately.
2. **Two pushes = help.** The second push must arrive between 1.0 s and 2.5 s after the first.
3. **A 3-second cancel time.** Because about 1 single push in 12 can look like two, a help request first shows on
   screen for 3 s with a Cancel button, then goes out (alert, notification, sound).
4. **A hand can never cancel a help request.** Only a person at the screen can. Another push sends it at once.
   A real call must never be dropped by accident.
5. While a help alert is open, gestures press nothing, and another push repeats the request.

### "No movement"

Raised when someone is present and no sign of life has been seen for the set time (default 60 s; the demo setting
is 15 s). Signs of life are: body movement the engine reports, any gesture, walking, and by default a confidently
tracked breathing rate.

- **It needs a confirmed person.** The engine must report "present" with at least 60 % confidence twice in a row.
  A noisy empty room can report a faint "present" (5 % confidence was seen on a desktop Mac); that never arms it.
- **The demo setting does not let breathing cancel the alert**, because breathing detection is experimental and
  the alert has to be showable on stage. The alert then says "Breathing is still detected" when it is.
- It resolves by itself when movement returns or the person leaves, and stays visible for 20 s so a caregiver
  still sees that it happened. Acknowledging it while the person is still keeps it quiet until they move.
- An engine outage or a calibration is never counted as stillness.
- **The alert's own sound cannot cancel it.** A loud sound through the same speakers can look like movement to the
  sonar, so movement seen while the sound plays is ignored.

### "No breathing movement"

The same logic as the Care Monitor page, moved into the service: after a steady rate has been seen for 5 s on a
still person, the alert fires when breathing movement stops for 10 s (the clinical length of an apnea). It never
fires within 6 s of a gesture or body movement. Breathing is experimental on one microphone; the screen shows a
dash when the engine is not confident.

### Alert life cycle

`pending` (help only, cancellable) → `active` (notification + sound; red alerts repeat the sound every 20 s) →
`acknowledged` and/or `resolved` → put away. Every step is a line in the timeline; a help request records the
seconds from going out to being acknowledged.

---

## For developers

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
| `POST /api/help/cancel` | cancel a help request that is still counting down |
| `POST /api/calibrate` | recalibrate (about 5 s, keep still) |
| `POST /api/practice` | `{"active": true}`: tutorial mode, nothing is pressed or raised |
| `GET /api/timeline` | recent timeline entries |
| `GET /api/boot.js` | tells a page where the relay is |

The server listens on 127.0.0.1 only. A POST must be JSON from one of the app's own pages, so another website open
in the browser cannot change settings or make the app press keys.

### Tests

```bash
.venv/bin/python -m pytest -m "not slow" -q     # core engine + app + Sonar Care (about 2 minutes)
node --test app/tests/                          # the browser code: adapter, store, live client
```

- `app/tests/test_care_rules.py`: every decision above, with a fake clock, including the same no-breathing
  scenarios the web monitor's own tests use.
- `app/tests/test_care_settings.py`: defaults, validation, saving.
- `app/tests/test_care_service.py`: the real service against the mock engine: gesture to (dry-run) key press, the
  help countdown, cancel, acknowledge, no movement, settings, practice mode, and refusing other websites.

---

## Design direction for the screens *(planned, phases 3 and 4)*

One product, one design system: the dashboard, settings, onboarding, launcher and the pitch deck all use
`app/web/css/tokens.css` (daylight and night themes, Atkinson Hyperlegible Next for text, Lexend for digits,
teal "tide" for live data, amber for attention, red only for alerts).

- **Calm first.** A big, plain statement of presence ("Someone is here · still for 40 s"), large numerals, generous
  space. Colour means state: green ok, amber check on them, red act now. Nothing flashes.
- **Honest readings.** Breathing carries an "early estimate" tag and shows a dash when the engine is unsure.
  Simulated data is always labelled.
- **Readable from two metres**, keyboard accessible, both themes at WCAG AA contrast, reduced-motion respected,
  no web fonts or scripts from the internet (it works offline).
- **Alerts in three weights:** a help request takes the whole screen with one large Acknowledge button; "No
  movement" is a wide banner with Acknowledge; a resolved alert is a quiet line that can be dismissed.
- **One page, three views** (`care.html`: Dashboard, Settings, Setup), so an alert shows wherever you are.

## What is not done yet

- The engine is not started or supervised by the app (phase 2). Phase 1 attaches to a running one.
- Keys are not really pressed; notifications and sounds are recorded, not shown (phase 2).
- No dashboard, settings or onboarding pages yet (phases 3 and 4). `care.html` does not exist: use `--no-browser`.
- The help gesture has never been tried with a deliberate double push on the demo MacBook.
- Breathing on one microphone is experimental; the no-breathing alert inherits that.

# App layer: Sonar Care, the pitch slides and the Care Monitor

Everything a person sees or touches. Plain HTML/CSS/JS with no build step, plus a small Python service. All of it
reads the sonar engine's events from a local WebSocket (`ws://127.0.0.1:8765`).

| File | What it is |
| --- | --- |
| `care/` + `sonar_care.py` | **Sonar Care**, the product: the service behind the menu bar item and the dashboard (touch-free control, room monitor, help gesture). Full guide: `docs/SONAR_CARE.md`. |
| `web/care.html` | The Sonar Care page: Dashboard, Settings and Setup. |
| `web/presenter.html` | 10 pitch slides changed by gestures, with a corner HUD and a live panel. Slide 5 is the live demo, slide 6 is Sonar Care. |
| `web/monitor.html` | The earlier single-screen Care Monitor: breathing rate and waveform, presence, gestures, Doppler strip, alarms. |
| `web/index.html` | Launcher that links to all three. |
| `web/js/adapter.js` | **The only place in the UI that knows the wire format** (the engine's messages and Sonar Care's own). |
| `adapter.py` | The same adapter in Python, used by the service and the tests. |
| `serve.py` | Tiny static server for the slides and the monitor alone (standard library only). |
| `mock_engine.py` | Fake engine: scripted scenarios or keyboard control, flagged as "Simulated data" in the UI. |
| `keys_bridge.py` | The earlier stand-alone gesture-to-arrow-key bridge. Sonar Care replaces it (and reuses its gesture filter). |
| `tests/` | Adapter fixtures shared by Python and JS, rule tests, and end-to-end tests. |

## Quick start on the MacBook (from the repo root)

```bash
scripts/sonar_care.sh                # everything: engine, Sonar Care, menu bar item, dashboard
scripts/sonar_care.sh --dry-run      # the same, but gestures only show the key they WOULD press
scripts/sonar_care.sh --mock         # no sonar: scripted fake data, to rehearse the screens on any Mac
```

While it runs: the dashboard is `http://127.0.0.1:8800/care.html`, the slides are
`http://127.0.0.1:8800/presenter.html?ws=ws://127.0.0.1:8765` (the terminal prints both). `docs/SONAR_CARE.md` has
the permissions, the 90-second demo and the troubleshooting table.

**Only the slides or the monitor, without Sonar Care:**

```bash
SONAR_MODEL_DIR=models-candidate-mac-calm50 scripts/run_best_mac.sh    # the engine, in one terminal
.venv/bin/python app/serve.py                    # the slides, in another (--view monitor for the Care Monitor)
.venv/bin/python app/serve.py --mock             # or: slides against the mock engine, no sonar needed
```

Press **F** for full screen. The HUD in the bottom-left shows a green dot and "Sonar live" when data is flowing,
and "Simulated data" when it comes from the mock or the simulator.

## Presenter controls

| Gesture | Key | Does |
| --- | --- | --- |
| swipe right | Right, Down, PgDn, Space | Next slide (the slide follows your hand to the right) |
| swipe left | Left, Up, PgUp | Previous slide |
| push | O | Live panel on or off (breathing, presence, waveform, Doppler) over any slide |
| | 1-9, 0 | Jump to slide 1-10 |
| | C | Open the Sonar Care dashboard in a new tab |
| | L | Pause gesture control (gestures still show in the HUD) |
| | F, T, H, D, M, ? | Full screen, light/dark theme, hide HUD, raw event inspector, open the monitor, help |

A gesture only acts if it passes three filters on top of the engine's own: confidence of at least 0.6, a 0.7 s
cooldown, and ignoring the opposite gesture for 1.1 s (the hand coming back after a swipe). Rejected gestures show
greyed in the HUD with the reason.

The slides and Sonar Care can run together. When the slides page is in front and Sonar Care also presses the arrow
key for the same swipe, the page drops that key (it arrives within 0.45 s of the gesture), and it ignores the Space
key Sonar Care presses about 2.5 s after a push. One movement never moves two slides.

## Care Monitor: what the alerts mean

- **No breathing detected** (red banner, red orb counting seconds). Someone is present and still, breathing was
  measured steadily for at least 5 s, and then breathing movement stops for **10 s** (the clinical length of an
  apnea). It never fires within 6 s of a gesture, a presence change or engine-reported motion. Breathing on one
  microphone is experimental.
- **Person entered** (blue, 6 s). Presence went from empty to entering or present.
- **Sensor offline** (amber). No data for 3 s. The page reconnects by itself every 0.25 to 2 s.
- **Calibrating** and **Weak echo signal** (SNR below 15 dB) come from the engine's status messages.

The Sonar Care dashboard does not use this page's built-in alarm: there the service decides every alert.

## The mock engine

```bash
.venv/bin/python app/mock_engine.py --interactive
.venv/bin/python app/mock_engine.py --scenario care      # the 80-second Sonar Care story on a loop
```

Interactive keys: arrows or `a`/`d`/`w` fire swipe left, swipe right and push; Down or `s` fires two pushes (the
help gesture); `x` fires a low-confidence gesture; `1`/`2`/`3` set empty, entering or present; `m` makes the person
move for 2 s; `h` holds the breath; `f` freezes all movement; `+`/`-` change the breathing rate; `c` calibrates;
`g` plays a fall (one fast one-way movement, then nothing moves for 30 s: the experimental fall alert); `q`
quits. Any client can also send JSON commands such as `{"cmd":"gesture","label":"push","confidence":0.9}`,
`{"cmd":"double_push"}`, `{"cmd":"moving","seconds":2}` or `{"cmd":"fall","direction":"toward","still_s":30}`.

It sends the engine's own message shapes (`INTERFACES.md`): three gestures, pilot tones at 18.5 and 19.0 kHz, a
19.5-21.5 kHz chirp, 50 frames/s, 64 Doppler bins, presence with distance, movement flag and stillness counter,
breathing as a whole window once a second.

## Settings in the URL (slides and monitor)

`presenter.html?ws=ws://127.0.0.1:8765&minConf=0.6&cooldown=700&returnMs=1100&theme=dark&slide=5`

`ws`, `minConf`, `cooldown` (ms), `returnMs` (ms), `apneaSec`, `breathConf`, `motionHold` (s), `breathMin` (s),
`lowSnr` (dB), `waveSec`, `theme` (`light`/`dark`), `room` (monitor label), `hud=0`, `debug=1`, `slide`. Without
`ws`, the pages connect to port 8765 on whatever host served them.

## When the engine's messages change

1. Compare `INTERFACES.md` with the alias tables at the top of `web/js/adapter.js` and `adapter.py`. Add new field
   or label names there, add a case to `tests/fixtures/engine_events.json`, and run the tests. Nothing else in the
   UI reads raw fields.
2. Press **D** on any page to see each raw message next to what the adapter made of it. Anything shown as
   `UNKNOWN` needs an alias.

## Tests

```bash
.venv/bin/python -m pytest -m "not slow" -q     # the engine's tests plus everything in app/tests (about 2 minutes)
node --test app/tests/                          # the browser code: adapters, store, live client (Node 22+)
```

## Troubleshooting (slides and monitor; for Sonar Care see docs/SONAR_CARE.md)

- **Blank or half-styled page**: open it through `serve.py` or Sonar Care, not by double-clicking the file.
- **"Sensor offline"**: the engine isn't running or is on another port. The page reconnects by itself. Try `?ws=`.
- **Port 8000 busy**: `serve.py` takes the next free port and prints the address.
- **Gestures show in the HUD but slides don't move**: the HUD says why ("too unsure", "too soon", "return stroke",
  or "paused" after L).
- **No sound from the browser**: by design. The pages never play audio; Sonar Care's alert chime comes from the
  service, not from the page.
- The screen won't dim while a page is visible (wake lock).

## Editing the slides

The slides are `<section class="slide">` blocks in `web/presenter.html`. Team names are near the end (add
`<li><b>Name</b><span>School</span></li>`). Slide 8 shows the model's honest numbers; their source is in the
comment above them. Every statistic on slides 2 and 7 has its source on the slide.

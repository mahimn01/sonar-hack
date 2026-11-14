# Laptop Sonar: app layer

The pitch slides and the live **Care Monitor** for the laptop-sonar demo. Both are plain HTML/CSS/JS with no build step. They read the sonar engine's events from a local WebSocket (`ws://127.0.0.1:8765`), so the slides change when a judge swipes a hand at the laptop.

| File | What it is |
| --- | --- |
| `serve.py` | Tiny static server (standard library only). Opens the slides in the browser. `--mock` also starts the mock engine. |
| `web/presenter.html` | 9 pitch slides driven by gestures, with a corner HUD and a live panel. Slide 5 is the live demo. |
| `web/monitor.html` | Care Monitor: breathing rate and waveform, presence, gestures, activity log, Doppler strip, signal health, alarms. |
| `web/index.html` | Launcher that links to both views. |
| `web/js/adapter.js` | **The only place in the UI that knows the engine's message format.** |
| `adapter.py` | The same adapter in Python, used by `keys_bridge.py` and the tests. |
| `mock_engine.py` | Fake engine: scripted scenarios or keyboard control, flagged as "Simulated data" in the UI. |
| `keys_bridge.py` | Turns gestures into arrow-key presses, to drive PowerPoint or Google Slides instead. |
| `tests/` | Adapter fixtures shared by Python and JS, alert-logic tests, and an end-to-end smoke test. |

## Quick start on the Windows demo laptop (PowerShell, from the repo root)

One-time setup (needed for the mock engine and the key bridge; `serve.py` itself needs nothing):

```powershell
py -3.12 -m venv app\.venv
app\.venv\Scripts\python -m pip install -r app\requirements.txt
```

**Rehearse without the sonar hardware:**

```powershell
app\.venv\Scripts\python app\serve.py --mock                     # 2-minute story: walk in, breathe, gestures, breath-hold, leave
app\.venv\Scripts\python app\serve.py --mock --scenario steady   # person present, a gesture every ~8 s (slide practice)
```

**Live with the real engine:**

1. Start the engine so it serves `ws://127.0.0.1:8765` (see the repo README / INTERFACES.md).
2. `py -3.12 app\serve.py`. Use `--view monitor` for the Care Monitor, or `--app` for a clean Chrome/Edge window without tabs.
3. Press **F** for full screen. Check the HUD in the bottom-left: a green dot with "Sonar live" means data is flowing. If it says "Simulated data", you are on the mock.

Both pages can be open at once (for example slides on the projector and the monitor on the laptop). The engine must then accept more than one WebSocket client.

## Presenter controls

| Gesture | Key | Does |
| --- | --- | --- |
| swipe right | Right, Down, PgDn, Space | Next slide (the slide follows your hand to the right) |
| swipe left | Left, Up, PgUp | Previous slide |
| push | O | Live panel on or off (breathing, presence, waveform, Doppler) over any slide |
| pull / wave | | Shown in the HUD and on the live slide only |
| | 1-9 | Jump to a slide |
| | L | Pause gesture control (gestures still show in the HUD) |
| | F, T, H, D, M, ? | Fullscreen, light/dark theme, hide HUD, raw event inspector, open the monitor, help |

A gesture only acts if it passes three filters on top of the engine's own: confidence of at least 0.6, a 0.7 s cooldown, and ignoring the opposite gesture for 1.1 s (the hand coming back after a swipe). Rejected gestures show greyed in the HUD with the reason, so you can tell "not recognised" from "recognised but ignored".

## Care Monitor: what the alerts mean

- **No breathing detected** (red banner, red orb counting seconds). Someone is present and still, breathing was measured steadily for at least 5 s, and then breathing movement stops for **10 s** (the clinical length of an apnea). It is detected from the waveform going flat, the engine's own apnea flag if it sends one, or valid breathing estimates stopping. It never fires within 6 s of a gesture, a presence change or engine-reported motion, so waving at the laptop cannot set it off.
- **Person entered** (blue, 6 s). Presence went from empty to entering or present.
- **Sensor offline** (amber). No data for 3 s. The page reconnects by itself every 0.25 to 2 s.
- **Calibrating** and **Weak echo signal** (SNR below 15 dB) come from the engine's status messages.

## The mock engine

```powershell
app\.venv\Scripts\python app\mock_engine.py --interactive
```

Interactive keys: arrows or WASD fire gestures; `x` fires a low-confidence one; `1`/`2`/`3` set empty, entering or present; `h` holds the breath (the alert fires about 10 s later); `+`/`-` change the rate; `c` calibrates; `q` quits. Any client can also send JSON commands such as `{"cmd":"gesture","label":"push","confidence":0.9}` or `{"cmd":"apnea","on":true}`.

By default it imitates `sonar/` as written so far: pilot tones at 18 and 18.5 kHz, a 19-21.5 kHz chirp, 50 frames/s, and 64 Doppler bins. Breathing comes once a second as a whole window (10 Hz, up to 30 s) scaled to its own peak, with `state`, `amplitude_mm`, `range_m` and `window_s`. Use `--waveform chunk` to send only new samples instead.

## Driving PowerPoint or Google Slides (`keys_bridge.py`)

```powershell
app\.venv\Scripts\python app\keys_bridge.py --dry-run           # prints what it would press
app\.venv\Scripts\python app\keys_bridge.py                     # click into the slideshow within 3 s
app\.venv\Scripts\python app\keys_bridge.py --map push=b        # push also blanks the screen in PowerPoint
```

It presses keys in whatever window has focus. Don't also drive the web slides with gestures at the same time (the page drops a duplicate key that arrives within 0.45 s, but keep it to one). If PowerPoint runs as administrator, run the bridge as administrator too.

## Settings in the URL

`presenter.html?ws=ws://127.0.0.1:8765&minConf=0.6&cooldown=700&returnMs=1100&theme=dark&slide=5`

`ws`, `minConf`, `cooldown` (ms), `returnMs` (ms), `apneaSec`, `breathConf`, `motionHold` (s), `breathMin` (s), `lowSnr` (dB), `waveSec`, `theme` (`light`/`dark`), `room` (monitor label), `hud=0`, `debug=1`, `slide`. Without `ws`, the pages connect to port 8765 on whatever host served them, so a phone on the same Wi-Fi can open the monitor. That needs `serve.py --host 0.0.0.0` and the engine listening on 0.0.0.0.

## Hooking up the real engine

1. When `INTERFACES.md` is final, compare it with the alias tables at the top of `web/js/adapter.js` and `adapter.py`. Add new field or label names there, add a case to `tests/fixtures/engine_events.json`, and run the tests. Nothing else in the UI reads raw engine fields.
2. Press **D** on either page to see each raw message next to what the adapter made of it. Anything shown as `UNKNOWN` needs an alias.
3. The adapter already copes with: labels in any case or with `-`/space; confidence in 0-1 or percent; timestamps in seconds, milliseconds or relative engine time; `{type, data}` envelopes, batches and arrays; a breathing waveform sent as a whole window or as new chunks; status with a carrier, tones, a chirp band, or per-channel SNR; spectrogram rows in 0-1 or dB; and the engine's breathing states (`tracking`, `acquiring`, `moving`, `no_signal`).

## Tests

```powershell
app\.venv\Scripts\python -m unittest discover -s app\tests -v   # adapter fixtures, bridge filter, end-to-end smoke
node --test app\tests\                                          # same fixtures for adapter.js, alert logic, live client (Node 22+)
```

## Troubleshooting

- **Blank or half-styled page**: open it through `serve.py`, not by double-clicking. The page reloads itself up to 3 times if a script fails to load.
- **"Sensor offline"**: the engine isn't running or is on another port. Fix the engine; the page reconnects by itself. Try `?ws=` if the port differs.
- **Port 8000 busy**: `serve.py` takes the next free port and prints the address.
- **Gestures show in the HUD but slides don't move**: the HUD says why ("too unsure", "too soon", "return stroke", or "paused" after L).
- **No sound from the browser**: by design. The page never plays audio, because the engine needs the speaker in exclusive mode.
- The screen won't dim while the page is visible (wake lock), and the mouse pointer hides after 2.5 s.

## Editing the slides

The slides are `<section class="slide">` blocks in `web/presenter.html`. Team names are near the end (add `<li><b>Name</b><span>School</span></li>`). Slide 7 has a hidden `.metrics` block for model results: fill it in and remove `hidden` once training numbers exist. Every statistic on slides 2 and 6 has its source on the slide.

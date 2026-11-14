# Backup demo video: shot list and narration

A 75-90 second video recorded tonight on the HP. It is the fallback when the live demo can't run (`DEMO_RUNBOOK.md`, section 5), and it can double as the Devpost video. Judges trust an uncut shot more than a montage, so the core of it is **one continuous take** of the laptop and the person together. Close-ups of the screen are inserts on top of that take, never replacements for it.

## What you need

- **Phone A** on a tripod, or propped on books, behind and beside the chair, about 1.5 m back. It must see the laptop screen, the taped webcam and S's chest in one frame. Start recording, then step out of the 2 m zone: the person holding the phone counts as movement.
- **Phone B** (optional), side-on to S's chest at the same height, for a close-up of the shoulders rising and falling.
- **Screen recording** on the HP at the same time: Win+Alt+R (Xbox Game Bar) records the browser window. Start it before S sits down.
- **Narration** recorded afterwards on a phone or the Mac, in a quiet spot. The HP can't record it: the engine holds its microphone.
- **Editing:** Clipchamp (built into Windows 11), iMovie or CapCut. Cuts and captions only.
- The engine and slides started as in the runbook (`--profile clean --blocksize 3840`). The HUD shows "Sonar live" with **no** "Simulated data" tag. That tag must never appear in this video.

## The rules for an honest video

1. Every reading on screen is live from the HP's own speaker and microphones. No mock and no replay. If you ever have to use a replay, caption it "replay of a recording" for as long as it is on screen.
2. The hold and its alarm come from the same take. Never splice an alarm from one take onto a hold from another.
3. The narration only says what the screen shows. If the self-test prints different numbers tonight, the narration doesn't recite them. The screen shows them.
4. First frame caption: "Recorded live on an HP ENVY x360 laptop, [DATE, TIME]. Camera taped over. Cuts only, no edits to the data."

## Shot list

Target 80 s. Shots 2 and 7 can be cut to reach 60 s.

| # | Time | Camera | Screen | Action |
|---|---|---|---|---|
| 1 | 0:00-0:06 | Phone A close on the taped webcam, then pull back (or cut) to the whole laptop on the table | Slide 1 | Nobody moves |
| 2 | 0:06-0:14 | Screen insert: the self-test's results block in PowerShell, zoomed so the lines are readable | Self-test output: SNR lines, lock, drift, audible leak, verdict | Recorded before the main take |
| 3 | 0:14-0:24 | Phone A, the main take starts | Care Monitor (`M` on the slides, or `.venv\Scripts\python app\serve.py --view monitor --app`) | S walks into frame and sits at the 0.6 m mark. Presence changes to "Person present" (it shows "Someone entering" first only if the room was empty). |
| 4 | 0:24-0:42 | Phone A, with the Phone B chest close-up as a picture-in-picture or a 3-second cutaway | The breathing line appears ("Hold still · N of 10 s"), then the rate | S sits still, hands in lap, breathing steadily and a little deeper than normal |
| 5 | 0:42-1:00 | Phone A, same take, no cut | The rate holds for 5 s, then the orb turns red, the banner says "No breathing movement detected" and counts; then the activity log shows "Breathing resumed" | S counts five seconds after the number appears, holds the breath, and breathes again about 5 s after the alarm appears. The counter starts from when breathing stopped, so it already reads about 10 s when the alarm first shows. |
| 6 | 1:00-1:12 | Phone A, or a new take on the slides | Slide 5 or 6 of the deck; the HUD shows the gesture and its confidence | The trained person swipes right with the left hand (the next slide slides in), then pushes (the live panel opens) |
| 7 | 1:12-1:22 | Screen insert of slide 7's live panel, then slide 9 | Echo SNR, frame rate, latency; then the title line | Nobody moves |

## Narration (about 170 words, read at a calm pace)

Timings match the shot list. Record it in one go, then trim the gaps in the editor.

**[1, 0:00]** "This laptop's camera is taped over. It's still going to watch my teammate breathe."

**[2, 0:06]** "Its speakers play sound above 18 kilohertz, higher than most adults can hear. A one-minute self-test checks that it hears its own echoes and that nothing leaks into the range we can hear."

**[3, 0:14]** "Lance sits down. There's no camera, but the echoes show someone is there."

**[4, 0:24]** "Each breath moves his chest a few millimetres. The sound's wavelength is under two centimetres, so one millimetre turns the echo by 43 degrees. That's the line you see. After ten seconds of stillness, it gives a rate."

**[5, 0:42]** "Now he holds his breath. Ten seconds without breathing movement is an apnea, and the alarm goes off. He breathes again, and it clears."

**[6, 1:00]** "The same echoes read hands. A swipe changes the slide. A push opens the live panel."

**[7, 1:12]** "All of it runs on the laptop's own processor and sends numbers, never audio or images. Laptop Sonar: camera-free care, from the laptop already in the room."

## Before and after recording

- [ ] Quiet corner, nobody walking within 2 m, no fan blowing across the table.
- [ ] Notifications off, volume at `[VOLUME %]`, charger in.
- [ ] Three full takes. Keep the best one whole. Don't build a "best of".
- [ ] Watch it once with the sound off: can you see the taped camera, the "Sonar live" HUD, S's chest and the alarm without being told where to look?
- [ ] Export 1080p MP4. Copy it to both phones and the Mac as a local file. Upload an unlisted copy for Devpost (`[VIDEO LINK]` in `DEVPOST.md`).

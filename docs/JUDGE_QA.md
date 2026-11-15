# Judge Q&A: the 15 hardest questions

Short answers you can say out loud, each grounded in the code or a measurement. Lead with the first sentence, stop there if the judge nods, and keep the rest for follow-ups. If you haven't measured something, say so and say how you would measure it.

Fill before judging: `[HONEST SCORE]`, `[LIVE BREATHING CHECK]`, `[ALARM TIME]` (see `DEMO_RUNBOOK.md`, section 1). Also: the presence/breathing retuning was in progress in another session at 00:41 on Nov 14 (uncommitted). If it lands, presence and breathing look at 0.30-1.3 m instead of 2 m / 0.25-1.6 m, and short disturbances (up to 3 s) no longer restart breathing. Then update the 2 m and 0.25-1.6 m figures here.

---

**1. How accurate is it?**

It depends on the feature, so here are three answers. Gestures: on people the model never saw, [HONEST SCORE]. We hold out whole people, and never the presenter, because a judge is a stranger. Breathing: in simulation it is within 1 breath per minute in all 7 test cases (10-20 breaths per minute, 0.5-1 m away, shallow and deep breathing, drifting clocks, a direct path 50 dB louder than the chest echo). On this laptop: [LIVE BREATHING CHECK]. Published phone-sonar systems report errors of about 0.1-0.3 breaths per minute (ApneaApp, SonarBeat), and that is the bar a proper validation would hold us to. Across all of it, the engine only shows a rate when a confidence gate passes and three readings agree. Otherwise you see a dash.

**2. Your problem slide is about falls. Can it detect a fall?**

Not yet, and we don't claim it. Today it has the ingredients: presence within 2 m, the person's distance, a timer of how long nothing has moved, and whether a still person is breathing. The fall rule we'd add is in `DESIGN.md` section 10: a fast movement whose distance changes by more than half a metre in under a second, then stillness. Writing it is easy. Testing it needs a mattress and volunteers, which we didn't have this weekend.

**3. A microphone that's always on in a bedroom. How is that more private than a camera?**

The first thing the code does with the audio is a high-pass filter that cuts everything below about 15 kHz by 100 dB, and speech lives well below that. Everything after that filter sees only our own echo band. The engine writes nothing to disk and sends only numbers (a rate, a state, a gesture name, the chest waveform, a coarse Doppler strip) to a socket that, by default, only the laptop itself can reach. The honest limit: today this is software on a general-purpose laptop, so you trust the software, as with any app that has microphone access. A product should do the filtering inside the audio chip, so raw audio never reaches the operating system.

**4. Why not a camera that processes everything on the device, or blurs?**

A camera that never records is still a camera in the bedroom, and the person has to trust that it doesn't. Ours has no image to leak because it never forms one. Cameras also need light, and night is when breathing matters most. Sound doesn't care about the lights.

**5. Why not mmWave radar or Wi-Fi sensing? They already measure breathing.**

They do, and radar is probably more accurate and reaches farther. But radar is a new box in every room that someone has to buy and install, and Wi-Fi sensing needs particular routers and firmware. We use the speaker and microphone that are already there. Radar is a fine answer wherever you can install hardware. We're aiming at the rooms where you can't, or where nobody will.

**6. Why not a wearable or a pendant?**

A wearable only helps while it is on, charged and, for a pendant, pressed. In the BMJ study of people over 90, call alarms were widely available but went unused in most of the falls that ended with a long time on the floor. Ours asks nothing of the person. It works next to a wearable, too.

**7. Does it work on any laptop? And who keeps a laptop by the bed?**

We've tested one: this HP ENVY x360. A laptop needs a speaker and microphone that pass 18.5-21.5 kHz at 48 kHz, and an audio path that doesn't filter that band (on Windows that means exclusive mode). The one-minute self-test answers it for any machine and recommends a signal profile, including lower bands down to 17 kHz for weaker speakers. Breathing and presence need no training. The gesture model is trained per laptop. A MacBook path exists in the code but hasn't run on a MacBook. As for the bedside: the laptop is our proof that ordinary audio hardware is enough. The target is whatever already sits in the room with a speaker and a microphone: a tablet on the nightstand, a smart speaker, a TV.

**8. What about pets?**

Two separate problems. Sensing: a pet moving within 2 m counts as presence, and a sleeping pet at the right distance could be read as breathing. It can't tell a cat from a person today. Distance helps, since the pet's echo comes from somewhere else, but we haven't tested it. Hearing: dogs hear up to about 45 kHz and cats past 60 kHz, so they can hear our band. We don't know whether it bothers them, because we haven't tested with animals. Until we have, the answer is the lowest volume that passes the self-test, and in a product, only transmitting when someone needs watching.

**9. What if two people are in the room?**

Presence says "someone", not how many. Breathing looks for moving echoes between 0.25 and 1.6 m, considers up to three distances and reports one rate: the clearest, usually the nearest person. Two people at different distances are separable in principle (the chirp resolves about 8 cm), but today we track one. Two people at the same distance would blur together.

**10. Does it see through walls? How far does it reach?**

No. Sound at 20 kHz barely passes through a wall. Presence counts movement within 2 m. Breathing searches from 0.25 to 1.6 m, was tested in simulation at 0.5-1 m, and we demo it at 0.5-0.8 m. Gestures work within about 0.6 m. The reach is limited mostly by how loud a laptop speaker can play. So it senses the room it is in, which is also a privacy property: it can't listen to the next room.

**11. Is the sound safe, for people and for pets?**

Health Canada's guideline for airborne ultrasound (Safety Code 24) sets 75 dB in the one-third-octave band around 20 kHz. A published measurement of a laptop's near-ultrasound output found about 63 dB SPL at 20 cm (arXiv 2103.11261). We haven't measured our own level with a calibrated microphone (phone apps can't measure 20 kHz reliably), so that is on the list. The same guideline notes that some people get headaches or fatigue from high-frequency sound, so we run at the lowest volume that passes the self-test, and the self-test checks that nothing leaks into the audible band. Last night's run: not audible above room noise. Older adults, the people this is for, hear this band least. For pets, see question 8: they can hear it, and we haven't tested their reaction.

**12. How often will the no-breathing alarm go off by mistake, or miss a real stop?**

We haven't measured either rate. That is the first thing a validation study has to measure. What the design does about it: the alarm arms only after 5 s of steady breathing from a still person, any gesture or presence change holds it off for 6 s, it needs 10 s with no breathing movement, and the engine refuses to keep "measuring" from old data once the chest stops. Known weak spots: shallow breathing can look flat (a false alarm); a person who keeps moving restarts the measurement, so there's no breathing alarm while they move; standing people sway in the breathing band; lying down and blankets are untested. In rehearsal the alarm came [ALARM TIME] seconds into a breath-hold.

**13. What's the regulatory path?**

Today it is a prototype, and the Care Monitor says so on screen. In the US, a device that alarms when breathing stops is an apnea monitor (21 CFR 868.2377, Class II with special controls). One that measures respiratory rate is a breathing frequency monitor (21 CFR 868.2375, also Class II). Both go through a 510(k). So the path is a clinical comparison against a reference (a respiration belt or capnography), then a 510(k). Because it is software on hardware we don't make, validation has to cover each device model it runs on. A first product could avoid medical claims and show breathing trends for wellness, but the moment it promises to catch apnea, it is a medical device.

**14. Who pays? What's the business model?**

We haven't validated a market, so this is a hypothesis. First: a per-room software licence for care homes and home-care agencies, running on tablets and laptops they already own, for overnight breathing checks and presence in rooms where cameras aren't acceptable. Second: licensing the sensing to device makers. Ultrasonic presence is already a licensed feature on some laptops (Elliptic Labs on Lenovo), so device makers know how to buy this kind of software. Later, after clearance, clinical remote monitoring could be reimbursed.

**15. What's new here? Phone sonar measured breathing in 2015.**

The physics isn't new, and we credit it: `DESIGN.md` section 2 lists the prior systems we built on. Most of them do one task with one signal on a phone. We built one engine that runs gestures, presence and breathing from a single tone-plus-chirp signal, live, on a stock Windows laptop. Getting there took a receiver that synchronises itself (Windows never reports dropped audio), a two-microphone direction cue after we measured that the speakers aren't stereo at these frequencies, a breathing estimator that shows a dash rather than guess, and a physics simulator that let us train with almost no recordings. On real public recordings, physics alone told push from pull 96 of 96 times, and with our pre-training one real example per class reached 94%, against 76% without.

---

## Numbers to quote

| Fact | Value | Source |
|---|---|---|
| Signal | 18.5 kHz pilot tone on one speaker, 19.5-21.5 kHz chirp on the other, repeating every 20 ms (profile `clean`) | `sonar/config.py` |
| Self-test on the HP | weakest pilot tone 63 dB, chirp band 73 dB (in 1 Hz); lock True, coherence 0.81; drift -0.3 ppm; not audible above room noise | self-test, Nov 13-14 |
| Live engine on the HP | weakest pilot tone about 53 dB over the Doppler noise floor (the screen's "Echo SNR"); status read 55 frames/s at that moment, while the long-run rate is the signal's 50; about 85 ms processing latency (audio block plus compute; a swipe also waits about 0.5 s to be sure it ended); detected a person at about 0.8 m | engine status, Nov 13-14 |
| Speaker channels | arrive at the microphones within 4 µs of each other, so left/right comes from the two microphones | self-test |
| Windows shared mode | low-passes the microphone: 15-22 kHz lands at the noise floor. Exclusive mode at 48 kHz fixes it | `DESIGN.md` section 1 |
| Chest movement to phase | 1 mm turns the echo phase by 43 degrees (wavelength about 1.7 cm) | `DESIGN.md` section 4 |
| Doppler | about 108 Hz per m/s of hand speed at 18.5 kHz; ±2.3 m/s range | `DESIGN.md` section 4 |
| Distance steps | 2.7 cm per tap, about 8 cm resolution, out to 2.6 m | `sonar/config.py` |
| Breathing | searches 0.25-1.6 m (simulation tested 0.5-1 m, demo at 0.5-0.8 m); 6-36 breaths/min; first rate after about 10-15 s of stillness | `sonar/config.py`, `tests/test_breathing.py`, `README.md` |
| Alarm | arms after 5 s of steady breathing; fires after 10 s without breathing movement | `app/web/js/store.js`, `config.js` |
| Presence | within 2 m; empty, entering, present; stillness timer | `sonar/presence.py` |
| Gesture model | 4 conv blocks, about 136,000 weights at the default width; 1.6 s Doppler picture; ONNX Runtime on the CPU | `sonar/model.py`, `DESIGN.md` section 6 |
| Gesture rules | confidence 0.6; 0.4 s dead time; opposite gesture ignored for 1.2 s; range gate at 0.6 m | `sonar/engine.py`, `sonar/config.py` |
| Push vs pull by physics alone | 96 of 96 real public clips (other hardware) | `DATA_STRATEGY.md` section 1 |
| Synthetic-only model on real clips | 83 of 96 real pushes and pulls, never confused | `DATA_STRATEGY.md` section 1 |
| Pre-training benefit | one real clip per class: 94% with synthetic pre-training, 76% without | `DATA_STRATEGY.md` section 1 |
| Rules vs model, harsh simulated hall | ordinary movement called a gesture: 49% (rules) vs 11% (model) | `DATA_STRATEGY.md` section 1 |
| Tests | 89 core + 8 simulator tests, plus the web app's Python and Node tests | `tests/`, `datagen/tests/`, `app/tests/` |
| Held-out gesture score | `[HONEST SCORE]` | `models-release/TRAINING_RUN.json` |
| Falls | 1 in 4 adults 65+ report a fall each year | CDC |
| Alone | 82% of falls among people over 90 happened alone; call alarms unused in most long lies | Fleming and Brayne, BMJ 2008 |
| Living alone | 28% of older Americans living in the community | ACL, 2023 Profile of Older Americans |
| Respiratory rate | least often recorded vital sign; over 24 breaths/min, an adult is likely critically ill | Cretikos et al., Med J Aust 2008 |
| Apnea | 10 s without breathing | AASM scoring definition |
| Ultrasound guideline | 75 dB in the 20 kHz one-third-octave band | Health Canada Safety Code 24, Table 5 |

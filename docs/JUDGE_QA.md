# Judge Q&A: the hardest questions

Short answers you can say out loud, each grounded in the code or a measurement. Lead with the first sentence, stop there if the judge nods, and keep the rest for follow-ups. If you haven't measured something, say so and say how you would measure it.

Fill before judging: `[SNR ON THE MACBOOK]` in the table at the end. Read it off the dashboard's Sensor card.

---

**1. How accurate is it?**

For gestures, on people held out of training: 133 of 206 recognised (65%), and 22 of 499 other movements set off a gesture (4.4%). We trained three times, each time leaving one of our three people out, and tested on the person left out. On the best-matched of the three it was 70 of 81 (86%), with 5 of 114 other movements triggering. It is three people on one laptop, so treat it as a first measurement. The spread between people is large: one of us swipes faster than the signal can follow (about 2.3 m/s) and stayed under half in every version we trained. The model in the demo is trained on all three of us, so it knows the presenter better than it knows a stranger. The help gesture has its own numbers (question 4). The alerts are rules, and we have no false-alert rate from long real use (question 18). Breathing is an early estimate (question 3).

**2. How do you tell left from right with one microphone?**

With the two speakers. The left speaker plays 18.5 kHz and the right plays 19.0 kHz, so the microphone can tell which echo came from which speaker by its pitch. The speakers sit apart. A hand crossing the screen moves away from one speaker while it approaches the other, so the two tones change in opposite ways and at different moments. A small neural network reads two pictures, one per tone, and names the direction. The price: the model depends on where this MacBook's speakers and microphone sit, so it only works on this laptop.

**3. Does the breathing reading work?**

As an early estimate, and nothing in the demo depends on it. The method reads the echo's timing: the wavelength is about 1.7 cm, so 1 mm of chest movement turns the echo's phase by about 43 degrees. In simulation it stays within 1 breath per minute in all 7 test cases (10-20 breaths per minute, 0.5-1 m away, shallow and deep breathing, drifting clocks, a direct path 50 dB louder than the chest echo), and also in a simulated busy hall with one microphone. On the real MacBook we have not checked it against a reference such as a respiration belt. A MacBook gives one microphone channel, which is half the looks at the chest the method was designed around. So the card is labelled "Early estimate", and it shows a dash unless a confidence gate passes and three readings in a row agree. Published phone-sonar systems report errors of about 0.1-0.3 breaths per minute (ApneaApp, SonarBeat). That is the bar a proper validation would hold us to.

**4. What stops a false help call?**

Timing first, then a cancel time. A help request needs two push events between 1.0 and 2.5 seconds apart. We counted in our recordings: 28 of 69 single pushes were reported twice, because the hand coming back also looks like a push, and most of those repeats came within a second. Treating a second push sooner than 1.0 s as the same push leaves 6 of 69, about 1 in 12, that still look like two. Because of that 1 in 12, the request shows on screen for 3 seconds with a Cancel button before any alert goes out. The other mistake is covered too: a hand gesture can never cancel a help request. Only a person at the screen can, and another push sends it at once. Swipes almost never read as a push (1 of 138). What we have not measured: how reliably a deliberate double push is caught.

**5. Does it press keys in any app?**

Yes. A gesture presses a real key in whatever app is in front, the same as a keyboard would, so the app being controlled needs no plug-in. Presets cover slides and PDFs (right arrow, left arrow, space), media (next, previous, play/pause) and scrolling (page down, page up). Custom lets you pick a key for each gesture from a fixed list. It cannot type text or click. There is no volume or mute action on purpose: the sonar plays through the same speakers, and changing the volume would change the sensor's own signal. A web page cannot abuse it: the app's web server listens on this computer only and refuses requests from other websites.

**6. What permission does it need?**

Two, both for the terminal app it is started from. Microphone, which macOS asks for on the first run: the sonar listens through it. Accessibility (System Settings > Privacy & Security > Accessibility), which macOS requires before any app may press keys in other apps. Only touch-free control needs Accessibility. The room monitor and the help gesture work without it, and the app then says "Keys blocked" instead of pretending.

**7. What if the laptop is playing music?**

Loud audio through the same speakers can weaken the sonar's signal, so the app shows signal strength on the dashboard and in the menu bar. Most of the music never reaches the sensing code: the first filter removes everything below about 15 kHz. We already handle one case of this. The alert chime and voice come out of the same speakers and can look like movement, so movement seen while they play is ignored. A product should sit inside the audio driver so it can share the speakers properly.

**8. Your problem slide is about falls. Can it detect a fall?**

Not in a way we would claim. What works today sits around a fall: a help gesture that needs nothing worn, and a "No movement" alert when someone is present but has not moved for a set time. That alert is a prompt to check on someone. It is not fall detection: a person who falls out of the sensor's range (about 1.3 m) reads as "room empty". The app does carry a first fall rule, switched off by default and labelled experimental: one sudden one-way movement, then 15 s of stillness. We ran it over 36 minutes of our own recordings of gestures, walking and sitting: no false alarm at 15 s, and at 10 s one, a person stepping away from the laptop and standing still. It has never seen a real fall. Testing that needs a mattress and volunteers, which we didn't have this weekend (`docs/SONAR_CARE.md`, section 10).

**9. How does the alert reach a caregiver who is in another room?**

Today it stays on the laptop: a full-screen alert, a notification, the words spoken aloud, and a chime that repeats every 20 seconds while nobody has acknowledged. That reaches someone within earshot. The app has an option (`--lan`) that lets another device on the same network open the dashboard and acknowledge an alert. Alerts on a caregiver's phone outside the home are the next step.

**10. A microphone that's always on in a bedroom. How is that more private than a camera?**

The first thing the code does with the audio is a high-pass filter that cuts everything below about 15 kHz by 100 dB, and speech lives well below that. Everything after that filter sees only our own echo band. No audio is written to disk or sent anywhere. The engine sends only small messages (a state, a gesture name, a rate, the chest waveform, a coarse Doppler strip) on a connection that only the laptop itself can reach. Sonar Care keeps a settings file and a text log of events on the laptop. The honest limit: today this is software on a general-purpose laptop, so you trust the software, as with any app that has microphone access. A product should do the filtering inside the audio chip, so raw audio never reaches the operating system.

**11. Why not a camera that processes everything on the device, or blurs?**

A camera that never records is still a camera in the bedroom, and the person has to trust that it doesn't. Ours has no image to leak because it never forms one. Cameras also need light, and night is when a room monitor matters most. Sound doesn't care about the lights.

**12. Why not mmWave radar or Wi-Fi sensing? They already measure breathing.**

They do, and radar is probably more accurate and reaches farther. But radar is a new box in every room that someone has to buy and install, and Wi-Fi sensing needs particular routers and firmware. We use the speaker and microphone that are already there. Radar is a fine answer wherever you can install hardware. We're aiming at the rooms where you can't, or where nobody will.

**13. Why not a wearable or a pendant?**

A wearable only helps while it is on, charged and, for a pendant, pressed. In the BMJ study of people over 90, call alarms were widely available but went unused in most of the falls that ended with a long time on the floor. Two pushes at the laptop need nothing worn, and the "No movement" alert asks nothing of the person at all. The limit is reach: the help gesture works a hand-length from the screen, so it does not replace a pendant across the room. It works next to one.

**14. Does it work on any laptop? And who keeps a laptop by the bed?**

The gestures don't. Today it runs on this one MacBook Pro, and the gesture model fits only this laptop, because left and right depend on where its two speakers and its microphone sit. Presence and breathing use no trained model, so they are not tied to one laptop in the same way. Any machine still needs speakers and a microphone that pass 18.5-21.5 kHz at 48 kHz. A one-minute self-test in the repo checks that and recommends a signal profile, including lower bands down to 17 kHz for weaker speakers. As for the bedside: the laptop is our proof that ordinary audio hardware is enough. The target is whatever already sits in the room with a speaker and a microphone: a tablet on the nightstand, a smart speaker, a TV.

**15. What about pets?**

Two separate problems. Sensing: a pet moving within about 1.3 m counts as presence, and a sleeping pet at the right distance could be read as breathing. It can't tell a cat from a person today. Distance helps, since the pet's echo comes from somewhere else, but we haven't tested it. Hearing: dogs hear up to about 45 kHz and cats past 60 kHz, so they can hear our band. We don't know whether it bothers them, because we haven't tested with animals. A product should transmit only while someone needs watching.

**16. What if two people are in the room?**

Presence says "someone", not how many. It follows the person in front of the laptop: once somebody is tracked, only movement up to about 0.35 m behind them counts as theirs. Breathing looks for moving echoes between 0.3 and 1.3 m, considers up to three distances and reports one rate: the clearest, usually the nearest person. Two people at different distances are separable in principle (the chirp resolves about 8 cm), but today we track one. Two people at the same distance would blur together. For the "No movement" alert that means a second person moving close to the laptop resets the count.

**17. Does it see through walls? How far does it reach?**

No. Sound at 20 kHz barely passes through a wall. Presence and breathing look between 0.3 and 1.3 m from the laptop. Gestures are made a hand-length (20 to 25 cm) from the screen. The reach is limited by how loud a laptop speaker can play, and we cut it at 1.3 m on purpose so that people walking past farther away are not counted. So it senses the space in front of it, which is also a privacy property: it can't listen to the next room.

**18. How often will the alerts go off by mistake, or miss something real?**

We haven't measured either rate in long real use. That is the first thing a validation study has to measure. What the design does about it: the "No movement" alert only arms for a confirmed person, meaning the engine reported "present" at 60% confidence or more twice in a row, because a noisy empty room can report a faint "present". An engine outage or a calibration never counts as stillness. The alert's own chime cannot cancel it. It resolves by itself when movement returns. Known weak spots: a person who sits very still for a minute raises it, so it is a prompt to check; a person who moves out of range reads as "room empty"; someone else moving close to the laptop resets the count. The "No breathing movement" alert (10 seconds without breathing movement, after 5 seconds of a steady rate) inherits everything that is experimental about breathing on one microphone.

**19. Is the sound safe, for people and for pets?**

Health Canada's guideline for airborne ultrasound (Safety Code 24) sets 75 dB in the one-third-octave band around 20 kHz. A published measurement of a laptop's near-ultrasound output found about 63 dB SPL at 20 cm (arXiv 2103.11261). We haven't measured our own level with a calibrated microphone (phone apps can't measure 20 kHz reliably), so that is on the list. The same guideline notes that some people get headaches or fatigue from high-frequency sound, so a product should run at the lowest volume that still gives a good signal. Our lowest tone is 18.5 kHz, and a few young people can hear that high, so a young judge may notice a faint whine. Older adults, the people this is for, hear this band least. For pets, see question 15: they can hear it, and we haven't tested their reaction.

**20. What's the regulatory path?**

Today it is a prototype, and the dashboard says so on screen. Pressing a key and raising a help request are not medical claims. Monitoring is where that changes. In the US, a device that alarms when breathing stops is an apnea monitor (21 CFR 868.2377, Class II with special controls). One that measures respiratory rate is a breathing frequency monitor (21 CFR 868.2375, also Class II). Both go through a 510(k). So the path for the breathing part is a clinical comparison against a reference (a respiration belt or capnography), then a 510(k). Because it is software on hardware we don't make, validation has to cover each device model it runs on. A first product could avoid medical claims and offer touch-free control, a help request and presence. The moment it promises to catch apnea, it is a medical device.

**21. Who pays? What's the business model?**

We haven't validated a market, so this is a hypothesis. First: a per-room software licence for care homes and home-care agencies, running on tablets and laptops they already own, for touch-free control, a help request and presence in rooms where cameras aren't acceptable. Second: licensing the sensing to device makers. Ultrasonic presence is already a licensed feature on some laptops (Elliptic Labs on Lenovo), so device makers know how to buy this kind of software. Later, after clearance, clinical remote monitoring could be reimbursed.

**22. What's new here? Phone sonar measured breathing in 2015.**

The physics isn't new, and we credit it: `DESIGN.md` section 2 lists the prior systems we built on. Most of them do one task with one signal on a phone. Ours runs gestures, presence and an early breathing estimate from a single tone-plus-chirp signal, live, on a stock MacBook with one microphone. Around that we built something a patient and a caregiver can use: real key presses in any app, a help gesture designed around its own false alarms, a "No movement" alert and a guided setup. The engineering behind it: left and right from two speakers heard by one microphone, a receiver that finds its own timing, and a physics simulator that replays real tracked hand paths, so we could train with only three people's recordings. On real public recordings made on other hardware, physics alone told a hand coming toward the microphone from a hand going away 96 of 96 times.

**23. You say no camera. Did you use one to build it?**

Only while collecting training data, and only hand points. Our recorder can use the webcam to follow the hand, so a gesture gets its label and its path without anyone timing it. It saves 21 hand points per picture and never a picture or a face. The running product never opens the camera, which is why the camera can stay taped over.

*Check before judging: confirm with whoever made the recordings that the hand paths came from this recorder (`sonar/camrecord.py`). If they did not, answer with how they were tracked.*

---

## Numbers to quote

| Fact | Value | Source |
|---|---|---|
| Signal | 18.5 kHz tone from the left speaker, 19.0 kHz tone from the right, 19.5-21.5 kHz chirp (it shares the left speaker); one waveform repeating every 20 ms; 48 kHz | `sonar/config.py` |
| Microphone | one channel; left and right come from the two speakers | `TRAINING.md` section 9, `models-candidate-mac-calm50/gesture.json` |
| Signal strength on the MacBook | `[SNR ON THE MACBOOK]` dB. 35 dB or more shows as "good signal", under 25 dB as "weak signal". | dashboard Sensor card; `app/care/service.py` |
| Held-out gesture score | 133 of 206 gestures recognised (65%); 22 of 499 other movements triggered (4.4%); each of 3 people held out of training in turn | recall session summary, Nov 14, section 2b |
| Best-matched held-out person | 70 of 81 (86%), with 5 of 114 other movements triggering. Never quote it alone. | recall session summary, sections 1 and 2b |
| People in the recordings | 3, all on this MacBook | `docs/SONAR_CARE.md` section 9 |
| Gestures | swipe left, swipe right, push. The model has no other classes. | `models-candidate-mac-calm50/gesture.json` |
| Gesture model | three copies of a 4-block CNN, about 136,000 weights each, averaged; a 1.6 s Doppler picture per tone; ONNX Runtime on the CPU | `sonar/model.py`, `models-candidate-mac-calm50/` |
| Gesture rules in the engine | confidence 0.6; 0.4 s dead time; opposite swipe ignored for 1.2 s; a swipe is reported about 0.5 s after the hand stops | `sonar/engine.py`, `sonar/config.py` |
| Sonar Care's own checks | confidence 0.6; 0.7 s between actions; opposite swipe ignored for 1.5 s; same gesture ignored for 1.3 s | `docs/SONAR_CARE.md` section 7 |
| Fastest hand the signal follows | about 2.3 m/s | `sonar/config.py` |
| One push reported twice | 28 of 69 single pushes (41%) | `docs/SONAR_CARE.md` section 7 |
| Single push that still looks like two | 6 of 69, about 1 in 12, after the 1.0 s rule | `docs/SONAR_CARE.md` section 7 |
| A swipe read as a push | 1 of 138 | `docs/SONAR_CARE.md` section 7 |
| Help gesture | second push 1.0 to 2.5 s after the first; 3 s to cancel on screen; a single push presses its key after 2.5 s; the chime repeats every 20 s while unanswered, up to 10 times | `app/care/settings.py`, `app/care/rules.py`, `docs/SONAR_CARE.md` section 7 |
| Deliberate double push | not measured yet | `docs/SONAR_CARE.md` sections 7 and 9 |
| "No movement" alert | 60 s by default, 15 s in the demo choice; needs "present" at 60% confidence or more twice in a row | `app/care/settings.py`, `app/care/rules.py` |
| "No breathing movement" alert | arms after 5 s of a steady rate; fires after 10 s without breathing movement; experimental | `app/care/rules.py`, `app/care/settings.py` |
| Presence | 0.3 to 1.3 m; empty, entering, present; a stillness timer | `sonar/config.py`, `sonar/presence.py` |
| Breathing | looks between 0.3 and 1.3 m; 6-36 breaths/min; needs at least 10 s of stillness; experimental on one microphone | `sonar/config.py` |
| Breathing in simulation | within 1 breath per minute in 7 cases, and in a busy hall with one microphone | `tests/test_breathing.py`, `tests/test_hall.py` |
| Chest movement to phase | 1 mm turns the echo phase by about 43 degrees (wavelength about 1.7 cm) | `DESIGN.md` section 4 |
| Doppler | about 108 Hz per m/s of hand speed at 18.5 kHz | `DESIGN.md` section 4 |
| Distance steps | 2.7 cm per tap, about 8 cm resolution, out to 2.6 m | `sonar/config.py` |
| First filter | cuts everything below about 15 kHz by 100 dB | `sonar/config.py`, `DESIGN.md` section 5.1 |
| Toward vs away by physics alone | 96 of 96 real public clips (other hardware) | `DATA_STRATEGY.md` section 1 |
| Simulator-only model on real clips | 83 of those 96, the two directions never confused | `DATA_STRATEGY.md` section 1 |
| Pre-training benefit | one real clip per class: 94% with simulated pre-training, 76% without | `DATA_STRATEGY.md` section 1 |
| Rules vs model, harsh simulated hall | ordinary movement called a gesture: 49% (rules) vs 11% (model) | `DATA_STRATEGY.md` section 1 |
| Falls | 1 in 4 adults 65+ report a fall each year | CDC |
| Alone | 82% of falls among people over 90 happened alone; call alarms unused in most long lies | Fleming and Brayne, BMJ 2008 |
| Living alone | 28% of older Americans living in the community | ACL, 2023 Profile of Older Americans |
| Respiratory rate | least often recorded vital sign; over 24 breaths/min, an adult is likely critically ill | Cretikos et al., Med J Aust 2008 |
| Apnea | 10 s without breathing | AASM scoring definition |
| Ultrasound guideline | 75 dB in the 20 kHz one-third-octave band | Health Canada Safety Code 24, Table 5 |

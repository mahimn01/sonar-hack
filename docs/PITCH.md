# Pitch scripts: 3 minutes and 60 seconds

Two speakers' parts are written out: **P** is the presenter, who talks. **S** is the subject, who sits at the laptop and breathes for the demo (Lance, unless you swap). Read the five rules before rehearsing. They come straight from how the code behaves, and each one can sink the demo.

Placeholders: `[RATE]` is whatever number the screen shows (read it, never guess). `[HONEST SCORE]` is the held-out training score, as one short spoken phrase. `[ALARM TIME]` is the seconds from breath-hold to alarm that you time in rehearsal.

## Five rules for the live part

1. **Nothing moves within 2 m while breathing is measured.** Any walking, reaching or hand-waving within about 2 m restarts the measurement, and big movements farther away can too, because the tone's movement check has no distance limit (`sonar/pipeline.py` feeds presence's "moving" flag into the breathing estimator). P keeps both hands down and stands still at least 2 m back. S doesn't talk. Judges at a table get asked to stand still.
2. **The number takes 10-15 seconds.** The estimator needs 10 s of stillness and three readings in a row that agree before it shows a rate. Until then the orb says "Hold still · N of 10 s". Fill that time with talk, not with movement.
3. **Wait 5 seconds after the number appears before the breath-hold.** The alarm only arms once a steady rate has been on screen for 5 s (`app/web/js/store.js`). Ask too early and no alarm will come.
4. **Expect the alarm 10-15 s into the hold.** That window is worked out from the code, not timed on the HP. Time it in rehearsal and write it here: `[ALARM TIME]`. S holds for up to 25 s, then breathes whether or not the alarm came.
5. **Gestures come last**, after the breathing demo. Swipe a hand-length (20-25 cm) from the screen, briskly, then let the hand drop. For **next slide** (swipe right) use the **left** hand, starting on the left. For **back**, use the right hand from the right. Whoever recorded the most gesture sessions does the swipes, because the model knows that person best. One miss: press the arrow key and keep talking. Never retry twice.

---

## The 3-minute version (stage, or a long judging slot)

Setup when the clock starts: engine running and calibrated, HUD shows a green "Sonar live", slide 1 on screen. The chair is at the tape mark (chest 0.6 m from the screen). S stands at least 2 m from the laptop. P stands at the laptop for slides 1-4 and changes slides with the arrow keys.

### 0:00-0:15 · Slide 1, title

**P:** "This laptop's camera is taped over. In about a minute it will measure Lance's breathing from across the table, and sound an alarm when he stops. It does it with sound you can't hear."

### 0:15-0:40 · Slide 2, the problem

**P** presses →.

**P:** "One in four adults over 65 falls every year. In a study of people over 90, 82 percent of falls happened when they were alone, and in most of the long waits on the floor, nobody used their call alarm. Those falls happen in bedrooms and bathrooms, where nobody wants a camera."

### 0:40-1:00 · Slide 3, the insight

**P** presses →.

**P:** "But almost every room already has a speaker and a microphone. These speakers are playing sound above 18 kilohertz right now, and most adults can't hear it. The microphones catch the echo. A moving hand changes its pitch. A breathing chest changes its timing: one millimetre of chest movement turns the echo by 43 degrees."

### 1:00-1:12 · Slide 4, how it works

**P** presses →.

**P:** "Five steps, all on the laptop's own processor. The strip at the bottom is the echo, live. Breathing and presence need no training at all. A small neural network names hand gestures."

### 1:12-2:05 · Slide 5, the live demo

**P** presses →, then walks to the 2 m mark and stands still, hands down. **S** sits in the chair: hands in lap, facing the screen, breathing steadily and a little deeper than normal (about 2 seconds in, 2 out). S doesn't talk until the gesture.

**P** (while the orb says "Hold still"): "Lance just sat down, and it shows a person present. Now it needs ten seconds of stillness and three readings that agree. The line rises when his chest moves toward the laptop. When it isn't sure it shows a dash, because a wrong number is worse than none."

*The rate appears, about 1:30.*

**P:** "[RATE] breaths a minute." *(Count five seconds. If you need words: "It has to hold steady for five seconds before the alarm arms.")* "Lance, hold your breath."

*S holds, about 1:37.*

**P** (while waiting): "Ten seconds without breathing is the clinical definition of an apnea, so that's what it waits for. Nothing here has taken a single picture."

*The orb turns red and counts seconds, about 1:50.*

**P:** "There's the alarm, counting. Breathe, Lance." *(It clears when breathing resumes.)* "One more thing. The same echoes read hands."

*The gesture person swipes right with the left hand. Slide 6 slides in from the left, about 2:05.*

### 2:05-2:25 · Slide 6, healthcare impact

**P:** "Why breathing? It's the vital sign hospitals record least, and above 24 breaths a minute an adult is likely critically ill. So today, one ordinary laptop gives a breathing rate, a no-breathing alarm, presence without images, and touch-free control. Nothing to wear or charge."

### 2:25-2:45 · Slide 7, under the hood

Swipe right again, or press → if the first swipe needed the keyboard.

**P:** "Measured on this laptop: our tone arrives 63 dB above the noise in a one-hertz band, clock drift is a third of a part per million, and processing runs in real time, about 85 milliseconds behind the microphone. Windows' normal audio mode deletes everything above 15 kilohertz, so we go around it. With almost no data, we wrote a physics simulator to pre-train on. On people the model never saw: [HONEST SCORE]."

### 2:45-2:55 · Slide 8, what's next

**P:** "Next: test it against a respiration belt, move it to the bedside for overnight breathing, add fall detection, and run it on the tablets and speakers already in these rooms."

### 2:55-3:00 · Slide 9, close

**P:** "Camera-free care, from the laptop already in the room. Come watch your own breathing at our table."

### If you have a wireless clicker

S sits from 0:00 and P changes slides from 2 m away. The rate is then ready when slide 5 appears, which frees about 20 seconds: go straight to the breath-hold and spend the time on slide 7. Don't use the laptop's own keyboard from close range during the acquisition, because your hands at the keyboard count as movement.

### If you're running long

Cut slide 4's lines and slide 8's lines. Never cut the breath-hold.

---

## The 60-second version (judges at the table)

Setup: S already sitting at the tape mark, still. Slide 5 on screen. P stands beside the table, out of the 2 m zone if the table allows.

**P (0:00):** "Hi, we're Laptop Sonar. One favour: please stand still for twenty seconds. This laptop is measuring Lance's breathing with sound, and it notices anyone moving nearby."

**P (0:10):** "The camera is taped. The speakers play a tone we can't hear, and the microphones catch the echo off his chest. A millimetre of breathing turns that echo by 43 degrees. That's this line." *(Point, don't wave.)*

**P (0:22, once the number has been up for 5 seconds):** "[RATE] breaths a minute. When it isn't sure, it shows a dash. Lance, hold your breath."

**P (0:28):** "Respiratory rate is the vital sign hospitals record least. Ten seconds without breathing is an apnea." *(Alarm, about 0:40.)* "There's the alarm. Breathe."

**P (0:45):** "The same echoes read hands." *(Swipe right with the left hand. The next slide comes in.)*

**P (0:50):** "It all runs on the laptop's processor and sends numbers, never audio or images. On people the gesture model never saw: [HONEST SCORE]. Camera-free care, from the laptop already in the room."

If the number isn't up by 0:22, keep talking (the falls statistic, the 43 degrees, the self-test numbers) and do the breath-hold when it arrives. If judges must leave first, offer: "Sit here for 20 seconds and watch your own breathing."

---

## When the live part goes wrong (say this, then act)

| What happens | Say | Do |
|---|---|---|
| No number after 25 s | "It's waiting for three steady readings. Anything moving nearby restarts it." | Check that nobody near the table is moving. Give it 20 s more. |
| Still no number at 45 s | "The hall is moving more than our chair can hide. Here's the same demo on this laptop last night." | Play the backup video from the breathing shot (`DEMO_RUNBOOK.md`, failure B1). |
| No alarm 25 s into the hold | "Breathe, Lance." Then: "It watches for any movement at all, and in rehearsal it took [ALARM TIME] seconds." | Move on. Offer to retry after the pitch. |
| Alarm while S is breathing | "That's a false alarm, and that's why this is a prototype and not a medical device." | Ask S to take two deep breaths. Move on. |
| Swipe does nothing | "It didn't catch that one, I'll use the keyboard." | Press →. |
| Swipe goes the wrong way | "Wrong way, which is why we test on people the model has never seen." | Press → twice, or the slide number. |

## Words to use and avoid

| Avoid | Say instead | Why |
|---|---|---|
| "ultrasound" for the whole signal | "sound above what most adults can hear", "near-ultrasound" | The tone is 18.5 kHz, and ultrasound usually means above 20 kHz. An acoustics judge will notice. |
| "works on any laptop" | "an ordinary laptop"; "tested on this HP" | One laptop has been tested. |
| "detects falls" | "fall detection is next" | It doesn't yet. Presence and the stillness timer are the parts that exist. |
| "medical grade", "diagnoses apnea" | "a no-breathing alarm", "a prototype" | Not validated, not cleared. |
| any accuracy number except the held-out one | `[HONEST SCORE]` | Scores on people in the training set flatter the model. |
| "it sees" | "it measures", "it hears" | The point is that nothing sees. |

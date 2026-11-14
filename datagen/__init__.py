"""datagen — synthetic ultrasonic-echo data for the laptop sonar.

What this package is for
------------------------
We cannot hand-record hours of labelled gestures in a 36-hour hackathon, so this
package *computes* what the laptop microphones would hear instead:

    speaker signal  ->  [virtual laptop + room + moving hand/arm/torso]  ->  2-mic audio

The output is ordinary raw microphone audio (same sample rate, same channel
layout as a real recording), so it goes through exactly the same feature code
and the same CNN as real data.

Modules
-------
physics.py             speed of sound, air absorption, small geometry helpers
hardware.py            one random "virtual laptop + room" (geometry, responses, noise)
kinematics.py          how hands, arms, torsos and walkers move
scenes.py              one random scene per class (swipe_left, push, none, entering ...)
render.py              turns (speaker signal, hardware, scene) into microphone audio
background.py          lays synthetic gestures over REAL background recordings
adapter.py             the ONLY place that knows about the core `sonar/` package
generate.py            command line: make a training set in the core's dataset format
render_examples.py     command line: sanity pictures per class
train_check.py         command line: train on synthetic data, check that it transfers
external_real_check.py command line: test on real public recordings
quicklook.py           minimal Doppler spectrogram used for plots, tests and rules
heuristics.py          training-free gesture cues (pitch-shift sign, left/right)

Everything here is read-only with respect to the rest of the repo: it only
writes inside the output folder you pass on the command line.
"""

__all__ = [
    "physics",
    "hardware",
    "kinematics",
    "scenes",
    "render",
    "quicklook",
    "adapter",
]

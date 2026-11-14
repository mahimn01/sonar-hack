# Motion library: real hand paths for the synthetic gestures

`unica_v1.npz` holds 123 hand paths (the position of the palm and the direction it faces, 30 times a
second) tracked from short webcam videos of 15 people moving a hand left, right, towards the camera and
away from it. `python -m datagen.generate --motion datagen/motion_library/unica_v1.npz` (and
`scripts/train_4090.py`, by default, for half of the synthetic gestures) replays these movements in the
simulator instead of the formula strokes of `datagen/kinematics.py`. `--motion none` turns it off.

| in the library | paths |
|---|---|
| swipe_left | 42 |
| swipe_right | 28 |
| push | 25 |
| pull | 28 |

## Where the videos come from, and the licence

Fronteddu, Porcu, Floris, Atzori: "A dynamic hand gesture recognition dataset for human-computer
interfaces", Computer Networks, 2022, DOI 10.1016/j.comnet.2022.108781. Data: "Dataset for Dynamic Hand
Gesture Recognition Systems", University of Cagliari, https://zenodo.org/records/16040322, licence
**Creative Commons Attribution 4.0 International (CC BY 4.0)**.

What was changed: only classes 01, 02, 18 and 19 of users 1-15 were used (125 clips); the hand was tracked
with Google MediaPipe Hand Landmarker (Apache 2.0); the 21 hand points were turned into one palm path in
metres, smoothed, and labelled by the direction the hand really travels (`datagen/motion.py`). The file
contains no pictures and nothing that identifies a person. It is shared under the same CC BY 4.0 licence;
keep this credit with it.

## How to rebuild or extend it

```bash
# 1. only the clips needed, straight out of the 22.9 GB archive (about 1 GB for 9 users x 4 classes)
mkdir -p datagen/motion/sources/unica_hgd && cd datagen/motion/sources/unica_hgd
python3 -I ../../../../scripts/fetch_zip_members.py "https://zenodo.org/api/records/16040322/files/hand_gesture_dataset_videos.zip/content" --match 'class_(01|02|18|19)/User[1-9]_/' --out .
curl -L -o hand_gesture_timing_stats.csv "https://zenodo.org/api/records/16040322/files/hand_gesture_timing_stats.csv/content"
cd ../../../..
# 2. the hand tracker has its own environment (it pins other library versions than the training one)
uv venv --python 3.12 .venv-mocap && uv pip install --python .venv-mocap/bin/python mediapipe opencv-python-headless numpy scipy
# 3. track and convert (about 1 second per clip)
.venv-mocap/bin/python -m datagen.mocap_extract --unica datagen/motion/sources/unica_hgd --out datagen/motion/unica_v1.npz
cp datagen/motion/unica_v1.npz datagen/motion_library/unica_v1.npz
```
Your own clips work the same way (`--clips clips.csv`, see the top of `datagen/mocap_extract.py`): a
minute of each gesture filmed with the demo laptop's webcam gives paths with exactly the demo's geometry
and no licence question at all.

## What it does and does not give

- Real: how the speed of a stroke rises and falls, how the path curves, how the palm turns, the wind-up
  before and the movement after the stroke.
- Not real: size and tempo. They are set to the demo's gesture range when a path is placed in a scene
  (`motion.place_motion`), because these videos show smaller movements than people make at the laptop.
- The distance from the camera comes from the apparent size of the hand (one camera, no depth sensor): good
  to about 20 %. Sideways movement is more exact than towards / away.
- A webcam sees only part of a wide sweep. Where the hand entered or left the picture while moving, a
  quarter of a second of smooth speeding up / slowing down was added (`motion._complete`).

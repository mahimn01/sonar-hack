# Handoff for Lance's RTX 4090 box: gesture recall work

For Lance (and Lance's bot). Run the steps in order, from the repo folder on the 4090 box
(`C:\Users\lance\sonar-hack`). Commit as yourself, no AI attribution lines. Every step says how long it takes
and exactly what to push back.

Why: the model scored 0.30 "gestures recognised" on the held-out person. The recordings in the repo do not
reproduce that number (the same recipe scores about 0.78 on them). The one session that was in the 4090 run but
is not in the repo, `mac/data/20261114-001248_mahimn_gestures`, was half of that test. The two "left/right wiring"
numbers your runs printed (-2.52 with it, +9.4 without it) only fit if the swipes in that session go the
opposite way to their prompts. Step A sends that session back so this can be checked directly.

---

## Step A — send back the missing session and the held-out run (about 1 minute, no GPU)

```powershell
cd C:\Users\lance\sonar-hack
git status --short | Select-Object -First 30
git fetch origin
git switch model-sat-am
git switch -c recall-data-1

# 1. the recording that was used in the 03:02 run but never committed (about 32 MB)
dir mac\data
git add -f mac/data/20261114-001248_mahimn_gestures

# 2. the held-out model of that run (the one that scored 0.296) and its log: small files
New-Item -ItemType Directory -Force recall_handoff\run_20261114-030255 | Out-Null
Copy-Item train_out\runs\20261114-030255\log.txt recall_handoff\run_20261114-030255\ -ErrorAction Continue
Copy-Item train_out\runs\20261114-030255\summary.json recall_handoff\run_20261114-030255\ -ErrorAction Continue
Copy-Item train_out\runs\20261114-030255\sweep\w24_s0 recall_handoff\run_20261114-030255\w24_s0 -Recurse -ErrorAction Continue
dir train_out\runs | Out-File recall_handoff\runs_listing.txt
dir train_out\synth, train_out\synth_bg, train_out\pretrain, train_out\sessions -ErrorAction Continue | Out-File recall_handoff\cache_listing.txt
git add -f recall_handoff

git commit -m "Add mahimn's first gesture session and the held-out run files"
git push -u origin recall-data-1
```

Push back: branch **`recall-data-1`**. It must contain
`mac/data/20261114-001248_mahimn_gestures/audio.wav` + `session.json` and the folder `recall_handoff/`.

If `mac\data\20261114-001248_mahimn_gestures` does not exist on the box: say so, and look for it on the MacBook
that recorded it (`data/20261114-001248_mahimn_gestures` in the repo folder there); send it any way you can.
If any other session folders exist under `mac\data` or `data` that are not in the repo, add them the same way
(`git add -f <folder>`).

One question for Mahimn while this runs: in your FIRST gesture session (about 00:12), did you sweep the opposite
way to the prompt (for example "swipe left" done as a hand moving towards your right, or the picture mirrored)?

---

## Step B — the leave-one-person-out experiment suite (GPU, about 40 minutes)

What it does: trains 15 recipes ("arms"), each one three times over with a different person held out
(lance+lance2 count as one person), 3 seeds each, and scores every one the same honest way. It also checks
every session's left/right labels against the others, and exports the best recipes as engine-ready models.
It is one command; everything lands in `train_out\recall\suite\<time>\`. Nothing in `models\` or
`models-release*\` is touched. The synthetic data is made on the GPU with the unchanged generator (the picture
check that already passed on this box is reused, so it should not run again).

Needs branch `recall` (Mahimn pushes it). Do Step A first.

```powershell
cd C:\Users\lance\sonar-hack
git fetch origin
git switch recall
git pull

# The recordings are not part of this branch. Put them into the working folder (they stay uncommitted here):
git restore --source origin/model-sat-am --worktree -- mac/data
git restore --source origin/recall-data-1 --worktree -- mac/data/20261114-001248_mahimn_gestures
dir mac\data                       # expect 13 session folders (12 if Step A found no 001248 session: still run)

# 2-minute self-test on the CPU first (its numbers mean nothing; it must end with "DONE")
powershell -ExecutionPolicy Bypass -File scripts\recall_suite.ps1 --device cpu --smoke

# the real run (about 40 minutes: ~10 min making data, ~25 min training 240 small models 4 at a time, ~2 min export)
powershell -ExecutionPolicy Bypass -File scripts\recall_suite.ps1
```

If it stops with an error: run the same last command again (it continues where it stopped). If it stops
again, push what exists anyway (below) so the log comes back.

Push back (branch **`recall-results-1`**), about 30 MB:

```powershell
$run = (Get-ChildItem train_out\recall\suite -Directory | Where-Object { $_.Name -notlike "smoke_*" } | Sort-Object Name | Select-Object -Last 1).FullName
New-Item -ItemType Directory -Force recall_results | Out-Null
Copy-Item $run recall_results\ -Recurse -Force
git switch -c recall-results-1
git add -f recall_results
git commit -m "Recall suite results (leave-one-human-out)"
git push -u origin recall-results-1
git switch recall
```

The files that must be in it: `recall_results\<time>\report.md`, `results.json`, `log.txt`, `label_audit.json`,
`manifest.json`, `table.npz`, the folder `jobs\` (one small `.npz` per trained model, plus a few `.pt`) and the
folder `final\` (three or four sub-folders, each with `gesture.onnx`, `gesture.json`, `gesture.pt`).

Also paste into the chat: the first table of `report.md` and the "label audit" block near the top of `log.txt`.

Run time measured here: the self-test takes 1 minute on a Mac CPU. The 40 minutes for the full run is an
estimate from the earlier 4090 timings (12,000 clips in 5.7 min, one fine-tuning run in 19 s); it has not been
run on a 4090.

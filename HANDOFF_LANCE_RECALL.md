# Handoff for Lance's RTX 4090 box: gesture recall work

For Lance (and Lance's bot). Run the steps in order, from the repo folder on the 4090 box
(`C:\Users\lance\sonar-hack`). Commit as yourself, no AI attribution lines. Every step says how long it takes
and exactly what to push back.

Why: the model scored 0.30 "gestures recognised" on the held-out person. Half of that test was mahimn's first
gesture session, and its swipes go the opposite way to their prompts (27 of 30, measured on the audio), so 30 of
the 81 test gestures could never count. The round-2 runs (0.31 / 0.32) were scored on the same 81. Step B scores
every recipe again with that session's labels fixed, holding out each person in turn.

---

## Step A — send back the held-out run's files (about 1 minute, no GPU)

The missing session is already back (Lance committed it as `mac/excluded/20261114-001248_mahimn_gestures` on
`model-sat-am`; checked here on the audio: its swipes do go the opposite way to their prompts, 27 of 30).
What is still useful: the held-out models of the earlier runs, so they can be re-scored with the labels fixed.

```powershell
cd C:\Users\lance\sonar-hack
git fetch origin
git switch model-sat-am
git switch -c recall-data-1
New-Item -ItemType Directory -Force recall_handoff | Out-Null
foreach ($d in "train_out", "train_out_formula") {
  Get-ChildItem $d\runs -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $dst = "recall_handoff\$d" + "_" + $_.Name
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item "$($_.FullName)\log.txt", "$($_.FullName)\summary.json" $dst -ErrorAction SilentlyContinue
    Copy-Item "$($_.FullName)\sweep" $dst -Recurse -ErrorAction SilentlyContinue
  }
}
git add -f recall_handoff
git commit -m "Held-out run files of the earlier training runs"
git push -u origin recall-data-1
```

Push back: branch **`recall-data-1`** with the folder `recall_handoff/` (each run's `log.txt`, `summary.json`
and `sweep\w24_s0\gesture.onnx` + `gesture.json` + `metrics.json` + `confusion.txt`). A few MB.

**Important for every other training command on this box:** `--data mac` still finds the flipped session,
because `mac\excluded\` is inside `mac\`. Until it is fixed, train with `--data mac\data` (or use the suite
below, which exchanges that session's left/right labels itself).

---

## Step B — the leave-one-person-out experiment suite (GPU, about 3 hours with the GPU to itself)

What it does: trains 36 recipes ("arms"), each one three times over with a different person held out
(lance+lance2 count as one person), 3 seeds each, and scores every one the same honest way. It checks every
session's left/right labels against the others and exchanges them where the swipes measurably go the
opposite way to the prompts (today that is mahimn's first session only; the log's "label audit" block shows
every session), and exports the best recipes as engine-ready models. It is one command; everything lands in
`train_out\recall\suite\<time>\`. Nothing in `models\` or `models-release*\` is touched. The synthetic data
is made on the GPU with the unchanged generator (the picture check that already passed on this box is reused,
so it should not run again).

Thirteen more recipes use the simulator fitted to this MacBook. They run by themselves if
`train_out\synth_cal\macbook_v*\` already exists (the simulator handoff, `HANDOFF_LANCE_CALIBRATION.md`,
makes it) and are skipped otherwise; Step C adds them later.

The GPU-against-CPU picture check runs again this time (two engine files it covers gained an option that is off
by default; about 5 minutes). If it does not pass, the suite says so, makes its synthetic data with the CPU
generator instead (about 15 minutes longer) and carries on: please send `parity.txt` back with the results.

Needs branch `recall` (Mahimn pushes it).

```powershell
cd C:\Users\lance\sonar-hack
git fetch origin
git switch recall
git pull

# The recordings are not part of this branch. Put them into the working folder (they stay uncommitted here):
git restore --source origin/model-sat-am --worktree -- mac/data mac/excluded
dir mac\data, mac\excluded        # expect 12 session folders + 1 (20261114-001248_mahimn_gestures)

# 2-minute self-test on the CPU first (its numbers mean nothing; it must end with "DONE")
powershell -ExecutionPolicy Bypass -File scripts\recall_suite.ps1 --device cpu --smoke

# the real run (about 3 hours: ~17 min checks and data, ~2.5 h training 540 small models 4 at a time, a few min export)
powershell -ExecutionPolicy Bypass -File scripts\recall_suite.ps1
```

If it stops with an error: run it again with the same folder, `--stamp <time>` (the folder name it printed); it
continues where it stopped. If it stops again, push what exists anyway (below) so the log comes back.

**What happened on 2026-11-14 (run `20261114-061651`), and what the suite now does about it.** Every big training
run took about 73 s until about 08:25, then 260 to 4,900 s; one of the four worker processes never came back and its
31 remaining runs were lost (509 of 540 arrived). From about 10:45 the other three ran at 38 s per run. Lance's
gap fill (11:22, `--stamp 20261114-061651 --workers 2`, 501 s) trained the 31: **Step B is complete, 540 of 540.** The cause is
not known from the Mac; the likely one is GPU memory filling up, which on Windows does not fail but spills into
normal memory and slows every run on the card. Since then (branch `recall`, after fcaeb40):

- every worker hands unused GPU memory back after each run and prints what it still holds (`gpu: peak .. GB, kept .. GB`);
- a worker that finishes no run for 15 minutes is stopped and its remaining runs restart in a fresh process
  (`--stall-minutes N` changes that; the log says "worker k finished no run for ...").

If a run still crawls, check `nvidia-smi` for another program on the GPU, and send back the `worker_folds_*.txt` files.

Push back (branch **`recall-results-1`**), about 45 MB:

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

Run time: the self-test takes 2-3 minutes on a Mac CPU. The full run was first estimated at 70 minutes; measured on
the 4090 on 2026-11-14 it needs about 3 hours with the GPU to itself (small runs 33-45 s each, big ones about 73 s,
four at a time).

---

## Step B2 — not needed any more

Step C's run of 12:46 (branch `recall-results-2`) built the final models of `share50_none50` and `real_pre`. The
third candidate, `share75_aux`, is weak over three people (120/206) and is not worth GPU time.

---

## Step C — add the recipes on the fitted simulator (GPU, about 1.5 to 2 hours with nothing else on the GPU)

**State on 2026-11-14 14:17.** All 20 fitted sets exist: Lance built 12 at 13:15-13:40 (`780ae3a`,
`round3/r3_sets.log`) and the 8 `_rel12` sets at 13:43-13:59 (`19dedf0`, `round3/r3_sets_rel12.log`), in
`C:\Users\lance\sonar-hack\train_out\synth_cal`. So all 13 recipes train in one pass.

| sets (each as all / no_lance / no_aarsh / no_mahimn) | recipes |
|---|---|
| `macbook_v2` | cal50, cal25, cal50_aux, cal_pre_real |
| `macbook_v2_motion` | calm50, calm50_aux |
| `macbook_v2_bg` | calbg50, calbg50_aux, calbg25_aux |
| `macbook_v2_rel12` | cal50_aux_rel12 |
| `macbook_v2_bg_rel12` | calbg50_aux_rel12, calbg25_aux_rel12, calbg75_aux_rel12 |

**Is it worth the GPU time?** Yes, as the one open lead. Step B found no recipe that clearly beats the shipped model
(`.runs/recall_partial_ranking.md` on the Mac). Round 3 part B's single held-out run then showed the fitted
simulator at 32/36 on Mahimn's second session against 26/36 for the old simulator, at 5 against 2 false triggers
of 101 (`models-release-cal/TRAINING_RUN.json`). Step C repeats that comparison with three seeds, all three people
and the same test as the shipped model. Nothing of Step B is redone.

It must run in the folder that holds the complete run (`C:\Users\lance\sonar-wt-recall`), or the same command
starts from zero (3 hours). Run nothing else on the GPU meanwhile (no set generation); the run time assumes it.

```powershell
cd C:\Users\lance\sonar-wt-recall
@(Get-ChildItem train_out\recall\suite\20261114-061651\jobs\*.npz).Count    # must print 560 or more (2 + 540 + 18); a small number = not the folder with the complete run: stop
git fetch origin
git merge origin/recall

# the sets must be visible HERE as train_out\synth_cal: a link to where they were built, not a copy (many GB)
if (@(Get-ChildItem train_out\synth_cal\*\*\dataset.npz -ErrorAction SilentlyContinue).Count -lt 20) {
  if (Test-Path train_out\synth_cal) { cmd /c rmdir train_out\synth_cal }   # takes away a link or an empty folder only, never files
  cmd /c mklink /J train_out\synth_cal C:\Users\lance\sonar-hack\train_out\synth_cal
}
# while that link exists, never delete train_out here with Remove-Item -Recurse (PowerShell would empty the sets it points to)
@(Get-ChildItem train_out\synth_cal\*\*\dataset.npz).Count                   # must print 20; if less, stop and send back what `dir train_out\synth_cal` prints

powershell -ExecutionPolicy Bypass -File scripts\recall_suite.ps1 --stamp 20261114-061651 --workers 3
```

The log must show these lines, in this order, and no line `arm ...: skipped`:

- `calibrated simulator data: family macbook_v2 (20 sets found)`
- `=== pretrain: 22 training runs, 20 to do, 3 at a time on cuda`
- `=== folds: 735 training runs, 195 to do, 3 at a time on cuda`
- `DONE`

If it says the folder "was built from other sessions, labels or code", stop and send that line back instead of
starting a fresh run. If single runs take longer than 6 minutes (`worker_folds_*.txt` in the run folder prints one
`done ... s` line per run), stop it with Ctrl+C and run the same line with `--workers 2`; it continues where it
stopped.

Run time: about 1.5 to 2 hours. This is an estimate from this run's own timings, not a measurement: 20
pre-training runs of 90 to 120 s and 195 runs of 40 to 95 s, three at a time (three at a time ran at 38 s per run
this morning, on sets less than half this size).

Push back (branch **`recall-results-3`**; `recall-results-2` is taken by the run of 12:46): the same folder again,
about 135 MB (104 MB today).

```powershell
$run = "train_out\recall\suite\20261114-061651"
Remove-Item recall_results -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force recall_results | Out-Null
Copy-Item $run recall_results\ -Recurse -Force
git switch -c recall-results-3
git add -f recall_results
git commit -m "Recall suite results with the fitted-simulator recipes"
git push -u origin recall-results-3
git switch -
```

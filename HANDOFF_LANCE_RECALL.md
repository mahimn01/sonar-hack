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

## Step C — add the recipes on the fitted simulator (GPU, about 1.5 hours with the GPU to itself)

**Is it worth the GPU time?** Step B is complete (540 of 540) and its answer is that 36 recipes cannot be told
apart on these three people, and none clearly beats the model already shipped
(`.runs/recall_partial_ranking.md` on the Mac). Step C adds 13 recipes that train on the simulator fitted to this
MacBook. Run it if the question "does the fitted simulator help?" is wanted for the write-up; do not expect it to
change the shipped model. Nothing of Step B has to be redone.

It needs the calibrated sets from `HANDOFF_LANCE_CALIBRATION.md`
(`train_out\synth_cal\macbook_v*\...\dataset.npz`). **The run of 12:46 found none** ("family macbook_v2 (0 sets
found)") and trained nothing new; run this step again only when `dir train_out\synth_cal` below lists the sets.

Which sets feed which recipes (the suite trains what it finds and names the rest as "skipped"; running the step
again later adds them):

| sets | recipes | state on 2026-11-14 13:00 (GPU session) |
|---|---|---|
| `macbook_v2` | cal50, cal25, cal50_aux, cal_pre_real | not affected by the hang |
| `macbook_v2_motion` | calm50, calm50_aux | not affected |
| `macbook_v2_rel12` | cal50_aux_rel12 | made by part E |
| `macbook_v2_bg`, `macbook_v2_bg_rel12` | calbg50, calbg50_aux, calbg25_aux and the three `calbg*_rel12` | blocked until the set generator's fix is pulled and parts C and E are rerun |

**It must run in the folder that holds the complete run**, or the same command starts from zero (3 hours). The
gap fill's log names it: `C:\Users\lance\sonar-wt-recall` (the first run was in `sonar-wt-recall-suite`). The
first lines below check that instead of trusting the name.

```powershell
# which folder holds the complete run? (554 files = 2 + 540 + 12)
Get-ChildItem C:\Users\lance -Directory -Filter "sonar*" | ForEach-Object { $j = Join-Path $_.FullName "train_out\recall\suite\20261114-061651\jobs"; if (Test-Path $j) { "{0}   {1} result files" -f $_.FullName, (Get-ChildItem $j -Filter *.npz).Count } }
cd C:\Users\lance\sonar-wt-recall         # the folder printed with 554 (change this line if another one has them)
git fetch origin
git merge origin/recall                   # newest suite code (watchdog, memory release); the recordings in mac\ stay

# The calibrated sets must be visible HERE as train_out\synth_cal. The calibration handoff may have made them in
# another folder: find the one that holds finished sets and link it in instead of copying about 8 GB.
$cal = Get-ChildItem C:\Users\lance -Directory -Filter "sonar*" | ForEach-Object { Join-Path $_.FullName "train_out\synth_cal" } | Where-Object { (Test-Path $_) -and (Get-ChildItem $_ -Recurse -Filter dataset.npz -ErrorAction SilentlyContinue | Select-Object -First 1) } | Select-Object -First 1
$cal                                      # must print a folder; nothing printed = the calibrated sets are not on this box yet
if ($cal -and -not (Test-Path train_out\synth_cal)) { cmd /c mklink /J train_out\synth_cal "$cal" }
dir train_out\synth_cal                  # expect macbook_v2, macbook_v2_motion, macbook_v2_bg, macbook_v2_rel12, macbook_v2_bg_rel12; each with all / no_lance / no_aarsh / no_mahimn

powershell -ExecutionPolicy Bypass -File scripts\recall_suite.ps1 --stamp 20261114-061651
```

What to expect in the log: "calibrated simulator data: family macbook_v2 (.. sets found)", then
"folds: 735 training runs, 195 to do" (13 recipes x 15; fewer if a set is missing, and the log then names the
recipe as "skipped"), and "DONE" at the end. If it says the folder "was built from other sessions, labels or code", stop and send that line back
instead of starting a fresh run.

Push back (branch **`recall-results-3`**; `recall-results-2` is taken by the run of 12:46): the same folder again.

```powershell
$run = "train_out\recall\suite\20261114-061651"
Remove-Item recall_results -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force recall_results | Out-Null
Copy-Item $run recall_results\ -Recurse -Force
git switch -c recall-results-3
git add -f recall_results
git commit -m "Recall suite results with the added recipes"
git push -u origin recall-results-3
git switch -
```

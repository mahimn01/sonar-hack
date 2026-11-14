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

**If it is running but crawling** (seen on 2026-11-14: run `20261114-061651` did every big training run in about
73 s until about 08:25, then 260 to 1,800 s each on all four workers; the partial push at 10:11 had 413 of 540
runs). The cause is not known from the Mac. Check, in the folder the suite runs in (its log says
`C:\Users\lance\sonar-wt-recall-suite`):

```powershell
nvidia-smi                                                                    # is another program using the GPU? is memory near 24 GB?
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
Get-Content train_out\recall\suite\20261114-061651\worker_folds_0.txt -Tail 3   # the last finished runs and their seconds
```

- Another job is on the GPU: let one of the two finish first (the suite needs about 40 more minutes alone).
- Only the suite's four python processes are there and GPU memory is full: stop the suite (Ctrl+C), then start
  it again with two workers. It keeps everything already trained and does only what is missing:
  `powershell -ExecutionPolicy Bypass -File scripts\recall_suite.ps1 --stamp 20261114-061651 --workers 2`
- Either way, when it ends with "DONE", push the whole folder again with the commands below, with the branch
  name changed to **`recall-results-1b`** in all three places (the partial push already took `recall-results-1`).

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

## Step C — add what is new to the same run (GPU, about 1 to 1.5 hours with the GPU to itself)

Do this when either is true:
- `HANDOFF_LANCE_CALIBRATION.md` (the simulator session's handoff) has written
  `train_out\synth_cal\macbook_v*\...\dataset.npz` after Step B ran (Step B's log then says "arm cal50: skipped");
- branch `recall` got new recipes after Step B ran (`git pull` brings changes under `recall/`).

It runs the suite again INTO THE SAME FOLDER: everything already trained is kept and skipped, only the missing
recipes are trained, and the report is rebuilt with all of them. (If the suite says the folder "was built from
other sessions, labels or code", run the Step B command instead: it starts a fresh folder.)

```powershell
cd C:\Users\lance\sonar-hack
git switch recall
git pull
dir train_out\synth_cal           # if the calibrated sets exist: macbook_v1 or macbook_v2, plus _motion and _bg, each with all / no_lance / no_aarsh / no_mahimn
$stamp = (Get-ChildItem train_out\recall\suite -Directory | Where-Object { $_.Name -notlike "smoke_*" } | Sort-Object Name | Select-Object -Last 1).Name
powershell -ExecutionPolicy Bypass -File scripts\recall_suite.ps1 --stamp $stamp
```

Push back (branch **`recall-results-2`**): the same folder again, same commands as in Step B with the branch
name changed:

```powershell
$run = (Get-ChildItem train_out\recall\suite -Directory | Where-Object { $_.Name -notlike "smoke_*" } | Sort-Object Name | Select-Object -Last 1).FullName
Remove-Item recall_results -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force recall_results | Out-Null
Copy-Item $run recall_results\ -Recurse -Force
git switch -c recall-results-2
git add -f recall_results
git commit -m "Recall suite results with the added recipes"
git push -u origin recall-results-2
git switch recall
```

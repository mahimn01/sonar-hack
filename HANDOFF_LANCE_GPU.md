# Handoff: the GPU data pipeline on the RTX 4090 (for Lance)

Everything below runs on the 4090 box, from the repo folder, in **Windows PowerShell**. It assumes
`TRAINING.md` steps 1-3 are done (the `.venv` exists and `torch.cuda.is_available()` is `True`).
Nothing here was run on a CUDA GPU yet: the code was written and checked on a Mac without one, so the
first three steps are also the test of the GPU code. Each step says what you should see and what to send
back to Mahimn. If a step fails, the last section says what to do; training never depends on the new code
(`--gen-device cpu` is the old pipeline, unchanged).

What is new: the synthetic clips and the pictures of the recordings are now computed on the GPU instead of
on every CPU core, the slowest loop (the echo of every body part) is one hand-written CUDA kernel, the
pre-training runs while the next clips are made, and the sweep runs four models at a time.
`TRAINING.md` section 11 explains it; this file is only the commands.

## Round 2 (after your first run, Nov 14 about 03:00): what it showed and what to run now

Your run (commit 8fef500) did what it should: the picture check passed on the 4090 (`PARITY OK`, 1,000 clips), the
data was made 5.8x faster than on the 24 CPU threads, and it told us three things:

| what your numbers said | what changed since (pull to get it) |
|---|---|
| the echo loop is 14.3 of 18 ms per clip on the GPU | it is now one CUDA kernel (step 1 below is its self-check) |
| GPU load only 61 %; end to end 33 clips/s although rendering alone did 55; random draws only 103 clips/s with 24 processes | the GPU now renders in its own thread while the main program keeps the CPU workers fed, and the workers hand their results over through shared memory instead of one pipe (your 103 clips/s was that pipe). The bench line `END TO END (plans via shared memory)` and the GPU load next to it show what that bought |
| held-out person: 0.30 of gestures recognised, although 0.73 on held-out synthetic clips | half of the synthetic gestures now replay REAL hand paths (section 4b). In a test here, a model trained on formula gestures recognised 0.26 of real-path gestures, one trained on the mix 0.51 |

Please run, in this order, and send back what each section below asks for:
```powershell
git pull
.venv\Scripts\python -m datagen.echo_kernel                                         # step 1: one line, KERNEL OK or the reason
.venv\Scripts\python scripts\check_gpu_parity.py --json train_out\parity\manual.json # step 2: must end with PARITY OK
.venv\Scripts\python scripts\bench_gpu.py                                          # step 3: the table
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --data mac --val-person mahimn --release models-release-mac
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --data mac --val-person mahimn --motion none --out train_out_formula --models train_out_formula\models --release train_out_formula\models-release
```
The last two are the same training with and without the real hand paths. The second writes everything under
`train_out_formula\`, so it does not replace the first model.
**The two `Honest score` lines side by side are the most useful thing to send back.**
Close the other GPU programs first if you can (your bench listed Medal, Overwolf, Discord, Epic and others on
the card): they take GPU time and memory.

## 0. Get the code

```powershell
git pull
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: something like `2.15.0+cu132 True NVIDIA GeForce RTX 4090`. Close every other program that
uses the GPU (games, browsers with video, other training).

## 1. The CUDA kernel's self-check (about 1 minute)

```powershell
.venv\Scripts\python -m datagen.echo_kernel
```
It compiles `datagen\echo_paths.c` for the GPU (with the compiler library that ships inside PyTorch; no
Visual Studio, no CUDA toolkit needed), renders 6 clips with it and with the plain PyTorch code, and compares.

Expected, one line:
```
KERNEL OK (CUDA kernel (NVRTC), 6 clips): echoes within 3e-07 of the PyTorch path; echo loop 0.4 ms per clip against 8 ms
```
(the two times are guesses; yours are the first real ones). **Send this line back.**

If it says `KERNEL NOT USED: <reason>`: not a problem for training. Everything below then uses the plain
PyTorch code automatically (same pictures, slower). Send the line; the reason tells us what to fix.
Optional second route if the reason mentions NVRTC: `.venv\Scripts\python -m pip install cupy-cuda12x`
(for PyTorch `+cu12x`; `cupy-cuda13x` for `+cu13x`), delete `%TEMP%\sonar_echo_kernel_*.json`, run step 1 again.

## 2. The picture check (1-3 minutes)

```powershell
.venv\Scripts\python scripts\check_gpu_parity.py --json train_out\parity\manual.json
```
Makes 1,000 fixed clips with the old CPU code and with the GPU code and compares them (audio before and
after rounding, movement bursts, labels, pictures).

Expected at the end:
```
check                                                                   worst     median      limit  result
same number of redraws                                                      0          0          0  ok
moving echoes, before rounding                                          6e-07      4e-07      2e-05  ok
...                                                                                                  ok
same bursts and labels                                                  0.002      0.002       0.02  ok
pictures: largest single pixel (Doppler pictures)                        0.03      7e-05       0.12  ok
...
stored CPU results: close - not bit-identical to the stored CPU results but within rounding (another computer or library version)
echo loop: fused CUDA kernel
PARITY OK: the GPU path makes the same training pictures as the CPU path (1000 clips, ... s)
```
Every line must say `ok`; `stored CPU results` may say `same` or `close` (both fine; `changed` is not).
**Send back the whole table and `train_out\parity\manual.json`.**
`PARITY FAILED`: go to the last section; do not train with GPU data.

## 3. The speed test (3-6 minutes)

```powershell
.venv\Scripts\python scripts\bench_gpu.py
```
Times the old CPU code, then the GPU path (with the kernel, without it, and with float32 FFTs), then the
training step. Expected shape of the output:
```
1. CPU reference (numpy): ...   ~1000 ms of CPU per clip on one core; with 16 processes: ~15 clips/s
2. GPU clip maker on cuda, number format 'ref', 384 clips ...
   echo loop: fused CUDA kernel (...)
     random draws (CPU, 16 processes)              ... clips/s
     rendering (GPU)                               ... clips/s
        echo paths                                 ... ms/clip      <- the kernel
        direct sound + room (FFT)                  ... ms/clip
        ...
     END TO END                                    ... clips/s   GPU load: mean .. %, memory peak .. GB
     against the CPU reference on every core (.. clips/s): ..x
2. ... fused kernel OFF ...                        (the same with plain PyTorch, to see what the kernel buys)
2. ... number format 'fast' ...
3. training step on cuda ...  width 24: .. ms per step of 256 = ...,... examples/s (bf16)
saved train_out\bench\bench_<time>.json
```
**Send back the whole printed output and the `bench_<time>.json` file.** These are the first real numbers
for this machine; they decide what gets optimised next (if "random draws" is the slowest line, the CPU is
the limit and the next step is to move the noise shaping to the GPU too).

## 4. Smoke test, then the real run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --size tiny          # about 2 minutes, junk model under train_out\tiny\
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1                      # the full run
powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --sweep              # + 9 fine-tuning runs, 4 at a time
```
What the log shows that is new:
```
synthetic clips and session pictures are computed on: cuda
GPU parity check: passed earlier for this code on this GPU (1000 clips, ...)      (or it runs it: 1-3 minutes, once)
...
  echo loop: fused CUDA kernel (CUDA kernel (NVRTC): echoes within 3.0e-07 of the PyTorch path; ...)
  GPU path on cuda:0: batch 2xx, precision ref, NN batches, render .. s, features .. s, N redraws
done in .. s: 12000 clips -> ~20000 examples in train_out\synth\<key>
=== pre-training 1 model(s) in the background while the next clips are made ...
```
and the usual `DONE in .. min`, `Honest score ...`, `Verdict: ...` and `Time per step: ...` lines at the end.
**Send back the last 15 lines of the run and `models-release\TRAINING_RUN.json`** (as before, plus the
branch with `models-release\`, `TRAINING.md` step 8).

## 4b. New in the data: real hand movements (nothing to do, just know it)

Half of the synthetic gestures now replay real hand paths tracked from webcam videos of 15 people
(`datagen\motion_library\unica_v1.npz`, 132 kB, comes with `git pull`; CC BY 4.0, credit in the README next
to it). The run log says so near the top:
```
synthetic gestures: 50% replay real hand paths from datagen\motion_library\unica_v1.npz, the rest use the formulas (--motion none: formulas only)
```
- `--motion none` trains on the formula gestures only, exactly as before. If the honest score drops with the
  library, run once with `--motion none` and send both score lines: that comparison on real recordings is the
  one thing that could not be measured without them.
- Nothing has to be downloaded and no video tool has to be installed on the 4090 box.
- The picture check (step 2) also runs 40 of these clips through the CPU and the GPU code.

## 5. If something goes wrong

| what you see | do this | and send |
|---|---|---|
| step 1 says `KERNEL NOT USED` | nothing: the PyTorch code is used automatically | the line |
| the run prints `WARNING: the fused kernel failed at run time and was switched off` | nothing for this run (it carried on with PyTorch code); next time add `--gen-kernel off` | the warning line |
| a Python error mentioning `echo_kernel`, `nvrtc` or `cuLaunchKernel` | add `--gen-kernel off` to the command | the error text |
| `PARITY FAILED` in step 2 or inside a run | add `--gen-kernel off` and run step 2 again with `--kernel off`; if it still fails, train with `--gen-device cpu` (the old pipeline) | the table, `train_out\parity\*.json` |
| `... batch(es) did not fit in GPU memory and were split` | nothing (it halved the batch itself); to avoid the retries add `--gen-batch 128` | the line |
| `CUDA out of memory` during training | `--sweep-jobs 2` (or `1`), `--no-overlap` | the last 30 lines |
| the machine is needed for something else during a run | `--workers 8` limits the CPU processes | |
| anything else odd in the GPU data path | `--gen-device cpu` = exactly the old CPU pipeline | the log `train_out\runs\<time>\log.txt` |
| the score is worse than an earlier run on the same recordings | `--motion none` (formula gestures only), compare | both `Honest score` lines |

Every option, one line each: `powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --help`.

## 6. On a rented Linux GPU box (H100 etc.)

The same code; all visible GPUs are used automatically (clips are rendered on every GPU, training runs are
dealt out in turn; `--gen-gpus N` limits it).
```bash
git clone https://github.com/mahimn01/sonar-hack.git && cd sonar-hack
python3.12 -m venv .venv && .venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126      # the index that matches `nvidia-smi`
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python -m datagen.echo_kernel && .venv/bin/python scripts/check_gpu_parity.py && .venv/bin/python scripts/bench_gpu.py
scripts/train_4090.sh --sweep --compile        # --compile (torch.compile) works on Linux; it falls back by itself if it cannot
```
Copy `data/` there first (zip, `TRAINING.md` step 5) and bring `models-release/` back.


## 7. On a MacBook (Apple GPU)

By default nothing changes there: `scripts/train_4090.sh --device mps ...` still makes the data on the CPU
cores. `--gen-device mps` moves it to the Apple GPU. That was run on an M1 Max: it passes the picture check
with looser limits (Apple GPUs have no 64-bit numbers, so about 1 clip in 13 gets a movement burst moved by a
frame) and makes 13.7 clips a second against 6.8 on six CPU cores. Worth it only when the synthetic set must
be remade often; on a MacBook with 16 GB add `--gen-batch 8`.
```bash
.venv/bin/python scripts/check_gpu_parity.py --device mps --clips 120      # must end with PARITY OK
scripts/train_4090.sh --device mps --gen-device mps --size small --release models-release-mac
```

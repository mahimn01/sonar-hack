# recall_suite.ps1 - run the leave-one-person-out experiment suite on the RTX 4090 box (Windows).
# What it does: checks the Python environment, then runs `python -m recall.suite run` with whatever
# options you pass (default: the full run on CUDA, about 3 hours as measured on 2026-11-14). It writes only under train_out\recall\.
# How to run (from the repo folder):
#   powershell -ExecutionPolicy Bypass -File scripts\recall_suite.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\recall_suite.ps1 --device cpu --smoke     (2-minute self-test)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "No Python environment at .venv (TRAINING.md, steps 2-3)"; exit 1 }
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
& $py -m recall.suite run @args
exit $LASTEXITCODE

# train_4090.ps1 - one command for the whole gesture-model training (Windows PowerShell; also pwsh on Linux).
#
# What it does: synthetic data -> pre-training -> fine-tuning on every recorded session
# under data\ (if any) -> export to models\ and models-release\ -> engine check.
# All the logic lives in scripts\train_4090.py; this file only finds the project's Python.
#
# How to run, from the repo folder:
#   powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1                          # full run on the GPU
#   powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --sweep                  # + widths x seeds
#   powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --device cpu --size tiny  # smoke test, no GPU
#   powershell -ExecutionPolicy Bypass -File scripts\train_4090.ps1 --help                   # every option
#
# What it changes: train_out\, models\, models-release\. Reads data\ only. No audio devices.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
if ($env:OS -eq "Windows_NT") { $py = Join-Path ".venv" "Scripts\python.exe" } else { $py = Join-Path ".venv" "bin/python" }
if (-not (Test-Path $py)) {
    Write-Host "No Python environment at .venv\ - do TRAINING.md steps 2 and 3 first." -ForegroundColor Red
    exit 1
}
# Windows PowerShell 5.1 turns a native program's stderr lines into errors when output is redirected;
# with "Stop" a warning printed by Python would end the script. The exit code below is what counts.
$ErrorActionPreference = "Continue"
# Typed at a PowerShell prompt, "--val-person ann,bob" arrives as an array; give Python the "ann,bob" it expects.
$pass = @(foreach ($a in $args) { if ($a -is [array]) { $a -join "," } else { "$a" } })
& $py scripts/train_4090.py @pass
exit $LASTEXITCODE

# PIT WALL - start both servers.
#
#   .\run.ps1            start backend + frontend
#   .\run.ps1 -Build     rebuild the demo race first (slow: ~1 hr of CPU inference)
#
# The race data is precomputed, so the normal path starts in seconds and needs
# no network.

param(
    [switch]$Build,
    [string]$Race = "2021_Abu_Dhabi_Grand_Prix"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

if ($Build) {
    Write-Host "Fetching clips for $Race ..." -ForegroundColor Cyan
    python "$root\backend\data\fetch_race_clips.py" $Race

    Write-Host "Stage 1: running models (this takes about an hour, and resumes if interrupted) ..." -ForegroundColor Cyan
    python "$root\backend\data\build_race.py" $Race

    Write-Host "Stage 2: calibrating ..." -ForegroundColor Cyan
    python "$root\backend\data\calibrate.py" $Race
}

$raceFile = Join-Path $root "backend\races\$Race.json"
if (-not (Test-Path $raceFile)) {
    Write-Host "No precomputed race found at $raceFile" -ForegroundColor Yellow
    Write-Host "Run:  .\run.ps1 -Build" -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting backend on http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\backend'; python -W ignore -m uvicorn main:app --port 8000 --reload"
)

Write-Host "Starting frontend on http://localhost:3000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root\frontend'; npm run dev"
)

Write-Host ""
Write-Host "PIT WALL is starting. Open http://localhost:3000" -ForegroundColor Green
Write-Host "(if port 3000 was taken, Next.js will say which port it used)"

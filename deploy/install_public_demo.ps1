param(
    [int]$Port = 8720,
    [int]$IntervalMinutes = 10
)

$ErrorActionPreference = "Stop"
$deployDir = $PSScriptRoot
$vbs = Join-Path $deployDir "run_keepalive_hidden.vbs"
if (-not (Test-Path $vbs)) { throw "missing run_keepalive_hidden.vbs" }

$taskMinute = "SpaceFLPublicDemo"
$taskBoot = "SpaceFLPublicDemoBoot"
# Use wscript.exe (GUI subsystem) + hidden VBS: a .bat task flashes a cmd window every run.
$taskCommand = '"' + (Join-Path $env:WINDIR "System32\wscript.exe") + '" //nologo "' + $vbs + '"'

Write-Host "[1/3] Registering hidden keepalive task..."
schtasks.exe /Delete /TN $taskMinute /F 2>$null | Out-Null
schtasks.exe /Delete /TN $taskBoot /F 2>$null | Out-Null
schtasks.exe /Create /TN $taskMinute /TR $taskCommand /SC MINUTE /MO $IntervalMinutes /RL LIMITED /F | Out-Host

Write-Host "[2/3] Running keepalive once to start server and tunnel..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $deployDir "keepalive_public_demo.ps1") -Port $Port

Write-Host "[3/3] Public URL:"
$urlFile = Join-Path $deployDir "public_url.txt"
if (Test-Path $urlFile) {
    Get-Content $urlFile
} else {
    Write-Host "URL not ready yet. Check deploy\logs\keepalive.log"
}

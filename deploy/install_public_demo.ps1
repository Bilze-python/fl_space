param(
    [int]$Port = 8720
)

$ErrorActionPreference = "Stop"
$deployDir = $PSScriptRoot
$bat = Join-Path $deployDir "run_keepalive.bat"
if (-not (Test-Path $bat)) { throw "missing run_keepalive.bat" }

$taskMinute = "SpaceFLPublicDemo"
$taskBoot = "SpaceFLPublicDemoBoot"
$taskCommand = '"' + $bat + '" ' + $Port

Write-Host "[1/3] Registering scheduled tasks..."
schtasks.exe /Delete /TN $taskMinute /F 2>$null | Out-Null
schtasks.exe /Delete /TN $taskBoot /F 2>$null | Out-Null
schtasks.exe /Create /TN $taskMinute /TR $taskCommand /SC MINUTE /MO 5 /RL LIMITED /F | Out-Host
schtasks.exe /Create /TN $taskBoot /TR $taskCommand /SC ONLOGON /RL LIMITED /F | Out-Host

Write-Host "[2/3] Running keepalive once to start server and tunnel..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $deployDir "keepalive_public_demo.ps1") -Port $Port

Write-Host "[3/3] Public URL:"
$urlFile = Join-Path $deployDir "public_url.txt"
if (Test-Path $urlFile) {
    Get-Content $urlFile
} else {
    Write-Host "URL not ready yet. Check deploy\logs\keepalive.log"
}

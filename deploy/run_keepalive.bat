@echo off
rem Keepalive entry used by Windows Task Scheduler (ASCII only for cmd.exe).
setlocal
set "PORT=%~1"
if "%PORT%"=="" set "PORT=8720"
powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0keepalive_public_demo.ps1" -Port %PORT%
endlocal

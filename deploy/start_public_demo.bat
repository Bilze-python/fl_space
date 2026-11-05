@echo off
rem One-click public demo launcher: starts demo-mode server + Cloudflare quick tunnel,
rem registers keepalive tasks, then prints the public URL.
setlocal
cd /d "%~dp0"
set "PORT=8720"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_public_demo.ps1" -Port %PORT%
echo.
pause
endlocal

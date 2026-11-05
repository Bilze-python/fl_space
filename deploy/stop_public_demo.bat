@echo off
rem Stop public demo: scheduled task, SSH tunnel, demo server, startup shortcut.
setlocal
cd /d "%~dp0"
echo Stopping SpaceFL public demo...
schtasks.exe /Delete /TN "SpaceFLPublicDemo" /F >nul 2>&1
schtasks.exe /Delete /TN "SpaceFLPublicDemoBoot" /F >nul 2>&1
if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\SpaceFLPublicDemo.lnk" (
  del /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\SpaceFLPublicDemo.lnk" >nul 2>&1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'ssh.exe' -and $_.CommandLine -match 'localhost.run') -or ($_.Name -eq 'python.exe' -and $_.CommandLine -match 'create_app' -and $_.CommandLine -match '8720') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
if exist "%~dp0public_url.txt" del /q "%~dp0public_url.txt" >nul 2>&1
echo Done. Public demo stopped.
echo Local full platform on port 8700 is not affected.
pause
endlocal

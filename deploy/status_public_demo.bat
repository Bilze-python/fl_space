@echo off
rem Show public demo status (ASCII only for cmd.exe).
setlocal
cd /d "%~dp0"
echo.
echo === Local demo server (port 8720) ===
curl.exe -s --max-time 5 http://127.0.0.1:8720/api/health
echo.
echo.
echo === Public URL ===
if exist "%~dp0public_url.txt" (
  type "%~dp0public_url.txt"
) else (
  echo Not ready. Run start_public_demo.bat or check deploy\logs\keepalive.log
)
echo.
echo === SSH tunnel process ===
tasklist /FI "IMAGENAME eq ssh.exe" /FO TABLE | findstr /I ssh.exe
echo.
echo === Scheduled task ===
schtasks.exe /Query /TN "SpaceFLPublicDemo" /FO LIST 2>nul | findstr /I "TaskName Status Next Run"
echo.
pause
endlocal

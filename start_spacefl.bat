@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8700/api/health' -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } } catch { exit 1 }"
if errorlevel 1 (
  if exist "%~dp0venv\Scripts\python.exe" (
    powershell -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath '%~dp0venv\Scripts\python.exe' -ArgumentList @('%~dp0web\server.py','--host','127.0.0.1','--port','8700') -WorkingDirectory '%~dp0' -WindowStyle Hidden"
  ) else (
    powershell -NoProfile -WindowStyle Hidden -Command "Start-Process -FilePath 'py' -ArgumentList @('-3.12','%~dp0web\server.py','--host','127.0.0.1','--port','8700') -WorkingDirectory '%~dp0' -WindowStyle Hidden"
  )
  timeout /t 2 /nobreak >nul
)

start "" "http://127.0.0.1:8700/launcher.html"
endlocal

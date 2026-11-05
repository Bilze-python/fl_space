@echo off
rem SpaceFL research platform launcher.
rem NOTE: keep this file ASCII-only (cmd.exe cannot parse UTF-8 Chinese comments safely).
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_EXE="
set "PYTHON_USER_SITE="
set "NPM_CMD="
set "RUN_PANEL=0"
set "PANEL_ARG1="
set "PANEL_ARG2="
set "PORT=8700"
set "STOP_ONLY=0"
set "RESTART_FIRST=0"

rem Arguments: --panel (from control_panel.bat), --stop (stop background servers),
rem --restart (stop old servers first, then start the current version).
if /i "%~1"=="--panel" set "RUN_PANEL=1"
if /i "%~1"=="--panel" set "PANEL_ARG1=%~2"
if /i "%~1"=="--panel" set "PANEL_ARG2=%~3"
if /i "%~1"=="--stop" set "STOP_ONLY=1"
if /i "%~1"=="--restart" set "RESTART_FIRST=1"

if "%STOP_ONLY%"=="1" (
  call :stop_project_servers
  endlocal
  exit /b 0
)
if "%RESTART_FIRST%"=="1" call :stop_project_servers

rem Reuse a healthy current-version SpaceFL server on ports 8700-8710.
rem :try_reuse sets REUSED=1 and PORT when a current server is found.
rem A server is considered current only if both /api/health and
rem /api/research/overview respond, so an old-version process is never reused.
set "REUSED=0"
if "%RUN_PANEL%"=="0" call :try_reuse
if "%REUSED%"=="1" goto :open_launcher

call :find_python
if not defined PYTHON_EXE (
  echo [SETUP] Python 3.9 or newer was not found. Trying Windows Package Manager...
  call :install_python
  call :find_python
)
if not defined PYTHON_EXE goto :python_missing

if not exist "%VENV_PYTHON%" (
  echo [SETUP] Creating the project virtual environment...
  "%PYTHON_EXE%" -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Could not create the project virtual environment.
    echo Make sure this folder is writable, then run this launcher again.
    pause
    exit /b 1
  )
)
set "PYTHON_EXE=%VENV_PYTHON%"
set "PATH=%VENV_DIR%\Scripts;%PATH%"

rem Reuse PyTorch installed for the user's base Python. Virtual environments
rem disable the user site by default, so expose it explicitly to the project.
for /f "delims=" %%P in ('"%PYTHON_EXE%" -c "import site; print(site.getusersitepackages())" 2^>nul') do set "PYTHON_USER_SITE=%%P"
if defined PYTHON_USER_SITE if exist "%PYTHON_USER_SITE%\torch" if exist "%PYTHON_USER_SITE%\torchvision" (
  set "PYTHONPATH=%PYTHON_USER_SITE%;%PYTHONPATH%"
  echo [SETUP] Reusing local PyTorch from "%PYTHON_USER_SITE%".
)

rem The same environment powers both the web server and the control panel.
"%PYTHON_EXE%" -c "import fastapi, uvicorn, numpy, matplotlib, fl_space" >nul 2>&1
if errorlevel 1 (
  echo [SETUP] Installing Python dependencies. The first run may take a few minutes...
  "%PYTHON_EXE%" -m pip install --disable-pip-version-check -e ".[web]"
  if errorlevel 1 (
    echo [ERROR] Python dependency installation failed.
    echo Check the network connection and try again.
    pause
    exit /b 1
  )
)

rem FedLEO validation requires both torch and torchvision. Install the full
rem optional dependencies only when a usable local copy was not found.
"%PYTHON_EXE%" -c "import torch, torchvision" >nul 2>&1
if errorlevel 1 (
  echo [SETUP] PyTorch was not found. Installing FedLEO experiment dependencies...
  "%PYTHON_EXE%" -m pip install --disable-pip-version-check -e ".[full]"
  if errorlevel 1 (
    echo [ERROR] PyTorch installation failed. The Web UI cannot run FedLEO experiments.
    echo Install torch and torchvision for this Python, then run the launcher again.
    pause
    exit /b 1
  )
)

"%PYTHON_EXE%" -c "import torch, torchvision" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] PyTorch or torchvision is not importable in the project environment.
  echo The Web UI cannot run FedLEO experiments until both packages are available.
  pause
  exit /b 1
)

if "%RUN_PANEL%"=="1" goto :run_panel

rem A busy default port is harmless: choose the first free local port.
:find_port
powershell.exe -NoProfile -Command "$l = Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue; if ($l) { exit 1 } else { exit 0 }" >nul 2>&1
if not errorlevel 1 goto :port_ready
set /a PORT+=1
if %PORT% GTR 8710 (
  echo [ERROR] Ports 8700-8710 are already in use.
  echo Run stop_spacefl.bat to stop stale SpaceFL servers, then start again.
  pause
  exit /b 1
)
goto :find_port

:port_ready
if exist "node_modules\d3\dist\d3.min.js" if exist "node_modules\marked\lib\marked.umd.js" if exist "node_modules\world-atlas\countries-110m.json" if exist "node_modules\cesium\Build\Cesium\Cesium.js" goto :web_assets_ready

call :find_npm
if not defined NPM_CMD (
  echo [SETUP] Node.js/npm was not found. Trying Windows Package Manager...
  call :install_node
  call :find_npm
)
if not defined NPM_CMD goto :node_missing

echo [SETUP] Installing Web assets. The first run may take a few minutes...
call "%NPM_CMD%" install --no-audit --no-fund --cache "%VENV_DIR%\npm-cache"
if errorlevel 1 (
  echo [ERROR] Web asset installation failed.
  echo Check the network connection and try again.
  pause
  exit /b 1
)

if not exist "node_modules\d3\dist\d3.min.js" goto :assets_failed
if not exist "node_modules\marked\lib\marked.umd.js" goto :assets_failed
if not exist "node_modules\world-atlas\countries-110m.json" goto :assets_failed
if not exist "node_modules\cesium\Build\Cesium\Cesium.js" goto :assets_failed

:web_assets_ready
echo [START] Starting the SpaceFL Web server on port %PORT%...
rem Log files are named per port (with random fallback) so a previously running
rem server cannot lock them and break the next start.
call :prepare_log_files
powershell.exe -NoProfile -WindowStyle Hidden -Command "$args = @('-m','fl_space.cli','run','serve','--host','127.0.0.1','--port',$env:PORT); $p = Start-Process -FilePath $env:PYTHON_EXE -ArgumentList $args -WorkingDirectory $env:PROJECT_DIR -WindowStyle Hidden -RedirectStandardOutput $env:SPACEFL_OUT -RedirectStandardError $env:SPACEFL_ERR -PassThru; if ($null -eq $p) { exit 1 }" 2>"%VENV_DIR%\spacefl-start.err.log"
if errorlevel 1 (
  echo [ERROR] Could not start the Web server process.
  if exist "%VENV_DIR%\spacefl-start.err.log" (
    echo ---- startup error ----
    type "%VENV_DIR%\spacefl-start.err.log"
    echo ------------------------
  )
  echo If a previous SpaceFL server is still running, run stop_spacefl.bat and try again.
  pause
  exit /b 1
)

rem Give Uvicorn time to import the app, then verify /api/health and the research API.
set /a START_ATTEMPTS=0
:wait_for_server
call :check_health
if not errorlevel 1 goto :server_ready
set /a START_ATTEMPTS+=1
if %START_ATTEMPTS% GEQ 45 goto :server_failed
ping.exe -n 2 127.0.0.1 >nul
goto :wait_for_server

:server_ready
goto :open_launcher

:server_failed
echo [ERROR] The Web server did not become ready within 45 seconds.
echo ---- server error log tail ----
if exist "%SPACEFL_ERR%" powershell.exe -NoProfile -Command "Get-Content -LiteralPath $env:SPACEFL_ERR -Tail 20"
echo ----------------------------
echo See "%SPACEFL_ERR%" for details.
echo If port %PORT% is held by a stale SpaceFL server, run stop_spacefl.bat and try again.
pause
exit /b 1

:open_launcher
echo [READY] Opening http://127.0.0.1:%PORT%/
start "" "http://127.0.0.1:%PORT%/#overview"
endlocal
exit /b 0

:run_panel
echo [START] Opening the SpaceFL Control Panel...
"%PYTHON_EXE%" control_panel.py %PANEL_ARG1% %PANEL_ARG2%
set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo [INFO] Control Panel closed with code %EXIT_CODE%.
pause
endlocal & exit /b %EXIT_CODE%

:python_missing
echo [ERROR] Python 3.9 or newer is required.
echo Install Python from https://www.python.org/downloads/windows/ and run this launcher again.
echo If winget is available, rerun the launcher to try the automatic installation.
pause
exit /b 1

:node_missing
echo [ERROR] Node.js LTS and npm are required for the Web assets.
echo Install Node.js LTS from https://nodejs.org/ and run this launcher again.
echo If winget is available, rerun the launcher to try the automatic installation.
pause
exit /b 1

:assets_failed
echo [ERROR] Required Web assets are still missing after npm installation.
echo Delete node_modules and run this launcher again if the installation was interrupted.
pause
exit /b 1

:find_python
if defined PYTHON_EXE exit /b 0
if exist "%VENV_PYTHON%" call :accept_python "%VENV_PYTHON%"
if defined PYTHON_EXE exit /b 0
for /f "delims=" %%P in ('where.exe python 2^>nul') do call :accept_python "%%P"
if defined PYTHON_EXE exit /b 0
for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do call :accept_python "%%P"
if defined PYTHON_EXE exit /b 0
for /d %%D in ("%USERPROFILE%\.pyenv\pyenv-win\versions\*") do if exist "%%~fD\python.exe" call :accept_python "%%~fD\python.exe"
if defined PYTHON_EXE exit /b 0
for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do if exist "%%~fD\python.exe" call :accept_python "%%~fD\python.exe"
if defined PYTHON_EXE exit /b 0
for /d %%D in ("%ProgramFiles%\Python3*") do if exist "%%~fD\python.exe" call :accept_python "%%~fD\python.exe"
exit /b 0

:accept_python
if defined PYTHON_EXE exit /b 0
if not exist "%~1" exit /b 0
if /i "%~x1"==".bat" call "%~1" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if /i "%~x1"==".cmd" call "%~1" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if /i not "%~x1"==".bat" if /i not "%~x1"==".cmd" "%~1" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_EXE=%~1"
exit /b 0

:find_npm
if defined NPM_CMD exit /b 0
for /f "delims=" %%N in ('where.exe npm.cmd 2^>nul') do call :accept_npm "%%N"
if defined NPM_CMD exit /b 0
if exist "%ProgramFiles%\nodejs\npm.cmd" call :accept_npm "%ProgramFiles%\nodejs\npm.cmd"
if defined NPM_CMD exit /b 0
if exist "%LocalAppData%\Programs\nodejs\npm.cmd" call :accept_npm "%LocalAppData%\Programs\nodejs\npm.cmd"
if defined NPM_CMD exit /b 0
if exist "%AppData%\npm\npm.cmd" call :accept_npm "%AppData%\npm\npm.cmd"
exit /b 0

:accept_npm
if defined NPM_CMD exit /b 0
if exist "%~1" set "NPM_CMD=%~1"
exit /b 0

:install_python
where.exe winget.exe >nul 2>&1
if errorlevel 1 exit /b 0
echo [SETUP] Installing Python 3.12 through winget...
winget.exe install --id Python.Python.3.12 --exact --accept-package-agreements --accept-source-agreements
exit /b 0

:install_node
where.exe winget.exe >nul 2>&1
if errorlevel 1 exit /b 0
echo [SETUP] Installing Node.js LTS through winget...
winget.exe install --id OpenJS.NodeJS.LTS --exact --accept-package-agreements --accept-source-agreements
exit /b 0

rem ==============================================================
rem Helper subroutines
rem ==============================================================

rem Try to reuse a running current-version SpaceFL server.
rem Sets REUSED=1 and PORT when found; always returns 0.
:try_reuse
set "FOUND_PORT="
for /f "delims=" %%R in ('powershell.exe -NoProfile -Command "$found = ''; foreach ($p in 8700..8710) { try { $h = Invoke-WebRequest -UseBasicParsing -Uri ('http://127.0.0.1:' + $p + '/api/health') -TimeoutSec 1; $b = $h.Content | ConvertFrom-Json; if ($h.StatusCode -eq 200 -and $b.status -eq 'ok' -and $b.service -match 'SpaceFL') { $r = Invoke-WebRequest -UseBasicParsing -Uri ('http://127.0.0.1:' + $p + '/api/research/overview') -TimeoutSec 2; if ($r.StatusCode -eq 200) { $found = $p; break } } } catch {} }; if ($found) { Write-Output $found }" 2^>nul') do set "FOUND_PORT=%%R"
if not defined FOUND_PORT goto :try_reuse_none
set "PORT=%FOUND_PORT%"
set "REUSED=1"
echo [INFO] Reusing running SpaceFL server: http://127.0.0.1:%FOUND_PORT%/
exit /b 0

:try_reuse_none
set "PORT=8700"
call :check_legacy_health
if not errorlevel 1 (
  echo [WARN] Port 8700 is served by an old SpaceFL version without the research API.
  echo [WARN] A new server will be started on the next free port.
  echo [WARN] Run stop_spacefl.bat to stop the old server if it is no longer needed.
)
exit /b 0

rem Check whether %PORT% serves the current SpaceFL platform
rem (/api/health OK AND /api/research/overview reachable).
:check_health
powershell.exe -NoProfile -Command "$ok = $false; try { $h = Invoke-WebRequest -UseBasicParsing -Uri ('http://127.0.0.1:' + $env:PORT + '/api/health') -TimeoutSec 2; $b = $h.Content | ConvertFrom-Json; if ($h.StatusCode -eq 200 -and $b.status -eq 'ok' -and $b.service -match 'SpaceFL') { $r = Invoke-WebRequest -UseBasicParsing -Uri ('http://127.0.0.1:' + $env:PORT + '/api/research/overview') -TimeoutSec 3; if ($r.StatusCode -eq 200) { $ok = $true } } } catch {}; if ($ok) { exit 0 } else { exit 1 }" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

rem Check whether %PORT% serves any SpaceFL version (health only).
:check_legacy_health
powershell.exe -NoProfile -Command "$ok = $false; try { $h = Invoke-WebRequest -UseBasicParsing -Uri ('http://127.0.0.1:' + $env:PORT + '/api/health') -TimeoutSec 2; $b = $h.Content | ConvertFrom-Json; if ($h.StatusCode -eq 200 -and $b.status -eq 'ok' -and $b.service -match 'SpaceFL') { $ok = $true } } catch {}; if ($ok) { exit 0 } else { exit 1 }" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

rem Stop background SpaceFL servers started from this project's venv.
:stop_project_servers
echo [STOP] Stopping SpaceFL background servers of this project...
powershell.exe -NoProfile -Command "$root = $env:PROJECT_DIR.TrimEnd('\'); $killed = @(); Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -and $_.CommandLine -like ('*' + $root + '*') -and ($_.CommandLine -match 'fl_space\.cli' -or $_.CommandLine -match 'web\.server') } | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; $killed += $_.ProcessId } catch {} }; if ($killed.Count -gt 0) { Write-Host ('[STOP] Stopped process id(s): ' + ($killed -join ', ')) } else { Write-Host '[STOP] No running SpaceFL server found.' }"
exit /b 0

rem Prepare log files for this start: per-port names, unique suffix when locked.
:prepare_log_files
set "SPACEFL_OUT=%VENV_DIR%\spacefl-web.%PORT%.out.log"
set "SPACEFL_ERR=%VENV_DIR%\spacefl-web.%PORT%.err.log"
del /q "%SPACEFL_OUT%" "%SPACEFL_ERR%" >nul 2>&1
copy /y nul "%SPACEFL_OUT%" >nul 2>&1
if errorlevel 1 goto :prepare_log_files_fallback
copy /y nul "%SPACEFL_ERR%" >nul 2>&1
if errorlevel 1 goto :prepare_log_files_fallback
exit /b 0

:prepare_log_files_fallback
set "SPACEFL_SUFFIX=%RANDOM%%RANDOM%"
set "SPACEFL_OUT=%VENV_DIR%\spacefl-web.%PORT%.%SPACEFL_SUFFIX%.out.log"
set "SPACEFL_ERR=%VENV_DIR%\spacefl-web.%PORT%.%SPACEFL_SUFFIX%.err.log"
exit /b 0

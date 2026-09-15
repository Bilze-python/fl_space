@echo off
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

rem control_panel.bat calls this file with --panel. Keep its optional --lang argument.
if /i "%~1"=="--panel" (
  set "RUN_PANEL=1"
  set "PANEL_ARG1=%~2"
  set "PANEL_ARG2=%~3"
)

rem Reuse a healthy SpaceFL server when one is already running.
if "%RUN_PANEL%"=="0" (
  powershell.exe -NoProfile -Command "$r = $null; $body = $null; try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8700/api/health' -TimeoutSec 1; $body = $r.Content | ConvertFrom-Json } catch {}; if ($r -and $r.StatusCode -eq 200 -and $body.service -eq 'SpaceFL Visual Experiment Platform') { exit 0 } else { exit 1 }" >nul 2>&1
  if not errorlevel 1 goto :open_launcher
)

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
set "SPACEFL_IN=%VENV_DIR%\spacefl-web.stdin"
set "SPACEFL_OUT=%VENV_DIR%\spacefl-web.out.log"
set "SPACEFL_ERR=%VENV_DIR%\spacefl-web.err.log"
type nul >"%SPACEFL_IN%"
del /q "%SPACEFL_OUT%" "%SPACEFL_ERR%" >nul 2>&1
powershell.exe -NoProfile -WindowStyle Hidden -Command "$inputFile = $env:SPACEFL_IN; $outFile = $env:SPACEFL_OUT; $errFile = $env:SPACEFL_ERR; $args = @('-m','fl_space.cli','run','serve','--host','127.0.0.1','--port',$env:PORT); $p = Start-Process -FilePath $env:PYTHON_EXE -ArgumentList $args -WorkingDirectory $env:PROJECT_DIR -WindowStyle Hidden -RedirectStandardInput $inputFile -RedirectStandardOutput $outFile -RedirectStandardError $errFile -PassThru; if ($null -eq $p) { exit 1 }"
if errorlevel 1 (
  echo [ERROR] Could not start the Web server process.
  echo See "%SPACEFL_ERR%" for details.
  pause
  exit /b 1
)

rem Give Uvicorn time to import the app, then verify the actual endpoint.
set /a START_ATTEMPTS=0
:wait_for_server
powershell.exe -NoProfile -Command "$r = $null; $body = $null; try { $r = Invoke-WebRequest -UseBasicParsing -Uri ('http://127.0.0.1:' + $env:PORT + '/api/health') -TimeoutSec 1; $body = $r.Content | ConvertFrom-Json } catch {}; if ($r -and $r.StatusCode -eq 200 -and $body.service -eq 'SpaceFL Visual Experiment Platform') { exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 goto :server_ready
set /a START_ATTEMPTS+=1
if %START_ATTEMPTS% GEQ 30 goto :server_failed
ping.exe -n 2 127.0.0.1 >nul
goto :wait_for_server

:server_ready
goto :open_launcher

:server_failed
echo [ERROR] The Web server did not become ready within 30 seconds.
echo Check "%VENV_DIR%\spacefl-web.err.log" for details.
pause
exit /b 1

:open_launcher
echo [READY] Opening http://127.0.0.1:%PORT%/launcher.html
start "" "http://127.0.0.1:%PORT%/launcher.html"
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

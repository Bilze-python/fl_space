@echo off
rem Stop SpaceFL background web servers started from this project.
rem Use it after upgrading the platform, or when ports/log files are held by an
rem old instance. This file must stay ASCII-only (cmd.exe limitation).
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
call "%~dp0start_spacefl.bat" --stop
echo.
echo Done. Run start_spacefl.bat to start the current version again.
pause
endlocal
exit /b 0

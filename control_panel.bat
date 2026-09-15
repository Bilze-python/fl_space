@echo off
setlocal EnableExtensions DisableDelayedExpansion
call "%~dp0start_spacefl.bat" --panel %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%

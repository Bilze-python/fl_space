' SpaceFL public demo hidden launcher (used by Task Scheduler / Startup).
' wscript.exe is a GUI-subsystem host, so this does not create a visible cmd window.
Option Explicit
Dim shell, projectRoot, port, command
Set shell = CreateObject("WScript.Shell")
projectRoot = "D:\fl_space-master"
port = 8720
command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & projectRoot & "\deploy\keepalive_public_demo.ps1"" -Port " & port
shell.Run command, 0, False

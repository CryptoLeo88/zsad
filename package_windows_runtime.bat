@echo off
setlocal

cd /d %~dp0

powershell -ExecutionPolicy Bypass -File "%~dp0package_windows_runtime.ps1" %*

endlocal

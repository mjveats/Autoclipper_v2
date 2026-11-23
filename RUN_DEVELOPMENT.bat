@echo off
setlocal
cd /d "%~dp0"
title Auto Clipper (Development Version) - By MJ Mark Jonel

set "VENVPY=%~dp0venv\Scripts\python.exe"
if not exist "%VENVPY%" (
    echo [ERROR] Virtual environment not found. Running setup...
    call "%~dp0INSTALL_AND_RUN.bat"
    exit /b
)

echo Starting Auto Clipper in Development Mode...
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0autoclipper.py"
exit /b 0

@echo off
setlocal
cd /d "%~dp0"
title Auto Clipper
set "TOOLS=%~dp0tools"
set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
set "VENVPY=%~dp0venv\Scripts\python.exe"
set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
if not exist "%TOOLS%" mkdir "%TOOLS%"

if exist "%~dp0.installed" goto RUN

echo ============================================================
echo  FIRST-TIME SETUP (one time only, about 15-25 minutes)
echo  Do not close this window.
echo ============================================================

rem ---- 1. Python 3.11 (private install, does not touch your system Python)
if exist "%PY%" goto PYOK
echo [1/6] Installing Python 3.11...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%TOOLS%\python_setup.exe'"
if errorlevel 1 goto FAIL
start /wait "" "%TOOLS%\python_setup.exe" /quiet InstallAllUsers=0 PrependPath=0 Include_test=0 Include_launcher=0
if not exist "%PY%" goto FAIL
:PYOK

rem ---- 2. FFmpeg
if exist "%TOOLS%\ffmpeg\bin\ffmpeg.exe" goto FFOK
echo [2/6] Installing FFmpeg...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' -OutFile '%TOOLS%\ffmpeg.zip'; Expand-Archive '%TOOLS%\ffmpeg.zip' '%TOOLS%\ffmpeg_tmp' -Force; Get-ChildItem '%TOOLS%\ffmpeg_tmp' | Select-Object -First 1 | Move-Item -Destination '%TOOLS%\ffmpeg' -Force; Remove-Item '%TOOLS%\ffmpeg.zip','%TOOLS%\ffmpeg_tmp' -Recurse -Force"
if not exist "%TOOLS%\ffmpeg\bin\ffmpeg.exe" goto FAIL
:FFOK

rem ---- 3. Ollama (local LLM server)
if exist "%OLLAMA%" goto OLOK
echo [3/6] Installing Ollama...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest 'https://ollama.com/download/OllamaSetup.exe' -OutFile '%TOOLS%\OllamaSetup.exe'"
if errorlevel 1 goto FAIL
start /wait "" "%TOOLS%\OllamaSetup.exe" /SILENT /NORESTART
if not exist "%OLLAMA%" goto FAIL
:OLOK

rem ---- 4. Virtual env + Python packages
echo [4/6] Installing Python packages...
if not exist "%VENVPY%" "%PY%" -m venv venv
if errorlevel 1 goto FAIL
"%VENVPY%" -m pip install --upgrade pip
"%VENVPY%" -m pip install -r requirements.txt
if errorlevel 1 goto FAIL

rem ---- 5. NVIDIA CUDA runtime libs (only if an NVIDIA GPU is present)
where nvidia-smi >nul 2>&1
if errorlevel 1 goto NOCUDA
echo [5/6] NVIDIA GPU found - installing CUDA libraries for Whisper...
"%VENVPY%" -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
:NOCUDA

rem ---- 6. Models (Whisper, face detector)
echo [6/6] Downloading AI models (Whisper, face detector)...
"%VENVPY%" autoclipper.py --setup
if errorlevel 1 goto FAIL

echo done> "%~dp0.installed"
echo.
echo Setup complete. Launching...

:RUN
rem Launch Auto Clipper workstation immediately with zero startup delay
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0autoclipper.py"
rem Keep yt-dlp updated asynchronously in background (never blocks UI launch)
start /b "" "%VENVPY%" -m pip install -U yt-dlp --quiet >nul 2>&1
exit /b 0

:FAIL
echo.
echo *** SETUP FAILED. Check your internet connection and run this file again. ***
echo *** It resumes where it stopped. ***
pause
exit /b 1

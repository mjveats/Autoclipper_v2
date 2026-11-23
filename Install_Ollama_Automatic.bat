@echo off
title Auto Clipper - Automatic Ollama Setup
cd /d "%~dp0"

echo ============================================================
echo   AUTO CLIPPER - ONE-CLICK AUTOMATIC OLLAMA INSTALLER
echo   Owner: MJ Mark Jonel
echo ============================================================
echo.
echo Checking if Ollama is already installed...
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    echo [OK] Ollama is already installed!
    echo You can close this window and launch AutoClipper.exe directly.
    echo.
    pause
    exit /b 0
)

echo Ollama is not installed yet.
echo [1/2] Downloading official Ollama installer (one time only)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest 'https://ollama.com/download/OllamaSetup.exe' -OutFile '%TEMP%\OllamaSetup.exe'"

if not exist "%TEMP%\OllamaSetup.exe" (
    echo [ERROR] Could not download Ollama installer. Please check your internet connection.
    pause
    exit /b 1
)

echo [2/2] Installing Ollama silently in the background...
start /wait "" "%TEMP%\OllamaSetup.exe" /SILENT /NORESTART
if exist "%TEMP%\OllamaSetup.exe" del /f /q "%TEMP%\OllamaSetup.exe"

echo.
echo ============================================================
echo   [SUCCESS] OLLAMA IS INSTALLED!
echo   You can now double-click AutoClipper.exe to start clipping!
echo ============================================================
echo.
pause
exit /b 0

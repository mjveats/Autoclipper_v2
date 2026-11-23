@echo off
title Build Auto Clipper Executable - By MJ Mark Jonel
cd /d "%~dp0"

echo ============================================================
echo   AUTO CLIPPER - BUILD STANDALONE EXECUTABLE (.exe)
echo   Owner: MJ Mark Jonel
echo ============================================================
echo.

set "VENVPY=%~dp0venv\Scripts\python.exe"
if exist "%VENVPY%" goto :found_venv
echo [ERROR] Python virtual environment not found in %VENVPY%
pause
exit /b 1

:found_venv
echo [1/3] Ensuring PyInstaller is up to date...
"%VENVPY%" -m pip install pyinstaller --quiet

echo [2/3] Compiling autoclipper.py into standalone AutoClipper.exe...
"%VENVPY%" -m PyInstaller --noconsole --name "AutoClipper" --clean --onedir ^
    --add-data "models;models" ^
    --add-data "music;music" ^
    --add-data "resources;resources" ^
    --add-data "web;web" ^
    --add-data "app;app" ^
    --collect-all "faster_whisper" ^
    --collect-all "ctranslate2" ^
    --collect-all "pycaps" ^
    --collect-all "movielite" ^
    --collect-all "mediapipe" ^
    --collect-all "webview" ^
    --collect-all "soundfile" ^
    --collect-submodules "app" ^
    --hidden-import "PIL" ^
    --hidden-import "PIL.ImageTk" ^
    --hidden-import "cv2" ^
    --hidden-import "soundfile" ^
    --hidden-import "faster_whisper" ^
    --hidden-import "yt_dlp" ^
    --hidden-import "requests" ^
    --hidden-import "app" ^
    --hidden-import "app.services" ^
    --hidden-import "app.services.reframer" ^
    --hidden-import "app.services.caption_renderer" ^
    autoclipper.py

if errorlevel 1 goto :build_fail

echo.
echo ============================================================
echo   [3/3] Running Post-Build Organizer and Packaging Tester Archive...
echo ============================================================
"%VENVPY%" "%~dp0build_post.py"
if errorlevel 1 goto :build_fail

:: Cleanup build intermediate files
if exist "%~dp0build" rmdir /s /q "%~dp0build"
if exist "%~dp0AutoClipper.spec" del /f /q "%~dp0AutoClipper.spec"

echo.
echo ============================================================
echo   ALL DONE! AutoClipper is fully compiled and ready.
echo ============================================================
echo.
exit /b 0

:build_fail
echo.
echo *** Compilation Failed! Check the error log above. ***
pause
exit /b 1

@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%~dp0core\src;%~dp0;%~dp0apps\transcribe"

REM Pin Hugging Face model cache to this folder.
REM First launch downloads the Whisper model (~3GB) here.
set "HF_HOME=%~dp0models\hf_cache"
set "HF_HUB_CACHE=%~dp0models\hf_cache"

if not exist "%~dp0python\python.exe" (
    echo [Error] python.exe not found in %~dp0python
    echo Please re-extract the ZIP completely and try again.
    pause
    exit /b 1
)

echo Starting podcast-transcriber-jp...
echo First launch downloads model ~3GB (5-15 minutes). Please wait.
"%~dp0python\python.exe" -m transcribe_gui

if errorlevel 1 (
    echo.
    echo [Error] Application exited abnormally (exit code %errorlevel%)
    if exist "%~dp0transcribe_gui_crash.log" (
        echo --- crash log ---
        type "%~dp0transcribe_gui_crash.log"
        echo -----------------
    )
    pause
)

endlocal

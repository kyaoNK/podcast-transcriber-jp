@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONPATH=%~dp0core\src;%~dp0;%~dp0apps\transcribe"

REM Hugging Face モデルキャッシュをZIP内に固定。
REM 初回起動時に Whisper モデル (約3GB) がこのフォルダにダウンロードされます。
set "HF_HOME=%~dp0models\hf_cache"
set "HF_HUB_CACHE=%~dp0models\hf_cache"

if not exist "%~dp0python\python.exe" (
    echo [Error] python.exe が見つかりません: %~dp0python
    echo ZIPの解凍が完全に終わっているか確認してください。
    pause
    exit /b 1
)

echo 起動中...初回はモデルのダウンロードに5-10分かかります。
"%~dp0python\python.exe" -m transcribe_gui

if errorlevel 1 (
    echo.
    echo [Error] アプリが異常終了しました (exit %errorlevel%)
    if exist "%~dp0transcribe_gui_crash.log" (
        echo --- クラッシュログ ---
        type "%~dp0transcribe_gui_crash.log"
        echo ----------------------
    )
    pause
)

endlocal

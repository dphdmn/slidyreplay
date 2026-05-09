@echo off
REM Build ReplayVideoGenerator.exe with bundled ffmpeg
REM Run from the replay_gui directory.

setlocal
set GUI_DIR=%~dp0

echo Copying dependency modules from parent project...
copy /Y "%GUI_DIR%..\splits.py" "%GUI_DIR%splits.py" >nul
copy /Y "%GUI_DIR%..\replay_generator.py" "%GUI_DIR%replay_generator.py" >nul
copy /Y "%GUI_DIR%..\replay_video.py" "%GUI_DIR%replay_video.py" >nul

if not exist "%GUI_DIR%ffmpeg\ffmpeg.exe" (
    where ffmpeg >nul 2>&1
    if not errorlevel 1 (
        for /f "tokens=*" %%i in ('where ffmpeg') do (
            copy /Y "%%i" "%GUI_DIR%ffmpeg\ffmpeg.exe" >nul
            goto :copy_done
        )
    ) else (
        echo WARNING: ffmpeg.exe not found in PATH. Place it in ffmpeg\ manually.
    )
)
:copy_done

echo Cleaning previous build...
if exist "%GUI_DIR%dist" rmdir /S /Q "%GUI_DIR%dist"
if exist "%GUI_DIR%build" rmdir /S /Q "%GUI_DIR%build"

echo Building exe...
cd /d "%GUI_DIR%"
python -m PyInstaller --onefile --windowed --name "ReplayVideoGenerator" ^
    --icon "assets\15PUZZLE_ICON.ico" ^
    --add-data "assets\15PUZZLE_ICON.ico;assets" ^
    --add-data "ffmpeg\ffmpeg.exe;." ^
    --hidden-import "ttkbootstrap" ^
    --hidden-import "PIL" ^
    --hidden-import "PIL._tkinter_finder" ^
    --hidden-import "tkinter" ^
    "main.py"

REM Clean up build artifacts and generated spec (only keep dist)
if exist "%GUI_DIR%build" rmdir /S /Q "%GUI_DIR%build"
if exist "%GUI_DIR%ReplayVideoGenerator.spec" del /Q "%GUI_DIR%ReplayVideoGenerator.spec"

echo.
if exist "%GUI_DIR%dist\ReplayVideoGenerator.exe" (
    echo Build complete: %GUI_DIR%dist\ReplayVideoGenerator.exe
) else (
    echo Build FAILED.
)
pause

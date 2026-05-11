@echo off
REM SlidyReplay — CPU-only standalone EXE build
REM Requires: pip install pyinstaller

python -m PyInstaller --onefile --windowed ^
  --name "ReplayVideoGenerator" ^
  --icon assets\15PUZZLE_ICON.ico ^
  --add-data "fonts;fonts" ^
  --add-data "assets;assets" ^
  --add-data "ffmpeg\ffmpeg.exe;." ^
  --exclude-module torch ^
  --exclude-module caffe2 ^
  --exclude-module torchvision ^
  --exclude-module torchaudio ^
  main.py

echo.
echo Build complete: dist\ReplayVideoGenerator.exe

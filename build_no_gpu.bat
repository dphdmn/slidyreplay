@echo off
REM SlidyReplay — CPU-only standalone EXE build
REM Requires: pip install pyinstaller

python -m PyInstaller --onefile ^
  --name "slidyreplay" ^
  --icon assets\15PUZZLE_ICON.ico ^
  --add-data "fonts;fonts" ^
  --add-data "assets;assets" ^
  --add-data "ffmpeg\ffmpeg.exe;." ^
  --hidden-import "PIL._tkinter_finder" ^
  --hidden-import "ttkbootstrap" ^
  --exclude-module torch ^
  --exclude-module caffe2 ^
  --exclude-module torchvision ^
  --exclude-module torchaudio ^
  --exclude-module cuda ^
  --exclude-module cupy ^
  main.py

echo.
echo Build complete: dist\slidyreplay.exe

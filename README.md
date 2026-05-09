# ReplayVideoGenerator

Generates MP4 videos of sliding puzzle replays from [slidysim](https://slidysim.github.io/) replay URLs or manual solution strings.

## Usage

```
python main.py
```

Paste replay URLs or manual solution strings, configure TPS/time/size, and generate videos.

## Build

```
build.bat
```

Builds a standalone `dist\ReplayVideoGenerator.exe` with PyInstaller. Requires `ffmpeg\ffmpeg.exe` (auto-copied from PATH if missing).

## Dependencies

- Python 3.13+
- `ttkbootstrap`, `Pillow`, `tabulate`
- `ffmpeg` (bundled into the exe at build time)

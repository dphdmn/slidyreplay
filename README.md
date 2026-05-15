# SlidyReplay

Generates MP4 videos of sliding puzzle replays from [slidysim](https://slidysim.github.io/) replay URLs, input files, or manual solution strings.

## Quick Start

### GUI

```
python main.py
```

Launches a dark-themed GUI with URL, File, and Manual input tabs, FPS slider, GPU toggle, force fringe toggle, progress panel with ETA, and a generated replays list.

### CLI

```
python main.py --solution R2D2L2U2 --size 3x3 --tps 10 -o replay.mp4
python main.py --url "https://slidysim.github.io/?replay=..." -o replay.mp4
python main.py --file input.txt -o replay.mp4
python main.py --batch solutions.txt
```

## CLI Reference

| Flag | Description |
|------|-------------|
| `--solution` / `-s` | Solution string |
| `--url` / `-u` | Replay URL |
| `--file` | File containing a replay URL or solution string |
| `--tps` | Tiles per second (omit if using `--time`) |
| `--time` | Total time in seconds (omit if using `--tps`) |
| `--size` | Puzzle size (e.g. `3x3`, `10x10`) |
| `--scramble` | Scramble string |
| `--output` / `-o` | Output file (default: `replay.mp4`) |
| `--quality` | Render quality (canvas height, min: 720) |
| `--compression` | Video encoder quality 10–40, lower = fewer artifacts but larger file (default: 18) |
| `--movetimes` | Comma-separated move timings in seconds (overrides `--tps`/`--time`) |
| `--speedup` | Speed multiplier (e.g. 2.0 = 2× faster, 0.5 = half speed) (default: 1.0) |
| `--force-fringe` | Force fringe colors (disable grids detection) |
| `--fps` | Output framerate (default: 60) |
| `--no-gpu` | Disable GPU acceleration (GPU is auto-detected by default) |
| `--force-fringe` | Force fringe colors (disable grids detection) |
| `--batch` | File with solutions/URLs (one per line) |
| `--log` | Enable debug logging to `logs/debug_\<timestamp\>.log` |
| `--no-layout` | Render only the puzzle grid — no timer bar, no stats panel |
| `--no-border` | Suppress tile border outlines |
| `--no-secondary-border` | Suppress secondary color bar borders |
| `--no-numbers` | Suppress tile number text (improves compression) |

Debug logging is **disabled by default**. Pass `--log` to enable:

```
python main.py --log                       # GUI with logging
python main.py --solution R2D2L2U2 --log   # CLI with logging
```

Logs are written to `logs/debug_YYYYMMDD_HHMMSS.log`.

## Output Format

Generated files follow the pattern: `<size>_<total_time>_<moves>_<tps>_movetimes.mp4`

Example: `8x8_23.564_707_30.003_movetimes_5.mp4`

## GPU Acceleration

The GUI shows the GPU name when available, or "Not available — install CUDA" when PyTorch/CUDA is missing. GPU is enabled by default when available and falls back to CPU automatically.

### Benchmarks (NVIDIA GeForce GTX 1660 SUPER)

**Settings:** quality=720, 60 FPS. 

15.05 (v5.0) version benchmark results (Layout / No Layout):
======================================================================
      Puzzle      Moves     Unique    GPU Lay    GPU NoL    CPU Lay    CPU NoL
  -----------------------------------------------------------------------------
         4x4         26         25       0.9s       0.7s       0.8s       0.7s
         5x5         98         95       1.1s       0.9s       1.2s       0.9s
         6x6        213        208       1.5s       1.2s       1.7s       1.4s
         7x7        425        401       2.2s       1.7s       2.9s       2.2s
         8x8        707        652       3.0s       2.3s       4.5s       3.1s
         9x9       1251       1079       4.8s       3.3s       7.6s       4.7s
       10x10       1569       1362       6.9s       4.5s       9.7s       6.1s
       12x12       2883       2470      11.2s       8.1s      16.5s      10.7s
       16x16       7132       5692      20.7s      18.8s      39.4s      23.6s
       20x20      14203      11177      43.6s      32.3s      78.0s      43.0s

Run your own benchmarks with `python benchmark.py`.
"Unique" = number of distinct puzzle states rendered (video may duplicate frames via frame mapping).

### Installing GPU support

1. **Check your NVIDIA driver supports CUDA 12.x:**
   ```
   nvidia-smi
   ```
   Look for "CUDA Version" in the output (top-right).

2. **Install PyTorch with CUDA:**
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Verify:**
   ```
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

### Benchmark script

```
python benchmark.py         # all puzzles: small (CPU+GPU) + big (GPU only)
python benchmark.py --small # small puzzles only (CPU+GPU)
python benchmark.py --big   # big puzzles only (GPU only)
```

All outputs saved to logs/ folder (for performance testing only).

## Build (Windows)
Builds a standalone `dist\ReplayVideoGenerator.exe` (no-GPU version only):
```
build_no_gpu.bat
```
Requires `ffmpeg\ffmpeg.exe`.

For GPU acceleration, use the repository directly (`python main.py`) instead of the standalone build.

## Dependencies

- Python 3.9+
- `ttkbootstrap`, `Pillow`, `tabulate`, `numpy`, `psutil` (optional — for RAM logging with `--log`)
- `torch` (optional — for GPU acceleration)
- `ffmpeg` — must be installed and available in your PATH (or same folder as the script)

Tested on Windows 11 with Python 3.13.5.

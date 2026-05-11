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
| `--quality` | Render quality 1.0–4.0 (default: 1.0) |
| `--compression` | Video encoder quality 10–40, lower = fewer artifacts but larger file (default: 18) |
| `--movetimes` | Comma-separated move timings in seconds (overrides `--tps`/`--time`) |
| `--fps` | Output framerate (default: 60) |
| `--no-gpu` | Disable GPU acceleration (GPU is auto-detected by default) |
| `--batch` | File with solutions/URLs (one per line) |
| `--log` | Enable debug logging to `logs/debug_\<timestamp\>.log` |

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

### Benchmarks (NVIDIA GeForce GTX 1660 SUPER) (v2.1)

**Settings:** quality=1.0, 60 FPS

| Puzzle | Moves | CPU    | GPU    | Speedup |
| ------ | ----- | ------ | ------ | ------- |
| 4×4    | 26    | 4.3s   | 0.8s   | 5.4×    |
| 5×5    | 98    | 4.6s   | 1.4s   | 3.3×    |
| 6×6    | 213   | 6.1s   | 2.7s   | 2.3×    |
| 7×7    | 425   | 9.9s   | 5.4s   | 1.8×    |
| 8×8    | 707   | 18.9s  | 8.5s   | 2.2×    |
| 9×9    | 1251  | 25.9s  | 14.0s  | 1.8×    |
| 10×10  | 1569  | 26.9s  | 16.2s  | 1.7×    |
| 12×12  | 2883  | —      | 36.4s  | —       |
| 16×16  | 7132  | —      | 112.6s | —       |
| 20×20  | 14203 | —      | 217.0s | —       |

The most benefit from GPU rendering is currently achieved by running encoding in parallel with the rendering stage.

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

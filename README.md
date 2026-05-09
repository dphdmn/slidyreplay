# SlidyReplay

Generates MP4 videos of sliding puzzle replays from [slidysim](https://slidysim.github.io/) replay URLs or manual solution strings.

## GUI

```
python main.py
```

Launches a dark-themed GUI with:
- **URL tab** — paste replay URLs (one per line) and hit Generate
- **Manual tab** — enter solution strings, TPS, time, size, scramble, movetimes
- **Quality** slider (1.0–4.0) controls output resolution
- **Force fringe** toggle to disable grids color detection
- **GPU acceleration** toggle with status indicator (shows GPU name when available)
- **Progress** panel with ETA, active/completed counts, elapsed time
- **Generated Replays** list — double-click to open a video, Folder to open the directory

## CLI

```
python main.py --solution R2D2L2U2 --size 3x3 --tps 10 -o replay.mp4
python main.py --url "https://slidysim.github.io/?replay=..." -o replay.mp4
python main.py --batch solutions.txt
```

CLI flags:
| Flag | Description |
|------|-------------|
| `--solution` / `-s` | Solution string |
| `--url` / `-u` | Replay URL |
| `--tps` | Tiles per second |
| `--time` | Total time in seconds |
| `--size` | Puzzle size (e.g. `3x3`, `10x10`) |
| `--scramble` | Scramble string |
| `--output` / `-o` | Output file (default: `replay.mp4`) |
| `--quality` | Render quality 1.0–4.0 (default: 2.0) |
| `--gpu` | Force GPU acceleration |
| `--no-gpu` | Disable GPU acceleration |
| `--batch` | File with solutions/URLs (one per line) |
| `--stats-path` | Write per-batch JSONL stats to this path (for benchmarking/debugging) |

Example batch file (`urls.txt`):
```
https://slidysim.github.io/?replay=...
https://slidysim.github.io/?replay=...
```

```
python main.py --batch urls.txt --gpu
```

## GPU Acceleration

The GUI shows the GPU name when available, or "Not available — install CUDA" when PyTorch/CUDA is missing. GPU is used by default when available.

### How GPU rendering works

Frame rendering is GPU-accelerated via PyTorch/CUDA:

- **Auto-calibrating batch sizing**: The renderer measures per-frame memory cost on the first batch (from `cuda.memory_reserved()`), then automatically computes the largest batch size that fits within available VRAM (soft 50% ceiling, with 256MB safety margin). No manual `memory_usage` parameter needed.

- **Row-chunked tile rendering**: For large puzzles, tiles are rendered in row chunks (2 rows at a time by default) to prevent combinatorial memory explosion.

- **In-place blending**: All tile compositing (background colors, borders, number text) happens in-place on GPU tensors.

- **Static/dynamic stats optimization**: The stats panel is split into static and dynamic parts. Static values (Time total, Moves total, TPS total, Cubic estimate, MD total, M/MD total, grid stages) are rendered **once**. Dynamic values (Predicted moves, MD current, M/MD current, current stage highlight) are rendered per frame.

### Benchmarks (NVIDIA GeForce GTX 1660 SUPER)

**12×12 puzzle, 2884 frames, quality=2.0:**

| Mode | Time | Notes |
|------|------|-------|
| GPU | ~72s | Auto-calibrated batches (~27-32 frames/batch) |

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

If PyTorch is not installed or CUDA is unavailable, the program falls back to CPU rendering automatically.

### Benchmark script

```
python benchmark.py
python benchmark.py --skip-cpu
```

Uses `test_input.txt` and renders with GPU acceleration. All outputs (MP4, JSONL stats, benchmark log) are saved to `logs/{run_id}/`.


## Build

```
build.bat
```

Builds a standalone `dist\ReplayVideoGenerator.exe` with PyInstaller. Requires `ffmpeg\ffmpeg.exe` (auto-copied from PATH if missing).

## Dependencies

- Python 3.13+
- `ttkbootstrap`, `Pillow`, `tabulate`, `numpy`
- `torch` (optional — for GPU acceleration)
- `ffmpeg` (bundled into the exe at build time)

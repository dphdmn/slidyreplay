# ReplayVideoGenerator

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

Example batch file (`urls.txt`):
```
https://slidysim.github.io/?replay=...
https://slidysim.github.io/?replay=...
```

```
python main.py --batch urls.txt --gpu
```

## GPU Acceleration

The GUI shows the GPU name when available, or "Not available — install CUDA" when PyTorch/CUDA is missing. The CLI prints `GPU ON (device name)` or `GPU OFF (CPU fallback)`. GPU is used by default when available.

### How GPU acceleration works

Frame rendering is GPU-accelerated via PyTorch/CUDA when available. The tile grid (colored backgrounds, number text, borders) is rendered as batched tensor operations on the GPU, and the full frame (timer, stats panel, solution text) is composited on the GPU. Each frame is rendered individually to fit within GPU memory.

### Benchmarks (GTX 1660 SUPER)

**25×25 puzzle, 200 frames:**

| Mode | Time | vs CPU |
|------|------|--------|
| CPU  | 56.7s | 1.0× |
| GPU  | 7.7s  | **7.4×** |

**10×10 puzzle, 1570 frames:**

| Mode | Time | vs CPU |
|------|------|--------|
| CPU  | 293.4s | 1.0× |
| GPU  | 144.3s | **2.0×** |

The speedup is larger for bigger puzzles because the GPU's parallel tile processing wins more over per-tile CPU loops. For smaller puzzles the overhead of tensor ops and CPU-GPU transfer is more significant.

### Installing GPU support

1. **Check your NVIDIA driver supports CUDA 12.x:**
   ```
   nvidia-smi
   ```
   Look for "CUDA Version" in the output (top-right). If you don't have an NVIDIA GPU or the driver doesn't support CUDA 12, GPU acceleration won't work.

2. **Install PyTorch with CUDA:**
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
   This installs PyTorch with CUDA 12.4 support. The download is ~3 GB.

3. **Verify:**
   ```
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```
   Should print `True` and your GPU name (e.g. `NVIDIA GeForce GTX 1660 SUPER`).

If PyTorch is not installed or CUDA is unavailable, the program falls back to CPU rendering automatically — no changes to usage are needed.

### Benchmark script

Run your own benchmark:
```
python benchmark.py
```
Uses `test_input.txt` (a 10×10 replay URL) and times GPU vs CPU rendering.

## Build

```
build.bat
```

Builds a standalone `dist\ReplayVideoGenerator.exe` with PyInstaller. Requires `ffmpeg\ffmpeg.exe` (auto-copied from PATH if missing).

## Dependencies

- Python 3.13+
- `ttkbootstrap`, `Pillow`, `tabulate`
- `torch` (optional — for GPU acceleration)
- `ffmpeg` (bundled into the exe at build time)

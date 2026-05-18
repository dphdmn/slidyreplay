# SlidyReplay

Generates MP4 videos of sliding puzzle replays from [slidysim](https://slidysim.github.io/) replay URLs, input files, or manual solution strings.

## Quick Start

### GUI

```
python main.py
```

Launches a fully functional GUI with nearly all custom options, backed by the same CLI interface.

### CLI

```
python main.py --solution R2D2L2U2 --size 3x3 --tps 10          # auto-named in replays/
python main.py -u "https://slidysim.github.io/?replay=..."     # auto-named in replays/
python main.py -f input.txt                                     # auto-named in replays/
python main.py -b solutions.txt -q 2160                         # batch, auto-named in replays/
python main.py --solution R2D2L2U2 --time 30 -c 28 --slow-render -o small.mp4
python main.py --solution R2D2L2U2 --no-layout --no-numbers -o clean.mp4
python main.py --solution R2D2L2U2 --upscale --encoder libx265 -o high_quality.mp4
python main.py --solution R2D2L2U2 -q 720 -s 2.0 -o fast.mp4
python main.py --solution R2D2L2U2 --no-header                 # hide timer bar, keep stats panel
python main.py --solution R2D2L2U2 --no-details                # hide stats panel, keep timer
python main.py --solution R2D2L2U2 --dynamic-md                # show right-side MD timer
python main.py --solution R2D2L2U2 --no-header --no-details    # puzzle grid only (= --no-layout)
python main.py --solution R2D2L2U2 --saturation 0.5          # set both min/max saturation to 0.5
python main.py --solution R2D2L2U2 --saturation-min 0.3 --saturation-max 0.9  # gradient across tiles
python main.py --solution R2D2L2U2 --brightness-min 0.2 --brightness-max 0.8  # brightness gradient
python main.py --settings my_settings.json --solution R2D2L2U2   # load settings from JSON
python main.py --solution R2D2L2U2 --font-family Arial --font-bold --font-size 36  # custom tile font
```

```
python main.py --image --size 4x4                                 # solved puzzle image
python main.py --image --size 4x4 --scramble "4 1 2/7 8 3/6 5 0" # custom scramble image
python main.py --image --solution R2D2L2U2 --size 3x3            # scrambled + solved images
python main.py --image --size 5x5 --no-numbers --main-scheme rows
python main.py --image -f replay.txt --size 4x4 -o output.png    # explicit output path
```

## CLI Reference

| Category | Flag | Short | Description |
|----------|------|-------|-------------|
| **Input** | `--solution` | | Solution string |
| | `--url` | `-u` | Replay URL |
| | `--file` | `-f` | File containing a replay URL or solution string |
| | `--batch` | `-b` | File with solutions/URLs (one per line) |
| | `--image` | | Render PNG image(s) instead of video. Requires `--size`. See [Image Mode](#image-mode) below |
| **Puzzle** | `--size` | | Puzzle size (e.g. `3x3`, `10x10`) |
| | `--scramble` | | Scramble string |
| **Timing** | `--tps` | | Tiles per second (omit if using `--time`) |
| | `--time` | | Total time in seconds (omit if using `--tps`) |
| | `--movetimes` | | Comma-separated move timings in seconds (overrides `--tps`/`--time`) |
| | `--speedup` | `-s` | Speed multiplier (e.g. `2.0` = 2× faster, `0.5` = half speed) (default: 1.0) |
| **Output** | `--output` | `-o` | Output file path (default: auto-generated name in `replays/` folder) |
| | `--quality` | `-q` | Render quality — canvas height (720, 1080, 1440, 2160) |
| | `--fps` | | Output framerate (default: 60) |
| **Encoder** | `--compression` | `-c` | Video encoder quality 10–40, lower = fewer artifacts but larger file (default: 18) |
| | `--slow-render` | | Slower encode, ~33% smaller file (p7 for NVENC, slow for libx264). Auto-enabled on CPU. |
| | `--encoder` | | Force video encoder (choices below). Auto-detected if not set. |
| | `--upscale` | | Re-encode to 2K (2560×1440). Only applies when quality < 1440p. Keeps original too. |
| **Render** | `--no-layout` | | Shortcut for `--no-header --no-details` (puzzle grid only) |
| | `--no-header` | | Hide the timer header bar (time/moves/tps and MD display) |
| | `--no-details` | | Hide the stats panel on the right side of the puzzle |
| | `--dynamic-md` | | Show MD/predicted/MMD timer on the right of the header (off by default — timer centered instead) |
| | `--adjust-height` | | Crop canvas height to puzzle content instead of fixed quality preset. Useful when big puzzle doesn't fill the frame due to small tile size |
| | `--no-border` | | Suppress tile border outlines |
| | `--no-secondary-border` | | Suppress secondary color bar borders |
| | `--no-grid-bars` | | Suppress secondary grid bar indicators inside tiles |
| | `--no-numbers` | | Suppress tile number text (improves compression) |
| | `--main-scheme` | | Color scheme: `fringe`, `rows`, or `columns` (default: `fringe`) |
| | `--force-main` | | Force main scheme everywhere (disable grids detection) |
| | `--animate-moves` | | Animate tile sliding between moves (smooth transitions) |
| | `--cycles-detection` | | EXPERIMENTAL: detect and display cycling tiles in grid stats (may increase analysis time) |
| **Font** | `--font-family` | | Tile number font family (system font name, e.g. `Arial`). Default: `Roboto` |
| | `--font-bold` | | Use bold variant of the tile number font |
| | `--font-size` | | Override font size in px (auto-computed by default). ⚠ Numbers may overflow tiles |
| **Colors** | `--grid1-color` | | Grid 1 (red sections) color as hex, e.g. `FF0000` |
| | `--grid2-color` | | Grid 2 (blue sections) color as hex, e.g. `0000FF` |
| | `--tile-bg-color` | | Tile background color as hex, e.g. `969696` |
| | `--hue-start` | | Hue range start (0–330, default: 0) |
| | `--hue-end` | | Hue range end (0–330, default: 330) |
| | `--saturation` | | Saturation, sets both min and max (0–1). Overrides `--saturation-{min,max}` |
| | `--saturation-min` | | Saturation minimum for tile gradient (0–1, default: 0.78) |
| | `--saturation-max` | | Saturation maximum for tile gradient (0–1, default: 0.78) |
| | `--brightness` | | Brightness, sets both min and max (0–1). Overrides `--brightness-{min,max}` |
| | `--brightness-min` | | Brightness minimum for tile gradient (0–1, default: 0.6) |
| | `--brightness-max` | | Brightness maximum for tile gradient (0–1, default: 0.6) |
| **Hardware** | `--no-gpu` | `-g` | Disable GPU acceleration (GPU is auto-detected by default) |
| **Settings** | `--settings` | | Load settings from a JSON file saved from the GUI. Explicit CLI flags override file values |
| **Debug** | `--log` | `-l` | Enable debug logging to `logs/debug_\<timestamp\>.log` |

## Settings Management

The GUI includes three settings controls at the bottom of the Settings panel:

- **Reset to Defaults** — resets all controls to their default values (render options, colors, sliders, etc.)
- **Save Settings...** — exports all current settings to a JSON file. Output folder is excluded.
- **Load Settings...** — imports settings from a previously saved JSON file and applies them to the GUI.

### CLI `--settings`

Use a settings file saved from the GUI directly on the command line:

```
python main.py --settings my_settings.json --solution R2D2L2U2 --size 4x4
```

Explicit CLI flags override values from the settings file, so you can use a base config and tweak individual parameters:

```
python main.py --settings base.json --solution R2D2L2U2 --fps 120 --hue-start 180
```

### JSON format

Settings files use a simple flat JSON structure. Example:

```json
{
  "quality": "1080p",
  "fps": 60,
  "compression": 18,
  "speed_factor": "1",
  "use_gpu": true,
  "slow_render": false,
  "upscale": false,
  "encoder": "Auto",
  "main_scheme": "fringe",
  "force_main": false,
  "no_layout": false,
  "no_border": false,
  "no_secondary_border": false,
  "no_grid_bars": false,
  "no_numbers": false,
  "no_header": false,
  "no_details": false,
  "dynamic_md": false,
  "cycles_detection": false,
  "adjust_height": false,
  "animate_moves": false,
  "hue_start": 0.0,
  "hue_end": 330.0,
  "saturation_min": 78,
  "saturation_max": 78,
  "brightness_min": 60,
  "brightness_max": 60,
  "grid1_color": "C86767",
  "grid2_color": "8DB3FF",
  "tile_bg_color": "454545",
  "font_family": "",
  "font_bold": false,
  "font_size_override": 0
}
```

Saturation and brightness values are stored at GUI scale (0–100), they are automatically converted to the 0–1 range when used via `--settings`.

## Encoder Options

Available encoders are auto-detected in priority order — the first supported encoder is used:

| Priority | Encoder | GPU Required | Quality Flag |
|----------|---------|-------------|-------------|
| 1 | `hevc_nvenc` | NVIDIA | `-cq` (CRF-like, 21–51) |
| 2 | `hevc_amf` | AMD | `-qp_p` (CQP, 18–48) |
| 3 | `hevc_qsv` | Intel | `-global_quality` (CQP, 18–48) |
| 4 | `libx265` | — | `-crf` (0–51) |
| 5 | `h264_nvenc` | NVIDIA | `-cq` (CRF-like, 22–52) |
| 6 | `h264_amf` | AMD | `-qp_p` (CQP, 18–48) |
| 7 | `h264_qsv` | Intel | `-global_quality` (CQP, 18–48) |
| 8 | `libx264` | — | `-crf` (0–51, default fallback) |

- **`--compression`** (`-c`) maps to the encoder's quality flag with an offset to normalize across encoders (the value 10–40 is translated per-encoder).
- **`--slow-render`** switches to a slower preset: `p7` for NVENC, `quality` for AMF, `veryslow` for QSV, `slow` for libx264/libx265. This typically reduces file size by ~33% at the cost of ~33% longer encode time.
- **`--encoder`** overrides auto-detection to force a specific encoder. Useful when multiple GPUs are present or you want software encoding on a GPU-capable system.

Debug logging is **disabled by default**. Pass `--log` (`-l`) to enable:

```
python main.py -l                               # GUI with logging
python main.py --solution R2D2L2U2 -l          # CLI with logging
```

Logs are written to `logs/debug_YYYYMMDD_HHMMSS.log`.

## Image Mode

The `--image` flag renders single-frame PNG images rather than videos. Images always render the puzzle grid only.

### Behavior by input

| Input | Output |
|-------|--------|
| `--size 4x4` only | One image: solved puzzle → `4x4_puzzle.png` |
| `--size 4x4 --scramble "..."` | One image: that custom scramble → `4x4_scramble.png` |
| `--size 3x3 --solution R2D2L2U2` | Two images: `3x3_scrambled.png` (first state) + `3x3_solved.png` (solved with analysis colors) |
| `--url` / `--file` + `--size` | Two images, same as solution. Replay URL is parsed for the solution |

### GUI "Render Image" button

Located in the **Override** section of the GUI (next to the Scramble and Movetimes fields). Behavior:
- **With a queue item selected**: generates two images (scrambled + solved with analysis) using current color/display settings, saved to the output folder and listed in the Generated Replays panel
- **No queue item selected**: generates a single image from the override fields — custom scramble if entered, otherwise a solved puzzle of the given size

## Output Format

By default, files are saved to the `replays/` folder with auto-generated names following the pattern: `<size>_<total_time>_<moves>_<tps>_movetimes.mp4`

Example: `8x8_23.564_707_30.003_movetimes_5.mp4`

Use `-o <path>` to save to a custom location with a specific filename.

## GPU Acceleration

The GUI shows the GPU name when available, or "Not available — install CUDA" when PyTorch/CUDA is missing. GPU is enabled by default when available and falls back to CPU automatically.

### Benchmarks (NVIDIA GeForce GTX 1660 SUPER)

**Settings:** quality=720, 60 FPS. 

| Puzzle | Moves | Unique | GPU Layout | GPU No Layout | CPU Layout | CPU No Layout |
|--------|------:|-------:|-----------:|--------------:|-----------:|--------------:|
| 4×4 | 26 | 25 | 0.9s | 0.7s | 0.8s | 0.7s |
| 5×5 | 98 | 95 | 1.1s | 0.9s | 1.2s | 0.9s |
| 6×6 | 213 | 208 | 1.5s | 1.2s | 1.7s | 1.4s |
| 7×7 | 425 | 401 | 2.2s | 1.7s | 2.9s | 2.2s |
| 8×8 | 707 | 652 | 3.0s | 2.3s | 4.5s | 3.1s |
| 9×9 | 1251 | 1079 | 4.8s | 3.3s | 7.6s | 4.7s |
| 10×10 | 1569 | 1362 | 6.9s | 4.5s | 9.7s | 6.1s |
| 12×12 | 2883 | 2470 | 11.2s | 8.1s | 16.5s | 10.7s |
| 16×16 | 7132 | 5692 | 20.7s | 18.8s | 39.4s | 23.6s |
| 20×20 | 14203 | 11177 | 43.6s | 32.3s | 78.0s | 43.0s |

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
python benchmark.py                 # full benchmark run
python benchmark.py --gpu-only      # GPU only (both layout and no-layout)
python benchmark.py --cpu-only      # CPU only
python benchmark.py --layout        # layout only (both GPU and CPU)
python benchmark.py --no-layout     # no-layout only
python benchmark.py --quality-test  # test 10×10 at all quality presets
```

All outputs saved to `logs/<timestamp>/` folder.

## Build (Windows)
Builds a standalone `dist\slidyreplay.exe` (CPU-only):
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

# Renderer Optimization

## Goal
Drive total render time down for both CPU and GPU paths. CPU is already competitive. Now optimize further.

## Project Overview

This project renders sliding puzzle replays as MP4 videos. It takes a solution string (e.g. `R2D2L2U2`) or a Slidysim URL, simulates the moves on a puzzle board, and produces an MP4 with tiles sliding around and a stats panel showing time/TPS/other metrics.

Two render paths exist:

- **CPU path** (`--no-gpu`): renders frames with Pillow (PIL), serial loop, writes raw RGBA frames to an ffmpeg pipe.
- **GPU path** (default): renders frames with PyTorch CUDA tensors in batches, writes raw RGBA frames to an ffmpeg pipe.

Both produce identical output video. The CPU path was originally much slower; optimizations (overlapped pipe, pre-blended drawing, no static_base copy) closed the gap. Now both are comparable on 10×10, and CPU actually wins on larger puzzles.

## Code Architecture & Render Pipeline

### Files you will modify

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point. Parses args, calls `ReplayVideoGenerator.generate_simple_replay()`. |
| `replay_video.py` | Core rendering logic. Contains `generate_frames()` (shared data prep), CPU serial render loop, GPU render orchestration, ffmpeg pipe management. All optimization changes go here or in `gpu_renderer.py`. |
| `gpu_renderer.py` | GPU batched tile renderer. `render_frames()` method does: allocate GPU tensors → upload frame params in batches → launch CUDA kernels → composite overlays → download RGBA bytes → call `frame_handler` callback. |
| `track_progress.py` | `ProgressTracker` class with phase weights. Controls the terminal progress bar and external (GUI) progress callbacks. |
| `replay_generator.py` | Puzzle simulation: generates grid states from solution moves. |

### End-to-end render pipeline

```
main.py
  └─ ReplayVideoGenerator.generate_simple_replay()
       ├─ Parse input (solution string, URL, or file)
       ├─ Determine puzzle size (width × height)
       ├─ Expand solution (R2 → RR, etc.)
       │
       ├─ Stage: Grid analysis ────────────────────────── phase "Analysis" (8%)
       │   ├─ analyse_grids_initial() — detect grid patterns in solution
       │   ├─ generate_grids_stats() — build per-move grid state map
       │   └─ get_all_fringe_schemes() — compute fringe color schemes
       │
       ├─ Stage: Timing ───────────────────────────────── phase "Precompute" (7%)
       │   ├─ calculate_move_timings() — compute per-move delays & fake frame times
       │   └─ Build frame_state map: which puzzle state → how many video frames
       │
       ├─ generate_frames() ───────────────────────────── phase "Render" (85%, overlapped with encode)
       │   ├─ STAGE 1 (data prep):
       │   │   ├─ Filter grid stages for render range
       │   │   ├─ Build frame_params: precompute tile colors, stats text, layout
       │   │   ├─ Build states_needed: list of unique puzzle states to render
       │   │   └─ Build frame_state: mapping from state index → repeat count
       │   │
       │   ├─ STAGE 2 (CPU path only — serial loop):
       │   │   ├─ Open ffmpeg pipe (hevc_nvenc auto-detected)
       │   │   ├─ For each unique state (sequentially):
       │   │   │   ├─ render_frame(state) → PIL Image (full RGBA canvas)
       │   │   │   ├─ img.tobytes() → raw RGBA bytes
       │   │   │   └─ For each duplicate frame count:
       │   │   │       └─ ffmpeg_proc.stdin.write(bytes)  ← pipe to hardware encoder
       │   │   └─ Close ffmpeg pipe
       │   │
       │   ├─ STAGE 3 (GPU path only — batched):
       │   │   ├─ Open ffmpeg pipe
       │   │   ├─ gpu.render_frames(unique_params, frame_handler=handler)
       │   │   │   └─ For each batch (~53 frames):
       │   │   │       ├─ Upload params to GPU tensor
       │   │   │       ├─ Launch CUDA kernels: fill grid, draw tiles, composite overlays
       │   │   │       ├─ Download rendered RGBA bytes to host
       │   │   │       └─ For each frame, call frame_handler:
       │   │   │           └─ For each duplicate count:
       │   │   │               └─ enc_proc.stdin.write(bytes)
       │   │   └─ Close ffmpeg pipe
       │
       └─ prog.finish() — print "Done! Video saved to ..."
```

### Key CPU functions in `replay_video.py`

| Function | Location | What it does |
|----------|----------|-------------|
| `generate_frames()` | line 1119 | Shared data prep + dispatches to CPU or GPU render path |
| `render_frame()` | ~line 950 | Single-frame PIL renderer: draw puzzle grid, tiles, stats panel, timer |
| `_create_ffmpeg_pipe()` | ~line 1071 | Spawns ffmpeg with auto-detected encoder (hevc_nvenc > h264_nvenc > libx264) |
| `_close_pipe()` | ~line 1105 | Closes ffmpeg stdin, waits for process |
| CPU serial loop | lines 1643-1665 | Iterates `states_needed`, calls `render_frame()`, writes to pipe |

### Key GPU functions in `gpu_renderer.py`

| Function | Location | What it does |
|----------|----------|-------------|
| `render_frames()` | ~line 600 | Main GPU render loop: allocates tensors, batches frames, launches kernels |
| Tile rendering kernels | ~line 400 | CUDA-style PyTorch ops to fill tile grid, apply colors, draw borders |
| Overlay compositing | ~line 500 | Blends stats panel, timer, stage borders onto rendered canvas |
| `frame_handler` callback | `replay_video.py:1574-1578` | Receives rendered frame bytes, writes duplicates to ffmpeg pipe |

## How to Run Tests

### Quick smoke test (any puzzle size)
```bash
# CPU:
python main.py --file test_replays/10x10 --no-gpu --log

# GPU:
python main.py --file test_replays/10x10 --log
```

### Test files available
```
test_replays/10x10      — 1569 moves, 10×10 puzzle (fast, for iteration)
test_replays_gpu/16x16  — 7132 moves, 16×16 puzzle (medium, for validation)
test_replays_gpu/20x20  — 14203 moves, 20×20 puzzle (slow, for final validation)
```

### What to measure
1. **Total time** from the final output line: `"Done! Video saved to: replay.mp4 (…, took XX.Xs)"`
2. **Per-stage timing** from the log file in `logs/debug_*.log`:
   - `STAGE 1 DONE: X.Xs` — data prep time
   - `GPU RENDER START` / `CPU RENDER+ENCODE DONE` — render time
   - `STAGE 3 DONE: X.Xs` — GPU render + pipe (GPU only)
3. **Unique frames per second** = `unique_frames / total_time`
4. **Progress bar smoothness** — does it jump or stutter?

### Log file access and analysis

Every run with `--log` creates a timestamped log file:

```bash
# List all logs (most recent last):
Get-ChildItem -LiteralPath logs | Select-Object Name

# Read the most recent log:
Get-Content -LiteralPath (Get-ChildItem -LiteralPath logs -Name | Select-Object -Last 1) -Wait
```

On Windows, you can also open `logs/` in File Explorer and sort by date modified — the latest `.log` file is your last run.

Each log contains:
- **Encoder detection** — which codec ffmpeg chose (`hevc_nvenc`, `h264_nvenc`, or `libx264`)
- **Stage timing** — how long each pipeline stage took
- **GPU batch sizes and counts** — `total_batches=N` (GPU path only)
- **VRAM usage** — `RAM [before/after GPU render]` and batch size per progress tick (GPU path only)
- **State-to-count mapping** — e.g. `state_to_count: 11177 unique states, counts=[30, 3, 15, ...]`

Key log lines to search for:

```bash
# From a specific log:
Select-String -Path "logs/debug_20260514_093651.log" -Pattern "STAGE|RENDER.*DONE|render_frames.*DONE"
```

Lines to look for:
```
  render decision: use_gpu=True, total_video_frames=23562, unique_states=11177 (47%)
  CPU RENDER+ENCODE DONE: 23562 frames written, canvas=1380x1116
  render_frames: DONE. total_batches=212, frames_rendered=11177
  STAGE 1 DONE: 1.1s
  STAGE 3 DONE: 180.0s       (GPU path only)
```

## Git Branch Workflow for Testing

Each optimization variant lives on its own git branch so you can switch between them and compare results.

### Starting from a clean baseline
```bash
# Make sure you're on the main branch with no uncommitted changes
git checkout main
git status  # should show "nothing to commit, working tree clean"
```

### Creating a branch for a variant
```bash
# Create and switch to a new branch for your variant
git checkout -b variant-a-concat-demuxer
```

### Making changes
Edit the relevant files (`replay_video.py`, `gpu_renderer.py`, etc.) then run the tests:
```bash
python main.py --file test_replays/10x10 --no-gpu --log
python main.py --file test_replays/10x10 --log
```

### Committing progress (optional)
```bash
git add replay_video.py
git commit -m "variant A: write unique frames once via concat demuxer"
```

### Switching between variants to compare
```bash
# Save your current variant's work as a commit
git commit -am "variant A: working implementation"

# Go back to main to start another variant
git checkout main
git checkout -b variant-d-larger-batches

# Later, to compare results:
git checkout variant-a-concat-demuxer
python main.py --file test_replays/10x10 --no-gpu --log

git checkout variant-d-larger-batches
python main.py --file test_replays/10x10 --no-gpu --log
```

### Rolling back a failed experiment
```bash
# If you haven't committed:
git checkout -- replay_video.py
git checkout -- gpu_renderer.py

# If you committed but want to discard:
git checkout main
git branch -D variant-a-concat-demuxer   # deletes the branch
```

## Test Commands

```bash
# CPU:
python main.py --file test_replays/10x10 --no-gpu --log

# GPU:
python main.py --file test_replays/10x10 --log
```

> Avoid 16×16 and 20×20 during iteration — use only at the very final validation stage.

## Baselines

Test system: GTX 1660 Super 6GB, 16GB RAM, hevc_nvenc.

| Path | 10×10 (1362 u) | 16×16 (5692 u) | 20×20 (11177 u) |
|------|----------------|----------------|-----------------|
| CPU  | 13.1s (104 f/s) | **77.4s** (74 f/s) | **153.2s** (73 f/s) |
| GPU  | **12.4s** (110 f/s) | 85.9s (66 f/s) | 182.1s (61 f/s) |

**CPU beats GPU on 16×16 and 20×20.** GPU performance collapses as puzzle size grows — VRAM pressure (6GB) forces tiny batches, high overhead per batch. CPU scales linearly across all sizes (~73-104 f/s flat).

## Render Flow & Log Analysis

### CPU render flow
```
Stage 1 (0.5-1.1s): build tile color cache, compute frame_params
Stage 2: for each unique state:
  render_frame(state) -> PIL Image                  # ~13ms per frame
  img.tobytes() -> ffmpeg_proc.stdin.write(data)    # ~0.1ms per write
  for each duplicate: write same data again          # multiplied by repeat count
```

Key log entries from **20×20 CPU** (153.2s total):
```
STAGE 1 DONE: 1.1s
CPU RENDER+ENCODE DONE: 23562 frames written, canvas=1380x1116
```
Render+encode dominates at **151.7s**. O(1) memory — no RAM growth.

### GPU render flow
```
Stage 1 (0.9-1.9s): same prep as CPU
Stage 3: gpu.render_frames(unique_params, frame_handler=handler):
  for each batch:
    upload batch of ~52-65 frames to GPU
    render tiles on GPU
    composite overlays (5 alpha blends + static_base.copy per frame)
    download frame bytes
    for each duplicate: call frame_handler -> enc_proc.stdin.write(data)
```

Key log entries from **20×20 GPU** (182.1s total):
```
STAGE 1 DONE: 1.9s
render_frames: 11177 frames, canvas=1380x1116, chunk_rows=2, reserved_static=40MB, target_used_mem=3071MB
render_frames: DONE. total_batches=212, frames_rendered=11177
STAGE 3 DONE: 180.0s
RAM [after GPU render]: 794MB (+753MB vs baseline)
```

### Why GPU loses at scale

| Metric | 10×10 | 16×16 | 20×20 |
|--------|-------|-------|-------|
| GPU batches | ~25 | 99 | 212 |
| Avg batch size | ~54 | 57.5 | 52.7 |
| Stage 1 (GPU) | 0.6s | 0.9s | 1.9s |
| Stage 3 (GPU render) | ~11s | 84.8s | 180.0s |
| GPU RAM after render | ~600MB | 849MB | 794MB |

Batch size stays flat (~52-57) regardless of puzzle size because VRAM is maxed. Total batches scale with unique frames: **212 batches for 20×20**. Each batch has fixed overhead (upload, sync, kernel launches, download). The GPU is spending more time in overhead than actual rendering.

### CPU scaling is flat

| Metric | 10×10 | 16×16 | 20×20 |
|--------|-------|-------|-------|
| Stage 1 (CPU) | 0.4s | 0.5s | 1.1s |
| Render+encode | 12.7s | 76.9s | 151.7s |
| f/s (unique) | 104 | 74 | 73 |

CPU f/s stays nearly constant — the slight drop from 104 to 73 is due to larger canvas pixels (more PIL work per frame), not memory pressure. There is no batch overhead.

The bottleneck for **both** paths is the pipe write/encode. CPU writes 23562 × 4.6 MB = 108 GB through ffmpeg. GPU does the same. This dominates total time.

## Optimization Ideas

### Candidate table

| Variant | What | Path | Impact estimate | Effort |
|---------|------|------|-----------------|--------|
| A | Unique frames + concat demuxer | Both | **REGRESSION** (~2× slower) | Medium |
| B | Backport CPU overlays to GPU | GPU | **N/A** (already optimal) | Medium |
| C | Batch GPU frames in RAM, write once | GPU | **~10-20%** (fewer pipe writes, bigger batches) | High |
| D | Reduce GPU batch overhead | GPU | **~10-20%** (target the 212 batches) | High |
| E | Move pipe write to background thread | Both | **CPU −9.9%, GPU −4.0%** (confirmed) | Low |

### A: Unique frames + concat demuxer

**Path:** Both

**Idea:** Write each unique frame once with per-frame duration via ffmpeg concat demuxer instead of duplicating frame writes. Currently for 20×20: 23562 writes → 11177 unique (108 GB → 51 GB pipe data).

**Implementation:**
1. Render unique frames to temp PPM files.
2. Build concat file with `duration count/fps` per unique frame.
3. Run ffmpeg with concat demuxer.

**Result: FAILED — regression.** Tested 10×10 CPU: 24.4s vs 13.1s baseline (1.9× slower).

**Why it fails:**
- The concat demuxer serializes render and encode (render first, then encode), losing the pipeline overlap that the pipe approach provides naturally.
- Per-frame `duration` directives have floating-point precision issues, causing incorrect frame counts (3110 vs 3167 for 10×10).
- Writing 1362 PPM files to disk adds ~11s I/O overhead with no benefit.
- The pipe bottleneck was already encoder throughput, not pipe write bandwidth — halving pipe data doesn't speed up the encoder.
- Disk space requirement exceeds available space for 20×20 (51 GB needed, 26.5 GB free on C:).

**Lesson:** The pipe-based approach is fundamentally better because it overlaps render, pipe write, and encode. Any strategy that serializes these steps will be slower, regardless of data reduction. Focus optimization on encoding throughput, not pipe bandwidth.

### B: Backport CPU overlays to GPU

**Path:** GPU

**Idea:** GPU path still uses 5 full-canvas RGBA alpha blends and `static_base.copy()` per frame. Apply the same pre-blended `draw.rectangle()` pattern that saved ~2.5s on CPU.

**Result: SKIPPED — already optimal.** Analysis of `gpu_renderer.py` shows the GPU path already uses the efficient pattern:
- Static base built once per solution into a CUDA tensor.
- Static base composited once per batch (not per frame) via tensor alpha blending.
- Per-frame overlays are small text draws from font atlases (~5 micro-blends per frame).
- No `static_base.copy()` or full-canvas alpha composites exist per frame.
- The doc's description reflects the old CPU path, not current GPU code.
- Only marginal improvement found: simplify expand/contiguous to broadcasting (~negligible gain).

**Test:** N/A — nothing to implement.

### C: Batch GPU frames in host RAM, reduce pipe writes

**Path:** GPU

**Idea:** Instead of writing each frame to the pipe immediately after GPU download, buffer a batch of rendered frames in host RAM and write them all at once. This:
- Reduces the number of `stdin.write()` calls (each has Python<->C overhead).
- Avoids the current pattern where `frame_handler` is called per-duplicate-frame, causing a write per duplicate.

**Why it might help:** Currently each duplicate frame triggers a separate pipe write. For 20×20 that's 23562 writes. If we batch unique frames + their duplicates, we write once per unique state with the count.

**Actual pipe write throughput test needed:** How fast is `stdin.write()` for a 4.6 MB bytes object? If it's ~0.1ms per write, 23562 writes = 2.4s — not a bottleneck. If it's higher with Python overhead, this matters.

**Test:** 10×10 vs GPU baseline. Measure time spent in `frame_handler` vs total.

### D: Reduce GPU batch overhead

**Path:** GPU

**Idea:** The main GPU bottleneck is the large number of small batches (212 for 20×20, avg 53 frames/batch). Each batch incurs fixed overhead:
- Upload frame data to GPU
- Synchronization
- Kernel launches
- Download rendered frames

**Approaches:**
- **Increase target_used_mem** — currently 3071MB (~half of 6GB). It may be possible to use more, increasing batch size and reducing batch count.
- **Async pipeline** — overlap upload of next batch with download/render of current batch (CUDA streams).
- **Render more frames per batch** — the batch size limit comes from per-frame VRAM usage (~30MB/frame for 20×20). If per-frame memory can be reduced (e.g. reuse tile color cache across frames in same batch), batch sizes increase.

**Test:** Profile VRAM usage break per frame. Try increasing target_used_mem to 4000MB. Measure batch count change.

### E: Pipe write in background thread

**Path:** Both

**Idea:** The render loop does `tobytes() + stdin.write()` synchronously. Move the write to a background thread with a queue so render can start the next frame immediately.

**Implementation:** `_PipeWriter` class in `replay_video.py` — a daemon thread reads from a bounded queue (`maxsize=5`) and writes to the ffmpeg pipe. The render thread calls `writer.write(data, count)` which is non-blocking while queue has space, providing natural backpressure when the encoder can't keep up.

**Result: CPU 11.8s (−9.9%), GPU 11.9s (−4.0%).** 10×10 benchmarks:
- CPU: 13.1s → 11.8s (104→112 f/s, 9.9% faster)
- GPU: 12.4s → 11.9s (110→112 f/s, 4.0% faster)
- Both paths converge at ~112 f/s, suggesting the encoder is the shared bottleneck.

**Why it helps:** The main gain comes from overlapping duplicate writes with the next frame's render. For frames with high duplicate counts (e.g., static replay start/end), the direct approach blocks the renderer for `count × write_time` while writing duplicates. The background thread allows the renderer to immediately start the next unique frame.

**Code:** `replay_video.py` — `_PipeWriter` class (~1045) + applied to GPU handler (~1620) and CPU render loop (~1698).

## Context

- **10×10** (1569 moves): 1362 unique / 3167 total, 960×696 canvas, 2.0 MB/frame.
- **16×16** (7132 moves): 5692 unique / 12923 total, 1104×912 canvas, 3.9 MB/frame.
- **20×20** (14203 moves): 11177 unique / 23562 total, 1380×1116 canvas, 4.6 MB/frame.
- Progress phase weights: both paths now use `[8, 7, 85]` (overlapped render+encode). Any phase structure change must update these.

## Relevant Files
- `replay_video.py`: CPU serial render loop (~1690–1710), GPU render path (~1607–1651), `_PipeWriter` class (~1042–1080)
- `gpu_renderer.py`: GPU batched rendering, overlay compositing, frame_handler callback
- `track_progress.py`: phase weights `[8, 7, 85]`
- `test_replays/10x10`, `test_replays_gpu/16x16`, `test_replays_gpu/20x20`: test files

## Git Branches for Each Variant

```bash
# Start each variant from a clean main
git checkout main
git checkout -b variant-a-concat-demuxer
git checkout -b variant-b-gpu-overlays
git checkout -b variant-c-ram-batch-write
git checkout -b variant-d-larger-batches
git checkout -b variant-e-background-write
```

Run baseline before starting any variant:
```bash
git checkout main
python main.py --file test_replays/10x10 --no-gpu --log    # CPU baseline
python main.py --file test_replays/10x10 --log              # GPU baseline
```

## Test Results

| Variant | Branch | Path | 10×10 total | 10×10 f/s | vs baseline | Notes |
|---------|--------|------|-------------|-----------|-------------|-------|
| Baseline CPU | main | CPU | 13.1s | 104 | — | |
| Baseline GPU | main | GPU | 12.4s | 110 | — | |
| A | variant-a-concat-demuxer | CPU | 24.4s | 56 | −86% | Regression — concat serializes render+encode; precision issues; 1.9× slower |
| B | variant-b-gpu-overlays | GPU | | | | |
| C | variant-c-ram-batch-write | GPU | | | | |
| D | variant-d-larger-batches | GPU | | | | |
| E | variant-e-background-write | CPU | 11.8s | 112 | −9.9% | Background pipe writer; overlaps duplicate writes with next render |
| E | variant-e-background-write | GPU | 11.9s | 112 | −4.0% | Background pipe writer; smaller gain on GPU (less write overhead per frame) |

Filled results should include the `f/s` value and the % change vs the relevant baseline (CPU or GPU).
DO NOT RUN TESTS ON 16x16 or 20x20 UNLESS USER ALLOWS IT TO, ASK FIRST.
DO NOT RUN TESTS ON 16x16 or 20x20 UNLESS USER ALLOWS IT TO, ASK FIRST.
DO NOT RUN TESTS ON 16x16 or 20x20 UNLESS USER ALLOWS IT TO, ASK FIRST.
DO NOT RUN TESTS ON 16x16 or 20x20 UNLESS USER ALLOWS IT TO, ASK FIRST.
# CPU Rendering Speed Optimization

## Goal
Make CPU rendering (with `--no-gpu`) approach GPU rendering speed for the 10×10 puzzle test case.

## Final Results (10×10, 1569 moves, quality=1.0)

| Mode | Total | Render | Encode | vs Baseline |
|------|-------|--------|--------|-------------|
| **GPU** (hevc_nvenc) | **11.5s** | ~5s (CUDA) | ~6s (overlapped) | — |
| **CPU (current)** | **12.1s** | ~7.6s | ~7.3s (overlapped) | **16s faster (57%)** |
| CPU (original) | 28.1s | ~18s | ~15.5s (sw) | baseline |
| CPU (NVENC only) | ~25s | ~18s | ~7s (hwe) | +3s |
| CPU (+ panel optimizations) | ~23s | ~15.6s | ~7s | +5s |
| CPU (+ memmap IPC) | ~15s | ~7.7s | ~7.3s | +13s |

## Changes Implemented

### Change 1: Unified ffmpeg pipe with hardware encoder (NVENC)
- **Files**: `replay_video.py:_create_ffmpeg_pipe(), _get_best_encoder()`
- Auto-detects `hevc_nvenc` > `h264_nvenc` > `libx264 veryfast`
- Both GPU and CPU paths use the same pipe function
- **Saved**: ~3s (encode dropped from 15.5s → 7s)

### Change 2: Eliminated panel alpha compositing
- **File**: `replay_video.py:render_frame()` lines 438-456
- Replaced 2 full-canvas RGBA `Image.new()` + 2 `Image.alpha_composite()` + 1 `convert('RGBA')` + 1 `convert('RGB')` = 5 allocations per frame → pre-blended `draw.rectangle()` calls (in-place, no allocations)
- **Saved**: ~1.5s (fewer allocations and pixel-blend operations)

### Change 3: Eliminated `static_base.copy()` in stats panel
- **File**: `replay_video.py:_apply_stats_dynamic()`
- Modified to paste directly onto canvas instead of copying static_base to intermediate RGBA and returning
- **Saved**: ~1s (avoided panel-sized copy per frame)

### Change 4: Memory-mapped frame data (eliminate IPC pickle overhead)
- **File**: `replay_video.py:_render_chunk_mmap()`, parallel path in `generate_frames()`
- Added `_render_chunk_mmap()` function that writes PIL- rendered frames to a numpy memmap file
- Workers write directly to file-backed shared memory instead of returning PIL Images via pickle
- Main process reads from the same memmap during encode
- Added `import tempfile` for temp file creation
- **Saved**: ~8s (render dropped from 15.6s → 7.7s — IPC pickle was the dominant bottleneck)

### Change 5: Overlapped render + encode via streaming encode
- **File**: `replay_video.py:generate_frames()` parallel loop
- Opens read-only memmap in main process *while* workers still render
- After each chunk completes, writes any newly-ready frames to ffmpeg pipe immediately (in video order via `_write_cursor`)
- Encode interleaves with remaining render work, hiding most of the 7.3s encode time
- **Saved**: ~3s (total dropped from 15s → 12s)

### Minor optimization: memoryview writes
- **File**: `replay_video.py:generate_frames()` encode loop
- Replaced `mm[state_idx].tobytes()` → `mm[state_idx].data` to avoid copying pixel data before pipe write
- **Saved**: negligible (~0.06s)

## What We Tried and Rejected

- **Background writer thread**: Writer thread competed for CPU → render slowed 2×. REVERTED.
- **Pre-composed tile sprites**: Combined base+number+bar into single RGBA → more PIL ops, not fewer. REVERTED.
- **In-place prev_canvas mutation (no copy)**: Would break stored state references in the images list. NOT FEASIBLE without architecture change.
- `**_render_chunk returns bytes`**: Moving `tobytes()` to workers didn't help — the real bottleneck was pickle IPC for return values, not the conversion itself.

## Key Technical Insights

1. **Pickle IPC was the bottleneck**: Workers returning 2MB PIL Images via ProcessPoolExecutor required ~2.7GB of pickle serialization. Memmap eliminated this entirely, cutting render time in half.
2. **Alpha compositing is expensive**: 5 full-canvas allocations + 2 pixel-blend operations per frame for the panel. Pre-blending the panel color with BG_COLOR eliminated all of them.
3. **Overlap is free with memmap**: Memmap allows simultaneous read/write from different processes. Reading frames for encode while workers render more frames provides free parallelization.
4. **NVENC encode is pipe-bound**: 3167 × 2MB = 6.3GB through a pipe takes ~7s regardless of encoder speed.

## Bottleneck Analysis (Final)

Current breakdown of the 12.1s CPU total:
- **Frame param prep + pickle**: ~1s
- **Rendering** (8 workers, ~170 frames each): ~7.6s
  - `prev_canvas.copy()` per frame: ~0.3ms × 1362 = 0.4s
  - Tile pasting (3 per changed tile × avg 2 changed): 6 pastes × 0.2ms × 1362 = 1.6s
  - `np.array(img)` for memmap write: ~2ms × 1362 = 2.7s
  - Panel overlay + stats paste: ~1ms × 1362 = 1.4s
  - Timer bar rendering: ~0.5ms × 1362 = 0.7s
  - Worker overhead (pickle params, process management): ~0.8s
- **Encode** (overlapped, mostly hidden): ~7.3s becomes ~0.5s visible tail

## Remaining Improvement Ideas

1. **Dedup repeated states in encode**: Currently we write each frame individually, including duplicates. Could pre-concatenate runs of the same state into single large writes. Small gain.

2. **Render directly into memmap (no PIL)**: Eliminate PIL Image creation + `np.array(img)` copy per frame by writing tile data directly into numpy arrays. Would save ~2.7s but requires major refactor.

3. **Use `np.ndarray` as canvas instead of PIL Image**: Replace `ImageDraw`, `canvas.paste()`, etc. with numpy array operations. Avoids PIL entirely for the CPU path. High effort, potentially large gain.

4. **Reduce canvas resolution for CPU path**: Quality parameter could be lowered when `--no-gpu` is set. Smaller canvas = less data to render, copy, and encode.

5. **Pre-compute tile composite sprites**: Pre-compose the most common (base+number+bar) combinations into single opaque sprites to reduce 6 pastes → 2 pastes per delta frame. Would save ~1s.

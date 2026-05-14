# CPU Rendering Optimization — Current State (May 14, 2026)

## Baseline (before changes)

```
10x10 puzzle, 1569 moves, quality=1.0, fps=60
GPU:  11.3s total  (hevc_nvenc, Batch: 61)
CPU:  28.1s total  (libx264 slow)
```

## Changes Made So Far

### Change 1: Auto-detect best ffmpeg encoder (NVENC for CPU path too)

**File:** `replay_video.py`

- Modified `_get_best_encoder()` to detect `hevc_nvenc` > `h264_nvenc` > `libx264`
- Unified `_create_ffmpeg_pipe()` to use best encoder automatically
- `_create_ffmpeg_pipe_gpu()` now just calls `_create_ffmpeg_pipe()`
- libx264 fallback changed from `-preset slow` to `-preset veryfast`

**Impact:** The CPU path (`--no-gpu`) now uses `hevc_nvenc` for encoding instead of `libx264 slow`. Encode phase dropped from ~15.5s to ~7.0s.

### Change 2: Open ffmpeg pipe early (render + encode overlap for serial path)

**File:** `replay_video.py`

- Moved ffmpeg pipe creation to before the render loop (for both parallel and serial paths)
- Serial path: writes each rendered frame to pipe immediately after rendering (overlap)
- Parallel path: kept sequential render-then-encode (workers out of order makes streaming complex)

### Change 3 (REVERTED): Pre-composed tile cache

Was causing a regression. Composing tiles into intermediate PIL images added more operations than it saved (4 pastes vs 3 per tile). Reverted.

## Current Timing After Changes

```
CPU (27.0s total):
  ┌─ Analysis + Precompute: ~4s  (same as baseline)
  ├─ Render (parallel, 8 workers): 07:46:58 - 07:45:35 = ~18.5s  ← SUSPECT
  ├─ Encode (NVENC hevc_nvenc): 07:47:05 - 07:46:58 = ~7.0s
  └─ Total generate_frames: ~25.8s
```

**Total wall-clock: 27.0s** (vs baseline 28.1s — only 4% improvement)

## Key Finding: Render Phase Regression

The render phase is taking **~18.5s**, which appears to be **~10s slower than the baseline**.

Measured from log timestamps:
- `07:46:35.234` → `generate_frames` starts
- `07:46:53.698` → `CPU RENDER DONE` logged
- Render wall time: **18.464s**

In the baseline, using phase weights [8, 7, 30, 55] with total 28.1s:
- Expected render: 28.1 × 30% = 8.43s
- Expected encode: 28.1 × 55% = 15.46s

But phase weights are RELATIVE not absolute. The actual render time in the baseline is unknown because we don't have log timestamps.

**Possible explanations for 18.5s render:**
1. Render always took ~18.5s; the 8.4s "estimate" from phase weights was inaccurate
2. Something we changed made rendering 2× slower (open pipe early? `tobytes()` in workers?)
3. The `_render_chunk` now does `img.tobytes()` in the worker, which adds ~1ms per frame × 1362 = 1.4s extra (not ~10s)

## What Was Actually Changed vs Original

The original parallel path code (basically unchanged):
```python
state_images = [None] * (sol_len + 1)
num_needed = len(states_needed)
_render_prog_step = max(1, num_needed // 100)
chunks = _build_chunks(states_needed)
if parallel and len(chunks) > 1:
    workers = min(os.cpu_count() or 4, len(chunks))
    done = 0
    _render_prog_count = 0
    pool = ProcessPoolExecutor(max_workers=workers)
    fut_to_chunk = {}
    for chunk in chunks:
        fut = pool.submit(_render_chunk, chunk, frame_params)
        fut_to_chunk[fut] = chunk
    remaining = set(fut_to_chunk.keys())
    while remaining:
        done_set, _ = wait(remaining, timeout=0.2, return_when=FIRST_COMPLETED)
        for fut in done_set:
            chunk_indices = fut_to_chunk[fut]
            chunk_results = fut.result()
            for idx, img in zip(chunk_indices, chunk_results):
                state_images[idx] = img
                done += 1
                _render_prog_count += 1
                if progress_callback and (_render_prog_count % _render_prog_step == 0 or done == num_needed):
                    progress_callback(done, num_needed, desc="Render" if _render_prog_count == _render_prog_step else None)
            remaining.remove(fut)

# THEN open ffmpeg pipe and encode
ffmpeg_proc = _create_ffmpeg_pipe(output_path, ...)
for state_idx in frame_state:
    ffmpeg_proc.stdin.write(np.array(state_images[state_idx]).tobytes())
```

My parallel path code:
```python
# same as before BUT:
# 1. Open ffmpeg pipe BEFORE the render loop
ffmpeg_proc = _create_ffmpeg_pipe(...)

# 2. Then render (same as original)
# 3. Then encode with .tobytes() instead of np.array().tobytes()
for state_idx in frame_state:
    ffmpeg_proc.stdin.write(state_images[state_idx].tobytes())
```

## Remaining Ideas to Speed Up CPU

### Bottleneck Analysis

The render phase is 18.5s for 1362 unique frames. With 8 workers:
- Per-worker frames: ~170
- Per-frame time: 18.5/170 = ~109ms per frame

`render_frame()` does per frame (with delta_mask, ~2 tiles changed):
1. `prev_canvas.copy()` → ~3-5ms (full canvas copy, 960×696 × 3 bytes = 2MB)
2. Render 2 tile composites → ~1-2ms
3. Timer bar text rendering → ~5-10ms
4. Stats panel: `_apply_stats_dynamic()` → ~15-25ms (copies static base, renders dynamic text surfaces, pastes)
5. `canvas.convert('RGB')` at end → ~3-5ms
6. Panel compositing (alpha_composite) → ~5ms

Total per frame: ~32-52ms

But 109ms is more than this. Additional overhead:
- Pickle/unpickle frame_params for IPC (ProcessPoolExecutor)
- PIL Image serialization between processes (returning images)
- GC pauses from creating many PIL objects
- Overhead in `_render_chunk` loop

### Ideas to further improve

**Idea A: Avoid `prev_canvas.copy()` by mutating in-place**
Change `render_frame()` to mutate `prev_canvas` instead of copying it. Since `prev_canvas = img` replaces the reference, the old canvas is available for mutation.

**Estimated: ~2-3s saved** (1362 × 2ms)

**Idea B: Pre-render all dynamic stats values as numpy arrays**
Instead of rendering text to PIL surfaces each frame via `_apply_stats_dynamic()`, pre-render all unique dynamic text overlays during the prep phase. Store as numpy arrays and blit using numpy slicing in the render function.

**Estimated: ~3-5s saved** (dominates per-frame time)

**Idea C: Render directly to raw bytearray (avoid PIL entirely)**
Replace `render_frame()` with a numpy-based renderer that:
1. Composes the grid via numpy indexing
2. Renders text via pre-built font atlases (similar to GPU path)
3. Returns raw RGB bytes directly (no PIL Image creation)

This avoids PIL overhead entirely but requires significant work.

**Estimated: ~5-8s saved**

**Idea D: Multiprocessing pipe (direct write from workers)**
Have each worker process write rendered frames directly to the ffmpeg pipe instead of returning them. Workers could share the pipe fd. But pipe writes must be in video order, so this requires careful sequencing.

**Estimated: ~2-3s saved** (avoids IPC + serialization overhead)

**Idea E: Reduce unique frames / FPS**
Drop from 60fps to 30fps. This reduces:
- Total frames: 3167 → ~1584
- Unique states: 1362 → ~860 (fewer unique states at lower fps)
- Encode time: ~7s → ~3.5s
- Render time: ~18.5s → ~12s

**Estimated: ~10s saved** but changes output quality

**Idea F: Numba JIT for the `tile_sprites is not None` render path**
JIT-compile the inner tile rendering loop with numba. For the full-render path (first frame of each chunk), this could be 5-10× faster.

**Estimated: ~1-2s saved**

### Summary Table

| Optimization | Est. Save | Effort | Risk |
|---|---|---|---|
| A. Mutate canvas in-place | 2-3s | Low | Low |
| B. Pre-render stats values | 3-5s | Medium | Low |
| C. Numpy bytearray renderer | 5-8s | High | Medium |
| D. Direct pipe from workers | 2-3s | High | Medium |
| E. Drop FPS to 30 | ~10s | Low | Medium (visual) |
| F. Numba JIT | 1-2s | Medium | Low |

## Testing Commands

```powershell
# CPU test
python main.py --file test_replays/10x10 --no-gpu

# GPU test
python main.py --file test_replays/10x10

# With logging
python main.py --file test_replays/10x10 --no-gpu --log
Get-ChildItem -Path logs -Name | Select-Object -Last 1 | ForEach-Object { Get-Content "logs/$_" | Select-String "CPU RENDER DONE|FFMPEG DONE" | Select-Object -Last 2 }
```

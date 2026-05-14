# CPU Rendering Optimization

## Goal
Make CPU (`--no-gpu`) rendering of sliding puzzle replays fast and reliable for any puzzle size without using page file or SharedMemory.

## Constraints & Preferences
- No SharedMemory — it uses the Windows page file, fails on large puzzles (OSError 1455), and is slower than serial for 16×16.
- No file-backed memmap — slower due to Windows disk I/O overhead per batch.
- No ThreadPoolExecutor — GIL contention makes it slower than ProcessPoolExecutor.
- Must work for 20×20 without crashing or using 51 GB of page file.
- C:\ drive has 26.5 GB free; total system RAM is 16 GB with ~10.6 GB available.
- `parallel` and `shared_pool` parameters removed from `generate_frames()` — external callers must not pass them.

## Progress

### Done
- **Unified NVENC encode** — auto-detect hevc_nvenc for CPU path. Encode dropped ~15.5s → ~7s.
- **Eliminated panel alpha compositing** — replaced 5 full-canvas RGBA allocations per frame with pre-blended draw.rectangle() calls. Saved ~1.5s.
- **Eliminated static_base.copy()** — stats dynamic values paste directly onto canvas. Saved ~1s.
- **Overlapped render + encode** — frames are written to ffmpeg pipe immediately after rendering. Total dropped from 15s → 12s for 10×10.
- **10×10 result: CPU 10.7s vs GPU 11.5s** — CPU slightly faster than GPU.
- **16×16 baseline tested** — serial 74.4s vs parallel (SharedMemory) 77.0s. Serial is faster because pipe write/encode dominates, not render.
- **Removed parallel SharedMemory path entirely** — serial path is the only render path now. No SharedMemory, no ProcessPoolExecutor for rendering, no _render_chunk/_render_chunk_mmap/_render_chunk_shm functions.
- **Removed `_build_chunks` function** — no longer needed after parallel path removal.
- **Cleaned up unused imports** — `SharedMemory` import removed.
- **Removed `parallel` and `shared_pool` parameters** from `generate_frames()` signature.
- **Removed stale `parallel` parameter from `generate_simple_replay`** and from `_batch_cpu_worker` call — would have caused TypeError at runtime.
- **Fixed CPU progress tracking for overlapped pipe** — CPU now uses `GPU_PHASE_WEIGHTS = [8, 7, 85]` since there's no separate encode phase. Progress ramps smoothly 8%→100% instead of jumping from ~44%→100%.

### Baseline Benchmarks (10×10, 1569 moves, 1362 unique / 3167 total frames, 960×696 canvas)

| Path | Time | Notes |
|------|------|-------|
| CPU (`--no-gpu`) | **13.1s** | Serial render + pipe, hevc_nvenc, smooth progress |
| GPU (default) | **12.4s** | GPU render + pipe, hevc_nvenc |

Both are essentially tied on 10×10. The CPU optimizations (overlapped pipe, eliminated alpha compositing, no static_base copy) closed the gap entirely.

### In Progress
- (none)

### Blocked
- 20×20: SharedMemory fails at 51 GB (page file too small). Serial fallback is the only option — estimated ~159 s based on per-state render time × 11177 states + 108 GB encode through pipe.

## Important: Progress Bar Awareness

Any optimization that changes the phase structure (e.g. adding a separate encode pass, splitting render into stages, adding post-processing) **must** update `track_progress.py` phase weights accordingly. The phase weights are the single source of truth for tracking. The rule: each phase weight should match the proportion of total time that phase actually takes. When phases are overlapped, merge their weights into a single combined phase.

Current phase weights in `track_progress.py`:
- Both paths: `[8, 7, 85]` — Analysis, Precompute, Combined Render+Encode (overlapped)
- Batch: `[100]` — single-phase item-level progress

## Optimization Test Plan

### Test Commands

```bash
# CPU baseline:
python main.py --file test_replays/10x10 --no-gpu --log

# GPU baseline:
python main.py --file test_replays/10x10 --log
```

### Results Table (10×10, 1362 unique / 3167 total frames, 960×696 canvas)

| Variant | Optimization | Path | 10x10 total | 10x10 f/s | vs CPU base | vs GPU base |
|---------|-------------|------|-------------|-----------|-------------|-------------|
| Baseline CPU | Current serial + overlapped pipe (hevc_nvenc) | CPU | 13.1s | 104 | — | +5.6% |
| Baseline GPU | Current GPU path | GPU | 12.4s | 110 | -5.3% | — |
| A | Unique frames + concat demuxer | Both | TBD | TBD | TBD | TBD |
| B | Backport CPU overlays (pre-blended draw, cached static_base) to GPU | GPU | TBD | TBD | TBD | TBD |

### Variant Details

#### A: Unique frames + concat demuxer

**Path:** Both CPU and GPU

**Idea:** Instead of writing every duplicate frame to the ffmpeg pipe (e.g. 23562 total → 11177 unique for 20×20), write each unique frame once with its repeat count via ffmpeg concat demuxer (text file specifying per-frame duration).

**Gain:** Cuts pipe data from 108 GB to ~51 GB for 20×20 (11177 writes instead of 23562), proportionally reducing NVENC encode time.

**Tradeoff:** Requires temp file for the concat script; adds disk I/O for the concat file (negligible size).

**Implementation sketch:**
1. Render unique frames, write to temp rawvideo file (or pipe).
2. Build concat file: `file frame.raw` + `duration N/60` per unique frame.
3. Run ffmpeg with concat demuxer.

#### B: Backport CPU overlays to GPU

**Path:** GPU only

**Idea:** The GPU path still uses 5 full-canvas RGBA allocations per frame (alpha compositing) and `static_base.copy()` for stats updates. Apply the same techniques that saved ~2.5s on CPU:
- Pre-blended `draw.rectangle()` calls instead of full-canvas alpha blends.
- Cache static base, only paste dynamic values.

**Expected gain:** Modest (~5-10%) since GPU render time is already fast and dominated by tile rendering, not overlays.

## Key Decisions
- **Serial-only render path** — parallel SharedMemory path removed because: (1) serial is faster for 16×16 (no IPC overhead), (2) SharedMemory fails on large puzzles, (3) the pipe-write/encode is the bottleneck, not render.
- **Keep ProcessPoolExecutor for batch mode** (multiple separate solutions) — `_batch_cpu_worker` still uses it; single-solution rendering is always serial.
- **Keep NVENC p7 preset** — preset was never the bottleneck; encode speed is limited by the 108 GB of raw pipe data, not encoder settings.
- **`parallel` / `shared_pool` removed** from public API — callers should not pass these; `use_gpu` remains the only mode switch.

## Critical Context
- **10×10** (1569 moves): 1362 unique states / 3167 total frames, 960×696 canvas, 2.0 MB/frame. CPU serial 12.5s, GPU 12.4s.
- **16×16** (7132 moves): 5692 unique states / 12923 total frames. Serial 74.4s, parallel 77.0s.
- **20×20** (14203 moves): 11177 unique states / 23562 total frames, 1380×1116 canvas, 4.6 MB/frame. Total pipe data: 23562 × 4.6 MB = 108 GB. NVENC encode at ~700 MB/s takes ~154 s alone.
- Serial path is O(1) memory — only current frame canvas in RAM (< 5 MB).
- The bottleneck is the pipe write to ffmpeg (108 GB for 20×20), not the rendering itself.
- All previous optimization attempts (pool reuse, SHM reuse, thread pool, RAM buffer, file-backed memmap) were reverted — none improved total time enough to justify added complexity.
- `ProcessPoolExecutor` import kept for `_batch_cpu_worker`; it is no longer used in the single-solution render path.
- The `parallel` parameter is gone — there is no parallel option for single-solution render anymore.

## Relevant Files
- `replay_video.py`: serial render path only (lines ~1639–1665), removed SharedMemory imports, `_build_chunks`, `parallel`/`shared_pool` params
- `track_progress.py`: ProgressTracker with phase weights. CPU = [8, 7, 30, 55], GPU = [8, 7, 85], need to fix CPU to match overlapped pipe.
- `optimizing_cpu.md`: this file
- `test_replays/10x10`, `test_replays_gpu/16x16`, `test_replays_gpu/20x20`: test replay files for benchmarking

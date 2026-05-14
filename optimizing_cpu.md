# CPU Rendering Speed Optimization

## Goal
Make CPU rendering (with `--no-gpu`) approach GPU rendering speed.

## Test Cases
- 10×10: 1569 moves, 1362 unique / 3167 total frames, 960×696 canvas, 2.0 MB/frame
- 16×16: 7132 moves, 5692 unique / 12923 total frames
- 20×20: 14203 moves, 11177 unique / 23562 total frames, 1380×1116 canvas, 4.6 MB/frame

## Baseline Results (current committed code)

| Size | Total | Notes |
|------|-------|-------|
| **10×10** | **10.7s** | CPU slightly faster than GPU (11.5s) |
| **16×16** | **77.0s** | SharedMemory succeeded (5700×2.0MB = ~11GB) |
| **20×20** | N/A | SharedMemory fails (11177×4.6MB = 51GB), falls to serial |

## Current Architecture (committed)

### Single-shot parallel (SharedMemory)
When all states fit in shared memory:
1. Pre-allocate `SharedMemory(size=num_needed * frame_bytes)`
2. Workers (`ProcessPoolExecutor`) call `_render_chunk_shm()` — renders frames into the shared buffer
3. Main process reads from the same buffer via `np.ndarray()` and writes to ffmpeg pipe
4. Overlapped: as chunks complete, frames are written to pipe immediately

### Serial fallback
When SharedMemory allocation fails (too large):
1. Render each unique state one-at-a-time
2. Write each frame N times to ffmpeg pipe immediately (overlap render + encode)
3. O(1) memory — only current frame in RAM

### No batched parallel path exists in current committed code
The refactored batched parallel path (ProcessPoolExecutor + SharedMemory per batch) was reverted.

## Key Bottlenecks Identified

### 1. SharedMemory uses page file
On Windows, `SharedMemory` uses the system paging file. For 20×20:
- 11177 states × 4.6MB = 51GB needed
- `OSError: [WinError 1455] The paging file is too small`
- Falls to serial path, which is slow (~159s estimated)

### 2. Serial path is too slow for large puzzles
Each state rendered sequentially. For 20×20 with ~57ms/state, 11177×57ms = 637s of render.

### 3. Pipe write bandwidth
23562 frames × 4.6MB = 108GB through ffmpeg stdin pipe. At ~700MB/s (NVENC encode rate), encode takes ~154s.

### 4. GIL contention with threads
ThreadPoolExecutor for parallel rendering doesn't give real parallelism due to Python GIL. PIL/numpy C extensions release GIL, but Python-level loop overhead causes contention.

### 5. File-backed memmap IO
Using `np.memmap` backed by disk files avoids page file but adds disk I/O overhead (especially on Windows with sequential file creation/deletion per batch).

### 6. ProcessPoolExecutor pool creation overhead
Creating/destroying ProcessPoolExecutor for each batch is expensive (~1-2s per batch).

## Previous Optimizations Attempted (all reverted)

| Change | 10×10 Effect | 20×20 Effect | Reverted? |
|--------|-------------|-------------|-----------|
| Pool reuse across batches | — | -20s (207→187s) | Yes |
| SHM reuse across batches | — | -5s (187→182s) | Yes |
| ThreadPoolExecutor (GIL contention) | — | slower | Yes |
| RAM buffer (numpy zeros) | — | ~same (182s) | Yes |
| File-backed memmap per batch | — | slower | Yes |
| NVENC p1 preset | — | -8s (180→172s) | Yes |

## Next Steps

### Option A: Batched parallel with ProcessPoolExecutor + SharedMemory per batch
Already tested: 20×20 ran in ~187s (after pool reuse). Falls to serial if SHM fails per-batch too.

### Option B: Pre-allocate large file on disk, use as memmap
26.5GB free on C:\. 20×20 needs 51GB total. Batched approach would use e.g. 4GB files. Tested slower due to Windows file I/O overhead.

### Option C: Reduce canvas resolution for CPU path
Scale down frames before writing to pipe. Reduces both render and encode time. Quality trade-off.

### Option D: Fix serial path performance
The serial path is the fallback for 20×20. If we can make serial go from ~159s to ~90s, that might be acceptable.

### Option E: Write unique states only (skip duplicates)
Instead of writing each video frame (23562), write each unique state (11177) with frame duration metadata. Requires ffmpeg VFR support.

## System Specs
- CPU: 6 cores (12 logical)
- RAM: 16 GB (10.6 GB available)
- GPU: NVIDIA (hevc_nvenc available)
- Disk: C: 26.5 GB free, F: 13.4 GB free

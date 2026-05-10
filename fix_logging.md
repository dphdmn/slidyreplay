# Logging & RAM Monitoring — Implementation Plan

## Overview

Two categories of changes:
- **Logging improvements** — reduce noise, add signal, add structure
- **RAM monitoring** — track system RAM deltas at every major phase

---

## Part 1 — General Infrastructure

### 1a. Add `psutil` dependency

**File:** `requirements.txt` (line 6)

**Action:** Append `psutil` to the dependency list.

```txt
psutil
```

**Why:** Cross-platform process memory monitoring (`psutil.Process().memory_info().rss`). Used only when `--log` flag is active. Tiny, pure-Python package.

---

### 1b. Add RAM logging utility

**File:** `replay_video.py` — insert at module scope, after `log = get_logger()` (line 36)

**After line 36 (`log = get_logger()`), insert:**

```python
import psutil
_proc = psutil.Process()
_baseline_ram = _proc.memory_info().rss

def log_ram(label: str) -> int:
    """Log current process RSS in MB with delta from baseline. Returns delta in bytes."""
    cur = _proc.memory_info().rss
    delta = cur - _baseline_ram
    log.info(f"  RAM [{label}]: {cur // (1024*1024)}MB ({delta // (1024*1024):+d}MB vs baseline)")
    return delta
```

**Note:** If you don't want to import psutil at module load (lazy import), wrap the import inside the function:
```python
def log_ram(label: str) -> int:
    import psutil
    ...
```

---

## Part 2 — `replay_video.py` changes

### 2a. Delete HANDLER log noise

**File:** `replay_video.py`, line 1502

**Delete this entire line:**
```python
            log.info(f"  HANDLER: idx_in_unique={idx_in_unique}, state={states_needed[idx_in_unique]}, count={count}, data_size={len(data)} bytes")
```

**Why:** Produces 1840 near-identical lines (96% of log file). The only varying fields (`idx_in_unique`, `state`, `count`) are not useful for debugging — pipe errors surface as exceptions with tracebacks. Removed line count: 1.

**After deletion, the `handler` function becomes:**
```python
        def handler(img, idx_in_unique, total):
            count = state_to_count[states_needed[idx_in_unique]]
            data = img.tobytes()
            for _ in range(count):
                enc_proc.stdin.write(data)
```

---

### 2b. Add pipeline stage markers with durations

**File:** `replay_video.py` — inside `generate_frames`, add section markers that wrap major phases.

**There are 4 stages in the GPU path:**

#### Stage 1 — Data Prep (insert before line ~1229 `log.info("generate_frames: ...")`)

```python
    _t_stage1 = time_module.time()
    log.info("====== STAGE 1: DATA PREP ======")
    # ... existing code through line ~1440 ...
    log.info(f"====== STAGE 1 DONE: {time_module.time() - _t_stage1:.1f}s ======")
```

#### Stage 2 — Overlay Pre-Render (insert before line ~1466 `log.info("  OVERLAY PRE-RENDER: ...")`)

```python
    _t_stage2 = time_module.time()
    log.info("====== STAGE 2: OVERLAY PRE-RENDER ======")
    # ... existing overlay code (lines 1466-1481) ...
    log.info(f"====== STAGE 2 DONE: {time_module.time() - _t_stage2:.1f}s ======")
```

#### Stage 3 — GPU Render (insert before line ~1497 `log.info("  GPU RENDER START: ...")`)

```python
    _t_stage3 = time_module.time()
    log.info("====== STAGE 3: GPU RENDER ======")
    # ... existing GPU render call + handler code (lines 1497-1517) ...
    log.info(f"====== STAGE 3 DONE: {time_module.time() - _t_stage3:.1f}s ======")
```

**Note:** Stage 3 timing captures the render_frames call + ffmpeg pipe writes (the handler), which happen interleaved.

#### Stage 4 — Encode (optional, since ffmpeg pipe close is part of Stage 3 cleanup)

If you want a separate encode stage, it would be after `_close_pipe(enc_proc)` at line 1517, but on the GPU path encoding happens inline via the handler. Not needed as a separate stage.

**CPU path equivalent:**

```python
    _t_stage_cpu = time_module.time()
    log.info("====== STAGE: CPU RENDER ======")
    # ... existing CPU path code (lines 1523-1578) ...
    log.info(f"====== STAGE: CPU RENDER DONE: {time_module.time() - _t_stage_cpu:.1f}s ======")
```

---

### 2c. Add RAM measurement points — GPU path

**Insert before existing lines.** Place `log_ram()` calls at these strategic locations:

#### After overlay pre-render completes (after line 1481 `log.info("  OVERLAY PRE-RENDER DONE: ...")`)

This captures the peak RAM after all 1840 timer/stats images are stored in `frame_params`.

```python
        log.info(f"  OVERLAY PRE-RENDER DONE: {overlay_done}/{overlay_total}")
        log_ram("after overlay pre-render")          # <-- INSERT THIS LINE
```

**Expected:** Large delta — ~1GB for 11×11 at quality=1.0.

#### Just before GPU render starts (after line 1497 `log.info("  GPU RENDER START: ...")`)

This is the baseline before batch rendering begins.

```python
        log.info(f"  GPU RENDER START: {len(unique_params)} unique frames to render")
        log_ram("before GPU render")                  # <-- INSERT THIS LINE
```

#### After GPU render completes (after line 1519 `log.info("  GPU PATH COMPLETE: ...")`)

This captures whether RAM was freed after render.

```python
        log.info(f"  GPU PATH COMPLETE: returning {len(frame_state)} frame_state entries")
        log_ram("after GPU render")                   # <-- INSERT THIS LINE
```

---

### 2d. Add RAM measurement points — CPU path

#### Before font loading (insert before line ~1525)

```python
    log_ram("CPU: before font load")                 # <-- INSERT THIS LINE
    _font_start = time_module.time()
    get_font(font_size)
    ...
```

#### After font loading (insert after line 1533)

```python
    log.info(f"  fonts loaded: took {time_module.time() - _font_start:.3f}s")
    log_ram("CPU: after font load")                  # <-- INSERT THIS LINE
```

#### Before rendering all state_images (insert after line 1536)

```python
    state_images = [None] * (sol_len + 1)
    num_needed = len(states_needed)
    log_ram("CPU: before render")                    # <-- INSERT THIS LINE
```

#### After all state_images rendered (insert after line 1566)

This is the peak — all 1840 full-resolution images stored in `state_images[]`.

```python
    log.info(f"  CPU RENDER DONE: canvas={canvas_w}x{canvas_h}")
    log_ram("CPU: after render (all frames in mem)") # <-- INSERT THIS LINE
```

**Expected:** Very large delta — ~4.2GB for 11×11 (1840 frames × 2.3MB each).

#### After ffmpeg pipe done (insert after line 1578)

```python
    log.info(f"  CPU FFMPEG DONE: {written} frames written, returncode={ffmpeg_proc.returncode}")
    log_ram("CPU: after ffmpeg pipe")                # <-- INSERT THIS LINE
```

---

### 2e. Add environment snapshot (GPU path)

**File:** `replay_video.py` — inside `generate_frames`, somewhere between lines 1335 and 1440 (after GPU availability info, before overlay pre-render).

**Insert after line ~1335 (`log.info(f"  canvas={gpu.canvas_w}x{gpu.canvas_h}, ...")`)**

```python
    log.info(f"  Python={sys.version.split()[0]}, torch={torch.__version__}, CUDA={torch.version.cuda}")
```

**Note:** Need to `import sys` at the top of the file (already present — confirmed at line 16).

**Also need to check torch import at function scope**: torch may not be imported at the top of `replay_video.py`. It's imported inside `_nvenc_available()` or similar. If `torch` is not in scope at the `generate_frames` level, move this log line inside the `if use_gpu:` block where `gpu` object is available, and use `torch.version.cuda` from the already-imported torch (gpu_renderer imports torch at line 18). Or add a local import:

```python
    if use_gpu:
        import torch
        log.info(f"  Python={sys.version.split()[0]}, torch={torch.__version__}, CUDA={torch.version.cuda}")
```

---

## Part 3 — `gpu_renderer.py` changes

### 3a. Add `import psutil` and per-process RAM delta utility

**File:** `gpu_renderer.py` — at the top, after existing imports (after line 16 or at module scope)

```python
import psutil
_gpu_proc = psutil.Process()
_gpu_baseline_ram = _gpu_proc.memory_info().rss

def _ram_delta_mb() -> int:
    """Return current RSS delta from baseline in MB."""
    return (_gpu_proc.memory_info().rss - _gpu_baseline_ram) // (1024 * 1024)
```

### 3b. Rewrite BATCH log line

**File:** `gpu_renderer.py`, line 364

**Current:**
```python
log.info(f"  BATCH: batch_start={batch_start}, batch_n={batch_n}, batch_size={batch_size}, free_mem={free_mem // (1024*1024)}MB, usable={usable // (1024*1024)}MB, per_frame_ema={per_frame_ema // (1024*1024)}MB")
```

**Replace with:**
```python
elapsed = time.time() - _batch_t0
peak_mem = torch.cuda.max_memory_reserved(dev)
log.info(f"  BATCH[{self._batch_counter}]: start={batch_start}, n={batch_n}, sz={batch_size}, "
         f"free={free_mem//(1024*1024)}MB, reserved={reserved_mem//(1024*1024)}MB, "
         f"peak={peak_mem//(1024*1024)}MB, usable={usable//(1024*1024)}MB, "
         f"ema={per_frame_ema//(1024*1024)}MB, batch_mem={self._stats['batch_mem_mb']}MB, "
         f"t={elapsed:.2f}s, ram={_ram_delta_mb()}MB")
```

**Prerequisites for this change:**

1. **`_batch_t0` timer** — add before the while loop (see step 3c)

2. **`torch.cuda.reset_peak_memory_stats(dev)`** — verify it's already called at line 424 (it is: `torch.cuda.reset_peak_memory_stats(dev)`). The `peak_mem` value must be read AFTER the batch work is done (tensors freed) but before the next `reset_peak_memory_stats`. Currently `reset_peak_memory_stats` is at line 424, called before canvas allocation. The BATCH log is at line 364 (before any batch work!), so this is a problem — `peak_mem` at line 364 would reflect the previous batch's peak.

**Fix:** Add `torch.cuda.reset_peak_memory_stats(dev)` right before the BATCH log entry (so it starts fresh for the coming batch), and read `peak_mem` AFTER the batch work completes. But the BATCH log fires BEFORE the batch work...

**Better approach:** Move the peak memory reading to after the batch work (around line 518, after `del`). Add a separate post-batch log or include peak in the post-batch section. Simplest: just read `peak_mem` before `reset_peak_memory_stats(dev)` is called for the *next* batch. The sequence should be:

```
reset_peak_memory_stats  <- at line 424 (already there, start of batch work)
... batch work ...
read peak_mem             <- AFTER batch work, BEFORE freeing tensors? Or after freeing?
log BATCH line            <- after peak_mem is read
del tensors               <- existing lines 518-521
empty_cache               <- existing line 522
```

Wait, the BATCH log is BEFORE the batch work (before the tensor upload). This is the "we're about to render this batch" log. The peak would be from the previous batch.

**Simplest correct implementation:**

Add a second log AFTER the batch work, or include peak in the post-batch log. Change the approach:

```python
# BEFORE batch work:
log.info(f"  BATCH[{self._batch_counter}]: start={batch_start}, sz={batch_size}, "
         f"free={free_mem//(1024*1024)}MB, reserved={reserved_mem//(1024*1024)}MB, "
         f"usable={usable//(1024*1024)}MB, ema={per_frame_ema//(1024*1024)}MB, "
         f"batch_mem={self._stats['batch_mem_mb']}MB")

# ... batch work ...

# AFTER batch work (after line 517, before `batch_start = batch_end`):
log.info(f"  BATCH[{self._batch_counter}] DONE: t={time.time()-_batch_t0:.2f}s, "
         f"peak={torch.cuda.max_memory_reserved(dev)//(1024*1024)}MB, ram={_ram_delta_mb()}MB")
```

This is cleaner — the "before" line has the budget info, the "after" line has the actual measurements. But doubles the batch log lines from 1 to 2.

**Alternative (single line):** Read peak AFTER batch work but include it in a single log. Move the log to after the batch work. The before-line info (free_mem, reserved_mem, etc.) would be stale (from the *next* batch's budget calc). Not ideal.

**Recommended approach:** Two lines — one before (budget), one after (actuals). The extra line per batch is negligible compared to the HANDLER noise we're deleting.

---

### 3c. Add per-batch timer

**File:** `gpu_renderer.py` — before the while loop (before line 326)

**Insert:**
```python
        _batch_t0 = time.time()
```

This initializes the timer. The elapsed is read in the BATCH log (step 3b).

---

### 3d. Add post-batch DONE log with peak_mem

**File:** `gpu_renderer.py` — insert before line 524 (`batch_start = batch_end`)

**Insert:**
```python
                log.info(f"  BATCH[{self._batch_counter}] DONE: "
                         f"t={time.time()-_batch_t0:.2f}s, "
                         f"peak={torch.cuda.max_memory_reserved(dev)//(1024*1024)}MB, "
                         f"ram={_ram_delta_mb()}MB")
```

**Expected position (around line 523, after `torch.cuda.empty_cache()`):**
```python
                torch.cuda.empty_cache()
                log.info(f"  BATCH[{self._batch_counter}] DONE: "
                         f"t={time.time()-_batch_t0:.2f}s, "
                         f"peak={torch.cuda.max_memory_reserved(dev)//(1024*1024)}MB, "
                         f"ram={_ram_delta_mb()}MB")
                batch_start = batch_end
```

---

### 3e. Add calibration detail log

**File:** `gpu_renderer.py` — insert after line 516 (after `self._stats["per_frame_ema_mb"] = ...`)

**Insert:**
```python
                    log.info(f"  CALIBRATION: reserved_peak={reserved_peak//(1024*1024)}MB, "
                             f"reserved_permanent={reserved_permanent//(1024*1024)}MB, "
                             f"marginal_cost={marginal_cost//(1024*1024)}MB, "
                             f"per_frame_ema={per_frame_ema/(1024*1024):.0f}MB")
```

**Expected context (after insert):**
```python
                if self._batch_counter == 1 and batch_n == 1 and per_frame_ema == 0.0:
                    marginal_cost = reserved_peak - reserved_permanent
                    if marginal_cost > 0:
                        per_frame_ema = marginal_cost
                        self._stats["per_frame_ema_mb"] = per_frame_ema / (1024 * 1024)
                        log.info(f"  CALIBRATION: reserved_peak={reserved_peak//(1024*1024)}MB, "
                                 f"reserved_permanent={reserved_permanent//(1024*1024)}MB, "
                                 f"marginal_cost={marginal_cost//(1024*1024)}MB, "
                                 f"per_frame_ema={per_frame_ema/(1024*1024):.0f}MB")
```

---

### 3f. Add final GPU summary

**File:** `gpu_renderer.py` — insert after line 526 (`log.info(f"render_frames: DONE. ...")`)

**Insert:**
```python
        total_t = time.time() - _batch_t0
        log.info(f"===== GPU RENDER SUMMARY =====")
        log.info(f"  total_time={total_t:.1f}s, batches={self._batch_counter}, "
                 f"frames={n}, avg_batch_size={n/max(1,self._batch_counter):.1f}, "
                 f"throughput={n/total_t:.0f} f/s (unique)")
        log.info(f"  RAM delta: {_ram_delta_mb()}MB")
```

**Expected position (after `batch_start` is reset after the loop, at the DONE line):**
```python
        log.info(f"render_frames: DONE. total_batches={self._batch_counter}, frames_rendered={batch_start}")
        total_t = time.time() - _batch_t0
        log.info(f"===== GPU RENDER SUMMARY =====")
        log.info(f"  total_time={total_t:.1f}s, batches={self._batch_counter}, "
                 f"frames={n}, avg_batch_size={n/max(1,self._batch_counter):.1f}, "
                 f"throughput={n/total_t:.0f} f/s (unique)")
        log.info(f"  RAM delta: {_ram_delta_mb()}MB")
        torch.cuda.empty_cache()
        return frames if not frame_handler else []
```

---

## Implementation Order

### Phase 1 — Dependencies (1 file, 1 line)
1. Add `psutil` to `requirements.txt`

### Phase 2 — `replay_video.py` (5 changes)
2. Add `log_ram()` utility + `psutil` import (after line 36)
3. Delete HANDLER log (line 1502)
4. Add pipeline stage markers (4 insertion points around existing log lines)
5. Add RAM measurement points (6 insertions at strategic locations)
6. Add environment snapshot (near line 1335)

### Phase 3 — `gpu_renderer.py` (6 changes)
7. Add `_ram_delta_mb()` utility + `psutil` import (top of file)
8. Add `_batch_t0 = time.time()` before while loop (before line 326)
9. Rewrite BATCH line (line 364) — shorten, add fields, remove duplicate info
10. Add post-batch DONE log with peak_mem + ram (before line 524)
11. Add calibration detail log (after line 516)
12. Add final GPU summary (after line 526)

---

## Net File Changes

| File | Lines Added | Lines Deleted | Net |
|------|-------------|---------------|-----|
| `requirements.txt` | 1 | 0 | +1 |
| `replay_video.py` | ~25 | 1 | +24 |
| `gpu_renderer.py` | ~18 | 1 (rewrite) | +17 |
| **Total** | **~44** | **2** | **+42** |

## Expected Log Output (example, 11×11 GPU path)

```
====== STAGE 1: DATA PREP ======
  matrix source: provided scramble, 11x11
  matrix=11x11, sol_len=1839
  tps_val=15, ...
  grids_data: ...
  timing: ...
  calling generate_frames: ...
generate_frames: 11x11, ...
  tile_size=58, raw_tile=29, font_size=29
  frame_state: total_frames=7452, ...
  canvas=1018x754, GPU available=True, ...
  Python=3.11.5, torch=2.1.0, CUDA=12.1
  frame_params created: 1840 entries
  render decision: use_gpu=True, ...
====== STAGE 1 DONE: 0.5s ======
====== STAGE 2: OVERLAY PRE-RENDER ======
  RAM [baseline]: 180MB (+0MB vs baseline)
  OVERLAY PRE-RENDER: 1840 states, 6 workers
  OVERLAY PRE-RENDER DONE: 1840/1840
  RAM [after overlay pre-render]: 1100MB (+920MB vs baseline)
====== STAGE 2 DONE: 10.2s ======
====== STAGE 3: GPU RENDER ======
  RAM [before GPU render]: 1100MB (+920MB vs baseline)
render_frames: 1840 frames, canvas=1018x754, ...
  BATCH[1]: start=0, sz=1, free=5104MB, reserved=22MB, usable=2793MB, ema=0MB, batch_mem=9MB
  CALIBRATION: reserved_peak=284MB, reserved_permanent=22MB, marginal_cost=262MB, per_frame_ema=24MB
  BATCH[1] DONE: t=0.23s, peak=284MB, ram=+920MB
  BATCH[2]: start=1, sz=80, free=4332MB, reserved=152MB, usable=2021MB, ema=24MB, batch_mem=738MB
  BATCH[2] DONE: t=0.21s, peak=1920MB, ram=+1100MB
  BATCH[3]: start=81, sz=58, free=4476MB, reserved=182MB, usable=2165MB, ema=24MB, batch_mem=535MB
  ...
  BATCH[31]: start=1788, sz=52, free=4424MB, reserved=168MB, usable=2113MB, ema=24MB, batch_mem=480MB
  BATCH[31] DONE: t=0.14s, peak=1200MB, ram=+980MB
render_frames: DONE. total_batches=31, frames_rendered=1840
===== GPU RENDER SUMMARY =====
  total_time=17.2s, batches=31, frames=1840, avg_batch_size=59.4, throughput=107 f/s (unique)
  RAM delta: +980MB
====== STAGE 3 DONE: 17.5s ======
  RAM [after GPU render]: 1000MB (+820MB vs baseline)
  GPU PATH COMPLETE: returning 7452 frame_state entries
```

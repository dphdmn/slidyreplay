# Bulk Rendering Optimizations

## Current Problems

1. **CPU batch: sequential + per-item spawn overhead**
   CLI `--batch` and GUI both iterate solutions one at a time. Each `generate_frames` call creates a new `ProcessPoolExecutor`. On Windows spawn, each pool re-imports the entire module (~2-3s overhead per small puzzle). For 10× 4×4 puzzles: ~45s total when actual rendering is only ~9s.

2. **GPU batch: per-item GPURenderer re-creation**
   Each item creates a fresh `GPURenderer`, uploads number textures to VRAM, then `cleanup()` destroys them. Same-size puzzles use identical textures — entirely redundant.

3. **GUI: ThreadPoolExecutor(max_workers=1) per-item submits**
   Processes items sequentially with no cross-item parallelism. Individual future per item adds complexity for no benefit.

4. **CLI: sequential for-loop**
   `--batch` processes items one at a time in a Python for loop. No parallelism.

## Solution Overview

```
batch_render(items, use_gpu, max_workers)
  ├─ Phase 1: Quick-scan all items → annotate with (w, h, quality)
  ├─ Phase 2: Route to strategy
  │   ├─ GPU path → group by (w, h, quality), sequential per group
  │   └─ CPU path → single ProcessPoolExecutor, disable inner pools
  └─ Phase 3: Return list of output paths
```

## Implementation Plan

### 1. Size Resolution Priority (per item)

No guessing when size is already known — use the first available source:

1. `size` param provided by user → parse directly
2. Replay URL → `parse_replay_url()` may include size info
3. `scramble` string → `scramble_to_puzzle(scramble)` = exact (w, h)
4. Only solution string → `parse_scramble_guess()` first, if that fails → `math.isqrt(len(expand_solution(solution)))` as rough estimate (only needed for GPU grouping, not rendering correctness)

New helper: `_quick_infer_size(solution, scramble, size)` → `(width, height)` or `None`

### 2. CPU Batch: Cross-Solution Parallelism

**Key change:** One shared `ProcessPoolExecutor` at the batch level, `parallel=False` inside each worker to suppress nested pool creation.

```
batch_render(items, use_gpu=False):
    analyzed = _annotate_items(items)  # quick size scan
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futs = [pool.submit(_batch_cpu_worker, item) for item in analyzed]
        for fut in as_completed(futs):
            fut.result()          # propagate errors
            progress_cb(done, n)  # per-item progress
```

**Module-level worker** (pickle-able for ProcessPoolExecutor):
```
_batch_cpu_worker(item: dict) -> str:
    gen = ReplayVideoGenerator()
    gen.generate_simple_replay(solution=..., output_path=...,
                               use_gpu=False, parallel=False, ...)
    return output_path
```

**Heuristic:**
- `len(items) > 1` → outer pool, `parallel=False` per worker
- `len(items) == 1` → existing behavior (inner pool with `parallel=True`)

**Changes to generate_frames:**
- Accept optional `shared_pool: Optional[ProcessPoolExecutor] = None`
- When provided, use it instead of creating a new executor
- When `shared_pool` is provided, do NOT call `.shutdown()` on it
- When `parallel=False` and `shared_pool=None`, use no pool (serial rendering)

### 3. GPU Batch: Renderer Reuse per Size Group

**Key change:** Group items by `(w, h, quality)`, reuse same `GPURenderer` within group.

```
batch_render(items, use_gpu=True):
    analyzed = _annotate_items(items)
    
    groups = {}  # key=(w, h, quality) -> list of items
    for item in analyzed:
        key = (item["w"], item["h"], item.get("quality", 1.0))
        groups.setdefault(key, []).append(item)
    
    renderer = None
    prev_key = None
    for key, group in groups.items():
        if renderer is None or key != prev_key:
            if renderer is not None:
                renderer.cleanup()
            w, h, q = key
            raw_tile = pick_tile_size(w, h)
            renderer = GPURenderer(w, h, raw_tile, quality=q)
        
        for item in group:
            generate_simple_replay(solution=..., output_path=...,
                                   use_gpu=True, gpu_renderer=renderer, ...)
        
        prev_key = key
    
    if renderer:
        renderer.cleanup()
```

**Changes to generate_frames:**
- Accept optional `gpu_renderer: Optional[GPURenderer] = None`
- When provided, skip creating a new `GPURenderer` (line ~1387) and use the passed one
- Do NOT call `gpu.cleanup()` at end when renderer was passed in (caller manages lifecycle)

**Changes to generate_simple_replay:**
- Accept optional `gpu_renderer` and pass through to `generate_frames`

### 4. GUI Integration

Replace per-item `ThreadPoolExecutor` submits with a single `batch_render()` on a background thread.

**Current:**
```
self._executor = ThreadPoolExecutor(max_workers=1)
for idx, (mode, input_str) in enumerate(items):
    fut = self._executor.submit(self._process_item, idx, mode, input_str, ...)
```

**New:**
```
if len(items) == 1:
    # existing single-item path
    ...
else:
    # batch path
    batch_items = _build_batch_items(items, output_dir)
    self._executor = ThreadPoolExecutor(max_workers=1)
    self._executor.submit(self._process_batch, batch_items, ...)
```

`_process_batch` calls `ReplayVideoGenerator.batch_render()` with appropriate progress callback.

**Progress:** Per-item granularity via `external_progress_cb(cur, total)` fired from `as_completed` loop. UI shows "3/10 done" style.

### 5. CLI Integration

Replace:
```
for idx, item in enumerate(items):
    sol, tps, scramble, movetimes = parse_replay_url(val)
    run_single(sol, output_path, ...)
```

With:
```
if args.batch:
    batch_items = [{"solution": sol, "output_path": out_path, ...} for ...]
    gen = ReplayVideoGenerator()
    gen.batch_render(batch_items, use_gpu=use_gpu, show_progress=True)
    return
```

### 6. Files to Modify

| File | Changes |
|------|---------|
| `replay_video.py` | Add `_quick_infer_size()`, `_batch_cpu_worker()`, `ReplayVideoGenerator.batch_render()`. Modify `generate_frames` to accept `shared_pool` and `gpu_renderer`. Modify `generate_simple_replay` to accept `parallel` and `gpu_renderer`. |
| `gpu_renderer.py` | No changes needed. |
| `main.py` | Add `_process_batch()`, `_build_batch_items()`. Modify `_generate()` to dispatch >1 items to batch path. Modify CLI `--batch` to use `batch_render()`. |
| `test_bulk.py` | Update to use `batch_render()` API directly. |

### 7. Expected Speedups

| Scenario | Before | After | Ratio |
|----------|--------|-------|-------|
| CPU 10× 4×4 (batch) | ~45s | ~12s | 3.8× |
| GPU 10× 4×4 (batch) | ~11s | ~8s | 1.4× |
| CPU 30 mixed sizes | ~90s | ~30s | 3× |
| GPU 30 mixed sizes | ~22s | ~19s | 1.2× |

CPU gains: per-call pool spawn eliminated + cross-solution parallelism.
GPU gains: texture upload eliminated for same-size groups.

### 8. Implementation Order

1. `parallel` param on `generate_simple_replay` → `generate_frames`
2. `_quick_infer_size()` helper
3. `_batch_cpu_worker()` module-level function
4. `batch_render()` method on `ReplayVideoGenerator` (CPU path + GPU path with gpu_renderer reuse)
5. `gpu_renderer` param threading through `generate_simple_replay` → `generate_frames`
6. GUI `_generate()` → `_process_batch()` dispatch
7. CLI `--batch` → `batch_render()`
8. Update `test_bulk.py` to use `batch_render` API

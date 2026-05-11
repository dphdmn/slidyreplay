# SlidyReplay — GPU Render Bug Fixes Status

## Summary of All Changes

### Fix 1: Tile Colors Cache — Blank Tile (0) renders with correct background

**File:** `replay_video.py:1399-1403`

**Bug:** `cached_colors[num - 1]` wraps to last tile's color when `num == 0` (Python negative index).

**Fix:** Added explicit `if num == 0` check → `(TILE_BG, None)`.

**Status:** ✅ Fixed. Verified in 12x12 GPU run. The CPU path was already correct.

---

### Fix 2: Incremental Manhattan Distance — correct tile read after swap

**File:** `replay_video.py:~1309`

**Bug:** Read `matrix[nr][nc]` after swap (always 0, the blank). Should read moved tile's old position `matrix[zero_pos[0]][zero_pos[1]]`.

**Fix:** Read `moved_val = matrix[zero_pos[0]][zero_pos[1]]` (the tile that just moved into the zero's old spot). Then compute old MD at `(nr, nc)` and new MD at `zero_pos`.

**Status:** ✅ Fixed in prior session. Verified correct MD values in 12x12 run.

---

### Fix 3: Progress Bar — both GPU paths use 1–99% scale

**Files:** `main.py:~761`, `replay_video.py:~1635-1750`

**Bug:** Raw progress (0 → 2470) mapped to 0–100% directly, but 1% of bar weight should be reserved for the preceding prep stage.

**Fix:** `adjusted_cur = 1 + raw_cur * 99 // raw_tot`, `adjusted_tot = 100`. Applied in both `main.py:_on_item_progress` and `TerminalProgress.__call__`.

**Status:** ✅ Fixed in prior session. Verified 0% → 100% in 12x12 run.

---

### Fix 4: `empty_cache` removed entirely

**File:** `gpu_renderer.py`

**Bug (original):** `empty_cache()` was called every batch, nuking the CUDA caching allocator and causing re-allocation overhead. Even at every-5-batches, it caused VRAM oscillation: the allocator's cache was flushed, the budget algorithm saw fake headroom, allocated a huge batch (spiking VRAM), then the next iteration had no headroom (tiny batch). This 145→4→145→4 cycle repeated indefinitely.

**Fix:** Removed `empty_cache()` entirely. Let the CUDA caching allocator manage memory — it does this well, maintaining cached allocations for reuse across batches.

**Result (11x11, comparison):**
| Metric | With empty_cache (every 5) | Without empty_cache |
|---|---|---|
| Batch size pattern | 73→78→56→45→84→72→66→63→62→90→68→57→52→49→80→79… | 73→78→56→45→40→37→36→35→35→35→35→… (locked at 35) |
| VRAM | Oscillating 1.7–3.9GB | Steady 4.1GB |
| Time | 28.3s | 27.8s |

**Conclusion:** `empty_cache` was actively harmful. Without it, batch size converges to steady-state (35 for 11x11), VRAM stays flat, and throughput is slightly higher. The CUDA allocator's caching behavior is exactly what we want for batching.

**Status:** ✅ Removed entirely in this session.

---

### Fix 5: VRAM limits increased

**File:** `gpu_renderer.py:312, 346, 362`

| Parameter | Original | Current |
|---|---|---|
| `target_mem_fraction` | 0.50 | 0.70 |
| `reserve_margin` | 256MB | 128MB |
| Budget factor (`*0.85`) | *0.70 | *0.85 |

**Status:** ✅ Unchanged from prior session.

---

### Fix 6: Batch Size EMA Dampener — smooths transitions

**File:** `gpu_renderer.py:365-369, 317, 565`

**Bug:** Budget-based batch size could change abruptly from one iteration to the next (e.g., 159→8). With `empty_cache` this caused wild oscillation. Without `empty_cache`, the initial ramp-up from calibration (1 frame) to steady-state could overshoot.

**Fix:** Added an EMA (exponential moving average) dampener on batch size:
- `prev_batch_n` tracks the actual frames rendered in the previous batch
- After computing the budget-based `batch_size`, blend with `prev_batch_n` using `damp = 0.5`:
  `batch_size = prev_batch_n * 0.5 + budget_batch_size * 0.5`
- Prevents overshoot: 1→80→78→56→45→40→37→36→35→35→… (smooth ramp-down to steady state)

**Status:** ✅ Applied.

---

## Current State (11x11 GPU, 2026-05-11, no empty_cache)

| Metric | Value |
|---|---|
| Total time | 27.8s |
| Puzzle | 11x11, 1840 frames |
| Batches | 54 (1 calib + 53 real) |
| Steady-state batch size | 35 (locked for ~85% of render) |
| Peak VRAM | 4.1GB (steady, no oscillation) |
| Progress | 0% → 100% |

## Remaining / Untested

1. **CPU path not tested** — All changes target the GPU path. Run `python main.py --file test_bugs/11x11 --no-gpu` to verify.

2. **Larger puzzles (16x16, 20x20, 30x30, 50x50)** — Not yet tested. The 11x11 and 12x12 are verified. The steady-state batch size will be smaller for larger puzzles (due to per-frame VRAM cost scaling with canvas area).

3. **`target_mem_fraction=0.70`** — With `empty_cache` removed, the allocator keeps its cache and the budget algorithm converges to a stable batch size. 0.70 is fine for 6GB on 11x11/12x12; larger puzzles may need adjustment if VRAM hits ceiling.

4. **Fake time `.0` suffix** — Output timestamps may have `.0` suffix; not harmful.

## Relevant Files

- `replay_video.py` — tile colors (line 1399-1403), MD incremental (line ~1309), TerminalProgress (line ~1635)
- `main.py` — `_on_item_progress` progress multiplier (line ~761)
- `gpu_renderer.py` — memory params (lines 312, 346, 362), EMA dampener (lines 365-369, 317, 565)
- `test_replays_gpu/12x12` — 12x12 puzzle replay file
- `test_bugs/11x11` — 11x11 bug test replay used by `test_bug.py`

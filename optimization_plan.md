# SlidyReplay Optimization Plan

## Project Context
Replay renderer for sliding puzzle videos. Generates MP4 from solution strings/URLs.
GPU-accelerated rendering via PyTorch/CUDA with CPU-based overlay pre-render.

## Key Files
- `replay_video.py` — main rendering pipeline (generate_frames, generate_simple_replay)
- `gpu_renderer.py` — GPU batched tile rendering + overlay compositing
- `main.py` — CLI entry point
- `ai_brainstorm/` — investigation reports with full detail

## Log Observation (20x20 baseline)
- Overlay pre-render: **+9,971MB RAM**, 105.9s
- Stage 1 data prep: **28.5s** (no per-operation timing)
- GPU render: stable VRAM at 824MB reserved, 1942MB peak per batch
- Process killed at 5 min timeout (batch 155/295)
- `fake_times_range=[0.0, 391202.0]` — cosmetic float display

---

## Implementation Steps (execute in order)

### Step 1: C — Fake Times Cosmetic Fixes (trivial)

**1a: Fix log format** — `replay_video.py:1828`
```python
# Before:
log.info(f"  timing: delays_count={len(delays)}, fake_times_range=[{fake_times[0]:.1f}, {fake_times[-1]:.1f}]ms")
# After:
log.info(f"  timing: delays_count={len(delays)}, fake_times_range=[{fake_times[0]:g}, {fake_times[-1]:g}]ms")
```
`:g` strips trailing zeros: `391202` instead of `391202.0`.

**1b: Fix `[0.0]` prefix for type consistency** — `replay_video.py:1825`
```python
# Before:
fake_times = [0.0] + list(movetimes)
# After:
fake_times = [0] + list(movetimes)
```
When movetimes come from URL (Python `int[]`), this avoids creating a mixed-type list.

---

### Step 2: A-P0 — Drop PIL Images After Numpy Conversion (trivial)

`replay_video.py:1490-1495` — delete PIL objects immediately after numpy conversion:
```python
# After line 1495 (p["stats_arr"] = np.array(stats_img)):
p.pop("timer_img", None)
p.pop("stats_img", None)
```
GPU renderer only reads `timer_arr`/`stats_arr` (lines 496-506 in gpu_renderer.py).
CPU path at line 797 explicitly pops them if present — confirming they're unused.
**Saves ~6.5 GB RAM. Zero runtime cost.**

---

### Step 3: B-P3 — Add Per-Operation Timing Logs to Stage 1 (diagnostic)

Add timing markers inside `generate_frames` in `replay_video.py`:

**3a: Time the full frame_params loop** — replace line 1453:
```python
# Before line 1453:
log.info(f"  frame_params created: {len(frame_params)} entries, needed={len(states_needed)}")

# After:
_t_fp = time_module.time()
log.info(f"  frame_params loop: {_t_fp - _t_stage1:.3f}s total, {len(states_needed)} states built, {sol_len + 1} iterations")
```

**3b: Time the find_zero + move_matrix mutation loop** — after line 1451:
```python
# Inside the `if frame_idx < sol_len:` block, after `mc = move_matrix(...)`:
if frame_idx % 1000 == 0:
    log.info(f"  mutation for move {frame_idx / 1000}k: {time_module.time() - _t_stage1:.3f}s total so far")
```

**3c: Time the tile_colors hot loop** — wrap lines 1360-1367 with timing:
```python
# Before line 1360:
_t_tc_start = time_module.time()
# After line 1367 (end of tile_colors triple loop):
_t_tc = time_module.time()
log.info(f"    state {frame_idx}: tile_colors took {_t_tc - _t_tc_start:.3f}s")
```

**3d: Time Manhattan distance** — after line 1381:
```python
# After `cur_md = calculate_manhattan_distance(mc)`:
_t_md = time_module.time()
log.info(f"    state {frame_idx}: manhattan_distance took {_t_md - _t_tc:.3f}s (md={cur_md})")
```

**Caution on 3c/3d**: These fire per-state (11,177 lines). Use a sampling approach or enable only via log level check.

---

### Step 4: B-P0 — Cache Tile Colors Per Grid Stage (~12-16s savings)

**Observation**: `get_tile_colors(num, state, all_fringe_schemes, w)` depends ONLY on `(state, num, w)`, NOT on matrix position. Currently called 4.47M times (11,177 states × 400 tiles). Cache per grid stage (~7 stages) → ~2,800 calls.

**Implementation in `replay_video.py`**:

**4a: Build cache before frame_params loop** — insert after line 1351:
```python
# Build tile color cache per grid stage
# tile_colors depends only on (state, num), not on matrix position
_tile_color_cache = {}
for stage_move in sorted(list(set(filtered_stages))):
    if stage_move == 0:
        cache_state = grid_states[0]
    else:
        cache_state = get_grids_state(grid_states, stage_move - 1)
    if id(cache_state) not in _tile_color_cache:
        color_matrix = []
        for num in range(1, h * w + 1):
            main_bg, sec_bg = get_tile_colors(num, cache_state, all_fringe_schemes, w)
            color_matrix.append((main_bg or TILE_BG, sec_bg))
        _tile_color_cache[id(cache_state)] = color_matrix
```

**4b: Lookup tile colors from cache instead of recomputing** — replace lines 1360-1367:
```python
# In the frame_params loop, determine which grid stage this state belongs to
if frame_idx == 0:
    cache_key = id(grid_states[0])
else:
    cache_key = id(get_grids_state(grid_states, frame_idx - 1))
cached_colors = _tile_color_cache.get(cache_key)
if cached_colors is None:
    cached_colors = _tile_color_cache.get(id(grid_states[0]))

tile_colors = []
for row_idx in range(h):
    start = row_idx * w
    tile_colors.append(cached_colors[start:start + w])
```

**Note**: This approach avoids calling `get_tile_colors` inside the hot loop entirely while correctly handling grid stage transitions.

---

### Step 5: B-P1 — Incremental Manhattan Distance (~2-3s savings)

**Observation**: Each move swaps exactly 2 tiles (empty + one number). MD only changes for these 2 tiles. Recomputing from scratch (O(h×w)=400 cells) per state is wasteful.

**Implementation in `replay_video.py`**:

**5a: Add incremental MD update function** — near line 1300:
```python
def _update_manhattan_distance(md: int, matrix, move, zero_pos, w: int, h: int) -> int:
    """Update MD incrementally after a single swap move."""
    dr, dc = {'R': (0, -1), 'L': (0, 1), 'U': (1, 0), 'D': (-1, 0)}[move]
    nr, nc = zero_pos[0] + dr, zero_pos[1] + dc
    moved_val = matrix[nr][nc]
    old_md = abs((moved_val - 1) // w - nr) + abs((moved_val - 1) % w - nc)
    new_md = abs((moved_val - 1) // w - zero_pos[0]) + abs((moved_val - 1) % w - zero_pos[1])
    return md - old_md + new_md
```

**5b: Initialize current_md** — after line 1300:
```python
current_md = all_md
```

**5c: Replace MD recalculation** — line 1379. Change from:
```python
cur_md = calculate_manhattan_distance(mc)
```
To:
```python
cur_md = current_md
```

**5d: Update MD after each mutation** — after line 1451, inside `if frame_idx < sol_len:`:
```python
current_md = _update_manhattan_distance(current_md, mc, move, zp, w, h)
```

---

### Step 6: B-P2a — Track Zero Position Incrementally (~1-2s savings)

**Observation**: `find_zero(mc, w, h)` scans all 400 cells on each of 14,203 moves. After a swap, new zero position is deterministically known.

**Implementation in `replay_video.py`**:

**6a: Initialize zp before loop** — after line 1299:
```python
zp = find_zero(mc, w, h)
```

**6b: Replace lines 1449-1451** — change from:
```python
zp = find_zero(mc, w, h)
mc = move_matrix(mc, move, zp, w, h)
```
To:
```python
dr, dc = {'R': (0, -1), 'L': (0, 1), 'U': (1, 0), 'D': (-1, 0)}[move]
new_zp = (zp[0] + dr, zp[1] + dc)
mc = move_matrix(mc, move, zp, w, h)
zp = new_zp
```

---

### Step 6b: B-P2b — Cache Grid State Lookup

**Observation**: `get_grids_state(grid_states, frame_idx - 1)` at line 1357 re-filters all keys on every call.

**Implementation**: Pre-compute sorted valid keys after line 1243:
```python
_sorted_grid_keys = sorted([k for k in grid_states.keys() if isinstance(k, (int, float))])
_sorted_grid_keys.append(sol_len + 1)  # sentinel

def _fast_grid_state(grid_states, move_index):
    idx = bisect.bisect_left(_sorted_grid_keys, move_index + 1) - 1
    return grid_states[_sorted_grid_keys[idx]] if idx >= 0 else grid_states[0]
```

Then replace line 1357: `state = get_grids_state(...)` → `state = _fast_grid_state(...)`.

---

### Step 7: A-P1 — Per-Batch Overlay Rendering (~13 GB savings)

**The biggest refactor**. Move overlay rendering from Stage 2 into the GPU batch loop.

**7a: Remove Stage 2 overlay pre-render** — delete lines 1473-1501 from `replay_video.py`.

Keep only the initial overlay setup (lines 1468-1471) for `static_base` and `static_layout`.

Keep the state_to_count computation (lines 1503-1507).

Keep the ffmpeg pipe opening (lines 1509-1515).

**7b: Pass overlay render functions + static assets to GPU renderer**

In `replay_video.py` before calling `gpu.render_frames(...)`:
```python
extra_overlay_args = {
    "panel_w_val": panel_w_val,
    "static_base": static_base,
    "static_layout": static_layout,
}
```

Pass `extra_overlay_args` into `render_frames`.

**7c: Modify `render_frames` in `gpu_renderer.py`**

Add a new parameter `overlay_render_data=None` to `render_frames`.

Inside the per-frame loop (around line 491-506), replace the current overlay section:
```python
# ── Overlays (in‑place) ──
for i in range(batch_n):
    fi = batch_start + i
    fc = canvas[i]
    params = frame_params_list[fi]

    if overlay_render_data is not None:
        # Render overlays on-the-fly for this frame
        timer_img = _render_timer_text(params["timer_text"])
        stats_img = _apply_stats_dynamic(
            params["stats_data"],
            overlay_render_data["panel_w_val"],
            overlay_render_data["static_base"],
            overlay_render_data["static_layout"]
        )
        timer_arr = np.array(timer_img)
        stats_arr = np.array(stats_img)
    else:
        timer_arr = params.get("timer_arr")
        stats_arr = params.get("stats_arr")

    if timer_arr is not None:
        tt = torch.from_numpy(timer_arr).to(dev, non_blocking=True).float() / 255.0
        dx = max(tx1, tx1 + ((tx2 - tx1) - tt.shape[1]) // 2)
        dy = max(ty1, ty1 + ((ty2 - ty1) - tt.shape[0]) // 2)
        self._blend_rgba_inplace(fc, tt, dx, dy)

    if stats_arr is not None:
        stt = torch.from_numpy(stats_arr).to(dev, non_blocking=True).float() / 255.0
        self._blend_rgba_inplace(fc, stt, px, py)
```

**7d: Import `_render_timer_text` and `_apply_stats_dynamic`**

In `gpu_renderer.py`, either:
- Move `_render_timer_text` (currently at line 105) and `_apply_stats_dynamic` into gpu_renderer.py, OR
- Import them from replay_video.py (beware circular imports!)

Best approach: `_render_timer_text` is already in gpu_renderer.py. Import `_apply_stats_dynamic` from replay_video.py at the top of gpu_renderer.py.

Also import `np` (numpy) in gpu_renderer.py if not already.

---

### Step 8: Verify

Run after all implementations:
```powershell
python main.py --file test_replays_gpu/20x20 --log
```

Compare vs baseline (debug_20260511_072548.log):
- Overlay RAM: should drop from +9971MB → ~0MB (per-batch)
- Stage 1 time: should drop from 28.5s → ~8-12s
- VRAM: should remain stable (~824MB reserved, ~1942MB peak)
- Fake times log: `391202` instead of `391202.0`
- Overall render time: should be significantly reduced

---

## TODO List (checkpoint state)

```
[ ] Step 1a — C: Change log format `:.1f` → `:g` (replay_video.py:1828)
[ ] Step 1b — C: Change `[0.0]` → `[0]` (replay_video.py:1825)
[ ] Step 2 — A-P0: Drop PIL after numpy (replay_video.py:1494-1495)
[ ] Step 3 — B-P3: Add per-operation timing logs (replay_video.py:1250-1453)
[ ] Step 4 — B-P0: Cache tile colors per grid stage (replay_video.py:1351-1367)
[ ] Step 5 — B-P1: Incremental Manhattan distance (replay_video.py:1300-1381)
[ ] Step 6 — B-P2: Incremental zero + cache grid state (replay_video.py:1357-1451)
[ ] Step 7 — A-P1: Per-batch overlay rendering (replay_video.py + gpu_renderer.py)
[ ] Step 8 — Verify: Run 20x20, compare RAM + speed vs baseline
```

## Detailed Reports (for deeper reference)
- `ai_brainstorm/report_overlay_ram.md` — full overlay RAM analysis + 5 suggestions
- `ai_brainstorm/report_data_prep.md` — full Stage 1 time breakdown + 5 logging points
- `ai_brainstorm/report_fake_times.md` — full fake_times data flow + 3 fix options

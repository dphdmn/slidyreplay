# Tile Size Fix Plan — Resolution-Based Rendering

## Goal
Replace the broken `quality` system (which always doubled tile sizes via `quality = quality + 1.0`) with resolution presets (360p/480p/720p/1080p/1440p/2160p). The user picks a **maximum video height**, and the system derives tile sizes, font sizes, padding, and stats panel dimensions from it.

## Core Design

### Constants (geometry.py)
```
MIN_TILE = 10          # absolute floor — 100×100+ puzzles
MAX_TILE = 60          # absolute ceiling — tiny puzzles (capped)
REFERENCE_HEIGHT = 1080 # 1080p is the design reference
```

### Layout Computation (new function in geometry.py)
```
compute_layout(height_target, puzzle_w, puzzle_h, grid_only)
  → { pad, header_h, panel_w, tile_size, font_size }

Scale = height_target / 1080.0
pad         = max(2,  round(20 * scale))
header_h    = max(12, round(56 * scale))
panel_w     = max(80, round(320 * scale))
avail_h_for_puzzle = height_target - (pad*2 if grid_only else header_h + 3*pad)
tile_size   = clamp(avail_h_for_puzzle // max(puzzle_w, puzzle_h), MIN_TILE, MAX_TILE)
font_size   = compute_font_size(puzzle_w, puzzle_h, tile_size)
```

### Resulting Heights at 1080p (full layout, avail_h=972)
```
 5×5:  tile=60 (cap)   puzzle=300   canvas≈570
10×10: tile=60 (cap)   puzzle=600   canvas≈870
16×16: tile=60 (cap)   puzzle=960   canvas≈1230  (stats may be shorter)
20×20: tile=48         puzzle=960   canvas≈1230
30×30: tile=32         puzzle=960   canvas≈1230
50×50: tile=19         puzzle=950   canvas≈1220  (down from old 1660!)
64×64: tile=15         puzzle=960   canvas≈1230
100×100: tile=10 (floor) puzzle=1000 canvas≈1270
144×144: tile=10 (floor) puzzle=1440 canvas≈1710
```

### Different Presets for 50×50
```
360p:  tile=10 (floor)  puzzle=500   canvas≈770
480p:  tile=10 (floor)  puzzle=500   canvas≈770
720p:  tile=13          puzzle=650   canvas≈920
1080p: tile=19          puzzle=950   canvas≈1220   ← DEFAULT
1440p: tile=26          puzzle=1300  canvas≈1570
2160p: tile=39          puzzle=1950  canvas≈2220
```

## Files to Change

### 1. geometry.py
- Remove `BASE_SIZE = 15`
- Add `MIN_TILE = 10`, `MAX_TILE = 60`, `REFERENCE_HEIGHT = 1080`
- Remove old `pick_tile_size()` (replay_video.py)
- Add `compute_layout()` function
- Modify `compute_canvas_dimensions()` to accept optional `pad`, `header_h`, `panel_w` overrides
- Modify `compute_grid_position()` to accept optional `pad`, `header_h` overrides

### 2. replay_video.py
- **Signature**: replace `quality: float = 1.0` with `height_target: int = 1080`
- **Remove**: `quality = quality + 1.0` hack (line 1240)
- **Remove**: `pick_tile_size()` + `compute_tile_size()` calls (lines 1303-1304)
- **Remove**: `raw_tile` variable (no longer separate from tile_size)
- **Add**: `layout = compute_layout(height_target, w, h, opts.grid_only)` at data prep start
- **Thread**: pass `pad`, `header_h`, `panel_w` through all GPU/CPU paths
- **Stats layout**: `_stats_layout_info(panel_w, height_target)` — scales font sizes (24→round(24×H/1080), 20→round(20×H/1080), 18→round(18×H/1080), 13→round(13×H/1080), 16→round(16×H/1080)), scales pixel offsets (px=round(10×scale), inner_w=panel_w-round(20×scale))
- **Grid stages centering fix**: In `_make_stats_static_base`, instead of `add(px, y, line, ...)`, compute `text_w = gs_lf.getbbox(line)[2] - gs_lf.getbbox(line)[0]` and center: `add(max(px, (panel_w - text_w) // 2), y, line, ...)`. But preserve the existing columnar layout alignment — use a centered offset for the whole block.
- **batch_render function**: Replace `eff_q = quality + 1.0` with `layout = compute_layout(height_target, ...)`; remove inline `eff_ts = max(raw_ts, int(raw_ts * eff_q))`

### 3. gpu_renderer.py
- Constructor: replace `quality: float = 1.0` with `pad=None, header_h=None, panel_w=None`
- Remove `ts = compute_tile_size(tile_size, quality)` — tile_size already computed
- Store self.pad, self.header_h, self.panel_w
- Use them in `compute_canvas_dimensions()`, `compute_grid_position()`, timer_bbox
- Note: `tile_size` parameter is now the FINAL tile size (not raw)

### 4. main.py
- **GUI**: Replace `DoubleQualityVar` checkbox with a height preset dropdown/combobox
  - Options: "360p", "480p", "720p", "1080p (default)", "1440p (2K)", "2160p (4K)"
  - Store as `self.height_target_var = tk.IntVar(value=1080)`
  - Add warning label for 2K/4K on large puzzles
- **CLI**: Replace `--quality N` with `--height N` (accepts 360-2160)
  - Also accept `--preset 720p` shorthand
- Pass `height_target` instead of `quality` to `generate_simple_replay`

### 5. test_tilesize.py
- Update to use new height_target parameter instead of quality

## Edge Cases & Warnings

### Edge Case 1: Non-square puzzles (2×100 or 100×2)
- Largest side (100) determines tile size → tile = max(10, 972//100) = 10
- 2×100: puzzle=1000×20, canvas_h≈770 (stats-driven, puzzle too short)
- 100×2: puzzle=20×1000, canvas_h≈770
- Acceptable — these are weird puzzles, canvas can't always reach target

### Edge Case 2: Tiny puzzles (3×3, 4×4)
- tile caps at 60, puzzle_h=180-240, canvas≈400-500
- Well below target height — that's fine, no need to waste pixels

### Edge Case 3: Stats panel height > puzzle height
- Canvas height = max(puzzle_h + header_h + 3×pad, stats_h + header_h + 2×pad)
- If stats panel is taller, canvas exceeds target — this is expected

### Edge Case 4: Grid stages text overflowing panel width
- **IMPORTANT**: Grid stages format: `"  0:00.000 |  0:01.000 (100/50) | s1"`
- Line lengths vary based on cumulative time values (longer times = wider strings)
- Currently fine-tuned to barely fit inside 320px panel_w at fixed font 13px mono
- When panel_w scales (smaller at low presets), the grid stages may overflow
- **Fix**: Either truncate long values, or reduce font size further when lines don't fit

### Edge Case 5: 2K/4K on large puzzles → VRAM warnings
- 2K on ≥50×50: show warning
- 4K on ≥30×30: show strong warning
- 4K on ≥50×50: warn about probable OOM

### Edge Case 6: Grid stages centering
- **Current**: Grid stages block is always left-aligned at `px=10`
- **Fix**: The entire grid stages block should be centered within the panel
- After rendering all lines, compute max line width, offset by `(panel_w - max_line_w) // 2`
- Still respect minimum `px` inset for visual padding

## Stats Panel — CPU and GPU Both Must Work

Both `_render_stats_full` (CPU path) and `_make_stats_static_base`/`_render_stats_dynamic` (GPU path) use the same `_stats_layout_info()` function. Since both will be updated to accept `height_target`, both paths automatically stay in sync.

### CPU path
- `render_frame()` → `_render_stats_full()` → `_stats_layout_info(panel_w, height_target)`
- All font sizes and offsets scale with height_target

### GPU path
- `_make_stats_static_base(panel_w, stats_data, ..., height_target=height_target)` 
  → `_stats_layout_info(panel_w, height_target)`
- `_render_stats_dynamic(...)` → `_stats_layout_info(panel_w, height_target)`
- GPU renders stats into overlay textures using the same scaled layout

## Implementation Order
1. Add constants + `compute_layout()` to `geometry.py`; modify `compute_canvas_dimensions()`, `compute_grid_position()`
2. Update `replay_video.py`: change signature, remove quality hack, thread layout params, update stats layout
3. Update `gpu_renderer.py`: accept layout params, remove quality from constructor
4. Update `main.py`: replace quality slider/checkbox with preset dropdown
5. Fix grid stages centering in `_make_stats_static_base`
6. Update `test_tilesize.py`
7. Run `python test_tilesize.py` to verify tile sizes
8. Run `python main.py --file test_replays/50x50 --log` to verify no errors

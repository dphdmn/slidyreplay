# Puzzle Preview & Color Scheme Customization

## Overview

Two features:

1. **Color scheme customization** — adjustable hue range (start/end), saturation, and brightness for generated main color schemes (fringe, rows, columns)
2. **Live GUI preview** — renders a solved puzzle with current settings so the user sees how changes look immediately

---

## Phase 1: Color Parameter Plumbing

### 1a. Add fields to `RenderOptions` (`geometry.py:326-341`)

Add to the `RenderOptions` dataclass:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `hue_start` | `float` | `0.0` | Start of hue range (0-360) |
| `hue_end` | `float` | `360.0` | End of hue range (0-360) |
| `saturation` | `float` | `0.78` | Color saturation (0-1) |
| `brightness` | `float` | `0.6` | Color brightness/lightness (0-1) |

### 1b. Modify `get_colors()` (`replay_video.py:84-92`)

Replace hardcoded values with parameters:

```python
def get_colors(num_colors, saturation=0.78, lightness=0.6, hue_start=0, hue_end=360):
    if num_colors < 1:
        return []
    colors = []
    for i in range(num_colors):
        hue = hue_start + (i / max(num_colors, 1)) * (hue_end - hue_start)
        colors.append(hsl_to_rgb(hue / 360.0, saturation, lightness))
    return colors
```

### 1c. Modify `get_fringe_colors_nxm()` (`replay_video.py:167-201`)

Add params: `saturation=0.78, lightness=0.6, hue_start=0, hue_end=360`

Forward them to `get_colors()` at lines 169, 172, 181, 189.

### 1d. Modify `get_all_fringe_schemes()` (`replay_video.py:208-225`)

Add params: `saturation=0.78, lightness=0.6, hue_start=0, hue_end=360`

Forward them to `get_fringe_colors_nxm()` at line 223.

### 1e. Modify `generate_simple_replay()` (`replay_video.py:2294-2320`)

Extract color params from `opts` and forward to `get_all_fringe_schemes()`:

```python
all_fringe_schemes = get_all_fringe_schemes(
    grid_states, main_scheme,
    saturation=opts.saturation if opts else 0.78,
    lightness=opts.brightness if opts else 0.6,
    hue_start=opts.hue_start if opts else 0,
    hue_end=opts.hue_end if opts else 360,
)
```

### Verification

- Run: `python -c "from replay_video import *; c = get_colors(5, saturation=1.0, lightness=0.5, hue_start=0, hue_end=180); print(c)"`
- Existing behavior preserved when all params at defaults (0.78, 0.6, 0, 360)

---

## Phase 2: Preview Rendering Function

### 2a. Write `_render_preview()` helper in `main.py`

This function will be a method of `ReplayGUI`. It:

1. Determines puzzle size:
   - Read `self.size_var.get()`, parse as `WxH`
   - If empty → default to 4×4
   - If W*H > 400 (>20×20) → show 4×4 instead, with size info label noting the override

2. Creates solved matrix (`grid_only` mode doesn't need grid analysis):
   ```python
   matrix = [[r * w + c + 1 for c in range(w)] for r in range(h)]
   matrix[h-1][w-1] = 0
   ```

3. Builds a single grid state (same as `force_main=True` path):
   ```python
   grid_data = {
       "enableGridsStatus": -1,
       "width": w, "height": h, "offsetW": 0, "offsetH": 0
   }
   grid_states = generate_grids_stats(grid_data)
   ```

4. Computes fringe schemes with current GUI color params:
   ```python
   sat = self.saturation_var.get() / 100.0
   light = self.brightness_var.get() / 100.0
   hue_start = self.hue_start_var.get()
   hue_end = self.hue_end_var.get()
   main_scheme = self.main_scheme_var.get()
   
   all_fringe_schemes = get_all_fringe_schemes(
       grid_states, main_scheme, sat, light, hue_start, hue_end
   )
   ```

5. Builds `RenderOptions` from current GUI state (grid_only, no_border, no_numbers, no_grid_bars, no_secondary_border, tile_bg_color, grid1_color, grid2_color)

6. Renders at small tile_size:
   ```python
   tile_size = max(16, min(48, 360 // max(w, h)))
   font_size = max(8, tile_size // 3)
   
   first_state = list(grid_states.values())[0]
   stats_data = {
       "moves": [], "current_time": 0,
       "total_moves": 0, "total_time_ms": 0, "total_tps": 0,
       "is_movetimes_accurate": False,
       "score_title": "", "timer_text": "",
   }
   
   img = render_frame(
       matrix=matrix, grid_state=first_state,
       all_fringe_schemes=all_fringe_schemes,
       tile_size=tile_size, font_size=font_size,
       stats_data=stats_data,
       score_title_text="", timer_text="",
       is_movetimes_accurate=False,
       total_moves=0, total_time_ms=0, total_tps=0,
       opts=opts,
   )
   ```

7. Resizes to fit preview area:
   ```python
   PREVIEW_SIZE = 280  # px, fits the middle column nicely
   img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.LANCZOS)
   ```

8. Displays in GUI (keeps reference to prevent GC):
   ```python
   self._preview_photo = ImageTk.PhotoImage(img)
   self._preview_label.config(image=self._preview_photo)
   ```

### 2b. Debounce scheduling

```python
def _schedule_preview(self):
    if self._preview_job:
        self.after_cancel(self._preview_job)
    self._preview_job = self.after(300, self._render_preview)
```

### 2c. Trigger wiring

Trace `write` on all relevant vars to call `_schedule_preview`:
- `self.hue_start_var`, `self.hue_end_var`
- `self.saturation_var`, `self.brightness_var`
- `self.main_scheme_var`
- `self._color_vars["grid1"]`, `self._color_vars["grid2"]`, `self._color_vars["tile_bg"]`
- `self.no_border_var`, `self.no_numbers_var`, `self.no_grid_bars_var`, `self.no_secondary_border_var`

### Verification

- Call `self._render_preview()` manually from the Python console
- Check that the preview image updates correctly
- Check that slider changes trigger debounced updates

---

## Phase 3: GUI Layout Changes

### 3a. Preview frame in Column 1 (middle, top)

Insert before the "File:" label in the `mid` frame packing:

```python
# Preview frame (packed first = appears at top)
preview_frame = tb.LabelFrame(mid, text="Preview")
preview_frame.pack(fill="x", pady=(0, 4))
# Inner frame to center the image
preview_inner = tb.Frame(preview_frame)
preview_inner.pack(pady=4)
self._preview_label = tb.Label(preview_inner)
self._preview_label.pack()

# Size info below preview
self._preview_info = tb.Label(preview_frame, text="4×4 · Fringe",
                               font=(FONT_FAMILY, 8), foreground="#888")
self._preview_info.pack(anchor="w", padx=4, pady=(0, 2))
```

Preview area size: 280×280 px (fits within the 400px minsize column, with room for padding).

### 3b. Color scheme sliders in COLORS section

After the tile_bg color row (after line 543), add 4 rows:

```
─── COLOR SCHEME ───── (section header label + separator)

Row 1: Hue start: [===●========] 0       (tb.Scale 0-360 + value label)
Row 2: Hue end:   [========●===] 360     (tb.Scale 0-360 + value label)
Row 3: Saturation: [=====●=====] 78%     (tb.Scale 0-100 + "%" label)
Row 4: Brightness: [====●======] 60%     (tb.Scale 0-100 + "%" label)
```

Each row follows the same pattern as the existing FPS/Compression sliders:
- `tb.Frame` with grid layout
- `tb.Label` for description
- `tb.Scale` with appropriate range
- `tb.Label` for current value display
- `trace_add("write", ...)` to update value label + schedule preview

### Verification

- GUI shows preview frame at top of middle column
- 4 sliders appear in COLORS section
- Changing sliders updates the preview image after 300ms debounce

---

## Phase 4: Integration Testing

### Scenarios to test

1. **Default state** — no input, 4×4 solved puzzle shown in fringe pattern with default colors
2. **Hue range change** — set hue_start=0, hue_end=120 → colors limited to red-yellow-green range
3. **Saturation change** — set to 0% → grayscale puzzle
4. **Brightness change** — set to 100% → very bright colors
5. **Main scheme change** — switch to "rows" → row stripes in preview, switch to "columns" → column stripes
6. **Grid colors** — change grid1/grid2 hex → preview updates (solved state won't show grids, but the rendering pipeline uses these colors for fringe regions in some edge cases)
7. **No numbers** — numbers disappear from preview tiles
8. **No border** — tile borders disappear
9. **Size > 20×20** — preview shows 4×4 fallback with info label
10. **Custom size** — e.g., "5x5" in size field → preview shows 5×5 solved
11. **Full render** — generate a video with custom hue/sat/brightness → verify the video matches preview

### Known limitations

- Solved state only — no grid regions visible (no red/blue cells). The preview shows the base color scheme, not grid detection effects.
- Preview uses `grid_only` — no layout chrome (timer, stats panel). This focuses on the tile colors.
- Grid colors (grid1/grid2) don't appear in preview since there are no grid regions in a solved puzzle. Their preview relevance is limited to `tile_bg_color`.
- For very large puzzles (up to 20×20), rendering may take a moment on slow machines — debouncing prevents rapid re-renders.

---

## Files Modified Summary

| File | What |
|------|------|
| `geometry.py` | +4 fields to `RenderOptions` dataclass |
| `replay_video.py` | `get_colors()` — accept color params; `get_fringe_colors_nxm()` — forward color params; `get_all_fringe_schemes()` — forward color params; `generate_simple_replay()` — extract from opts |
| `main.py` | Preview frame + label + info in column 1; 4 slider rows in COLORS section; `_schedule_preview()`, `_render_preview()` methods; trace wiring; opts construction with new fields |
| `README.md` | Document `--hue-start`, `--hue-end`, `--saturation`, `--brightness` flags |

## Order of Implementation

1. Phase 1: Color parameters (geometry.py + replay_video.py) — no GUI changes yet
2. Phase 2: Preview rendering function (main.py) — add the method but don't wire GUI yet
3. Phase 3: GUI layout — preview frame + sliders + tracing
4. Phase 4: CLI flags + README update
5. Phase 5: Testing all scenarios

Each phase builds on the previous one and is testable independently.

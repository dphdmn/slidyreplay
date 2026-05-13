# Layout Path Optimizations

## Test Commands

```bash
# Baseline with layout (stats panel + timer):
python main.py --file test_replays/10x10 --log
python main.py --file test_replays/12x12 --log

# Reference without layout (grid only):
python main.py --file test_replays/10x10 --no-layout --log
python main.py --file test_replays/12x12 --no-layout --log
```

## Baseline (GTX 1660 Super, 6GB)

| Config | 10x10 (1362 unique) | 12x12 (2470 unique) |
|--------|---------------------|---------------------|
| No layout | 6.3s (216 f/s) | 14.7s (168 f/s) |
| With layout | 15.5s (88 f/s) | 34.7s (71 f/s) |
| Slowdown | 2.5x | 2.4x |

## Root Cause Summary

| Metric | 12x12 layout | 12x12 no layout | Ratio |
|--------|-------------|----------------|-------|
| Canvas | 1076×812 | 736×736 | **1.6x pixels** |
| Batch size (steady) | 79-83 | 132-133 | **0.6x** |
| Per-batch time | 1.5s | 0.7s | **2.1x** |
| Per-frame time | 18.8ms | 5.4ms | **3.5x** |
| actual_ppf (memory) | 28-34MB | 19MB | **1.5-1.8x** |
| ema | 33MB | 21MB | **1.6x** |

Three compounding factors:
1. Larger canvas (1.6x pixels) → more GPU work per frame.
2. Higher per-frame memory (33MB vs 21MB) → batch sizes capped at ~80 vs ~133.
3. More batches → more fixed overhead per batch (sync, cache flush, kernel launches).

---

## Independent Sub-Ideas (can be combined)

### Sub-idea X: Cache static_base on GPU, only upload tiny dynamic text
Upload the stats panel background once. Per frame, only compute 4 tiny text images (~30×10px each) via PIL and composite onto the cached base on GPU.

### Sub-idea Y: Inline overlay compute, no background thread
Compute overlays for the next batch during GPU tile rendering of the current batch. Removes thread+queue complexity.

### Sub-idea Z: Full GPU text rendering (font atlas)
Zero PIL. Pre-render a bitmap font atlas on GPU. Render all text with GPU kernels. Hardest but theoretically best.

### Sub-idea W: Shrink stats panel
Reduce panel width or make it configurable. Direct pixel savings.

### Sub-idea V: Faster EMA for batch sizing
Tune initial EMA estimate so the first batch doesn't overfill VRAM. Low effort, saves ~2-3s.

---

## Implementation Plan: Variants to Test

Implement in this order. Each is a git branch off the same baseline.

### Variant A — Inline overlays (sub-idea Y only)

**Goal**: Remove background thread. Compute overlays inline between batches.

**Files to change**: `gpu_renderer.py`

**Changes**:
1. Delete `_overlay_queue`, `_overlay_queue = None`, `Queue` import, `threading` import.
2. Before the batch loop: compute overlays for the first batch (PIL inline).
3. In the loop after tile rendering: apply current batch's overlays. Then compute next batch's overlays inline.
4. Keep the existing per-frame overlay PIL code (timer_img/stats_img from `overlay_render_data`).

**Test**:
```bash
python main.py --file test_replays/10x10 --log
python main.py --file test_replays/12x12 --log
```

**Expected**: Same or slightly worse than baseline (thread overhead was near zero). This is a simplification step.

---

### Variant B — Tiny text + cached static_base (sub-idea X only)

**Goal**: Instead of computing the full stats RGBA per frame (840KB), compute only 4 tiny text images (~1.2KB total) and composite onto a cached GPU static_base.

**Files to change**: `gpu_renderer.py`, `replay_video.py`, `geometry.py`

**Changes**:

`geometry.py`:
- Add `render_dynamic_text(text, font, pos) -> np.ndarray` — renders a single text string to a tight RGBA numpy array.

`replay_video.py`:
- Background thread computes 4 small text arrays + timer_arr per frame, NOT the full stats_arr.
- Pass `static_base` as a PIL Image (already done for `overlay_render_data`).

`gpu_renderer.py`:
- Add `self._stats_static_base: torch.Tensor` — uploaded once in `render_frames` (or `__init__`), converted from `overlay_render_data["static_base"]`.
- Overlay path per batch:
  ```python
  # For each frame in batch:
  timer_arr, text_arrays = queue.get()  # text_arrays = list of (arr, x, y) for 4 dynamic values
  tt = torch.from_numpy(timer_arr).to(dev).float() / 255.0
  self._blend_rgba_inplace(canvas[_i], tt, tx, ty)

  # Composite dynamic text onto cached static_base
  stats = self._stats_static_base.clone()  # clone once per frame (or once per batch then scatter)
  for text_arr, tx, ty in text_arrays:
      tt = torch.from_numpy(text_arr).to(dev).float() / 255.0
      self._blend_rgba_inplace(stats, tt, tx, ty)
  self._blend_rgba_inplace(canvas[_i], stats, px, py)
  ```

**Test**:
```bash
python main.py --file test_replays/10x10 --log
python main.py --file test_replays/12x12 --log
```

**Expected**: Per-frame memory drops from 33MB to ~22MB. Batch sizes roughly double (80→160). Total time: 34.7s → ~18-20s.

---

### Variant C — A + B combined

**Goal**: Inline overlay compute (no thread) + tiny text + cached static_base.

**Changes**: Merge Variant A (inline compute, no queue) with Variant B (tiny text, cached base).

**Test**:
```bash
python main.py --file test_replays/10x10 --log
python main.py --file test_replays/12x12 --log
```

**Expected**: Best practical result without GPU font atlas. ~16-18s for 12x12.

---

### Variant D — Full GPU font atlas (sub-idea Z only, optional)

**Goal**: Zero PIL. Render all overlay text on GPU.

**Changes**:

`geometry.py`:
- Add `build_font_atlas(font, chars, size) -> (atlas_im: Image, char_map: dict[str, (u,v,w,h)])`.

`gpu_renderer.py`:
- Upload font atlas as a 2D RGBA texture.
- For each text string: use `torch.nn.functional.grid_sample` to sample glyph quads from the atlas and copy to target positions.
- Composite rendered text onto `_stats_static_base` → blend onto canvas.

`replay_video.py`:
- Remove all PIL overlay pre-computation. No background thread.
- Pass font atlas + char map to GPU renderer.

**Test**:
```bash
python main.py --file test_replays/10x10 --log
python main.py --file test_replays/12x12 --log
```

**Expected**: Should match or beat Variant C. Adding GPU font atlas after Variant C gives the final ~15-17s result.

---

## Results Table

Fill this after all variants are tested:

| Variant | 10x10 total | 10x10 f/s | 12x12 total | 12x12 f/s | VS baseline |
|---------|-------------|-----------|-------------|-----------|-------------|
| Baseline (current) | 15.5s | 88 | 34.7s | 71 | — |
| A: Inline overlays | | | | | |
| B: Tiny text + cached base | | | | | |
| C: A + B combined | | | | | |
| D: Full GPU font atlas | | | | | |
| No layout (reference) | 6.3s | 216 | 14.7s | 168 | 2.4x |

## Rollback

Each variant on its own git branch:
```bash
git checkout -b variant-a-inline-overlays
# ... hack hack ...
git commit -m "variant a"
git checkout baseline-before-variants

git checkout -b variant-b-tiny-text
# ...
```

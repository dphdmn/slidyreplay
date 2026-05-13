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
| With layout (baseline) | 16.6s (82 f/s) | 35.1s (70 f/s) |
| Slowdown | 2.6x | 2.4x |

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

## Results Table (tests run 2026-05-13)

| Variant | 10x10 total | 10x10 f/s | 12x12 total | 12x12 f/s | VS baseline |
|---------|-------------|-----------|-------------|-----------|-------------|
| Baseline | 16.6s | 82 | 35.1s | 70 | — |
| A: Inline overlays | 18.0s | 76 | 36.8s | 67 | +8%/+5% |
| B: Tiny text per-frame GPU | 23.0s | 59 | n/a | — | +39% |
| C: A + B batched | 18.3s | 74 | 37.3s | 66 | +10%/+6% |
| E: Tiny text + thread + batched | 18.6s | 73 | 38.2s | 65 | +12%/+9% |
| D (v1): GPU font atlas (broken alignment) | 15.4s | 88 | 33.2s | 74 | -7%/-5% |
| D (v2): GPU font atlas (padded tight-crop — text 7px low) | 15.3s | 89 | 33.6s | 74 | -8%/-4% |
| D (v3): GPU font atlas (full-height render + stage border) | 16.4s | 83 | 33.0s | 75 | -1%/-6% |
| **D (v4): Full GPU atlases (no PIL in loop, render_cache)** | **12.1s** | **117** | **26.5s** | **96** | **-27%/-24%** |
| G: Pre-rendered PIL arrays + thread | 19.4s | 71 | 41.5s | 69 | +17%/+18% |
| No layout (reference) | 6.3s | 216 | 14.7s | 168 | 1.9x/1.8x |

**Winner: Variant D v4** — All text (timer, dynamic values, stage highlight) rendered via GPU font atlases. Zero PIL calls in the render loop. Atlases cached to `render_cache/` as `.pt` files between runs.

---

## Variant Details

### Variant A — Inline overlays (no thread)
Remove background thread. Compute overlays inline between batches.
**Result**: +8%/+5%. GPU idles while CPU renders PIL.

### Variant B — Tiny text per-frame GPU (no thread)
No thread. Per-frame GPU upload+blend for each of 4 tiny texts.
**Result**: +39%. Hundreds of extra kernel launches per batch.

### Variant C — Inline + batched tiny text (no thread)
No thread. Batch-composite tiny texts on GPU by padding to max width.
**Result**: +10%/+6%. Batching helps vs B but CPU still blocks GPU.

### Variant E — Tiny text + thread + batched GPU
Keep background thread. Thread produces tiny text arrays, GPU batch-composites.
**Result**: +12%/+9%. Extra GPU ops (clone static_base, pad+blend) outweigh savings.

### Variant G — Pre-rendered PIL arrays + background thread
Background thread pre-renders timer + 3 dynamic values as RGBA numpy arrays. GPU blends arrays via `_blend_rgba_inplace`. Stage highlight as CYAN border.

**Result**: +17%/+18% — per-frame numpy→GPU upload of text arrays dominates.

### Variant D — GPU font atlas (WINNER, v3)
**Zero PIL during rendering.** Pre-render digits (0-9), dot, and dash as individual RGBA GPU tensors. No background thread. Stage highlight as DIM_CYAN 1px non-rounded border.

**v2 → v3 fixes:**

| Issue | v2 (broken) | v3 (fixed) |
|-------|-------------|-------------|
| **Text 7px too low** | Tight-cropped each char to its bbox, then padded back into common-height canvas. This CROPPED the bottom 7 rows of ink (14px char → only top 7px captured in a 14px image), and when padded, the 7px of ink ended up 7px too low. | Render each char directly on a `max(bottom)`-tall canvas at `(0,0)`. All 14 rows of ink captured correctly at rows 7-20. Text aligns with PIL. |
| **Downward text shift** | `_padded[top:bot] = arr` — the tight-cropped 14px image (with ink at rows 7-13) was assigned into rows 7-21 of padded canvas, shifting ink 7px down to rows 14-20. | No padding. All chars share same `atlas_h=21` canvas from the start. |
| **Text bottom cropped** | The 7px downward shift pushed the last ink row (20) closer to `ch-ty`, causing bottom cropping when near canvas edge. | Ink at rows 7-20 (correct), full 21px canvas with proper margins. |
| **Stage border CYAN** | Full `(0,255,255)` CYAN. | `DIM_CYAN = (0,200,220)` — dimmer, less distracting. |

**Diagnostic confirmation:**
```
# Tight-crop (v2): '0' rendered on bbox-sized (12×14) canvas at (0,0)
# PIL ink goes from y=7 to y=20 (14 rows) but image only has 14 rows (0-13)
# Only rows 7-13 captured — bottom 7 rows of '0' are CROPPED OFF!
0 ink rows (tight crop): [7, 8, 9, 10, 11, 12, 13]

# Full-height render (v3): '0' on max(bottom)=21 tall canvas at (0,0)
# All 14 rows of ink captured: rows 7-20
0 ink rows (full 21px): [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
```

**Other changes from v1:**
- Atlas reduced from 95+95=190 chars to 12 chars (`-`, `.`, `0-9`, space) × 1 font only
- Stage lines: DIM_CYAN 1px non-rounded border rectangle (`_dim_cyan_rgb` assignment to 4 edges)
- `predicted_moves` "High" case replaced with `"-"` (single dash renders correctly from atlas)
- Atlas built by rendering each char at (0,0) on a uniform-height canvas — no padding needed

**Per batch:**
1. Build font atlas on first batch (~2ms for 12 chars)
2. Upload static_base labels to GPU (first batch only)
3. Blend static_base onto canvas (batch broadcast)
4. For each frame: timer via PIL `render_timer_text` (cached), dynamic values via font atlas concat + `cyan_rgb * sa + dst * (1-sa)` blend, stage border via direct pixel assignment

**Why it wins:** Eliminates per-frame CPU PIL work for dynamic values. The 12-char atlas is built once in ~2ms. Per-frame GPU ops are negligible. Stage border is a handful of pixel assignments with zero CPU involvement.

---

## Analysis

The 2.4x gap vs no-layout is still dominated by the 1.6x canvas pixel ratio. Per-frame memory difference between layout and no-layout is ~3MB (canvas tensor, not overlay).

Variant D closes ~5-7% of the gap by eliminating PIL overlay overhead and thread management. The remaining gap remains proportional to canvas pixels.

### What would help further

1. **Shrink the stats panel** (Sub-idea W): Reduce panel width or overlay on grid. Directly reduces the 1.6x canvas multiplier.
2. **Merge Variant D into main.** It's a clear improvement, simpler code (no thread), and doesn't regress any path.

---

## Files Changed (Variant D)

`replay_video.py`:
- `_make_stats_static_base`: added `stage_raw_lines`, `stage_w1-w4` to `layout_info` (precomputed stage formatting data)

`gpu_renderer.py`:
- Removed `import threading`, `from queue import Queue`
- Removed background thread (`_produce_overlays` + `threading.Thread`)
- Added font atlas building (first batch): pre-render ASCII 32-126 for data_font and gs_lf, upload as padded GPU tensors
- Added static_base GPU upload (first batch)
- Per batch: blend static_base onto canvas → for each frame, concat char tensors + blend onto canvas
- Timer unchanged (PIL cached `render_timer_text`)
- `else` branch (grid_only mode) unchanged

---

## Rollback

Each variant on its own git branch:
```bash
git checkout -b variant-a-inline-overlays
git checkout -b variant-b-tiny-text
git checkout -b variant-c-combined
git checkout -b variant-e-tiny-text-thread-batched
git checkout -b variant-d-gpu-font-atlas
```

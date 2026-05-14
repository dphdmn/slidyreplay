# Renderer Optimization

## Goal
Drive total render time down for both CPU and GPU paths. CPU is already competitive (~13s vs GPU ~12s on 10×10). Now optimize further and bring the same improvements to GPU.

## Test Commands

```bash
# CPU:
python main.py --file test_replays/10x10 --no-gpu --log

# GPU:
python main.py --file test_replays/10x10 --log
```

> Avoid 16×16 and 20×20 during iteration — use only at the very final validation stage. 10×10 is fast enough for rapid testing and all optimization signals appear there first.

## Baselines

| Path | 10×10 (1362 u) |
|------|-----------------|
| CPU  | 13.1s (104 f/s) |
| GPU  | 12.4s (110 f/s) |

u = unique frames. f/s = unique frames per second. Test system: GTX 1660 Super 6GB, 16GB RAM, hevc_nvenc.

## Optimization Ideas

| Variant | Optimization | Path | 10×10 total | 10×10 f/s | vs CPU base | vs GPU base |
|---------|-------------|------|-------------|-----------|-------------|-------------|
| Baseline CPU | current serial + pipe (hevc_nvenc) | CPU | 13.1s | 104 | — | +5.6% |
| Baseline GPU | current GPU path | GPU | 12.4s | 110 | -5.3% | — |
| A | Unique frames + concat demuxer | Both | TBD | TBD | TBD | TBD |
| B | Backport CPU overlays to GPU | GPU | TBD | TBD | TBD | TBD |

### A: Unique frames + concat demuxer

**Path:** Both

**Idea:** Write each unique frame once with per-frame duration via ffmpeg concat demuxer instead of duplicating frame writes.

**Implementation:**
1. Render unique frames, write to temp rawvideo file.
2. Build concat file with `duration N/fps` per unique frame (fps default 60, configurable).
3. Run ffmpeg with concat demuxer.

**Test:** 10×10 vs baseline. At final validation, also test 16×16.

### B: Backport CPU overlays to GPU

**Path:** GPU

**Idea:** GPU path still uses 5 full-canvas RGBA alpha blends and `static_base.copy()` per frame. Apply the same pre-blended `draw.rectangle()` pattern that saved ~2.5s on CPU.

**Implementation:**
- Replace alpha compositing in GPU overlay pass with pre-blended pixel writes.
- Cache static base on GPU, only upload dynamic values.

**Test:** 10×10 vs baseline. At final validation, also test 16×16.

## Context

- **10×10** (1569 moves): 1362 unique / 3167 total, 960×696 canvas, 2.0 MB/frame.
- **16×16** (7132 moves): 5692 unique / 12923 total, 1104×912 canvas, 3.9 MB/frame.
- **20×20** (14203 moves): 11177 unique / 23562 total, 1380×1116 canvas, 4.6 MB/frame.
- CPU serial path is O(1) memory (~5 MB canvas only). No SharedMemory, no page file usage.
- Bottleneck is pipe write to ffmpeg (108 GB for 20×20), not rendering.
- Progress phase weights: both paths now use `[8, 7, 85]` (overlapped render+encode). Any phase structure change must update these.
- `ProcessPoolExecutor` kept only for batch mode (`_batch_cpu_worker`); single-solution render is always serial.

## Relevant Files
- `replay_video.py`: CPU serial render loop (~1639–1665), GPU render path (~1565–1602)
- `gpu_renderer.py`: GPU overlay compositing (needs pre-blend backport)
- `track_progress.py`: phase weights `[8, 7, 85]`
- `test_replays/10x10`, `test_replays_gpu/16x16`, `test_replays_gpu/20x20`: test files

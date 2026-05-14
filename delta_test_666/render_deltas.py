"""Render 4x4 puzzle delta frames as dimmed-prev + bright-changed-tiles visualizations."""
import sys, os, numpy as np
from PIL import Image, ImageDraw, ImageEnhance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)
for f in os.listdir(OUT_DIR):
    if f.endswith('.png'):
        os.remove(os.path.join(OUT_DIR, f))

import replay_video
from replay_video import ReplayVideoGenerator, RenderOptions
from geometry import compute_grid_position
import inspect

_render_count = 0
_original_render_frame = replay_video.render_frame
_original_frame_args = set(inspect.signature(_original_render_frame).parameters.keys())

def _patched_render_frame(**kw):
    global _render_count
    img = _original_render_frame(**kw)
    has_delta = kw.get('prev_canvas') is not None and kw.get('delta_mask') is not None

    if has_delta:
        prev = kw['prev_canvas']
        dm = kw['delta_mask']
        matrix = kw['matrix']
        h, w = len(matrix), len(matrix[0])
        ts = kw['tile_size']
        opts = kw.get('opts', RenderOptions())
        gx, gy = compute_grid_position(opts.grid_only)

        rows, cols = np.where(dm)
        n_changed = len(rows)
        viz = prev.copy()
        viz = ImageEnhance.Brightness(viz).enhance(0.25)
        for r, c in zip(rows, cols):
            sx = gx + c * ts
            sy = gy + r * ts
            tile = img.crop((sx, sy, sx + ts, sy + ts))
            viz.paste(tile, (sx, sy))
            d = ImageDraw.Draw(viz)
            d.rectangle([sx, sy, sx + ts - 1, sy + ts - 1], outline=(0, 255, 0), width=2)
        label = f"frame{_render_count:04d}_DELTA_{n_changed}tiles"
        viz.save(os.path.join(OUT_DIR, f"{label}.png"))
    else:
        img.save(os.path.join(OUT_DIR, f"frame{_render_count:04d}_FULL.png"))

    _render_count += 1
    return img

replay_video.render_frame = _patched_render_frame
replay_video._RENDER_FRAME_ARGS = _original_frame_args

url_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "test_replays", "4x4")
with open(url_path, 'r') as f:
    replay_url = f.read().strip()

from sliding_puzzles import parse_replay_url
sol, tps, scramble, movetimes = parse_replay_url(replay_url)
print(f"4x4 puzzle: sol_len={len(sol)}, tps={tps}")

opts = RenderOptions()
gen = ReplayVideoGenerator(cleanup_frames=False)
print(f"Rendering delta visualizations to {OUT_DIR}/")
gen.generate_simple_replay(
    solution=sol, output_path=os.path.join(OUT_DIR, "replay.mp4"),
    tps=tps, scramble=scramble, movetimes=movetimes,
    use_gpu=False, show_progress=True, opts=opts,
)
print(f"DONE: {_render_count} frames")

"""Render puzzles at various sizes with default settings for visual review.
Run: python test_tilesize.py
Outputs go to big_test/
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_video import ReplayVideoGenerator, RenderOptions

OUT = "big_test"
os.makedirs(OUT, exist_ok=True)

sizes = list(range(4, 21)) + [30, 40, 50, 64, 100]

gen = ReplayVideoGenerator()
for sz in sizes:
    label = f"{sz}x{sz}"
    out = os.path.join(OUT, f"{label}.mp4")
    if os.path.exists(out):
        print(f"Skip {label}")
        continue
    print(f"Render {label}...", end=" ", flush=True)
    gen.generate_simple_replay(
        solution="L",
        output_path=out,
        use_gpu=True,
        show_progress=False,
        tps=30, fps=30,
        size=(sz, sz),
        opts=RenderOptions(grid_only=True),
    )
    print("done")

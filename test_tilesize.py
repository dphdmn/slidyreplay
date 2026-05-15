"""Render puzzles at various sizes with default settings for visual review.
Run: python test_tilesize.py
Outputs go to big_test/
"""
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_video import ReplayVideoGenerator, RenderOptions

OUT = "big_test"
os.makedirs(OUT, exist_ok=True)

sizes = list(range(3, 101))

gen = ReplayVideoGenerator()
for sz in sizes:
    label = f"{sz}x{sz}"
    mp4 = os.path.join(OUT, f"{label}.mp4")
    png = os.path.join(OUT, f"{label}.png")
    if os.path.exists(mp4) and os.path.exists(png):
        print(f"Skip {label}")
        os.remove(mp4)
        continue
    if not os.path.exists(mp4):
        print(f"Render {label}...", end=" ", flush=True)
        gen.generate_simple_replay(
            solution="L",
            output_path=mp4,
            use_gpu=True,
            show_progress=False,
            tps=30, fps=30, quality=1440,
            size=(sz, sz),
            opts=RenderOptions(grid_only=True),
        )
        print("done", end=" ", flush=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4, "-vframes", "1", png],
        capture_output=True,
    )
    os.remove(mp4)
    print(f"png")

"""
Bulk test: render all solutions from bulktest.txt via batch_render API.
Tests both CPU and GPU batch rendering directly from ReplayVideoGenerator.
"""

import os
import sys
import time

BULKTEST = "bulktest.txt"
OUT_DIR = "bulk_test"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_video import ReplayVideoGenerator


def run(tag: str, use_gpu: bool):
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(BULKTEST) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    items = []
    for idx, line in enumerate(lines):
        out_path = os.path.join(OUT_DIR, f"{tag}_{idx+1:03d}.mp4")
        items.append({"solution": line, "output_path": out_path, "tps": 15, "fps": 30})

    print(f"  {tag}: {len(items)} solutions")

    gen = ReplayVideoGenerator()
    t0 = time.time()
    gen.batch_render(items, use_gpu=use_gpu, show_progress=True)
    elapsed = time.time() - t0
    print(f"  {tag}: {elapsed:.1f}s total")
    return elapsed


def main():
    cpu = gpu = False
    for arg in sys.argv[1:]:
        if arg == "--cpu":
            cpu = True
        elif arg == "--gpu":
            gpu = True
        elif arg in ("-h", "--help"):
            print("Usage: python test_bulk.py [--cpu] [--gpu]")
            return

    if not cpu and not gpu:
        cpu = gpu = True

    results = []
    if cpu:
        print(f"\n{'='*60}\nCPU mode\n{'='*60}")
        elapsed = run("cpu", use_gpu=False)
        results.append(("CPU", elapsed))

    if gpu:
        print(f"\n{'='*60}\nGPU mode\n{'='*60}")
        elapsed = run("gpu", use_gpu=True)
        results.append(("GPU", elapsed))

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for tag, elapsed in results:
        print(f"  {tag}: {elapsed:.1f}s")
    print(f"Outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()

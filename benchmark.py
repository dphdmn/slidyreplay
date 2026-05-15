"""
Benchmark GPU renderer — layout vs no-layout comparison.
All outputs saved to logs/ folder.

Usage:
    python benchmark.py                  # all puzzles, both layout and no-layout
    python benchmark.py --layout         # layout only
    python benchmark.py --no-layout      # no-layout only
"""

import subprocess
import sys
import os
import json
import re
import datetime


_REPLAY_DIRS = ["test_replays", "test_replays_gpu"]


def run_bench(label: str, filepath: str, extra_args: list, output_path: str, no_layout=False) -> tuple:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = [
        sys.executable, os.path.join(script_dir, "main.py"),
        "--file", filepath,
        "--quality", "1.0",
        "--output", output_path,
        "--log",
    ]
    if no_layout:
        cmd.append("--no-layout")
    cmd += extra_args

    detail = {
        "label": label,
        "extra_args": extra_args,
        "output": os.path.basename(output_path),
        "returncode": None,
        "elapsed": None,
        "gpu_info": "",
        "error_lines": [],
    }

    print(f"  {label}...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
    detail["returncode"] = result.returncode

    for line in result.stdout.splitlines():
        if "GPU ON" in line or "GPU OFF" in line:
            detail["gpu_info"] = line.strip()
            print(f"[{line.strip()}]")
            break

    if result.returncode != 0:
        for line in result.stderr.splitlines():
            print(f"    {line}")
            detail["error_lines"].append(line)
        print("FAILED")
        return None, detail

    m = re.search(r"took\s+([\d.]+)s", result.stdout)
    elapsed = float(m.group(1)) if m else None
    detail["elapsed"] = elapsed

    m2 = re.search(r"(\d+)\s+unique\s*/\s*(\d+)\s+total\s+frames", result.stdout)
    unique_frames = int(m2.group(1)) if m2 else None
    total_frames = int(m2.group(2)) if m2 else None
    detail["unique_frames"] = unique_frames
    detail["total_frames"] = total_frames

    print(f"done ({elapsed:.1f}s)" if elapsed else "done")
    return elapsed, detail


def parse_puzzle_info(content: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from sliding_puzzles import parse_replay_url
    from replay_generator import parse_scramble_guess, count_moves
    try:
        sol, tps, scramble, movetimes = parse_replay_url(content)
    except Exception:
        sol = content
        tps = None
        scramble = None
        movetimes = -1
    matrix = parse_scramble_guess(sol)
    has_mt = isinstance(movetimes, list) and len(movetimes) > 0
    return {
        "puzzle_size": f"{len(matrix)}x{len(matrix[0])}",
        "moves": count_moves(sol),
        "movetimes_count": len(movetimes) if has_mt else 0,
        "movetimes_accurate": has_mt,
        "tps_from_url": tps,
    }


def collect_replay_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files = []
    seen = set()
    for d in _REPLAY_DIRS:
        full = os.path.join(script_dir, d)
        if not os.path.isdir(full):
            continue
        for f in sorted(os.listdir(full), key=lambda x: [int(n) for n in re.findall(r"\d+", x)] if re.findall(r"\d+", x) else [0]):
            fp = os.path.join(full, f)
            if os.path.isfile(fp) and not f.startswith(".") and fp not in seen:
                seen.add(fp)
                files.append(fp)
    return files


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark GPU — layout vs no-layout")
    parser.add_argument("--layout", action="store_true", help="Layout mode only")
    parser.add_argument("--no-layout", action="store_true", help="No-layout mode only")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(script_dir, "logs", run_id)
    os.makedirs(logs_dir, exist_ok=True)

    do_layout = not args.no_layout or args.layout
    do_no_layout = not args.layout or args.no_layout

    replay_files = collect_replay_files()
    if not replay_files:
        print("No replay files found.")
        sys.exit(1)

    gpu_info_once = ""
    summary_data = []

    for filepath in replay_files:
        filename = os.path.basename(filepath)
        content = open(filepath, encoding="utf-8").read().strip()
        if not content:
            print(f"  Skipping {filename}: empty")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Puzzle: {filename}")
        print(f"{'=' * 60}")

        puzzle_info = parse_puzzle_info(content)
        puzzle_info["label"] = filename

        layout_time = None
        no_layout_time = None
        layout_detail = None
        no_layout_detail = None
        unique_frames = None
        total_frames = None

        if do_layout:
            out = os.path.join(logs_dir, f"{filename}_layout.mp4")
            t, d = run_bench(f"{filename} Layout", filepath, [], out, no_layout=False)
            layout_time = t
            layout_detail = d
            if d and d.get("gpu_info") and not gpu_info_once:
                gpu_info_once = d["gpu_info"]
            if d and d.get("unique_frames") is not None:
                unique_frames = d["unique_frames"]
                total_frames = d["total_frames"]

        if do_no_layout:
            out = os.path.join(logs_dir, f"{filename}_nolayout.mp4")
            t, d = run_bench(f"{filename} No-layout", filepath, [], out, no_layout=True)
            no_layout_time = t
            no_layout_detail = d
            if d and d.get("gpu_info") and not gpu_info_once:
                gpu_info_once = d["gpu_info"]
            if d and d.get("unique_frames") is not None:
                unique_frames = d["unique_frames"]
                total_frames = d["total_frames"]

        speedup = no_layout_time / layout_time if layout_time and no_layout_time and layout_time > 0 else 0

        print(f"  {'-' * 40}")
        print(f"  {'Mode':<25} {'Time':>8s}")
        print(f"  {'-' * 25} {'-' * 8}")
        if layout_time is not None:
            print(f"  {'Layout':<25} {layout_time:>7.1f}s")
        if no_layout_time is not None:
            print(f"  {'No-layout':<25} {no_layout_time:>7.1f}s")
        if layout_time and no_layout_time:
            print(f"  {'Speedup':<25} {speedup:>7.2f}x")

        summary_data.append({
            "puzzle": filename,
            "size": puzzle_info["puzzle_size"],
            "moves": puzzle_info["moves"],
            "unique_frames": unique_frames,
            "total_frames": total_frames,
            "layout_time_seconds": layout_time,
            "no_layout_time_seconds": no_layout_time,
            "speedup_removing_layout": round(speedup, 2) if speedup else None,
        })

    if not summary_data:
        print("No benchmarks were run.")
        sys.exit(1)

    print(f"\n\n{'=' * 60}")
    if gpu_info_once:
        print(f"  {gpu_info_once}")
    print(f"  OVERALL BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Puzzle':<12} {'Moves':<7} {'Unique':<7} {'Layout':>8s} {'No-layout':>10s} {'Speedup':>8s}")
    print(f"  {'-' * 12} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 10} {'-' * 8}")

    for d in summary_data:
        label = d["puzzle"]
        moves = d["moves"]
        unique = d["unique_frames"] if d["unique_frames"] else "?"
        layout_str = f"{d['layout_time_seconds']:.1f}s" if d["layout_time_seconds"] else "—"
        no_layout_str = f"{d['no_layout_time_seconds']:.1f}s" if d["no_layout_time_seconds"] else "—"
        speedup_str = f"{d['speedup_removing_layout']:.2f}x" if d["speedup_removing_layout"] else "—"
        print(f"  {label:<12} {moves:<7} {str(unique):<7} {layout_str:>8s} {no_layout_str:>10s} {speedup_str:>8s}")

    print("=" * 60)

    log = {
        "run_id": run_id,
        "summary": summary_data,
    }

    log_path = os.path.join(logs_dir, "benchmark_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nRun summary saved to: {log_path}")
    print(f"All outputs in: {logs_dir}")


if __name__ == "__main__":
    main()

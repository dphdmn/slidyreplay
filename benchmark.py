"""
Benchmark GPU renderer with stats logging.
All outputs saved to logs/ folder.

Usage:
    python benchmark.py                  # all puzzles: small (CPU+GPU) + big (GPU only)
    python benchmark.py --small          # small puzzles only (CPU+GPU)
    python benchmark.py --big            # big puzzles only (GPU only)
    python benchmark.py --no-layout      # all renders without timer/stats panel
"""

import subprocess
import sys
import os
import json
import re


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
        print(f"FAILED")
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
    from replay_video import parse_replay_url
    from replay_generator import parse_scramble_guess, expand_solution, guess_size
    try:
        sol, tps, scramble, movetimes = parse_replay_url(content)
    except Exception:
        sol = content
        tps = None
        scramble = None
        movetimes = -1
    matrix = parse_scramble_guess(sol)
    expanded = expand_solution(sol)
    has_mt = isinstance(movetimes, list) and len(movetimes) > 0
    return {
        "puzzle_size": f"{len(matrix)}x{len(matrix[0])}",
        "moves": len(expanded),
        "movetimes_count": len(movetimes) if has_mt else 0,
        "movetimes_accurate": has_mt,
        "tps_from_url": tps,
    }


def process_replays(replays_dir, logs_dir, gpu_only, no_layout=False):
    """Scan a replay directory, run benchmarks, return list of result dicts."""
    if not os.path.isdir(replays_dir):
        return []

    replay_files = sorted(
        (f for f in os.listdir(replays_dir)
         if os.path.isfile(os.path.join(replays_dir, f)) and not f.startswith(".")),
        key=lambda x: [int(n) for n in re.findall(r"\d+", x)] if re.findall(r"\d+", x) else [0],
    )

    results_list = []

    for replay_file in replay_files:
        filepath = os.path.join(replays_dir, replay_file)
        content = open(filepath, encoding="utf-8").read().strip()
        if not content:
            print(f"  Skipping {replay_file}: empty")
            continue

        puzzle_label = replay_file
        print(f"\n{'=' * 60}")
        print(f"  Puzzle: {puzzle_label}", end="")
        if gpu_only:
            print(" (GPU only)", end="")
        print()
        print(f"{'=' * 60}")

        puzzle_info = parse_puzzle_info(content)
        puzzle_info["label"] = puzzle_label

        results = []
        details = []

        if not gpu_only:
            cpu_out = os.path.join(logs_dir, f"{puzzle_label}_cpu.mp4")
            t, d = run_bench(f"{puzzle_label} CPU", filepath, ["--no-gpu"], cpu_out, no_layout=no_layout)
            results.append(("CPU baseline", t))
            if d:
                details.append(d)

        mode_label = "GPU no-layout" if no_layout else "GPU"
        gpu_out = os.path.join(logs_dir, f"{puzzle_label}_gpu.mp4")
        t, d = run_bench(f"{puzzle_label} {mode_label}", filepath, [], gpu_out, no_layout=no_layout)
        results.append(("GPU", t))
        if d:
            details.append(d)

        cpu_result = results[0] if results and results[0][0].startswith("CPU") else None
        cpu_time = cpu_result[1] if cpu_result else None
        has_cpu = cpu_time is not None

        print(f"  {'-' * 40}")
        print(f"  {'Mode':<25} {'Time':>8s}", end="")
        if has_cpu:
            print(f" {'vs CPU':>8s}", end="")
        print()
        print(f"  {'-' * 25} {'-' * 8}", end="")
        if has_cpu:
            print(f" {'-' * 8}", end="")
        print()

        for name, t in results:
            if t is not None:
                ratio = cpu_time / t if cpu_time and t > 0 else 0
                print(f"  {name:<25} {t:>7.1f}s", end="")
                if has_cpu and ratio > 0:
                    print(f" {ratio:>7.1f}x", end="")
                print()
            else:
                print(f"  {name:<25} {'FAILED':>8s}")

        # Extract times and unique frames for summary
        gpu_time = None
        gpu_info = ""
        unique_frames = None
        total_frames = None
        for (name, t), d in zip(results, details):
            if name.startswith("GPU") and t is not None:
                gpu_time = t
            if d and d.get("gpu_info"):
                gpu_info = d["gpu_info"]
            if d and d.get("unique_frames") is not None:
                unique_frames = d["unique_frames"]
                total_frames = d["total_frames"]

        speedup = cpu_time / gpu_time if cpu_time and gpu_time and gpu_time > 0 else 0

        results_list.append({
            "puzzle": puzzle_label,
            "size": puzzle_info["puzzle_size"],
            "moves": puzzle_info["moves"],
            "unique_frames": unique_frames,
            "total_frames": total_frames,
            "cpu_time_seconds": cpu_time,
            "gpu_time_seconds": gpu_time,
            "speedup_vs_cpu": round(speedup, 2) if speedup else None,
            "gpu_info": gpu_info,
            "gpu_only": gpu_only,
            "no_layout": no_layout,
        })

    return results_list


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run GPU benchmark with stats logging")
    parser.add_argument("--small", action="store_true",
                        help="Run small puzzles (test_replays/, CPU+GPU)")
    parser.add_argument("--big", action="store_true",
                        help="Run big puzzles (test_replays_gpu/, GPU only)")
    parser.add_argument("--no-layout", action="store_true",
                        help="Render with --no-layout (no timer bar or stats panel)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    import datetime
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(script_dir, "logs", run_id)
    os.makedirs(logs_dir, exist_ok=True)

    do_small = not args.big or args.small
    do_big = not args.small or args.big

    summary_data = []

    if do_small:
        summary_data += process_replays(
            os.path.join(script_dir, "test_replays"),
            logs_dir, gpu_only=False, no_layout=args.no_layout,
        )

    if do_big:
        summary_data += process_replays(
            os.path.join(script_dir, "test_replays_gpu"),
            logs_dir, gpu_only=True, no_layout=args.no_layout,
        )

    if not summary_data:
        print("No benchmarks were run.")
        sys.exit(1)

    print(f"\n\n{'=' * 60}")
    title = "OVERALL BENCHMARK SUMMARY (no-layout)" if args.no_layout else "OVERALL BENCHMARK SUMMARY"
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  {'Puzzle':<10} {'Moves':<7} {'Unique':<7} {'CPU':>8s} {'GPU':>8s} {'Speedup':>8s} {'GPU Info'}")
    print(f"  {'-' * 10} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 30}")

    for d in summary_data:
        label = d["puzzle"]
        moves = d["moves"]
        unique = d["unique_frames"] if d["unique_frames"] else "?"
        cpu_str = f"{d['cpu_time_seconds']:.1f}s" if d["cpu_time_seconds"] else "—"
        gpu_str = f"{d['gpu_time_seconds']:.1f}s" if d["gpu_time_seconds"] else "—"
        speedup_str = f"{d['speedup_vs_cpu']:.1f}x" if d["speedup_vs_cpu"] else "—"
        gpu_info = d.get("gpu_info", "")
        print(f"  {label:<10} {moves:<7} {str(unique):<7} {cpu_str:>8s} {gpu_str:>8s} {speedup_str:>8s} {gpu_info}")

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

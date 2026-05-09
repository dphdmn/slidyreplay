"""
Benchmark GPU renderer with stats logging.
All outputs saved to logs/ folder.

Usage:
    python benchmark.py         # all puzzles: small (CPU+GPU) + big (GPU only)
    python benchmark.py --small # small puzzles only (CPU+GPU)
    python benchmark.py --big   # big puzzles only (GPU only)
"""

import subprocess
import time
import sys
import os
import json
import re
import tempfile


def run_bench(label: str, url: str, extra_args: list, output_path: str, stats_path: str) -> tuple:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    url_file = tempfile.NamedTemporaryFile(mode="w", suffix=".url", delete=False)
    url_file.write(url)
    url_file.close()

    cmd = [
        sys.executable, os.path.join(script_dir, "main.py"),
        "--url-file", url_file.name,
        "--quality", "1.0",
        "--output", output_path,
        "--stats-path", stats_path,
    ] + extra_args

    detail = {
        "label": label,
        "extra_args": extra_args,
        "output": os.path.basename(output_path),
        "stats_path": os.path.abspath(stats_path),
        "returncode": None,
        "elapsed": None,
        "gpu_info": "",
        "stats": [],
        "error_lines": [],
    }

    print(f"  {label}...", end=" ", flush=True)
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
    elapsed = time.time() - start
    detail["elapsed"] = round(elapsed, 3)
    detail["returncode"] = result.returncode
    try:
        os.unlink(url_file.name)
    except OSError:
        pass

    for line in result.stdout.splitlines():
        if "GPU ON" in line or "GPU OFF" in line:
            detail["gpu_info"] = line.strip()
            print(f"[{line.strip()}]")
            break

    if os.path.exists(stats_path):
        with open(stats_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    detail["stats"].append(json.loads(line))

    if result.returncode != 0:
        print(f"FAILED ({elapsed:.1f}s)")
        for line in result.stderr.splitlines():
            print(f"    {line}")
            detail["error_lines"].append(line)
        return None, detail

    print(f"done ({elapsed:.1f}s)")
    return elapsed, detail


def parse_puzzle_info(url: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from replay_video import parse_replay_url
    from replay_generator import parse_scramble_guess, expand_solution
    sol, tps, scramble, movetimes = parse_replay_url(url)
    matrix = parse_scramble_guess(sol)
    expanded = expand_solution(sol)
    has_mt = isinstance(movetimes, list) and len(movetimes) > 0
    return {
        "puzzle_size": f"{len(matrix)}x{len(matrix[0])}",
        "moves": len(expanded),
        "frames": len(expanded) + 1,
        "movetimes_count": len(movetimes) if has_mt else 0,
        "movetimes_accurate": has_mt,
        "tps_from_url": tps,
    }


def process_replays(replays_dir, logs_dir, gpu_only):
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
        url = open(filepath, encoding="utf-8").read().strip()
        if not url:
            print(f"  Skipping {replay_file}: empty")
            continue

        puzzle_label = replay_file
        print(f"\n{'=' * 60}")
        print(f"  Puzzle: {puzzle_label}", end="")
        if gpu_only:
            print(" (GPU only)", end="")
        print()
        print(f"{'=' * 60}")

        puzzle_info = parse_puzzle_info(url)
        puzzle_info["label"] = puzzle_label

        results = []
        details = []

        if not gpu_only:
            cpu_out = os.path.join(logs_dir, f"{puzzle_label}_cpu.mp4")
            cpu_stats = os.path.join(logs_dir, f"{puzzle_label}_stats_cpu.jsonl")
            t, d = run_bench(f"{puzzle_label} CPU", url, ["--no-gpu"], cpu_out, cpu_stats)
            results.append(("CPU baseline", t))
            if d:
                details.append(d)

        gpu_out = os.path.join(logs_dir, f"{puzzle_label}_gpu.mp4")
        gpu_stats = os.path.join(logs_dir, f"{puzzle_label}_stats_gpu.jsonl")
        t, d = run_bench(f"{puzzle_label} GPU", url, ["--gpu"], gpu_out, gpu_stats)
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

        # Extract times for summary
        gpu_time = None
        gpu_info = ""
        for (name, t), d in zip(results, details):
            if name.startswith("GPU") and t is not None:
                gpu_time = t
            if d and d.get("gpu_info"):
                gpu_info = d["gpu_info"]

        speedup = cpu_time / gpu_time if cpu_time and gpu_time and gpu_time > 0 else 0

        results_list.append({
            "puzzle": puzzle_label,
            "size": puzzle_info["puzzle_size"],
            "moves": puzzle_info["moves"],
            "frames": puzzle_info["frames"],
            "cpu_time_seconds": cpu_time,
            "gpu_time_seconds": gpu_time,
            "speedup_vs_cpu": round(speedup, 2) if speedup else None,
            "gpu_info": gpu_info,
            "gpu_only": gpu_only,
        })

    return results_list


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run GPU benchmark with stats logging")
    parser.add_argument("--small", action="store_true",
                        help="Run small puzzles (test_replays/, CPU+GPU)")
    parser.add_argument("--big", action="store_true",
                        help="Run big puzzles (test_replays_gpu/, GPU only)")
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
            logs_dir, gpu_only=False,
        )

    if do_big:
        summary_data += process_replays(
            os.path.join(script_dir, "test_replays_gpu"),
            logs_dir, gpu_only=True,
        )

    if not summary_data:
        print("No benchmarks were run.")
        sys.exit(1)

    print(f"\n\n{'=' * 60}")
    print(f"  OVERALL BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Puzzle':<10} {'Moves':<7} {'Frames':<7} {'CPU':>8s} {'GPU':>8s} {'Speedup':>8s} {'GPU Info'}")
    print(f"  {'-' * 10} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 30}")

    for d in summary_data:
        label = d["puzzle"]
        moves = d["moves"]
        frames = d["frames"]
        cpu_str = f"{d['cpu_time_seconds']:.1f}s" if d["cpu_time_seconds"] else "—"
        gpu_str = f"{d['gpu_time_seconds']:.1f}s" if d["gpu_time_seconds"] else "—"
        speedup_str = f"{d['speedup_vs_cpu']:.1f}x" if d["speedup_vs_cpu"] else "—"
        gpu_info = d.get("gpu_info", "")
        print(f"  {label:<10} {moves:<7} {frames:<7} {cpu_str:>8s} {gpu_str:>8s} {speedup_str:>8s} {gpu_info}")

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

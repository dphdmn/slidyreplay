"""
Benchmark GPU vs CPU + GPU memory scaling using test_input.txt.
Saves a detailed log (benchmark_log.json) with per-batch render stats.

Usage:
    python benchmark.py
"""

import subprocess
import time
import sys
import os
import json
import shutil


def run_bench(label: str, extra_args: list, output_name: str, stats_path: str, run_id: str = "") -> tuple:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    url = open(os.path.join(script_dir, "test_input.txt")).read().strip()
    cmd = [
        sys.executable, os.path.join(script_dir, "main.py"),
        "--url", url,
        "--quality", "2.0",
        "--output", os.path.join(script_dir, output_name),
        "--stats-path", stats_path,
    ] + extra_args

    detail = {
        "label": label, "extra_args": extra_args, "output": output_name,
        "stats_path": os.path.abspath(stats_path), "returncode": None, "elapsed": None,
        "gpu_info": "", "stats": [], "error_lines": [], "run_id": run_id,
    }

    print(f"  {label}...", end=" ", flush=True)
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
    elapsed = time.time() - start
    detail["elapsed"] = round(elapsed, 3)
    detail["returncode"] = result.returncode

    for line in result.stdout.splitlines():
        if "GPU ON" in line or "GPU OFF" in line:
            detail["gpu_info"] = line.strip()
            print(f"[{line.strip()}]")
            break

    # Read per-batch stats file if it exists
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    detail["stats"].append(json.loads(line))
        pass  # Keep raw stats file in benchmark_stats/{run_id}/

    if result.returncode != 0:
        print(f"FAILED ({elapsed:.1f}s)")
        for line in result.stderr.splitlines():
            print(f"    {line}")
            detail["error_lines"].append(line)
        return None, detail

    print(f"done ({elapsed:.1f}s)")
    return elapsed, detail


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run GPU vs CPU benchmark")
    parser.add_argument("--skip-cpu", action="store_true",
                        help="Skip CPU baseline (GPU tests only)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    url_path = os.path.join(script_dir, "test_input.txt")
    if not os.path.exists(url_path):
        print(f"Error: {url_path} not found")
        sys.exit(1)

    url = open(url_path).read().strip()
    if not url:
        print("Error: test_input.txt is empty")
        sys.exit(1)

    sys.path.insert(0, script_dir)
    from replay_video import parse_replay_url
    from replay_generator import parse_scramble_guess, expand_solution
    sol, tps, scramble, movetimes = parse_replay_url(url)
    matrix = parse_scramble_guess(sol)
    expanded = expand_solution(sol)

    has_mt = isinstance(movetimes, list) and len(movetimes) > 0

    import datetime
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    puzzle_info = {
        "run_id": run_id,
        "puzzle_size": f"{len(matrix)}x{len(matrix[0])}",
        "moves": len(expanded),
        "frames": len(expanded) + 1,
        "movetimes_count": len(movetimes) if has_mt else 0,
        "movetimes_accurate": has_mt,
        "tps_from_url": tps,
    }

    print("=" * 55)
    print(f"  Puzzle: {puzzle_info['puzzle_size']}")
    print(f"  Moves: {puzzle_info['moves']}")
    print(f"  Frames: {puzzle_info['frames']}")
    print(f"  Movetimes: {puzzle_info['movetimes_count']} entries {'(accurate timing)' if has_mt else '(no movetimes)'}")
    print(f"  TPS (from URL): {tps}")
    print("=" * 55)

    mem_levels = [50, 70, 90]
    results = []
    details = []

    stats_dir = os.path.join(script_dir, "benchmark_stats", run_id)
    os.makedirs(stats_dir, exist_ok=True)

    if not args.skip_cpu:
        t, d = run_bench("CPU baseline", ["--no-gpu"], "bench_cpu.mp4",
                          os.path.join(stats_dir, f"stats_cpu_{run_id}.jsonl"), run_id)
        results.append(("CPU", None, t))
        if d: details.append(d)

    for pct in mem_levels:
        label = f"GPU mem={pct}%"
        out = f"bench_gpu_{pct}pct.mp4"
        stats_file = os.path.join(stats_dir, f"stats_gpu_{pct}pct_{run_id}.jsonl")
        t, d = run_bench(label, ["--gpu", "--memory-usage", f"{pct / 100:.2f}"], out, stats_file, run_id)
        results.append((f"GPU @ {pct}% mem", pct / 100, t))
        if d: details.append(d)

    # Print results table
    cpu_result = results[0] if results[0][0] == "CPU" else None
    cpu_time = cpu_result[2] if cpu_result else None
    has_cpu = cpu_time is not None

    print("-" * 55)
    if has_cpu:
        print(f"  {'Method':<20} {'Time':>8s} {'vs CPU':>8s}")
        print(f"  {'-'*20} {'-'*8} {'-'*8}")
    else:
        print(f"  {'Method':<20} {'Time':>8s}")
        print(f"  {'-'*20} {'-'*8}")
    for name, _, t in results:
        if t is not None:
            ratio = cpu_time / t if cpu_time else 0
            if has_cpu:
                print(f"  {name:<20} {t:>7.1f}s {ratio:>7.1f}x")
            else:
                print(f"  {name:<20} {t:>7.1f}s")
        else:
            print(f"  {name:<20} {'FAILED':>8s}")
    print("=" * 55)

    # Build detailed log
    log = {"puzzle": puzzle_info, "results": []}
    for (name, mem, elapsed), d in zip(results, details):
        stats = [s for s in d.get("stats", []) if s.get("event") == "batch_start"]
        batch_count = len(stats)
        batch_sizes = [s.get("batch_size", 0) for s in stats]
        free_mems = [s.get("free_mem_mb", 0) for s in stats]

        total_mem = None
        for s in stats:
            if "total_mem_mb" in s:
                total_mem = s["total_mem_mb"]
                break
        entry = {
            "method": name,
            "memory_fraction": mem,
            "elapsed_seconds": elapsed,
            "batch_count": batch_count,
            "stats_log_path": d.get("stats_path"),
            "batch_sizes": batch_sizes,
            "free_mem_mb_per_batch": free_mems,
            "used_mem_mb_per_batch": [s.get("used_mem_mb", 0) for s in stats],
            "total_mem_mb": total_mem,
            "batch_mem_mb_per_batch": [s.get("batch_mem_mb", 0) for s in stats],
            "alloc_baseline_mb_per_batch": [s.get("alloc_baseline_mb") for s in stats],
            "returncode": d.get("returncode"),
            "gpu_info": d.get("gpu_info", ""),
        }
        if cpu_time and elapsed:
            entry["speedup_vs_cpu"] = round(cpu_time / elapsed, 3)
        else:
            entry["speedup_vs_cpu"] = None
        log["results"].append(entry)

    log_path = os.path.join(script_dir, f"benchmark_log_{run_id}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nDetailed log saved to: {log_path}")

    print(f"\nPer-run stats saved in: {stats_dir}")


if __name__ == "__main__":
    main()

"""
Benchmark GPU renderer comparing two modes:
  - "logged" = with --stats-path (per-batch JSONL logging)
  - "stripped" = without logging (pure render time)

All outputs saved to logs/ folder (nothing left in root).

Usage:
    python benchmark.py
    python benchmark.py --skip-cpu
"""

import subprocess
import time
import sys
import os
import json
import shutil


def run_bench(label: str, extra_args: list, output_path: str, stats_path: str = None) -> tuple:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    url = open(os.path.join(script_dir, "test_input.txt")).read().strip()
    
    cmd = [
        sys.executable, os.path.join(script_dir, "main.py"),
        "--url", url,
        "--quality", "2.0",
        "--output", output_path,
    ]
    
    if stats_path:
        cmd.extend(["--stats-path", stats_path])
    
    cmd.extend(extra_args)

    detail = {
        "label": label,
        "extra_args": extra_args,
        "output": os.path.basename(output_path),
        "stats_path": os.path.abspath(stats_path) if stats_path else None,
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

    for line in result.stdout.splitlines():
        if "GPU ON" in line or "GPU OFF" in line:
            detail["gpu_info"] = line.strip()
            print(f"[{line.strip()}]")
            break

    if stats_path and os.path.exists(stats_path):
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run GPU benchmark: logged vs stripped modes")
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

    print("=" * 60)
    print(f"  Puzzle: {puzzle_info['puzzle_size']}")
    print(f"  Moves: {puzzle_info['moves']}")
    print(f"  Frames: {puzzle_info['frames']}")
    print(f"  Run ID: {run_id}")
    print("=" * 60)

    logs_dir = os.path.join(script_dir, "logs", run_id)
    os.makedirs(logs_dir, exist_ok=True)

    results = []
    details = []

    if not args.skip_cpu:
        cpu_out = os.path.join(logs_dir, "bench_cpu.mp4")
        cpu_stats = os.path.join(logs_dir, "stats_cpu.jsonl")
        t, d = run_bench("CPU baseline", ["--no-gpu"], cpu_out, cpu_stats)
        results.append(("CPU baseline", t))
        if d: details.append(d)

    stripped_out = os.path.join(logs_dir, "bench_gpu_stripped.mp4")
    t1, d1 = run_bench("GPU (stripped, no logging)", ["--gpu"], stripped_out, None)
    results.append(("GPU (stripped)", t1))
    if d1: details.append(d1)

    logged_out = os.path.join(logs_dir, "bench_gpu_logged.mp4")
    logged_stats = os.path.join(logs_dir, "stats_gpu_logged.jsonl")
    t2, d2 = run_bench("GPU (logged, JSONL per-batch)", ["--gpu"], logged_out, logged_stats)
    results.append(("GPU (logged)", t2))
    if d2: details.append(d2)

    cpu_result = results[0] if results[0][0].startswith("CPU") else None
    cpu_time = cpu_result[1] if cpu_result else None
    has_cpu = cpu_time is not None

    print("-" * 60)
    print(f"  {'Mode':<30} {'Time':>8s}", end="")
    if has_cpu:
        print(f" {'vs CPU':>8s}", end="")
    print()
    print(f"  {'-'*30} {'-'*8}", end="")
    if has_cpu:
        print(f" {'-'*8}", end="")
    print()

    for name, t in results:
        if t is not None:
            ratio = cpu_time / t if cpu_time and t > 0 else 0
            print(f"  {name:<30} {t:>7.1f}s", end="")
            if has_cpu and ratio > 0:
                print(f" {ratio:>7.1f}x", end="")
            print()
        else:
            print(f"  {name:<30} {'FAILED':>8s}")

    if t1 and t2:
        diff = t2 - t1
        diff_pct = (diff / t1 * 100) if t1 > 0 else 0
        print("-" * 60)
        print(f"  Logging overhead: {diff:.1f}s ({diff_pct:+.1f}%)")

    print("=" * 60)

    log = {
        "puzzle": puzzle_info,
        "comparison": {
            "gpu_stripped_sec": t1,
            "gpu_logged_sec": t2,
            "logging_overhead_sec": t2 - t1 if (t1 and t2) else None,
            "logging_overhead_pct": (t2 - t1) / t1 * 100 if (t1 and t2 and t1 > 0) else None,
        },
        "results": [],
    }

    for (name, elapsed), d in zip(results, details):
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
            "elapsed_seconds": elapsed,
            "batch_count": batch_count,
            "batch_sizes": batch_sizes,
            "free_mem_mb_per_batch": free_mems,
            "used_mem_mb_per_batch": [s.get("used_mem_mb", 0) for s in stats],
            "total_mem_mb": total_mem,
            "batch_mem_mb_per_batch": [s.get("batch_mem_mb", 0) for s in stats],
            "returncode": d.get("returncode"),
            "gpu_info": d.get("gpu_info", ""),
        }
        if cpu_time and elapsed:
            entry["speedup_vs_cpu"] = round(cpu_time / elapsed, 3)
        else:
            entry["speedup_vs_cpu"] = None
        log["results"].append(entry)

    log_path = os.path.join(logs_dir, "benchmark_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nRun summary saved to: {log_path}")
    print(f"All outputs in: {logs_dir}")


if __name__ == "__main__":
    main()

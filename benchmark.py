"""
Benchmark: GPU/CPU × Layout/No-layout comparison.
All outputs saved to logs/ folder.

Usage:
    python benchmark.py                    # 4-way: GPU+CPU × Layout+No-layout
    python benchmark.py --gpu-only         # GPU only (both layout modes)
    python benchmark.py --cpu-only         # CPU only (both layout modes)
    python benchmark.py --no-layout        # No-layout only (both GPU+CPU)
    python benchmark.py --layout           # Layout only (both GPU+CPU)
"""

import subprocess
import sys
import os
import json
import re
import datetime


_REPLAY_DIR = "test_replays"


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


def run_bench(label: str, filepath: str, extra_args: list, output_path: str) -> tuple:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    cmd = [
        sys.executable, os.path.join(script_dir, "main.py"),
        "--file", filepath,
        "--quality", "720",
        "--output", output_path,
        "--log",
    ]
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark GPU vs CPU × Layout vs No-layout")
    parser.add_argument("--gpu-only", action="store_true", help="GPU only")
    parser.add_argument("--cpu-only", action="store_true", help="CPU only")
    parser.add_argument("--layout", action="store_true", help="Layout only")
    parser.add_argument("--no-layout", dest="no_layout", action="store_true", help="No-layout only")
    parser.add_argument("--quality-test", action="store_true", help="Test 10x10 at all quality presets")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(script_dir, "logs", run_id)
    os.makedirs(logs_dir, exist_ok=True)

    do_gpu = not args.cpu_only or args.gpu_only
    do_cpu = not args.gpu_only or args.cpu_only
    do_layout = not args.no_layout or args.layout
    do_nolayout = not args.layout or args.no_layout

    replay_dir = os.path.join(script_dir, _REPLAY_DIR)
    if not os.path.isdir(replay_dir):
        print(f"Directory not found: {replay_dir}")
        sys.exit(1)

    replay_files = sorted(
        (f for f in os.listdir(replay_dir)
         if os.path.isfile(os.path.join(replay_dir, f)) and not f.startswith(".")),
        key=lambda x: [int(n) for n in re.findall(r"\d+", x)] if re.findall(r"\d+", x) else [0],
    )

    if not replay_files:
        print("No replay files found.")
        sys.exit(1)

    gpu_info_once = ""
    summary_data = []

    if args.quality_test:
        # Find 10x10 replay file
        target = "10x10"
        filepath = None
        for fname in replay_files:
            if target in fname:
                filepath = os.path.join(replay_dir, fname)
                filename = fname
                break
        if filepath is None:
            print(f"No {target} replay file found.")
            sys.exit(1)

        content = open(filepath, encoding="utf-8").read().strip()
        puzzle_info = parse_puzzle_info(content)
        print(f"\n{'=' * 60}")
        print(f"  Quality test: {filename} ({puzzle_info['puzzle_size']}, {puzzle_info['moves']} moves)")
        print(f"{'=' * 60}")

        qualities = [720, 1080, 1440, 2160]
        results = {}
        for q in qualities:
            out = os.path.join(logs_dir, f"{filename}_q{q}.mp4")
            label = f"GPU Layout {q}p"
            extra = ["--no-gpu"] if args.cpu_only else []
            t, d = run_bench(label, filepath, extra + ["--quality", str(q)], out)
            results[q] = t
            if d and d.get("gpu_info") and not gpu_info_once:
                gpu_info_once = d["gpu_info"]

        print(f"\n{'=' * 60}")
        if gpu_info_once:
            print(f"  {gpu_info_once}")
        print(f"  Quality benchmark: {filename}")
        print(f"{'=' * 60}")
        print(f"  {'Quality':>8} {'Time':>10}")
        print(f"  {'-' * 20}")
        for q in qualities:
            t = results.get(q)
            line = f"  {q:>4}p      {t:>7.1f}s" if t else f"  {q:>4}p      {'FAIL':>8}"
            print(line)
        print(f"{'=' * 60}")

        log = {
            "run_id": run_id,
            "type": "quality_test",
            "puzzle": filename,
            "qualities": {str(q): results[q] for q in qualities},
        }
        log_path = os.path.join(logs_dir, "benchmark_log.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\nRun summary saved to: {log_path}")
        print(f"All outputs in: {logs_dir}")
        sys.exit(0)

    for filename in replay_files:
        filepath = os.path.join(replay_dir, filename)
        content = open(filepath, encoding="utf-8").read().strip()
        if not content:
            print(f"  Skipping {filename}: empty")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Puzzle: {filename}")
        print(f"{'=' * 60}")

        puzzle_info = parse_puzzle_info(content)
        puzzle_info["label"] = filename

        times = {}
        details = {}
        unique_frames = None
        total_frames = None

        configs = []
        if do_gpu and do_layout:
            configs.append(("gpu_layout", [], "GPU Layout"))
        if do_gpu and do_nolayout:
            configs.append(("gpu_nolayout", ["--no-layout"], "GPU No-lay"))
        if do_cpu and do_layout:
            configs.append(("cpu_layout", ["--no-gpu"], "CPU Layout"))
        if do_cpu and do_nolayout:
            configs.append(("cpu_nolayout", ["--no-gpu", "--no-layout"], "CPU No-lay"))

        for key, extra, label in configs:
            out = os.path.join(logs_dir, f"{filename}_{key}.mp4")
            t, d = run_bench(label, filepath, extra, out)
            times[key] = t
            details[key] = d
            if d and d.get("gpu_info") and not gpu_info_once:
                gpu_info_once = d["gpu_info"]
            if d and d.get("unique_frames") is not None and unique_frames is None:
                unique_frames = d["unique_frames"]
                total_frames = d["total_frames"]

        print(f"  {'-' * 52}")
        for key, _, label in configs:
            t = times.get(key)
            print(f"  {label:<25} {t:>7.1f}s" if t else f"  {label:<25} {'FAIL':>8s}")

        if all(times.get(k) for k, _, _ in configs):
            print(f"  {'-' * 52}")
            gpu_l = times.get("gpu_layout")
            gpu_nl = times.get("gpu_nolayout")
            cpu_l = times.get("cpu_layout")
            cpu_nl = times.get("cpu_nolayout")
            if gpu_l and cpu_l:
                print(f"  GPUvsCPU (layout):{'':>9} {gpu_l:>7.1f}s vs {cpu_l:>7.1f}s  ({cpu_l/gpu_l:.1f}x)")
            if gpu_nl and cpu_nl:
                print(f"  GPUvsCPU (no-lay):{'':>7} {gpu_nl:>7.1f}s vs {cpu_nl:>7.1f}s  ({cpu_nl/gpu_nl:.1f}x)")
            if gpu_l and gpu_nl:
                print(f"  Layout vs NoLay (GPU): {gpu_l:>7.1f}s vs {gpu_nl:>7.1f}s  ({gpu_l/gpu_nl:.1f}x)")
            if cpu_l and cpu_nl:
                print(f"  Layout vs NoLay (CPU): {cpu_l:>7.1f}s vs {cpu_nl:>7.1f}s  ({cpu_l/cpu_nl:.1f}x)")

        row = {
            "puzzle": filename,
            "size": puzzle_info["puzzle_size"],
            "moves": puzzle_info["moves"],
            "unique_frames": unique_frames,
            "total_frames": total_frames,
        }
        for key, _, label in configs:
            row[key] = times.get(key)
            row[f"{key}_failed"] = details.get(key, {}).get("returncode") != 0 if details.get(key) else True
        summary_data.append(row)

    if not summary_data:
        print("No benchmarks were run.")
        sys.exit(1)

    print(f"\n\n{'=' * 70}")
    if gpu_info_once:
        print(f"  {gpu_info_once}")
    print(f"  OVERALL BENCHMARK SUMMARY")
    print(f"{'=' * 70}")

    has_gpu_lay = any(d.get("gpu_layout") for d in summary_data)
    has_gpu_nol = any(d.get("gpu_nolayout") for d in summary_data)
    has_cpu_lay = any(d.get("cpu_layout") for d in summary_data)
    has_cpu_nol = any(d.get("cpu_nolayout") for d in summary_data)

    cols = ["Puzzle", "Moves", "Unique"]
    if has_gpu_lay: cols.append("GPU Lay")
    if has_gpu_nol: cols.append("GPU NoL")
    if has_cpu_lay: cols.append("CPU Lay")
    if has_cpu_nol: cols.append("CPU NoL")

    headers = [f"{h:>10}" for h in cols]
    print(f"  {' '.join(headers)}")
    print(f"  {'-' * (len(cols) * 11)}")

    for d in summary_data:
        vals = [d["puzzle"][:10], str(d["moves"]), str(d["unique_frames"] if d.get("unique_frames") else "?")]
        for key, col in [("gpu_layout", "GPU Lay"), ("gpu_nolayout", "GPU NoL"),
                          ("cpu_layout", "CPU Lay"), ("cpu_nolayout", "CPU NoL")]:
            t = d.get(key)
            if t is not None:
                vals.append(f"{t:.1f}s")
            elif d.get(f"{key}_failed"):
                vals.append("FAIL")
            elif key.startswith("gpu") == has_gpu_lay:
                continue
            else:
                vals.append("—")
        line = "  " + " ".join(f"{v:>10}" for v in vals)
        print(line)

    print("=" * 70)

    log = {
        "run_id": run_id,
        "args": vars(args),
        "summary": summary_data,
    }

    log_path = os.path.join(logs_dir, "benchmark_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nRun summary saved to: {log_path}")
    print(f"All outputs in: {logs_dir}")


if __name__ == "__main__":
    main()

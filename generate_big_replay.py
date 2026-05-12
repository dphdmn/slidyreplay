import argparse
from replay_generator import ReplayGenerator


def read_solution(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def read_movetimes(path: str) -> list:
    with open(path) as f:
        raw = f.read().strip()
    if not raw:
        return -1
    if raw.startswith("[") and raw.endswith("]"):
        import json
        return json.loads(raw)
    return [int(x) for x in raw.replace(",", "\n").split()]


def main():
    parser = argparse.ArgumentParser(description="Generate simple replay URL from large solution/movetimes files.")
    parser.add_argument("--solution-file", required=True, help="Path to file containing the solution string")
    parser.add_argument("--movetimes-file", default=None, help="Path to file containing move timings (one per line, comma-separated, or JSON array)")
    parser.add_argument("--tps", type=float, default=None, help="Tiles per second")
    parser.add_argument("--time", type=float, default=None, help="Total time in seconds (alternative to tps)")
    parser.add_argument("--scramble", type=str, default=None, help="Scramble string (overrides size-based auto-detection)")
    parser.add_argument("--size", type=str, default=None, help="Grid dimensions as WxH (e.g. 4x4)")
    parser.add_argument("--output", default="big_replay.txt", help="Output file path (default: big_replay.txt)")
    args = parser.parse_args()

    solution = read_solution(args.solution_file)
    movetimes = read_movetimes(args.movetimes_file) if args.movetimes_file else -1
    size = tuple(int(x) for x in args.size.split("x")) if args.size else None

    gen = ReplayGenerator()
    url = gen.generate_simple_replay(
        solution=solution,
        tps=args.tps,
        scramble=args.scramble,
        size=size,
        movetimes=movetimes,
        time=args.time
    )

    with open(args.output, "w") as f:
        f.write(url + "\n")

    print(f"Replay URL saved to {args.output}")
    print(url)


if __name__ == "__main__":
    main()

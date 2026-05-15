from typing import List, Dict, Tuple, Optional, Union
from tabulate import tabulate

from sliding_puzzles import (
    decompress_string_to_array, read_solve_data,
    guess_size_square, parse_scramble_guess_square,
)
from grids_analysis import get_grid_states, filter_grid_stages
from replay_generator import (
    count_moves, expand_solution, scramble_to_puzzle, puzzle_to_scramble,
    calculate_manhattan_distance, get_cubic_estimate,
)


def splits_file(filename: str, grid_data: Optional[Dict] = None):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            return splits_formatted(content, grid_data=grid_data)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def splits_formatted(sol: str, grid_data: Optional[Dict] = None):
    try:
        data = splits(sol, grid_data=grid_data)
    except Exception:
        return "splits are failed"

    if data is None:
        return "Invalid splits data"

    splits_data = data[0]
    # [splits, time, moves, tps, scramble, size, md, mmd, cubic]
    time, moves_total, tps_final, scramble, size, md, mmd, cubic = data[1:9]

    if not splits_data or len(splits_data) < 6:
        return (
            f"{size} solved in {time} ({moves_total} / {tps_final})\n"
            f"MD: {md} ({mmd})\n"
            f"{scramble}\n"
            f"Cubic Estimate: {cubic}"
        )

    try:
        converted = []
        for item in splits_data:
            try:
                converted.append(float(item))
            except (ValueError, TypeError):
                converted.append(0.0)

        half = len(converted) // 2
        times = converted[:half]
        moves = [int(round(x)) for x in converted[half:]]

        cum_time = 0.0
        lines = []

        # Generate labels based on number of lines
        n = len(times)
        if n == 3:
            labels = ["S", "F1", "F2"]
        elif n == 7:
            labels = ["s1", "ss1", "f1", "f2", "ss2", "f3", "f4"]
        else:
            labels = [f"Step {i + 1}" for i in range(n)]

        def fmt_time(seconds):
            hours = int(seconds // 3600)
            remaining = seconds % 3600
            minutes = int(remaining // 60)
            remaining %= 60
            secs = remaining
            
            # Format milliseconds to 3 digits, padding with zeros if needed
            milliseconds = "{:.3f}".format(secs).split(".")[1]
            seconds_int = int(secs)
            
            if hours > 0:
                return f"{hours}:{minutes:02d}:{seconds_int:02d}.{milliseconds}"
            elif minutes > 0:
                return f"{minutes}:{seconds_int:02d}.{milliseconds}"
            else:
                return f"{seconds_int}.{milliseconds}"

        # Calculate widths
        compact_w = max(len(f"{fmt_time(t)}") if t > 0 else 1 for t in times)
        compact2_w= max(len(f"({m}/{m/t:.1f})") if t > 0 else 1 for t, m in zip(times, moves))
        total_w = max(len(fmt_time(sum(times[:i+1]))) for i in range(n))
        label_w = max(len(l) for l in labels)

        lines.append(f"Size: {size}")
        lines.append(f"Time: {time} ({moves_total} / {tps_final})")
        lines.append(f"MD: {md} ({mmd})")
        if size != "10x10":
            lines.append(f"Cubic Estimate: {cubic}")
        lines.append("")  
        
        # Header
        header = f"{'Total':>{total_w}} | {'Split (Moves/TPS)':<{compact_w+compact2_w+1}} | {'Step':<{label_w}}"
        lines.append(header)
        lines.append("-" * len(header))

        for t, m, label in zip(times, moves, labels):
            cum_time += t
            tps = m / t if t > 0 else 0.0
            compact = f"{fmt_time(t)}"
            compact2= f"({m}/{tps:.1f})"
            line = f"{fmt_time(cum_time):>{total_w}} | {compact:>{compact_w}} {compact2:>{compact2_w}} | {label:<{label_w}}"
            lines.append(line)

        return "\n".join(lines)

    except Exception as e:
        return f"Error formatting splits: {str(e)}"


def splits(sol: str, grid_data: Optional[Dict] = None) -> Optional[List[List[Union[float, int]]]]:
    if "?r=" not in sol:
        return None

    try:
        query_start = sol.index('?')
        query_params = sol[query_start + 1:].split('&')
        replay_param = ''
        for param in query_params:
            key_value = param.split('=')
            if len(key_value) == 2 and key_value[0] == 'r':
                replay_param = key_value[1]
                break
    except Exception:
        return None

    try:
        replay_data = decompress_string_to_array(replay_param)
    except Exception:
        return None

    solution, scramble, move_times, tps_from_url = None, None, None, None

    try:
        if len(replay_data) < 10:
            solution = replay_data[0]
            tps_from_url = replay_data[1] / 1000.0
            scramble = replay_data[2]
            move_times = replay_data[3]
        else:
            solve_data = read_solve_data(replay_data[1])
            solution = solve_data['solutions']
            try:
                scramble = puzzle_to_scramble(parse_scramble_guess_square(solution))
            except Exception:
                scramble = ""
            move_times = solve_data['move_times'][0]
    except Exception:
        return None

    if grid_data is not None:
        grids_states = grid_data
    else:
        try:
            grids_states = get_grid_states(solution, scramble)
        except Exception:
            grids_states = {}

    try:
        return calculate_splits(grids_states, move_times, solution, scramble, tps_from_url)
    except Exception:
        return None

def calculate_splits(grids_states: Dict, move_times, solution: str, scramble: str, tps_from_url=None) -> List[List[Union[float, int]]]:
    try:
        solution_length = count_moves(solution)
    except Exception:
        solution_length = 0

    puzzle_matrix = [[0]]
    try:
        puzzle_matrix = scramble_to_puzzle(scramble)
    except Exception:
        pass

    split_times = []
    split_moves = []

    can_compute_splits = isinstance(move_times, list) and len(move_times) > 0

    if can_compute_splits:
        try:
            w = len(puzzle_matrix[0])
            h = len(puzzle_matrix)
            relevant_grid_indices = filter_grid_stages(grids_states, w, h, add_last=solution_length - 1)
            previous_time = 0
            previous_move_count = 0

            for current_index in relevant_grid_indices[1:]:
                current_move_count = current_index + 1
                current_time = move_times[current_index]
                split_duration = current_time - previous_time
                moves_in_split = current_move_count - previous_move_count

                split_times.append(split_duration / 1000)
                split_moves.append(moves_in_split)
                previous_time = current_time
                previous_move_count = current_move_count
        except Exception:
            pass

    splits = split_times + split_moves

    time = 0
    moves = 0
    tps = "0"
    width = 0
    height = 0
    size = "?x?"

    try:
        width = len(puzzle_matrix[0])
        height = len(puzzle_matrix)
        size = f"{width}x{height}"
    except Exception:
        pass

    try:
        moves = solution_length
        if isinstance(move_times, list) and len(move_times) > 0:
            time = move_times[-1]
            tps = f"{(moves*1000 / time):.3f}"
        elif tps_from_url is not None and tps_from_url > 0:
            tps_val = tps_from_url
            time = moves * 1000 / tps_val
            tps = f"{tps_val:.3f}"
    except Exception:
        pass

    try:
        md = calculate_manhattan_distance(puzzle_matrix)
        mmd = f"{(moves / md):.3f}"
    except Exception:
        md = 0
        mmd = "0"

    try:
        cubic = f"{(get_cubic_estimate(time, width, height)/1000):.3f}"
    except Exception:
        cubic = "0"

    try:
        time = f"{(time/1000):.3f}"
    except Exception:
        time = "0"

    return [splits, time, moves, tps, scramble, size, md, mmd, cubic]

def format_time(milliseconds: float, cut: bool = False) -> str:
    hours = int(milliseconds // 3600000)
    remaining_millis = milliseconds % 3600000
    minutes = int(remaining_millis // 60000)
    remaining_seconds = int((remaining_millis % 60000) // 1000)
    milliseconds_part = int(remaining_millis % 1000)
    
    if cut:
        if hours > 0:
            return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
        elif minutes > 0:
            return f"{minutes}:{remaining_seconds:02d}"
        else:
            return f"{remaining_seconds}.{milliseconds_part:03d}"
    else:
        if hours > 0:
            return f"{hours}:{minutes:02d}:{remaining_seconds:02d}.{milliseconds_part:03d}"
        elif minutes > 0:
            return f"{minutes}:{remaining_seconds:02d}.{milliseconds_part:03d}"
        else:
            return f"{remaining_seconds}.{milliseconds_part:03d}"

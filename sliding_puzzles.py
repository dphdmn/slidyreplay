"""
sliding_puzzles.py - Pure sliding-puzzle operations.

Decompress replay URLs, guess puzzle sizes, validate scrambles,
expand matrices — no formatting, no grid analysis, no rendering.
"""

import json
import base64
import zlib
import re
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from urllib.parse import unquote

from replay_generator import guess_size, parse_scramble, create_puzzle


def decompress_string_to_array(compressed_string: str) -> List:
    if "r=" in compressed_string:
        compressed_string = compressed_string.split("r=")[1]
    decoded_url = unquote(compressed_string)
    padding_needed = len(decoded_url) % 4
    if padding_needed:
        decoded_url += "=" * (4 - padding_needed)
    binary_data = base64.b64decode(decoded_url)
    inflated_data = zlib.decompress(binary_data).decode("utf-8")
    return json.loads(inflated_data)


def read_solve_data(input_str: str) -> Dict:
    decoded_string = input_str
    binary_string = base64.b64decode(decoded_string)
    decompressed = zlib.decompress(binary_string).decode("utf-8")

    move_times = -1
    remaining_decompressed = decompressed

    open_bracket_index = decompressed.find("[")
    close_bracket_index = decompressed.find("]")

    if open_bracket_index != -1 and close_bracket_index != -1 and close_bracket_index > open_bracket_index:
        move_times_content = decompressed[open_bracket_index + 1 : close_bracket_index]
        move_times = [[int(num) for num in move_times_content.split(",")]]
        remaining_decompressed = decompressed[:open_bracket_index] + decompressed[close_bracket_index + 1 :]

    remaining_decompressed = remaining_decompressed.rstrip(";")
    parts = remaining_decompressed.split(";")

    return {
        "solutions": parts[0] if len(parts) > 0 else -1,
        "times": parts[1] if len(parts) > 1 else -1,
        "moves": parts[2] if len(parts) > 2 else -1,
        "tps": parts[3] if len(parts) > 3 else -1,
        "bld_times": parts[4] if len(parts) > 4 else -1,
        "move_times": move_times,
    }


def guess_size_square(solution: str) -> int:
    width, height = guess_size(solution)
    return max(width, height)


def validate_scramble(input_str: str) -> bool:
    if not re.match(r"^[0-9\s/]*$", input_str):
        return False

    parts = input_str.split("/")
    num_counts = [len(part.split()) for part in parts]
    all_equal = all(count == num_counts[0] for count in num_counts)

    all_numbers = []
    for part in parts:
        all_numbers.extend([int(num) for num in part.split()])

    sorted_numbers = sorted(all_numbers)
    is_sequential = all(num == i for i, num in enumerate(sorted_numbers))

    return all_equal and is_sequential


def parse_scramble_guess_square(solution: str) -> List[List[int]]:
    size = guess_size_square(solution)
    return parse_scramble(size, size, solution)


def expand_matrix(matrix: List[List[int]], W: int, H: int) -> List[List[int]]:
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    num_rows_diff = W - num_rows
    num_cols_diff = H - num_cols
    expanded = create_puzzle(W, H)
    mapping = np.array(create_puzzle(W, H))

    arr = np.array(matrix, dtype=np.int32)
    nonzero = arr != 0
    vals = arr[nonzero]
    row_idx = (vals - 1) // num_cols
    col_idx = (vals - 1) % num_cols
    expanded_np = np.array(expanded, dtype=np.int32)
    expanded_np[num_rows_diff:, num_cols_diff:][nonzero] = mapping[row_idx + num_rows_diff, col_idx + num_cols_diff]
    return expanded_np.tolist()


def parse_replay_url(url: str):
    parsed = decompress_string_to_array(url)
    if len(parsed) < 10:
        solution = parsed[0]
        tps = parsed[1] / 1000.0 if len(parsed) > 1 else None
        scramble = parsed[2] if len(parsed) > 2 else None
        movetimes = parsed[3] if len(parsed) > 3 else -1
    else:
        solve_data = read_solve_data(parsed[1])
        solution = solve_data['solutions']
        tps = solve_data.get('tps', None)
        if tps == -1 or tps == "-1" or tps is None:
            tps = None
        # Fallback to top-level TPS field (parsed[3]) if solve data has none
        if tps is None and len(parsed) > 3:
            try:
                tps = float(parsed[3]) / 1000.0
            except (ValueError, TypeError, ZeroDivisionError):
                tps = None
        scramble = None
        movetimes = solve_data.get('move_times', -1)
        if isinstance(movetimes, list) and len(movetimes) > 0:
            movetimes = movetimes[0]

    if isinstance(movetimes, str):
        movetimes = -1

    if tps is not None:
        try:
            tps = float(tps)
        except (ValueError, TypeError):
            tps = None

    return solution, tps, scramble, movetimes


# ─── Puzzle Move Operations ────────────────────────────────────────

_MOVE_DIRS = {
    'R': (0, -1), 'L': (0, 1),
    'U': (1, 0), 'D': (-1, 0),
}


def move_matrix_inplace(mc_flat, move, zp_idx, w):
    dr, dc = _MOVE_DIRS[move]
    nr_idx = zp_idx + dr * w + dc
    mc_flat[zp_idx], mc_flat[nr_idx] = mc_flat[nr_idx], mc_flat[zp_idx]


def find_zero(matrix, w, h):
    result = np.argwhere(np.asarray(matrix, dtype=np.int32) == 0)
    if result.shape[0] == 0:
        return -1, -1
    return int(result[0, 0]), int(result[0, 1])


def update_md_flat(md, mc_flat, move, zp_idx, w, h):
    dr, dc = _MOVE_DIRS[move]
    nr_idx = zp_idx + dr * w + dc
    moved_val = mc_flat[zp_idx]
    sr, sc = (moved_val - 1) // w, (moved_val - 1) % w
    old_md = abs(sr - nr_idx // w) + abs(sc - nr_idx % w)
    new_md = abs(sr - zp_idx // w) + abs(sc - zp_idx % w)
    return md - old_md + new_md

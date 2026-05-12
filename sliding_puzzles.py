"""
sliding_puzzles.py - Pure sliding-puzzle operations.

Decompress replay URLs, guess puzzle sizes, validate scrambles,
expand matrices — no formatting, no grid analysis, no rendering.
"""

import json
import base64
import zlib
import re
from typing import List, Dict, Optional, Union
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
    expanded_matrix = create_puzzle(W, H)
    mapping_matrix = create_puzzle(W, H)

    for i in range(num_rows):
        for j in range(num_cols):
            value = matrix[i][j]
            original_value = 0
            if value != 0:
                row_index = (value - 1) // num_cols
                col_index = (value - 1) % num_cols
                original_value = mapping_matrix[row_index + num_rows_diff][col_index + num_cols_diff]
            expanded_matrix[i + num_rows_diff][j + num_cols_diff] = original_value
    return expanded_matrix

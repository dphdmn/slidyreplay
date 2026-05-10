import json
import base64
import zlib
from typing import List, Dict, Tuple, Optional, Union
from urllib.parse import unquote
from tabulate import tabulate


def splits_file(filename: str):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            return splits_formatted(content)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def splits_formatted(sol: str):
    try:
        data = splits(sol)
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


def splits(sol: str) -> Optional[List[List[Union[float, int]]]]:
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
            tps_from_url = replay_data[1]
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

    try:
        grids_states = get_grids_states(solution, scramble)
    except Exception:
        grids_states = {}

    try:
        return calculate_splits(grids_states, move_times, solution, scramble, tps_from_url)
    except Exception:
        return None

def calculate_splits(grids_states: Dict, move_times, solution: str, scramble: str, tps_from_url=None) -> List[List[Union[float, int]]]:
    try:
        solution_length = len(expand_solution(solution))
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
            relevant_grid_indices = [int(key) for key in grids_states.keys()
                                   if grids_states[key]['activeZone']['width'] + 1 >= len(puzzle_matrix[0]) / 2
                                   and grids_states[key]['activeZone']['height'] + 1 >= len(puzzle_matrix) / 2]
            relevant_grid_indices.append(solution_length - 1)
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
            tps_val = tps_from_url / 1000
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

def read_solve_data(input_str: str) -> Dict:
    decoded_string = input_str
    binary_string = base64.b64decode(decoded_string)
    decompressed = zlib.decompress(binary_string).decode('utf-8')
    
    move_times = -1
    remaining_decompressed = decompressed
    
    open_bracket_index = decompressed.find('[')
    close_bracket_index = decompressed.find(']')
    
    if open_bracket_index != -1 and close_bracket_index != -1 and close_bracket_index > open_bracket_index:
        move_times_content = decompressed[open_bracket_index + 1:close_bracket_index]
        move_times = [[int(num) for num in move_times_content.split(',')]]
        remaining_decompressed = decompressed[:open_bracket_index] + decompressed[close_bracket_index + 1:]
    
    remaining_decompressed = remaining_decompressed.rstrip(';')
    parts = remaining_decompressed.split(';')
    
    return {
        'solutions': parts[0] if len(parts) > 0 else -1,
        'times': parts[1] if len(parts) > 1 else -1,
        'moves': parts[2] if len(parts) > 2 else -1,
        'tps': parts[3] if len(parts) > 3 else -1,
        'bld_times': parts[4] if len(parts) > 4 else -1,
        'move_times': move_times
    }

def get_grids_states(solution: str, custom_scramble: str) -> Dict:
    scramble_matrix = scramble_to_puzzle(custom_scramble)
    width = len(scramble_matrix[0])
    height = len(scramble_matrix)
    cycled_numbers = get_cycles_numbers(scramble_matrix, expand_solution(solution))
    grids_data = analyse_grids_initial(scramble_matrix, expand_solution(solution), cycled_numbers)
    grids_states = generate_grids_stats(grids_data)
    return grids_states

c_t_map = {
    'fringe': 1,
    'grids1': 2,
    'grids2': 3
}

def decompress_string_to_array(compressed_string: str) -> List:
    # Remove URL prefix if present (e.g., "https://slidysim.online/replay?r=")
    if "r=" in compressed_string:
        compressed_string = compressed_string.split("r=")[1]
    
    # URL-decode the string (handles %2B → '+', %2F → '/', %3D → '=')
    decoded_url = unquote(compressed_string)
    
    # Ensure proper base64 padding (length must be a multiple of 4)
    padding_needed = len(decoded_url) % 4
    if padding_needed:
        decoded_url += "=" * (4 - padding_needed)
    
    # Decode base64 → decompress zlib → parse JSON
    binary_data = base64.b64decode(decoded_url)
    inflated_data = zlib.decompress(binary_data).decode('utf-8')
    return json.loads(inflated_data)

def get_cubic_estimate(time: float, n: int, m: int) -> int:
    return int(2000 * time / (n * m * (n + m)))

def calculate_manhattan_distance(scrambled_matrix: List[List[int]]) -> int:
    height = len(scrambled_matrix)
    width = len(scrambled_matrix[0])
    total_distance = 0
    
    for i in range(height):
        for j in range(width):
            current_value = scrambled_matrix[i][j]
            if current_value != 0:
                target_row = (current_value - 1) // width
                target_col = (current_value - 1) % width
                distance = abs(target_row - i) + abs(target_col - j)
                total_distance += distance
    return total_distance

def expand_solution(solution: str) -> str:
    import re
    def expand(match):
        letter = match.group(1)
        count = int(match.group(2))
        return letter * count
    return re.sub(r'([A-Z])(\d+)', expand, solution)

def compress_solution(input_str: str) -> str:
    import re
    def compress(match):
        char = match.group(1)
        return char + str(len(match.group(0)))
    return re.sub(r'(.)\1+', compress, input_str)

def get_repeated_lengths(input_string: str) -> Dict[str, int]:
    repeated_width = 0
    repeated_height = 0
    
    for i in range(1, len(input_string)):
        if input_string[i] == input_string[i - 1]:
            if input_string[i] in 'DU':
                repeated_height += 1
            if input_string[i] in 'RL':
                repeated_width += 1
    return {'repeatedWidth': repeated_width, 'repeatedHeight': repeated_height}

def move_matrix(matrix: List[List[int]], move_type: str, zero_pos: Tuple[int, int], width: int, height: int) -> List[List[int]]:
    zero_row, zero_col = zero_pos
    updated_matrix = [row[:] for row in matrix]
    
    if move_type == 'R':
        if zero_col > 0:
            updated_matrix[zero_row][zero_col], updated_matrix[zero_row][zero_col - 1] = updated_matrix[zero_row][zero_col - 1], updated_matrix[zero_row][zero_col]
        else:
            raise ValueError(f"Invalid move: {move_type}\nPuzzle state: {puzzle_to_scramble(matrix)}")
    elif move_type == 'L':
        if zero_col < width - 1:
            updated_matrix[zero_row][zero_col], updated_matrix[zero_row][zero_col + 1] = updated_matrix[zero_row][zero_col + 1], updated_matrix[zero_row][zero_col]
        else:
            raise ValueError(f"Invalid move: {move_type}\nPuzzle state: {puzzle_to_scramble(matrix)}")
    elif move_type == 'U':
        if zero_row < height - 1:
            updated_matrix[zero_row][zero_col], updated_matrix[zero_row + 1][zero_col] = updated_matrix[zero_row + 1][zero_col], updated_matrix[zero_row][zero_col]
        else:
            raise ValueError(f"Invalid move: {move_type}\nPuzzle state: {puzzle_to_scramble(matrix)}")
    elif move_type == 'D':
        if zero_row > 0:
            updated_matrix[zero_row][zero_col], updated_matrix[zero_row - 1][zero_col] = updated_matrix[zero_row - 1][zero_col], updated_matrix[zero_row][zero_col]
        else:
            raise ValueError(f"Invalid move: {move_type}\nPuzzle state: {puzzle_to_scramble(matrix)}")
    else:
        raise ValueError(f"Unexpected move character: {move_type}\nPuzzle state: {puzzle_to_scramble(matrix)}")
    
    return updated_matrix

def apply_moves(matrix: List[List[int]], moves: str) -> Union[List[List[int]], int]:
    h = len(matrix)
    w = len(matrix[0])
    y, x = -1, -1
    
    for i in range(h):
        for j in range(w):
            if matrix[i][j] == 0:
                y, x = i, j
    
    for move in moves:
        dy, dx = 0, 0
        if move == 'U':
            dy, dx = 1, 0
        elif move == 'D':
            dy, dx = -1, 0
        elif move == 'L':
            dy, dx = 0, 1
        elif move == 'R':
            dy, dx = 0, -1
        
        ny, nx = y + dy, x + dx
        if ny < 0 or ny >= h or nx < 0 or nx >= w:
            return -1
        
        matrix[y][x], matrix[ny][nx] = matrix[ny][nx], matrix[y][x]
        y, x = ny, nx
    
    return matrix

def find_zero(matrix: List[List[int]], width: int, height: int) -> Tuple[int, int]:
    for i in range(height):
        for j in range(width):
            if matrix[i][j] == 0:
                return (i, j)
    return (-1, -1)

def reverse_solution(solution: str) -> str:
    reverse_map = {'U': 'D', 'D': 'U', 'L': 'R', 'R': 'L'}
    return ''.join([reverse_map.get(c, c) for c in reversed(solution)])

def guess_size(solution: str) -> Tuple[int, int]:
    solution = reverse_solution(expand_solution(solution))
    x, y = 1, 1
    width, height = 0, 0
    
    for move in solution:
        if move == 'D':
            y += 1
        elif move == 'R':
            x += 1
        elif move == 'U':
            y -= 1
        elif move == 'L':
            x -= 1
        
        width = max(width, x)
        height = max(height, y)
    
    return (max(2, width), max(2, height))

def guess_size_square(solution: str) -> int:
    width, height = guess_size(solution)
    return max(width, height)

def validate_scramble(input_str: str) -> bool:
    import re
    if not re.match(r'^[0-9\s/]*$', input_str):
        return False
    
    parts = input_str.split('/')
    num_counts = [len(part.split()) for part in parts]
    all_equal = all(count == num_counts[0] for count in num_counts)
    
    all_numbers = []
    for part in parts:
        all_numbers.extend([int(num) for num in part.split()])
    
    sorted_numbers = sorted(all_numbers)
    is_sequential = all(num == i for i, num in enumerate(sorted_numbers))
    
    return all_equal and is_sequential

def scramble_to_puzzle(input_string: str) -> List[List[int]]:
    return [[int(num) for num in row.split()] for row in input_string.split('/')]

def create_puzzle(width: int, height: int) -> List[List[int]]:
    counter = 1
    puzzle = []
    for i in range(height):
        row = []
        for j in range(width):
            if i == height - 1 and j == width - 1:
                row.append(0)
            else:
                row.append(counter)
                counter += 1
        puzzle.append(row)
    return puzzle

def puzzle_to_scramble(puzzle: List[List[int]]) -> str:
    return '/'.join([' '.join(map(str, row)) for row in puzzle])

def parse_scramble(width: int, height: int, solution: str) -> List[List[int]]:
    return apply_moves(create_puzzle(width, height), reverse_solution(expand_solution(solution)))

def parse_scramble_guess(solution: str) -> List[List[int]]:
    width, height = guess_size(solution)
    return parse_scramble(width, height, solution)

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

def get_cycles_numbers(matrix: List[List[int]], solution: str, moves_early: float = 0.96, moves_late: float = 0.98, safe_rect: float = 0.5) -> List[int]:
    width = len(matrix[0])
    height = len(matrix)
    sol_len = len(solution)
    early_count = int(moves_early * sol_len)
    late_count = int(moves_late * sol_len)
    safe_width = int(width * safe_rect)
    safe_height = int(height * safe_rect)
    unsolved_info = []
    matrix_copy = [row[:] for row in matrix]
    
    for move_index in range(late_count):
        move = solution[move_index]
        zero_pos = find_zero(matrix_copy, width, height)
        matrix_copy = move_matrix(matrix_copy, move, zero_pos, width, height)
        
        if move_index > early_count:
            unsolved_info.append(get_solve_elements_amount(matrix_copy, safe_width, safe_height))
    
    if unsolved_info:
        min_info = min(unsolved_info, key=lambda x: x['amount'])
        return min_info['arrayOfUnsolved']
    return []

def analyse_grids_initial(matrix: List[List[int]], solution: str, cycled_numbers: List[int]) -> Dict:
    width = len(matrix[0])
    height = len(matrix)
    return analyse_grids(matrix, solution, width, height, width, height, 0, 0, 0, cycled_numbers)

def generate_grids_stats(grids_data: Dict) -> Dict:
    levels = {}
    
    def traverse(node, id):
        if node:
            levels[id] = get_data_by_level(node)
            if 'nextLayerFirst' in node:
                traverse(node['nextLayerFirst'], node['gridsStarted'])
            if 'nextLayerSecond' in node:
                traverse(node['nextLayerSecond'], node['gridsStopped'])
    
    traverse(grids_data, 0)
    return levels

def get_grids_state(grids_states: Dict, move_index: int) -> Dict:
    keys = [int(key) for key in grids_states.keys()]
    highest_key = max([key for key in keys if key <= move_index], default=-1)
    return grids_states.get(highest_key, {})

def get_data_by_level(current_level: Dict) -> Dict:
    return {
        'secondaryColors': get_secondary_colors_by_level(current_level),
        'mainColors': get_main_colors_by_level(current_level),
        'activeZone': get_active_zone_by_level(current_level)
    }

def get_active_zone_by_level(current_level: Dict) -> Dict:
    return get_sizes_for_layer(0, current_level)

def get_main_colors_by_level(current_level: Dict) -> List[Dict]:
    if current_level.get('enableGridsStatus') == -1:
        return [get_sizes_for_layer(c_t_map['fringe'], current_level)]
    return [
        get_sizes_for_layer(c_t_map['grids1'], current_level.get('nextLayerFirst', {})),
        get_sizes_for_layer(c_t_map['grids2'], current_level.get('nextLayerSecond', {}))
    ]

def get_secondary_colors_by_level(current_level: Dict) -> List[Dict]:
    secondary_colors = []
    if current_level.get('enableGridsStatus') == -1:
        return secondary_colors
    
    f_l = current_level.get('nextLayerFirst', {})
    s_l = current_level.get('nextLayerSecond', {})
    
    if f_l and 'nextLayerFirst' in f_l:
        secondary_colors.append(get_sizes_for_layer(c_t_map['grids1'], f_l['nextLayerFirst']))
        secondary_colors.append(get_sizes_for_layer(c_t_map['grids2'], f_l['nextLayerSecond']))
    elif f_l:
        secondary_colors.append(get_sizes_for_layer(c_t_map['fringe'], f_l))
    
    if s_l and 'nextLayerSecond' in s_l:
        secondary_colors.append(get_sizes_for_layer(c_t_map['grids1'], s_l['nextLayerFirst']))
        secondary_colors.append(get_sizes_for_layer(c_t_map['grids2'], s_l['nextLayerSecond']))
    elif s_l:
        secondary_colors.append(get_sizes_for_layer(c_t_map['fringe'], s_l))
    
    return secondary_colors

def get_sizes_for_layer(type_n: int, layer: Dict) -> Dict:
    return {
        'type': type_n,
        'width': layer.get('width', 0),
        'height': layer.get('height', 0),
        'offsetW': layer.get('offsetW', 0),
        'offsetH': layer.get('offsetH', 0)
    }

def analyse_grids(matrix: List[List[int]], solution: str, width_initial: int, height_initial: int, 
                 width: int, height: int, offset_w: int, offset_h: int, 
                 moves_offset_counter: int, cycled_numbers: List[int]) -> Dict:
    matrix_copy = [row[:] for row in matrix]
    
    for move_index in range(len(solution)):
        move = solution[move_index]
        zero_pos = find_zero(matrix_copy, width_initial, height_initial)
        matrix_copy = move_matrix(matrix_copy, move, zero_pos, width_initial, height_initial)
        grids_status = guess_grids(matrix_copy, width, height, offset_w, offset_h, width_initial)
        
        if grids_status != 0:
            grids_started = move_index
            enable_grids_status = grids_status
            grids_unsolved_last = None
            matrix_before_grids = [row[:] for row in matrix_copy]
            
            for grids_stopped_temp_id in range(grids_started + 1, len(solution)):
                move = solution[grids_stopped_temp_id]
                zero_pos = find_zero(matrix_copy, width_initial, height_initial)
                matrix_copy = move_matrix(matrix_copy, move, zero_pos, width_initial, height_initial)
                
                if not grids_solved(matrix_copy, width, height, offset_w, offset_h, 
                                  enable_grids_status, width_initial, cycled_numbers):
                    grids_unsolved_last = grids_stopped_temp_id
                else:
                    break
            
            if grids_unsolved_last is None:
                return "Error, grids never stopped"
            else:
                grids_stopped = grids_unsolved_last + 1
                sol1 = solution[grids_started + 1:grids_stopped + 2]
                sol2 = solution[grids_stopped + 2:]
                new_parts = get_grids_parts(matrix_before_grids, sol1, width_initial, height_initial)
                
                if new_parts is not None:
                    if enable_grids_status == 1:
                        width_first = width
                        width_second = width
                        offset_w_first = offset_w
                        offset_w_second = offset_w
                        height_first = (height + 1) // 2
                        height_second = height - height_first
                        offset_h_first = offset_h
                        offset_h_second = height_first + offset_h
                        
                        return {
                            'enableGridsStatus': enable_grids_status,
                            'gridsStarted': grids_started + moves_offset_counter,
                            'gridsStopped': grids_stopped + moves_offset_counter,
                            'width': width,
                            'height': height,
                            'offsetW': offset_w,
                            'offsetH': offset_h,
                            'nextLayerFirst': analyse_grids(new_parts[0], sol1, width_initial, height_initial, 
                                                          width_first, height_first, offset_w_first, offset_h_first, 
                                                          moves_offset_counter + grids_started + 1, cycled_numbers),
                            'nextLayerSecond': analyse_grids(new_parts[1], sol2, width_initial, height_initial, 
                                                            width_second, height_second, offset_w_second, offset_h_second, 
                                                            moves_offset_counter + grids_stopped + 1, cycled_numbers)
                        }
                    elif enable_grids_status == 2:
                        width_first = (width + 1) // 2
                        width_second = width - width_first
                        offset_w_first = offset_w
                        offset_w_second = width_first + offset_w
                        height_first = height
                        height_second = height
                        offset_h_first = offset_h
                        offset_h_second = offset_h
                        
                        return {
                            'enableGridsStatus': enable_grids_status,
                            'gridsStarted': grids_started + moves_offset_counter,
                            'gridsStopped': grids_stopped + moves_offset_counter,
                            'width': width,
                            'height': height,
                            'offsetW': offset_w,
                            'offsetH': offset_h,
                            'nextLayerFirst': analyse_grids(new_parts[0], sol1, width_initial, height_initial, 
                                                          width_first, height_first, offset_w_first, offset_h_first, 
                                                          moves_offset_counter + grids_started + 1, cycled_numbers),
                            'nextLayerSecond': analyse_grids(new_parts[1], sol2, width_initial, height_initial, 
                                                            width_second, height_second, offset_w_second, offset_h_second, 
                                                            moves_offset_counter + grids_stopped + 1, cycled_numbers)
                        }
                
                return {
                    'enableGridsStatus': enable_grids_status,
                    'gridsStarted': grids_started,
                    'gridsStopped': grids_stopped,
                    'width': width,
                    'height': height,
                    'offsetW': offset_w,
                    'offsetH': offset_h,
                    'nextLayerFirst': None,
                    'nextLayerSecond': None
                }
    
    return {
        'enableGridsStatus': -1,
        'width': width,
        'height': height,
        'offsetW': offset_w,
        'offsetH': offset_h
    }

def get_grids_parts(matrix_before_grids: List[List[int]], solution: str, width: int, height: int) -> Optional[List[List[List[int]]]]:
    if width < 6 and height < 6:
        return None
    
    first_matrix = [row[:] for row in matrix_before_grids]
    
    for move in solution:
        zero_pos = find_zero(matrix_before_grids, width, height)
        matrix_before_grids = move_matrix(matrix_before_grids, move, zero_pos, width, height)
    
    second_matrix = matrix_before_grids
    return [first_matrix, second_matrix]

def guess_grids(matrix: List[List[int]], width: int, height: int, offset_w: int, offset_h: int, width_initial: int) -> int:
    if width < 6 and height < 6:
        return 0
    
    if height > 5:
        if check_top_bottom(matrix, width, height, offset_w, offset_h, width_initial):
            return 1
    
    if width > 5:
        if check_left_right(matrix, width, height, offset_w, offset_h, width_initial):
            return 2
    
    return 0

def check_top_bottom(matrix: List[List[int]], width: int, height: int, offset_w: int, offset_h: int, width_initial: int) -> bool:
    new_h = (height + 1) // 2 + offset_h
    solved_counter = 0
    
    for row in range(offset_h, new_h):
        for col in range(offset_w, width + offset_w):
            number = matrix[row][col]
            if number != 0 and (number - 1) // width_initial >= new_h:
                return False
            if number_is_solved(number, row, col, width_initial):
                solved_counter += 1
    
    return width * (new_h - offset_h) / 3 > solved_counter

def check_left_right(matrix: List[List[int]], width: int, height: int, offset_w: int, offset_h: int, width_initial: int) -> bool:
    new_w = (width + 1) // 2 + offset_w
    solved_counter = 0
    
    for row in range(offset_h, height + offset_h):
        for col in range(offset_w, new_w):
            number = matrix[row][col]
            if number != 0 and (number - 1) % width_initial >= new_w:
                return False
            if number_is_solved(number, row, col, width_initial):
                solved_counter += 1
    
    return height * (new_w - offset_w) / 3 > solved_counter

def grids_solved(matrix: List[List[int]], width: int, height: int, offset_w: int, offset_h: int, 
                grids_type: int, width_initial: int, cycled_numbers: List[int]) -> bool:
    if grids_type == 1:
        new_h = (height + 1) // 2 + offset_h
        for row in range(offset_h, new_h):
            for col in range(offset_w, width + offset_w):
                number = matrix[row][col]
                if number != 0 and not number_is_solved(number, row, col, width_initial):
                    if number not in cycled_numbers:
                        return False
    elif grids_type == 2:
        new_w = (width + 1) // 2 + offset_w
        for row in range(offset_h, height + offset_h):
            for col in range(offset_w, new_w):
                number = matrix[row][col]
                if number != 0 and not number_is_solved(number, row, col, width_initial):
                    if number not in cycled_numbers:
                        return False
    return True

def number_is_solved(number: int, row: int, col: int, width: int) -> bool:
    if number == 0:
        return False
    return (number - 1) // width == row and (number - 1) % width == col

def get_solve_elements_amount(matrix: List[List[int]], safe_width: int = 0, safe_height: int = 0) -> Dict:
    flat_matrix = [num for row in matrix for num in row]
    unsolved = []
    
    for index, num in enumerate(flat_matrix):
        if num == 0:
            continue
        
        expected_row = index // len(matrix[0])
        expected_col = index % len(matrix[0])
        
        if (num != expected_row * len(matrix[0]) + expected_col + 1 and
            not (expected_row >= len(matrix) - safe_height and 
                 expected_col >= len(matrix[0]) - safe_width)):
            unsolved.append(num)
    
    return {
        'amount': len(unsolved),
        'arrayOfUnsolved': unsolved
    }
    
if __name__ == '__main__':
    with open("big_solve.txt", 'r') as file:
        sol = file.read()  
    query_start = sol.index('?')
    query_params = sol[query_start + 1:].split('&')
    replay_param = ''
    
    for param in query_params:
        key_value = param.split('=')
        if len(key_value) == 2 and key_value[0] == 'r':
            replay_param = key_value[1]
            break
    
    replay_data = decompress_string_to_array(replay_param)
    solution, scramble, move_times = None, None, None
    
    if len(replay_data) < 10:
        solution = replay_data[0]
        scramble = replay_data[2]
        move_times = replay_data[3]
    else:
        solve_data = read_solve_data(replay_data[1])
        solution = solve_data['solutions']
        scramble = puzzle_to_scramble(parse_scramble_guess_square(solution))
        move_times = solve_data['move_times'][0]
    with open("sol.txt", 'w') as file:
        file.write(solution)
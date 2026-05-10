import json
import base64
import zlib
import re
from datetime import datetime


def encode_uri_component(string):
    return ''.join(c if c.isalnum() or c in ['-', '_', '.', '~'] else f'%{ord(c):02X}' for c in string)


def compress_array_to_string(input_array):
    json_string = json.dumps(input_array)
    compressed_data = zlib.compress(json_string.encode(), level=9)
    base64_encoded = base64.b64encode(compressed_data).decode()
    return encode_uri_component(base64_encoded)


def compress_solution(solution):
    return re.sub(r'(.)\1+', lambda m: m.group(1) + str(len(m.group(0))), solution)


def expand_solution(solution):
    return re.sub(r'([RULD])(\d+)', lambda m: m.group(1) * int(m.group(2)), solution)


def reverse_solution(solution):
    rev = {'U': 'D', 'D': 'U', 'L': 'R', 'R': 'L'}
    return ''.join(rev.get(c, c) for c in reversed(solution))


def create_puzzle(width, height):
    counter = 1
    matrix = []
    for i in range(height):
        row = []
        for j in range(width):
            if i == height - 1 and j == width - 1:
                row.append(0)
            else:
                row.append(counter)
                counter += 1
        matrix.append(row)
    return matrix


def puzzle_to_scramble(matrix):
    return '/'.join(' '.join(str(n) for n in row) for row in matrix)


def scramble_to_puzzle(scramble_str):
    return [[int(n) for n in row.split()] for row in scramble_str.split('/')]


def apply_moves(matrix, moves):
    h = len(matrix)
    w = len(matrix[0])
    y = x = -1
    for i in range(h):
        for j in range(w):
            if matrix[i][j] == 0:
                y, x = i, j
                break
        if y != -1:
            break
    dirs = {'U': (1, 0), 'D': (-1, 0), 'L': (0, 1), 'R': (0, -1)}
    for move in moves:
        dy, dx = dirs[move]
        ny, nx = y + dy, x + dx
        if ny < 0 or ny >= h or nx < 0 or nx >= w:
            return None
        matrix[y][x], matrix[ny][nx] = matrix[ny][nx], matrix[y][x]
        y, x = ny, nx
    return matrix


def guess_size(solution):
    sol = reverse_solution(expand_solution(solution))
    x = y = 1
    width = height = 0
    for move in sol:
        if move == 'D':
            y += 1
        if move == 'R':
            x += 1
        if move == 'U':
            y -= 1
        if move == 'L':
            x -= 1
        width = max(width, x)
        height = max(height, y)
    return max(2, width), max(2, height)


def parse_scramble(width, height, solution):
    puzzle = create_puzzle(width, height)
    result = apply_moves(puzzle, reverse_solution(expand_solution(solution)))
    return result


def parse_scramble_guess(solution):
    w, h = guess_size(solution)
    return parse_scramble(w, h, solution)


def calculate_manhattan_distance(matrix):
    h = len(matrix)
    w = len(matrix[0])
    total = 0
    for i in range(h):
        for j in range(w):
            val = matrix[i][j]
            if val != 0:
                tr = (val - 1) // w
                tc = (val - 1) % w
                total += abs(tr - i) + abs(tc - j)
    return total


def get_cubic_estimate(time_ms, n, m):
    return int(2000 * time_ms / (n * m * (n + m)))


def get_repeated_lengths(solution):
    repeated_width = 0
    repeated_height = 0
    for i in range(1, len(solution)):
        if solution[i] == solution[i - 1]:
            if solution[i] in 'DU':
                repeated_height += 1
            if solution[i] in 'RL':
                repeated_width += 1
    return repeated_width, repeated_height


class ReplayGenerator:
    BASE_URL = "https://slidysim.github.io/replay?r="

    def __init__(self):
        pass

    def generate_simple_replay(self, solution, tps=None, scramble=None, size=None, movetimes=-1, time=None):
        if tps is not None and time is not None:
            raise ValueError("Provide either tps or time, not both")
        if scramble is not None:
            matrix = scramble_to_puzzle(scramble)
            width = len(matrix[0])
            height = len(matrix)
        elif size is not None:
            width, height = size
            matrix = parse_scramble(width, height, solution)
            scramble = puzzle_to_scramble(matrix)
        else:
            matrix = parse_scramble_guess(solution)
            width = len(matrix[0])
            height = len(matrix)
            scramble = puzzle_to_scramble(matrix)
        solution_expanded = expand_solution(solution)
        if time is not None:
            tps = len(solution_expanded) / time
        elif tps is None:
            tps = 15
        tps_int = int(tps * 1000) if tps < 1000 else int(tps)
        replay_data = [solution, tps_int, scramble, movetimes]
        return self.BASE_URL + compress_array_to_string(replay_data)

    def generate_complex_replay(self, width, height, controls_text, solve_type, display_type, timestamp,
                                solutions_list, times_list, moves_list, tps_list,
                                bld_memo_list=-1, movetimes_list=None,
                                final_time_ms=-1, final_moves_ms=-1, final_tps_ms=-1):
        item = self._generate_item(
            width=width,
            height=height,
            controls_text=controls_text,
            solve_type=solve_type,
            display_type=display_type,
            avg_len=1 if ('relay' in solve_type.lower() or 'marathon' in solve_type.lower()) else len(solutions_list),
            time_ms=final_time_ms,
            moves_ms=final_moves_ms,
            tps_ms=final_tps_ms,
            timestamp=timestamp
        )
        solve_data = self._generate_solve_data(
            solutions_list=solutions_list,
            times_list=times_list,
            moves_list=moves_list,
            tps_list=tps_list,
            bld_memo_list=bld_memo_list,
            movetimes_list=movetimes_list if movetimes_list is not None else []
        )
        event = self._generate_event()
        score_title = self._generate_score_title(controls_text, timestamp)
        return self._create_replay_url(
            item=item,
            solve_data=solve_data,
            event=event,
            tps=final_tps_ms,
            width=width,
            height=height,
            score_title=score_title,
            video_link=-1,
            score_tier="alpha",
            is_wr=False
        )

    def generate_replay(self, solution, **kwargs):
        has_complex = all(k in kwargs for k in ['width', 'height', 'controls_text', 'solve_type',
                                                  'display_type', 'timestamp', 'solutions_list',
                                                  'times_list', 'moves_list', 'tps_list'])
        if has_complex:
            return self.generate_complex_replay(
                width=kwargs['width'],
                height=kwargs['height'],
                controls_text=kwargs['controls_text'],
                solve_type=kwargs['solve_type'],
                display_type=kwargs['display_type'],
                timestamp=kwargs['timestamp'],
                solutions_list=kwargs['solutions_list'],
                times_list=kwargs['times_list'],
                moves_list=kwargs['moves_list'],
                tps_list=kwargs['tps_list'],
                bld_memo_list=kwargs.get('bld_memo_list', -1),
                movetimes_list=kwargs.get('movetimes_list'),
                final_time_ms=kwargs.get('final_time_ms', -1),
                final_moves_ms=kwargs.get('final_moves_ms', -1),
                final_tps_ms=kwargs.get('final_tps_ms', -1)
            )
        tps = kwargs.get('tps')
        time = kwargs.get('time')
        scramble = kwargs.get('scramble')
        size = kwargs.get('size')
        movetimes = kwargs.get('movetimes', -1)
        return self.generate_simple_replay(
            solution=solution,
            tps=tps,
            time=time,
            scramble=scramble,
            size=size,
            movetimes=movetimes
        )

    def _generate_item(self, width, height, controls_text, solve_type, display_type,
                       avg_len, time_ms, moves_ms, tps_ms, timestamp):
        return {
            "width": width,
            "height": height,
            "leaderboardType": "time",
            "controls": controls_text,
            "gameMode": solve_type,
            "displayType": display_type,
            "nameFilter": "Player",
            "avglen": avg_len,
            "time": time_ms,
            "moves": moves_ms,
            "tps": tps_ms,
            "timestamp": timestamp,
            "solve_data_available": True,
            "videolink": -1
        }

    def _generate_solve_data(self, solutions_list, times_list, moves_list, tps_list,
                             bld_memo_list, movetimes_list):
        solutions_str = ",".join(solutions_list)
        times_str = ",".join(str(x) for x in times_list)
        moves_str = ",".join(str(x) for x in moves_list)
        tps_str = ",".join(str(x) for x in tps_list)
        bld_str = ",".join(str(x) for x in bld_memo_list) if isinstance(bld_memo_list, list) else str(bld_memo_list)
        movetimes_str = ";".join(
            ",".join(str(x) for x in sublist) for sublist in movetimes_list
        )
        movetimes_str = f"[{movetimes_str}]"
        combined = ";".join([solutions_str, times_str, moves_str, tps_str, bld_str, movetimes_str])
        compressed = zlib.compress(combined.encode(), level=9)
        return base64.b64encode(compressed).decode()

    def _generate_event(self):
        return {"isTrusted": True}

    def _generate_score_title(self, controls, timestamp):
        dt = datetime.fromtimestamp(timestamp / 1000)
        date_str = dt.strftime("%Y.%m.%d")
        time_str = dt.strftime("%H:%M:%S")
        return (
            f'<span>'
            f'Custom replay<br>'
            f'{controls} | {date_str} {time_str}'
            f'</span>'
        )

    def _create_replay_url(self, item, solve_data, event, tps, width, height,
                           score_title, video_link, score_tier, is_wr):
        replay_data = [item, solve_data, event, tps, width, height,
                       score_title, video_link, score_tier, is_wr]
        return self.BASE_URL + compress_array_to_string(replay_data)

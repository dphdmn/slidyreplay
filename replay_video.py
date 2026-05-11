"""
replay_video.py - Generate MP4 videos of sliding puzzle replays

Ports the frontend rendering logic (fringeColors.js, gridsAnalysis.js,
replayGeneration.js, etc.) to Python using Pillow + ffmpeg.
Supports all color schemes, grid detection, and stats display.
"""

import os
import re
import math
import json
import base64
import zlib
import shutil
import subprocess
import sys
import time as time_module
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Union, Dict
import bisect
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

from replay_generator import (
    expand_solution, scramble_to_puzzle, puzzle_to_scramble,
    create_puzzle, apply_moves, reverse_solution, parse_scramble,
    parse_scramble_guess, calculate_manhattan_distance,
    get_repeated_lengths, compress_solution
)

from splits import decompress_string_to_array, read_solve_data
from gpu_renderer import GPURenderer, _render_timer_text, CancelError
from debug_log import get_logger

log = get_logger()

import psutil as _psutil
_proc = _psutil.Process()
_baseline_ram = _proc.memory_info().rss

def log_ram(label: str) -> int:
    cur = _proc.memory_info().rss
    delta = cur - _baseline_ram
    log.info(f"  RAM [{label}]: {cur // (1024*1024)}MB ({delta // (1024*1024):+d}MB vs baseline)")
    return delta

# ─── Replay URL Parsing ────────────────────────────────────────────

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
        if tps == -1 or tps is None:
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

BASE_SIZE = 15
BG_COLOR = (18, 18, 18)
TILE_BG = (51, 51, 51)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_BORDER_COLOR = (0, 0, 0)
NULL_COLOR = (248, 24, 148)
PANEL_BG = (17, 17, 17)
PANEL_ALPHA = 0.69
TIMER_BG = (22, 22, 22)
ACCURATE_COLOR = (0, 255, 0)
INACCURATE_COLOR = (255, 255, 255)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)

_font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_FAMILY = os.path.join(_font_dir, "Roboto-Regular.ttf")
FONT_FAMILY_BOLD = os.path.join(_font_dir, "Roboto-Bold.ttf")
FONT_FAMILY_MONO = os.path.join(_font_dir, "JetBrainsMono-Regular.ttf")
FONT_FAMILY_MONO_BOLD = os.path.join(_font_dir, "JetBrainsMono-Bold.ttf")
TILE_BORDER_RADIUS_RATIO = 0.4
TILE_BORDER_WIDTH = 1

PADDING = 20
STATS_PANEL_WIDTH = 320
TIMER_HEIGHT = 30
HEADER_H = 56
INFO_H = 40

# ─── Color Utilities (ported from fringeColors.js) ──────────────────

def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    def hue2rgb(p, q, t):
        if t < 0: t += 1
        if t > 1: t -= 1
        if t < 1/6: return p + (q - p) * 6 * t
        if t < 1/2: return q
        if t < 2/3: return p + (q - p) * (2/3 - t) * 6
        return p

    if s == 0:
        r = g = b = l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue2rgb(p, q, h + 1/3)
        g = hue2rgb(p, q, h)
        b = hue2rgb(p, q, h - 1/3)

    return (round(r * 255), round(g * 255), round(b * 255))


def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    if num_colors < 1:
        return []
    colors = []
    color_step = 360.0 / num_colors
    for i in range(num_colors):
        hue = i * color_step
        colors.append(hsl_to_rgb(hue / 360.0, 0.78, 0.6))
    return colors


def generate_color_fringe(colors_list: List[Tuple[int, int, int]], size: int) -> List[List[Tuple[int, int, int]]]:
    matrix = [[None] * size for _ in range(size)]
    for i, color in enumerate(colors_list):
        if i == 0:
            for j in range(size):
                matrix[0][j] = color
        elif i % 2 == 0:
            row_idx = i // 2
            for j in range(size):
                if matrix[row_idx][j] is None:
                    matrix[row_idx][j] = color
        else:
            col_idx = i // 2
            for j in range(size):
                if matrix[j][col_idx] is None:
                    matrix[j][col_idx] = color
    matrix[size - 1][size - 1] = NULL_COLOR
    return matrix


def split_matrix(matrix: List[List[int]]):
    height = len(matrix)
    width = len(matrix[0])
    square_size = min(width, height)
    square_matrix = [[0] * square_size for _ in range(square_size)]
    for i in range(square_size):
        for j in range(square_size):
            square_matrix[i][j] = matrix[height - square_size + i][width - square_size + j]

    other_part = None
    if width != height:
        if width > height:
            op_width = width - square_size
            other_part = [[0] * op_width for _ in range(height)]
            for i in range(height):
                for j in range(op_width):
                    other_part[i][j] = matrix[i][j]
        else:
            op_height = height - square_size
            other_part = [[0] * width for _ in range(op_height)]
            for i in range(op_height):
                for j in range(width):
                    other_part[i][j] = matrix[i][j]
    return square_matrix, other_part


def get_columns_colors(colors_list: List[Tuple[int, int, int]], width: int, height: int):
    return [[colors_list[c % len(colors_list)] for c in range(width)] for _ in range(height)]


def get_rows_colors(colors_list: List[Tuple[int, int, int]], width: int, height: int):
    return [[colors_list[r % len(colors_list)]] * width for r in range(height)]


def merge_matrices_by_dimension(matrix1: List[List], matrix2: List[List], match_by_width: bool):
    if match_by_width:
        return matrix1 + matrix2
    else:
        return [row1 + (matrix2[i] if i < len(matrix2) else []) for i, row1 in enumerate(matrix1)]


def get_fringe_colors_nxm(width: int, height: int) -> List[List[Tuple[int, int, int]]]:
    puzzle = create_puzzle(width, height)
    sq_matrix, start_matrix = split_matrix(puzzle)
    sq_size = len(sq_matrix)

    if start_matrix is None:
        num_colors = sq_size * 2 - 2
        colors_list = get_colors(num_colors)
        return generate_color_fringe(colors_list, sq_size)

    orig_w, orig_h = width, height
    start_w = len(start_matrix[0])
    start_h = len(start_matrix)
    sq_size = len(sq_matrix)
    extra_size = max(orig_w, orig_h) - sq_size
    num_colors = extra_size + sq_size * 2 - 2
    colors_list = get_colors(num_colors)
    start_colors = colors_list[:extra_size]
    square_colors = colors_list[extra_size:]
    colors_matrix_sq = generate_color_fringe(square_colors, sq_size)

    match_by_width = orig_w < orig_h
    if not match_by_width:
        extra_colors_matrix = get_columns_colors(start_colors, start_w, start_h)
    else:
        extra_colors_matrix = get_rows_colors(start_colors, start_w, start_h)

    return merge_matrices_by_dimension(extra_colors_matrix, colors_matrix_sq, match_by_width)


def get_mono_colors(color: Tuple[int, int, int], width: int, height: int):
    return [[color] * width for _ in range(height)]


# ─── Grids Analysis (ported from gridsAnalysis.js) ──────────────────

CT_MAP = {'fringe': 1, 'grids1': 2, 'grids2': 3}


def find_zero(matrix, w, h):
    for i in range(h):
        for j in range(w):
            if matrix[i][j] == 0:
                return i, j
    return -1, -1


def move_matrix(matrix, move, zero_pos, w, h):
    zr, zc = zero_pos
    updated = [row[:] for row in matrix]
    moves = {
        'R': (0, -1), 'L': (0, 1),
        'U': (1, 0), 'D': (-1, 0),
    }
    dr, dc = moves[move]
    nr, nc = zr + dr, zc + dc
    updated[zr][zc], updated[nr][nc] = updated[nr][nc], updated[zr][zc]
    return updated


def number_is_solved(num, row, col, w):
    if num == 0:
        return False
    return (num - 1) // w == row and (num - 1) % w == col


def get_solve_elements_amount(matrix, safe_w=0, safe_h=0):
    h = len(matrix)
    w = len(matrix[0])
    unsolved = []
    for idx, num in enumerate(matrix_flat := [v for row in matrix for v in row]):
        if num == 0:
            continue
        exp_row = idx // w
        exp_col = idx % w
        if (num != exp_row * w + exp_col + 1 and
                not (exp_row >= h - safe_h and exp_col >= w - safe_w)):
            unsolved.append(num)
    return len(unsolved), unsolved


def get_cycles_numbers(matrix, solution, moves_early=0.96, moves_late=0.98, safe_rect=0.5):
    w = len(matrix[0])
    h = len(matrix)
    sol_len = len(solution)
    early_count = int(moves_early * sol_len)
    late_count = int(moves_late * sol_len)
    safe_w = round(w * safe_rect)
    safe_h = round(h * safe_rect)
    unsolved_info = []
    mc = [row[:] for row in matrix]
    for mi in range(late_count):
        move = solution[mi]
        zp = find_zero(mc, w, h)
        mc = move_matrix(mc, move, zp, w, h)
        if mi > early_count:
            amt, arr = get_solve_elements_amount(mc, safe_w, safe_h)
            unsolved_info.append((amt, arr))
    if not unsolved_info:
        return []
    return min(unsolved_info, key=lambda x: x[0])[1]


def check_top_bottom(matrix, width, height, offset_w, offset_h, width_initial):
    new_h = math.ceil(height / 2) + offset_h
    solved_counter = 0
    for row in range(offset_h, new_h):
        for col in range(offset_w, width + offset_w):
            num = matrix[row][col]
            if num != 0 and (num - 1) // width_initial >= new_h:
                return False
            if number_is_solved(num, row, col, width_initial):
                solved_counter += 1
    return width * (new_h - offset_h) / 3 > solved_counter


def check_left_right(matrix, width, height, offset_w, offset_h, width_initial):
    new_w = math.ceil(width / 2) + offset_w
    solved_counter = 0
    for row in range(offset_h, height + offset_h):
        for col in range(offset_w, new_w):
            num = matrix[row][col]
            if num != 0 and (num - 1) % width_initial >= new_w:
                return False
            if number_is_solved(num, row, col, width_initial):
                solved_counter += 1
    return height * (new_w - offset_w) / 3 > solved_counter


def guess_grids(matrix, width, height, offset_w, offset_h, width_initial):
    if width < 6 and height < 6:
        return 0
    if height > 5 and check_top_bottom(matrix, width, height, offset_w, offset_h, width_initial):
        return 1
    if width > 5 and check_left_right(matrix, width, height, offset_w, offset_h, width_initial):
        return 2
    return 0


def grids_solved(matrix, width, height, offset_w, offset_h, grids_type, width_initial, cycled_numbers):
    if grids_type == 1:
        new_h = math.ceil(height / 2) + offset_h
        for row in range(offset_h, new_h):
            for col in range(offset_w, width + offset_w):
                num = matrix[row][col]
                if num != 0 and not number_is_solved(num, row, col, width_initial):
                    if num not in cycled_numbers:
                        return False
    if grids_type == 2:
        new_w = math.ceil(width / 2) + offset_w
        for row in range(offset_h, height + offset_h):
            for col in range(offset_w, new_w):
                num = matrix[row][col]
                if num != 0 and not number_is_solved(num, row, col, width_initial):
                    if num not in cycled_numbers:
                        return False
    return True


def get_grids_parts(matrix_before, solution, width, height):
    if width < 6 and height < 6:
        return None
    first = [row[:] for row in matrix_before]
    mc = [row[:] for row in matrix_before]
    for move in solution:
        zp = find_zero(mc, width, height)
        mc = move_matrix(mc, move, zp, width, height)
    return first, mc


def analyse_grids(matrix, solution, width_initial, height_initial, width, height, offset_w, offset_h, moves_offset, cycled_numbers):
    mc = [row[:] for row in matrix]
    for mi in range(len(solution)):
        move = solution[mi]
        zp = find_zero(mc, width_initial, height_initial)
        mc = move_matrix(mc, move, zp, width_initial, height_initial)
        gs = guess_grids(mc, width, height, offset_w, offset_h, width_initial)
        if gs != 0:
            grids_started = mi
            enable_gs = gs
            girds_unsolved_last = None
            matrix_before = [row[:] for row in mc]
            for gst_id in range(grids_started + 1, len(solution)):
                move2 = solution[gst_id]
                zp2 = find_zero(mc, width_initial, height_initial)
                mc = move_matrix(mc, move2, zp2, width_initial, height_initial)
                if not grids_solved(mc, width, height, offset_w, offset_h, enable_gs, width_initial, cycled_numbers):
                    girds_unsolved_last = gst_id
                else:
                    break
            if girds_unsolved_last is None:
                return {"enableGridsStatus": -1, "width": width, "height": height, "offsetW": offset_w, "offsetH": offset_h}
            grids_stopped = girds_unsolved_last + 1
            sol1 = solution[grids_started + 1: grids_stopped + 2]
            sol2 = solution[grids_stopped + 2:]
            parts = get_grids_parts(matrix_before, sol1, width_initial, height_initial)
            if parts is not None and enable_gs == 1:
                w1 = w2 = width
                ow1 = ow2 = offset_w
                h1 = math.ceil(height / 2)
                h2 = height - h1
                oh1 = offset_h
                oh2 = h1 + offset_h
                return {
                    "enableGridsStatus": enable_gs,
                    "gridsStarted": grids_started + moves_offset,
                    "gridsStopped": grids_stopped + moves_offset,
                    "width": width, "height": height,
                    "offsetW": offset_w, "offsetH": offset_h,
                    "nextLayerFirst": analyse_grids(parts[0], sol1, width_initial, height_initial, w1, h1, ow1, oh1, moves_offset + grids_started + 1, cycled_numbers),
                    "nextLayerSecond": analyse_grids(parts[1], sol2, width_initial, height_initial, w2, h2, ow2, oh2, moves_offset + grids_stopped + 1, cycled_numbers)
                }
            if parts is not None and enable_gs == 2:
                w1 = math.ceil(width / 2)
                w2 = width - w1
                ow1 = offset_w
                ow2 = w1 + offset_w
                h1 = h2 = height
                oh1 = oh2 = offset_h
                return {
                    "enableGridsStatus": enable_gs,
                    "gridsStarted": grids_started + moves_offset,
                    "gridsStopped": grids_stopped + moves_offset,
                    "width": width, "height": height,
                    "offsetW": offset_w, "offsetH": offset_h,
                    "nextLayerFirst": analyse_grids(parts[0], sol1, width_initial, height_initial, w1, h1, ow1, oh1, moves_offset + grids_started + 1, cycled_numbers),
                    "nextLayerSecond": analyse_grids(parts[1], sol2, width_initial, height_initial, w2, h2, ow2, oh2, moves_offset + grids_stopped + 1, cycled_numbers)
                }
            return {
                "enableGridsStatus": enable_gs,
                "gridsStarted": grids_started + moves_offset,
                "gridsStopped": grids_stopped + moves_offset,
                "width": width, "height": height,
                "offsetW": offset_w, "offsetH": offset_h,
                "nextLayerFirst": None,
                "nextLayerSecond": None
            }
    return {"enableGridsStatus": -1, "width": width, "height": height, "offsetW": offset_w, "offsetH": offset_h}


def analyse_grids_initial(matrix, solution, cycled_numbers):
    h = len(matrix)
    w = len(matrix[0])
    return analyse_grids(matrix, solution, w, h, w, h, 0, 0, 0, cycled_numbers)


def get_sizes_for_layer(type_n, layer):
    return {"type": type_n, "width": layer["width"], "height": layer["height"],
            "offsetW": layer["offsetW"], "offsetH": layer["offsetH"]}


def get_main_colors_by_level(cl):
    if cl["enableGridsStatus"] == -1:
        return [get_sizes_for_layer(CT_MAP['fringe'], cl)]
    return [
        get_sizes_for_layer(CT_MAP['grids1'], cl["nextLayerFirst"]),
        get_sizes_for_layer(CT_MAP['grids2'], cl["nextLayerSecond"])
    ]


def get_secondary_colors_by_level(cl):
    sec = []
    if cl["enableGridsStatus"] == -1:
        return sec
    fl = cl["nextLayerFirst"]
    sl = cl["nextLayerSecond"]
    if fl and fl.get("nextLayerFirst"):
        sec.append(get_sizes_for_layer(CT_MAP['grids1'], fl["nextLayerFirst"]))
        sec.append(get_sizes_for_layer(CT_MAP['grids2'], fl["nextLayerSecond"]))
    elif fl:
        sec.append(get_sizes_for_layer(CT_MAP['fringe'], fl))
    if sl and sl.get("nextLayerSecond"):
        sec.append(get_sizes_for_layer(CT_MAP['grids1'], sl["nextLayerFirst"]))
        sec.append(get_sizes_for_layer(CT_MAP['grids2'], sl["nextLayerSecond"]))
    elif sl:
        sec.append(get_sizes_for_layer(CT_MAP['fringe'], sl))
    return sec


def get_active_zone_by_level(cl):
    return get_sizes_for_layer(0, cl)


def get_data_by_level(cl):
    return {
        "mainColors": get_main_colors_by_level(cl),
        "secondaryColors": get_secondary_colors_by_level(cl),
        "activeZone": get_active_zone_by_level(cl)
    }


def generate_grids_stats(grids_data):
    levels = {}

    def traverse(node, nid):
        if node:
            levels[nid] = get_data_by_level(node)
            traverse(node.get("nextLayerFirst"), node.get("gridsStarted", 0))
            traverse(node.get("nextLayerSecond"), node.get("gridsStopped", 0))

    traverse(grids_data, 0)
    return levels


def get_grids_state(grids_states, move_index):
    keys = [k for k in grids_states.keys() if isinstance(k, (int, float))]
    valid = [k for k in keys if k <= move_index]
    if not valid:
        return grids_states[0]
    return grids_states[max(valid)]


def get_all_fringe_schemes(grid_states):
    schemes = {}
    for key, state in grid_states.items():
        for mc in state["mainColors"]:
            if len(state["mainColors"]) == 1:
                pair = f"{mc['width']}x{mc['height']}"
                if pair not in schemes:
                    schemes[pair] = get_fringe_colors_nxm(mc['width'], mc['height'])
        for sc in state["secondaryColors"]:
            if sc["type"] == CT_MAP["fringe"]:
                pair = f"{sc['width']}x{sc['height']}"
                if pair not in schemes:
                    schemes[pair] = get_fringe_colors_nxm(sc['width'], sc['height'])
    return schemes


RED_GRIDS = (200, 103, 103)
BLUE_GRIDS = (141, 179, 255)


# ─── Color Application (ported from replayGeneration.js) ────────────

def apply_color_any(colors_matrix, number, box_w, box_h, offset_w, offset_h, main_w, secondary=False):
    row = (number - offset_h * main_w - 1) // main_w
    col = (number - offset_w - 1) % main_w
    if 0 <= row < box_h and 0 <= col < box_w and row >= 0 and col >= 0:
        return colors_matrix[row][col]
    return None


def get_tile_colors(number, state, all_fringe_schemes, main_w):
    main_bg = None
    secondary_bg = None
    if number == 0:
        return main_bg, secondary_bg
    if len(state["mainColors"]) == 1:
        mc = state["mainColors"][0]
        key = f"{mc['width']}x{mc['height']}"
        scheme = all_fringe_schemes[key]
        main_bg = apply_color_any(scheme, number, mc['width'], mc['height'], mc['offsetW'], mc['offsetH'], main_w)
    else:
        for cs in state["mainColors"]:
            if cs["type"] == CT_MAP["grids1"]:
                c = RED_GRIDS
            elif cs["type"] == CT_MAP["grids2"]:
                c = BLUE_GRIDS
            else:
                continue
            scheme = get_mono_colors(c, cs["width"], cs["height"])
            if main_bg is None:
                main_bg = apply_color_any(scheme, number, cs["width"], cs["height"], cs["offsetW"], cs["offsetH"], main_w)
        for sc in state["secondaryColors"]:
            if secondary_bg is not None:
                break
            if sc["type"] == CT_MAP["fringe"]:
                key = f"{sc['width']}x{sc['height']}"
                scheme = all_fringe_schemes[key]
            elif sc["type"] == CT_MAP["grids1"]:
                scheme = get_mono_colors(RED_GRIDS, sc["width"], sc["height"])
            elif sc["type"] == CT_MAP["grids2"]:
                scheme = get_mono_colors(BLUE_GRIDS, sc["width"], sc["height"])
            else:
                continue
            if secondary_bg is None:
                secondary_bg = apply_color_any(scheme, number, sc["width"], sc["height"], sc["offsetW"], sc["offsetH"], main_w, secondary=True)
    return main_bg, secondary_bg


# ─── Font Loading ──────────────────────────────────────────────────

_font_cache = {}

def get_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold, mono)
    if key in _font_cache:
        return _font_cache[key]
    try:
        if mono:
            name = FONT_FAMILY_MONO_BOLD if bold else FONT_FAMILY_MONO
        else:
            name = FONT_FAMILY_BOLD if bold else FONT_FAMILY
        font = ImageFont.truetype(name, size)
    except Exception:
        try:
            font = ImageFont.truetype(FONT_FAMILY_MONO if mono else FONT_FAMILY, size)
        except Exception:
            font = ImageFont.load_default()
    _font_cache[key] = font
    return font


_number_texture_cache: dict = {}

def _get_number_texture(num: int, tile_size: int, font_size: int) -> Image.Image:
    key = (num, tile_size, font_size)
    cached = _number_texture_cache.get(key)
    if cached is not None:
        return cached
    im = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    text = str(num)
    tf = get_font(font_size)
    tb = draw.textbbox((0, 0), text, font=tf)
    tx = tile_size // 2 - (tb[0] + tb[2]) // 2
    ty = tile_size // 2 - (tb[1] + tb[3]) // 2
    draw.text((tx, ty), text, fill=(0, 0, 0, 255), font=tf)
    _number_texture_cache[key] = im
    return im


def format_time_str(ms: int) -> str:
    if ms < 1000:
        return f"0.{ms:03d}"
    total_sec = ms / 1000
    if total_sec < 60:
        return f"{total_sec:.3f}"
    minutes = int(total_sec // 60)
    sec = total_sec % 60
    return f"{minutes}:{sec:06.3f}"


# ─── Puzzle Rendering ──────────────────────────────────────────────

def pick_tile_size(width: int, height: int) -> int:
    max_dim = max(width, height)
    min_dim = min(width, height)
    if max_dim >= 30:
        return BASE_SIZE
    elif max_dim >= 20:
        return BASE_SIZE + 10
    elif max_dim >= 10:
        return BASE_SIZE + 14
    else:
        return min(BASE_SIZE * 40 // 11, max(BASE_SIZE * 28 // 11, BASE_SIZE * 140 // 11 // min_dim))


def draw_filled_rect(draw, bbox, color):
    x1, y1, x2, y2 = bbox
    draw.rectangle(bbox, fill=color)


def draw_multiline_text(draw, xy, text, fill, font, line_spacing=4):
    x, y = xy
    for line in text.split('\n'):
        draw.text((x, y), line, fill=fill, font=font)
        bbox = draw.textbbox((0, 0), line, font=font)
        y += (bbox[3] - bbox[1]) + line_spacing


def render_frame(
    matrix: List[List[int]],
    grid_state: dict,
    all_fringe_schemes: dict,
    tile_size: int,
    font_size: int,
    stats_data: dict,
    score_title_text: str,
    timer_text: str,
    is_movetimes_accurate: bool,
    total_moves: int,
    total_time_ms: int,
    total_tps: float,
    gpu_grid: Optional[Image.Image] = None,
    static_stats_base: Optional[Image.Image] = None,
    static_stats_layout: Optional[dict] = None,
) -> Image.Image:
    h = len(matrix)
    w = len(matrix[0])
    puzzle_w = w * tile_size
    puzzle_h = h * tile_size
    HEADER_H = 56

    canvas_w = (puzzle_w + STATS_PANEL_WIDTH + PADDING * 3 + 1) // 2 * 2

    panel_w_est = canvas_w - (PADDING + puzzle_w + PADDING) - PADDING
    stats_h = _compute_stats_full_height(
        panel_w_est,
        has_grid_stages=len(stats_data.get("grid_stages", [])) > 0
    )
    canvas_h = max(
        (HEADER_H + puzzle_h + PADDING * 3 + 1) // 2 * 2,
        HEADER_H + PADDING + stats_h + PADDING
    )
    canvas_h = (canvas_h + 1) // 2 * 2
    canvas = Image.new('RGB', (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # ─── Timer Bar (centered, compact) ──────────────────────────
    timer_bg_bbox = (PADDING, PADDING, canvas_w - PADDING, PADDING + HEADER_H)
    draw_filled_rect(draw, timer_bg_bbox, TIMER_BG)

    timer_img = _render_timer_text(timer_text)
    tw, th = timer_img.size
    tx = (canvas_w - tw) // 2
    ty = PADDING + (HEADER_H - th) // 2
    canvas.paste(timer_img, (tx, ty), timer_img)

    # ─── Puzzle Grid ──────────────────────────────────────────────
    grid_x = PADDING
    grid_y = PADDING + HEADER_H + PADDING

    if gpu_grid is not None:
        canvas.paste(gpu_grid, (grid_x, grid_y))
    else:
        for row_idx in range(h):
            for col_idx in range(w):
                num = matrix[row_idx][col_idx]
                sx = grid_x + col_idx * tile_size
                sy = grid_y + row_idx * tile_size
                sq_bbox = (sx, sy, sx + tile_size, sy + tile_size)

                main_bg, sec_bg = get_tile_colors(num, grid_state, all_fringe_schemes, w)
                bg_color = main_bg if main_bg else TILE_BG
                draw_filled_rect(draw, sq_bbox, bg_color)

                if tile_size > 1:
                    draw.rectangle(sq_bbox, outline=TILE_BORDER_COLOR, width=TILE_BORDER_WIDTH)

                if sec_bg:
                    bar_h = max(2, int(tile_size * 0.1))
                    bar_y = sy + tile_size - bar_h - max(2, int(tile_size * 0.06))
                    bar_bbox = (sx + max(2, int(tile_size * 0.1)), bar_y,
                                sx + tile_size - max(2, int(tile_size * 0.1)), bar_y + bar_h)
                    draw_filled_rect(draw, bar_bbox, sec_bg)
                    if tile_size > 1:
                        draw.rectangle(bar_bbox, outline=TILE_BORDER_COLOR, width=1)

                if num != 0:
                    tex = _get_number_texture(num, tile_size, font_size)
                    canvas.paste(tex, (sx, sy), tex)

    # ─── Stats Panel ──────────────────────────────────────────────
    panel_x = grid_x + puzzle_w + PADDING
    panel_y = grid_y
    panel_w = canvas_w - panel_x - PADDING
    panel_h = canvas_h - panel_y - PADDING

    if panel_w > 0 and panel_h > 0:
        panel_bbox = (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h)
        panel_img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
        pdraw = ImageDraw.Draw(panel_img)
        pdraw.rectangle(panel_bbox, fill=(*PANEL_BG, int(255 * PANEL_ALPHA)))
        panel_overlay = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
        pdraw2 = ImageDraw.Draw(panel_overlay)
        pdraw2.rectangle(panel_bbox, outline=CYAN, width=1)
        canvas = Image.alpha_composite(canvas.convert('RGBA'), panel_img)
        canvas = Image.alpha_composite(canvas, panel_overlay)

        if static_stats_base is not None and static_stats_layout is not None:
            stats_img = _apply_stats_dynamic(stats_data, panel_w, static_stats_base, static_stats_layout)
        else:
            stats_img = _render_stats_full(stats_data, is_movetimes_accurate, panel_w)
        canvas.paste(stats_img, (panel_x, panel_y), stats_img)

    return canvas.convert('RGB')


# ─── Timing Calculation (ported from replayGeneration.js) ──────────

def calculate_move_timings(solution: str, tps: float, width: int, height: int, speed_factor: float = 1.0):
    expanded = expand_solution(solution)
    sol_len = len(expanded)
    if sol_len <= 1:
        return [0], [0]

    repeated_width, repeated_height = get_repeated_lengths(expanded)
    longer_factor = 2
    k_w = width / longer_factor if speed_factor == 1.0 else speed_factor
    k_h = height / longer_factor if speed_factor == 1.0 else speed_factor

    base_delay_ms = 1000 * sol_len / (tps * (sol_len - 1))
    denom = (sol_len - 1 - repeated_width - repeated_height
             + repeated_width / k_w + repeated_height / k_h)
    delay_for_move = base_delay_ms * (sol_len - 1) / denom if denom != 0 else base_delay_ms
    short_delay_w = delay_for_move / k_w if k_w != 0 else delay_for_move
    short_delay_h = delay_for_move / k_h if k_h != 0 else delay_for_move

    delays = [delay_for_move]
    fake_times = [0.0]
    for mi in range(1, sol_len):
        if expanded[mi] == expanded[mi - 1]:
            if expanded[mi] in 'DU':
                delays.append(short_delay_h)
            else:
                delays.append(short_delay_w)
        else:
            delays.append(delay_for_move)
        fake_times.append(fake_times[mi - 1] + delays[mi])

    return delays, fake_times



# ─── Frame Generation ──────────────────────────────────────────────

def _render_one_frame(params: dict) -> Image.Image:
    params.pop("colors", None)
    params.pop("timer_img", None)
    params.pop("stats_img", None)
    return render_frame(**params)


def _render_frame_batch(params_list: List[dict]) -> List[Image.Image]:
    return [_render_one_frame(params) for params in params_list]


def _calc_render_batch_size(num_needed: int, workers: int) -> int:
    return max(1, min(20, num_needed // (workers * 2) + 1))


def _make_stats_text(stats_data, is_movetimes_accurate, panel_w):
    """Full render of stats panel (CPU path)."""
    return _render_stats_full(stats_data, is_movetimes_accurate, panel_w)


def _stats_layout_info(panel_w):
    """Compute layout constants for the stats panel."""
    inner_w = panel_w - 20
    px = 10
    data_font = get_font(20, mono=True)
    hf = get_font(24, bold=True)
    gs_hf = get_font(18, bold=True)
    gs_lf = get_font(13, mono=True)
    acc_font = get_font(16, mono=True)

    data_line_h = data_font.getbbox("Xy")[3] - data_font.getbbox("Xy")[1] + 4
    row_h = data_line_h + 8

    labels = ["Time (total):", "Moves (total):", "TPS (total):", "Cubic est:",
              "Predicted moves:", "MD (total):", "MD (current):",
              "M/MD (total):", "M/MD (current):"]

    return {
        "panel_w": panel_w, "inner_w": inner_w, "px": px,
        "data_font": data_font, "header_font": hf,
        "gs_header_font": gs_hf, "gs_data_font": gs_lf,
        "acc_font": acc_font, "row_h": row_h, "labels": labels,
    }


def _render_stats_full(stats_data, is_movetimes_accurate, panel_w):
    """Render full stats panel - left-aligned labels, right-aligned values."""
    li = _stats_layout_info(panel_w)
    inner_w = li["inner_w"]; px = li["px"]
    data_font = li["data_font"]; hf = li["header_font"]
    gs_hf = li["gs_header_font"]; gs_lf = li["gs_data_font"]
    acc_font = li["acc_font"]; row_h = li["row_h"]

    lines = []
    def add(x, y, text, fill, font):
        lines.append((x, y, text, fill, font))

    def lv_line(label, value, font, color):
        nonlocal y
        add(px, y, label, color, font)
        vb = font.getbbox(value)
        vw = vb[2] - vb[0]
        add(px + inner_w - vw, y, value, color, font)
        y += row_h

    y = 10

    # "Stats" header (left-aligned)
    hb = hf.getbbox("Stats")
    add(px, y, "Stats", CYAN, hf)
    y += (hb[3] - hb[1]) + 16

    # Row 0: Time (total)
    ce = stats_data.get("cubic_estimate")
    lv_line("Time (total): ", stats_data.get('time_all', '0.000'), data_font, WHITE)

    # Row 1: Moves (total)
    lv_line("Moves (total): ", stats_data.get('moves_all', '0'), data_font, WHITE)

    # Row 2: TPS (total)
    lv_line("TPS (total): ", stats_data.get('tps_all', '0.000'), data_font, WHITE)

    # Row 3: Cubic est
    lv_line("Cubic est: ", ce if ce else '---', data_font, WHITE)

    # Row 4: Predicted moves
    lv_line("Predicted moves: ", stats_data.get('predicted_moves', ''), data_font, CYAN)

    # Row 5: MD (total)
    lv_line("MD (total): ", stats_data.get('md_all', '0'), data_font, WHITE)

    # Row 6: MD (current)
    lv_line("MD (current): ", stats_data.get('md_cur', '0'), data_font, CYAN)

    # Row 7: M/MD (total)
    lv_line("M/MD (total): ", stats_data.get('mmd_all', '0.000'), data_font, WHITE)

    # Row 8: M/MD (current)
    lv_line("M/MD (current): ", stats_data.get('mmd_cur', '0.000'), data_font, CYAN)

    y += 4

    # Movetimes accuracy (left-aligned)
    acc_text = "Movetimes accurate" if is_movetimes_accurate else "NOT movetimes accurate"
    acc_color = ACCURATE_COLOR if is_movetimes_accurate else INACCURATE_COLOR
    ab = acc_font.getbbox(acc_text)
    add(px, y, acc_text, acc_color, acc_font)
    y += (ab[3] - ab[1]) + 6

    # Grid stages
    stages = stats_data.get("grid_stages", [])
    cur_stage = stats_data.get("grid_current", 0)
    if stages:
        gb = gs_hf.getbbox("Grid stages")
        add(px, y, "Grid stages", CYAN, gs_hf)
        y += (gb[3] - gb[1]) + 14
        raw_lines = []
        for st in stages:
            if st["cum_time"] > 0:
                cum_s = format_time_str(st['cum_time'])
                split_s = format_time_str(st['split_time'])
                mvtps_s = f"({st['split_moves']}/{st['split_tps']:.1f})"
            else:
                cum_s = str(st['cum_moves'])
                split_s = f"(+{st['split_moves']})"
                mvtps_s = ""
            raw_lines.append((cum_s, split_s, mvtps_s, st['label']))
        if raw_lines:
            w1 = max(len(l[0]) for l in raw_lines)
            w2 = max(len(l[1]) for l in raw_lines)
            w3 = max(len(l[2]) for l in raw_lines) if any(l[2] for l in raw_lines) else 0
            w4 = max(len(l[3]) for l in raw_lines)
        for i, (cum_s, split_s, mvtps_s, label) in enumerate(raw_lines):
            if '.' in cum_s:
                line = f"{cum_s:>{w1}} | {split_s:>{w2}} {mvtps_s:<{w3}} | {label:<{w4}}"
            else:
                line = f"{cum_s:>{w1}} | {split_s:<{w2}}  | {label:<{w4}}"
            color = CYAN if i == cur_stage else WHITE
            add(px, y, line, color, gs_lf)
            y += (gs_lf.getbbox(line)[3] - gs_lf.getbbox(line)[1]) + 4

    total_h = y + 30
    im = Image.new("RGBA", (panel_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    for x, y, text, fill, font in lines:
        draw.text((x, y), text, fill=(*fill, 255), font=font)
    return im


def _compute_stats_full_height(panel_w, has_grid_stages=True):
    """Estimate total height of the rendered stats panel."""
    li = _stats_layout_info(panel_w)
    data_font = li["data_font"]; hf = li["header_font"]
    gs_hf = li["gs_header_font"]; gs_lf = li["gs_data_font"]
    acc_font = li["acc_font"]; row_h = li["row_h"]

    y = 10
    y += (hf.getbbox("Stats")[3] - hf.getbbox("Stats")[1]) + 16
    y += 9 * row_h
    y += 4
    y += (acc_font.getbbox("Movetimes accurate")[3] - acc_font.getbbox("Movetimes accurate")[1]) + 6
    if has_grid_stages:
        y += (gs_hf.getbbox("Grid stages")[3] - gs_hf.getbbox("Grid stages")[1]) + 14
        y += 4 * ((gs_lf.getbbox("Xy")[3] - gs_lf.getbbox("Xy")[1]) + 4)
    y += 30
    return y


def _make_stats_static_base(panel_w, stats_data, is_movetimes_accurate, grid_stages_list):
    """Render static parts of stats panel (left-aligned labels, right-aligned values). Returns (image, layout_info)."""
    li = _stats_layout_info(panel_w)
    inner_w = li["inner_w"]; px = li["px"]
    data_font = li["data_font"]; hf = li["header_font"]
    gs_hf = li["gs_header_font"]; gs_lf = li["gs_data_font"]
    acc_font = li["acc_font"]; row_h = li["row_h"]

    lines = []
    def add(x, y, text, fill, font):
        lines.append((x, y, text, fill, font))

    def lv_line(label, value, font, color):
        nonlocal y
        add(px, y, label, color, font)
        vb = font.getbbox(value)
        vw = vb[2] - vb[0]
        add(px + inner_w - vw, y, value, color, font)
        y += row_h

    y = 10

    # "Stats" header (left-aligned)
    hb = hf.getbbox("Stats")
    add(px, y, "Stats", CYAN, hf)
    y += (hb[3] - hb[1]) + 16

    # Row 0: Time (total) [static, WHITE]
    ce = stats_data.get("cubic_estimate")
    lv_line("Time (total): ", stats_data.get('time_all', '0.000'), data_font, WHITE)

    # Row 1: Moves (total) [static, WHITE]
    lv_line("Moves (total): ", stats_data.get('moves_all', '0'), data_font, WHITE)

    # Row 2: TPS (total) [static, WHITE]
    lv_line("TPS (total): ", stats_data.get('tps_all', '0.000'), data_font, WHITE)

    # Row 3: Cubic est [static, WHITE]
    lv_line("Cubic est: ", ce if ce else '---', data_font, WHITE)

    # Row 4: Predicted moves [dynamic label in static base]
    add(px, y, "Predicted moves: ", CYAN, data_font)
    y_predicted = y
    y += row_h

    # Row 5: MD (total) [static, WHITE]
    lv_line("MD (total): ", stats_data.get('md_all', '0'), data_font, WHITE)

    # Row 6: MD (current) [dynamic label in static base]
    add(px, y, "MD (current): ", CYAN, data_font)
    y_md_cur = y
    y += row_h

    # Row 7: M/MD (total) [static, WHITE]
    lv_line("M/MD (total): ", stats_data.get('mmd_all', '0.000'), data_font, WHITE)

    # Row 8: M/MD (current) [dynamic label in static base]
    add(px, y, "M/MD (current): ", CYAN, data_font)
    y_mmd_cur = y
    y += row_h

    y += 4

    # Movetimes accuracy (left-aligned)
    acc_text = "Movetimes accurate" if is_movetimes_accurate else "NOT movetimes accurate"
    acc_color = ACCURATE_COLOR if is_movetimes_accurate else INACCURATE_COLOR
    ab = acc_font.getbbox(acc_text)
    add(px, y, acc_text, acc_color, acc_font)
    y += (ab[3] - ab[1]) + 6

    # Grid stages (all white in static base)
    stage_y_positions = []
    if grid_stages_list:
        gb = gs_hf.getbbox("Grid stages")
        add(px, y, "Grid stages", CYAN, gs_hf)
        y += (gb[3] - gb[1]) + 14
        raw_lines = []
        for st in grid_stages_list:
            if st["cum_time"] > 0:
                cum_s = format_time_str(st['cum_time'])
                split_s = format_time_str(st['split_time'])
                mvtps_s = f"({st['split_moves']}/{st['split_tps']:.1f})"
            else:
                cum_s = str(st['cum_moves'])
                split_s = f"(+{st['split_moves']})"
                mvtps_s = ""
            raw_lines.append((cum_s, split_s, mvtps_s, st['label']))
        if raw_lines:
            w1 = max(len(l[0]) for l in raw_lines)
            w2 = max(len(l[1]) for l in raw_lines)
            w3 = max(len(l[2]) for l in raw_lines) if any(l[2] for l in raw_lines) else 0
            w4 = max(len(l[3]) for l in raw_lines)
        for i, (cum_s, split_s, mvtps_s, label) in enumerate(raw_lines):
            if '.' in cum_s:
                line = f"{cum_s:>{w1}} | {split_s:>{w2}} {mvtps_s:<{w3}} | {label:<{w4}}"
            else:
                line = f"{cum_s:>{w1}} | {split_s:<{w2}}  | {label:<{w4}}"
            add(px, y, line, WHITE, gs_lf)
            stage_y_positions.append(y)
            y += (gs_lf.getbbox(line)[3] - gs_lf.getbbox(line)[1]) + 4

    total_h = y + 30
    im = Image.new("RGBA", (panel_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    for x, y, text, fill, font in lines:
        draw.text((x, y), text, fill=(*fill, 255), font=font)

    layout_info = {
        "px": px, "inner_w": inner_w,
        "data_font": data_font, "row_h": row_h,
        "y_predicted": y_predicted,
        "y_md_cur": y_md_cur,
        "y_mmd_cur": y_mmd_cur,
        "stage_y_positions": stage_y_positions,
        "grid_stages_list": grid_stages_list,
        "gs_lf": gs_lf,
    }
    return im, layout_info


def _apply_stats_dynamic(stats_data, panel_w, static_base, layout_info):
    """Overlay dynamic values and stage highlight onto a static base (labels already drawn in base)."""
    overlay = Image.new("RGBA", static_base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    px = layout_info["px"]
    inner_w = layout_info["inner_w"]
    data_font = layout_info["data_font"]
    gs_lf = layout_info["gs_lf"]

    def draw_value(value, fill, y_pos):
        vb = data_font.getbbox(value)
        vw = vb[2] - vb[0]
        draw.text((px + inner_w - vw, y_pos), value, fill=(*fill, 255), font=data_font)

    # Predicted moves (dynamic value only)
    draw_value(stats_data.get("predicted_moves", ""), CYAN, layout_info["y_predicted"])

    # MD (current) (dynamic value only)
    draw_value(stats_data.get("md_cur", "0"), CYAN, layout_info["y_md_cur"])

    # M/MD (current) (dynamic value only)
    draw_value(stats_data.get("mmd_cur", "0.000"), CYAN, layout_info["y_mmd_cur"])

    # Current grid stage highlight (CYAN)
    stages = stats_data.get("grid_stages", [])
    cur_stage = stats_data.get("grid_current", 0)
    stage_y_positions = layout_info["stage_y_positions"]
    if stages and cur_stage < len(stage_y_positions):
        raw_lines = []
        for st in stages:
            if st["cum_time"] > 0:
                cum_s = format_time_str(st['cum_time'])
                split_s = format_time_str(st['split_time'])
                mvtps_s = f"({st['split_moves']}/{st['split_tps']:.1f})"
            else:
                cum_s = str(st['cum_moves'])
                split_s = f"(+{st['split_moves']})"
                mvtps_s = ""
            raw_lines.append((cum_s, split_s, mvtps_s, st['label']))
        if raw_lines:
            w1 = max(len(l[0]) for l in raw_lines)
            w2 = max(len(l[1]) for l in raw_lines)
            w3 = max(len(l[2]) for l in raw_lines) if any(l[2] for l in raw_lines) else 0
            w4 = max(len(l[3]) for l in raw_lines)
        cum_s, split_s, mvtps_s, label = raw_lines[cur_stage]
        if '.' in cum_s:
            line = f"{cum_s:>{w1}} | {split_s:>{w2}} {mvtps_s:<{w3}} | {label:<{w4}}"
        else:
            line = f"{cum_s:>{w1}} | {split_s:<{w2}}  | {label:<{w4}}"
        draw.text((px, stage_y_positions[cur_stage]),
                  line, fill=(*CYAN, 255), font=gs_lf)

    result = static_base.copy()
    result = Image.alpha_composite(result, overlay)
    return result


_nvenc_cache: Optional[bool] = None

def _close_pipe(proc: subprocess.Popen) -> None:
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        proc.wait()


def _nvenc_available() -> bool:
    global _nvenc_cache
    if _nvenc_cache is not None:
        return _nvenc_cache
    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'],
            capture_output=True, text=True, timeout=10,
        )
        _nvenc_cache = 'h264_nvenc' in result.stdout
    except Exception:
        _nvenc_cache = False
    log.info(f"_nvenc_available: {_nvenc_cache}")
    return _nvenc_cache


def _create_ffmpeg_pipe(output_path: str, width: int, height: int, fps: int = 60, compression: int = 18):
    """Spawn ffmpeg with libx264 (software) reading rawvideo from stdin."""
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', str(compression),
        '-profile:v', 'high',
        '-level', '4.1',
        '-pix_fmt', 'yuv420p',
        '-fps_mode', 'cfr',
        '-movflags', '+faststart',
        output_path,
    ]
    log.info(f"_create_ffmpeg_pipe (libx264): cmd={' '.join(cmd)}")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _create_ffmpeg_pipe_nvenc(output_path: str, width: int, height: int, fps: int = 60, compression: int = 18):
    """Spawn ffmpeg with h264_nvenc (hardware) reading rawvideo from stdin."""
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'h264_nvenc',
        '-preset', 'p7',
        '-rc', 'vbr',
        '-cq', str(compression),
        '-profile:v', 'high',
        '-pix_fmt', 'yuv420p',
        '-fps_mode', 'cfr',
        '-movflags', '+faststart',
        output_path,
    ]
    log.info(f"_create_ffmpeg_pipe_nvenc: cmd={' '.join(cmd)}")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def generate_frames(
    matrix: List[List[int]],
    solution: str,
    tps: float,
    all_fringe_schemes: dict,
    grid_states: dict,
    fake_times: List[float],
    delays: List[float],
    is_movetimes_accurate: bool,
    score_title_text: str = "",
    custom_move_times: Optional[List[float]] = None,
    cumulative_data: Optional[dict] = None,
    progress_callback=None,
    overlay_progress_callback=None,
    quality: float = 1.0,
    parallel: bool = True,
    use_gpu: bool = True,
    cancel_check=None,
    output_path: str = None,
    fps: int = 60,
    compression: int = 18,
    shared_pool: Optional[ProcessPoolExecutor] = None,
    gpu_renderer: Optional[GPURenderer] = None,
) -> Tuple[List[Image.Image], List[int]]:
    quality = quality + 1.0
    expanded = expand_solution(solution)
    sol_len = len(expanded)
    h = len(matrix)
    w = len(matrix[0])
    _t_stage1 = time_module.time()
    log.info("====== STAGE 1: DATA PREP ======")
    log.info(f"generate_frames: {w}x{h}, sol_len={sol_len}, quality={quality}, fps={fps}, use_gpu={use_gpu}, output_path={output_path}")

    grid_keys = sorted([k for k in grid_states.keys() if isinstance(k, (int, float))])
    filtered_stages = [0]
    for k in grid_keys:
        if k == 0:
            continue
        az = grid_states[k]["activeZone"]
        if az["width"] + 1 >= w / 2 and az["height"] + 1 >= h / 2:
            filtered_stages.append(k)
    filtered_stages.append(sol_len)

    _sorted_grid_keys = sorted([k for k in grid_states.keys() if isinstance(k, (int, float))])
    _sorted_grid_keys.append(sol_len + 1)

    def _fast_grid_state(grid_states, move_index):
        idx = bisect.bisect_left(_sorted_grid_keys, move_index + 1) - 1
        return grid_states[_sorted_grid_keys[idx]] if idx >= 0 else grid_states[0]

    grid_stages_list = []
    last_moves = 0
    last_time = 0
    n_stages = len(filtered_stages) - 1
    if n_stages == 3:
        labels = ["S", "F1", "F2"]
    elif n_stages == 7:
        labels = ["s1", "ss1", "f1", "f2", "ss2", "f3", "f4"]
    else:
        labels = [f"S{i+1}" for i in range(n_stages)]
    for i, s in enumerate(filtered_stages):
        if i == 0:
            continue
        if s == sol_len:
            cum_moves = sol_len
            split_moves = cum_moves - last_moves
        else:
            cum_moves = s + 1
            split_moves = cum_moves - last_moves
        stage_time = 0
        cum_stage_time = 0
        if custom_move_times and len(custom_move_times) > s - 1 and s > 0:
            if s == sol_len:
                cum_stage_time = custom_move_times[-1]
                stage_time = custom_move_times[-1] - last_time
            else:
                cum_stage_time = custom_move_times[s - 1]
                stage_time = custom_move_times[s - 1] - last_time
            last_time = custom_move_times[s - 1] if s < sol_len else custom_move_times[-1]
        split_tps = split_moves * 1000 / stage_time if stage_time > 0 else 0
        grid_stages_list.append({
            "cum_moves": cum_moves,
            "split_moves": split_moves,
            "cum_time": round(cum_stage_time),
            "split_time": round(stage_time),
            "split_tps": split_tps,
            "label": labels[i - 1] if i - 1 < len(labels) else f"S{i}",
        })
        last_moves = cum_moves

    log.info(f"  grid_stages: n_stages={n_stages}, filtered_stages={filtered_stages}")
    raw_tile = pick_tile_size(w, h)
    tile_size = max(raw_tile, int(raw_tile * quality))
    font_size = max(11, tile_size // 2)
    log.info(f"  tile_size={tile_size}, raw_tile={raw_tile}, font_size={font_size}")

    def _update_manhattan_distance(md: int, matrix, move, zero_pos, w: int, h: int) -> int:
        dr, dc = {'R': (0, -1), 'L': (0, 1), 'U': (1, 0), 'D': (-1, 0)}[move]
        nr, nc = zero_pos[0] + dr, zero_pos[1] + dc
        moved_val = matrix[zero_pos[0]][zero_pos[1]]
        old_md = abs((moved_val - 1) // w - nr) + abs((moved_val - 1) % w - nc)
        new_md = abs((moved_val - 1) // w - zero_pos[0]) + abs((moved_val - 1) % w - zero_pos[1])
        return md - old_md + new_md

    mc = [row[:] for row in matrix]
    all_md = calculate_manhattan_distance(mc)
    current_md = all_md
    zp = find_zero(mc, w, h)

    total_time_ms = fake_times[-1] if fake_times else 0
    total_tps = tps

    if not custom_move_times:
        custom_move_times = []

    # ── Stage 1: determine which states are actually needed ──
    frame_time_ms = 1000.0 / fps
    preview_ms = 500.0
    final_ms = 1000.0

    if custom_move_times and len(custom_move_times) == sol_len:
        move_times = custom_move_times
    else:
        move_times = []
        cum = 0.0
        for d in delays:
            cum += d
            move_times.append(cum)

    starts = [0.0]
    for mt in move_times:
        starts.append(preview_ms + mt)
    starts.append(preview_ms + move_times[-1] + final_ms)

    total_frames = max(1, int(round(starts[-1] / frame_time_ms)))

    frame_state = []
    for j in range(total_frames):
        t = j * frame_time_ms
        idx = bisect.bisect_right(starts, t) - 1
        frame_state.append(max(0, min(idx, sol_len)))

    states_needed = sorted(set(frame_state))
    states_needed_set = set(states_needed)
    log.info(f"  frame_state: total_frames={total_frames}, unique_states={len(states_needed)}, frame_time_ms={frame_time_ms:.3f}")

    # Optional GPU acceleration for tile grid rendering
    puzzle_w_est = w * tile_size
    canvas_w_est = (puzzle_w_est + STATS_PANEL_WIDTH + PADDING * 3 + 1) // 2 * 2
    panel_w_est = canvas_w_est - (PADDING + puzzle_w_est + PADDING) - PADDING
    stats_h_est = _compute_stats_full_height(panel_w_est, has_grid_stages=True)
    min_canvas_h = HEADER_H + PADDING + stats_h_est + PADDING
    if gpu_renderer is not None:
        gpu = gpu_renderer
    else:
        gpu = GPURenderer(w, h, raw_tile, quality, min_canvas_h=min_canvas_h)
    use_gpu = use_gpu and gpu.available
    log.info(f"  canvas={gpu.canvas_w}x{gpu.canvas_h}, GPU available={gpu.available}, use_gpu={use_gpu}, max_batch_pixels={getattr(gpu, '_max_pixels_per_batch', '?')}")
    if use_gpu:
        import torch as _torch_snapshot
        log.info(f"  Python={sys.version.split()[0]}, torch={_torch_snapshot.__version__}, CUDA={_torch_snapshot.version.cuda}")

    # ── Build tile color cache for every grid state lookup may need ──
    _tile_color_cache = {}
    for key in _sorted_grid_keys:
        if key == sol_len + 1:
            continue
        cache_state = grid_states[key]
        cache_key = id(cache_state)
        if cache_key not in _tile_color_cache:
            color_matrix = []
            for num in range(1, h * w + 1):
                main_bg, sec_bg = get_tile_colors(num, cache_state, all_fringe_schemes, w)
                color_matrix.append((main_bg or TILE_BG, sec_bg))
            _tile_color_cache[cache_key] = color_matrix

    # ── Stage 2: precompute data only for states that will be rendered ──
    frame_params = [None] * (sol_len + 1)
    for frame_idx in range(sol_len + 1):
        if frame_idx in states_needed_set:
            mc_snapshot = [row[:] for row in mc]
            state = _fast_grid_state(grid_states, frame_idx - 1) if frame_idx > 0 else grid_states[0]
            current_moves = frame_idx

            _t_tc_start = time_module.time()
            cached_colors = _tile_color_cache.get(id(state))
            if cached_colors is None:
                cached_colors = _tile_color_cache.get(id(grid_states[0]))
            tile_colors = []
            for row_idx in range(h):
                row_colors = []
                for col_idx in range(w):
                    num = mc[row_idx][col_idx]
                    if num == 0:
                        row_colors.append((TILE_BG, None))
                    else:
                        row_colors.append(cached_colors[num - 1])
                tile_colors.append(row_colors)
            if frame_idx % 500 == 0 and frame_idx > 0:
                log.info(f"    state {frame_idx}: tile_colors took {time_module.time() - _t_tc_start:.3f}s")

            if frame_idx == 0:
                cur_time_ms = 0
                cur_md = all_md
                moved_md = 0
                cur_tps_val = 0.0
            else:
                if custom_move_times and len(custom_move_times) > frame_idx - 1:
                    cur_time_ms = custom_move_times[frame_idx - 1]
                else:
                    cur_time_ms = fake_times[frame_idx - 1] if frame_idx - 1 < len(fake_times) else 0
                cur_md = current_md
                moved_md = all_md - cur_md
                cur_tps_val = current_moves * 1000 / cur_time_ms if cur_time_ms > 0 else 0

            cur_mmd = "High" if moved_md <= 0 else (current_moves / moved_md)
            all_mmd = sol_len / all_md if all_md > 0 else 0
            mmd_display = cur_mmd if isinstance(cur_mmd, str) else f"{cur_mmd:.3f}"

            if cumulative_data:
                base_time = cur_time_ms
                base_moves = current_moves
                if cumulative_data["time"] > 0:
                    cur_time_display = format_time_str(round(base_time + cumulative_data["time"]))
                    tps_display = ((cumulative_data["moves"] + base_moves) * 1000 / (base_time + cumulative_data["time"]))
                    cur_tps_display = f"{tps_display:.3f}"
                else:
                    cur_time_display = format_time_str(round(base_time))
                    cur_tps_display = f"{cur_tps_val:.3f}"
                moves_display = str(base_moves + cumulative_data["moves"])
            else:
                cur_time_display = format_time_str(round(cur_time_ms))
                cur_tps_display = f"{cur_tps_val:.3f}".replace("inf", "Inf.")
                moves_display = str(current_moves)

            timer_text = f"{cur_time_display} ({moves_display} / {cur_tps_display})"

            if isinstance(cur_mmd, str):
                predicted_moves = cur_mmd
            else:
                predicted_moves = f"{round(cur_mmd * all_md)}"

            stats_data = {
                "time_all": format_time_str(round(total_time_ms)),
                "moves_all": str(sol_len),
                "md_all": str(all_md),
                "md_cur": str(moved_md),
                "mmd_all": f"{all_mmd:.3f}",
                "mmd_cur": mmd_display,
                "tps_all": f"{total_tps:.3f}",
                "predicted_moves": predicted_moves,
                "cubic_estimate": None,
            }

            move_idx = frame_idx - 1 if frame_idx > 0 else 0
            cur_stage_idx = max(0, sum(1 for s in filtered_stages if s <= move_idx) - 1)
            stats_data["grid_stages"] = grid_stages_list
            stats_data["grid_current"] = cur_stage_idx

            if w * h > 99:
                from replay_generator import get_cubic_estimate
                ce = get_cubic_estimate(round(total_time_ms), w, h)
                stats_data["cubic_estimate"] = format_time_str(ce)

            frame_params[frame_idx] = dict(
                matrix=mc_snapshot,
                grid_state=state,
                all_fringe_schemes=all_fringe_schemes,
                tile_size=tile_size,
                font_size=font_size,
                stats_data=stats_data,
                score_title_text=score_title_text,
                timer_text=timer_text,
                is_movetimes_accurate=is_movetimes_accurate,
                total_moves=sol_len,
                total_time_ms=round(total_time_ms),
                total_tps=total_tps,
                colors=tile_colors,
            )

        if frame_idx < sol_len:
            move = expanded[frame_idx]
            dr, dc = {'R': (0, -1), 'L': (0, 1), 'U': (1, 0), 'D': (-1, 0)}[move]
            new_zp = (zp[0] + dr, zp[1] + dc)
            mc = move_matrix(mc, move, zp, w, h)
            current_md = _update_manhattan_distance(current_md, mc, move, zp, w, h)
            zp = new_zp
            if frame_idx % 1000 == 0 and frame_idx > 0:
                log.info(f"  mutation for move {frame_idx // 1000}k: {time_module.time() - _t_stage1:.3f}s total so far")

    _t_fp = time_module.time()
    log.info(f"  frame_params loop: {_t_fp - _t_stage1:.3f}s total, {len(states_needed)} states built, {sol_len + 1} iterations")

    log.info(f"  render decision: use_gpu={use_gpu}, total_video_frames={len(frame_state)}, unique_states={len(states_needed)} ({len(states_needed)*100//len(frame_state) if frame_state else 0}%%)")

    # Pre-compute static stats base + layout for static/dynamic split (both paths)
    first_needed = states_needed[0] if states_needed else 0
    first_stats = frame_params[first_needed]["stats_data"]
    first_is_accurate = frame_params[first_needed]["is_movetimes_accurate"]
    grid_stages_list = first_stats.get("grid_stages", [])
    puzzle_w_pre = w * tile_size
    canvas_w_pre = (puzzle_w_pre + STATS_PANEL_WIDTH + PADDING * 3 + 1) // 2 * 2
    panel_x_pre = PADDING + puzzle_w_pre + PADDING
    panel_w_pre = canvas_w_pre - panel_x_pre - PADDING
    static_base, static_layout = _make_stats_static_base(panel_w_pre, first_stats, first_is_accurate, grid_stages_list)
    for fp_idx in states_needed:
        frame_params[fp_idx]["static_stats_base"] = static_base
        frame_params[fp_idx]["static_stats_layout"] = static_layout

    log.info(f"====== STAGE 1 DONE: {time_module.time() - _t_stage1:.1f}s ======")
    # ── GPU path: render unique states, pipe via frame mapping ──
    if use_gpu and len(frame_params) > 1:
        puzzle_w = w * tile_size
        puzzle_h = h * tile_size
        canvas_w = gpu.canvas_w
        canvas_h = gpu.canvas_h
        panel_x = PADDING + puzzle_w + PADDING
        panel_w_val = canvas_w - panel_x - PADDING

        extra_overlay_args = {
            "panel_w_val": panel_w_val,
            "static_base": static_base,
            "static_layout": static_layout,
        }

        # Pre-compute how many video frames each puzzle state spans
        state_to_count = {}
        for state_idx in frame_state:
            state_to_count[state_idx] = state_to_count.get(state_idx, 0) + 1
        log.info(f"  state_to_count: {len(state_to_count)} unique states, counts={list(state_to_count.values())[:20]}...")

        # Open ffmpeg pipe — prefer NVENC on GPU path, fall back to libx264
        encoder_name = "NVENC" if _nvenc_available() else "libx264"
        log.info(f"  OPENING FFMPEG PIPE: output={output_path}, canvas={canvas_w}x{canvas_h}, fps={fps}, compression={compression}, encoder={encoder_name}")
        if _nvenc_available():
            enc_proc = _create_ffmpeg_pipe_nvenc(output_path, canvas_w, canvas_h, fps=fps, compression=compression)
        else:
            enc_proc = _create_ffmpeg_pipe(output_path, canvas_w, canvas_h, fps=fps, compression=compression)
        unique_params = [frame_params[i] for i in states_needed]
        _t_stage3 = time_module.time()
        log.info("====== STAGE 3: GPU RENDER ======")
        log.info(f"  GPU RENDER START: {len(unique_params)} unique frames to render")
        log_ram("before GPU render")

        def handler(img, idx_in_unique, total):
            count = state_to_count[states_needed[idx_in_unique]]
            data = img.tobytes()
            for _ in range(count):
                enc_proc.stdin.write(data)

        try:
            try:
                gpu.render_frames(
                    unique_params,
                    progress_callback=lambda cur, tot, **kw: progress_callback(cur, tot, _use_gpu=True, **kw) if progress_callback else None,
                    cancel_check=cancel_check,
                    frame_handler=handler,
                    overlay_render_data=extra_overlay_args,
                )
            finally:
                if gpu_renderer is None:
                    gpu.cleanup()
        finally:
            _close_pipe(enc_proc)

        log.info(f"====== STAGE 3 DONE: {time_module.time() - _t_stage3:.1f}s ======")

        log.info(f"  GPU PATH COMPLETE: returning {len(frame_state)} frame_state entries")
        log_ram("after GPU render")

        return [], frame_state

    # ── CPU path: render only states_needed, pipe via frame mapping ──
    log.info(f"  CPU PATH: {len(states_needed)} unique states to render")
    log_ram("CPU: before font load")
    _font_start = time_module.time()
    get_font(font_size)
    get_font(24, bold=True)
    get_font(20, mono=True)
    get_font(18, bold=True)
    get_font(14, mono=True)
    get_font(16, mono=True)
    get_font(9)
    get_font(36, bold=True, mono=True)
    log.info(f"  fonts loaded: took {time_module.time() - _font_start:.3f}s")
    log_ram("CPU: after font load")

    state_images = [None] * (sol_len + 1)
    num_needed = len(states_needed)
    log_ram("CPU: before render")

    RENDER_W = 50

    if parallel and num_needed > 1:
        workers = min(os.cpu_count() or 4, num_needed)
        done = 0
        pool = shared_pool if shared_pool is not None else ProcessPoolExecutor(max_workers=workers)
        try:
            batch_size = _calc_render_batch_size(num_needed, workers)
            fut_to_indices = {}
            for i in range(0, num_needed, batch_size):
                batch_indices = states_needed[i:i + batch_size]
                batch_params = [frame_params[idx] for idx in batch_indices]
                fut = pool.submit(_render_frame_batch, batch_params)
                fut_to_indices[fut] = batch_indices

            remaining = set(fut_to_indices.keys())
            while remaining:
                if cancel_check and cancel_check():
                    raise CancelError()
                done_set, _ = wait(remaining, timeout=0.2, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    batch_indices = fut_to_indices[fut]
                    batch_results = fut.result()
                    for idx, img in zip(batch_indices, batch_results):
                        state_images[idx] = img
                        done += 1
                        if progress_callback:
                            cur = done * RENDER_W // num_needed
                            progress_callback(cur, 100, _desc="Render" if done == 1 else None)
                    remaining.remove(fut)
        finally:
            if shared_pool is None:
                pool.shutdown()
    else:
        for seq_idx, i in enumerate(states_needed):
            if cancel_check and cancel_check():
                raise CancelError()
            state_images[i] = _render_one_frame(frame_params[i])
            if progress_callback:
                cur = (seq_idx + 1) * RENDER_W // num_needed
                progress_callback(cur, 100, _desc="Render" if seq_idx == 0 else None)

    first_rendered = next(img for img in state_images if img is not None)
    canvas_w, canvas_h = first_rendered.size
    log.info(f"  CPU RENDER DONE: canvas={canvas_w}x{canvas_h}")
    log_ram("CPU: after render (all frames in mem)")

    total_video_frames = len(frame_state)
    log.info(f"  CPU FFMPEG PIPE: output={output_path}, total_frames={total_video_frames}, compression={compression}")
    ffmpeg_proc = _create_ffmpeg_pipe(output_path, canvas_w, canvas_h, fps=fps, compression=compression)
    written = 0
    try:
        for idx, state_idx in enumerate(frame_state):
            ffmpeg_proc.stdin.write(np.array(state_images[state_idx]).tobytes())
            written += 1
            if progress_callback:
                cur = RENDER_W + written * (100 - RENDER_W) // total_video_frames
                progress_callback(cur, 100, _desc="Encode" if written == 1 else None)
    finally:
        _close_pipe(ffmpeg_proc)
    log.info(f"  CPU FFMPEG DONE: {written} frames written, returncode={ffmpeg_proc.returncode}")
    log_ram("CPU: after ffmpeg pipe")

    return [], frame_state


# ─── Progress Display ──────────────────────────────────────────────

class TerminalProgress:
    def __init__(self, total: int, desc: str = "Generating frames", hide_rate: bool = False):
        self.total = total
        self.desc = desc
        self.hide_rate = hide_rate
        self.start_time = time_module.time()
        self.last_update_time = self.start_time
        self.last_current = 0
        self.window_rate = 0.0
        self.last_draw = 0
        self._last_print_time = -999.0
        self.width = 40
        self._gpu_stats = None
        self._is_tty = sys.stdout.isatty()
        self._term_width = shutil.get_terminal_size().columns if self._is_tty else 120
        # Phase-combining state (overlay pre-render + gpu render → single bar)
        self._phase_offset = 0
        self._phase0_total = None
        self._phase_prev_cur = None
        self._max_line_width = 0

    def _time_str(self, t: float) -> str:
        total_sec = int(round(t))
        if total_sec >= 3600:
            return f"{total_sec // 3600}h{total_sec % 3600 // 60}m"
        elif total_sec >= 60:
            return f"{total_sec // 60}m{total_sec % 60}s"
        else:
            return f"{t:.1f}s"

    def _build_line(self, current: int, elapsed: float, rate: float, eta: float) -> str:
        frac = current / self.total if self.total > 0 else 0
        pct = frac * 100
        total_t = elapsed + eta
        suffix = f" {pct:.0f}% | {self._time_str(elapsed)}/{self._time_str(total_t)}" if self.hide_rate else f" {pct:.0f}% | {rate:.0f}/s | {self._time_str(elapsed)}/{self._time_str(total_t)}"
        if self._gpu_stats and self._gpu_stats.get("batch_size"):
            s = self._gpu_stats
            mb = s.get('mem_used_mb', 0) / 1024
            tb = s.get('total_mem_mb', 0) / 1024
            suffix += f" | {mb:.1f}/{tb:.1f}GB | Batch: {s.get('batch_size', 0)}"
        prefix = f"{self.desc}: ["
        fixed_len = len(prefix) + 1 + len(suffix)  # +1 for "]"
        bar_w = max(2, min(40, self._term_width - fixed_len))
        filled = int(bar_w * frac)
        bar = "#" * filled + "-" * (bar_w - filled)
        line = f"{prefix}{bar}]{suffix}"
        return line

    def update(self, current: int, actual_current: Optional[int] = None, actual_total: Optional[int] = None):
        now = time_module.time()
        # Throttle: redraw at most every ~100ms to avoid flooding console scrollback
        if current < self.total and now - self._last_print_time < 1.0:
            return
        elapsed = now - self.start_time
        window_elapsed = now - self.last_update_time
        _rate_source = actual_current if actual_current is not None else current
        if window_elapsed > 0.5 and _rate_source > self.last_current:
            instant = (_rate_source - self.last_current) / window_elapsed
            if self.window_rate <= 0:
                self.window_rate = instant
            else:
                self.window_rate = self.window_rate * 0.5 + instant * 0.5
            self.last_update_time = now
            self.last_current = _rate_source
        rate = self.window_rate if self.window_rate > 0 else _rate_source / elapsed if elapsed > 0 else 0
        _remaining_source = (actual_total - _rate_source) if (actual_total is not None and actual_current is not None) else (self.total - current)
        eta = _remaining_source / rate if rate > 0 else 0
        line = self._build_line(current, elapsed, rate, eta)
        if self._is_tty:
            # Track max width so shorter updates fully overwrite longer ones
            self._max_line_width = max(self._max_line_width, len(line))
            print(f"\r{line.ljust(self._max_line_width)}", end="", flush=True)
        else:
            # Pipe/file: each update on its own line
            print(line, flush=True)
        self._last_print_time = now
        if current >= self.total:
            print()

    def set_desc(self, desc: str):
        self.desc = desc

    def __call__(self, current, total, **kwargs):
        # Phase transition: when cur goes backwards, a new phase started
        if (self._phase_prev_cur is not None and
            current < self._phase_prev_cur and
            self._phase0_total is not None):
            self._phase_offset += self._phase0_total
        self._phase_prev_cur = current
        if self._phase0_total is None:
            self._phase0_total = total
        _use_gpu = kwargs.get("_use_gpu", False)
        if _use_gpu:
            adjusted_cur = 1 + current * 99 // total if total > 0 else 0
            adjusted_tot = 100
        else:
            adjusted_cur = current + self._phase_offset
            adjusted_tot = self._phase0_total
        self.total = adjusted_tot
        gpu_stats = kwargs.get("gpu_stats")
        if gpu_stats is not None:
            self._gpu_stats = gpu_stats
        _desc = kwargs.get("_desc")
        if _desc:
            self.set_desc(_desc)
        self.update(adjusted_cur, actual_current=current, actual_total=total)


# ─── Batch Helpers ─────────────────────────────────────────────────

def _quick_infer_size(solution: str, scramble: Optional[str] = None, size=None) -> Optional[Tuple[int, int]]:
    """Quickly determine puzzle (width, height) without full render setup.
    Uses the first available source: size param > scramble > guess from solution."""
    if size is not None:
        if isinstance(size, str) and 'x' in size:
            parts = size.lower().split('x')
            return (int(parts[0]), int(parts[1]))
        if isinstance(size, (tuple, list)) and len(size) == 2:
            return (int(size[0]), int(size[1]))
        return None
    if scramble:
        try:
            from replay_generator import scramble_to_puzzle
            matrix = scramble_to_puzzle(scramble)
            return (len(matrix[0]), len(matrix))
        except Exception:
            pass
    try:
        from replay_generator import parse_scramble_guess, expand_solution
        matrix = parse_scramble_guess(solution)
        if matrix:
            return (len(matrix[0]), len(matrix))
    except Exception:
        pass
    try:
        sol_len = len(expand_solution(solution))
        side = math.isqrt(sol_len)
        if side * side == sol_len:
            return (side, side)
    except Exception:
        pass
    return None


def _batch_cpu_worker(item: dict) -> dict:
    """ProcessPoolExecutor worker — renders one solution with inner parallelism disabled.
    Returns metadata dict for clean progress display."""
    gen = ReplayVideoGenerator()
    kwargs = {k: v for k, v in item.items() if k != "solution" and k != "output_path"}
    t0 = time_module.time()
    gen.generate_simple_replay(
        solution=item["solution"],
        output_path=item["output_path"],
        use_gpu=False,
        parallel=False,
        show_progress=False,
        **kwargs,
    )
    elapsed = time_module.time() - t0
    return {"path": item["output_path"], "elapsed": elapsed}


# ─── Main API ──────────────────────────────────────────────────────

class ReplayVideoGenerator:
    def __init__(self, temp_dir: Optional[str] = None, cleanup_frames: bool = True):
        self.temp_dir = temp_dir
        self.cleanup_frames = cleanup_frames

    def generate_simple_replay(
        self,
        solution: str,
        output_path: str = "replay.mp4",
        tps: Optional[Union[float, int]] = None,
        time: Optional[float] = None,
        scramble: Optional[str] = None,
        size: Optional[Tuple[int, int]] = None,
        movetimes: Union[int, List[float]] = -1,
        force_fringe: bool = False,
        show_progress: bool = True,
        speed_factor: float = 1.0,
    quality: float = 1.0,
        external_progress_cb = None,
        use_gpu: bool = True,
        cancel_check=None,
        fps: int = 60,
        compression: int = 18,
        parallel: bool = True,
        gpu_renderer=None,
    ):
        log.info(f"generate_simple_replay: output={output_path}, force_fringe={force_fringe}, fps={fps}, compression={compression}, quality={quality}, use_gpu={use_gpu}")
        log.info(f"  tps={tps}, time={time}, scramble_len={len(scramble) if scramble else 0}, size={size}")
        log.info(f"  movetimes_type={type(movetimes).__name__}, sol_len_approx={len(expand_solution(solution)) if isinstance(solution, str) else '?'}")
        if tps is not None and time is not None:
            raise ValueError("Provide either tps or time, not both")

        # Determine puzzle matrix
        if scramble is not None:
            matrix = scramble_to_puzzle(scramble)
            width = len(matrix[0])
            height = len(matrix)
        elif size is not None:
            if isinstance(size, str) and 'x' in size:
                parts = size.lower().split('x')
                size = (int(parts[0]), int(parts[1]))
            width, height = size
            matrix = parse_scramble(width, height, solution)
            scramble = puzzle_to_scramble(matrix)
        else:
            matrix = parse_scramble_guess(solution)
            width = len(matrix[0])
            height = len(matrix)
            scramble = puzzle_to_scramble(matrix)

        solution_expanded = expand_solution(solution)
        sol_len = len(solution_expanded)
        log.info(f"  matrix source: {'provided scramble' if scramble is not None else 'size' if size is not None else 'guessed from solution'}, {width}x{height}")
        log.info(f"  matrix={width}x{height}, sol_len={sol_len}")

        # Compute TPS and movetimes
        real_tps = None
        if isinstance(movetimes, list) and len(movetimes) > 0:
            custom_move_times = movetimes
            is_movetimes_accurate = True
            total_real_time_s = movetimes[-1] / 1000.0 if movetimes[-1] > 0 else 0
            if total_real_time_s > 0:
                real_tps = sol_len / total_real_time_s
        else:
            custom_move_times = None
            is_movetimes_accurate = False

        if time is not None:
            tps_val = sol_len / time
        elif tps is not None:
            tps_val = tps
        elif real_tps is not None:
            tps_val = real_tps
        else:
            tps_val = 15

        log.info(f"  tps_val={tps_val}, is_movetimes_accurate={is_movetimes_accurate}, custom_move_times={'list' if isinstance(custom_move_times, list) else None}")

        # Grids analysis
        cycled_numbers = get_cycles_numbers(matrix, solution_expanded)
        if not force_fringe:
            grids_data = analyse_grids_initial(matrix, solution_expanded, cycled_numbers)
        else:
            grids_data = {
                "enableGridsStatus": -1,
                "width": width,
                "height": height,
                "offsetW": 0,
                "offsetH": 0
            }

        try:
            grid_states = generate_grids_stats(grids_data)
        except Exception:
            grids_data = {
                "enableGridsStatus": -1,
                "width": width,
                "height": height,
                "offsetW": 0,
                "offsetH": 0
            }
            grid_states = generate_grids_stats(grids_data)

        all_fringe_schemes = get_all_fringe_schemes(grid_states)
        log.info(f"  grids_data: enableGridsStatus={grids_data.get('enableGridsStatus')}, fringe_schemes={len(all_fringe_schemes)}, force_fringe={force_fringe}")

        # Calculate timing
        if isinstance(movetimes, list) and len(movetimes) > 0:
            delays, fake_times = calculate_move_timings(solution, tps_val, width, height, speed_factor)
            fake_times = [0] + list(movetimes)
        else:
            delays, fake_times = calculate_move_timings(solution, tps_val, width, height, speed_factor)
        log.info(f"  timing: delays_count={len(delays)}, fake_times_range=[{fake_times[0]:g}, {fake_times[-1]:g}]ms")

        # Score title
        score_title_text = f"{width}x{height} sliding puzzle"

        # Progress
        total_frames = sol_len + 1

        if show_progress:
            print(f"Puzzle: {width}x{height}, Moves: {sol_len}, TPS: {tps_val:.3f}")
            print(f"Tile size: {pick_tile_size(width, height)}px x quality={quality}, Frames: {total_frames}")
            print(f"Output: {output_path}")
            prog = TerminalProgress(total_frames, "Render")
        else:
            prog = None

        def progress_cb(cur, tot, **kwargs):
            if prog:
                prog(cur, tot, **kwargs)
            if external_progress_cb:
                external_progress_cb(cur, tot, **kwargs)

        log.info(f"  calling generate_frames: fringe_schemes={len(all_fringe_schemes)}, grid_states_keys={len(grid_states)}, delays={len(delays)}, fake_times={len(fake_times)}")
        frames, frame_state_map = generate_frames(
            matrix=matrix,
            solution=solution,
            tps=tps_val,
            all_fringe_schemes=all_fringe_schemes,
            grid_states=grid_states,
            fake_times=fake_times,
            delays=delays,
            is_movetimes_accurate=is_movetimes_accurate,
            score_title_text=score_title_text,
            custom_move_times=custom_move_times,
            cumulative_data=None,
            progress_callback=progress_cb if (show_progress or external_progress_cb) else None,
            overlay_progress_callback=progress_cb if (show_progress or external_progress_cb) else None,
            quality=quality,
            parallel=parallel,
            use_gpu=use_gpu,
            cancel_check=cancel_check,
            output_path=output_path,
            fps=fps,
            compression=compression,
            gpu_renderer=gpu_renderer,
        )

        log.info(f"  generate_frames returned: frames_count={len(frames)}, frame_state_map_len={len(frame_state_map)}")

        if show_progress:
            print()

        if show_progress:
            elapsed = time_module.time() - (prog.start_time if prog else time_module.time())
            print(f"Done! Video saved to: {output_path} (took {elapsed:.1f}s)")

        return output_path

    def batch_render(
        self,
        items: List[dict],
        use_gpu: bool = True,
        max_workers: Optional[int] = None,
        show_progress: bool = True,
        external_progress_cb=None,
        cancel_check=None,
    ) -> List[str]:
        """Render multiple solutions in a single batch.

        CPU mode: one ProcessPoolExecutor for all items (cross-solution parallelism),
        inner per-solution pools disabled.  GPU mode: sequential, grouped by puzzle
        size to reuse GPURenderer between same-size items.

        items: list of dicts with keys matching generate_simple_replay params
               (at minimum: 'solution', 'output_path').
        returns: list of output paths.
        """
        if not items:
            return []

        n = len(items)
        output_paths = []

        # Phase 1: quick-scan items for size grouping
        for item in items:
            size = _quick_infer_size(
                item.get("solution", ""),
                scramble=item.get("scramble"),
                size=item.get("size"),
            )
            item["_inferred_size"] = size

        if use_gpu:
            # ── GPU path: sequential, group by size ──
            groups: Dict[tuple, List[dict]] = {}
            for item in items:
                sz = item.get("_inferred_size") or (0, 0)
                kval = (sz[0], sz[1], item.get("quality", 1.0))
                groups.setdefault(kval, []).append(item)

            renderer = None
            prev_key = None
            try:
                for key in sorted(groups.keys()):
                    group = groups[key]
                    if renderer is None or key != prev_key:
                        if renderer is not None:
                            renderer.cleanup()
                        w, h, quality = key
                        eff_q = quality + 1.0
                        raw_ts = pick_tile_size(w, h)
                        eff_ts = max(raw_ts, int(raw_ts * eff_q))
                        pw_est = w * eff_ts
                        cw_est = (pw_est + STATS_PANEL_WIDTH + PADDING * 3 + 1) // 2 * 2
                        pnl_w = cw_est - (PADDING + pw_est + PADDING) - PADDING
                        min_ch = HEADER_H + PADDING + _compute_stats_full_height(pnl_w, has_grid_stages=True) + PADDING
                        renderer = GPURenderer(w, h, raw_ts, quality=eff_q, min_canvas_h=min_ch)

                    for item in group:
                        if cancel_check and cancel_check():
                            raise CancelError()
                        kwargs = {k: v for k, v in item.items()
                                  if k not in ("solution", "output_path", "_inferred_size")}
                        self.generate_simple_replay(
                            solution=item["solution"],
                            output_path=item["output_path"],
                            use_gpu=True,
                            show_progress=show_progress,
                            cancel_check=cancel_check,
                            gpu_renderer=renderer,
                            **kwargs,
                        )
                        output_paths.append(item["output_path"])
                        if external_progress_cb:
                            external_progress_cb(len(output_paths), n)

                    prev_key = key
            finally:
                if renderer is not None:
                    renderer.cleanup()
        else:
            # ── CPU path: one pool, cross-solution parallelism ──
            max_workers = max_workers or os.cpu_count() or 4
            cpu_items = []
            for item in items:
                kwargs = {k: v for k, v in item.items()
                          if k not in ("solution", "output_path", "_inferred_size")}
                cpu_items.append({
                    "solution": item["solution"],
                    "output_path": item["output_path"],
                    **kwargs,
                })

            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                fut_to_idx = {}
                for idx, citem in enumerate(cpu_items):
                    fut = pool.submit(_batch_cpu_worker, citem)
                    fut_to_idx[fut] = idx

                from concurrent.futures import as_completed
                _batch_prog = TerminalProgress(n, "Batch", hide_rate=True) if show_progress else None
                done_set = set()
                while len(done_set) < len(cpu_items):
                    if cancel_check and cancel_check():
                        raise CancelError()
                    for fut in as_completed(fut_to_idx):
                        if fut in done_set:
                            continue
                        result = fut.result()
                        done_set.add(fut)
                        out_path = result["path"]
                        output_paths.append(out_path)
                        if external_progress_cb:
                            external_progress_cb(len(output_paths), n)
                        if _batch_prog:
                            _batch_prog(len(output_paths), n)
                        break

        return output_paths


# ─── CLI Entry Point ───────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate sliding puzzle replay video")
    parser.add_argument("--url", type=str, default=None, help="SlidySim replay URL (e.g. https://slidysim.github.io/replay?r=...)")
    parser.add_argument("solution", nargs="?", default=None, help="Solution string (e.g. R3D3L3U3)")
    parser.add_argument("--tps", type=float, default=None, help="TPS value")
    parser.add_argument("--time", type=float, default=None, help="Total time in seconds")
    parser.add_argument("--scramble", type=str, default=None, help="Scramble string")
    parser.add_argument("--size", type=str, default=None, help="Puzzle size (e.g. 4x4)")
    parser.add_argument("--output", type=str, default="replay.mp4", help="Output video path")
    parser.add_argument("--force-fringe", action="store_true", default=False, help="Force fringe colors (disable grids detection)")
    parser.add_argument("--quality", type=float, default=1.0, help="Render quality multiplier")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor")
    parser.add_argument("--fps", type=int, default=60, help="Output video frame rate (default: 60)")
    parser.add_argument("--temp-dir", type=str, default=None, help="Temp directory for frames")

    args = parser.parse_args()

    if args.url:
        solution, tps, scramble, movetimes = parse_replay_url(args.url)
        size = None
    else:
        if args.solution is None:
            parser.error("provide a solution string or --url")
        solution = args.solution
        tps = args.tps
        scramble = args.scramble
        movetimes = -1
        size = None
        if args.size:
            parts = args.size.lower().split("x")
            if len(parts) == 2:
                size = (int(parts[0]), int(parts[1]))

    gen = ReplayVideoGenerator(temp_dir=args.temp_dir)
    gen.generate_simple_replay(
        solution=solution,
        output_path=args.output,
        tps=tps,
        time=args.time,
        scramble=scramble,
        size=size,
        movetimes=movetimes,
        force_fringe=args.force_fringe,
        speed_factor=args.speed,
        quality=args.quality,
        fps=args.fps,
        show_progress=True
    )


if __name__ == "__main__":
    main()

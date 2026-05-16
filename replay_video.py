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
import subprocess
import sys
import tempfile
import time as time_module
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Union, Dict
import bisect
import threading
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

from replay_generator import (
    expand_solution, scramble_to_puzzle, puzzle_to_scramble,
    create_puzzle, parse_scramble,
    parse_scramble_guess, calculate_manhattan_distance,
    get_repeated_lengths,
)

from sliding_puzzles import decompress_string_to_array, read_solve_data, parse_replay_url, _MOVE_DIRS, move_matrix_inplace, update_md_flat, find_zero

from grids_analysis import (
    CT_MAP, analyse_grids_initial, generate_grids_stats, filter_grid_stages,
    collect_all_cycled_tiles,
)

from track_progress import ProgressTracker, CPU_PHASE_WEIGHTS, GPU_PHASE_WEIGHTS, BATCH_PHASE_WEIGHTS
from debug_log import get_logger, CancelError, log_ram, reset_ram_baseline

log = get_logger()
reset_ram_baseline()

from geometry import (PADDING, HEADER_H, STATS_PANEL_WIDTH, INFO_H, TIMER_HEIGHT,
    BG_COLOR, TILE_BG, TILE_TEXT_COLOR, TILE_BORDER_COLOR, NULL_COLOR,
    PANEL_BG, PANEL_ALPHA, TIMER_BG, ACCURATE_COLOR, INACCURATE_COLOR,
    WHITE, CYAN, GREEN, GRAY, LIGHT_GRAY,
    TILE_BORDER_WIDTH, TILE_BORDER_RADIUS_RATIO,
    compute_canvas_dimensions, compute_layout, RenderOptions,
    get_font, render_number_texture, render_timer_text,
    compute_font_size,
    compute_grid_position, compute_panel_rect, compute_secondary_bar_rect,
    should_draw_numbers, should_draw_tile_border, should_draw_secondary_border_rect,
    round_canvas_height,
    TileSpriteCache, _solid_base, _bar_sprite, select_base, select_bar,
    prerender_composite_tile, parse_hex_color)

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


def _fringe_to_np(fringe_list):
    """Convert list-of-lists-of-tuples fringe to (H, W, 3) uint8 numpy array."""
    return np.array(fringe_list, dtype=np.uint8)


def generate_color_fringe(colors_list, size: int):
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
    return _fringe_to_np(matrix)


def split_matrix(matrix):
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


def get_columns_colors(colors_list, width: int, height: int):
    arr = np.empty((height, width, 3), dtype=np.uint8)
    for c in range(width):
        arr[:, c] = colors_list[c % len(colors_list)]
    return arr


def get_rows_colors(colors_list, width: int, height: int):
    arr = np.empty((height, width, 3), dtype=np.uint8)
    for r in range(height):
        arr[r, :] = colors_list[r % len(colors_list)]
    return arr


def merge_matrices_by_dimension(matrix1, matrix2, match_by_width: bool):
    if match_by_width:
        return np.vstack([matrix1, matrix2])
    else:
        return np.hstack([matrix1, matrix2])


def get_fringe_colors_nxm(width: int, height: int, force_rows=False, force_columns=False):
    if force_rows:
        colors_list = get_colors(height)
        return get_rows_colors(colors_list, width, height)
    if force_columns:
        colors_list = get_colors(width)
        return get_columns_colors(colors_list, width, height)

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


def get_mono_colors(color, width: int, height: int):
    return np.full((height, width, 3), color, dtype=np.uint8)


def get_all_fringe_schemes(grid_states, force_rows=False, force_columns=False):
    _t0 = time_module.time()
    # Pre-scan to count unique sizes
    needed = set()
    for key, state in grid_states.items():
        if isinstance(key, (int, float)):
            for mc in state["mainColors"]:
                needed.add(f"{mc['width']}x{mc['height']}")
        for sc in state["secondaryColors"]:
            if sc["type"] == CT_MAP["fringe"]:
                needed.add(f"{sc['width']}x{sc['height']}")
    schemes = {}
    for i, pair in enumerate(needed):
        parts = pair.split('x')
        w = int(parts[0]); h = int(parts[1])
        schemes[pair] = get_fringe_colors_nxm(w, h, force_rows, force_columns)
    log.info(f"  get_all_fringe_schemes took {time_module.time() - _t0:.3f}s, {len(needed)} unique schemes")
    return schemes


RED_GRIDS = np.array((200, 103, 103), dtype=np.uint8)
BLUE_GRIDS = np.array((141, 179, 255), dtype=np.uint8)


# ─── Color Application (ported from replayGeneration.js) ────────────

def apply_color_any(colors_matrix, number, box_w, box_h, offset_w, offset_h, main_w, secondary=False):
    row = (number - offset_h * main_w - 1) // main_w
    col = (number - offset_w - 1) % main_w
    if 0 <= row < box_h and 0 <= col < box_w and row >= 0 and col >= 0:
        return colors_matrix[row][col]
    return None


def get_tile_colors(number, state, all_fringe_schemes, main_w, opts=None):
    grid1_color = np.array(opts.grid1_color, dtype=np.uint8) if opts and opts.grid1_color else RED_GRIDS
    grid2_color = np.array(opts.grid2_color, dtype=np.uint8) if opts and opts.grid2_color else BLUE_GRIDS
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
                c = grid1_color
            elif cs["type"] == CT_MAP["grids2"]:
                c = grid2_color
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
                scheme = get_mono_colors(grid1_color, sc["width"], sc["height"])
            elif sc["type"] == CT_MAP["grids2"]:
                scheme = get_mono_colors(grid2_color, sc["width"], sc["height"])
            else:
                continue
            if secondary_bg is None:
                secondary_bg = apply_color_any(scheme, number, sc["width"], sc["height"], sc["offsetW"], sc["offsetH"], main_w, secondary=True)
    if isinstance(main_bg, np.ndarray):
        main_bg = tuple(main_bg.ravel())
    if isinstance(secondary_bg, np.ndarray):
        secondary_bg = tuple(secondary_bg.ravel())
    return main_bg, secondary_bg


def format_time_str(ms: int) -> str:
    if ms < 1000:
        return f"0.{ms:03d}"
    total_sec = ms / 1000
    if total_sec < 60:
        return f"{total_sec:.3f}"
    minutes = int(total_sec // 60)
    sec = total_sec % 60
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}:{minutes:02d}:{sec:06.3f}"
    return f"{minutes}:{sec:06.3f}"


def _wrap_cycles_text(cyc_text, font, max_line_w):
    if not cyc_text:
        return []
    prefix = "cyc: "
    body = cyc_text[len(prefix):]
    items = [x.strip() for x in body.split(",")]
    lines = []
    prefix_w = font.getbbox(prefix)[2]
    comma_w = font.getbbox(", ")[2]
    current = prefix
    current_w = prefix_w
    for item in items:
        item_w = font.getbbox(item)[2]
        sep_w = 0 if current == prefix else comma_w
        if current_w + sep_w + item_w > max_line_w and current != prefix:
            lines.append(current)
            current = prefix + item
            current_w = prefix_w + item_w
        else:
            if current != prefix:
                current += ", "
                current_w += comma_w
            current += item
            current_w += item_w
    if current:
        lines.append(current)
    return lines


def _render_cycles_lines_on_draw(draw, lines, x, y, cur_time_ms, cycles_fix_times, font, done_color, pending_color):
    line_h = font.getbbox("Xy")[3] - font.getbbox("Xy")[1]
    for line in lines:
        prefix = "cyc: "
        body = line[len(prefix):]
        entries = [e.strip() for e in body.split(",")]
        cx = x
        draw.text((cx, y), prefix, fill=(*pending_color, 255), font=font)
        cx += font.getbbox(prefix)[2]
        for i, entry in enumerate(entries):
            if i > 0:
                draw.text((cx, y), ", ", fill=(*pending_color, 255), font=font)
                cx += font.getbbox(", ")[2]
            tile_str = entry.split("(")[0] if "(" in entry else entry
            try:
                tile = int(tile_str)
                fix_ms = cycles_fix_times.get(tile)
                color = done_color if (fix_ms is not None and cur_time_ms >= fix_ms) else pending_color
            except ValueError:
                color = pending_color
            draw.text((cx, y), entry, fill=(*color, 255), font=font)
            cx += font.getbbox(entry)[2]
        y += line_h + 8
    return y


# ─── Puzzle Rendering ──────────────────────────────────────────────

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
    opts: RenderOptions = RenderOptions(),
    tile_sprites: Optional[TileSpriteCache] = None,
    prev_canvas: Optional[Image.Image] = None,
    changed_tiles: Optional[np.ndarray] = None,
    timer_arr: Optional[np.ndarray] = None,
    stats_arr: Optional[np.ndarray] = None,
    composite_atlas: Optional[List[Image.Image]] = None,
    composite_lookup: Optional[Dict] = None,
    quality: int = 1080,
    use_gpu: bool = True,
    fps: int = 60,
    compression: int = 18,
    encoder_preset: str = "",
    puzzle_size: str = "",
    total_frames: int = 0,
    codec_name: str = "",
    resolved_preset: str = "",
    canvas_size: str = "",
    unique_frames: int = 0,
    pad: int = None,
    header_h: int = None,
    panel_w: int = None,
    canvas_w: int = None,
    canvas_h: int = None,
) -> Image.Image:
    h = len(matrix)
    w = len(matrix[0])
    if pad is None:
        pad = PADDING
    if header_h is None:
        header_h = HEADER_H
    if panel_w is None:
        panel_w = STATS_PANEL_WIDTH
    puzzle_w = w * tile_size
    puzzle_h = h * tile_size
    grid_x, grid_y = compute_grid_position(
        opts.grid_only, pad=pad, header_h=header_h,
        canvas_h=canvas_h, puzzle_h=puzzle_h,
        no_header=opts.no_header,
        align_top=opts.adjust_height,
    )

    if prev_canvas is not None and changed_tiles is not None:
        canvas = prev_canvas
        canvas_w, canvas_h = canvas.size
        draw = ImageDraw.Draw(canvas)
    else:
        if canvas_w is None or canvas_h is None:
            canvas_w, canvas_h = compute_canvas_dimensions(w, h, tile_size, grid_only=opts.grid_only, pad=pad, header_h=header_h, panel_w=panel_w, no_details=opts.no_details, adjust_height=opts.adjust_height)
            if not opts.grid_only and not opts.no_details:
                panel_w_est = canvas_w - puzzle_w - pad
                stats_h = _compute_stats_full_height(
                    panel_w_est,
                    has_grid_stages=len(stats_data.get("grid_stages", [])) > 1,
                    has_cycles=bool(stats_data.get("cycles_display", "")),
                    quality=quality,
                )
                canvas_h = max(canvas_h, (0 if opts.no_header else header_h) + stats_h)
                canvas_h = round_canvas_height(canvas_h)
        canvas = Image.new('RGB', (canvas_w, canvas_h), BG_COLOR)
        draw = ImageDraw.Draw(canvas)

    # ─── Timer Bar (left: time, right: MD/predicted/MMD) ─────────
    if not opts.grid_only and not opts.no_header:
        timer_bg_bbox = (0, 0, canvas_w, header_h)
        draw_filled_rect(draw, timer_bg_bbox, TIMER_BG)
        tf = max(12, header_h - 12)

        if timer_arr is not None:
            timer_img = Image.fromarray(timer_arr)
        else:
            timer_img = render_timer_text(timer_text, font_size=tf)
        tw, th = timer_img.size
        if opts.dynamic_md:
            tx = pad
        else:
            tx = (canvas_w - tw) // 2
        ty = (header_h - th) // 2
        canvas.paste(timer_img, (tx, ty), timer_img)

        if opts.dynamic_md:
            right_text = stats_data.get("timer_right_text", "") if stats_data else ""
            if right_text:
                right_img = render_timer_text(right_text, font_size=tf)
                rtw, rth = right_img.size
                rx = canvas_w - rtw - pad
                ry = (header_h - rth) // 2
                canvas.paste(right_img, (rx, ry), right_img)

    # ─── Puzzle Grid ──────────────────────────────────────────────
    if gpu_grid is not None:
        canvas.paste(gpu_grid, (grid_x, grid_y))
    elif composite_atlas is not None and composite_lookup is not None and prev_canvas is not None and changed_tiles is not None:
        state_sig = id(grid_state)
        lookup = composite_lookup[state_sig]
        for r, c in changed_tiles:
            num = matrix[r][c]
            sx = grid_x + c * tile_size
            sy = grid_y + r * tile_size
            canvas.paste(composite_atlas[lookup[num]], (sx, sy), composite_atlas[lookup[num]])
    elif prev_canvas is not None and changed_tiles is not None:
        for r, c in changed_tiles:
            num = matrix[r][c]
            sx = grid_x + c * tile_size
            sy = grid_y + r * tile_size
            main_bg, sec_bg = get_tile_colors(num, grid_state, all_fringe_schemes, w, opts)
            base = select_base(main_bg, num, tile_sprites)
            canvas.paste(base, (sx, sy), base)
            if not opts.no_numbers and num != 0:
                canvas.paste(tile_sprites.number_texts[num], (sx, sy), tile_sprites.number_texts[num])
            if sec_bg is not None:
                bar = select_bar(sec_bg, tile_sprites)
                if bar is not None:
                    canvas.paste(bar, (sx, sy), bar)
    elif composite_atlas is not None and composite_lookup is not None:
        state_sig = id(grid_state)
        lookup = composite_lookup[state_sig]
        for row_idx in range(h):
            for col_idx in range(w):
                num = matrix[row_idx][col_idx]
                sx = grid_x + col_idx * tile_size
                sy = grid_y + row_idx * tile_size
                canvas.paste(composite_atlas[lookup[num]], (sx, sy), composite_atlas[lookup[num]])
    elif tile_sprites is not None:
        for row_idx in range(h):
            for col_idx in range(w):
                num = matrix[row_idx][col_idx]
                sx = grid_x + col_idx * tile_size
                sy = grid_y + row_idx * tile_size

                main_bg, sec_bg = get_tile_colors(num, grid_state, all_fringe_schemes, w, opts)
                base = select_base(main_bg, num, tile_sprites)
                canvas.paste(base, (sx, sy), base)

                if not opts.no_numbers and num != 0:
                    canvas.paste(tile_sprites.number_texts[num], (sx, sy), tile_sprites.number_texts[num])

                if sec_bg is not None:
                    bar = select_bar(sec_bg, tile_sprites)
                    if bar is not None:
                        canvas.paste(bar, (sx, sy), bar)
    else:
        for row_idx in range(h):
            for col_idx in range(w):
                num = matrix[row_idx][col_idx]
                sx = grid_x + col_idx * tile_size
                sy = grid_y + row_idx * tile_size
                sq_bbox = (sx, sy, sx + tile_size, sy + tile_size)

                main_bg, sec_bg = get_tile_colors(num, grid_state, all_fringe_schemes, w, opts)
                bg_color = main_bg if main_bg is not None else (opts.tile_bg_color or TILE_BG)
                bg_color = tuple(bg_color.ravel()) if isinstance(bg_color, np.ndarray) else bg_color
                draw_filled_rect(draw, sq_bbox, bg_color)

                if should_draw_tile_border(tile_size) and not opts.no_border:
                    draw.line([(sx, sy), (sx + tile_size - 1, sy)], fill=TILE_BORDER_COLOR, width=TILE_BORDER_WIDTH)
                    draw.line([(sx, sy), (sx, sy + tile_size - 1)], fill=TILE_BORDER_COLOR, width=TILE_BORDER_WIDTH)

                if sec_bg is not None:
                    bx0, by0, bx1, by1 = compute_secondary_bar_rect(tile_size, sx, sy, font_size=font_size)
                    bar_bbox = (bx0, by0, max(bx0, bx1 - 1), max(by0, by1 - 1))
                    draw_filled_rect(draw, bar_bbox, sec_bg)
                    if should_draw_secondary_border_rect(tile_size, (bx0, by0, bx1, by1)) and not opts.no_secondary_border:
                        draw.rectangle(bar_bbox, outline=TILE_BORDER_COLOR, width=1)

                if not opts.no_numbers and num != 0 and should_draw_numbers(tile_size, font_size):
                    tex = render_number_texture(num, tile_size, font_size)
                    canvas.paste(tex, (sx, sy), tex)

    # ─── Stats Panel ──────────────────────────────────────────────
    if not opts.grid_only and not opts.no_details:
        panel_y_start = 0 if (opts.no_header or opts.grid_only) else header_h
        panel_x, panel_y, panel_w, panel_h = compute_panel_rect(
            grid_x, puzzle_w, canvas_w, grid_y, canvas_h,
            pad=pad, panel_y=panel_y_start,
        )

        if panel_w > 0 and panel_h > 0:
            blended_bg = tuple(
                int(a * PANEL_ALPHA + b * (1 - PANEL_ALPHA))
                for a, b in zip(PANEL_BG, BG_COLOR)
            )
            panel_bbox = (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h)
            draw.rectangle(panel_bbox, fill=blended_bg)

            if stats_arr is not None:
                stats_img = Image.fromarray(stats_arr)
                canvas.paste(stats_img, (panel_x, panel_y), stats_img)
            elif static_stats_base is not None and static_stats_layout is not None:
                _apply_stats_dynamic(stats_data, panel_w, static_stats_base, static_stats_layout, canvas, panel_x, panel_y)
            else:
                stats_img = _render_stats_full(stats_data, is_movetimes_accurate, panel_w, quality=quality, use_gpu=use_gpu, fps=fps, compression=compression, codec_name=codec_name, resolved_preset=resolved_preset, puzzle_size=puzzle_size, total_frames=total_frames, canvas_size=canvas_size, unique_frames=unique_frames, tile_size=tile_size)
                canvas.paste(stats_img, (panel_x, panel_y), stats_img)

    return canvas


# ─── Timing Calculation (ported from replayGeneration.js) ──────────

def calculate_move_timings(solution: str, tps: float, width: int, height: int, speed_factor: float = 1.0, expanded_solution: Optional[str] = None):
    expanded = expanded_solution if expanded_solution is not None else expand_solution(solution)
    sol_len = len(expanded)
    if sol_len <= 1:
        return [0], [0]

    repeated_width, repeated_height = get_repeated_lengths(expanded)
    longer_factor = 2
    k_w = width / longer_factor
    k_h = height / longer_factor

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

    if speed_factor != 1.0:
        inv = 1.0 / speed_factor
        delays = [d * inv for d in delays]
        fake_times = [0.0]
        for mi in range(1, sol_len):
            fake_times.append(fake_times[mi - 1] + delays[mi])

    return delays, fake_times



# ─── Frame Generation ──────────────────────────────────────────────

def _render_one_frame(params: dict) -> Image.Image:
    params.pop("timer_img", None)
    params.pop("stats_img", None)
    return render_frame(**params)


def _render_frame_batch(params_list: List[dict]) -> List[Image.Image]:
    return [_render_one_frame(params) for params in params_list]


def _calc_render_batch_size(num_needed: int, workers: int) -> int:
    return max(1, min(20, num_needed // (workers * 2) + 1))


def _make_stats_text(stats_data, is_movetimes_accurate, panel_w, quality=1080, use_gpu=True, fps=60, compression=18, codec_name="", resolved_preset="", puzzle_size="", total_frames=0, canvas_size="", unique_frames=0, tile_size=0):
    """Full render of stats panel (CPU path)."""
    return _render_stats_full(stats_data, is_movetimes_accurate, panel_w, quality=quality, use_gpu=use_gpu, fps=fps, compression=compression, codec_name=codec_name, resolved_preset=resolved_preset, puzzle_size=puzzle_size, total_frames=total_frames, canvas_size=canvas_size, unique_frames=unique_frames, tile_size=tile_size)


def _stats_layout_info(panel_w, quality=1080):
    """Compute layout constants for the stats panel, scaled to quality."""
    scale = quality / 1080.0
    px = max(4, round(10 * scale))
    inner_w = panel_w - 2 * px
    data_font = get_font(max(7, round(20 * scale)), mono=True, bold=True)
    hf = get_font(max(9, round(24 * scale)), bold=True)
    gs_hf = get_font(max(7, round(18 * scale)), bold=True)
    gs_lf = get_font(max(6, round(13 * scale)), mono=True, bold=True)
    acc_font = get_font(max(7, round(16 * scale)), mono=True, bold=True)

    data_line_h = data_font.getbbox("Xy")[3] - data_font.getbbox("Xy")[1] + 4
    row_pad = max(4, round(10 * scale))
    row_h = data_line_h + row_pad

    labels = ["Time (total):", "Moves (total):", "TPS (total):", "Cubic est:",
              "Predicted moves:", "MD (total):", "MD (current):",
              "M/MD (total):", "M/MD (current):"]

    return {
        "panel_w": panel_w, "inner_w": inner_w, "px": px,
        "data_font": data_font, "header_font": hf,
        "gs_header_font": gs_hf, "gs_data_font": gs_lf,
        "acc_font": acc_font, "row_h": row_h, "labels": labels,
    }


def _render_stats_full(stats_data, is_movetimes_accurate, panel_w, quality=1080, use_gpu=True, fps=60, compression=18, codec_name="", resolved_preset="", puzzle_size="", total_frames=0, canvas_size="", unique_frames=0, tile_size=0):
    """Full stats panel with section headers (no dynamic overlays needed)."""
    if stats_data is None:
        return Image.new("RGBA", (panel_w, 1), (0, 0, 0, 0))
    li = _stats_layout_info(panel_w, quality)
    inner_w = li["inner_w"]; px = li["px"]
    data_font = li["data_font"]; hf = li["header_font"]
    gs_hf = li["gs_header_font"]; gs_lf = li["gs_data_font"]
    acc_font = li["acc_font"]; row_h = li["row_h"]

    lines = []
    def add(x, y, text, fill, font):
        lines.append((x, y, text, fill, font))

    def lv_line(label, value, color=WHITE):
        nonlocal y
        add(px, y, label, color, data_font)
        if value:
            vb = data_font.getbbox(value)
            vw = vb[2] - vb[0]
            add(px + inner_w - vw, y, value, color, data_font)
        y += row_h

    def section_header(text):
        nonlocal y
        hb = gs_hf.getbbox(text)
        add(px, y, text, CYAN, gs_hf)
        y += (hb[3] - hb[1]) + 8

    y = 10

    hb = hf.getbbox("Stats")
    add(px, y, "Stats", CYAN, hf)
    y += (hb[3] - hb[1]) + 14

    # ── Render Info ──
    section_header("Render Info")
    lv_line("Quality: ", f"{quality}p")
    if canvas_size:
        lv_line("Canvas: ", canvas_size)
    lv_line("Render: ", "GPU" if use_gpu else "CPU")
    if codec_name:
        lv_line("Codec: ", codec_name)
    if resolved_preset:
        lv_line("Preset: ", resolved_preset)
    if tile_size:
        lv_line("Tile: ", f"{tile_size}px")
    lv_line("FPS: ", str(fps))
    lv_line("Compression: ", str(compression))
    if total_frames:
        lv_line("Frames: ", str(total_frames))
    if unique_frames:
        lv_line("Unique: ", str(unique_frames))
    lv_line("Speed: ", stats_data.get("speed_playback", "1.00x"))
    y += 6

    # ── Puzzle Info ──
    section_header("Puzzle Info")
    if puzzle_size:
        lv_line("Puzzle: ", puzzle_size)
    lv_line("Time (total): ", stats_data.get("time_all", "0.000"))
    lv_line("Moves (total): ", stats_data.get("moves_all", "0"))
    lv_line("TPS (total): ", stats_data.get("tps_all", "0.000"))
    ce = stats_data.get("cubic_estimate")
    lv_line("Cubic est: ", ce if ce else "---")
    lv_line("MD (total): ", stats_data.get("md_all", "0"))
    lv_line("M/MD (total): ", stats_data.get("mmd_all", "0.000"))
    acc_text = "Movetimes accurate" if is_movetimes_accurate else "NOT movetimes accurate"
    acc_color = ACCURATE_COLOR if is_movetimes_accurate else INACCURATE_COLOR
    ab = acc_font.getbbox(acc_text)
    add(px, y, acc_text, acc_color, acc_font)
    y += (ab[3] - ab[1]) + 6
    y += 6

    # ── Grid stages ──
    stages = stats_data.get("grid_stages", [])
    cur_stage = stats_data.get("grid_current", 0)
    show_stages = len(stages) > 1 or (len(stages) == 1 and stats_data.get("cycles_display", ""))
    if show_stages:
        gb = gs_hf.getbbox("Grid stages")
        add(px, y, "Grid stages", CYAN, gs_hf)
        y += (gb[3] - gb[1]) + 14
        if len(stages) > 1:
            raw_lines = []
            for st in stages:
                if st["cum_time"] > 0:
                    cum_s = format_time_str(st["cum_time"])
                    split_s = format_time_str(st["split_time"])
                    mvtps_s = f"({st['split_moves']}/{st['split_tps']:.1f})"
                else:
                    cum_s = str(st["cum_moves"])
                    split_s = f"(+{st['split_moves']})"
                    mvtps_s = ""
                raw_lines.append((cum_s, split_s, mvtps_s, st["label"]))
            if raw_lines:
                w1 = max(len(l[0]) for l in raw_lines)
                w2 = max(len(l[1]) for l in raw_lines)
                w3 = max(len(l[2]) for l in raw_lines) if any(l[2] for l in raw_lines) else 0
                w4 = max(len(l[3]) for l in raw_lines)
                formatted = []
                for cum_s, split_s, mvtps_s, label in raw_lines:
                    if '.' in cum_s:
                        line = f"{cum_s:>{w1}} | {split_s:>{w2}} {mvtps_s:<{w3}} | {label:<{w4}}"
                    else:
                        line = f"{cum_s:>{w1}} | {split_s:<{w2}}  | {label:<{w4}}"
                    formatted.append(line)
                max_line_w = max(gs_lf.getbbox(l)[2] for l in formatted)
                gs_x = max(6, (panel_w - max_line_w) // 2)
            for i, line in enumerate(formatted):
                color = CYAN if i == cur_stage else WHITE
                add(gs_x, y, line, color, gs_lf)
                y += (gs_lf.getbbox(line)[3] - gs_lf.getbbox(line)[1]) + 8

        # ── Cycles lines ──
        cycles_display = stats_data.get("cycles_display", "")
        cur_time_ms = stats_data.get("cur_time_ms", 0)
        cycles_fix_times = stats_data.get("cycles_fix_times", {})
        cyc_lines_data = None
        if cycles_display:
            y += 4
            max_line_w = max(gs_lf.getbbox(l)[2] for l in formatted) if formatted else panel_w - 2 * gs_x
            cyc_lines = _wrap_cycles_text(cycles_display, gs_lf, max_line_w)
            cyc_start_y = y
            for _ in cyc_lines:
                y += (gs_lf.getbbox("Xy")[3] - gs_lf.getbbox("Xy")[1]) + 8
            cyc_lines_data = (cyc_lines, cyc_start_y, cur_time_ms, cycles_fix_times)

    total_h = y + 30
    im = Image.new("RGBA", (panel_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    for x, y, text, fill, font in lines:
        draw.text((x, y), text, fill=(*fill, 255), font=font)
    if cyc_lines_data:
        cyc_lines, cyc_start_y, ctm, cft = cyc_lines_data
        _render_cycles_lines_on_draw(draw, cyc_lines, gs_x, cyc_start_y, ctm, cft, gs_lf, CYAN, WHITE)
    return im


def _compute_stats_full_height(panel_w, has_grid_stages=True, has_cycles=False, quality=1080):
    """Estimate total height of the rendered stats panel (new section layout)."""
    li = _stats_layout_info(panel_w, quality)
    data_font = li["data_font"]; hf = li["header_font"]
    gs_hf = li["gs_header_font"]; gs_lf = li["gs_data_font"]
    acc_font = li["acc_font"]; row_h = li["row_h"]

    y = 10
    y += (hf.getbbox("Stats")[3] - hf.getbbox("Stats")[1]) + 14

    # Render Info: section header + 11 rows + gap
    y += (gs_hf.getbbox("Render Info")[3] - gs_hf.getbbox("Render Info")[1]) + 8
    y += 11 * row_h
    y += 6

    # Puzzle Info: section header + 8 rows + gap
    y += (gs_hf.getbbox("Puzzle Info")[3] - gs_hf.getbbox("Puzzle Info")[1]) + 8
    y += 8 * row_h

    # Accuracy line (part of Puzzle Info, uses acc_font not row_h)
    y += (acc_font.getbbox("Movetimes accurate")[3] - acc_font.getbbox("Movetimes accurate")[1]) + 6
    y += 6

    if has_grid_stages:
        y += (gs_hf.getbbox("Grid stages")[3] - gs_hf.getbbox("Grid stages")[1]) + 14
        y += 4 * ((gs_lf.getbbox("Xy")[3] - gs_lf.getbbox("Xy")[1]) + 8)
    if has_cycles:
        y += 4 + 4 * ((gs_lf.getbbox("Xy")[3] - gs_lf.getbbox("Xy")[1]) + 8)
    y += 30
    return y


def _make_stats_static_base(panel_w, stats_data, is_movetimes_accurate, grid_stages_list,
                            quality=1080, use_gpu=True, fps=60, compression=18,
                            codec_name="", resolved_preset="", puzzle_size="",
                            total_frames=0, canvas_size="", unique_frames=0, tile_size=0):
    """Render static stats panel with section headers. Returns (image, layout_info)."""
    if stats_data is None:
        return Image.new("RGBA", (panel_w, 1), (0, 0, 0, 0)), {}
    li = _stats_layout_info(panel_w, quality)
    inner_w = li["inner_w"]; px = li["px"]
    data_font = li["data_font"]; hf = li["header_font"]
    gs_hf = li["gs_header_font"]; gs_lf = li["gs_data_font"]
    acc_font = li["acc_font"]; row_h = li["row_h"]

    lines = []
    def add(x, y, text, fill, font):
        lines.append((x, y, text, fill, font))

    def lv_line(label, value, color=WHITE):
        nonlocal y
        add(px, y, label, color, data_font)
        if value:
            vb = data_font.getbbox(value)
            vw = vb[2] - vb[0]
            add(px + inner_w - vw, y, value, color, data_font)
        y += row_h

    def section_header(text):
        nonlocal y
        hb = gs_hf.getbbox(text)
        add(px, y, text, CYAN, gs_hf)
        y += (hb[3] - hb[1]) + 8

    y = 10

    # "Stats" page header
    hb = hf.getbbox("Stats")
    add(px, y, "Stats", CYAN, hf)
    y += (hb[3] - hb[1]) + 14

    # ── Render Info ──
    section_header("Render Info")
    lv_line("Quality: ", f"{quality}p")
    if canvas_size:
        lv_line("Canvas: ", canvas_size)
    render_dev = "GPU" if use_gpu else "CPU"
    lv_line("Render: ", render_dev)
    if codec_name:
        lv_line("Codec: ", codec_name)
    if resolved_preset:
        lv_line("Preset: ", resolved_preset)
    if tile_size:
        lv_line("Tile: ", f"{tile_size}px")
    lv_line("FPS: ", str(fps))
    lv_line("Compression: ", str(compression))
    if total_frames:
        lv_line("Frames: ", str(total_frames))
    if unique_frames:
        lv_line("Unique: ", str(unique_frames))
    lv_line("Speed: ", stats_data.get("speed_playback", "1.00x"))
    y += 6

    # ── Puzzle Info ──
    section_header("Puzzle Info")
    if puzzle_size:
        lv_line("Puzzle: ", puzzle_size)
    lv_line("Time (total): ", stats_data.get("time_all", "0.000"))
    lv_line("Moves (total): ", stats_data.get("moves_all", "0"))
    lv_line("TPS (total): ", stats_data.get("tps_all", "0.000"))
    ce = stats_data.get("cubic_estimate")
    lv_line("Cubic est: ", ce if ce else "---")
    lv_line("MD (total): ", stats_data.get("md_all", "0"))
    lv_line("M/MD (total): ", stats_data.get("mmd_all", "0.000"))
    acc_text = "Movetimes accurate" if is_movetimes_accurate else "NOT movetimes accurate"
    acc_color = ACCURATE_COLOR if is_movetimes_accurate else INACCURATE_COLOR
    ab = acc_font.getbbox(acc_text)
    add(px, y, acc_text, acc_color, acc_font)
    y += (ab[3] - ab[1]) + 6
    y += 6

    # ── Grid stages ──
    stage_y_positions = []
    gs_x = px
    raw_lines = []
    w1 = w2 = w3 = w4 = 0
    if len(grid_stages_list) > 1:
        gb = gs_hf.getbbox("Grid stages")
        add(px, y, "Grid stages", CYAN, gs_hf)
        y += (gb[3] - gb[1]) + 14
        for st in grid_stages_list:
            if st["cum_time"] > 0:
                cum_s = format_time_str(st["cum_time"])
                split_s = format_time_str(st["split_time"])
                mvtps_s = f"({st['split_moves']}/{st['split_tps']:.1f})"
            else:
                cum_s = str(st["cum_moves"])
                split_s = f"(+{st['split_moves']})"
                mvtps_s = ""
            raw_lines.append((cum_s, split_s, mvtps_s, st["label"]))
        if raw_lines:
            w1 = max(len(l[0]) for l in raw_lines)
            w2 = max(len(l[1]) for l in raw_lines)
            w3 = max(len(l[2]) for l in raw_lines) if any(l[2] for l in raw_lines) else 0
            w4 = max(len(l[3]) for l in raw_lines)
            formatted = []
            for cum_s, split_s, mvtps_s, label in raw_lines:
                if '.' in cum_s:
                    line = f"{cum_s:>{w1}} | {split_s:>{w2}} {mvtps_s:<{w3}} | {label:<{w4}}"
                else:
                    line = f"{cum_s:>{w1}} | {split_s:<{w2}}  | {label:<{w4}}"
                formatted.append(line)
            max_line_w = max(gs_lf.getbbox(l)[2] for l in formatted)
            gs_x = max(6, (panel_w - max_line_w) // 2)
        for i, line in enumerate(formatted):
            add(gs_x, y, line, WHITE, gs_lf)
            stage_y_positions.append(y)
            y += (gs_lf.getbbox(line)[3] - gs_lf.getbbox(line)[1]) + 8

    stage_line_h = gs_lf.getbbox("Xy")[3] - gs_lf.getbbox("Xy")[1] + 4

    # ── Cycles (static white — same pattern as grid stages) ──
    cycles_display = stats_data.get("cycles_display", "")
    cycles_lines_list = None
    cycles_y_pos = None
    cycles_entry_data = []
    if cycles_display:
        y += 4
        cycles_y_pos = y
        try:
            max_line_w = max(gs_lf.getbbox(l)[2] for l in formatted)
        except (NameError, ValueError):
            max_line_w = panel_w - 2 * max(6, px)
        cycles_lines_list = _wrap_cycles_text(cycles_display, gs_lf, max_line_w)
        prefix = "cyc: "
        pre_w = gs_lf.getbbox(prefix)[2]
        comma_w = gs_lf.getbbox(", ")[2]
        cycles_line_h = gs_lf.getbbox("Xy")[3] - gs_lf.getbbox("Xy")[1] + 8
        for li, line in enumerate(cycles_lines_list):
            add(gs_x, y, line, WHITE, gs_lf)
            body = line[len(prefix):]
            entries = [e.strip() for e in body.split(",")]
            cx = pre_w
            for entry in entries:
                tile_str = entry.split("(")[0]
                try:
                    tile = int(tile_str)
                except ValueError:
                    tile = None
                ew = gs_lf.getbbox(entry)[2]
                cycles_entry_data.append({
                    "tile": tile,
                    "text": entry,
                    "panel_x": gs_x + cx,
                    "panel_y": cycles_y_pos + li * cycles_line_h,
                })
                cx += ew + comma_w
            y += cycles_line_h

    total_h = y + 30
    im = Image.new("RGBA", (panel_w, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    for x, y, text, fill, font in lines:
        draw.text((x, y), text, fill=(*fill, 255), font=font)

    layout_info = {
        "px": px, "gs_x": gs_x, "inner_w": inner_w,
        "data_font": data_font, "row_h": row_h,
        "header_font": hf,
        "gs_header_font": gs_hf,
        "acc_font": acc_font,
        "stage_y_positions": stage_y_positions,
        "stage_line_h": stage_line_h,
        "total_h": total_h,
        "grid_stages_list": grid_stages_list,
        "gs_lf": gs_lf,
        "stage_raw_lines": raw_lines if len(grid_stages_list) > 1 else None,
        "stage_w1": w1 if len(grid_stages_list) > 1 and raw_lines else 0,
        "stage_w2": w2 if len(grid_stages_list) > 1 and raw_lines else 0,
        "stage_w3": w3 if len(grid_stages_list) > 1 and raw_lines else 0,
        "stage_w4": w4 if len(grid_stages_list) > 1 and raw_lines else 0,
        "cycles_lines": cycles_lines_list,
        "cycles_y": cycles_y_pos,
        "cycles_entry_data": cycles_entry_data,
    }
    return im, layout_info
# ─── Font Atlas Disk Cache for CPU path ───────────────────────
_NP_CACHE_DIR = "render_cache"

def _get_np_font_key(font) -> str | None:
    try:
        return f"{os.path.splitext(os.path.basename(font.path))[0]}_{font.size}"
    except AttributeError:
        return None

def _build_np_font_atlas(font, cache_name=None):
    codes = range(32, 127)
    h_max = 1
    for code in codes:
        b = font.getbbox(chr(code))
        h_max = max(h_max, b[3])
    atlas = {}
    for code in codes:
        c = chr(code)
        b = font.getbbox(c)
        w = max(b[2] - b[0], 1)
        im = Image.new("RGBA", (w, h_max), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((0, 0), c, fill=(255, 255, 255, 255), font=font)
        atlas[code] = np.array(im)
    if cache_name:
        fkey = _get_np_font_key(font)
        if fkey:
            os.makedirs(_NP_CACHE_DIR, exist_ok=True)
            path = os.path.join(_NP_CACHE_DIR, f"{cache_name}_{fkey}.npz")
            np.savez_compressed(path, **{f"c{k}": v for k, v in atlas.items()})
    return atlas

def _load_np_atlas(font, cache_name):
    fkey = _get_np_font_key(font)
    if not fkey:
        return None
    path = os.path.join(_NP_CACHE_DIR, f"{cache_name}_{fkey}.npz")
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        atlas = {}
        for key in data:
            code = int(key[1:])
            atlas[code] = data[key]
        return atlas
    except Exception:
        return None

_np_font_atlas_cache: Dict = {}

def _get_np_font_atlas(font, cache_name=None):
    atlas = _np_font_atlas_cache.get(font)
    if atlas is not None:
        return atlas
    if cache_name:
        atlas = _load_np_atlas(font, cache_name)
    if atlas is None:
        atlas = _build_np_font_atlas(font, cache_name)
    _np_font_atlas_cache[font] = atlas
    return atlas

# ─── Cached PIL Text Surfaces for Stats Panel ──────────────────
_cache_stats_surfaces: Dict = {}



def _apply_stats_dynamic(stats_data, panel_w, static_base, layout_info, canvas, panel_x, panel_y):
    """Paste static base and overlay current stage highlight."""
    canvas.paste(static_base, (panel_x, panel_y), static_base)

    if stats_data is None:
        return

    px = layout_info["px"]
    gs_lf = layout_info["gs_lf"]

    # Pre-build font atlases for disk cache (.npz files for GPU to share)
    if gs_lf:
        _get_np_font_atlas(gs_lf, "gs_lf")

    def overlay_surface(text, font, fill, x, y):
        if not text:
            return
        key = (text, font.size, fill)
        surface = _cache_stats_surfaces.get(key)
        if surface is None:
            b = font.getbbox(text)
            surface = Image.new('RGBA', (b[2], b[3]), (0, 0, 0, 0))
            ImageDraw.Draw(surface).text((0, 0), text, fill=(255, 255, 255, 255), font=font)
            s_arr = np.array(surface)
            s_arr[:,:,:3] = fill
            surface = Image.fromarray(s_arr, 'RGBA')
            _cache_stats_surfaces[key] = surface
        canvas.paste(surface, (panel_x + x, panel_y + y), surface)

    # Stage highlight only
    stages = stats_data.get("grid_stages", [])
    cur_stage = stats_data.get("grid_current", 0)
    stage_y_positions = layout_info["stage_y_positions"]
    gs_x = layout_info.get("gs_x", px)
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
        overlay_surface(line, gs_lf, CYAN, gs_x, stage_y_positions[cur_stage])

    # ── Cycles (cyan overlay for entries whose fix time has been reached) ──
    cycles_entry_data = layout_info.get("cycles_entry_data", [])
    if cycles_entry_data:
        cur_time_ms = stats_data.get("cur_time_ms", 0)
        cycles_fix_times = stats_data.get("cycles_fix_times", {})
        for ed in cycles_entry_data:
            tile = ed["tile"]
            if tile is not None:
                fix_ms = cycles_fix_times.get(tile)
                if fix_ms is not None and cur_time_ms >= fix_ms:
                    overlay_surface(ed["text"], gs_lf, CYAN, ed["panel_x"], ed["panel_y"])


def prerender_tile_layers(width, height, tile_size, font_size, opts, all_fringe_schemes, grid_states, cancel_check=None):
    w, h = width, height
    ts = tile_size

    base_sprites = {}

    for state in grid_states.values():
        if cancel_check and cancel_check():
            raise CancelError()
        if len(state["mainColors"]) != 1:
            continue
        mc = state["mainColors"][0]
        key = f"{mc['width']}x{mc['height']}"
        scheme = all_fringe_schemes.get(key)
        if scheme is None:
            continue
        for num in range(1, w * h + 1):
            color = apply_color_any(
                scheme, num,
                mc['width'], mc['height'],
                mc['offsetW'], mc['offsetH'],
                w
            )
            if color is None:
                continue
            color = tuple(int(x) for x in color)
            if color not in base_sprites:
                base_sprites[color] = _solid_base(color, ts, opts)

    grid1_color_arr = opts.grid1_color if opts and opts.grid1_color else RED_GRIDS
    grid2_color_arr = opts.grid2_color if opts and opts.grid2_color else BLUE_GRIDS
    tile_bg = opts.tile_bg_color if opts and opts.tile_bg_color else TILE_BG
    red_t = tuple(int(x) for x in grid1_color_arr)
    blue_t = tuple(int(x) for x in grid2_color_arr)
    base_sprites[red_t] = _solid_base(red_t, ts, opts)
    base_sprites[blue_t] = _solid_base(blue_t, ts, opts)
    base_sprites[tile_bg] = _solid_base(tile_bg, ts, opts)
    base_sprites[NULL_COLOR] = _solid_base(NULL_COLOR, ts, opts)

    number_texts = {}
    for num in range(w * h + 1):
        number_texts[num] = render_number_texture(num, ts, font_size)

    bar_sprites = {}
    seen = set()
    for state in grid_states.values():
        for sc in state.get("secondaryColors", []):
            if sc["type"] == CT_MAP["fringe"]:
                skey = f"{sc['width']}x{sc['height']}"
                scheme = all_fringe_schemes[skey]
                for r in range(scheme.shape[0]):
                    for c in range(scheme.shape[1]):
                        color = tuple(int(x) for x in scheme[r, c])
                        if color not in seen:
                            seen.add(color)
                            bar_sprites[color] = _bar_sprite(color, ts, opts, font_size=font_size)
            elif sc["type"] == CT_MAP["grids1"]:
                color = red_t
                if color not in seen:
                    seen.add(color)
                    bar_sprites[color] = _bar_sprite(color, ts, opts, font_size=font_size)
            elif sc["type"] == CT_MAP["grids2"]:
                color = blue_t
                if color not in seen:
                    seen.add(color)
                    bar_sprites[color] = _bar_sprite(color, ts, opts, font_size=font_size)

    for col in (red_t, blue_t):
        if col not in seen:
            bar_sprites[col] = _bar_sprite(col, ts, opts, font_size=font_size)

    return TileSpriteCache(
        tile_size=ts,
        base_sprites=base_sprites,
        number_texts=number_texts,
        bar_sprites=bar_sprites,
        opts=opts,
    )


def build_composite_atlas(tile_sprites, w, h, font_size, opts, grid_states, all_fringe_schemes, cancel_check=None):
    """Pre-render composite tiles: base + number + bar into single RGBA PIL Image per (state, num).
    Returns (composite_images, composite_lookup):
      composite_images: list of PIL RGBA Images (atlas entries, all tile_size×tile_size)
      composite_lookup: dict[state_sig] → list[num] → atlas index
    """
    composite_images: List[Image.Image] = []
    composite_lookup = {}
    ts = tile_sprites.tile_size

    for state_key, state in grid_states.items():
        if cancel_check and cancel_check():
            raise CancelError()
        if not isinstance(state_key, (int, float)):
            continue
        state_sig = id(state)
        lookup = [0] * (w * h + 1)
        for num in range(w * h + 1):
            main_bg, sec_bg = get_tile_colors(num, state, all_fringe_schemes, w, opts)
            composite = prerender_composite_tile(num, main_bg, sec_bg, tile_sprites, opts)
            idx = len(composite_images)
            composite_images.append(composite)
            lookup[num] = idx
        composite_lookup[state_sig] = lookup

    log.info(f"  build_composite_atlas: {len(composite_images)} entries, {len(composite_lookup)} states, "
             f"tile_size={ts}, total_mem={len(composite_images)*ts*ts*4//(1024*1024)}MB")
    return composite_images, composite_lookup


import functools
import inspect as _inspect


_RENDER_FRAME_ARGS = None

def _get_render_frame_args():
    global _RENDER_FRAME_ARGS
    if _RENDER_FRAME_ARGS is None:
        _RENDER_FRAME_ARGS = set(_inspect.signature(render_frame).parameters.keys())
    return _RENDER_FRAME_ARGS


class _PipeWriter:
    """Write frames to ffmpeg pipe in a background thread.
    
    Overlaps pipe writes with the next frame's render by moving per-frame
    duplicate writes off the render thread. Blocks render if queue fills
    (maxsize), providing natural backpressure against the encoder.
    """

    def __init__(self, proc, maxsize=6):
        self._proc = proc
        self._queue = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, data, count=1, free_event=None):
        """Queue a frame write (data repeats count times)."""
        self._queue.put((data, count, free_event))

    def close(self):
        """Flush all queued writes then close the pipe."""
        self._queue.join()
        self._stop.set()
        self._thread.join(timeout=5)
        _close_pipe(self._proc)

    def _run(self):
        while not self._stop.is_set():
            try:
                data, count, free_event = self._queue.get(timeout=0.2)
                try:
                    for _ in range(count):
                        self._proc.stdin.write(data)
                except Exception:
                    pass
                finally:
                    if free_event is not None:
                        free_event.set()
                    self._queue.task_done()
            except queue.Empty:
                pass


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


_ENCODER_PRIORITY = ['hevc_nvenc', 'hevc_amf', 'hevc_qsv', 'libx265', 'h264_nvenc', 'h264_amf', 'h264_qsv', 'libx264']

_ENCODER_CONFIG = {
    'hevc_nvenc': {'profile': 'main', 'preset_fast': 'p4', 'preset_slow': 'p7', 'preset_flag': '-preset', 'quality_flag': '-cq', 'quality_offset': 11, 'rc_cqp': False},
    'h264_nvenc': {'profile': 'high', 'preset_fast': 'p4', 'preset_slow': 'p7', 'preset_flag': '-preset', 'quality_flag': '-cq', 'quality_offset': 12, 'rc_cqp': False},
    'hevc_amf':   {'profile': None, 'preset_fast': 'balanced', 'preset_slow': 'quality', 'preset_flag': '-quality', 'quality_flag': '-qp_p', 'quality_offset': 8, 'rc_cqp': True},
    'h264_amf':   {'profile': 'high', 'preset_fast': 'balanced', 'preset_slow': 'quality', 'preset_flag': '-quality', 'quality_flag': '-qp_p', 'quality_offset': 8, 'rc_cqp': True},
    'hevc_qsv':   {'profile': None, 'preset_fast': 'medium', 'preset_slow': 'veryslow', 'preset_flag': '-preset', 'quality_flag': '-global_quality', 'quality_offset': 8, 'rc_cqp': False},
    'h264_qsv':   {'profile': 'high', 'preset_fast': 'medium', 'preset_slow': 'veryslow', 'preset_flag': '-preset', 'quality_flag': '-global_quality', 'quality_offset': 8, 'rc_cqp': False},
    'libx265':    {'profile': 'main', 'preset_fast': 'veryfast', 'preset_slow': 'slow', 'preset_flag': '-preset', 'quality_flag': '-crf', 'quality_offset': 0, 'rc_cqp': False},
    'libx264':    {'profile': 'high', 'preset_fast': 'veryfast', 'preset_slow': 'slow', 'preset_flag': '-preset', 'quality_flag': '-crf', 'quality_offset': 0, 'rc_cqp': False},
}


@functools.lru_cache(maxsize=1)
def _detect_gpu_vendors() -> set:
    vendors = set()
    try:
        r = subprocess.run(['nvidia-smi'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        if r.returncode == 0:
            vendors.add('nvidia')
    except Exception:
        pass
    if sys.platform == 'win32':
        try:
            r = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                             capture_output=True, text=True, timeout=5)
            lower = r.stdout.lower()
            if any(x in lower for x in ('amd', 'radeon', 'advanced micro', 'ati ')):
                vendors.add('amd')
            if 'intel' in lower:
                vendors.add('intel')
        except Exception:
            pass
    return vendors


@functools.lru_cache(maxsize=1)
def _get_available_encoders() -> list:
    vendors = _detect_gpu_vendors()
    try:
        r = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=5)
        available = []
        for name in _ENCODER_PRIORITY:
            if name not in r.stdout:
                continue
            if 'nvenc' in name and 'nvidia' not in vendors:
                continue
            if 'amf' in name and 'amd' not in vendors:
                continue
            if 'qsv' in name and 'intel' not in vendors:
                continue
            available.append(name)
        return available if available else ['libx264']
    except Exception:
        return ['libx264']


@functools.lru_cache(maxsize=1)
def _get_best_encoder(encoder_override: str = "") -> str:
    if encoder_override:
        return encoder_override
    available = _get_available_encoders()
    return available[0]


def _build_encoder_cmd_base(encoder_name: str, compression: int, slow_render: bool, encoder_preset: str) -> tuple:
    cfg = _ENCODER_CONFIG.get(encoder_name, _ENCODER_CONFIG['libx264'])
    preset = encoder_preset or (cfg['preset_slow'] if slow_render else cfg['preset_fast'])
    quality = compression + cfg['quality_offset']
    return cfg, preset, quality


def _create_ffmpeg_pipe(output_path: str, width: int, height: int, fps: int = 60, compression: int = 18, slow_render: bool = False, encoder_preset: str = "", encoder_override: str = ""):
    """Spawn ffmpeg with best available encoder reading rawvideo from stdin.
    slow_render=True: slower preset (smaller file, slower encode).
    slow_render=False: faster preset.
    encoder_preset: override preset name. Takes priority over slow_render.
    encoder_override: force a specific encoder name."""
    encoder = _get_best_encoder(encoder_override)
    cfg, p, quality = _build_encoder_cmd_base(encoder, compression, slow_render, encoder_preset)

    cmd = [
        'ffmpeg', '-y', '-hide_banner',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
        '-c:v', encoder, cfg['preset_flag'], p,
    ]
    if cfg['profile']:
        cmd += ['-profile:v', cfg['profile']]
    cmd += [cfg['quality_flag'], str(quality)]
    if cfg['rc_cqp']:
        cmd += ['-rc', 'cqp']
    cmd += ['-pix_fmt', 'yuv420p']
    if encoder == 'libx264':
        cmd += ['-level', '4.1']
    cmd += ['-fps_mode', 'cfr', '-movflags', '+faststart', output_path]

    log.info(f"_create_ffmpeg_pipe ({encoder}, preset={p}): cmd={' '.join(cmd)}")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _create_ffmpeg_pipe_gpu(output_path: str, width: int, height: int, fps: int = 60, compression: int = 18, slow_render: bool = False, encoder_preset: str = "", encoder_override: str = ""):
    return _create_ffmpeg_pipe(output_path, width, height, fps, compression, slow_render, encoder_preset, encoder_override)


def _get_upscaled_path(input_path: str) -> str:
    stem, ext = os.path.splitext(input_path)
    return f"{stem}_1440p60{ext}"


def upscale_video(
    input_path: str,
    output_path: str,
    fps: int = 60,
    compression: int = 18,
    slow_render: bool = False,
    encoder_preset: str = "",
    encoder_override: str = "",
) -> str:
    encoder = _get_best_encoder(encoder_override)
    cfg, p, quality = _build_encoder_cmd_base(encoder, compression, slow_render, encoder_preset)
    scale_filter = "scale=-1:1440:flags=lanczos,pad=2560:1440:(ow-iw)/2:(oh-ih)/2"

    cmd = [
        'ffmpeg', '-i', input_path,
        '-vf', scale_filter,
        '-c:v', encoder, cfg['preset_flag'], p,
    ]
    if cfg['profile']:
        cmd += ['-profile:v', cfg['profile']]
    cmd += [cfg['quality_flag'], str(quality)]
    if cfg['rc_cqp']:
        cmd += ['-rc', 'cqp']
    cmd += ['-pix_fmt', 'yuv420p', '-r', str(fps), '-y', output_path]

    log.info(f"Upscaling to 2K: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode('utf-8', errors='replace')[:500] if result.stderr else ""
        raise RuntimeError(f"Upscale failed (code {result.returncode}): {err}")
    log.info(f"Upscale completed: {output_path}")
    return output_path


def generate_frames(
    matrix: List[List[int]],
    solution: str,
    tps: float,
    all_fringe_schemes: dict,
    grid_states: dict,
    fake_times: List[float],
    delays: List[float],
    original_fake_times: Optional[List[float]],
    original_custom_move_times: Optional[List[float]],
    is_movetimes_accurate: bool,
    score_title_text: str = "",
    custom_move_times: Optional[List[float]] = None,
    cumulative_data: Optional[dict] = None,
    progress_callback=None,
    quality: int = 1080,
    use_gpu: bool = True,
    cancel_check=None,
    output_path: str = None,
    fps: int = 60,
    compression: int = 18,
    slow_render: bool = False,
    encoder_preset: str = "",
    encoder_override: str = "",
    gpu_renderer: Optional['GPURenderer'] = None,
    speed_factor: float = 1.0,
    opts: RenderOptions = RenderOptions(),
    expanded_solution: Optional[str] = None,
    grids_data: Optional[dict] = None,
) -> Tuple[List[Image.Image], List[int]]:
    expanded = expanded_solution if expanded_solution is not None else expand_solution(solution)
    sol_len = len(expanded)
    h = len(matrix)
    w = len(matrix[0])
    _t_stage1 = time_module.time()
    log.info("====== STAGE 1: DATA PREP ======")
    log.info(f"generate_frames: {w}x{h}, sol_len={sol_len}, quality={quality}, fps={fps}, use_gpu={use_gpu}, output_path={output_path}")

    filtered_stages = filter_grid_stages(grid_states, w, h, add_last=sol_len)

    _sorted_grid_keys = sorted([k for k in grid_states.keys() if isinstance(k, (int, float))])
    _sorted_grid_keys.append(sol_len + 1)
    _grid_ptr = 0

    def _fast_grid_state(grid_states, move_index):
        nonlocal _grid_ptr
        while _grid_ptr + 1 < len(_sorted_grid_keys) and _sorted_grid_keys[_grid_ptr + 1] <= move_index:
            _grid_ptr += 1
        return grid_states[_sorted_grid_keys[_grid_ptr]]

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
        _grid_ct = original_custom_move_times if original_custom_move_times else custom_move_times
        if _grid_ct and len(_grid_ct) > s - 1 and s > 0:
            if s == sol_len:
                cum_stage_time = _grid_ct[-1]
                stage_time = _grid_ct[-1] - last_time
            else:
                cum_stage_time = _grid_ct[s - 1]
                stage_time = _grid_ct[s - 1] - last_time
            last_time = _grid_ct[s - 1] if s < sol_len else _grid_ct[-1]
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

    # ── Cycles data ──
    all_cycled_tiles = []
    cycles_fix_times = {}
    if opts.cycles_detection and grids_data is not None:
        all_cycled_tiles = collect_all_cycled_tiles(grids_data)
    if all_cycled_tiles:
        mc_flat = np.array(matrix, dtype=np.int32).flatten()
        zpt = find_zero(matrix, w, h)
        zp_idx = zpt[0] * w + zpt[1]
        tile_pos = np.zeros(w * h + 1, dtype=np.int32)
        for i, t in enumerate(mc_flat):
            tile_pos[t] = i
        tile_set = set(all_cycled_tiles)
        last_wrong = {}
        for t in all_cycled_tiles:
            if tile_pos[t] != t - 1:
                last_wrong[t] = 0
        for mi in range(sol_len):
            move = expanded[mi]
            dr, dc = _MOVE_DIRS[move]
            new_zp = zp_idx + dr * w + dc
            moving_tile = mc_flat[new_zp]
            mc_flat[zp_idx] = moving_tile
            mc_flat[new_zp] = 0
            tile_pos[moving_tile] = zp_idx
            tile_pos[0] = new_zp
            zp_idx = new_zp
            if moving_tile in tile_set and tile_pos[moving_tile] != moving_tile - 1:
                last_wrong[moving_tile] = mi + 1
        time_arr = original_fake_times if original_fake_times and len(original_fake_times) > sol_len else fake_times
        for t in all_cycled_tiles:
            lw = last_wrong.get(t, -1)
            if lw >= 0 and lw + 1 < len(time_arr):
                cycles_fix_times[t] = round(time_arr[lw + 1])
            elif lw >= 0:
                cycles_fix_times[t] = round(time_arr[-1])

    layout = compute_layout(quality, w, h, opts.grid_only, no_header=opts.no_header, no_details=opts.no_details, adjust_height=opts.adjust_height)
    tile_size = layout["tile_size"]
    font_size = layout["font_size"]
    log.info(f"  tile_size={tile_size}, pad={layout['pad']}, header_h={layout['header_h']}, panel_w={layout['panel_w']}, font_size={font_size}")

    tile_sprites = prerender_tile_layers(w, h, tile_size, font_size, opts, all_fringe_schemes, grid_states, cancel_check=cancel_check)
    if cancel_check and cancel_check():
        raise CancelError()
    log.info(f"  tile_sprites: {len(tile_sprites.base_sprites)} bases, {len(tile_sprites.number_texts)} numbers, {len(tile_sprites.bar_sprites)} bars")

    all_md = calculate_manhattan_distance(matrix)
    current_md = all_md
    zp = find_zero(matrix, w, h)

    total_time_ms = (original_fake_times[-1] if original_fake_times
                     else fake_times[-1]) if fake_times else 0
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

    total_frames = max(1, int(round((preview_ms + move_times[-1] + final_ms) / frame_time_ms)))

    frame_state = np.zeros(total_frames, dtype=np.int32)
    mi = 0
    for j in range(total_frames):
        t = j * frame_time_ms
        while mi < sol_len and preview_ms + move_times[mi] <= t:
            mi += 1
        frame_state[j] = mi

    unique_vals = np.unique(frame_state)
    states_needed = [int(x) for x in unique_vals]
    states_needed_set = set(states_needed)
    _t_frame_state_end = time_module.time()
    log.info(f"  frame_state: total_frames={total_frames}, unique_states={len(states_needed)}, frame_time_ms={frame_time_ms:.3f} (took {_t_frame_state_end - _t_stage1:.3f}s)")

    # Resolve codec + encoder preset for stats display
    _best_codec = _get_best_encoder(encoder_override)
    _codec_name = _best_codec
    _cfg = _ENCODER_CONFIG.get(_best_codec, _ENCODER_CONFIG['libx264'])
    _resolved_preset = encoder_preset or (_cfg['preset_slow'] if slow_render else _cfg['preset_fast'])
    canvas_size = f"{layout['canvas_w']}x{layout['canvas_h']}"
    unique_frames_count = len(states_needed)

    # Build composites only for grid states that will actually be rendered
    grid_keys_for_needed = _sorted_grid_keys[:-1]
    needed_state_ids = {id(grid_states[0])}
    for i, gk in enumerate(grid_keys_for_needed):
        if gk == 0:
            continue
        next_gk = grid_keys_for_needed[i + 1] if i + 1 < len(grid_keys_for_needed) else sol_len + 1
        first_fn = gk + 1
        last_fn_exclusive = next_gk + 1
        idx = bisect.bisect_left(states_needed, first_fn)
        if idx < len(states_needed) and states_needed[idx] < last_fn_exclusive:
            needed_state_ids.add(id(grid_states[gk]))

    filtered_grid_states = {k: v for k, v in grid_states.items()
                            if not isinstance(k, (int, float)) or id(v) in needed_state_ids}
    composite_images, composite_lookup = build_composite_atlas(
        tile_sprites, w, h, font_size, opts, filtered_grid_states, all_fringe_schemes, cancel_check=cancel_check
    )
    if cancel_check and cancel_check():
        raise CancelError()

    # Optional GPU acceleration for tile grid rendering
    puzzle_w_est = w * tile_size
    if use_gpu:
        if gpu_renderer is not None:
            gpu = gpu_renderer
        else:
            from gpu_renderer import GPURenderer
            gpu = GPURenderer(w, h, tile_size, pad=layout["pad"], header_h=layout["header_h"], panel_w=layout["panel_w"], canvas_w=layout["canvas_w"], canvas_h=layout["canvas_h"], opts=opts)
        if gpu.available:
            if cancel_check and cancel_check():
                raise CancelError()
            gpu.upload_composite_atlas(composite_images, composite_lookup)
        use_gpu = use_gpu and gpu.available
        log.info(f"  canvas={gpu.canvas_w}x{gpu.canvas_h}, GPU available={gpu.available}, use_gpu={use_gpu}")
    else:
        log.info(f"  GPU disabled, use_gpu=False")
    if use_gpu:
        import torch as _torch_snapshot
        log.info(f"  Python={sys.version.split()[0]}, torch={_torch_snapshot.__version__}, CUDA={_torch_snapshot.version.cuda}")

    # (tile color cache removed — was only consumed by dead CPU-only _build_tile_colors_np)

    if cancel_check and cancel_check():
        raise CancelError()

    # ── Stage 2: precompute data only for states that will be rendered ──

    # Convert matrix to flat numpy array for O(1) moves
    mc_flat = np.array(matrix, dtype=np.int32).flatten()
    zp_idx = zp[0] * w + zp[1]

    frame_params = {}
    num_needed = len(states_needed)
    _t_last = _t_stage1

    # Helper to build stats_data for a needed state
    def _build_stats_data(frame_idx, cur_time_ms, current_md, current_moves):
        moved_md = all_md - current_md
        cur_tps_val = current_moves * 1000 / cur_time_ms if cur_time_ms > 0 else 0
        cur_mmd = "High" if moved_md <= 0 else (current_moves / moved_md)
        all_mmd = sol_len / all_md if all_md > 0 else 0
        mmd_display = cur_mmd if isinstance(cur_mmd, str) else f"{cur_mmd:.3f}"
        if opts.grid_only:
            return None, ""
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
            predicted_moves = "-"
        else:
            predicted_moves = f"{round(cur_mmd * all_md)}"
        sd = {
            "time_all": format_time_str(round(total_time_ms)),
            "moves_all": str(sol_len),
            "md_all": str(all_md),
            "md_cur": str(moved_md),
            "mmd_all": f"{all_mmd:.3f}",
            "mmd_cur": mmd_display,
            "tps_all": f"{total_tps:.3f}",
            "predicted_moves": predicted_moves,
            "cubic_estimate": None,
            "speed_playback": f"{speed_factor:.2f}x" if speed_factor != 1.0 else "1.00x",
            "timer_right_text": f"{moved_md} ({predicted_moves} / {mmd_display})",
            "cur_time_ms": cur_time_ms,
        }
        move_idx = frame_idx - 1 if frame_idx > 0 else 0
        cur_stage_idx = max(0, sum(1 for s in filtered_stages if s <= move_idx) - 1)
        sd["grid_stages"] = grid_stages_list
        sd["grid_current"] = cur_stage_idx
        if all_cycled_tiles:
            parts = []
            tile_data = []
            for t in all_cycled_tiles:
                ft = cycles_fix_times.get(t)
                if ft is not None:
                    parts.append(f"{t}(→{format_time_str(ft)})")
                    tile_data.append({"tile": t, "fix_ms": ft})
                else:
                    parts.append(str(t))
                    tile_data.append({"tile": t, "fix_ms": None})
            sd["cycles_display"] = "cyc: " + ", ".join(parts)
            sd["cycles_tile_data"] = tile_data
            sd["cycles_fix_times"] = cycles_fix_times
        else:
            sd["cycles_display"] = ""
            sd["cycles_tile_data"] = []
            sd["cycles_fix_times"] = {}
        if w * h > 99:
            from replay_generator import get_cubic_estimate
            ce = get_cubic_estimate(round(total_time_ms), w, h)
            sd["cubic_estimate"] = format_time_str(ce)
        return sd, timer_text

    # Process state 0
    state0 = grid_states[0]
    sd0, tt0 = _build_stats_data(0, 0, all_md, 0)
    frame_params[0] = dict(
        matrix=mc_flat.reshape(h, w).copy(),
        grid_state=state0,
        all_fringe_schemes=all_fringe_schemes,
        tile_size=tile_size,
        font_size=font_size,
        stats_data=sd0,
        score_title_text=score_title_text,
        timer_text=tt0,
        is_movetimes_accurate=is_movetimes_accurate,
        total_moves=sol_len,
        total_time_ms=round(total_time_ms),
        total_tps=total_tps,
        opts=opts,
        tile_sprites=tile_sprites,
        composite_atlas=composite_images,
        composite_lookup=composite_lookup,
        quality=quality,
        pad=layout["pad"],
        header_h=layout["header_h"],
        panel_w=layout["panel_w"],
        canvas_w=layout["canvas_w"],
        canvas_h=layout["canvas_h"],
        use_gpu=use_gpu,
        fps=fps,
        compression=compression,
        encoder_preset=encoder_preset,
        codec_name=_codec_name,
        resolved_preset=_resolved_preset,
        canvas_size=canvas_size,
        unique_frames=unique_frames_count,
        puzzle_size=f"{w}x{h}",
        total_frames=len(frame_state),
    )

    # Walk all states sequentially with O(1) in-place moves, build frame_params only for needed states
    _prog_step = max(1, num_needed // 100)
    _prog_count = 0 if states_needed[0] == 0 else 1  # state 0 already processed separately
    prev_delta_matrix = None
    prev_state_sig = None
    _cancel_check_counter = 0
    for frame_idx in range(sol_len + 1):
        _cancel_check_counter += 1
        if _cancel_check_counter % 500 == 0 and cancel_check and cancel_check():
            raise CancelError()
        if frame_idx in states_needed_set:
            if frame_idx == 0:
                state = state0
                cur_time_ms = 0
                sd, tt = sd0, tt0
            else:
                state = _fast_grid_state(grid_states, frame_idx - 1)
                _use_orig_ct = original_custom_move_times if original_custom_move_times else custom_move_times
                if _use_orig_ct and len(_use_orig_ct) > frame_idx - 1:
                    cur_time_ms = _use_orig_ct[frame_idx - 1]
                else:
                    _ft = original_fake_times if original_fake_times else fake_times
                    cur_time_ms = _ft[frame_idx - 1] if frame_idx - 1 < len(_ft) else 0
                sd, tt = _build_stats_data(frame_idx, cur_time_ms, current_md, frame_idx)

            current_matrix = mc_flat.reshape(h, w).copy()
            current_state_sig = id(state)
            if prev_delta_matrix is None:
                changed_tiles = None
            else:
                if current_state_sig != prev_state_sig:
                    changed_tiles = None
                else:
                    _mask = (current_matrix != prev_delta_matrix)
                    changed_tiles = np.argwhere(_mask)
            prev_delta_matrix = current_matrix
            prev_state_sig = current_state_sig

            frame_params[frame_idx] = dict(
                matrix=current_matrix,
                grid_state=state,
                all_fringe_schemes=all_fringe_schemes,
                tile_size=tile_size,
                font_size=font_size,
                stats_data=sd,
                score_title_text=score_title_text,
                timer_text=tt,
                is_movetimes_accurate=is_movetimes_accurate,
                total_moves=sol_len,
                total_time_ms=round(total_time_ms),
                total_tps=total_tps,
                opts=opts,
                tile_sprites=tile_sprites,
                changed_tiles=changed_tiles,
                composite_atlas=composite_images,
                composite_lookup=composite_lookup,
                quality=quality,
                pad=layout["pad"],
                header_h=layout["header_h"],
                panel_w=layout["panel_w"],
                canvas_w=layout["canvas_w"],
                canvas_h=layout["canvas_h"],
                use_gpu=use_gpu,
                fps=fps,
                compression=compression,
                encoder_preset=encoder_preset,
                codec_name=_codec_name,
                resolved_preset=_resolved_preset,
                canvas_size=canvas_size,
                unique_frames=unique_frames_count,
                puzzle_size=f"{w}x{h}",
                total_frames=len(frame_state),
            )

            _prog_count += 1
            if progress_callback and (_prog_count % _prog_step == 0 or _prog_count == num_needed):
                progress_callback(_prog_count, num_needed, desc="Precompute" if _prog_count == _prog_step else None)

            if _prog_count % 1000 == 0:
                _now = time_module.time()
                log.info(f"  precompute state {_prog_count}/{num_needed}: {_now - _t_stage1:.1f}s total")

        if frame_idx < sol_len:
            move = expanded[frame_idx]
            move_matrix_inplace(mc_flat, move, zp_idx, w)
            current_md = update_md_flat(current_md, mc_flat, move, zp_idx, w, h)
            zp_idx = zp_idx + _MOVE_DIRS[move][0] * w + _MOVE_DIRS[move][1]

    _t_fp = time_module.time()
    log.info(f"  frame_params loop: {_t_fp - _t_stage1:.3f}s total, {num_needed} states built")

    log.info(f"  render decision: use_gpu={use_gpu}, total_video_frames={len(frame_state)}, unique_states={len(states_needed)} ({len(states_needed)*100//len(frame_state) if len(frame_state) > 0 else 0}%%)")

    # Pre-compute static stats base + layout for static/dynamic split (both paths)
    if not opts.grid_only and not opts.no_details:
        first_needed = states_needed[0] if states_needed else 0
        first_stats = frame_params[first_needed]["stats_data"]
        first_is_accurate = frame_params[first_needed]["is_movetimes_accurate"]
        grid_stages_list = first_stats.get("grid_stages", [])
        puzzle_w_pre = w * tile_size
        canvas_w_pre = (puzzle_w_pre + layout["panel_w"] + layout["pad"] + 1) // 2 * 2
        panel_x_pre = puzzle_w_pre + layout["pad"]
        panel_w_pre = canvas_w_pre - panel_x_pre
        static_base, static_layout = _make_stats_static_base(panel_w_pre, first_stats, first_is_accurate, grid_stages_list, quality=quality, use_gpu=use_gpu, fps=fps, compression=compression, codec_name=_codec_name, resolved_preset=_resolved_preset, puzzle_size=f"{w}x{h}", total_frames=len(frame_state), canvas_size=canvas_size, unique_frames=unique_frames_count, tile_size=tile_size)
        for fp_idx in states_needed:
            frame_params[fp_idx]["static_stats_base"] = static_base
            frame_params[fp_idx]["static_stats_layout"] = static_layout

    else:
        static_base = None
        static_layout = None

    log.info(f"====== STAGE 1 DONE: {time_module.time() - _t_stage1:.1f}s ======")

    # Pre-compute canvas dimensions for CPU path
    canvas_w_cpu, canvas_h_cpu = layout["canvas_w"], layout["canvas_h"]

    # ── GPU path: render unique states, pipe via frame mapping ──
    if use_gpu and len(frame_params) > 1:
        puzzle_w = w * tile_size
        puzzle_h = h * tile_size
        canvas_w = gpu.canvas_w
        canvas_h = gpu.canvas_h
        panel_x = puzzle_w + layout["pad"]
        panel_w_val = canvas_w - panel_x

        extra_overlay_args = dict(
            panel_w_val=panel_w_val,
            static_base=static_base,
            static_layout=static_layout,
        ) if not opts.grid_only else None

        # Pre-compute how many video frames each puzzle state spans
        state_to_count = {}
        for state_idx in frame_state:
            state_to_count[state_idx] = state_to_count.get(state_idx, 0) + 1
        log.info(f"  state_to_count: {len(state_to_count)} unique states, counts={list(state_to_count.values())[:20]}...")

        # Open ffmpeg pipe with selected encoder
        log.info(f"  OPENING FFMPEG PIPE: output={output_path}, canvas={canvas_w}x{canvas_h}, fps={fps}, compression={compression}, encoder=hevc_nvenc")
        enc_proc = _create_ffmpeg_pipe_gpu(output_path, canvas_w, canvas_h, fps=fps, compression=compression, slow_render=slow_render, encoder_preset=encoder_preset, encoder_override=encoder_override)
        writer = _PipeWriter(enc_proc)
        unique_params = [frame_params[i] for i in states_needed]
        _t_stage3 = time_module.time()
        log.info("====== STAGE 3: GPU RENDER ======")
        log.info(f"  GPU RENDER START: {len(unique_params)} unique frames to render")
        log_ram("before GPU render")

        def handler(img, idx_in_unique, total, free_event=None):
            count = state_to_count[states_needed[idx_in_unique]]
            if isinstance(img, np.ndarray):
                data = memoryview(img)
            else:
                data = img.tobytes()
            writer.write(data, count, free_event)

        _gpu_render_step = max(1, len(unique_params) // 100)
        _gpu_render_count = 0

        def _gpu_progress_cb(cur, tot, **kw):
            nonlocal _gpu_render_count
            _gpu_render_count += 1
            if progress_callback and (_gpu_render_count % _gpu_render_step == 0 or cur == tot):
                progress_callback(cur, tot, use_gpu=True, desc="Render" if _gpu_render_count == _gpu_render_step else None, **kw)

        try:
            try:
                gpu.render_frames(
                    unique_params,
                    progress_callback=_gpu_progress_cb,
                    cancel_check=cancel_check,
                    frame_handler=handler,
                    overlay_render_data=extra_overlay_args,
                )
            finally:
                if gpu_renderer is None:
                    gpu.cleanup()
        finally:
            writer.close()

        log.info(f"====== STAGE 3 DONE: {time_module.time() - _t_stage3:.1f}s ======")

        log.info(f"  GPU PATH COMPLETE: returning {len(frame_state)} frame_state entries")
        log_ram("after GPU render")

        return [], frame_state.tolist()

    # ── CPU path: render unique states, pipe via frame mapping ──
    log.info(f"  CPU PATH: {len(states_needed)} unique states to render, canvas={canvas_w_cpu}x{canvas_h_cpu}")
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

    # Pre-compute how many video frames each puzzle state spans (shared with serial path)
    state_to_count = {}
    for state_idx in frame_state:
        state_to_count[state_idx] = state_to_count.get(state_idx, 0) + 1

    num_needed = len(states_needed)
    total_video_frames = len(frame_state)
    log_ram("CPU: before render")

    # Open ffmpeg pipe early so render + encode overlap
    log.info(f"  CPU FFMPEG PIPE: output={output_path}, total_frames={total_video_frames}, compression={compression}")
    ffmpeg_proc = _create_ffmpeg_pipe(output_path, canvas_w_cpu, canvas_h_cpu, fps=fps, compression=compression, slow_render=slow_render, encoder_preset=encoder_preset, encoder_override=encoder_override)
    writer = _PipeWriter(ffmpeg_proc)

    _render_prog_step = max(1, num_needed // 100)
    try:
        written = 0
        prev_canvas = None
        for seq_idx, i in enumerate(states_needed):
            if cancel_check and cancel_check():
                raise CancelError()
            p = frame_params[i]
            kw = {k: v for k, v in p.items() if k in _get_render_frame_args()}
            if prev_canvas is not None:
                kw["prev_canvas"] = prev_canvas
            img = render_frame(**kw)
            prev_canvas = img

            count = state_to_count[i]
            data = img.tobytes()
            writer.write(data, count)
            written += count

            _render_prog_count = seq_idx + 1
            if progress_callback and (_render_prog_count % _render_prog_step == 0 or _render_prog_count == num_needed):
                progress_callback(_render_prog_count, num_needed, desc="Render" if _render_prog_count == _render_prog_step else None)

        log.info(f"  CPU RENDER+ENCODE DONE: {written} frames written, canvas={canvas_w_cpu}x{canvas_h_cpu}")
    finally:
        writer.close()
    log_ram("CPU: after ffmpeg pipe")

    return [], frame_state.tolist()


# ─── Progress Display ──────────────────────────────────────────────
# (handled by track_progress.ProgressTracker)


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
        from replay_generator import parse_scramble_guess
        matrix = parse_scramble_guess(solution)
        if matrix:
            return (len(matrix[0]), len(matrix))
    except Exception:
        pass
    try:
        from replay_generator import count_moves
        sol_len = count_moves(solution)
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
    opts = item.get("opts", RenderOptions())
    kwargs = {k: v for k, v in item.items()
              if k not in ("solution", "output_path", "_inferred_size", "opts")}
    t0 = time_module.time()
    gen.generate_simple_replay(
        solution=item["solution"],
        output_path=item["output_path"],
        use_gpu=False,
        show_progress=False,
        opts=opts,
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
        force_rows: bool = False,
        force_columns: bool = False,
        show_progress: bool = True,
        speed_factor: float = 1.0,
    quality: int = 1080,
        external_progress_cb = None,
        use_gpu: bool = True,
        cancel_check=None,
        fps: int = 60,
        compression: int = 18,
        slow_render: bool = False,
        encoder_preset: str = "",
        encoder_override: str = "",
        gpu_renderer=None,
        opts: RenderOptions = RenderOptions(),
        upscale: bool = False,
    ):
        _start_time = time_module.time()
        log.info(f"generate_simple_replay: output={output_path}, force_fringe={force_fringe}, force_rows={force_rows}, force_columns={force_columns}, fps={fps}, compression={compression}, slow_render={slow_render}, quality={quality}, use_gpu={use_gpu}, upscale={upscale}, encoder_override={encoder_override}")
        log.info(f"  tps={tps}, time={time}, scramble_len={len(scramble) if scramble else 0}, size={size}")
        if tps is not None and time is not None:
            raise ValueError("Provide either tps or time, not both")

        _t_matrix_start = time_module.time()

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

        _t_expand_start = time_module.time()
        solution_expanded = expand_solution(solution)
        sol_len = len(solution_expanded)
        _t_expand_end = time_module.time()
        log.info(f"  matrix source: {'provided scramble' if scramble is not None else 'size' if size is not None else 'guessed from solution'}, {width}x{height}")
        log.info(f"  matrix={width}x{height}, sol_len={sol_len}")
        log.info(f"  expand_solution took {_t_expand_end - _t_expand_start:.3f}s, matrix resolution took {_t_expand_start - _t_matrix_start:.3f}s")

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
        elif real_tps is not None:
            tps_val = real_tps
        elif tps is not None:
            tps_val = tps
        else:
            tps_val = 15

        log.info(f"  tps_val={tps_val}, is_movetimes_accurate={is_movetimes_accurate}, custom_move_times={'list' if isinstance(custom_move_times, list) else None}")

        # ── Progress setup (moved before analysis so early stages are tracked) ──
        total_frames = sol_len + 1
        pw = GPU_PHASE_WEIGHTS  # both GPU and CPU use overlapped pipe (no separate encode phase)

        if show_progress:
            print(f"Puzzle: {width}x{height}, Moves: {sol_len}, TPS: {tps_val:.3f}")
            print(f"Frames: {total_frames}")
            print(f"Output: {output_path}")

        prog = None
        if show_progress or external_progress_cb:
            prog = ProgressTracker(
                total=total_frames,
                desc="Render",
                phase_weights=pw,
                external_cb=external_progress_cb,
                show_terminal=show_progress,
            )
        # Fire initial analysis progress tick to enter phase 0
        _analysis_weight = pw[0]  # e.g. 2 (GPU) or 2 (CPU)
        if prog:
            prog(0, _analysis_weight, desc="Analysis")

        # ── Stage: Grid analysis ──
        _t_analysis_start = time_module.time()

        def _analysis_prog(cur, tot):
            if prog:
                scaled = int(round(_analysis_weight * cur / tot)) if tot > 0 else _analysis_weight
                prog(scaled, _analysis_weight)

        force_fringe_final = force_fringe or force_rows or force_columns
        if not force_fringe_final:
            grids_data = analyse_grids_initial(matrix, solution_expanded, progress_callback=_analysis_prog, cancel_check=cancel_check, cycles_detection=opts.cycles_detection)
        else:
            grids_data = {
                "enableGridsStatus": -1,
                "width": width,
                "height": height,
                "offsetW": 0,
                "offsetH": 0
            }

        # Grid analysis fully consumed the phase weight
        _t_gridstats_start = time_module.time()
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
        log.info(f"  generate_grids_stats took {time_module.time() - _t_gridstats_start:.3f}s, {len(grid_states)} states")

        _t_schemes_start = time_module.time()
        all_fringe_schemes = get_all_fringe_schemes(grid_states, force_rows, force_columns)
        log.info(f"  get_all_fringe_schemes took {time_module.time() - _t_schemes_start:.3f}s, {len(all_fringe_schemes)} schemes")
        _t_analysis_end = time_module.time()
        log.info(f"  analysis total took {_t_analysis_end - _t_analysis_start:.3f}s, enableGridsStatus={grids_data.get('enableGridsStatus')}")

        # Consume full analysis weight now (children send no progress)
        if prog:
            prog(_analysis_weight, _analysis_weight)

        if cancel_check and cancel_check():
            raise CancelError()

        # ── Stage: Timing ──
        _t_timing_start = time_module.time()
        if isinstance(movetimes, list) and len(movetimes) > 0:
            delays, _fake_unused = calculate_move_timings(solution, tps_val, width, height, speed_factor, expanded_solution=solution_expanded)
            inv = 1.0 / speed_factor if speed_factor != 1.0 else 1.0
            custom_move_times_sped = [t * inv for t in movetimes]
            original_fake_times = [0] + list(movetimes)
            original_custom_move_times = list(movetimes)
            fake_times = [0] + custom_move_times_sped
        else:
            delays, fake_times = calculate_move_timings(solution, tps_val, width, height, speed_factor, expanded_solution=solution_expanded)
            original_fake_times = [t * speed_factor for t in fake_times] if speed_factor != 1.0 else fake_times
            custom_move_times_sped = None
            original_custom_move_times = None
        _t_timing_end = time_module.time()
        log.info(f"  calculate_move_timings took {_t_timing_end - _t_timing_start:.3f}s, delays_count={len(delays)}, fake_times_range=[{fake_times[0]:g}, {fake_times[-1]:g}]ms")

        # Score title
        score_title_text = f"{width}x{height} sliding puzzle"

        _t_prerender_end = time_module.time()
        log.info(f"  pre-render stages total took {_t_prerender_end - _t_analysis_start:.3f}s (analysis={_t_analysis_end - _t_analysis_start:.3f}s, timing={_t_timing_end - _t_timing_start:.3f}s)")
        if cancel_check and cancel_check():
            raise CancelError()
        log.info(f"  calling generate_frames: fringe_schemes={len(all_fringe_schemes)}, grid_states_keys={len(grid_states)}, delays={len(delays)}, fake_times={len(fake_times)}")
        _orig_cmt = original_custom_move_times
        frames, frame_state_map = generate_frames(
            matrix=matrix,
            solution=solution,
            tps=tps_val,
            all_fringe_schemes=all_fringe_schemes,
            grid_states=grid_states,
            fake_times=fake_times,
            delays=delays,
            original_fake_times=original_fake_times,
            original_custom_move_times=_orig_cmt,
            is_movetimes_accurate=is_movetimes_accurate,
            score_title_text=score_title_text,
            custom_move_times=custom_move_times_sped,
            cumulative_data=None,
            progress_callback=prog,
            quality=quality,
            use_gpu=use_gpu,
            cancel_check=cancel_check,
            output_path=output_path,
            fps=fps,
            compression=compression,
            slow_render=slow_render,
            encoder_preset=encoder_preset,
            encoder_override=encoder_override,
            gpu_renderer=gpu_renderer,
            speed_factor=speed_factor,
            opts=opts,
            expanded_solution=solution_expanded,
            grids_data=grids_data,
        )

        log.info(f"  generate_frames returned: frames_count={len(frames)}, frame_state_map_len={len(frame_state_map)}")

        if prog:
            prog.finish()
            elapsed = time_module.time() - _start_time
            unique_frames = len(set(frame_state_map))
            total_frames = len(frame_state_map)
            print(f"Done! Video saved to: {output_path} ({unique_frames} unique / {total_frames} total frames, took {elapsed:.1f}s)")

        if upscale:
            if quality < 1440:
                upscaled_path = _get_upscaled_path(output_path)
                print("Upscaling to 2K (2560x1440)...")
                upscale_video(
                    input_path=output_path,
                    output_path=upscaled_path,
                    fps=fps,
                    compression=compression,
                    slow_render=slow_render,
                    encoder_preset=encoder_preset,
                    encoder_override=encoder_override,
                )
                print(f"Upscaled version saved to: {upscaled_path}")
            else:
                print("[Note] Upscale skipped — quality already >=1440p (no upscaling needed).")

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
            from gpu_renderer import GPURenderer
            # ── GPU path: sequential, group by size ──
            groups: Dict[tuple, List[dict]] = {}
            for item in items:
                sz = item.get("_inferred_size") or (0, 0)
                kval = (sz[0], sz[1], item.get("quality", 1080), item.get("opts", RenderOptions()))
                groups.setdefault(kval, []).append(item)

            _batch_prog = ProgressTracker(
                n, "Batch",
                phase_weights=BATCH_PHASE_WEIGHTS,
                hide_rate=True,
                show_terminal=show_progress,
            ) if show_progress else None
            renderer = None
            prev_key = None
            try:
                for key in sorted(groups.keys()):
                    group = groups[key]
                    if renderer is None or key != prev_key:
                        if renderer is not None:
                            renderer.cleanup()
                        w, h, quality, batch_opts = key
                        from geometry import compute_layout
                        layout = compute_layout(quality, w, h, batch_opts.grid_only, adjust_height=batch_opts.adjust_height)
                        tile_size = layout["tile_size"]
                        renderer = GPURenderer(w, h, tile_size, pad=layout["pad"], header_h=layout["header_h"], panel_w=layout["panel_w"], canvas_w=layout["canvas_w"], canvas_h=layout["canvas_h"], opts=batch_opts)

                    for item in group:
                        if cancel_check and cancel_check():
                            raise CancelError()
                        kwargs = {k: v for k, v in item.items()
                                  if k not in ("solution", "output_path", "_inferred_size")}
                        self.generate_simple_replay(
                            solution=item["solution"],
                            output_path=item["output_path"],
                            use_gpu=True,
                            show_progress=False,
                            cancel_check=cancel_check,
                            gpu_renderer=renderer,
                            **kwargs,
                        )
                        output_paths.append(item["output_path"])
                        if external_progress_cb:
                            external_progress_cb(len(output_paths), n)
                        if _batch_prog:
                            _batch_prog(len(output_paths), n)

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
                _batch_prog = ProgressTracker(
                    n, "Batch",
                    phase_weights=BATCH_PHASE_WEIGHTS,
                    hide_rate=True,
                    show_terminal=show_progress,
                ) if show_progress else None
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
    parser.add_argument("--force-rows", action="store_true", default=False, help="Force row stripes (disable grids detection)")
    parser.add_argument("--force-columns", action="store_true", default=False, help="Force column stripes (disable grids detection)")
    parser.add_argument("--quality", type=int, default=1080, help="Target video quality (360/480/720/1080/1440/2160)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor")
    parser.add_argument("--fps", type=int, default=60, help="Output video frame rate (default: 60)")
    parser.add_argument("--temp-dir", type=str, default=None, help="Temp directory for frames")
    parser.add_argument("--grid1-color", type=str, default=None, help="Grid 1 color as hex (e.g. FF0000 for red)")
    parser.add_argument("--grid2-color", type=str, default=None, help="Grid 2 color as hex (e.g. 0000FF for blue)")
    parser.add_argument("--tile-bg-color", type=str, default=None, help="Tile background color as hex")

    args = parser.parse_args()
    opts = RenderOptions(
        grid1_color=parse_hex_color(args.grid1_color),
        grid2_color=parse_hex_color(args.grid2_color),
        tile_bg_color=parse_hex_color(args.tile_bg_color),
    )

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
        force_rows=args.force_rows,
        force_columns=args.force_columns,
        speed_factor=args.speed,
        quality=args.quality,
        fps=args.fps,
        show_progress=True,
        opts=opts,
    )


if __name__ == "__main__":
    main()

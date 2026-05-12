import math
import numpy as np
from typing import List, Dict, Tuple, Optional

_MOVE_DIRS = {
    'R': (0, -1), 'L': (0, 1),
    'U': (1, 0), 'D': (-1, 0),
}

CT_MAP = {'fringe': 1, 'grids1': 2, 'grids2': 3}


def move_matrix_inplace(mc_flat, move, zp_idx, w):
    dr, dc = _MOVE_DIRS[move]
    nr_idx = zp_idx + dr * w + dc
    mc_flat[zp_idx], mc_flat[nr_idx] = mc_flat[nr_idx], mc_flat[zp_idx]


def find_zero(matrix, w, h):
    for i in range(h):
        for j in range(w):
            if matrix[i][j] == 0:
                return i, j
    return -1, -1


def number_is_solved(num, row, col, w):
    if num == 0:
        return False
    return (num - 1) // w == row and (num - 1) % w == col


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


def grids_solved(matrix, width, height, offset_w, offset_h, grids_type, width_initial):
    if grids_type == 1:
        new_h = math.ceil(height / 2) + offset_h
        for row in range(offset_h, new_h):
            for col in range(offset_w, width + offset_w):
                num = matrix[row][col]
                if num != 0 and not number_is_solved(num, row, col, width_initial):
                    return False
    if grids_type == 2:
        new_w = math.ceil(width / 2) + offset_w
        for row in range(offset_h, height + offset_h):
            for col in range(offset_w, new_w):
                num = matrix[row][col]
                if num != 0 and not number_is_solved(num, row, col, width_initial):
                    return False
    return True


def get_grids_parts(matrix_before, solution, width, height):
    if width < 6 and height < 6:
        return None
    first = [row[:] for row in matrix_before]
    mc_flat = np.array(matrix_before, dtype=np.int32).flatten()
    zp = find_zero(matrix_before, width, height)
    zp_idx = zp[0] * width + zp[1]
    for move in solution:
        move_matrix_inplace(mc_flat, move, zp_idx, width)
        dr, dc = _MOVE_DIRS[move]
        zp_idx += dr * width + dc
    second = mc_flat.reshape(height, width).tolist()
    return first, second


def analyse_grids(matrix, solution, width_initial, height_initial, width, height, offset_w, offset_h, moves_offset):
    mc_flat = np.array(matrix, dtype=np.int32).flatten()
    zp = find_zero(matrix, width_initial, height_initial)
    zp_idx = zp[0] * width_initial + zp[1]

    for mi in range(len(solution)):
        move = solution[mi]
        move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
        dr, dc = _MOVE_DIRS[move]
        zp_idx += dr * width_initial + dc

        mc_2d = mc_flat.reshape(height_initial, width_initial)
        gs = guess_grids(mc_2d, width, height, offset_w, offset_h, width_initial)
        if gs != 0:
            grids_started = mi
            enable_gs = gs
            girds_unsolved_last = None
            before_flat = mc_flat.copy()
            before_zp = zp_idx

            for gst_id in range(grids_started + 1, len(solution)):
                move2 = solution[gst_id]
                move_matrix_inplace(mc_flat, move2, zp_idx, width_initial)
                dr2, dc2 = _MOVE_DIRS[move2]
                zp_idx += dr2 * width_initial + dc2
                mc_2d = mc_flat.reshape(height_initial, width_initial)
                if not grids_solved(mc_2d, width, height, offset_w, offset_h, enable_gs, width_initial):
                    girds_unsolved_last = gst_id
                else:
                    break

            if girds_unsolved_last is None:
                return {"enableGridsStatus": -1, "width": width, "height": height, "offsetW": offset_w, "offsetH": offset_h}
            grids_stopped = girds_unsolved_last + 1
            sol1 = solution[grids_started + 1: grids_stopped + 2]
            sol2 = solution[grids_stopped + 2:]
            matrix_before = before_flat.reshape(height_initial, width_initial).tolist()
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
                    "nextLayerFirst": analyse_grids(parts[0], sol1, width_initial, height_initial, w1, h1, ow1, oh1, moves_offset + grids_started + 1),
                    "nextLayerSecond": analyse_grids(parts[1], sol2, width_initial, height_initial, w2, h2, ow2, oh2, moves_offset + grids_stopped + 1)
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
                    "nextLayerFirst": analyse_grids(parts[0], sol1, width_initial, height_initial, w1, h1, ow1, oh1, moves_offset + grids_started + 1),
                    "nextLayerSecond": analyse_grids(parts[1], sol2, width_initial, height_initial, w2, h2, ow2, oh2, moves_offset + grids_stopped + 1)
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


def analyse_grids_initial(matrix, solution):
    h = len(matrix)
    w = len(matrix[0])
    return analyse_grids(matrix, solution, w, h, w, h, 0, 0, 0)


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


def filter_grid_stages(grid_states, width, height, add_last=None):
    """Return sorted move‑indices of relevant grid‑stage boundaries.

    A grid stage is relevant when its active zone covers at least half
    the puzzle width *and* height.  Always includes ``0``.  When
    *add_last* is given it is appended as the terminal boundary.

    Logic shared by splits formatting and frame generation.
    """
    keys = sorted([k for k in grid_states.keys() if isinstance(k, (int, float))])
    out = [0]
    for k in keys:
        if k == 0:
            continue
        az = grid_states[k]["activeZone"]
        if az["width"] + 1 >= width / 2 and az["height"] + 1 >= height / 2:
            out.append(k)
    if add_last is not None:
        out.append(add_last)
    return out


def get_grid_states(solution: str, scramble: str) -> Dict:
    """Convenience: puzzle matrix + grids analysis + stats in one call."""
    from replay_generator import scramble_to_puzzle, expand_solution
    matrix = scramble_to_puzzle(scramble)
    expanded = expand_solution(solution)
    grids_data = analyse_grids_initial(matrix, expanded)
    return generate_grids_stats(grids_data)

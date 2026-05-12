import math
import numpy as np
from typing import List, Dict, Tuple, Optional

from sliding_puzzles import _MOVE_DIRS, move_matrix_inplace, find_zero

CT_MAP = {'fringe': 1, 'grids1': 2, 'grids2': 3}


def _check_top_bottom_np(matrix, width, height, offset_w, offset_h, width_initial):
    new_h = math.ceil(height / 2) + offset_h
    sub = matrix[offset_h:new_h, offset_w:offset_w + width]
    nonzero = sub != 0
    if not np.any(nonzero):
        return True
    targets = (sub[nonzero] - 1) // width_initial
    if np.any(targets >= new_h):
        return False
    actual_rows = (sub - 1) // width_initial
    actual_cols = (sub - 1) % width_initial
    exp_rows = np.arange(offset_h, new_h)[:, None]
    exp_cols = np.arange(offset_w, offset_w + width)[None, :]
    solved = nonzero & (actual_rows == exp_rows) & (actual_cols == exp_cols)
    return width * (new_h - offset_h) / 3 > np.sum(solved)


def _check_left_right_np(matrix, width, height, offset_w, offset_h, width_initial):
    new_w = math.ceil(width / 2) + offset_w
    sub = matrix[offset_h:offset_h + height, offset_w:new_w]
    nonzero = sub != 0
    if not np.any(nonzero):
        return True
    targets = (sub[nonzero] - 1) % width_initial
    if np.any(targets >= new_w):
        return False
    actual_rows = (sub - 1) // width_initial
    actual_cols = (sub - 1) % width_initial
    exp_rows = np.arange(offset_h, offset_h + height)[:, None]
    exp_cols = np.arange(offset_w, new_w)[None, :]
    solved = nonzero & (actual_rows == exp_rows) & (actual_cols == exp_cols)
    return height * (new_w - offset_w) / 3 > np.sum(solved)


def guess_grids(matrix, width, height, offset_w, offset_h, width_initial):
    if width < 6 and height < 6:
        return 0
    if height > 5 and _check_top_bottom_np(matrix, width, height, offset_w, offset_h, width_initial):
        return 1
    if width > 5 and _check_left_right_np(matrix, width, height, offset_w, offset_h, width_initial):
        return 2
    return 0


def _grids_solved_np(matrix, width, height, offset_w, offset_h, grids_type, width_initial):
    if grids_type == 1:
        new_h = math.ceil(height / 2) + offset_h
        sub = matrix[offset_h:new_h, offset_w:offset_w + width]
        nonzero = sub != 0
        if not np.any(nonzero):
            return True
        actual_rows = (sub[nonzero] - 1) // width_initial
        actual_cols = (sub[nonzero] - 1) % width_initial
        exp_rows = np.repeat(np.arange(offset_h, new_h), width)[nonzero.ravel()]
        exp_cols = np.tile(np.arange(offset_w, offset_w + width), new_h - offset_h)[nonzero.ravel()]
        return np.all((actual_rows == exp_rows) & (actual_cols == exp_cols))
    if grids_type == 2:
        new_w = math.ceil(width / 2) + offset_w
        sub = matrix[offset_h:offset_h + height, offset_w:new_w]
        nonzero = sub != 0
        if not np.any(nonzero):
            return True
        actual_rows = (sub[nonzero] - 1) // width_initial
        actual_cols = (sub[nonzero] - 1) % width_initial
        exp_rows = np.repeat(np.arange(offset_h, offset_h + height), new_w - offset_w)[nonzero.ravel()]
        exp_cols = np.tile(np.arange(offset_w, new_w), height)[nonzero.ravel()]
        return np.all((actual_rows == exp_rows) & (actual_cols == exp_cols))
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
    return first, mc_flat.reshape(height, width).tolist()


def _to_relative(node, base_offset):
    """Convert absolute move indices to relative (0-based at base_offset)."""
    if node is None or node.get("enableGridsStatus") == -1:
        return node
    shifted = {
        "enableGridsStatus": node["enableGridsStatus"],
        "gridsStarted": node["gridsStarted"] - base_offset,
        "gridsStopped": node["gridsStopped"] - base_offset,
        "width": node["width"], "height": node["height"],
        "offsetW": node["offsetW"], "offsetH": node["offsetH"],
        "nextLayerFirst": _to_relative(node.get("nextLayerFirst"), base_offset),
        "nextLayerSecond": _to_relative(node.get("nextLayerSecond"), base_offset),
    }
    return shifted


def _apply_offset(skeleton, new_offset):
    """Clone a relative skeleton, shifting indices by new_offset."""
    if skeleton is None or skeleton.get("enableGridsStatus") == -1:
        return skeleton
    shifted = {
        "enableGridsStatus": skeleton["enableGridsStatus"],
        "gridsStarted": skeleton["gridsStarted"] + new_offset,
        "gridsStopped": skeleton["gridsStopped"] + new_offset,
        "width": skeleton["width"], "height": skeleton["height"],
        "offsetW": skeleton["offsetW"], "offsetH": skeleton["offsetH"],
        "nextLayerFirst": _apply_offset(skeleton.get("nextLayerFirst"), new_offset),
        "nextLayerSecond": _apply_offset(skeleton.get("nextLayerSecond"), new_offset),
    }
    return shifted


def analyse_grids(matrix, solution, width_initial, height_initial, width, height, offset_w, offset_h, moves_offset, shape_cache=None):
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

            for gst_id in range(grids_started + 1, len(solution)):
                move2 = solution[gst_id]
                move_matrix_inplace(mc_flat, move2, zp_idx, width_initial)
                dr2, dc2 = _MOVE_DIRS[move2]
                zp_idx += dr2 * width_initial + dc2
                mc_2d = mc_flat.reshape(height_initial, width_initial)
                if not _grids_solved_np(mc_2d, width, height, offset_w, offset_h, enable_gs, width_initial):
                    girds_unsolved_last = gst_id
                else:
                    break

            if girds_unsolved_last is None:
                return {"enableGridsStatus": -1, "width": width, "height": height, "offsetW": offset_w, "offsetH": offset_h}
            grids_stopped = girds_unsolved_last + 1
            sol1 = solution[grids_started + 1: grids_stopped + 2]
            sol2 = solution[grids_stopped + 2:]

            if shape_cache is None:
                shape_cache = {}

            # Compute half dimensions
            if enable_gs == 1:
                w1 = w2 = width
                ow1 = ow2 = offset_w
                h1 = math.ceil(height / 2)
                h2 = height - h1
                oh1 = offset_h
                oh2 = h1 + offset_h
            else:
                w1 = math.ceil(width / 2)
                w2 = width - w1
                ow1 = offset_w
                ow2 = w1 + offset_w
                h1 = h2 = height
                oh1 = oh2 = offset_h

            key1 = (w1, h1, ow1, oh1)
            key2 = (w2, h2, ow2, oh2)

            # Only call get_grids_parts if at least one half needs analysis
            parts = None
            if key1 not in shape_cache or key2 not in shape_cache:
                matrix_before = before_flat.reshape(height_initial, width_initial).tolist()
                parts = get_grids_parts(matrix_before, sol1, width_initial, height_initial)

            # Resolve first half
            if key1 in shape_cache:
                next_first = _apply_offset(shape_cache[key1], moves_offset + grids_started + 1)
            elif parts is not None:
                next_first = analyse_grids(parts[0], sol1, width_initial, height_initial, w1, h1, ow1, oh1, moves_offset + grids_started + 1, shape_cache)
                shape_cache[key1] = _to_relative(next_first, moves_offset + grids_started + 1)
            else:
                next_first = None

            # Resolve second half
            if key2 in shape_cache:
                next_second = _apply_offset(shape_cache[key2], moves_offset + grids_stopped + 1)
            elif parts is not None:
                next_second = analyse_grids(parts[1], sol2, width_initial, height_initial, w2, h2, ow2, oh2, moves_offset + grids_stopped + 1, shape_cache)
                shape_cache[key2] = _to_relative(next_second, moves_offset + grids_stopped + 1)
            else:
                next_second = None

            return {
                "enableGridsStatus": enable_gs,
                "gridsStarted": grids_started + moves_offset,
                "gridsStopped": grids_stopped + moves_offset,
                "width": width, "height": height,
                "offsetW": offset_w, "offsetH": offset_h,
                "nextLayerFirst": next_first,
                "nextLayerSecond": next_second,
            }
    return {"enableGridsStatus": -1, "width": width, "height": height, "offsetW": offset_w, "offsetH": offset_h}


def analyse_grids_initial(matrix, solution):
    h = len(matrix)
    w = len(matrix[0])
    return analyse_grids(matrix, solution, w, h, w, h, 0, 0, 0, shape_cache={})


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

    # Deduplicate: states with identical (mainColors, secondaryColors)
    # share the same dict object so tile color cache id() hits.
    seen = {}
    for k in list(levels.keys()):
        sig = (str(levels[k]["mainColors"]), str(levels[k]["secondaryColors"]))
        if sig in seen:
            levels[k] = seen[sig]
        else:
            seen[sig] = levels[k]

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
    keys = [k for k in grid_states.keys() if isinstance(k, (int, float))]
    keys.sort()
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


def get_grid_states(solution: str = None, scramble: str = None, *, matrix=None, expanded_solution=None, grids_data=None) -> Dict:
    """Convenience: puzzle matrix + grids analysis + stats in one call.
    
    Accepts either (solution, scramble) strings to parse from scratch,
    or pre-computed ``matrix`` + ``expanded_solution``, or a ready-made
    ``grids_data`` tree (skips all analysis).
    """
    if grids_data is not None:
        return generate_grids_stats(grids_data)
    from replay_generator import scramble_to_puzzle, expand_solution
    if matrix is None:
        matrix = scramble_to_puzzle(scramble)
    if expanded_solution is None:
        expanded_solution = expand_solution(solution)
    grids_data = analyse_grids_initial(matrix, expanded_solution)
    return generate_grids_stats(grids_data)

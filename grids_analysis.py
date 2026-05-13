import math
import time
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from sliding_puzzles import _MOVE_DIRS, move_matrix_inplace, find_zero
from debug_log import get_logger

log = get_logger()

CT_MAP = {'fringe': 1, 'grids1': 2, 'grids2': 3}


def _check_gs(mc_2d, width, height, offset_w, offset_h,
              can_split_tb, can_split_lr, tb_new_h, lr_new_w,
              tb_exp_rows_bc, tb_exp_cols_bc,
              lr_exp_rows_bc, lr_exp_cols_bc,
              _row_of, _col_of):
    gs = 0
    if can_split_tb:
        sub = mc_2d[offset_h:tb_new_h, offset_w:offset_w + width]
        nonzero = sub != 0
        if not np.any(nonzero):
            gs = 1
        else:
            vals = sub[nonzero] - 1
            targets = _row_of[vals]
            if not np.any(targets >= tb_new_h):
                all_vals = sub - 1
                actual_rows = _row_of[all_vals]
                actual_cols = _col_of[all_vals]
                solved = nonzero & (actual_rows == tb_exp_rows_bc) & (actual_cols == tb_exp_cols_bc)
                if width * (tb_new_h - offset_h) / 3 > np.sum(solved):
                    gs = 1
    if gs == 0 and can_split_lr:
        sub = mc_2d[offset_h:offset_h + height, offset_w:lr_new_w]
        nonzero = sub != 0
        if not np.any(nonzero):
            gs = 2
        else:
            vals = sub[nonzero] - 1
            targets = _col_of[vals]
            if not np.any(targets >= lr_new_w):
                all_vals = sub - 1
                actual_rows = _row_of[all_vals]
                actual_cols = _col_of[all_vals]
                solved = nonzero & (actual_rows == lr_exp_rows_bc) & (actual_cols == lr_exp_cols_bc)
                if height * (lr_new_w - offset_w) / 3 > np.sum(solved):
                    gs = 2
    return gs


def _check_solved(mc_2d, enable_gs, width, height, offset_w, offset_h,
                  tb_new_h, lr_new_w, can_split_tb, can_split_lr,
                  tb_gs_full_rows, tb_gs_full_cols,
                  lr_gs_full_rows, lr_gs_full_cols,
                  _row_of, _col_of):
    if enable_gs == 1 and can_split_tb:
        sub = mc_2d[offset_h:tb_new_h, offset_w:offset_w + width]
        nonzero = sub != 0
        if not np.any(nonzero):
            return True
        vals = sub[nonzero] - 1
        actual_rows = _row_of[vals]
        actual_cols = _col_of[vals]
        nzf = nonzero.ravel()
        exp_rows_sub = tb_gs_full_rows[nzf]
        exp_cols_sub = tb_gs_full_cols[nzf]
        return np.all((actual_rows == exp_rows_sub) & (actual_cols == exp_cols_sub))
    if enable_gs == 2 and can_split_lr:
        sub = mc_2d[offset_h:offset_h + height, offset_w:lr_new_w]
        nonzero = sub != 0
        if not np.any(nonzero):
            return True
        vals = sub[nonzero] - 1
        actual_rows = _row_of[vals]
        actual_cols = _col_of[vals]
        nzf = nonzero.ravel()
        exp_rows_sub = lr_gs_full_rows[nzf]
        exp_cols_sub = lr_gs_full_cols[nzf]
        return np.all((actual_rows == exp_rows_sub) & (actual_cols == exp_cols_sub))
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



def analyse_grids(matrix, solution, width_initial, height_initial, width, height, offset_w, offset_h, moves_offset, shape_cache=None, progress_callback=None, progress_total=None, _timing=None, _lock=None, _row_of=None, _col_of=None):
    if _timing is None:
        _timing = {"main_loop": 0.0, "scan_fwd": 0.0, "get_parts": 0.0, "n_calls": 0}
    if _lock is None:
        _lock = threading.Lock()
    _timing["n_calls"] += 1

    mc_flat = np.array(matrix, dtype=np.int32).flatten()
    zp = find_zero(matrix, width_initial, height_initial)
    zp_idx = zp[0] * width_initial + zp[1]
    init_flat = mc_flat.copy()
    init_zp = zp_idx

    sol_len = len(solution)
    total_for_progress = progress_total if progress_total is not None else sol_len

    tb_new_h = math.ceil(height / 2) + offset_h
    lr_new_w = math.ceil(width / 2) + offset_w
    can_split_tb = height > 5
    can_split_lr = width > 5
    if can_split_tb:
        tb_sub_h = tb_new_h - offset_h
        tb_exp_rows_bc = np.arange(offset_h, tb_new_h)[:, None]
        tb_exp_cols_bc = np.arange(offset_w, offset_w + width)[None, :]
        tb_gs_full_rows = np.repeat(np.arange(offset_h, tb_new_h), width)
        tb_gs_full_cols = np.tile(np.arange(offset_w, offset_w + width), tb_sub_h)
    else:
        tb_exp_rows_bc = np.empty((0, 0))
        tb_exp_cols_bc = np.empty((0, 0))
        tb_gs_full_rows = np.empty(0)
        tb_gs_full_cols = np.empty(0)
    if can_split_lr:
        lr_sub_w = lr_new_w - offset_w
        lr_exp_rows_bc = np.arange(offset_h, offset_h + height)[:, None]
        lr_exp_cols_bc = np.arange(offset_w, lr_new_w)[None, :]
        lr_gs_full_rows = np.repeat(np.arange(offset_h, offset_h + height), lr_sub_w)
        lr_gs_full_cols = np.tile(np.arange(offset_w, lr_new_w), height)
    else:
        lr_exp_rows_bc = np.empty((0, 0))
        lr_exp_cols_bc = np.empty((0, 0))
        lr_gs_full_rows = np.empty(0)
        lr_gs_full_cols = np.empty(0)

    if not can_split_tb and not can_split_lr:
        return {"enableGridsStatus": -1, "width": width, "height": height, "offsetW": offset_w, "offsetH": offset_h}

    # Shared args dict to reduce parameter passing
    _gs_args = (width, height, offset_w, offset_h, can_split_tb, can_split_lr, tb_new_h, lr_new_w, tb_exp_rows_bc, tb_exp_cols_bc, lr_exp_rows_bc, lr_exp_cols_bc, _row_of, _col_of)

    _t_loop = time.time()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Find grids_started (first position where gs != 0)
    # ═══════════════════════════════════════════════════════════════
    gs_type = 0
    grids_started = -1

    if sol_len <= 16:
        # Linear scan for tiny solutions
        for mi in range(sol_len):
            move = solution[mi]
            move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
            dr, dc = _MOVE_DIRS[move]
            zp_idx += dr * width_initial + dc
            gs = _check_gs(mc_flat.reshape(height_initial, width_initial), *_gs_args)
            if progress_callback:
                with _lock:
                    progress_callback(moves_offset + mi, total_for_progress)
            if gs:
                gs_type = gs
                grids_started = mi
                break
        if grids_started == -1:
            with _lock:
                _timing["main_loop"] += time.time() - _t_loop
            return {"enableGridsStatus": -1, "width": width, "height": height, "offsetW": offset_w, "offsetH": offset_h}
    else:
        # Linear check first 3 positions
        n_linear = min(3, sol_len)
        for mi in range(n_linear):
            move = solution[mi]
            move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
            dr, dc = _MOVE_DIRS[move]
            zp_idx += dr * width_initial + dc
            gs = _check_gs(mc_flat.reshape(height_initial, width_initial), *_gs_args)
            if progress_callback:
                with _lock:
                    progress_callback(moves_offset + mi, total_for_progress)
            if gs:
                gs_type = gs
                grids_started = mi
                break

        if grids_started == -1:
            # Gallop
            cur_pos = n_linear - 1
            lo_state = mc_flat.copy()
            lo_zp = zp_idx
            step = 4

            while cur_pos + step < sol_len:
                target = cur_pos + step
                snapshot = lo_state
                snapshot_zp = lo_zp

                for i in range(cur_pos + 1, target + 1):
                    move = solution[i]
                    move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                    dr, dc = _MOVE_DIRS[move]
                    zp_idx += dr * width_initial + dc

                cur_pos = target
                gs = _check_gs(mc_flat.reshape(height_initial, width_initial), *_gs_args)

                if progress_callback:
                    with _lock:
                        progress_callback(moves_offset + cur_pos, total_for_progress)

                if gs:
                    snapshot_pos = target - step
                    lo, hi = 0, step

                    while lo < hi - 1:
                        mid = (lo + hi) // 2
                        mc_flat[:] = snapshot
                        zp_idx = snapshot_zp
                        for i in range(1, mid + 1):
                            pos = snapshot_pos + i
                            move = solution[pos]
                            move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                            dr, dc = _MOVE_DIRS[move]
                            zp_idx += dr * width_initial + dc
                        gs_mid = _check_gs(mc_flat.reshape(height_initial, width_initial), *_gs_args)
                        if gs_mid:
                            hi = mid
                        else:
                            lo = mid

                    grids_started = snapshot_pos + hi
                    # Restore state to grids_started
                    mc_flat[:] = snapshot
                    zp_idx = snapshot_zp
                    for i in range(1, hi + 1):
                        pos = snapshot_pos + i
                        move = solution[pos]
                        move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                        dr, dc = _MOVE_DIRS[move]
                        zp_idx += dr * width_initial + dc
                    gs_type = _check_gs(mc_flat.reshape(height_initial, width_initial), *_gs_args)
                    break

                lo_state = mc_flat.copy()
                lo_zp = zp_idx
                step = min(step * 2, 256)

        if grids_started == -1:
            # Fallback: restore to initial state, do full linear scan
            # (gs can be non-monotonic — galloping may skip the first non-zero)
            mc_flat[:] = init_flat
            zp_idx = init_zp
            for mi in range(sol_len):
                move = solution[mi]
                move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                dr, dc = _MOVE_DIRS[move]
                zp_idx += dr * width_initial + dc
                gs = _check_gs(mc_flat.reshape(height_initial, width_initial), *_gs_args)
                if progress_callback:
                    with _lock:
                        progress_callback(moves_offset + mi, total_for_progress)
                if gs:
                    gs_type = gs
                    grids_started = mi
                    break

    with _lock:
        _timing["main_loop"] += time.time() - _t_loop

    if grids_started == -1:
        return {"enableGridsStatus": -1, "width": width, "height": height, "offsetW": offset_w, "offsetH": offset_h}

    # Save transition state for child calls
    transition_flat = mc_flat.copy()
    transition_zp = zp_idx
    enable_gs = gs_type

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2: Find grids_stopped (first solved position)
    # ═══════════════════════════════════════════════════════════════
    _t_scan = time.time()
    scan_solution = solution[grids_started + 1:]
    scan_len = len(scan_solution)
    grids_stopped = -1

    _solv_args = (enable_gs, width, height, offset_w, offset_h, tb_new_h, lr_new_w, can_split_tb, can_split_lr, tb_gs_full_rows, tb_gs_full_cols, lr_gs_full_rows, lr_gs_full_cols, _row_of, _col_of)

    if scan_len == 0:
        with _lock:
            _timing["scan_fwd"] += time.time() - _t_scan
        return {"enableGridsStatus": -1, "width": width, "height": height, "offsetW": offset_w, "offsetH": offset_h}

    if scan_len <= 16:
        for si in range(scan_len):
            move = scan_solution[si]
            move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
            dr, dc = _MOVE_DIRS[move]
            zp_idx += dr * width_initial + dc
            if _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args):
                grids_stopped = grids_started + 1 + si
                break
    else:
        n_linear_s = min(4, scan_len)
        for si in range(n_linear_s):
            move = scan_solution[si]
            move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
            dr, dc = _MOVE_DIRS[move]
            zp_idx += dr * width_initial + dc
            if _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args):
                grids_stopped = grids_started + 1 + si
                break

        if grids_stopped == -1:
            cur_pos = n_linear_s - 1
            lo_state = mc_flat.copy()
            lo_zp = zp_idx
            step = 4

            while cur_pos + step < scan_len:
                target = cur_pos + step
                snapshot = lo_state
                snapshot_zp = lo_zp

                for i in range(cur_pos + 1, target + 1):
                    move = scan_solution[i]
                    move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                    dr, dc = _MOVE_DIRS[move]
                    zp_idx += dr * width_initial + dc

                cur_pos = target
                solved = _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args)

                if solved:
                    snapshot_pos = target - step
                    lo, hi = 0, step

                    while lo < hi - 1:
                        mid = (lo + hi) // 2
                        mc_flat[:] = snapshot
                        zp_idx = snapshot_zp
                        for i in range(1, mid + 1):
                            pos = snapshot_pos + i
                            move = scan_solution[pos]
                            move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                            dr, dc = _MOVE_DIRS[move]
                            zp_idx += dr * width_initial + dc
                        if _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args):
                            hi = mid
                        else:
                            lo = mid

                    grids_stopped = grids_started + 1 + snapshot_pos + hi
                    break

                lo_state = mc_flat.copy()
                lo_zp = zp_idx
                step = min(step * 2, 256)

        if grids_stopped == -1:
            # Fallback: restore to start of scan, do full linear scan
            mc_flat[:] = transition_flat
            zp_idx = transition_zp
            for si in range(scan_len):
                move = scan_solution[si]
                move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                dr, dc = _MOVE_DIRS[move]
                zp_idx += dr * width_initial + dc
                if _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args):
                    grids_stopped = grids_started + 1 + si
                    break

    with _lock:
        _timing["scan_fwd"] += time.time() - _t_scan

    if grids_stopped == -1:
        return {"enableGridsStatus": -1, "width": width, "height": height, "offsetW": offset_w, "offsetH": offset_h}

    girds_unsolved_last = grids_stopped - 1
    grids_stopped_final = girds_unsolved_last + 1

    sol1 = solution[grids_started + 1: grids_stopped_final + 2]
    sol2 = solution[grids_stopped_final + 2:]

    if shape_cache is None:
        shape_cache = {}

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

    cache_hit = (key1 in shape_cache and key2 in shape_cache)

    _t_parts = time.time()
    if cache_hit:
        next_first = _apply_offset(shape_cache[key1], moves_offset + grids_started + 1)
        next_second = _apply_offset(shape_cache[key2], moves_offset + grids_stopped_final + 1)
    else:
        parts = get_grids_parts(transition_flat.reshape(height_initial, width_initial).tolist(), sol1, width_initial, height_initial)
        with _lock:
            _timing["get_parts"] += time.time() - _t_parts

        log.info(f"  analyse_grids split@{moves_offset + grids_started}: {width}x{height} region, sol_len={sol_len}, scan_range={grids_stopped - grids_started - 1}, cache_hit=False, parts={time.time() - _t_parts:.3f}s")

        child_args = []
        if key1 in shape_cache:
            next_first = _apply_offset(shape_cache[key1], moves_offset + grids_started + 1)
        elif parts is not None:
            child_args.append((1, parts[0], sol1, w1, h1, ow1, oh1, moves_offset + grids_started + 1, key1))

        if key2 in shape_cache:
            next_second = _apply_offset(shape_cache[key2], moves_offset + grids_stopped_final + 1)
        elif parts is not None:
            child_args.append((2, parts[1], sol2, w2, h2, ow2, oh2, moves_offset + grids_stopped_final + 1, key2))

        if child_args:
            def run_child(idx, mat, sol, w, h, ow, oh, off, key):
                r = analyse_grids(mat, sol, width_initial, height_initial, w, h, ow, oh, off, shape_cache, progress_callback, progress_total, _timing, _lock, _row_of, _col_of)
                shape_cache[key] = _to_relative(r, off)
                return r

            if len(child_args) == 2:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    f1 = pool.submit(run_child, *child_args[0])
                    f2 = pool.submit(run_child, *child_args[1])
                    r1, r2 = f1.result(), f2.result()
                if child_args[0][0] == 1:
                    next_first, next_second = r1, r2
                else:
                    next_first, next_second = r2, r1
            else:
                r = run_child(*child_args[0])
                if child_args[0][0] == 1:
                    next_first = r
                else:
                    next_second = r
        else:
            # Both cached from parts computation
            pass

    if not cache_hit and not child_args:
        next_first = None
        next_second = None

    return {
        "enableGridsStatus": enable_gs,
        "gridsStarted": grids_started + moves_offset,
        "gridsStopped": grids_stopped_final + moves_offset,
        "width": width, "height": height,
        "offsetW": offset_w, "offsetH": offset_h,
        "nextLayerFirst": next_first,
        "nextLayerSecond": next_second,
    }


def analyse_grids_initial(matrix, solution, progress_callback=None):
    h = len(matrix)
    w = len(matrix[0])
    total_cells = w * h
    _row_of = np.arange(total_cells) // w
    _col_of = np.arange(total_cells) % w
    _timing = {"main_loop": 0.0, "scan_fwd": 0.0, "get_parts": 0.0, "n_calls": 0}
    _lock = threading.Lock()
    result = analyse_grids(matrix, solution, w, h, w, h, 0, 0, 0, shape_cache={}, progress_callback=progress_callback, progress_total=len(solution), _timing=_timing, _lock=_lock, _row_of=_row_of, _col_of=_col_of)
    total = _timing["main_loop"] + _timing["scan_fwd"] + _timing["get_parts"]
    log.info(f"  analysis timing: main_loop={_timing['main_loop']:.3f}s, scan_fwd={_timing['scan_fwd']:.3f}s, get_parts={_timing['get_parts']:.3f}s, n_calls={_timing['n_calls']}, total={total:.3f}s")
    return result


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


def get_grid_states(solution=None, scramble=None, *, matrix=None, expanded_solution=None, grids_data=None):
    if grids_data is not None:
        return generate_grids_stats(grids_data)
    from replay_generator import scramble_to_puzzle, expand_solution
    if matrix is None:
        matrix = scramble_to_puzzle(scramble)
    if expanded_solution is None:
        expanded_solution = expand_solution(solution)
    grids_data = analyse_grids_initial(matrix, expanded_solution)
    return generate_grids_stats(grids_data)

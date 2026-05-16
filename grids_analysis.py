import math
import time
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

from sliding_puzzles import _MOVE_DIRS, move_matrix_inplace, find_zero
from debug_log import get_logger, CancelError

log = get_logger()

CT_MAP = {'fringe': 1, 'grids1': 2, 'grids2': 3}


def _wrong_tiles(mc_2d, full_h, full_w, _row_of, _col_of):
    all_safe = mc_2d - 1
    all_safe = np.where(mc_2d == 0, 0, all_safe)
    ar_all = _row_of[all_safe]
    ac_all = _col_of[all_safe]
    exp_rows_all = np.arange(full_h)[:, None]
    exp_cols_all = np.arange(full_w)[None, :]
    solved_all = (mc_2d != 0) & (ar_all == exp_rows_all) & (ac_all == exp_cols_all)
    wrong = (mc_2d != 0) & ~solved_all
    return np.unique(mc_2d[wrong])

def _corner_solved(mc_2d, full_h, full_w, _row_of, _col_of):
    corner_h = min(5, full_h)
    corner_w = min(5, full_w)
    if corner_h < 5 or corner_w < 5:
        return False
    corner = mc_2d[full_h - corner_h:, full_w - corner_w:]
    mask = corner != 0
    if np.sum(mask) < corner_h * corner_w * 0.5:
        return False
    corner_vals = corner[mask] - 1
    ar = _row_of[corner_vals]
    ac = _col_of[corner_vals]
    rows, cols = np.where(mask)
    solved_count = np.sum((ar == rows + full_h - corner_h) & (ac == cols + full_w - corner_w))
    return solved_count >= corner_h * corner_w * 0.5

def _check_gs(mc_2d, width, height, offset_w, offset_h,
              can_split_tb, can_split_lr, tb_new_h, lr_new_w,
              tb_exp_rows_bc, tb_exp_cols_bc,
              lr_exp_rows_bc, lr_exp_cols_bc,
              _row_of, _col_of, full_h, full_w, cycled=np.array([], dtype=np.int32)):
    if len(cycled) == 0:
        cycled = _wrong_tiles(mc_2d, full_h, full_w, _row_of, _col_of) if _corner_solved(mc_2d, full_h, full_w, _row_of, _col_of) else np.array([], dtype=np.int32)
    gs = 0
    if can_split_tb:
        sub = mc_2d[offset_h:tb_new_h, offset_w:offset_w + width]
        nonzero = sub != 0
        if not np.any(nonzero):
            gs = 1
        else:
            all_vals = sub - 1
            actual_rows = _row_of[all_vals]
            actual_cols = _col_of[all_vals]
            solved = nonzero & (actual_rows == tb_exp_rows_bc) & (actual_cols == tb_exp_cols_bc)
            n_solved = np.sum(solved)
            total = width * (tb_new_h - offset_h)
            vals = sub[nonzero] - 1
            targets = _row_of[vals]
            if not np.any(targets >= tb_new_h) and total / 3 > n_solved:
                gs = 1
            elif gs == 0 and len(cycled) > 0:
                not_cycled = ~np.isin(vals + 1, cycled)
                filtered = targets[not_cycled]
                if len(filtered) == 0 or not np.any(filtered >= tb_new_h):
                    gs = 1
    if gs == 0 and can_split_lr:
        sub = mc_2d[offset_h:offset_h + height, offset_w:lr_new_w]
        nonzero = sub != 0
        if not np.any(nonzero):
            gs = 2
        else:
            all_vals = sub - 1
            actual_rows = _row_of[all_vals]
            actual_cols = _col_of[all_vals]
            solved = nonzero & (actual_rows == lr_exp_rows_bc) & (actual_cols == lr_exp_cols_bc)
            n_solved = np.sum(solved)
            total = height * (lr_new_w - offset_w)
            vals = sub[nonzero] - 1
            targets = _col_of[vals]
            if not np.any(targets >= lr_new_w) and total / 3 > n_solved:
                gs = 2
            elif gs == 0 and len(cycled) > 0:
                not_cycled = ~np.isin(vals + 1, cycled)
                filtered = targets[not_cycled]
                if len(filtered) == 0 or not np.any(filtered >= lr_new_w):
                    gs = 2
    return gs


def _check_solved(mc_2d, enable_gs, width, height, offset_w, offset_h,
                  tb_new_h, lr_new_w, can_split_tb, can_split_lr,
                  tb_gs_full_rows, tb_gs_full_cols,
                  lr_gs_full_rows, lr_gs_full_cols,
                  _row_of, _col_of, cycled_tiles=None):
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
        correct = (actual_rows == exp_rows_sub) & (actual_cols == exp_cols_sub)
        if np.all(correct):
            return True
        if cycled_tiles is not None and len(cycled_tiles) > 0:
            wrong_vals = sub[nonzero][~correct]
            if np.all(np.isin(wrong_vals, cycled_tiles)):
                return True
        return False
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
        correct = (actual_rows == exp_rows_sub) & (actual_cols == exp_cols_sub)
        if np.all(correct):
            return True
        if cycled_tiles is not None and len(cycled_tiles) > 0:
            wrong_vals = sub[nonzero][~correct]
            if np.all(np.isin(wrong_vals, cycled_tiles)):
                return True
        return False
    return True


def _detect_region_cycled(mc_2d, enable_gs, width, height, offset_w, offset_h,
                          width_initial, _row_of, _col_of, max_cycled=4):
    if enable_gs == 1:
        new_h = math.ceil(height / 2) + offset_h
        sub = mc_2d[offset_h:new_h, offset_w:offset_w + width]
    elif enable_gs == 2:
        new_w = math.ceil(width / 2) + offset_w
        sub = mc_2d[offset_h:offset_h + height, offset_w:new_w]
    else:
        return np.array([], dtype=np.int32)

    nonzero = sub != 0
    if not np.any(nonzero):
        return np.array([], dtype=np.int32)

    nonzero_vals = sub[nonzero]
    vals_idx = nonzero_vals - 1
    actual_rows = _row_of[vals_idx]
    actual_cols = _col_of[vals_idx]

    nz_flat_idx = np.flatnonzero(nonzero)
    if enable_gs == 1:
        sub_w = width
        exp_rows = np.arange(offset_h, new_h)[nz_flat_idx // sub_w]
        exp_cols = np.arange(offset_w, offset_w + width)[nz_flat_idx % sub_w]
    else:
        sub_w = new_w - offset_w
        sub_h = height
        exp_rows = np.arange(offset_h, offset_h + height)[nz_flat_idx // sub_w]
        exp_cols = np.arange(offset_w, new_w)[nz_flat_idx % sub_w]

    correct = (actual_rows == exp_rows) & (actual_cols == exp_cols)
    wrong_mask = ~correct
    if not np.any(wrong_mask):
        return np.array([], dtype=np.int32)

    wrong_vals = nonzero_vals[wrong_mask]
    region_area = sub.shape[0] * sub.shape[1]
    effective_max = min(max_cycled, max(1, region_area // 6))
    if len(wrong_vals) > effective_max:
        return np.array([], dtype=np.int32)

    return np.unique(wrong_vals)


def _detect_global_cycled(matrix, solution, width_initial, height_initial, _row_of, _col_of):
    sol_len = len(solution)
    if sol_len < 50:
        return np.array([], dtype=np.int32)
    mc_flat = np.array(matrix, dtype=np.int32).flatten()
    zp = find_zero(matrix, width_initial, height_initial)
    zp_idx = zp[0] * width_initial + zp[1]
    early = int(0.97 * sol_len)
    late = int(0.99 * sol_len)
    safe_w = max(1, width_initial // 2)
    safe_h = max(1, height_initial // 2)
    safe_r = max(0, height_initial - safe_h)
    safe_c = max(0, width_initial - safe_w)
    core = np.ones((height_initial, width_initial), dtype=bool)
    core[safe_r:, safe_c:] = False
    min_unsolved = 10**9
    best = np.array([], dtype=np.int32)
    for mi in range(late):
        move = solution[mi]
        move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
        dr, dc = _MOVE_DIRS[move]
        zp_idx += dr * width_initial + dc
        if mi < early:
            continue
        mc_2d = mc_flat.reshape(height_initial, width_initial)
        all_vals = mc_2d - 1
        ar = _row_of[all_vals.ravel()].reshape(height_initial, width_initial)
        ac = _col_of[all_vals.ravel()].reshape(height_initial, width_initial)
        exp_r = np.arange(height_initial)[:, None]
        exp_c = np.arange(width_initial)[None, :]
        solved = (ar == exp_r) & (ac == exp_c)
        wrong = (mc_2d != 0) & ~solved & core
        count = np.count_nonzero(wrong)
        if count < min_unsolved:
            min_unsolved = count
            best = np.unique(mc_2d[wrong])
    return best


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
        "cycledTiles": node.get("cycledTiles", []),
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
        "cycledTiles": skeleton.get("cycledTiles", []),
    }
    return shifted



def analyse_grids(matrix, solution, width_initial, height_initial, width, height, offset_w, offset_h, moves_offset, shape_cache=None, progress_callback=None, progress_total=None, _timing=None, _lock=None, _row_of=None, _col_of=None, cancel_check=None, global_cycled=None):
    if cancel_check and cancel_check():
        raise CancelError()
    if _timing is None:
        _timing = {"main_loop": 0.0, "scan_fwd": 0.0, "scan_cycles": 0.0, "get_parts": 0.0, "n_calls": 0}
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
    _gs_args = (width, height, offset_w, offset_h, can_split_tb, can_split_lr, tb_new_h, lr_new_w, tb_exp_rows_bc, tb_exp_cols_bc, lr_exp_rows_bc, lr_exp_cols_bc, _row_of, _col_of, height_initial, width_initial, global_cycled if global_cycled is not None else np.array([], dtype=np.int32))

    _t_loop = time.time()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1: Find grids_started (first position where gs != 0)
    # ═══════════════════════════════════════════════════════════════
    gs_type = 0
    grids_started = -1

    # Only send progress from top-level call (moves_offset == 0) to keep monotonic
    _send_progress = progress_callback is not None and moves_offset == 0

    if sol_len <= 16:
        # Linear scan for tiny solutions
        for mi in range(sol_len):
            move = solution[mi]
            move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
            dr, dc = _MOVE_DIRS[move]
            zp_idx += dr * width_initial + dc
            gs = _check_gs(mc_flat.reshape(height_initial, width_initial), *_gs_args)
            if _send_progress:
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
            if _send_progress:
                with _lock:
                    progress_callback(moves_offset + mi, total_for_progress)
            if gs:
                gs_type = gs
                grids_started = mi
                break

        if grids_started == -1:
            # Gallop with delta checkpoints (positions only, no board copies)
            cur_pos = n_linear - 1
            lo_state = mc_flat.copy()
            lo_zp = zp_idx
            step = 4
            checkpoints = []  # (pos, step_len) — deltas from previous state

            while cur_pos + step < sol_len:
                target = cur_pos + step
                snapshot = lo_state
                snapshot_zp = lo_zp

                for i in range(cur_pos + 1, target + 1):
                    move = solution[i]
                    move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                    dr, dc = _MOVE_DIRS[move]
                    zp_idx += dr * width_initial + dc

                checkpoints.append((cur_pos, step))
                cur_pos = target
                gs = _check_gs(mc_flat.reshape(height_initial, width_initial), *_gs_args)

                if _send_progress:
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
            # Fallback: replay forward scanning gaps between checkpoints + tail
            mc_flat[:] = init_flat
            zp_idx = init_zp

            # Advance from position 0 to n_linear-1 (state where gallop starts)
            for mi in range(n_linear):
                move = solution[mi]
                move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                dr, dc = _MOVE_DIRS[move]
                zp_idx += dr * width_initial + dc
            last_pos = n_linear - 1

            for snap_pos, step_len in checkpoints:
                for mi in range(last_pos + 1, snap_pos + 1):
                    move = solution[mi]
                    move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                    dr, dc = _MOVE_DIRS[move]
                    zp_idx += dr * width_initial + dc
                last_pos = snap_pos

                end_scan = min(snap_pos + step_len, sol_len)
                for mi in range(snap_pos + 1, end_scan):
                    move = solution[mi]
                    move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                    dr, dc = _MOVE_DIRS[move]
                    zp_idx += dr * width_initial + dc
                    last_pos = mi
                    gs = _check_gs(mc_flat.reshape(height_initial, width_initial), *_gs_args)
                    if gs:
                        grids_started = mi
                        gs_type = gs
                        break
                if grids_started != -1:
                    break

            if grids_started == -1:
                for mi in range(last_pos + 1, sol_len):
                    move = solution[mi]
                    move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                    dr, dc = _MOVE_DIRS[move]
                    zp_idx += dr * width_initial + dc
                    gs = _check_gs(mc_flat.reshape(height_initial, width_initial), *_gs_args)
                    if gs:
                        grids_started = mi
                        gs_type = gs
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
            if _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args, cycled_tiles=global_cycled):
                grids_stopped = grids_started + 1 + si
                break
    else:
        n_linear_s = min(4, scan_len)
        for si in range(n_linear_s):
            move = scan_solution[si]
            move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
            dr, dc = _MOVE_DIRS[move]
            zp_idx += dr * width_initial + dc
            if _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args, cycled_tiles=global_cycled):
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
                solved = _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args, cycled_tiles=global_cycled)

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
                        if _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args, cycled_tiles=global_cycled):
                            hi = mid
                        else:
                            lo = mid

                    grids_stopped = grids_started + 1 + snapshot_pos + hi
                    break

                lo_state = mc_flat.copy()
                lo_zp = zp_idx
                step = min(step * 2, 256)

        if grids_stopped == -1:
            # New fallback – forward binary search on the tail
            # lo_state is state at cur_pos (the last gallop target)
            mc_flat[:] = lo_state
            zp_idx = lo_zp
            lo = cur_pos + 1
            hi = scan_len - 1
            while lo < hi:
                mid = (lo + hi) // 2
                mc_flat[:] = lo_state
                zp_idx = lo_zp
                for i in range(cur_pos + 1, mid + 1):
                    move = scan_solution[i]
                    move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                    dr, dc = _MOVE_DIRS[move]
                    zp_idx += dr * width_initial + dc
                if _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args, cycled_tiles=global_cycled):
                    hi = mid
                else:
                    lo = mid + 1
            if lo <= hi:
                mc_flat[:] = lo_state
                zp_idx = lo_zp
                for i in range(cur_pos + 1, lo + 1):
                    move = scan_solution[i]
                    move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                    dr, dc = _MOVE_DIRS[move]
                    zp_idx += dr * width_initial + dc
                if _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args, cycled_tiles=global_cycled):
                    grids_stopped = grids_started + 1 + lo
            # As a safety net (should never be needed for monotonic solved)
            if grids_stopped == -1:
                mc_flat[:] = lo_state
                zp_idx = lo_zp
                for si in range(cur_pos + 1, scan_len):
                    move = scan_solution[si]
                    move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                    dr, dc = _MOVE_DIRS[move]
                    zp_idx += dr * width_initial + dc
                    if _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args, cycled_tiles=global_cycled):
                        grids_stopped = grids_started + 1 + si
                        break

    with _lock:
        _timing["scan_fwd"] += time.time() - _t_scan

    if grids_stopped == -1:
        return {"enableGridsStatus": -1, "width": width, "height": height, "offsetW": offset_w, "offsetH": offset_h}

    girds_unsolved_last = grids_stopped - 1
    grids_stopped_final = girds_unsolved_last + 1

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3: Per-region cycle detection
    # ═══════════════════════════════════════════════════════════════
    _t_cycles = time.time()
    per_region_cycled = np.array([], dtype=np.int32)
    combined = np.array([], dtype=np.int32)
    if global_cycled is not None and len(global_cycled) > 0:
        combined = global_cycled.copy()
    if grids_stopped - grids_started > 2:
        mc_flat[:] = transition_flat
        zp_idx = transition_zp
        for mi in range(grids_started + 1, grids_stopped):
            move = solution[mi]
            move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
            dr, dc = _MOVE_DIRS[move]
            zp_idx += dr * width_initial + dc
        mc_2d = mc_flat.reshape(height_initial, width_initial)
        cycled_candidates = _detect_region_cycled(
            mc_2d, enable_gs, width, height, offset_w, offset_h,
            width_initial, _row_of, _col_of
        )
        if len(cycled_candidates) > 0:
            lookahead = min(15, len(solution) - grids_stopped - 1)
            if lookahead > 0:
                target = grids_stopped + lookahead - 1
                for mi in range(grids_stopped, target + 1):
                    move = solution[mi]
                    move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                    dr, dc = _MOVE_DIRS[move]
                    zp_idx += dr * width_initial + dc
                mc_2d = mc_flat.reshape(height_initial, width_initial)
                still_wrong = _detect_region_cycled(
                    mc_2d, enable_gs, width, height, offset_w, offset_h,
                    width_initial, _row_of, _col_of
                )
                per_region_cycled = cycled_candidates[np.isin(cycled_candidates, still_wrong)]
            else:
                per_region_cycled = cycled_candidates

            seg_len = grids_stopped - grids_started
            if seg_len > 100 and len(cycled_candidates) > 0:
                backward = min(500, seg_len // 4)
                check_at = grids_stopped - backward
                if check_at > grids_started + 1:
                    mid_at = grids_started + seg_len // 2
                    use_midpoint = seg_len > 200 and mid_at > grids_started + 1 and mid_at < check_at - 1

                    mc_flat[:] = transition_flat
                    zp_idx = transition_zp
                    if use_midpoint:
                        for mi in range(grids_started + 1, mid_at + 1):
                            move = solution[mi]
                            move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                            dr, dc = _MOVE_DIRS[move]
                            zp_idx += dr * width_initial + dc
                        mid_cy = _detect_region_cycled(
                            mc_flat.reshape(height_initial, width_initial),
                            enable_gs, width, height, offset_w, offset_h,
                            width_initial, _row_of, _col_of
                        )
                        for mi in range(mid_at + 1, check_at + 1):
                            move = solution[mi]
                            move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                            dr, dc = _MOVE_DIRS[move]
                            zp_idx += dr * width_initial + dc
                    else:
                        for mi in range(grids_started + 1, check_at + 1):
                            move = solution[mi]
                            move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                            dr, dc = _MOVE_DIRS[move]
                            zp_idx += dr * width_initial + dc

                    earlier_cy = _detect_region_cycled(
                        mc_flat.reshape(height_initial, width_initial),
                        enable_gs, width, height, offset_w, offset_h,
                        width_initial, _row_of, _col_of
                    )

                    if len(earlier_cy) > 0:
                        sets = []
                        if use_midpoint and len(mid_cy) > 0:
                            sets.append(mid_cy)
                        sets.append(earlier_cy)
                        sets.append(cycled_candidates)

                        persistent = np.array([], dtype=np.int32)
                        if len(sets) >= 3:
                            pairs = []
                            for i in range(len(sets)):
                                for j in range(i + 1, len(sets)):
                                    inter = np.intersect1d(sets[i], sets[j])
                                    if len(inter) > 0:
                                        pairs.append(inter)
                            if pairs:
                                persistent = np.unique(np.concatenate(pairs))
                        else:
                            persistent = np.intersect1d(sets[0], sets[1])

                        if len(persistent) > 0:
                            log.info(f"  persistent cycles@{moves_offset+grids_started}: {persistent.tolist()}")
                            per_region_cycled = np.union1d(per_region_cycled, persistent)
        else:
            per_region_cycled = cycled_candidates

        if global_cycled is not None and len(global_cycled) > 0:
            combined = np.union1d(per_region_cycled, global_cycled)
        else:
            combined = per_region_cycled
        if len(combined) > 0:
            lo = grids_started + 1
            hi = grids_stopped - 1
            while lo < hi:
                mid = (lo + hi) // 2
                mc_flat[:] = transition_flat
                zp_idx = transition_zp
                for mi in range(grids_started + 1, mid + 1):
                    move = solution[mi]
                    move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                    dr, dc = _MOVE_DIRS[move]
                    zp_idx += dr * width_initial + dc
                if _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args, cycled_tiles=combined):
                    hi = mid
                else:
                    lo = mid + 1
            mc_flat[:] = transition_flat
            zp_idx = transition_zp
            for mi in range(grids_started + 1, lo + 1):
                move = solution[mi]
                move_matrix_inplace(mc_flat, move, zp_idx, width_initial)
                dr, dc = _MOVE_DIRS[move]
                zp_idx += dr * width_initial + dc
            if _check_solved(mc_flat.reshape(height_initial, width_initial), *_solv_args, cycled_tiles=combined):
                if lo < grids_stopped:
                    log.info(f"    cycles@{moves_offset+grids_started}: {per_region_cycled.tolist()}, grids_stopped {grids_stopped}->{lo}")
                    grids_stopped = lo
                    grids_stopped_final = lo
                    girds_unsolved_last = lo - 1

    with _lock:
        _timing["scan_cycles"] += time.time() - _t_cycles

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
                r = analyse_grids(mat, sol, width_initial, height_initial, w, h, ow, oh, off, shape_cache, progress_callback, progress_total, _timing, _lock, _row_of, _col_of, cancel_check=cancel_check, global_cycled=combined if len(combined) > 0 else None)
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
        "cycledTiles": combined.tolist() if len(combined) > 0 else [],
    }



def _find_truncation_point(matrix, solution, full_w, full_h, _row_of, _col_of):
    if full_h < 5 or full_w < 5:
        return -1, np.array([], dtype=np.int32)
    corner_h = min(5, full_h)
    corner_w = min(5, full_w)
    mc_flat = np.array(matrix, dtype=np.int32).flatten()
    zp = find_zero(matrix, full_w, full_h)
    zp_idx = zp[0] * full_w + zp[1]
    for mi, move in enumerate(solution):
        move_matrix_inplace(mc_flat, move, zp_idx, full_w)
        dr, dc = _MOVE_DIRS[move]
        zp_idx += dr * full_w + dc
        mc_2d = mc_flat.reshape(full_h, full_w)
        if _corner_solved(mc_2d, full_h, full_w, _row_of, _col_of):
            tiles = _wrong_tiles(mc_2d, full_h, full_w, _row_of, _col_of)
            # Remove tiles that belong to the corner (they're legit unsolved, not cycle victims)
            if len(tiles) > 0:
                vals = tiles - 1
                tr = vals // full_w
                tc = vals % full_w
                corner_mask = ~((tr >= full_h - corner_h) & (tc >= full_w - corner_w))
                tiles = tiles[corner_mask]
            return mi, tiles
    return -1, np.array([], dtype=np.int32)


def analyse_grids_initial(matrix, solution, progress_callback=None, cancel_check=None):
    _t0 = time.time()
    h = len(matrix)
    w = len(matrix[0])
    total_cells = w * h
    _row_of = np.arange(total_cells) // w
    _col_of = np.arange(total_cells) % w
    _timing = {"main_loop": 0.0, "scan_fwd": 0.0, "scan_cycles": 0.0, "get_parts": 0.0, "n_calls": 0}
    _lock = threading.Lock()
    trunc_at, global_cycled = _find_truncation_point(matrix, solution, w, h, _row_of, _col_of)
    if trunc_at > 0:
        log.info(f"  truncated solution at move {trunc_at}, earliest corner solved; global cycled {global_cycled.tolist() if len(global_cycled) > 0 else 'none'}")
        solution = solution[:trunc_at]
    else:
        global_cycled = _detect_global_cycled(matrix, solution, w, h, _row_of, _col_of)
    if len(global_cycled) > 0:
        log.info(f"  global cycled tiles: {global_cycled.tolist()}")
    result = analyse_grids(matrix, solution, w, h, w, h, 0, 0, 0, shape_cache={}, progress_callback=progress_callback, progress_total=len(solution), _timing=_timing, _lock=_lock, _row_of=_row_of, _col_of=_col_of, cancel_check=cancel_check, global_cycled=global_cycled)
    elapsed = time.time() - _t0
    log.info(f"  grids analysis: {_timing['n_calls']} calls, {elapsed:.3f}s wall clock")
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
        "activeZone": get_active_zone_by_level(cl),
        "cycledTiles": cl.get("cycledTiles", []),
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
            existing = seen[sig]
            existing_ct = existing.get("cycledTiles", [])
            current_ct = levels[k].get("cycledTiles", [])
            if existing_ct or current_ct:
                merged = list(set(existing_ct + current_ct))
                existing["cycledTiles"] = merged
            levels[k] = existing
        else:
            seen[sig] = levels[k]

    return levels


def collect_all_cycled_tiles(grids_data):
    tiles = set()
    def walk(node):
        if node:
            ct = node.get("cycledTiles", [])
            tiles.update(ct)
            walk(node.get("nextLayerFirst"))
            walk(node.get("nextLayerSecond"))
    walk(grids_data)
    return sorted(tiles)


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

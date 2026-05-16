import sys; sys.path.insert(0, '.')
import numpy as np
from splits import splits
from replay_generator import scramble_to_puzzle, expand_solution
from sliding_puzzles import decompress_string_to_array, read_solve_data, _MOVE_DIRS, find_zero, move_matrix_inplace
from urllib.parse import urlparse, parse_qs
import math
from grids_analysis import _check_gs

with open('test_bugs/40x40 cycle.txt', 'r') as f:
    url = f.read().strip()
data = splits(url)
scramble = data[4]
qp = parse_qs(urlparse(url).query)
rp = qp.get('r', qp.get('p', [None]))[0]
rd = decompress_string_to_array(rp)
solution = read_solve_data(rd[1])['solutions'] if len(rd) >= 10 else rd[0]
matrix = scramble_to_puzzle(scramble)
expanded = expand_solution(solution)
w, h = len(matrix[0]), len(matrix)
_row_of = np.arange(w*h)//w; _col_of = np.arange(w*h)%w

mc = np.array(matrix, dtype=np.int32).flatten()
zp = find_zero(matrix, w, h); zi = zp[0]*w+zp[1]

lr_new_w = math.ceil(10/2)+30
tb_new_h = math.ceil(5/2)+25
lr_sub_w = lr_new_w - 30
lr_exp_rows_bc = np.arange(25, 30)[:, None]
lr_exp_cols_bc = np.arange(30, lr_new_w)[None, :]
gs_args = (10, 5, 30, 25, 5>5, 10>5, tb_new_h, lr_new_w,
    np.empty((0,0)), np.empty((0,0)),
    lr_exp_rows_bc, lr_exp_cols_bc,
    _row_of, _col_of)

for i in range(108143):
    m = expanded[i]; move_matrix_inplace(mc, m, zi, w)
    dr, dc = _MOVE_DIRS[m]; zi += dr*w+dc

for i in range(108143, 108625):
    m = expanded[i]; move_matrix_inplace(mc, m, zi, w)
    dr, dc = _MOVE_DIRS[m]; zi += dr*w+dc
    internal_idx = i - 108143
    gs = _check_gs(mc.reshape(h,w), *gs_args)
    if gs:
        sub = mc.reshape(h,w)[25:30, 30:35]
        nz = sub != 0; nzv = sub[nz]; vi = nzv-1
        ac = _col_of[vi]
        has_cross = np.any(ac >= 35)
        cross_tiles = np.unique(nzv[ac >= 35]).tolist() if has_cross else []
        print(f'i={i} internal={internal_idx} gs={gs} cross={cross_tiles}')

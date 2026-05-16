import sys; sys.path.insert(0, ".")
import numpy as np
from splits import splits
from grids_analysis import analyse_grids_initial
from replay_generator import scramble_to_puzzle, expand_solution
from sliding_puzzles import decompress_string_to_array, read_solve_data
from urllib.parse import urlparse, parse_qs

with open("test_bugs/40x40 cycle.txt", "r") as f:
    url = f.read().strip()
data = splits(url)
scramble = data[4]
qp = parse_qs(urlparse(url).query)
rp = qp.get("r", qp.get("p", [None]))[0]
rd = decompress_string_to_array(rp)
if len(rd) < 10:
    solution = rd[0]
else:
    solution = read_solve_data(rd[1])["solutions"]
matrix = scramble_to_puzzle(scramble)
expanded = expand_solution(solution)
gd = analyse_grids_initial(matrix, expanded)


def walk(node, d=0, path=""):
    en = node.get("enableGridsStatus")
    tag = {1: "TB", 2: "LR", -1: "Fringe"}.get(en, "?")
    cy = node.get("cycledTiles")
    cy_str = (" cy=" + str(cy)) if cy else ""
    w = node.get("width")
    h = node.get("height")
    ow = node.get("offsetW")
    oh = node.get("offsetH")
    gs = node.get("gridsStarted", "?")
    ge = node.get("gridsStopped", "?")
    indent = "  " * d
    print(f"{indent}{path} {tag} {w}x{h}@{ow},{oh} gs={gs}-{ge}{cy_str}")
    n1 = node.get("nextLayerFirst")
    n2 = node.get("nextLayerSecond")
    if n1:
        walk(n1, d + 1, path + "L")
    if n2:
        walk(n2, d + 1, path + "R")


walk(gd)

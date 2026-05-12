from dataclasses import dataclass

PADDING = 20
HEADER_H = 56
STATS_PANEL_WIDTH = 320
INFO_H = 40
TIMER_HEIGHT = 30

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

TILE_BORDER_WIDTH = 1
TILE_BORDER_RADIUS_RATIO = 0.4
BASE_SIZE = 15


def compute_canvas_dimensions(puzzle_w: int, puzzle_h: int, tile_size: int,
                              grid_only: bool = False) -> tuple[int, int]:
    if puzzle_w < 1 or puzzle_h < 1:
        raise ValueError(f"puzzle dimensions must be >= 1, got {puzzle_w}x{puzzle_h}")
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")

    puzzle_px_w = puzzle_w * tile_size
    puzzle_px_h = puzzle_h * tile_size

    if grid_only:
        cw = (puzzle_px_w + PADDING * 2 + 1) // 2 * 2
        ch = (puzzle_px_h + PADDING * 2 + 1) // 2 * 2
    else:
        panel_w_est = STATS_PANEL_WIDTH
        cw = (puzzle_px_w + panel_w_est + PADDING * 3 + 1) // 2 * 2
        base_h = HEADER_H + puzzle_px_h + PADDING * 3
        ch = (base_h + 1) // 2 * 2

    return cw, ch


@dataclass(frozen=True)
class RenderOptions:
    grid_only: bool = False
    no_border: bool = False
    no_secondary_border: bool = False
    no_numbers: bool = False

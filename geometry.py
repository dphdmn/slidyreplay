import os
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

# ─── Font Loading ──────────────────────────────────────────────────

_font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_FAMILY = os.path.join(_font_dir, "Roboto-Regular.ttf")
FONT_FAMILY_BOLD = os.path.join(_font_dir, "Roboto-Bold.ttf")
FONT_FAMILY_MONO = os.path.join(_font_dir, "JetBrainsMono-Regular.ttf")
FONT_FAMILY_MONO_BOLD = os.path.join(_font_dir, "JetBrainsMono-Bold.ttf")

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

def render_number_texture(num: int, tile_size: int, font_size: int) -> Image.Image:
    """Render a single number tile. Returns RGBA Image."""
    key = (num, tile_size, font_size)
    cached = _number_texture_cache.get(key)
    if cached is not None:
        return cached
    im = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    if num != 0:
        text = str(num)
        tf = get_font(font_size)
        tb = draw.textbbox((0, 0), text, font=tf)
        tx = tile_size // 2 - (tb[0] + tb[2]) // 2
        ty = tile_size // 2 - (tb[1] + tb[3]) // 2
        draw.text((tx, ty), text, fill=(0, 0, 0, 255), font=tf)
    _number_texture_cache[key] = im
    return im


@lru_cache(maxsize=128)
def render_timer_text(timer_text: str) -> Image.Image:
    font = get_font(36, bold=True, mono=True)
    b = font.getbbox(timer_text)
    w = b[2] - b[0]
    h = b[3] - b[1]
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.text((-b[0], -b[1]), timer_text, fill=(*CYAN, 255), font=font)
    return im


def render_dynamic_text(text: str, font, color=WHITE):
    b = font.getbbox(text)
    w = b[2] - b[0]
    h = b[3] - b[1]
    if w <= 0 or h <= 0:
        return None
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.text((-b[0], -b[1]), text, fill=(*color, 255), font=font)
    return im


def compute_tile_size(raw_tile: int, quality: float) -> int:
    return max(raw_tile, int(raw_tile * quality))


def compute_font_size(width: int, height: int, tile_size: int) -> int:
    max_num = width * height - 1
    num_digits = len(str(max_num))
    divisor = 25 if num_digits <= 3 else 30
    return max(8, tile_size * 11 // divisor)


def compute_grid_position(grid_only: bool) -> tuple[int, int]:
    return PADDING, PADDING if grid_only else PADDING + HEADER_H + PADDING


def compute_panel_rect(grid_x: int, puzzle_w: int, canvas_w: int, grid_y: int, canvas_h: int) -> tuple[int, int, int, int]:
    panel_x = grid_x + puzzle_w + PADDING
    panel_y = grid_y
    panel_w = canvas_w - panel_x - PADDING
    panel_h = canvas_h - panel_y - PADDING
    return panel_x, panel_y, panel_w, panel_h


def compute_secondary_bar_rect(tile_size: int, tile_x: int = 0, tile_y: int = 0) -> tuple[int, int, int, int]:
    bar_h = max(2, int(tile_size * 0.1))
    bar_off = max(2, int(tile_size * 0.06))
    bar_inset = max(2, int(tile_size * 0.1))
    y0 = tile_y + tile_size - bar_h - bar_off
    y1 = tile_y + tile_size - bar_off
    x0 = tile_x + bar_inset
    x1 = tile_x + tile_size - bar_inset
    return x0, y0, x1, y1


def round_canvas_height(h: int) -> int:
    return (h + 1) // 2 * 2


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


@dataclass
class TileSpriteCache:
    tile_size: int
    base_sprites: dict
    number_texts: dict
    bar_sprites: dict
    opts: RenderOptions = RenderOptions()


_RED = (200, 103, 103)
_BLUE = (141, 179, 255)


def _solid_base(color, tile_size, opts):
    im = Image.new("RGBA", (tile_size, tile_size))
    draw = ImageDraw.Draw(im)
    draw.rectangle([(0, 0), (tile_size - 1, tile_size - 1)], fill=color)
    if tile_size > 1 and not opts.no_border:
        draw.rectangle(
            [(0, 0), (tile_size - 1, tile_size - 1)],
            outline=TILE_BORDER_COLOR, width=TILE_BORDER_WIDTH
        )
    return im


def _bar_sprite(color, tile_size, opts):
    bar_h = max(2, int(tile_size * 0.1))
    bar_off = max(2, int(tile_size * 0.06))
    bar_inset = max(2, int(tile_size * 0.1))
    y0 = tile_size - bar_h - bar_off
    y1 = tile_size - bar_off
    x0 = bar_inset
    x1 = tile_size - bar_inset

    im = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    bar_bbox = (x0, y0, x1, y1)
    draw.rectangle(bar_bbox, fill=color)
    if tile_size > 1 and not opts.no_secondary_border:
        draw.rectangle(bar_bbox, outline=TILE_BORDER_COLOR, width=1)
    return im


def select_base(main_bg, num, cache: TileSpriteCache):
    if main_bg is None:
        return cache.base_sprites[TILE_BG]
    if isinstance(main_bg, np.ndarray):
        main_bg = tuple(int(x) for x in main_bg.ravel())
    else:
        main_bg = tuple(int(x) for x in main_bg)
    return cache.base_sprites[main_bg]


def select_bar(sec_bg, cache: TileSpriteCache):
    if sec_bg is None:
        return None
    if isinstance(sec_bg, np.ndarray):
        sec_bg = tuple(int(x) for x in sec_bg.ravel())
    else:
        sec_bg = tuple(int(x) for x in sec_bg)
    return cache.bar_sprites.get(sec_bg)

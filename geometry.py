import os
import re
import sys
from typing import Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PADDING = 2


def parse_hex_color(hex_str: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if hex_str is None:
        return None
    hex_str = hex_str.strip().lstrip("#")
    if not re.match(r"^[0-9a-fA-F]{6}$", hex_str):
        return None
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (r, g, b)
HEADER_H = 32
STATS_PANEL_WIDTH = 340
INFO_H = 40
TIMER_HEIGHT = 30

BG_COLOR = (18, 18, 18)
TILE_BG = (69, 69, 69)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_BORDER_COLOR = (0, 0, 0)
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

TILE_BORDER_WIDTH = 1  # base; scaled by tile_size in render
TILE_BORDER_RADIUS_RATIO = 0.4
MIN_TILE = 2
MIN_NUMBER_TILE_SIZE = 12
MIN_BORDER_TILE_SIZE = 12
MIN_SECONDARY_BORDER_TILE_SIZE = 35

# ─── Font Loading ──────────────────────────────────────────────────

_font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT_FAMILY = os.path.join(_font_dir, "Roboto-Regular.ttf")
FONT_FAMILY_BOLD = os.path.join(_font_dir, "Roboto-Bold.ttf")
FONT_FAMILY_MONO = os.path.join(_font_dir, "JetBrainsMono-Regular.ttf")
FONT_FAMILY_MONO_BOLD = os.path.join(_font_dir, "JetBrainsMono-Bold.ttf")

_font_cache = {}

# ─── System Font Resolution ──────────────────────────────────────

_system_fonts = None

def _build_system_font_map():
    """Scan OS for installed font files.
    Returns {family_lower: {"name": original_name, "regular": path_or_None, "bold": path_or_None}}.
    """
    font_map = {}
    if sys.platform == "win32":
        font_dir = os.path.join(
            os.environ.get("SystemRoot", "C:\\Windows"), "Fonts"
        )
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts")
            for i in range(winreg.QueryInfoKey(key)[1]):
                try:
                    name, value, _ = winreg.EnumValue(key, i)
                    display = re.sub(r'\s*\(.*?\)\s*$', '', name).strip()
                    if not value or not display:
                        continue
                    path = value if os.path.isabs(value) else os.path.join(font_dir, value)
                    if not os.path.exists(path):
                        continue
                    # Determine if this entry is a bold variant
                    display_lower = display.lower()
                    is_bold = 'bold' in display_lower and 'thin' not in display_lower
                    # Derive base family by stripping known style suffixes
                    base = display
                    for suffix in [" Bold", " Italic", " Light", " Black",
                                   " Semibold", " Semilight", " ExtraBold", " Thin"]:
                        if base.endswith(suffix) and len(base) > len(suffix):
                            # Also try removing combined suffixes
                            test = base[:-len(suffix)]
                            if test.strip():
                                base = test.strip()
                            break
                    # Also handle "Bold Italic" two-word style
                    if base != display:
                        # Already stripped one suffix
                        pass
                    elif " bold " in display_lower or display_lower.endswith("bold italic"):
                        base = re.sub(r'\s*bold\s*italic\s*$', '', display, flags=re.IGNORECASE).strip()
                        if not base or base == display:
                            base = re.sub(r'\s*bold\s*', '', display, flags=re.IGNORECASE).strip()
                    for fam_key in (display, base):
                        if not fam_key:
                            continue
                        fl = fam_key.lower()
                        if fl not in font_map:
                            font_map[fl] = {"name": fam_key, "regular": None, "bold": None}
                        if is_bold:
                            if font_map[fl]["bold"] is None:
                                font_map[fl]["bold"] = path
                            # Exact family entries like "Arial Bold" should also resolve as regular
                            # (loading the bold file is the intended bold look for that family)
                            if font_map[fl]["regular"] is None and fam_key == display:
                                font_map[fl]["regular"] = path
                        else:
                            if font_map[fl]["regular"] is None:
                                font_map[fl]["regular"] = path
                except Exception:
                    pass
        except Exception:
            pass
    elif sys.platform in ("linux", "darwin"):
        try:
            import subprocess
            result = subprocess.run(
                ["fc-list", "--format=%{family}\n%{file}\n%{style}\n"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split("\n")
            i = 0
            while i + 2 < len(lines):
                families = [f.strip() for f in lines[i].split(",")]
                path = lines[i + 1].strip()
                style = lines[i + 2].strip().lower()
                i += 3
                if not path or not families[0]:
                    continue
                for fam in families:
                    if not fam:
                        continue
                    fl = fam.lower()
                    if fl not in font_map:
                        font_map[fl] = {"name": fam, "regular": None, "bold": None}
                    is_bold = "bold" in style and "thin" not in style
                    if is_bold:
                        if font_map[fl]["bold"] is None:
                            font_map[fl]["bold"] = path
                    else:
                        if font_map[fl]["regular"] is None:
                            font_map[fl]["regular"] = path
        except Exception:
            pass
        # Fallback: scan common font directories
        if not font_map:
            _scan_font_dirs(font_map)
    # Add bundled fonts
    bundled = [
        ("Roboto", FONT_FAMILY, FONT_FAMILY_BOLD),
        ("JetBrains Mono", FONT_FAMILY_MONO, FONT_FAMILY_MONO_BOLD),
    ]
    for name, regular_path, bold_path in bundled:
        fl = name.lower()
        if fl not in font_map:
            font_map[fl] = {"name": name, "regular": None, "bold": None}
        if font_map[fl]["regular"] is None and os.path.exists(regular_path):
            font_map[fl]["regular"] = regular_path
        if font_map[fl]["bold"] is None and os.path.exists(bold_path):
            font_map[fl]["bold"] = bold_path
    return font_map


def _scan_font_dirs(font_map):
    """Fallback: scan common font directories for .ttf/.otf files."""
    candidates = []
    if sys.platform == "linux":
        candidates = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.local/share/fonts"),
            os.path.expanduser("~/.fonts"),
        ]
    elif sys.platform == "darwin":
        candidates = [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ]
    if sys.platform in ("linux", "darwin"):
        import subprocess
        for d in candidates:
            if not os.path.isdir(d):
                continue
            for root, dirs, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf")):
                        path = os.path.join(root, f)
                        try:
                            r = subprocess.run(
                                ["fc-scan", "--format=%{family}\n%{style}", path],
                                capture_output=True, text=True, timeout=2
                            )
                            if r.returncode == 0:
                                lines = r.stdout.strip().split("\n")
                                if len(lines) >= 2:
                                    families = [x.strip() for x in lines[0].split(",")]
                                    style = lines[1].strip().lower()
                                    for fam in families:
                                        if not fam:
                                            continue
                                        fl = fam.lower()
                                        if fl not in font_map:
                                            font_map[fl] = {"name": fam, "regular": None, "bold": None}
                                        is_bold = "bold" in style and "thin" not in style
                                        if is_bold:
                                            if font_map[fl]["bold"] is None:
                                                font_map[fl]["bold"] = path
                                        else:
                                            if font_map[fl]["regular"] is None:
                                                font_map[fl]["regular"] = path
                        except Exception:
                            pass


def get_system_font_families():
    """Return sorted list of available font family names (original casing)."""
    global _system_fonts
    if _system_fonts is None:
        _system_fonts = _build_system_font_map()
    return sorted(v["name"] for v in _system_fonts.values() if v.get("name"))


def get_font_path(family: str, bold: bool = False) -> Optional[str]:
    """Resolve a font family name to a file path."""
    global _system_fonts
    if _system_fonts is None:
        _system_fonts = _build_system_font_map()
    fl = family.lower().strip()
    entry = _system_fonts.get(fl)
    if entry:
        if bold and entry.get("bold"):
            return entry["bold"]
        if entry.get("regular"):
            if not bold:
                return entry["regular"]
    # Bold not found — try alternative family names (e.g. "Arial Bold" list entry)
    if bold:
        for alt in [f"{family} Bold", f"{family}-Bold", f"{family}Bold"]:
            alt_fl = alt.lower()
            alt_entry = _system_fonts.get(alt_fl)
            if alt_entry:
                if alt_entry.get("regular"):
                    return alt_entry["regular"]
                if alt_entry.get("bold"):
                    return alt_entry["bold"]
    return None


def get_font(size: int, bold: bool = False, mono: bool = False,
             family: str = None) -> ImageFont.FreeTypeFont:
    key = (size, bold, mono, family)
    if key in _font_cache:
        return _font_cache[key]
    if family is not None and family.lower().strip() not in ("roboto", "jetbrains mono"):
        font_path = get_font_path(family, bold)
        if font_path:
            try:
                font = ImageFont.truetype(font_path, size)
                _font_cache[key] = font
                return font
            except Exception:
                pass
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

def render_number_texture(num: int, tile_size: int, font_size: int,
                          font_family: str = None, font_bold: bool = False) -> Image.Image:
    """Render a single number tile. Returns RGBA Image."""
    key = (num, tile_size, font_size, font_family, font_bold)
    cached = _number_texture_cache.get(key)
    if cached is not None:
        return cached
    im = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    if num != 0 and should_draw_numbers(tile_size, font_size):
        text = str(num)
        tf = get_font(font_size, bold=font_bold, family=font_family)
        tb = draw.textbbox((0, 0), text, font=tf)
        text_w = max(1, tb[2] - tb[0])
        text_h = max(1, tb[3] - tb[1])
        pad = 2
        text_im = Image.new("RGBA", (text_w + pad * 2, text_h + pad * 2), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_im)
        text_draw.text((pad - tb[0], pad - tb[1]), text, fill=(0, 0, 0, 255), font=tf)
        ink_bbox = text_im.getbbox()
        if ink_bbox is not None:
            text_im = text_im.crop(ink_bbox)
            tx = (tile_size - text_im.width + 1) // 2
            ty = (tile_size - text_im.height) // 2
            im.alpha_composite(text_im, (tx, ty))
    _number_texture_cache[key] = im
    return im


@lru_cache(maxsize=128)
def render_timer_text(timer_text: str, font_size: int = 36) -> Image.Image:
    font = get_font(font_size, bold=True, mono=True)
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


def compute_font_size(width: int, height: int, tile_size: int,
                      font_size_override: Optional[int] = None) -> int:
    if font_size_override is not None and font_size_override > 0:
        return font_size_override
    if tile_size < MIN_NUMBER_TILE_SIZE:
        return 0
    # Sub-linear base: font grows proportionally up to 60px tile,
    # then only 25% of excess contributes, keeping fonts reasonable on small puzzles.
    base = tile_size if tile_size <= 60 else 60 + (tile_size - 60) * 0.25
    # Target font size from formula — no dependency on digit count, so no jumps.
    font_size = max(4, round(base * 0.68))
    # Verify against a fixed 4-char sample (covers all puzzles up to 100x100).
    # Using "8888" everywhere eliminates digit-count transitions entirely.
    max_text_w = max(1, round(tile_size * 0.85))
    max_text_h = max(1, round(tile_size * 0.58))
    font = get_font(font_size)
    bbox = font.getbbox("8888")
    while font_size > 4 and (bbox[2] - bbox[0] > max_text_w or bbox[3] - bbox[1] > max_text_h):
        font_size -= 1
        font = get_font(font_size)
        bbox = font.getbbox("8888")
    return font_size


def should_draw_numbers(tile_size: int, font_size: int) -> bool:
    return tile_size >= MIN_NUMBER_TILE_SIZE and font_size > 0


def should_draw_tile_border(tile_size: int) -> bool:
    return tile_size >= MIN_BORDER_TILE_SIZE


def should_draw_secondary_border(tile_size: int) -> bool:
    return tile_size >= MIN_SECONDARY_BORDER_TILE_SIZE


def should_draw_secondary_border_rect(tile_size: int, rect: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = rect
    return should_draw_secondary_border(tile_size) and (x1 - x0) >= 3 and (y1 - y0) >= 3


def compute_grid_position(grid_only: bool, pad: int = None, header_h: int = None,
                          canvas_h: int = None, puzzle_h: int = None,
                          no_header: bool = False, align_top: bool = False) -> tuple[int, int]:
    if header_h is None:
        header_h = HEADER_H
    hide_header = grid_only or no_header
    base_y = 0 if hide_header else header_h
    if hide_header and canvas_h is not None and puzzle_h is not None and not align_top:
        avail_h = canvas_h
        extra_h = max(0, avail_h - puzzle_h)
        base_y += extra_h // 2
    return 0, base_y


def compute_panel_rect(grid_x: int, puzzle_w: int, canvas_w: int, grid_y: int, canvas_h: int,
                       pad: int = None, panel_y: int = None) -> tuple[int, int, int, int]:
    if pad is None:
        pad = PADDING
    if panel_y is None:
        panel_y = grid_y
    panel_x = grid_x + puzzle_w + pad
    panel_w = canvas_w - panel_x
    panel_h = canvas_h - panel_y
    return panel_x, panel_y, panel_w, panel_h


def compute_number_visual_bottom(tile_size: int, font_size: int) -> int:
    if not should_draw_numbers(tile_size, font_size):
        return round(tile_size * 0.55)
    font = get_font(font_size)
    bbox = font.getbbox("8888")
    text_mid_y = (bbox[1] + bbox[3]) // 2
    text_y = tile_size // 2 - text_mid_y
    return text_y + bbox[3]


def compute_secondary_bar_rect(tile_size: int, tile_x: int = 0, tile_y: int = 0,
                               font_size: int = None) -> tuple[int, int, int, int]:
    bar_h = round(tile_size * 0.10)
    if tile_size >= MIN_NUMBER_TILE_SIZE:
        bar_h = max(3, bar_h)
    else:
        bar_h = max(1, bar_h)
    bar_inset = max(1, round(tile_size * 0.12))
    number_bottom = compute_number_visual_bottom(tile_size, font_size or 0)
    bar_mid_y = number_bottom + max(1, (tile_size - number_bottom) // 2)
    y0 = tile_y + max(0, min(tile_size - bar_h, bar_mid_y - bar_h // 2))
    y1 = y0 + bar_h
    x0 = tile_x + bar_inset
    x1 = tile_x + tile_size - bar_inset
    return x0, y0, x1, y1


def round_canvas_height(h: int) -> int:
    return (h + 1) // 2 * 2


def compute_layout(quality: int, puzzle_w: int, puzzle_h: int, grid_only: bool = False,
                   no_header: bool = False, no_details: bool = False,
                   adjust_height: bool = False,
                   font_size_override: Optional[int] = None) -> dict:
    """Compute layout parameters from a target video height preset.
    Canvas height is always the exact quality — no edge padding.
    pad = gap between puzzle grid and stats panel (small).
    header_h = timer bar height."""
    scale = quality / 1080
    header_h = max(8, int(round(32 * scale)))
    gap = max(1, int(round(4 * scale)))
    panel_w = max(80, int(round(STATS_PANEL_WIDTH * scale)))

    max_dim = max(puzzle_w, puzzle_h)
    hide_header = grid_only or no_header
    if hide_header:
        avail_h = quality
    else:
        avail_h = quality - header_h
    tile_size = avail_h // max_dim
    tile_size = max(MIN_TILE, tile_size)

    font_size = compute_font_size(puzzle_w, puzzle_h, tile_size, font_size_override)

    puzzle_px_w = puzzle_w * tile_size
    puzzle_px_h = puzzle_h * tile_size

    if grid_only or no_details:
        canvas_w = (puzzle_px_w + 1) // 2 * 2
    else:
        canvas_w = (puzzle_px_w + gap + panel_w + 1) // 2 * 2
    if adjust_height:
        if grid_only:
            canvas_h = (puzzle_px_h + 1) // 2 * 2
        else:
            content_h = (0 if hide_header else header_h) + puzzle_px_h
            canvas_h = (content_h + 1) // 2 * 2
    else:
        canvas_h = (quality + 1) // 2 * 2

    return {
        "pad": gap,
        "header_h": header_h,
        "panel_w": panel_w,
        "tile_size": tile_size,
        "font_size": font_size,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
    }


def compute_canvas_dimensions(puzzle_w: int, puzzle_h: int, tile_size: int,
                              grid_only: bool = False,
                              pad: int = None, header_h: int = None,
                              panel_w: int = None,
                              quality: int = None,
                              no_details: bool = False,
                              adjust_height: bool = False) -> tuple[int, int]:
    if puzzle_w < 1 or puzzle_h < 1:
        raise ValueError(f"puzzle dimensions must be >= 1, got {puzzle_w}x{puzzle_h}")
    if tile_size < 1:
        raise ValueError(f"tile_size must be >= 1, got {tile_size}")

    if pad is None:
        pad = PADDING
    if header_h is None:
        header_h = HEADER_H
    if panel_w is None:
        panel_w = STATS_PANEL_WIDTH

    puzzle_px_w = puzzle_w * tile_size
    puzzle_px_h = puzzle_h * tile_size

    if grid_only or no_details:
        cw = (puzzle_px_w + 1) // 2 * 2
        if quality is not None:
            if adjust_height:
                ch = (puzzle_px_h + 1) // 2 * 2 if grid_only else (header_h + puzzle_px_h + 1) // 2 * 2
            else:
                ch = (quality + 1) // 2 * 2
        else:
            if adjust_height and not grid_only:
                ch = (header_h + puzzle_px_h + 1) // 2 * 2
            else:
                ch = (puzzle_px_h + 1) // 2 * 2
    else:
        cw = (puzzle_px_w + panel_w + pad + 1) // 2 * 2
        if quality is not None:
            if adjust_height:
                ch = (header_h + puzzle_px_h + 1) // 2 * 2
            else:
                ch = (quality + 1) // 2 * 2
        else:
            base_h = header_h + puzzle_px_h
            ch = (base_h + 1) // 2 * 2

    return cw, ch


@dataclass(frozen=True)
class RenderOptions:
    grid_only: bool = False
    no_border: bool = False
    no_secondary_border: bool = False
    no_grid_bars: bool = False
    no_numbers: bool = False
    no_header: bool = False
    no_details: bool = False
    dynamic_md: bool = False
    cycles_detection: bool = False
    adjust_height: bool = False
    grid1_color: Optional[Tuple[int, int, int]] = None
    grid2_color: Optional[Tuple[int, int, int]] = None
    tile_bg_color: Optional[Tuple[int, int, int]] = None
    animate_moves: bool = False
    hue_start: float = 0.0
    hue_end: float = 330.0
    saturation_min: float = 0.78
    saturation_max: float = 0.78
    brightness_min: float = 0.6
    brightness_max: float = 0.6
    font_family: Optional[str] = None
    font_bold: bool = False
    font_size_override: Optional[int] = None
    cursor_dance_path: Optional[str] = None


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
    if should_draw_tile_border(tile_size) and not opts.no_border:
        draw.line([(0, 0), (tile_size - 1, 0)], fill=TILE_BORDER_COLOR, width=TILE_BORDER_WIDTH)
        draw.line([(0, 0), (0, tile_size - 1)], fill=TILE_BORDER_COLOR, width=TILE_BORDER_WIDTH)
    return im


def _bar_sprite(color, tile_size, opts, font_size: int = None):
    x0, y0, x1, y1 = compute_secondary_bar_rect(tile_size, font_size=font_size)

    im = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    bar_bbox = (x0, y0, max(x0, x1 - 1), max(y0, y1 - 1))
    draw.rectangle(bar_bbox, fill=color)
    if should_draw_secondary_border_rect(tile_size, (x0, y0, x1, y1)) and not opts.no_secondary_border:
        draw.rectangle(bar_bbox, outline=TILE_BORDER_COLOR, width=1)
    return im


def prerender_composite_tile(num: int, main_bg, sec_bg, tile_sprites: TileSpriteCache, opts: RenderOptions) -> Image.Image:
    """Pre-composite one tile: base + number + bar into single RGBA PIL Image."""
    base = select_base(main_bg, num, tile_sprites)
    composite = base.copy()
    if not opts.no_numbers and num != 0:
        nt = tile_sprites.number_texts[num]
        composite.paste(nt, (0, 0), nt)
    if sec_bg is not None and not opts.no_grid_bars:
        bar = select_bar(sec_bg, tile_sprites)
        if bar is not None:
            composite.paste(bar, (0, 0), bar)
    return composite.convert("RGBA")


def select_base(main_bg, num, cache: TileSpriteCache):
    if main_bg is None:
        bg_color = cache.opts.tile_bg_color or TILE_BG
        return cache.base_sprites[bg_color]
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

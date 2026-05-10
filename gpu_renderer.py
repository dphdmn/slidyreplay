import math
import json
import os
import time as _time_module
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional
from debug_log import get_logger

log = get_logger()

class CancelError(Exception):
    pass


_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

_font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT = os.path.join(_font_dir, "Roboto-Regular.ttf")
FONT_BOLD = os.path.join(_font_dir, "Roboto-Bold.ttf")
FONT_MONO = os.path.join(_font_dir, "JetBrainsMono-Regular.ttf")
FONT_MONO_BOLD = os.path.join(_font_dir, "JetBrainsMono-Bold.ttf")
BG_COLOR = (18, 18, 18)
TILE_BG = (51, 51, 51)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_BORDER_COLOR = (0, 0, 0)
TILE_BORDER_WIDTH = 1
PADDING = 20
STATS_PANEL_WIDTH = 300
HEADER_H = 56
INFO_H = 40
TIMER_BG = (22, 22, 22)
PANEL_BG = (17, 17, 17)
PANEL_ALPHA = 0.69
CYAN = (0, 255, 255)
GREEN = (0, 255, 0)
WHITE_C = (255, 255, 255)
GRAY_C = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
ACCURATE_COLOR = (0, 255, 0)
INACCURATE_COLOR = (255, 255, 255)
NULL_COLOR = (248, 24, 148)


_font_cache = {}

def _font(size, bold=False, mono=False):
    key = (size, bold, mono)
    cached = _font_cache.get(key)
    if cached is not None:
        return cached
    try:
        if mono:
            name = FONT_MONO_BOLD if bold else FONT_MONO
        else:
            name = FONT_BOLD if bold else FONT
        f = ImageFont.truetype(name, size)
    except Exception:
        try:
            f = ImageFont.truetype(FONT_MONO if mono else FONT, size)
        except Exception:
            f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def _render_number_textures(max_num: int, ts: int, font_size: int) -> dict:
    textures = {}
    for num in range(max_num + 1):
        im = Image.new("RGBA", (ts, ts), (0, 0, 0, 0))
        draw = ImageDraw.Draw(im)
        if num != 0:
            text = str(num)
            f = _font(font_size)
            b = draw.textbbox((0, 0), text, font=f)
            tx = ts // 2 - (b[0] + b[2]) // 2
            ty = ts // 2 - (b[1] + b[3]) // 2
            draw.text((tx, ty), text, fill=(0, 0, 0, 255), font=f)
        textures[num] = im
    return textures


def _render_text_rgba(text, font, fill, pad=0):
    b = font.getbbox(text)
    w = b[2] - b[0] + pad * 2
    h = b[3] - b[1] + pad * 2
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.text((pad - b[0], pad - b[1]), text, fill=(*fill, 255), font=font)
    return im


def _render_timer_text(timer_text: str) -> Image.Image:
    font = _font(36, bold=True, mono=True)
    return _render_text_rgba(timer_text, font, CYAN)



class GPURenderer:
    def __init__(self, width: int, height: int, tile_size: int, quality: float = 1.0, min_canvas_h: int = None):
        """
        GPU renderer with automatic optimal batching.

        Self‑calibrates on the first frame, then uses as large a batch as the
        available free memory allows, while respecting a soft 50% VRAM ceiling.
        No knobs required.
        """
        self.w = width
        self.h = height
        ts = max(tile_size, int(tile_size * quality))
        self.tile_size = ts
        self.font_size = max(11, ts // 2)
        self.pw = width * ts
        self.ph = height * ts

        puzzle_w = width * ts
        puzzle_h = height * ts
        self.canvas_w = (puzzle_w + STATS_PANEL_WIDTH + PADDING * 3 + 1) // 2 * 2
        default_h = (HEADER_H + puzzle_h + PADDING * 3 + 1) // 2 * 2
        if min_canvas_h is not None:
            default_h = max(default_h, min_canvas_h)
        self.canvas_h = (default_h + 1) // 2 * 2
        self.grid_x = PADDING
        self.grid_y = PADDING + HEADER_H + PADDING
        self.panel_x = self.grid_x + puzzle_w + PADDING
        self.panel_y = self.grid_y
        self.panel_w = self.canvas_w - self.panel_x - PADDING
        self.panel_h = self.canvas_h - self.panel_y - PADDING
        self.timer_bbox = (PADDING, PADDING, self.canvas_w - PADDING, PADDING + HEADER_H)

        self._device = None
        self._num_batch = None
        self._tile_mask = None
        self._timer_bg = None
        self._panel_bg = None
        self._static_stats_bg_rgb = None
        self._static_stats_bg_a = None
        self._overlay_text_positions = None

        if _HAS_TORCH:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._stats = {
            "gpu_name": "",
            "total_mem_mb": 0,
            "free_mem_mb": 0,
            "mem_used_mb": 0,
            "batch_size": 0,
            "batch_mem_mb": 0,
        }
        self._batch_counter = 0
        log.info(f"GPURenderer.__init__: {width}x{height}, tile_size={ts}, canvas={self.canvas_w}x{self.canvas_h}, device={self._device}")
        if self._stats["total_mem_mb"]:
            log.info(f"  GPU: {self._stats['gpu_name']}, total_mem={self._stats['total_mem_mb']}MB")
        if self._device is not None and self._device.type == "cuda":
            total = torch.cuda.get_device_properties(self._device).total_memory
            self._stats["gpu_name"] = torch.cuda.get_device_name(self._device)
            self._stats["total_mem_mb"] = total // (1024 * 1024)

        if self._device is not None and self._device.type == "cuda":
            cuda_dev = self._device

            tex_map = _render_number_textures(width * height, ts, self.font_size)
            n = len(tex_map)
            num_batch = torch.zeros(n, ts, ts, 4, device=cuda_dev, dtype=torch.float32)
            for num, pil in tex_map.items():
                arr = torch.from_numpy(np.array(pil)).to(cuda_dev).float() / 255.0
                num_batch[num] = arr
            self._num_batch = num_batch

            # Tile mask (all ones) and its complement
            mask = torch.ones(ts, ts, 1, device=cuda_dev, dtype=torch.float32)
            self._tile_mask = mask
            self._tile_mask_inv = 1.0 - mask

            # Tile border mask: 1 px border
            border = torch.zeros(ts, ts, 1, device=cuda_dev, dtype=torch.float32)
            border[0, :] = 1
            border[ts - 1, :] = 1
            border[:, 0] = 1
            border[:, ts - 1] = 1
            self._border_mask = border
            self._border_mask_inv = 1.0 - border

            # Secondary colour bar masks
            bar_h = max(2, int(ts * 0.1))
            bar_off = max(2, int(ts * 0.06))
            bar_inset = max(2, int(ts * 0.1))
            bar_fill = torch.zeros(ts, ts, 1, device=cuda_dev, dtype=torch.float32)
            bar_border = torch.zeros(ts, ts, 1, device=cuda_dev, dtype=torch.float32)
            by0 = ts - bar_h - bar_off
            by1 = ts - bar_off
            bx0 = bar_inset
            bx1 = ts - bar_inset
            bar_fill[by0:by1, bx0:bx1] = 1.0
            bar_border[by0:by1, bx0:bx1] = 1.0
            if by0 > 0:
                bar_border[by0, bx0:bx1] = 1.0
            if by1 < ts:
                bar_border[by1 - 1, bx0:bx1] = 1.0
            if bx0 > 0:
                bar_border[by0:by1, bx0] = 1.0
            if bx1 < ts:
                bar_border[by0:by1, bx1 - 1] = 1.0
            if by1 - by0 > 2 and bx1 - bx0 > 2:
                bar_border[by0 + 1:by1 - 1, bx0 + 1:bx1 - 1] = 0.0
            self._bar_fill = bar_fill
            self._bar_border = bar_border

            # Timer bar background
            tx1, ty1, tx2, ty2 = self.timer_bbox
            tw = tx2 - tx1
            th = ty2 - ty1
            timer_bg_pil = Image.new("RGB", (tw, th), TIMER_BG)
            self._timer_bg = torch.from_numpy(
                np.array(timer_bg_pil)
            ).to(cuda_dev).float() / 255.0

            # Stats panel background + border
            if self.panel_w > 0 and self.panel_h > 0:
                panel_pil = Image.new("RGBA", (self.panel_w, self.panel_h), (0, 0, 0, 0))
                pdraw = ImageDraw.Draw(panel_pil)
                pdraw.rectangle(
                    (0, 0, self.panel_w - 1, self.panel_h - 1),
                    fill=(*PANEL_BG, int(255 * PANEL_ALPHA))
                )
                pdraw.rectangle(
                    (0, 0, self.panel_w - 1, self.panel_h - 1),
                    outline=CYAN, width=1
                )
                panel_arr = torch.from_numpy(np.array(panel_pil)).to(cuda_dev).float() / 255.0
                self._panel_bg = panel_arr
                self._panel_bg_rgb = panel_arr[:, :, :3]
                self._panel_bg_a = panel_arr[:, :, 3:4]

    @property
    def available(self):
        return self._device is not None and self._device.type == "cuda" and self._num_batch is not None

    @staticmethod
    def _blend_rgba_inplace(dst: torch.Tensor, src_rgba: torch.Tensor, x0: int, y0: int):
        """Blend src RGBA onto dst at (x0, y0), modifying dst in-place."""
        h = min(src_rgba.shape[0], dst.shape[0] - y0)
        w = min(src_rgba.shape[1], dst.shape[1] - x0)
        if h <= 0 or w <= 0:
            return
        src = src_rgba[:h, :w]
        a = src[:, :, 3:4]
        dst_slice = dst[y0:y0 + h, x0:x0 + w]
        dst_slice.mul_(1.0 - a).add_(src[:, :, :3] * a)

    def render_frames(
        self,
        frame_params_list: List[dict],
        progress_callback=None,
        cancel_check=None,
        frame_handler=None,
    ) -> List[Image.Image]:
        if not self.available or not frame_params_list:
            return self._cpu_fallback(frame_params_list, progress_callback, cancel_check)

        n = len(frame_params_list)
        w, h, ts = self.w, self.h, self.tile_size
        pw, ph = self.pw, self.ph
        cw, ch = self.canvas_w, self.canvas_h
        gx, gy = self.grid_x, self.grid_y
        px, py = self.panel_x, self.panel_y
        tx1, ty1, tx2, ty2 = self.timer_bbox
        th, tw = ty2 - ty1, tx2 - tx1

        dev = self._device
        num_tex = self._num_batch
        # 6‑D broadcast views for blending
        tm = self._tile_mask[None, None, None, :, :, :]
        tm_inv = self._tile_mask_inv[None, None, None, :, :, :]
        bm = self._border_mask[None, None, None, :, :, :]
        bm_inv = self._border_mask_inv[None, None, None, :, :, :]
        bar_fill_mask = self._bar_fill[None, None, None, :, :, :]
        bar_border_mask = self._bar_border[None, None, None, :, :, :]
        timer_bg = self._timer_bg
        panel_bg_rgb = self._panel_bg_rgb
        panel_bg_a = self._panel_bg_a

        c_bg = torch.tensor(BG_COLOR, device=dev, dtype=torch.float32) / 255.0
        c_border = torch.tensor(TILE_BORDER_COLOR, device=dev, dtype=torch.float32) / 255.0
        c_bg_6d = c_bg.view(1, 1, 1, 1, 1, 3)
        c_border_6d = c_border.view(1, 1, 1, 1, 1, 3)
        tile_bg_t = torch.tensor(TILE_BG, device=dev, dtype=torch.float32) / 255.0

        total_mem = torch.cuda.get_device_properties(dev).total_memory
        batch_has_colors = any(p.get("colors") is not None for p in frame_params_list)

        # Soft 50% VRAM ceiling – we’ll exceed it if needed, bounded by free memory
        target_mem_fraction = 0.50
        target_used_mem = int(total_mem * target_mem_fraction)

        per_frame_ema = 0.0
        batch_size = 1

        self._stats["batch_size"] = batch_size
        self._stats["batch_mem_mb"] = 0
        self._stats["per_frame_ema_mb"] = 0.0
        self._stats["target_used_mem_mb"] = target_used_mem // (1024 * 1024)

        # Row‑chunked rendering: 2 rows per chunk for large puzzles
        chunk_rows = 2 if h > 6 else h

        frames = []
        batch_start = 0

        # Permanent reserved memory right after static tensors are loaded
        reserved_permanent = torch.cuda.memory_reserved(dev)
        log.info(f"render_frames: {n} frames, canvas={cw}x{ch}, chunk_rows={chunk_rows}, reserved_static={reserved_permanent // (1024*1024)}MB, target_used_mem={target_used_mem // (1024*1024)}MB")

        with torch.inference_mode():
            while batch_start < n:
                if cancel_check and cancel_check():
                    raise CancelError()

                remaining = n - batch_start

                # ── Memory‑aware batching ──
                reserved_mem = torch.cuda.memory_reserved(dev)
                free_mem, _ = torch.cuda.mem_get_info(dev)

                # 256 MB safety margin to avoid edge‑cases
                reserve_margin = 256 * 1024 * 1024

                # Soft target headroom – if reserved_mem exceeds the target we ignore it
                # and rely solely on physical headroom.
                target_headroom = target_used_mem - reserved_mem - reserve_margin
                physical_headroom = int(free_mem * 0.90)

                if target_headroom > 0:
                    usable = min(target_headroom, physical_headroom)
                else:
                    usable = physical_headroom

                if usable < 0:
                    usable = 0

                if per_frame_ema > 0:
                    max_by_budget = max(1, int(usable / per_frame_ema))
                    batch_size = max(1, min(remaining, max_by_budget))
                else:
                    batch_size = 1

                batch_end = min(batch_start + batch_size, n)
                batch_n = batch_end - batch_start

                self._stats["batch_size"] = batch_size
                self._stats["batch_mem_mb"] = (ch * cw * 3 * 4 * batch_size) // (1024 * 1024)

                log.info(f"  BATCH: batch_start={batch_start}, batch_n={batch_n}, batch_size={batch_size}, free_mem={free_mem // (1024*1024)}MB, usable={usable // (1024*1024)}MB, per_frame_ema={per_frame_ema // (1024*1024) if per_frame_ema > 0 else 'N/A'}MB")

                # Fallback to CPU only if physical memory cannot fit a single frame
                if batch_n == 1 and per_frame_ema > 0 and usable < per_frame_ema:
                    log.warning(f"  GPU OOM FALLBACK: remaining={remaining}, usable={usable}, per_frame_ema={per_frame_ema}")
                    return self._cpu_fallback(frame_params_list[batch_start:],
                                              progress_callback, cancel_check)

                self._stats["free_mem_mb"] = free_mem // (1024 * 1024)
                self._stats["mem_used_mb"] = self._stats["total_mem_mb"] - self._stats["free_mem_mb"]
                self._batch_counter += 1
                self._stats["batch_idx"] = self._batch_counter

                # ── Upload batch data ──
                batch_params = frame_params_list[batch_start:batch_end]
                mats_np = np.stack([p["matrix"] for p in batch_params], axis=0)
                mats = torch.from_numpy(mats_np).to(dev, non_blocking=True)

                if batch_has_colors:
                    main_np = np.full((batch_n, h, w, 3), TILE_BG, dtype=np.float32)
                    sec_np = np.zeros((batch_n, h, w, 3), dtype=np.float32)
                    has_sec_np = np.zeros((batch_n, h, w), dtype=np.float32)
                    for i, p in enumerate(batch_params):
                        cd = p.get("colors")
                        if cd:
                            flat = [(mc, sc) for row_c in cd for mc, sc in row_c]
                            mc_flat = np.array([c for c, _ in flat], dtype=np.float32)
                            main_np[i] = mc_flat.reshape(h, w, 3)
                            sc_idx = [idx for idx, (_, sc) in enumerate(flat) if sc is not None]
                            sc_colors = [flat[idx][1] for idx in sc_idx]
                            if sc_idx:
                                sc_arr = np.array(sc_colors, dtype=np.float32)
                                rs = np.array([idx // w for idx in sc_idx])
                                cs = np.array([idx % w for idx in sc_idx])
                                sec_np[i, rs, cs] = sc_arr
                                has_sec_np[i, rs, cs] = 1.0
                    cols = torch.from_numpy(main_np).to(dev, non_blocking=True) / 255.0
                    sec = torch.from_numpy(sec_np).to(dev, non_blocking=True) / 255.0
                    has_sec = torch.from_numpy(has_sec_np).to(dev, non_blocking=True).view(batch_n, h, w, 1, 1, 1)
                else:
                    cols = tile_bg_t.view(1, 1, 1, 3).expand(batch_n, h, w, 3)
                    sec = torch.zeros(batch_n, h, w, 3, device=dev, dtype=torch.float32)
                    has_sec = torch.zeros(batch_n, h, w, 1, 1, 1, device=dev, dtype=torch.float32)

                torch.cuda.reset_peak_memory_stats(dev)

                # ── Canvas ──
                canvas = torch.empty(batch_n, ch, cw, 3, device=dev, dtype=torch.float32)
                canvas[:] = c_bg.view(1, 1, 1, 3)
                canvas[:, ty1:ty2, tx1:tx2] = timer_bg.view(1, th, tw, 3)
                if panel_bg_rgb is not None and self.panel_h > 0 and self.panel_w > 0:
                    ph_p, pw_p = self.panel_h, self.panel_w
                    canvas[:, py:py + ph_p, px:px + pw_p] = (
                        panel_bg_rgb * panel_bg_a +
                        canvas[:, py:py + ph_p, px:px + pw_p] * (1 - panel_bg_a)
                    )

                # ── Row‑chunked tile rendering ──
                for row_start in range(0, h, chunk_rows):
                    row_end = min(row_start + chunk_rows, h)
                    n_rows = row_end - row_start

                    mats_chunk = mats[:, row_start:row_end, :]
                    nums_chunk = num_tex[mats_chunk]   # (B, n_rows, W, ts, ts, 4)
                    text_rgb_chunk = nums_chunk[..., :3]
                    text_a_chunk = nums_chunk[..., 3:]

                    cols_chunk = cols[:, row_start:row_end, ...]
                    if batch_has_colors:
                        sec_chunk = sec[:, row_start:row_end, ...]
                        has_sec_chunk = has_sec[:, row_start:row_end, ...]
                    else:
                        sec_chunk = None
                        has_sec_chunk = None

                    # In‑place tile blending
                    colored = cols_chunk.view(batch_n, n_rows, w, 1, 1, 3) * tm
                    colored.addcmul_(c_bg_6d, tm_inv)
                    colored.mul_(bm_inv).addcmul_(c_border_6d, bm)

                    if batch_has_colors:
                        bar_fill_factor = has_sec_chunk * bar_fill_mask
                        bar_border_factor = has_sec_chunk * bar_border_mask

                        colored.mul_(1 - bar_fill_factor)
                        sec_contrib = sec_chunk.view(batch_n, n_rows, w, 1, 1, 3) * bar_fill_mask
                        colored.add_(sec_contrib * bar_fill_factor)
                        del sec_contrib

                        colored.mul_(1 - bar_border_factor)
                        colored.addcmul_(c_border_6d, bar_border_factor)

                    tile_chunk = text_rgb_chunk * text_a_chunk + colored * (1 - text_a_chunk)
                    tile_chunk = tile_chunk.permute(0, 1, 3, 2, 4, 5).reshape(batch_n, n_rows * ts, pw, 3)
                    canvas_y = gy + row_start * ts
                    canvas[:, canvas_y:canvas_y + n_rows * ts, gx:gx + pw] = tile_chunk

                    del nums_chunk, text_rgb_chunk, text_a_chunk, colored, tile_chunk

                # ── Overlays (in‑place) ──
                for i in range(batch_n):
                    fi = batch_start + i
                    fc = canvas[i]
                    params = frame_params_list[fi]

                    timer_arr = params.get("timer_arr")
                    if timer_arr is not None:
                        tt = torch.from_numpy(timer_arr).to(dev, non_blocking=True).float() / 255.0
                        dx = max(tx1, tx1 + ((tx2 - tx1) - tt.shape[1]) // 2)
                        dy = max(ty1, ty1 + ((ty2 - ty1) - tt.shape[0]) // 2)
                        self._blend_rgba_inplace(fc, tt, dx, dy)

                    stats_arr = params.get("stats_arr")
                    if stats_arr is not None:
                        stt = torch.from_numpy(stats_arr).to(dev, non_blocking=True).float() / 255.0
                        self._blend_rgba_inplace(fc, stt, px, py)

                # ── GPU → CPU (uint8) ──
                batch_u8 = canvas.mul(255.0).clamp_(0, 255).to(torch.uint8).cpu().numpy()
                for i in range(batch_n):
                    img = Image.fromarray(batch_u8[i])
                    if frame_handler:
                        frame_handler(img, batch_start + i, n)
                    else:
                        frames.append(img)
                    if progress_callback:
                        progress_callback(batch_start + i + 1, n, gpu_stats=dict(self._stats))

                # ── One‑shot calibration (first batch only) ──
                torch.cuda.synchronize(dev)
                reserved_peak = torch.cuda.memory_reserved(dev)

                if self._batch_counter == 1 and batch_n == 1 and per_frame_ema == 0.0:
                    marginal_cost = reserved_peak - reserved_permanent
                    if marginal_cost > 0:
                        per_frame_ema = marginal_cost
                        self._stats["per_frame_ema_mb"] = per_frame_ema / (1024 * 1024)

                # ── Free batch tensors ──
                del canvas, mats
                if batch_has_colors:
                    del cols, sec, has_sec

                batch_start = batch_end

        log.info(f"render_frames: DONE. total_batches={self._batch_counter}, frames_rendered={batch_start}")
        torch.cuda.empty_cache()
        return frames if not frame_handler else []

    def cleanup(self):
        if self._device is not None and self._device.type == "cuda":
            attrs = [
                "_num_batch", "_tile_mask", "_tile_mask_inv",
                "_border_mask", "_border_mask_inv",
                "_bar_fill", "_bar_border",
                "_timer_bg", "_panel_bg", "_panel_bg_rgb", "_panel_bg_a",
            ]
            for attr in attrs:
                t = getattr(self, attr, None)
                if t is not None:
                    del t
                    setattr(self, attr, None)
            torch.cuda.empty_cache()
            log.info("GPURenderer.cleanup: CUDA tensors freed")

    def _cpu_fallback(self, frame_params_list, progress_callback, cancel_check=None):
        frames = []
        w, h, ts = self.w, self.h, self.tile_size
        tex_map = _render_number_textures(w * h, ts, self.font_size)

        for i, p in enumerate(frame_params_list):
            if cancel_check and cancel_check():
                raise CancelError()
            mat = p["matrix"]
            colors_data = p.get("colors", None)
            timer_img = p.get("timer_img")
            stats_img = p.get("stats_img")

            cw, ch = self.canvas_w, self.canvas_h
            canvas = Image.new("RGB", (cw, ch), BG_COLOR)
            draw = ImageDraw.Draw(canvas)

            tx1, ty1, tx2, ty2 = self.timer_bbox
            draw.rectangle((tx1, ty1, tx2, ty2), fill=TIMER_BG)

            if timer_img:
                ti_w, ti_h = timer_img.size
                tx = tx1 + ((tx2 - tx1) - ti_w) // 2
                ty = ty1 + ((ty2 - ty1) - ti_h) // 2
                canvas.paste(timer_img, (tx, ty), timer_img)

            gx, gy = self.grid_x, self.grid_y
            for row in range(h):
                for col in range(w):
                    num = mat[row][col]
                    sx, sy = gx + col * ts, gy + row * ts
                    if colors_data is not None:
                        mc, sc = colors_data[row][col]
                        if mc is None:
                            mc = TILE_BG
                    else:
                        mc = TILE_BG
                    draw.rectangle((sx, sy, sx + ts - 1, sy + ts - 1), fill=mc)
                    if ts > 1:
                        draw.rectangle((sx, sy, sx + ts - 1, sy + ts - 1),
                                       outline=TILE_BORDER_COLOR, width=TILE_BORDER_WIDTH)
                    if sc is not None and ts > 1:
                        bar_h = max(2, int(ts * 0.1))
                        bar_off = max(2, int(ts * 0.06))
                        bar_inset = max(2, int(ts * 0.1))
                        bar_y0 = sy + ts - bar_h - bar_off
                        bar_y1 = sy + ts - bar_off
                        bar_x0 = sx + bar_inset
                        bar_x1 = sx + ts - bar_inset
                        draw.rectangle((bar_x0, bar_y0, bar_x1, bar_y1), fill=sc)
                        draw.rectangle((bar_x0, bar_y0, bar_x1, bar_y1),
                                       outline=TILE_BORDER_COLOR, width=1)
                    if num != 0:
                        text = str(num)
                        f = _font(self.font_size)
                        b = draw.textbbox((0, 0), text, font=f)
                        tx2d = sx + ts // 2 - (b[0] + b[2]) // 2
                        ty2d = sy + ts // 2 - (b[1] + b[3]) // 2
                        draw.text((tx2d, ty2d), text, fill=TILE_TEXT_COLOR, font=f)

            pw, ph = self.panel_w, self.panel_h
            px, py = self.panel_x, self.panel_y
            if pw > 0 and ph > 0:
                panel = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
                pdraw = ImageDraw.Draw(panel)
                pdraw.rectangle((0, 0, pw - 1, ph - 1),
                                fill=(*PANEL_BG, int(255 * PANEL_ALPHA)))
                pdraw.rectangle((0, 0, pw - 1, ph - 1), outline=CYAN, width=1)
                canvas.paste(panel, (px, py), panel)
                if stats_img:
                    canvas.paste(stats_img, (px, py), stats_img)

            frames.append(canvas)
            if progress_callback:
                progress_callback(i + 1, len(frame_params_list))

        return frames
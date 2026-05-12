from __future__ import annotations
import math
import json
import os
import time as _time_module
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional
from debug_log import get_logger
import psutil as _psutil

_gpu_proc = _psutil.Process()
_gpu_baseline_ram = _gpu_proc.memory_info().rss

def _ram_delta_mb() -> int:
    return (_gpu_proc.memory_info().rss - _gpu_baseline_ram) // (1024 * 1024)

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
from geometry import (PADDING, HEADER_H, STATS_PANEL_WIDTH, INFO_H, TIMER_HEIGHT,
    BG_COLOR, TILE_BG, TILE_TEXT_COLOR, TILE_BORDER_COLOR, NULL_COLOR,
    PANEL_BG, PANEL_ALPHA, TIMER_BG, ACCURATE_COLOR, INACCURATE_COLOR,
    WHITE, CYAN, GREEN, GRAY, LIGHT_GRAY,
    TILE_BORDER_WIDTH, TILE_BORDER_RADIUS_RATIO, BASE_SIZE,
    compute_canvas_dimensions, RenderOptions)


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


@lru_cache(maxsize=128)
def _render_timer_text(timer_text: str) -> Image.Image:
    font = _font(36, bold=True, mono=True)
    return _render_text_rgba(timer_text, font, CYAN)



class GPURenderer:
    def __init__(self, width: int, height: int, tile_size: int, quality: float = 1.0,
                 min_canvas_h: int = None, opts: Optional[RenderOptions] = None):
        """
        GPU renderer with automatic optimal batching.

        Self‑calibrates on the first frame, then uses as large a batch as the
        available free memory allows, while respecting a soft 50% VRAM ceiling.
        No knobs required.
        """
        self.opts = opts or RenderOptions()
        self.w = width
        self.h = height
        ts = max(tile_size, int(tile_size * quality))
        self.tile_size = ts
        self.font_size = max(11, ts // 2)
        self.pw = width * ts
        self.ph = height * ts

        puzzle_w = width * ts
        puzzle_h = height * ts
        self.canvas_w, self.canvas_h = compute_canvas_dimensions(
            width, height, ts, grid_only=self.opts.grid_only
        )
        if not self.opts.grid_only and min_canvas_h is not None:
            self.canvas_h = max(self.canvas_h, min_canvas_h)
            self.canvas_h = (self.canvas_h + 1) // 2 * 2
        self.grid_x = PADDING
        self.grid_y = PADDING if self.opts.grid_only else PADDING + HEADER_H + PADDING
        self.panel_x = self.grid_x + puzzle_w + PADDING
        self.panel_y = self.grid_y
        self.panel_w = self.canvas_w - self.panel_x - PADDING
        self.panel_h = self.canvas_h - self.panel_y - PADDING
        self.timer_bbox = (0, 0, 0, 0) if self.opts.grid_only else (PADDING, PADDING, self.canvas_w - PADDING, PADDING + HEADER_H)

        self._init_success = False
        self._device = None
        self._num_batch = None
        self._tile_mask = None
        self._timer_bg = None
        self._panel_bg = None
        self._panel_bg_rgb = None
        self._panel_bg_a = None
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

            if self.opts.no_numbers:
                self._num_batch = None
            else:
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
            if th > 0 and tw > 0:
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

        self._init_success = True

    @property
    def available(self):
        return (self._device is not None and self._device.type == "cuda"
                and self._init_success)

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
        overlay_render_data=None,
    ) -> List[Image.Image]:
        if not self.available:
            raise RuntimeError(
                "GPU renderer is not available. CUDA device not found or "
                "initialization failed. Disable GPU acceleration to use CPU rendering."
            )
        if not frame_params_list:
            return []

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
        target_mem_fraction = 0.70
        target_used_mem = int(total_mem * target_mem_fraction)

        per_frame_ema = 0.0
        batch_size = 1
        prev_batch_n = 0

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
            _batch_t0 = _time_module.time()
            while batch_start < n:
                if cancel_check and cancel_check():
                    raise CancelError()

                remaining = n - batch_start

                # ── Memory‑aware batching ──
                allocated_mem = torch.cuda.memory_allocated(dev)
                free_mem, _ = torch.cuda.mem_get_info(dev)

                # 256 MB safety margin to avoid edge‑cases
                reserve_margin = 128 * 1024 * 1024

                # Soft target headroom – if allocated_mem exceeds the target we ignore it
                # and rely solely on physical headroom.
                target_headroom = target_used_mem - allocated_mem - reserve_margin
                physical_headroom = int((total_mem - allocated_mem) * 0.90)

                if target_headroom > 0:
                    usable = min(target_headroom, physical_headroom)
                else:
                    usable = physical_headroom

                if usable < 0:
                    usable = 0

                if per_frame_ema > 0:
                    max_by_budget = max(1, int(usable / per_frame_ema * 0.85))
                    batch_size = max(1, min(remaining, max_by_budget))
                    # EMA dampener: smooth batch size transitions
                    if prev_batch_n > 0:
                        damp = 0.5
                        batch_size = max(1, int(prev_batch_n * (1 - damp) + batch_size * damp))
                        batch_size = min(remaining, batch_size)
                else:
                    batch_size = 1

                batch_end = min(batch_start + batch_size, n)
                batch_n = batch_end - batch_start

                self._stats["batch_size"] = batch_size
                self._stats["batch_mem_mb"] = (ch * cw * 3 * 4 * batch_size) // (1024 * 1024)

                log.info(f"  BATCH[{self._batch_counter}]: start={batch_start}, sz={batch_size}, "
                         f"free={free_mem//(1024*1024)}MB, allocated={allocated_mem//(1024*1024)}MB, "
                         f"usable={usable//(1024*1024)}MB, ema={per_frame_ema//(1024*1024)}MB, "
                         f"batch_mem={self._stats['batch_mem_mb']}MB, t={_time_module.time()-_batch_t0:.2f}s, ram={_ram_delta_mb()}MB")

                # Early OOM guard — even the calibration batch needs VRAM
                if batch_n == 1 and per_frame_ema == 0 and usable <= 0:
                    msg = (
                        f"GPU out of memory: no usable VRAM available "
                        f"(0MB usable, {free_mem // (1024*1024)}MB free). "
                        f"Try disabling GPU acceleration."
                    )
                    log.critical(msg)
                    raise RuntimeError(msg)

                # GPU out of memory guard — retry with cache flush before giving up
                if batch_n == 1 and per_frame_ema > 0 and usable < per_frame_ema:
                    log.warning(f"  VRAM low: free={free_mem//(1024*1024)}MB, "
                                f"usable={usable//(1024*1024)}MB < "
                                f"ema={per_frame_ema//(1024*1024)}MB, "
                                f"flushing cache and retrying...")
                    torch.cuda.empty_cache()
                    r2 = torch.cuda.memory_reserved(dev)
                    f2, _ = torch.cuda.mem_get_info(dev)
                    th2 = target_used_mem - r2 - reserve_margin
                    ph2 = int(f2 * 0.90)
                    usable = min(th2, ph2) if th2 > 0 else ph2
                    if usable < 0:
                        usable = 0
                    if usable < per_frame_ema:
                        msg = (
                            f"GPU out of memory: cannot fit a single frame in available VRAM "
                            f"(needs ~{per_frame_ema // (1024*1024)}MB, "
                            f"only ~{usable // (1024*1024)}MB available). "
                            f"Try disabling GPU acceleration, reducing quality, "
                            f"or using a smaller puzzle."
                        )
                        log.critical(msg)
                        raise RuntimeError(msg)
                    log.info(f"  Cache flush recovered {usable//(1024*1024)}MB usable, continuing")

                self._stats["free_mem_mb"] = free_mem // (1024 * 1024)
                self._stats["mem_used_mb"] = self._stats["total_mem_mb"] - self._stats["free_mem_mb"]
                self._batch_counter += 1
                self._stats["batch_idx"] = self._batch_counter

                # ── Upload batch data ──
                batch_params = frame_params_list[batch_start:batch_end]
                if batch_params:
                    p_opts = batch_params[0].get("opts")
                    if not isinstance(p_opts, RenderOptions):
                        raise TypeError(f"batch_params[0]['opts'] must be RenderOptions, got {type(p_opts)}")
                    if p_opts != self.opts:
                        raise ValueError(
                            f"Item opts={p_opts} != renderer opts={self.opts}. "
                            "All items in a render batch must use identical RenderOptions."
                        )
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

                reserved_before_render = torch.cuda.memory_reserved(dev)
                torch.cuda.reset_peak_memory_stats(dev)

                # ── Canvas ──
                canvas = torch.empty(batch_n, ch, cw, 3, device=dev, dtype=torch.float32)
                canvas[:] = c_bg.view(1, 1, 1, 3)
                if timer_bg is not None and th > 0 and tw > 0:
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
                    if not self.opts.no_numbers and num_tex is not None:
                        nums_chunk = num_tex[mats_chunk]
                        text_rgb_chunk = nums_chunk[..., :3]
                        text_a_chunk = nums_chunk[..., 3:]
                    else:
                        nums_chunk = None
                        text_rgb_chunk = None
                        text_a_chunk = None

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
                    if not self.opts.no_border:
                        colored.mul_(bm_inv).addcmul_(c_border_6d, bm)

                    if batch_has_colors:
                        bar_fill_factor = has_sec_chunk * bar_fill_mask

                        colored.mul_(1 - bar_fill_factor)
                        sec_contrib = sec_chunk.view(batch_n, n_rows, w, 1, 1, 3) * bar_fill_mask
                        colored.add_(sec_contrib * bar_fill_factor)
                        del sec_contrib

                        if not self.opts.no_secondary_border:
                            bar_border_factor = has_sec_chunk * bar_border_mask
                            colored.mul_(1 - bar_border_factor)
                            colored.addcmul_(c_border_6d, bar_border_factor)

                    if not self.opts.no_numbers and num_tex is not None:
                        tile_chunk = text_rgb_chunk * text_a_chunk + colored * (1 - text_a_chunk)
                    else:
                        tile_chunk = colored
                    tile_chunk = tile_chunk.permute(0, 1, 3, 2, 4, 5).reshape(batch_n, n_rows * ts, pw, 3)
                    canvas_y = gy + row_start * ts
                    canvas[:, canvas_y:canvas_y + n_rows * ts, gx:gx + pw] = tile_chunk

                    del nums_chunk, text_rgb_chunk, text_a_chunk, colored, tile_chunk

                # ── Overlays (in‑place) ──
                if overlay_render_data is not None:
                    from replay_video import _apply_stats_dynamic as _apply_sd
                for i in range(batch_n):
                    fi = batch_start + i
                    fc = canvas[i]
                    params = frame_params_list[fi]

                    if overlay_render_data is not None:
                        timer_img = _render_timer_text(params["timer_text"])
                        stats_img = _apply_sd(
                            params["stats_data"],
                            overlay_render_data["panel_w_val"],
                            overlay_render_data["static_base"],
                            overlay_render_data["static_layout"]
                        )
                        timer_arr = np.array(timer_img)
                        stats_arr = np.array(stats_img)
                    else:
                        timer_arr = params.get("timer_arr")
                        stats_arr = params.get("stats_arr")

                    if timer_arr is not None:
                        tt = torch.from_numpy(timer_arr).to(dev, non_blocking=True).float() / 255.0
                        dx = max(tx1, tx1 + ((tx2 - tx1) - tt.shape[1]) // 2)
                        dy = max(ty1, ty1 + ((ty2 - ty1) - tt.shape[0]) // 2)
                        self._blend_rgba_inplace(fc, tt, dx, dy)

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
                        log.info(f"  CALIBRATION: reserved_peak={reserved_peak//(1024*1024)}MB, "
                                 f"reserved_permanent={reserved_permanent//(1024*1024)}MB, "
                                 f"marginal_cost={marginal_cost//(1024*1024)}MB, "
                                 f"per_frame_ema={per_frame_ema/(1024*1024):.0f}MB")

                # ── Free batch tensors ──
                del canvas, mats
                if batch_has_colors:
                    del cols, sec, has_sec
                if batch_n > 1:
                    peak_reserved = torch.cuda.max_memory_reserved(dev)
                    marginal = peak_reserved - reserved_before_render
                    if marginal > 0:
                        actual_ppf = marginal / batch_n
                        old_ema = per_frame_ema
                        per_frame_ema = max(per_frame_ema, actual_ppf * 1.15)
                        self._stats["per_frame_ema_mb"] = per_frame_ema / (1024 * 1024)
                        log.info(f"  POST-BATCH EMA: old={old_ema/(1024*1024):.0f}MB, "
                                 f"peak={peak_reserved//(1024*1024)}MB, "
                                 f"marginal={marginal//(1024*1024)}MB, "
                                 f"actual_ppf={actual_ppf/(1024*1024):.0f}MB, "
                                 f"new_ema={per_frame_ema/(1024*1024):.0f}MB")
                log.info(f"  BATCH[{self._batch_counter}] DONE: "
                         f"t={_time_module.time()-_batch_t0:.2f}s, "
                         f"peak={torch.cuda.max_memory_reserved(dev)//(1024*1024)}MB, "
                         f"ram={_ram_delta_mb()}MB")

                prev_batch_n = batch_n
                batch_start = batch_end

        log.info(f"render_frames: DONE. total_batches={self._batch_counter}, frames_rendered={batch_start}")
        total_t = _time_module.time() - _batch_t0
        log.info(f"===== GPU RENDER SUMMARY =====")
        log.info(f"  total_time={total_t:.1f}s, batches={self._batch_counter}, "
                 f"frames={n}, avg_batch_size={n/max(1,self._batch_counter):.1f}, "
                 f"throughput={n/total_t:.0f} f/s (unique)")
        log.info(f"  RAM delta: {_ram_delta_mb()}MB")
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

    def __del__(self):
        if self._device is not None:
            try:
                self.cleanup()
            except Exception:
                pass

    # _cpu_fallback removed — GPU errors now raise RuntimeError instead of silently falling back
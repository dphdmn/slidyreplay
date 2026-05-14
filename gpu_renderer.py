from __future__ import annotations
import math
import json
import os
import time as _time_module

import numpy as np
from PIL import Image, ImageDraw
from typing import List, Optional
from debug_log import get_logger, CancelError

log = get_logger()

import psutil as _psutil
_gpu_proc = _psutil.Process()
_gpu_baseline_ram = _gpu_proc.memory_info().rss


def _ram_delta_mb() -> int:
    return (_gpu_proc.memory_info().rss - _gpu_baseline_ram) // (1024 * 1024)


CACHE_DIR = "render_cache"


def _get_font_key(font) -> str | None:
    try:
        return f"{os.path.splitext(os.path.basename(font.path))[0]}_{font.size}"
    except AttributeError:
        return None


def _build_char_atlas(font, codes, dev):
    """Build {code: CUDA tensor} atlas for given font. Each char rendered in WHITE on transparent."""
    atlas = {}
    h_max = 0
    for c in codes:
        b = font.getbbox(chr(c))
        h_max = max(h_max, b[3])
    for c in codes:
        ch = chr(c)
        b = font.getbbox(ch)
        w = max(b[2] - b[0], 1)
        im = Image.new("RGBA", (w, h_max), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((0, 0), ch, fill=(255, 255, 255, 255), font=font)
        atlas[c] = torch.from_numpy(np.array(im)).to(dev, non_blocking=True).float() / 255.0
    return atlas


def _load_or_build_atlas(font, cache_name, codes, dev):
    """Load atlas from render_cache/ or build + cache it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    fkey = _get_font_key(font)
    if fkey:
        path = os.path.join(CACHE_DIR, f"{cache_name}_{fkey}.pt")
        if os.path.exists(path):
            try:
                cpu_data = torch.load(path, map_location="cpu")
                atlas = {}
                for k, v in cpu_data.items():
                    atlas[k] = v.to(dev, non_blocking=True)
                if all(c in atlas for c in codes):
                    return atlas
            except Exception:
                pass
    atlas = _build_char_atlas(font, codes, dev)
    if fkey:
        path = os.path.join(CACHE_DIR, f"{cache_name}_{fkey}.pt")
        torch.save({k: v.cpu() for k, v in atlas.items()}, path)
    return atlas



_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

from geometry import (PADDING, HEADER_H, STATS_PANEL_WIDTH, INFO_H, TIMER_HEIGHT,
    BG_COLOR, TILE_BG, TILE_TEXT_COLOR, TILE_BORDER_COLOR, NULL_COLOR,
    PANEL_BG, PANEL_ALPHA, TIMER_BG, ACCURATE_COLOR, INACCURATE_COLOR,
    WHITE, CYAN, GREEN, GRAY, LIGHT_GRAY,
    TILE_BORDER_WIDTH, TILE_BORDER_RADIUS_RATIO, BASE_SIZE,
    compute_canvas_dimensions, RenderOptions,
    get_font, render_number_texture,
    compute_tile_size, compute_font_size,
    compute_grid_position, compute_panel_rect, compute_secondary_bar_rect,
    round_canvas_height)


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
        ts = compute_tile_size(tile_size, quality)
        self.tile_size = ts
        self.font_size = compute_font_size(width, height, ts)
        self.pw = width * ts
        self.ph = height * ts

        puzzle_w = width * ts
        puzzle_h = height * ts
        self.canvas_w, self.canvas_h = compute_canvas_dimensions(
            width, height, ts, grid_only=self.opts.grid_only
        )
        if not self.opts.grid_only and min_canvas_h is not None:
            self.canvas_h = max(self.canvas_h, min_canvas_h)
            self.canvas_h = round_canvas_height(self.canvas_h)
        self.grid_x, self.grid_y = compute_grid_position(self.opts.grid_only)
        self.panel_x, self.panel_y, self.panel_w, self.panel_h = compute_panel_rect(
            self.grid_x, puzzle_w, self.canvas_w, self.grid_y, self.canvas_h
        )
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
                n = width * height
                num_batch = torch.zeros(n, ts, ts, 4, device=cuda_dev, dtype=torch.float32)
                for num in range(n):
                    pil = render_number_texture(num, ts, self.font_size)
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
            bx0, by0, bx1, by1 = compute_secondary_bar_rect(ts)
            bar_fill = torch.zeros(ts, ts, 1, device=cuda_dev, dtype=torch.float32)
            bar_border = torch.zeros(ts, ts, 1, device=cuda_dev, dtype=torch.float32)
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

    def upload_sprite_atlas(self, tile_sprites, grid_states, all_fringe_schemes, w, h):
        dev = self._device
        ts = self.tile_size

        base_items = []
        base_color_to_idx = {}
        for color, pil in tile_sprites.base_sprites.items():
            idx = len(base_items)
            arr = torch.from_numpy(np.array(pil)).to(dev).float() / 255.0
            base_items.append(arr)
            base_color_to_idx[color] = idx

        self._base_atlas = torch.stack(base_items) if base_items else None

        bar_items = []
        bar_color_to_idx = {}
        for color, pil in tile_sprites.bar_sprites.items():
            idx = len(bar_items)
            arr = torch.from_numpy(np.array(pil)).to(dev).float() / 255.0
            bar_items.append(arr)
            bar_color_to_idx[color] = idx

        self._bar_atlas = torch.stack(bar_items) if bar_items else None

        self._base_lookup = {}
        self._bar_lookup = {}
        for state_key, state in grid_states.items():
            if not isinstance(state_key, (int, float)):
                continue
            state_sig = id(state)
            num_to_base = torch.zeros(w * h + 1, device=dev, dtype=torch.int32)
            num_to_bar = torch.full((w * h + 1,), -1, device=dev, dtype=torch.int32)
            for num in range(w * h + 1):
                from replay_video import get_tile_colors
                main_bg, sec_bg = get_tile_colors(num, state, all_fringe_schemes, w)
                base_color = TILE_BG if main_bg is None else tuple(int(x) for x in main_bg)
                num_to_base[num] = base_color_to_idx.get(base_color, 0)
                if sec_bg is not None:
                    bar_color = tuple(int(x) for x in sec_bg)
                    if bar_color in bar_color_to_idx:
                        num_to_bar[num] = bar_color_to_idx[bar_color]
            self._base_lookup[state_sig] = num_to_base
            self._bar_lookup[state_sig] = num_to_bar

        log.info(f"  _upload_sprite_atlas: {len(base_items)} base sprites, {len(bar_items)} bar sprites, "
                 f"{len(self._base_lookup)} state lookups")

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
        batch_has_colors = any(p.get("colors_main") is not None for p in frame_params_list)

        # VRAM ceiling: stay well below total to avoid OOM / unified memory thrashing
        vram_ceiling = int(total_mem * 0.82)
        target_mem_fraction = 0.50
        target_used_mem = int(total_mem * target_mem_fraction)

        # Estimate per-frame cost from canvas size (conservative: 3x canvas bytes)
        per_frame_ema = max(1, ch * cw * 3 * 4 * 3)
        batch_size = 1
        prev_batch_n = 0

        self._stats["batch_size"] = batch_size
        self._stats["batch_mem_mb"] = 0
        self._stats["per_frame_ema_mb"] = per_frame_ema / (1024 * 1024)
        self._stats["target_used_mem_mb"] = target_used_mem // (1024 * 1024)
        self._stats["vram_ceiling_mb"] = vram_ceiling // (1024 * 1024)

        # Row‑chunked rendering: 2 rows per chunk for large puzzles
        chunk_rows = 2 if h > 6 else h

        frames = []
        batch_start = 0
        _overlay_ready = False

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
                reserved_mem = torch.cuda.memory_reserved(dev)

                # Enforce VRAM ceiling: flush cache if we're over
                if reserved_mem > vram_ceiling:
                    torch.cuda.empty_cache()
                    reserved_mem = torch.cuda.memory_reserved(dev)

                allocated_mem = torch.cuda.memory_allocated(dev)
                free_mem, _ = torch.cuda.mem_get_info(dev)

                # Safety margin
                reserve_margin = 384 * 1024 * 1024

                # Headroom based on reserved memory (accounts for cached allocations)
                headroom = vram_ceiling - reserved_mem - reserve_margin

                if headroom < 0:
                    headroom = 0

                if per_frame_ema > 0:
                    # Peak per-frame cost is ~1.5x marginal EMA (temp buffers during render)
                    peak_cost = per_frame_ema * 1.5
                    max_by_budget = max(1, int(headroom / peak_cost))
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
                         f"free={free_mem//(1024*1024)}MB, reserved={reserved_mem//(1024*1024)}MB, "
                         f"headroom={headroom//(1024*1024)}MB, ema={per_frame_ema//(1024*1024)}MB, "
                         f"batch_mem={self._stats['batch_mem_mb']}MB, t={_time_module.time()-_batch_t0:.2f}s, ram={_ram_delta_mb()}MB")

                # OOM guard — retry with cache flush before giving up
                if batch_n == 1 and headroom < per_frame_ema:
                    log.warning(f"  VRAM low: reserved={reserved_mem//(1024*1024)}MB, "
                                f"headroom={headroom//(1024*1024)}MB < "
                                f"ema={per_frame_ema//(1024*1024)}MB, "
                                f"flushing cache and retrying...")
                    torch.cuda.empty_cache()
                    r2 = torch.cuda.memory_reserved(dev)
                    headroom = vram_ceiling - r2 - reserve_margin
                    if headroom < 0:
                        headroom = 0
                    if headroom < per_frame_ema:
                        msg = (
                            f"GPU out of memory: cannot fit a single frame in available VRAM "
                            f"(needs ~{per_frame_ema // (1024*1024)}MB, "
                            f"only ~{headroom // (1024*1024)}MB available). "
                            f"Try disabling GPU acceleration, reducing quality, "
                            f"or using a smaller puzzle."
                        )
                        log.critical(msg)
                        raise RuntimeError(msg)
                    log.info(f"  Cache flush recovered {headroom//(1024*1024)}MB headroom, continuing")

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

                use_atlas = self._base_atlas is not None
                if use_atlas:
                    base_idx_np = np.zeros((batch_n, h, w), dtype=np.int32)
                    bar_idx_np = np.full((batch_n, h, w), -1, dtype=np.int32)
                    for i, p in enumerate(batch_params):
                        state_sig = id(p["grid_state"])
                        mats_p = np.asarray(p["matrix"])
                        if state_sig in self._base_lookup:
                            base_lookup_np = self._base_lookup[state_sig].cpu().numpy()
                            bar_lookup_np = self._bar_lookup[state_sig].cpu().numpy()
                            base_idx_np[i] = base_lookup_np[mats_p]
                            bar_idx_np[i] = bar_lookup_np[mats_p]
                    base_idx = torch.from_numpy(base_idx_np).to(dev, non_blocking=True)
                    bar_idx = torch.from_numpy(bar_idx_np).to(dev, non_blocking=True)
                else:
                    if batch_has_colors:
                        main_np = np.full((batch_n, h, w, 3), TILE_BG, dtype=np.float32)
                        sec_np = np.zeros((batch_n, h, w, 3), dtype=np.float32)
                        has_sec_np = np.zeros((batch_n, h, w), dtype=np.float32)
                        for i, p in enumerate(batch_params):
                            cm = p.get("colors_main")
                            if cm is not None:
                                main_np[i] = cm.reshape(h, w, 3)
                                cs = p.get("colors_sec")
                                if cs is not None:
                                    sec_np[i] = cs.reshape(h, w, 3)
                                csm = p.get("colors_sec_mask")
                                if csm is not None:
                                    has_sec_np[i] = csm.reshape(h, w).astype(np.float32)
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

                # ── Tile rendering (atlas path: batched full render, all frames in parallel) ──
                if use_atlas:
                    for row_start in range(0, h, chunk_rows):
                        row_end = min(row_start + chunk_rows, h)
                        n_rows = row_end - row_start

                        base_chunk = self._base_atlas[base_idx[:, row_start:row_end, :]]
                        tile_rgb = base_chunk[..., :3]

                        if not self.opts.no_numbers and num_tex is not None:
                            nums = num_tex[mats[:, row_start:row_end, :]]
                            tile_rgb = nums[..., :3] * nums[..., 3:] + tile_rgb * (1 - nums[..., 3:])

                        if self._bar_atlas is not None:
                            bi = bar_idx[:, row_start:row_end, :]
                            bar_mask = (bi >= 0).float().view(batch_n, n_rows, w, 1, 1, 1)
                            bar_safe = bi.clamp(min=0)
                            bar_s = self._bar_atlas[bar_safe]
                            tile_rgb = tile_rgb * (1 - bar_s[..., 3:] * bar_mask) + bar_s[..., :3] * bar_s[..., 3:] * bar_mask

                        tile_chunk = tile_rgb.permute(0, 1, 3, 2, 4, 5).reshape(batch_n, n_rows * ts, pw, 3)
                        canvas_y = gy + row_start * ts
                        canvas[:, canvas_y:canvas_y + n_rows * ts, gx:gx + pw] = tile_chunk
                        del base_chunk, tile_rgb, tile_chunk
                else:
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

                # ── Overlays (GPU font atlases — no PIL in render loop) ──
                if overlay_render_data is not None:
                    if not _overlay_ready:
                        _layout = overlay_render_data["static_layout"]
                        _data_font = _layout["data_font"]
                        _layout_px = _layout["px"]
                        _layout_inner_w = _layout["inner_w"]
                        _row_h = _layout["row_h"]
                        _y_predicted = _layout["y_predicted"]
                        _y_md_cur = _layout["y_md_cur"]
                        _y_mmd_cur = _layout["y_mmd_cur"]
                        _stage_y_positions = _layout.get("stage_y_positions", [])
                        _stage_raw_lines = _layout.get("stage_raw_lines", [])
                        _stage_w1 = _layout.get("stage_w1", 0)
                        _stage_w2 = _layout.get("stage_w2", 0)
                        _stage_w3 = _layout.get("stage_w3", 0)
                        _stage_w4 = _layout.get("stage_w4", 0)
                        _gs_lf = _layout.get("gs_lf")
                        _panel_w = overlay_render_data.get("panel_w_val", self.panel_w)

                        # Font atlases (PIL one-time, cached to disk)
                        _data_atlas = _load_or_build_atlas(
                            _data_font, "datafont", list(range(32, 127)), dev)
                        _acc_font = _layout.get("acc_font") or get_font(16, mono=True)
                        _acc_atlas = _load_or_build_atlas(
                            _acc_font, "accfont", list(range(32, 127)), dev)
                        _gs_atlas = _load_or_build_atlas(
                            _gs_lf, "gs_lf", list(range(32, 127)), dev) if _gs_lf else {}
                        _header_font = _layout.get("header_font")
                        _header_atlas = _load_or_build_atlas(
                            _header_font, "headerfont", list(range(32, 127)), dev) if _header_font else _data_atlas
                        _gs_hf = _layout.get("gs_header_font")
                        _gs_hf_atlas = _load_or_build_atlas(
                            _gs_hf, "gshfont", list(range(32, 127)), dev) if _gs_hf else _acc_atlas
                        _timer_font = get_font(36, bold=True, mono=True)
                        _timer_atlas = _load_or_build_atlas(
                            _timer_font, "timerfont",
                            [32, 40, 41, 46, 47] + list(range(48, 58)) + [58], dev)

                        # ── Static base: GPU composition from font atlases ──
                        _fp0 = frame_params_list[0]
                        _stats0 = _fp0.get("stats_data", {})
                        _is_acc = _fp0.get("is_movetimes_accurate", False)
                        _sb_h = _layout.get("total_h", 400)
                        _sb = torch.zeros(_sb_h, _panel_w, 4, device=dev, dtype=torch.float32)
                        _white_rgb = torch.tensor([1., 1., 1.], device=dev)
                        _cyan_rgb_sb = torch.tensor(CYAN, device=dev, dtype=torch.float32) / 255.0
                        _acc_rgb = torch.tensor(ACCURATE_COLOR, device=dev, dtype=torch.float32) / 255.0
                        _inacc_rgb = torch.tensor(INACCURATE_COLOR, device=dev, dtype=torch.float32) / 255.0

                        def _pt(cv, text, atl, xx, yy, col):
                            if not text: return
                            ts = [atl.get(ord(c), atl[32]) for c in text]
                            ti = torch.cat(ts, dim=1)
                            th, tw = ti.shape[:2]
                            if xx + tw > cv.shape[1]: tw = max(0, cv.shape[1] - xx)
                            if yy + th > cv.shape[0]: th = max(0, cv.shape[0] - yy)
                            if th <= 0 or tw <= 0: return
                            sa = ti[:th, :tw, 3:4]
                            dst = cv[yy:yy+th, xx:xx+tw]
                            dst[:, :, :3] = col.view(1,1,3) * sa + dst[:, :, :3] * (1 - sa)
                            dst[:, :, 3] = dst[:, :, 3] + sa.squeeze(-1) * (1 - dst[:, :, 3])

                        def _pv(cv, val, atl, rxx, yy, col):
                            if not val: return
                            tw = sum(atl.get(ord(c), atl[32]).shape[1] for c in val)
                            _pt(cv, val, atl, rxx - tw, yy, col)

                        def _pl(cv, lbl, val, atl, xx, yy, iw):
                            _pt(cv, lbl, atl, xx, yy, _white_rgb)
                            _pv(cv, val, atl, xx + iw, yy, _white_rgb)

                        _acc_h = _acc_atlas[32].shape[0]
                        _gs_hf_h = _gs_hf_atlas[32].shape[0]

                        # "Stats" header (24px bold)
                        _pt(_sb, "Stats", _header_atlas, _layout_px, 10, _cyan_rgb_sb)
                        # Static lines (label + value, WHITE)
                        _pl(_sb, "Time (total): ", _stats0.get("time_all","0.000"), _data_atlas,
                            _layout_px, _y_predicted - 5*_row_h, _layout_inner_w)
                        _pl(_sb, "Moves (total): ", _stats0.get("moves_all","0"), _data_atlas,
                            _layout_px, _y_predicted - 4*_row_h, _layout_inner_w)
                        _pl(_sb, "TPS (total): ", _stats0.get("tps_all","0.000"), _data_atlas,
                            _layout_px, _y_predicted - 3*_row_h, _layout_inner_w)
                        _pl(_sb, "Cubic est: ", _stats0.get("cubic_estimate","---"), _data_atlas,
                            _layout_px, _y_predicted - 2*_row_h, _layout_inner_w)
                        _pl(_sb, "Playback speed: ", _stats0.get("speed_playback","1.00×"), _data_atlas,
                            _layout_px, _y_predicted - 1*_row_h, _layout_inner_w)
                        # Predicted moves label (CYAN)
                        _pt(_sb, "Predicted moves: ", _data_atlas, _layout_px, _y_predicted, _cyan_rgb_sb)
                        # MD total (WHITE)
                        _pl(_sb, "MD (total): ", _stats0.get("md_all","0"), _data_atlas,
                            _layout_px, _y_predicted + _row_h, _layout_inner_w)
                        # MD current label (CYAN)
                        _pt(_sb, "MD (current): ", _data_atlas, _layout_px, _y_md_cur, _cyan_rgb_sb)
                        # M/MD total (WHITE)
                        _pl(_sb, "M/MD (total): ", _stats0.get("mmd_all","0.000"), _data_atlas,
                            _layout_px, _y_md_cur + _row_h, _layout_inner_w)
                        # M/MD current label (CYAN)
                        _pt(_sb, "M/MD (current): ", _data_atlas, _layout_px, _y_mmd_cur, _cyan_rgb_sb)
                        # Accuracy text
                        _acc_yy = _y_mmd_cur + _row_h + 4
                        _pt(_sb, "Movetimes accurate" if _is_acc else "NOT movetimes accurate",
                            _acc_atlas, _layout_px, _acc_yy, _acc_rgb if _is_acc else _inacc_rgb)
                        # Grid stages (WHITE, static only)
                        if _stage_raw_lines:
                            _pt(_sb, "Grid stages", _gs_hf_atlas, _layout_px,
                                _acc_yy + _acc_h + 6, _cyan_rgb_sb)
                            for _i in range(len(_stage_raw_lines)):
                                _cum_s, _split_s, _mvtps_s, _label = _stage_raw_lines[_i]
                                if '.' in _cum_s:
                                    _gl = f"{_cum_s:>{_stage_w1}} | {_split_s:>{_stage_w2}} {_mvtps_s:<{_stage_w3}} | {_label:<{_stage_w4}}"
                                else:
                                    _gl = f"{_cum_s:>{_stage_w1}} | {_split_s:<{_stage_w2}}  | {_label:<{_stage_w4}}"
                                _pt(_sb, _gl, _gs_atlas, _layout_px, _stage_y_positions[_i], _white_rgb)

                        _static_base_gpu = _sb
                        _sb_h, _sb_w = _static_base_gpu.shape[:2]
                        _cyan_rgb = torch.tensor(CYAN, device=dev, dtype=torch.float32) / 255.0
                        _overlay_ready = True

                    # Blend static_base onto canvas (batch broadcast, once per batch)
                    _dh = min(_sb_h, ch - py)
                    _dw = min(_sb_w, cw - px)
                    if _dh > 0 and _dw > 0:
                        canvas[:, py:py + _dh, px:px + _dw] = (
                            _static_base_gpu[:_dh, :_dw, :3].unsqueeze(0).expand(batch_n, -1, -1, -1).contiguous() *
                            _static_base_gpu[:_dh, :_dw, 3:4].unsqueeze(0).expand(batch_n, -1, -1, -1).contiguous() +
                            canvas[:, py:py + _dh, px:px + _dw] * (1 - _static_base_gpu[:_dh, :_dw, 3:4].unsqueeze(0).expand(batch_n, -1, -1, -1).contiguous())
                        )

                    def _blend_cyan(canvas_i, text, atlas, x, y, cw_max, ch_max, center_x=False, center_y=False):
                        """Blend text from atlas in CYAN onto canvas[i] at (x,y), optionally centered."""
                        if not text:
                            return
                        tensors = [atlas.get(ord(c), atlas[32]) for c in text]
                        ti = torch.cat(tensors, dim=1)
                        th, tw = ti.shape[:2]
                        if center_x:
                            x = x + ((cw_max - x) - tw) // 2
                        if center_y:
                            y = y + ((ch_max - y) - th) // 2
                        th_c = min(th, ch - y)
                        tw_c = min(tw, cw - x)
                        if th_c > 0 and tw_c > 0:
                            sa = ti[:th_c, :tw_c, 3:4]
                            dst = canvas_i[y:y + th_c, x:x + tw_c, :]
                            dst[:, :, :3] = _cyan_rgb.view(1, 1, 3) * sa + dst[:, :, :3] * (1 - sa)

                    for _i in range(batch_n):
                        _p = frame_params_list[batch_start + _i]

                        # Timer (GPU font atlas, no PIL)
                        _blend_cyan(canvas[_i], _p["timer_text"], _timer_atlas,
                                    tx1, ty1, tx2, ty2, center_x=True, center_y=True)

                        _sd = _p.get("stats_data")
                        if _sd is None:
                            continue

                        # Dynamic values (data font atlas, right-aligned, CYAN)
                        for _key, _y_val in [("predicted_moves", _y_predicted), ("md_cur", _y_md_cur), ("mmd_cur", _y_mmd_cur)]:
                            _text = _sd.get(_key, "")
                            if not _text:
                                continue
                            _tw = sum(_data_atlas.get(ord(c), _data_atlas[32]).shape[1] for c in _text)
                            _blend_cyan(canvas[_i], _text, _data_atlas,
                                        px + _layout_px + _layout_inner_w - _tw, py + _y_val,
                                        cw, ch)

                        # Stage highlight: CYAN from gs_lf atlas (same positioning as white static base)
                        _cur_stage = _sd.get("grid_current", 0)
                        if _stage_raw_lines and _cur_stage < len(_stage_y_positions):
                            _cums_s, _splits_s, _mvtpss_s, _label_s = _stage_raw_lines[_cur_stage]
                            if '.' in _cums_s:
                                _line_s = f"{_cums_s:>{_stage_w1}} | {_splits_s:>{_stage_w2}} {_mvtpss_s:<{_stage_w3}} | {_label_s:<{_stage_w4}}"
                            else:
                                _line_s = f"{_cums_s:>{_stage_w1}} | {_splits_s:<{_stage_w2}}  | {_label_s:<{_stage_w4}}"
                            _blend_cyan(canvas[_i], _line_s, _gs_atlas,
                                        px + _layout_px, py + _stage_y_positions[_cur_stage],
                                        cw, ch)
                else:
                    first_stats_arr = frame_params_list[batch_start].get("stats_arr")
                    if first_stats_arr is not None:
                        s_h, s_w = first_stats_arr.shape[:2]
                        snp = np.stack([frame_params_list[batch_start + i]["stats_arr"] for i in range(batch_n)], axis=0)
                        stats_t = torch.from_numpy(snp).to(dev, non_blocking=True).float() / 255.0
                        dh = min(s_h, ch - py)
                        dw = min(s_w, cw - px)
                        if dh > 0 and dw > 0:
                            canvas[:, py:py + dh, px:px + dw] = (
                                stats_t[:, :dh, :dw, :3] * stats_t[:, :dh, :dw, 3:] +
                                canvas[:, py:py + dh, px:px + dw] * (1 - stats_t[:, :dh, :dw, 3:])
                            )
                    for _i in range(batch_n):
                        ta = frame_params_list[batch_start + _i].get("timer_arr")
                        if ta is not None:
                            tt = torch.from_numpy(ta).to(dev, non_blocking=True).float() / 255.0
                            dx = max(tx1, tx1 + ((tx2 - tx1) - tt.shape[1]) // 2)
                            dy = max(ty1, ty1 + ((ty2 - ty1) - tt.shape[0]) // 2)
                            self._blend_rgba_inplace(canvas[_i], tt, dx, dy)

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

                # ── Free batch tensors + flush CUDA cache ──
                del canvas, mats
                if use_atlas:
                    del base_idx, bar_idx
                elif batch_has_colors:
                    del cols, sec, has_sec
                torch.cuda.synchronize(dev)
                torch.cuda.empty_cache()
                peak_reserved = torch.cuda.max_memory_reserved(dev)
                marginal = peak_reserved - reserved_before_render
                if marginal > 0 and batch_n >= 1:
                        actual_ppf = marginal / batch_n
                        old_ema = per_frame_ema
                        # EMA with decay – lets batch sizes grow when per-frame cost drops
                        alpha = 0.3
                        per_frame_ema = per_frame_ema * (1 - alpha) + max(actual_ppf, actual_ppf * 1.10) * alpha
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
                "_base_atlas", "_bar_atlas",
            ]
            for attr in attrs:
                t = getattr(self, attr, None)
                if t is not None:
                    del t
                    setattr(self, attr, None)
            self._base_lookup = {}
            self._bar_lookup = {}
            torch.cuda.empty_cache()
            log.info("GPURenderer.cleanup: CUDA tensors freed")

    def __del__(self):
        if self._device is not None:
            try:
                self.cleanup()
            except Exception:
                pass

    # _cpu_fallback removed — GPU errors now raise RuntimeError instead of silently falling back
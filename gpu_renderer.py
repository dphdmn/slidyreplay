from __future__ import annotations
import math
import json
import os
import threading
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
    TILE_BORDER_WIDTH, TILE_BORDER_RADIUS_RATIO,
    compute_canvas_dimensions, RenderOptions,
    get_font, render_number_texture, compute_font_size,
    compute_grid_position, compute_panel_rect, compute_secondary_bar_rect,
    should_draw_tile_border, should_draw_secondary_border_rect,
    round_canvas_height)


class GPURenderer:
    def __init__(self, width: int, height: int, tile_size: int,
                 pad: int = None, header_h: int = None, panel_w: int = None,
                 canvas_w: int = None, canvas_h: int = None,
                 opts: Optional[RenderOptions] = None):
        """
        GPU renderer with automatic optimal batching.

        Self‑calibrates on the first frame, then uses as large a batch as the
        available free memory allows, while respecting a soft 50% VRAM ceiling.
        No knobs required.
        """
        self.opts = opts or RenderOptions()
        self.w = width
        self.h = height
        self.pad = pad if pad is not None else PADDING
        self.header_h = header_h if header_h is not None else HEADER_H
        self.layout_panel_w = panel_w if panel_w is not None else STATS_PANEL_WIDTH
        self.tile_size = tile_size
        self.font_size = compute_font_size(width, height, tile_size)
        self.pw = width * tile_size
        self.ph = height * tile_size

        puzzle_w = width * tile_size
        puzzle_h = height * tile_size
        if canvas_w is not None and canvas_h is not None:
            self.canvas_w, self.canvas_h = canvas_w, canvas_h
        else:
            self.canvas_w, self.canvas_h = compute_canvas_dimensions(
                width, height, tile_size, grid_only=self.opts.grid_only,
                pad=self.pad, header_h=self.header_h, panel_w=self.layout_panel_w,
            )
        self.grid_x, self.grid_y = compute_grid_position(
            self.opts.grid_only, pad=self.pad, header_h=self.header_h,
            canvas_h=self.canvas_h, puzzle_h=self.ph,
            no_header=self.opts.no_header,
            align_top=self.opts.adjust_height,
        )
        self.panel_x, self.panel_y, self.panel_w, self.panel_h = compute_panel_rect(
            self.grid_x, puzzle_w, self.canvas_w, self.grid_y, self.canvas_h,
            pad=self.pad, panel_y=(0 if (self.opts.grid_only or self.opts.no_header) else self.header_h),
        )
        self.timer_bbox = (0, 0, 0, 0) if (self.opts.grid_only or self.opts.no_header) else (0, 0, self.canvas_w, self.header_h)

        self._init_success = False
        self._device = None
        self._tile_mask = None
        self._timer_bg = None
        self._panel_bg = None
        self._panel_bg_rgb = None
        self._panel_bg_a = None
        self._static_stats_bg_rgb = None
        self._static_stats_bg_a = None
        self._overlay_text_positions = None
        self._composite_atlas = None
        self._composite_lookup = {}
        self._composite_lookup_cpu = {}

        if _HAS_TORCH:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._stats = {
            "gpu_name": "",
            "total_mem_mb": 0,
        }
        self._batch_counter = 0
        log.info(f"GPURenderer.__init__: {width}x{height}, tile_size={tile_size}, canvas={self.canvas_w}x{self.canvas_h}, device={self._device}")
        if self._stats["total_mem_mb"]:
            log.info(f"  GPU: {self._stats['gpu_name']}, total_mem={self._stats['total_mem_mb']}MB")
        if self._device is not None and self._device.type == "cuda":
            total = torch.cuda.get_device_properties(self._device).total_memory
            self._stats["gpu_name"] = torch.cuda.get_device_name(self._device)
            self._stats["total_mem_mb"] = total // (1024 * 1024)

        if self._device is not None and self._device.type == "cuda":
            cuda_dev = self._device

            # Tile mask (all ones) and its complement
            mask = torch.ones(tile_size, tile_size, 1, device=cuda_dev, dtype=torch.float32)
            self._tile_mask = mask
            self._tile_mask_inv = 1.0 - mask

            # Tile border mask: 1 px border
            border = torch.zeros(tile_size, tile_size, 1, device=cuda_dev, dtype=torch.float32)
            if should_draw_tile_border(tile_size) and not self.opts.no_border:
                border[0, :] = 1
                border[:, 0] = 1
            self._border_mask = border
            self._border_mask_inv = 1.0 - border

            # Secondary colour bar masks
            bx0, by0, bx1, by1 = compute_secondary_bar_rect(tile_size, font_size=self.font_size)
            bar_fill = torch.zeros(tile_size, tile_size, 1, device=cuda_dev, dtype=torch.float32)
            bar_border = torch.zeros(tile_size, tile_size, 1, device=cuda_dev, dtype=torch.float32)
            bar_fill[by0:by1, bx0:bx1] = 1.0
            if should_draw_secondary_border_rect(tile_size, (bx0, by0, bx1, by1)) and not self.opts.no_secondary_border:
                bar_border[by0, bx0:bx1] = 1.0
                bar_border[by1 - 1, bx0:bx1] = 1.0
                bar_border[by0:by1, bx0] = 1.0
                bar_border[by0:by1, bx1 - 1] = 1.0
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

    def upload_composite_atlas(self, composite_images, composite_lookup):
        """Upload pre-rendered composite tiles as uint8 CUDA tensor."""
        dev = self._device
        ts = self.tile_size
        n = len(composite_images)
        arr = np.zeros((n, ts, ts, 4), dtype=np.uint8)
        for i, pil in enumerate(composite_images):
            arr[i] = np.array(pil)
        self._composite_atlas = torch.from_numpy(arr).to(dev)
        self._composite_lookup = {}
        self._composite_lookup_cpu = {}
        for state_sig, lookup_list in composite_lookup.items():
            self._composite_lookup[state_sig] = torch.tensor(lookup_list, device=dev, dtype=torch.int32)
        log.info(f"  upload_composite_atlas: {n} entries, "
                 f"size={n*ts*ts*4//(1024*1024)}MB (uint8)")

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
        timer_bg = self._timer_bg
        panel_bg_rgb = self._panel_bg_rgb
        panel_bg_a = self._panel_bg_a

        c_bg = torch.tensor(BG_COLOR, device=dev, dtype=torch.float32) / 255.0

        # Row‑chunked rendering: 2 rows per chunk for large puzzles
        chunk_rows = 2 if h > 6 else h

        frames = []
        batch_start = 0
        _overlay_ready = False

        log.info(f"render_frames: {n} frames, canvas={cw}x{ch}, chunk_rows={chunk_rows}")

        # Pre-allocate pinned memory buffers + CUDA stream for async GPU→CPU download (3B-b)
        _N_BUFS = 2
        _pinned_bufs = [
            torch.empty(1, ch, cw, 3, dtype=torch.uint8, pin_memory=True)
            for _ in range(_N_BUFS)
        ]
        _buf_free_events = [threading.Event() for _ in range(_N_BUFS)]
        for _e in _buf_free_events:
            _e.set()
        _dl_stream = torch.cuda.Stream()
        _dl_buf_idx = 0
        _prev_buf_idx = None
        _dl_prev_event = None
        _dl_prev_uint8 = None
        _dl_prev_start = 0

        # 666 delta rendering state
        _prev_canvas = None

        # Profiling accumulators (seconds)
        _prof_setup = 0.0
        _prof_delta_patch = 0.0
        _prof_full_render = 0.0
        _prof_overlays = 0.0
        _prof_uint8 = 0.0
        _prof_dl_handler = 0.0
        _prof_delta_count = 0
        _prof_full_count = 0
        _tick = _time_module.perf_counter

        with torch.inference_mode():
            _batch_t0 = _time_module.time()
            while batch_start < n:
                _pt_setup = _tick()
                if cancel_check and cancel_check():
                    raise CancelError()

                batch_end = batch_start + 1
                batch_n = 1

                self._batch_counter += 1
                self._stats["batch_idx"] = self._batch_counter

                # ── Upload batch data ──
                p = frame_params_list[batch_start]
                p_opts = p.get("opts")
                if not isinstance(p_opts, RenderOptions):
                    raise TypeError(f"batch_params['opts'] must be RenderOptions, got {type(p_opts)}")
                if p_opts != self.opts:
                    raise ValueError(
                        f"Item opts={p_opts} != renderer opts={self.opts}. "
                        "All items must use identical RenderOptions."
                    )
                mats = torch.from_numpy(np.asarray(p["matrix"])[None, :, :]).to(dev, non_blocking=True)

                use_composite = hasattr(self, '_composite_atlas') and self._composite_atlas is not None
                if not use_composite:
                    raise RuntimeError("upload_composite_atlas must be called before render_frames")

                # Check if this frame can use delta path
                _ct = p.get("changed_tiles")
                _can_delta = use_composite and _prev_canvas is not None and _ct is not None
                _delta_threshold = h * w // 6
                _will_delta = _can_delta and len(_ct) <= _delta_threshold

                composite_idx = None
                if not _will_delta:
                    state_sig = id(p["grid_state"])
                    if state_sig not in self._composite_lookup_cpu:
                        self._composite_lookup_cpu[state_sig] = self._composite_lookup[state_sig].cpu().numpy()
                    lookup_np = self._composite_lookup_cpu[state_sig]
                    mats_p = np.asarray(p["matrix"])
                    composite_idx_np = lookup_np[mats_p][None, :, :]
                    composite_idx = torch.from_numpy(composite_idx_np).to(dev, non_blocking=True)

                _prof_setup = _prof_setup + _tick() - _pt_setup

                # ── Canvas ──
                _pt0 = _tick()
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

                # ── Delta patch (copy prev canvas, update only changed tiles) ──
                _delta_applied = False
                if _will_delta:
                    canvas[0].copy_(_prev_canvas)
                    if len(_ct) > 0:
                        state_sig = id(p["grid_state"])
                        if state_sig not in self._composite_lookup_cpu:
                            self._composite_lookup_cpu[state_sig] = self._composite_lookup[state_sig].cpu().numpy()
                        lookup_np = self._composite_lookup_cpu[state_sig]
                        mats_p = np.asarray(p["matrix"])
                        ti_np = lookup_np[mats_p[_ct[:, 0], _ct[:, 1]]]
                        ti_t = torch.from_numpy(ti_np).to(dev)
                        _t_batch = self._composite_atlas[ti_t, :, :, :3]  # uint8
                        for _k in range(len(_ct)):
                            _r, _c = _ct[_k]
                            _sy, _sx = gy + _r * ts, gx + _c * ts
                            canvas[0, _sy:_sy + ts, _sx:_sx + ts] = _t_batch[_k].float() / 255.0
                        del _t_batch, ti_t
                    _delta_applied = True

                # ── Tile rendering (skipped when delta applied) ──
                if not _delta_applied:
                    for row_start in range(0, h, chunk_rows):
                        row_end = min(row_start + chunk_rows, h)
                        n_rows = row_end - row_start
                        tile_chunk = self._composite_atlas[composite_idx[:, row_start:row_end, :]]
                        tile_rgb = tile_chunk[..., :3].float() / 255.0
                        tile_chunk_out = tile_rgb.permute(0, 1, 3, 2, 4, 5).reshape(batch_n, n_rows * ts, pw, 3)
                        canvas_y = gy + row_start * ts
                        canvas[:, canvas_y:canvas_y + n_rows * ts, gx:gx + pw] = tile_chunk_out
                        del tile_chunk, tile_rgb, tile_chunk_out

                # ── Profile: tiles done ──
                if _delta_applied:
                    _prof_delta_patch += _tick() - _pt0
                    _prof_delta_count += 1
                else:
                    _prof_full_render += _tick() - _pt0
                    _prof_full_count += 1
                if use_composite:
                    _prev_canvas = canvas[0].clone()
                _pt0 = _tick()

                # ── Overlays (GPU font atlases — no PIL in render loop) ──
                if overlay_render_data is not None:
                    if not _overlay_ready:
                        _layout = overlay_render_data.get("static_layout")

                        # Timer font atlas — needed regardless of no_details/no_header
                        _timer_font = get_font(max(12, self.header_h - 12), bold=True, mono=True)
                        _timer_atlas = _load_or_build_atlas(
                            _timer_font, "timerfont",
                            [32, 40, 41, 46, 47] + list(range(48, 58)) + [58], dev)
                        _cyan_rgb = torch.tensor(CYAN, device=dev, dtype=torch.float32) / 255.0

                        if _layout is not None and not self.opts.no_details:
                            _data_font = _layout["data_font"]
                            _layout_px = _layout["px"]
                            _layout_inner_w = _layout["inner_w"]
                            _row_h = _layout["row_h"]
                            _y_predicted = 0
                            _y_md_cur = 0
                            _y_mmd_cur = 0
                            _stage_y_positions = _layout.get("stage_y_positions", [])
                            _gs_x = _layout.get("gs_x", _layout_px)
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

                            def _pl_white(label, value, atl):
                                nonlocal _y_sb
                                _pt(_sb, label, atl, _layout_px, _y_sb, _white_rgb)
                                if value:
                                    _pv(_sb, value, atl, _layout_px + _layout_inner_w, _y_sb, _white_rgb)
                                _y_sb += _row_h

                            _gs_hf_h = _gs_hf_atlas[32].shape[0]
                            _y_sb = 10

                            # "Stats" header
                            _pt(_sb, "Stats", _header_atlas, _layout_px, _y_sb, _cyan_rgb_sb)
                            _y_sb += _header_atlas[32].shape[0] + 14

                            # ── Render Info ──
                            _pt(_sb, "Render Info", _gs_hf_atlas, _layout_px, _y_sb, _cyan_rgb_sb)
                            _y_sb += _gs_hf_h + 8

                            _render_info_data = [
                                ("Quality: ", f"{_layout.get('quality', 1080)}p"),
                                ("Canvas: ", _fp0.get("canvas_size", "")),
                                ("Render: ", "GPU"),
                                ("Codec: ", _fp0.get("codec_name", "")),
                                ("Preset: ", _fp0.get("resolved_preset", "")),
                                ("Tile: ", f"{self.tile_size}px" if self.tile_size else ""),
                                ("FPS: ", str(_fp0.get("fps", 60))),
                                ("Compression: ", str(_fp0.get("compression", 18))),
                                ("Frames: ", str(_fp0.get("total_frames", 0))),
                                ("Unique: ", str(_fp0.get("unique_frames", 0))),
                                ("Speed: ", _stats0.get("speed_playback", "1.00x")),
                            ]
                            for _lbl, _val in _render_info_data:
                                _pl_white(_lbl, _val, _data_atlas)
                            _y_sb += 6

                            # ── Puzzle Info ──
                            _pt(_sb, "Puzzle Info", _gs_hf_atlas, _layout_px, _y_sb, _cyan_rgb_sb)
                            _y_sb += _gs_hf_h + 8

                            _puzzle_info_data = [
                                ("Puzzle: ", _fp0.get("puzzle_size", "")),
                                ("Time (total): ", _stats0.get("time_all","0.000")),
                                ("Moves (total): ", _stats0.get("moves_all","0")),
                                ("TPS (total): ", _stats0.get("tps_all","0.000")),
                                ("Cubic est: ", _stats0.get("cubic_estimate","---")),
                                ("MD (total): ", _stats0.get("md_all","0")),
                                ("M/MD (total): ", _stats0.get("mmd_all","0.000")),
                            ]
                            for _lbl, _val in _puzzle_info_data:
                                _pl_white(_lbl, _val, _data_atlas)
                            _acc_text = "Movetimes accurate" if _is_acc else "NOT movetimes accurate"
                            _pt(_sb, _acc_text, _acc_atlas, _layout_px, _y_sb,
                                _acc_rgb if _is_acc else _inacc_rgb)
                            _y_sb += _acc_atlas[32].shape[0] + 6
                            _y_sb += 6

                            # ── Grid stages ──
                            if _stage_raw_lines:
                                _pt(_sb, "Grid stages", _gs_hf_atlas, _layout_px, _y_sb, _cyan_rgb_sb)
                                _y_sb += _gs_hf_h + 14
                                for _i in range(len(_stage_raw_lines)):
                                    _cum_s, _split_s, _mvtps_s, _label = _stage_raw_lines[_i]
                                    if '.' in _cum_s:
                                        _gl = f"{_cum_s:>{_stage_w1}} | {_split_s:>{_stage_w2}} {_mvtps_s:<{_stage_w3}} | {_label:<{_stage_w4}}"
                                    else:
                                        _gl = f"{_cum_s:>{_stage_w1}} | {_split_s:<{_stage_w2}}  | {_label:<{_stage_w4}}"
                                    _pt(_sb, _gl, _gs_atlas, _gs_x, _stage_y_positions[_i], _white_rgb)

                            # ── Cycles (static white) ──
                            _cycles_lines = _layout.get("cycles_lines")
                            _cycles_y_val = _layout.get("cycles_y")
                            if _cycles_lines and _cycles_y_val is not None:
                                _cyc_line_h = _gs_atlas[32].shape[0] + 8
                                for _li, _line in enumerate(_cycles_lines):
                                    _pt(_sb, _line, _gs_atlas, _gs_x, _cycles_y_val + _li * _cyc_line_h, _white_rgb)

                            _static_base_gpu = _sb
                            _sb_h, _sb_w = _static_base_gpu.shape[:2]
                            _static_base_pil = overlay_render_data.get("static_base")
                            if _static_base_pil is not None:
                                _static_base_arr = np.array(_static_base_pil.convert("RGBA"), dtype=np.uint8)
                                _static_base_gpu = torch.from_numpy(_static_base_arr).to(dev, non_blocking=True).float() / 255.0
                                _sb_h, _sb_w = _static_base_gpu.shape[:2]
                            _stage_highlights = []
                            if _stage_raw_lines and _gs_lf:
                                for _cum_s, _split_s, _mvtps_s, _label in _stage_raw_lines:
                                    if '.' in _cum_s:
                                        _gl = f"{_cum_s:>{_stage_w1}} | {_split_s:>{_stage_w2}} {_mvtps_s:<{_stage_w3}} | {_label:<{_stage_w4}}"
                                    else:
                                        _gl = f"{_cum_s:>{_stage_w1}} | {_split_s:<{_stage_w2}}  | {_label:<{_stage_w4}}"
                                    _b = _gs_lf.getbbox(_gl)
                                    _surf = Image.new("RGBA", (max(_b[2], 1), max(_b[3], 1)), (0, 0, 0, 0))
                                    ImageDraw.Draw(_surf).text((0, 0), _gl, fill=(*CYAN, 255), font=_gs_lf)
                                    _stage_highlights.append(torch.from_numpy(np.array(_surf)).to(dev, non_blocking=True).float() / 255.0)
                        else:
                            _static_base_gpu = None
                            _stage_raw_lines = []
                            _gs_lf = None
                            _gs_x = 0
                            _stage_y_positions = []
                            _layout_px = None
                        _overlay_ready = True

                    # Blend static_base onto canvas (batch broadcast) — only when no_details
                    if not self.opts.no_details and _static_base_gpu is not None:
                        _dh = min(_sb_h, ch - py)
                        _dw = min(_sb_w, cw - px)
                        if _dh > 0 and _dw > 0:
                            _base_rgb = _static_base_gpu[:_dh, :_dw, :3]
                            _base_a = _static_base_gpu[:_dh, :_dw, 3:4]
                            canvas[0, py:py + _dh, px:px + _dw] = _base_rgb * _base_a + canvas[0, py:py + _dh, px:px + _dw] * (1 - _base_a)

                    _overlay_text_cache = {}
                    def _compose_text(text, atlas):
                        """Return (tensor, width) for text, cached."""
                        cached = _overlay_text_cache.get(text)
                        if cached is not None and cached[2] is atlas:
                            return cached[0], cached[1]
                        tensors = [atlas.get(ord(c), atlas[32]) for c in text]
                        ti = torch.cat(tensors, dim=1)
                        tw = ti.shape[1]
                        _overlay_text_cache[text] = (ti, tw, atlas)
                        return ti, tw

                    def _blend_cyan(canvas_i, text, atlas, x, y, cw_max, ch_max, center_x=False, center_y=False):
                        """Blend text from atlas in CYAN onto canvas[i] at (x,y), optionally centered."""
                        if not text:
                            return
                        ti, tw = _compose_text(text, atlas)
                        th = ti.shape[0]
                        if center_x:
                            x = x + ((cw_max - x) - tw) // 2
                        if center_y:
                            y = y + ((ch_max - y) - th) // 2
                        x = max(x, 0)
                        y = max(y, 0)
                        th_c = min(th, ch - y)
                        tw_c = min(tw, cw - x)
                        if th_c > 0 and tw_c > 0:
                            sa = ti[:th_c, :tw_c, 3:4]
                            dst = canvas_i[y:y + th_c, x:x + tw_c, :]
                            dst[:, :, :3] = _cyan_rgb.view(1, 1, 3) * sa + dst[:, :, :3] * (1 - sa)

                    if not self.opts.no_header:
                        _blend_cyan(canvas[0], p["timer_text"], _timer_atlas,
                                    tx1, ty1, tx2, ty2, center_x=not self.opts.dynamic_md, center_y=True)

                    _sd = p.get("stats_data")
                    if _sd is not None:
                        if self.opts.dynamic_md and not self.opts.no_header:
                            # Right timer text: MD (predicted / MMD)
                            _right_text = _sd.get("timer_right_text", "")
                            if _right_text:
                                _tw_right = _compose_text(_right_text, _timer_atlas)[1]
                                _rx_right = tx2 - _tw_right - 4
                                _blend_cyan(canvas[0], _right_text, _timer_atlas,
                                            _rx_right, ty1, tx2, ty2, center_y=True)

                        if not self.opts.no_details:
                            # Stage highlight: CYAN from gs_lf atlas (same positioning as white static base)
                            _cur_stage = _sd.get("grid_current", 0)
                            if _stage_raw_lines and _cur_stage < len(_stage_y_positions):
                                if _cur_stage < len(_stage_highlights):
                                    _hi = _stage_highlights[_cur_stage]
                                    _hx = px + _gs_x
                                    _hy = py + _stage_y_positions[_cur_stage]
                                    _hh = min(_hi.shape[0], ch - _hy)
                                    _hw = min(_hi.shape[1], cw - _hx)
                                    if _hh > 0 and _hw > 0:
                                        _src = _hi[:_hh, :_hw]
                                        _sa = _src[:, :, 3:4]
                                        _dst = canvas[0, _hy:_hy + _hh, _hx:_hx + _hw]
                                        _dst[:, :, :3] = _src[:, :, :3] * _sa + _dst[:, :, :3] * (1 - _sa)

                            # ── Cycles (cyan overlay for entries whose fix time reached) ──
                            _cycles_entry_data = _layout.get("cycles_entry_data", [])
                            if _cycles_entry_data:
                                _ctm = _sd.get("cur_time_ms", 0)
                                _cft = _sd.get("cycles_fix_times", {})
                                for _ed in _cycles_entry_data:
                                    _tile = _ed["tile"]
                                    if _tile is not None:
                                        _fix_ms = _cft.get(_tile)
                                        if _fix_ms is not None and _ctm >= _fix_ms:
                                            _pt(canvas[0:1], _ed["text"], _gs_atlas,
                                                px + _ed["panel_x"], py + _ed["panel_y"], _cyan_rgb_sb)
                else:
                    first_stats_arr = frame_params_list[batch_start].get("stats_arr")
                    if first_stats_arr is not None:
                        s_h, s_w = first_stats_arr.shape[:2]
                        stats_t = torch.from_numpy(first_stats_arr[None, :, :, :]).to(dev, non_blocking=True).float() / 255.0
                        dh = min(s_h, ch - py)
                        dw = min(s_w, cw - px)
                        if dh > 0 and dw > 0:
                            canvas[0, py:py + dh, px:px + dw] = (
                                stats_t[0, :dh, :dw, :3] * stats_t[0, :dh, :dw, 3:] +
                                canvas[0, py:py + dh, px:px + dw] * (1 - stats_t[0, :dh, :dw, 3:])
                            )
                    if not self.opts.no_header:
                        ta = frame_params_list[batch_start].get("timer_arr")
                        if ta is not None:
                            tt = torch.from_numpy(ta).to(dev, non_blocking=True).float() / 255.0
                            dx = max(tx1, tx1 + ((tx2 - tx1) - tt.shape[1]) // 2)
                            dy = max(ty1, ty1 + ((ty2 - ty1) - tt.shape[0]) // 2)
                            self._blend_rgba_inplace(canvas[0], tt, dx, dy)

                _prof_overlays += _tick() - _pt0; _pt0 = _tick()

                # ── GPU → CPU (uint8) ──
                uint8_gpu = canvas.mul(255.0).clamp_(0, 255).to(torch.uint8)
                _prof_uint8 += _tick() - _pt0; _pt0 = _tick()

                if frame_handler:
                    # Async download via pinned memory + CUDA stream (3B-b)
                    _default_stream = torch.cuda.current_stream(dev)
                    _buf_free_events[_dl_buf_idx].wait()
                    _buf_free_events[_dl_buf_idx].clear()
                    with torch.cuda.stream(_dl_stream):
                        _dl_stream.wait_stream(_default_stream)
                        _pinned_bufs[_dl_buf_idx].copy_(uint8_gpu, non_blocking=True)
                    _dl_event = _dl_stream.record_event()

                    # Free GPU tensors
                    del canvas, mats
                    if composite_idx is not None:
                        del composite_idx

                    # Process PREVIOUS batch's frame from the other pinned buffer
                    # (overlaps with THIS batch's DMA transfer)
                    if _dl_prev_event is not None:
                        _dl_prev_event.synchronize()
                        _prev_buf = _pinned_bufs[_prev_buf_idx]
                        frame_handler(_prev_buf[0].numpy(), _dl_prev_start, n, _buf_free_events[_prev_buf_idx])
                        if progress_callback:
                            progress_callback(_dl_prev_start + 1, n, gpu_stats=dict(self._stats))
                        if _dl_prev_uint8 is not None:
                            del _dl_prev_uint8

                    # Rotate for next batch
                    _dl_prev_uint8 = uint8_gpu
                    _dl_prev_event = _dl_event
                    _dl_prev_start = batch_start
                    _prev_buf_idx = _dl_buf_idx
                    _dl_buf_idx = (_dl_buf_idx + 1) % _N_BUFS
                    _prof_dl_handler += _tick() - _pt0
                else:
                    # Sync download (no handler → no overlap possible)
                    batch_u8 = uint8_gpu.cpu().numpy()
                    del uint8_gpu, canvas, mats
                    if composite_idx is not None:
                        del composite_idx
                    frames.append(Image.fromarray(batch_u8[0]))
                    if progress_callback:
                        progress_callback(batch_start + 1, n, gpu_stats=dict(self._stats))

                batch_start = batch_end

        # Process last batch's frame (async path)
        if _dl_prev_event is not None and frame_handler is not None:
            _dl_prev_event.synchronize()
            _prev_buf = _pinned_bufs[1 - _dl_buf_idx]
            frame_handler(_prev_buf[0].numpy(), _dl_prev_start, n)
            if progress_callback is not None:
                progress_callback(_dl_prev_start + 1, n, gpu_stats=dict(self._stats))
            if _dl_prev_uint8 is not None:
                del _dl_prev_uint8

        log.info(f"render_frames: DONE. total_batches={self._batch_counter}, frames_rendered={batch_start}")
        total_t = _time_module.time() - _batch_t0
        _prof_sum = _prof_setup + _prof_delta_patch + _prof_full_render + _prof_overlays + _prof_uint8 + _prof_dl_handler
        _prof_other = total_t - _prof_sum
        log.info(f"===== GPU RENDER SUMMARY =====")
        log.info(f"  total_time={total_t:.1f}s, batches={self._batch_counter}, "
                 f"frames={n}, avg_batch_size={n/max(1,self._batch_counter):.1f}, "
                 f"throughput={n/total_t:.0f} f/s (unique)")
        log.info(f"  profile: setup={_prof_setup:.2f}s "
                 f"delta_patch={_prof_delta_patch:.2f}s ({_prof_delta_count}f) "
                 f"full_render={_prof_full_render:.2f}s ({_prof_full_count}f) "
                 f"overlays={_prof_overlays:.2f}s "
                 f"uint8={_prof_uint8:.2f}s "
                 f"dl_handler={_prof_dl_handler:.2f}s "
                 f"other={_prof_other:.2f}s")
        log.info(f"  RAM delta: {_ram_delta_mb()}MB")
        torch.cuda.empty_cache()
        return frames if not frame_handler else []

    def cleanup(self):
        if self._device is not None and self._device.type == "cuda":
            attrs = [
                "_tile_mask", "_tile_mask_inv",
                "_border_mask", "_border_mask_inv",
                "_bar_fill", "_bar_border",
                "_timer_bg", "_panel_bg", "_panel_bg_rgb", "_panel_bg_a",
                "_composite_atlas",
                "_pinned_bufs", "_dl_stream",
            ]
            for attr in attrs:
                t = getattr(self, attr, None)
                if t is not None:
                    del t
                    setattr(self, attr, None)
            self._composite_lookup = {}
            self._composite_lookup_cpu = {}
            torch.cuda.empty_cache()
            log.info("GPURenderer.cleanup: CUDA tensors freed")

    def __del__(self):
        if self._device is not None:
            try:
                self.cleanup()
            except Exception:
                pass

    # _cpu_fallback removed — GPU errors now raise RuntimeError instead of silently falling back

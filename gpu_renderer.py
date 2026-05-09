import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional

class CancelError(Exception):
    pass


_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

FONT = "calibri.ttf"
FONT_MONO = "consola.ttf"
BG_COLOR = (18, 18, 18)
TILE_BG = (51, 51, 51)
TILE_TEXT_COLOR = (0, 0, 0)
TILE_BORDER_COLOR = (0, 0, 0)
TILE_BORDER_WIDTH = 1
PADDING = 20
STATS_PANEL_WIDTH = 330
HEADER_H = 52
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


def _font(size, bold=False, mono=False):
    name = FONT_MONO if mono else FONT
    try:
        f = ImageFont.truetype(name, size)
    except Exception:
        f = ImageFont.load_default()
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
    font = _font(22, bold=True)
    return _render_text_rgba(timer_text, font, CYAN)


def _render_solution_text(solution_text: str) -> Image.Image:
    font = _font(11, mono=True)
    return _render_text_rgba(solution_text, font, GREEN)


class GPURenderer:
    def __init__(self, width: int, height: int, tile_size: int, quality: float = 2.0,
                 memory_usage: float = 0.5):
        self.w = width
        self.h = height
        self.memory_usage = max(0.05, min(memory_usage, 0.95))
        ts = max(tile_size, int(tile_size * quality))
        self.tile_size = ts
        self.font_size = max(11, ts // 2)
        self.pw = width * ts
        self.ph = height * ts

        puzzle_w = width * ts
        puzzle_h = height * ts
        self.canvas_w = (puzzle_w + STATS_PANEL_WIDTH + PADDING * 3 + 1) // 2 * 2
        self.canvas_h = (HEADER_H + puzzle_h + PADDING * 3 + INFO_H + 1) // 2 * 2
        self.grid_x = PADDING
        self.grid_y = PADDING + HEADER_H + PADDING
        self.panel_x = self.grid_x + puzzle_w + PADDING
        self.panel_y = self.grid_y
        self.panel_w = self.canvas_w - self.panel_x - PADDING
        self.panel_h = self.canvas_h - INFO_H - self.panel_y - PADDING
        self.info_y = self.grid_y + puzzle_h + 4
        self.timer_bbox = (PADDING, PADDING, self.canvas_w - PADDING, PADDING + HEADER_H)

        self._device = None
        self._num_batch = None
        self._tile_mask = None
        self._timer_bg = None
        self._panel_bg = None

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

            # Tile mask: full rectangle (all ones) — no rounded corners
            mask = torch.ones(ts, ts, 1, device=cuda_dev, dtype=torch.float32)
            self._tile_mask = mask

            # Tile border mask: 1px border around tile
            border = torch.zeros(ts, ts, 1, device=cuda_dev, dtype=torch.float32)
            border[0, :] = 1
            border[ts - 1, :] = 1
            border[:, 0] = 1
            border[:, ts - 1] = 1
            self._border_mask = border

            # Secondary color bar mask + border
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
            # Remove fill interior from border mask to prevent double-counting
            if by1 - by0 > 2 and bx1 - bx0 > 2:
                bar_border[by0 + 1:by1 - 1, bx0 + 1:bx1 - 1] = 0.0
            self._bar_fill = bar_fill
            self._bar_border = bar_border

            # Pre-render static timer bar background
            tx1, ty1, tx2, ty2 = self.timer_bbox
            tw = tx2 - tx1
            th = ty2 - ty1
            timer_bg_pil = Image.new("RGB", (tw, th), TIMER_BG)
            self._timer_bg = torch.from_numpy(
                np.array(timer_bg_pil)
            ).to(cuda_dev).float() / 255.0

            # Pre-render static panel background + border
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
                self._panel_bg = panel_arr  # (panel_h, panel_w, 4) RGBA
                self._panel_bg_rgb = panel_arr[:, :, :3]
                self._panel_bg_a = panel_arr[:, :, 3:4]

    @property
    def available(self):
        return self._device is not None and self._device.type == "cuda" and self._num_batch is not None

    def render_frames(
        self,
        frame_params_list: List[dict],
        progress_callback=None,
        cancel_check=None,
        stats_path: str = None
    ) -> List[Image.Image]:
        if not self.available or not frame_params_list:
            return self._cpu_fallback(frame_params_list, progress_callback, cancel_check)

        n = len(frame_params_list)
        w, h, ts = self.w, self.h, self.tile_size
        pw, ph = self.pw, self.ph
        cw, ch = self.canvas_w, self.canvas_h
        gx, gy = self.grid_x, self.grid_y
        px, py = self.panel_x, self.panel_y
        info_y = self.info_y
        tx1, ty1, tx2, ty2 = self.timer_bbox

        dev = self._device
        num_tex = self._num_batch
        tile_mask = self._tile_mask
        border_mask = self._border_mask
        timer_bg = self._timer_bg
        panel_bg_rgb = self._panel_bg_rgb
        panel_bg_a = self._panel_bg_a

        c_bg = torch.tensor(BG_COLOR, device=dev, dtype=torch.float32) / 255.0
        c_border = torch.tensor(TILE_BORDER_COLOR, device=dev, dtype=torch.float32) / 255.0

        # Compute safe batch size from available GPU memory
        free_bytes = torch.cuda.mem_get_info(dev)[0]
        per_frame_est = (ch * cw * 3 * 4) + (h * w * ts * ts * 4 * 8) + (2 * 1024 * 1024)
        batch_size = max(1, min(n, int(free_bytes * self.memory_usage / max(per_frame_est, 1))))

        self._stats["batch_size"] = batch_size
        self._stats["batch_mem_mb"] = (ch * cw * 3 * 4 * batch_size) // (1024 * 1024)

        frames = []
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_n = batch_end - batch_start

            # Refresh GPU memory stats before allocating the batch
            free, _ = torch.cuda.mem_get_info(dev)
            self._stats["free_mem_mb"] = free // (1024 * 1024)
            self._stats["mem_used_mb"] = self._stats["total_mem_mb"] - self._stats["free_mem_mb"]
            self._stats["batch_idx"] = batch_start // batch_size + 1 if batch_size else 0

            if stats_path:
                import json, time as _time
                with open(stats_path, "a") as _sf:
                    _sf.write(json.dumps({
                        "event": "batch_start",
                        "t": _time.time(),
                        "batch_idx": self._stats["batch_idx"],
                        "batch_size": batch_n,
                        "free_mem_mb": self._stats["free_mem_mb"],
                        "used_mem_mb": self._stats["mem_used_mb"],
                        "total_mem_mb": self._stats["total_mem_mb"],
                        "batch_mem_mb": self._stats["batch_mem_mb"],
                        "frames_done": batch_start,
                        "frames_total": n,
                    }) + "\n")

            canvas = torch.empty(batch_n, ch, cw, 3, device=dev, dtype=torch.float32)
            canvas[:] = c_bg.view(1, 1, 3)

            for fi in range(batch_start, batch_end):
                if cancel_check and cancel_check():
                    raise CancelError()
                params = frame_params_list[fi]
                mat = params["matrix"]
                colors_data = params.get("colors", None)

                fc = canvas[fi - batch_start]

                # 1. Timer bar background
                fc[ty1:ty2, tx1:tx2] = timer_bg

                # 2. Timer text (centered)
                timer_img = params.get("timer_img")
                if timer_img is not None:
                    timer_t = torch.from_numpy(np.array(timer_img)).to(dev).float() / 255.0
                    ti_h, ti_w = timer_t.shape[:2]
                    tx = max(tx1, tx1 + ((tx2 - tx1) - ti_w) // 2)
                    ty = max(ty1, ty1 + ((ty2 - ty1) - ti_h) // 2)
                    use_h_t = min(ti_h, ch - ty)
                    use_w_t = min(ti_w, cw - tx)
                    if use_h_t > 0 and use_w_t > 0:
                        fc[ty:ty + use_h_t, tx:tx + use_w_t] = (
                            timer_t[:use_h_t, :use_w_t, :3] * timer_t[:use_h_t, :use_w_t, 3:4] +
                            fc[ty:ty + use_h_t, tx:tx + use_w_t] * (1 - timer_t[:use_h_t, :use_w_t, 3:4])
                        )

                # 3. Tile grid (vectorized)
                mat_t = torch.tensor(mat, device=dev, dtype=torch.long)

                if colors_data is not None:
                    col_arr = []
                    sec_arr = []
                    has_sec_arr = []
                    for row in range(h):
                        col_row = []
                        sec_row = []
                        hs_row = []
                        for ci in range(w):
                            mc, sc = colors_data[row][ci]
                            col_row.append(mc if mc is not None else TILE_BG)
                            if sc is not None:
                                sec_row.append(sc)
                                hs_row.append(1.0)
                            else:
                                sec_row.append(TILE_BG)
                                hs_row.append(0.0)
                        col_arr.append(col_row)
                        sec_arr.append(sec_row)
                        has_sec_arr.append(hs_row)
                    cols_t = torch.tensor(col_arr, device=dev, dtype=torch.float32) / 255.0
                    sec_color_t = torch.tensor(sec_arr, device=dev, dtype=torch.float32) / 255.0
                    has_sec_t = torch.tensor(has_sec_arr, device=dev, dtype=torch.float32).view(h, w, 1, 1, 1)
                else:
                    cols_t = torch.full((h, w, 3), 51.0 / 255.0, device=dev, dtype=torch.float32)
                    sec_color_t = torch.zeros(h, w, 3, device=dev, dtype=torch.float32)
                    has_sec_t = torch.zeros(h, w, 1, 1, 1, device=dev, dtype=torch.float32)

                num_batch = num_tex[mat_t]
                text_rgb = num_batch[:, :, :, :, :3]
                text_a = num_batch[:, :, :, :, 3:4]

                colored = cols_t[:, :, None, None, :] * tile_mask[None, None, :, :, :]
                colored += c_bg.view(1, 1, 1, 1, 3) * (1 - tile_mask[None, None, :, :, :])

                border_layer = c_border.view(1, 1, 1, 1, 3) * border_mask[None, None, :, :, :]
                inside_border = (1 - border_mask[None, None, :, :, :])
                colored = border_layer + colored * inside_border

                # Secondary color bar (fill + border below number text)
                bar_fill = sec_color_t[:, :, None, None, :] * self._bar_fill[None, None, :, :, :]
                bar_border = c_border.view(1, 1, 1, 1, 3) * self._bar_border[None, None, :, :, :]
                bar_blend_fill = has_sec_t * self._bar_fill[None, None, :, :, :]
                bar_blend_border = has_sec_t * self._bar_border[None, None, :, :, :]
                colored = colored * (1 - bar_blend_fill) + bar_fill * bar_blend_fill
                colored = colored * (1 - bar_blend_border) + bar_border * bar_blend_border

                tile_result = text_rgb * text_a + colored * (1 - text_a)

                fc[gy:gy + ph, gx:gx + pw] = tile_result.permute(0, 2, 1, 3, 4).reshape(ph, pw, 3)

                # 4. Solution text below puzzle
                sol_img = params.get("sol_img")
                if sol_img is not None:
                    sol_t = torch.from_numpy(np.array(sol_img)).to(dev).float() / 255.0
                    si_h, si_w = sol_t.shape[:2]
                    use_h_s = min(si_h, ch - info_y)
                    use_w_s = min(si_w, cw - gx)
                    if use_h_s > 0 and use_w_s > 0:
                        fc[info_y:info_y + use_h_s, gx:gx + use_w_s] = (
                            sol_t[:use_h_s, :use_w_s, :3] * sol_t[:use_h_s, :use_w_s, 3:4] +
                            fc[info_y:info_y + use_h_s, gx:gx + use_w_s] * (1 - sol_t[:use_h_s, :use_w_s, 3:4])
                        )

                # 5. Stats panel background + border
                if panel_bg_rgb is not None and self.panel_h > 0 and self.panel_w > 0:
                    pw_p, ph_p = self.panel_w, self.panel_h
                    fc[py:py + ph_p, px:px + pw_p] = (
                        panel_bg_rgb * panel_bg_a +
                        fc[py:py + ph_p, px:px + pw_p] * (1 - panel_bg_a)
                    )

                # 6. Stats text overlay
                stats_img = params.get("stats_img")
                if stats_img is not None:
                    st_t = torch.from_numpy(np.array(stats_img)).to(dev).float() / 255.0
                    st_h, st_w = st_t.shape[:2]
                    use_h = min(st_h, ch - py)
                    use_w = min(st_w, cw - px)
                    if use_h > 0 and use_w > 0:
                        fc[py:py + use_h, px:px + use_w] = (
                            st_t[:use_h, :use_w, :3] * st_t[:use_h, :use_w, 3:4] +
                            fc[py:py + use_h, px:px + use_w] * (1 - st_t[:use_h, :use_w, 3:4])
                        )

            # Copy batch to CPU
            batch_arr = (canvas.cpu().numpy() * 255).astype("uint8")
            for i in range(batch_n):
                frames.append(Image.fromarray(batch_arr[i]))
                if progress_callback:
                    progress_callback(batch_start + i + 1, n, gpu_stats=dict(self._stats))

        return frames

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
            sol_img = p.get("sol_img")
            stats_img = p.get("stats_img")

            cw, ch = self.canvas_w, self.canvas_h
            canvas = Image.new("RGB", (cw, ch), BG_COLOR)
            draw = ImageDraw.Draw(canvas)

            # Timer bar background
            tx1, ty1, tx2, ty2 = self.timer_bbox
            draw.rectangle((tx1, ty1, tx2, ty2), fill=TIMER_BG)

            # Timer text
            if timer_img:
                ti_w, ti_h = timer_img.size
                tx = tx1 + ((tx2 - tx1) - ti_w) // 2
                ty = ty1 + ((ty2 - ty1) - ti_h) // 2
                canvas.paste(timer_img, (tx, ty), timer_img)

            # Tile grid
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

            # Solution text
            if sol_img:
                canvas.paste(sol_img, (gx, self.info_y), sol_img)

            # Stats panel
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

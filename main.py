import tkinter as tk
from tkinter import colorchooser, filedialog, scrolledtext, ttk
import threading
import time
import os
import subprocess
import sys
import ctypes
from concurrent.futures import ThreadPoolExecutor

import ttkbootstrap as tb
from ttkbootstrap.constants import *

from replay_video import ReplayVideoGenerator, CancelError, _quick_infer_size, _get_available_encoders, render_frame, get_all_fringe_schemes
from grids_analysis import generate_grids_stats
from sliding_puzzles import parse_replay_url
from replay_generator import count_moves
from geometry import RenderOptions, parse_hex_color
from PIL import Image, ImageTk
from debug_log import get_logger, init_logfile

log = get_logger()

if getattr(sys, 'frozen', False):
    base = sys._MEIPASS
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.environ["PATH"] = base + os.pathsep + os.environ.get("PATH", "")
else:
    base = os.path.dirname(os.path.abspath(__file__))
    script_dir = base

_font_dir = os.path.join(base, "fonts")
FONT_REGULAR = os.path.join(_font_dir, "Roboto-Regular.ttf")
FONT_BOLD = os.path.join(_font_dir, "Roboto-Bold.ttf")
FONT_MONO = os.path.join(_font_dir, "JetBrainsMono-Regular.ttf")
FONT_MONO_BOLD = os.path.join(_font_dir, "JetBrainsMono-Bold.ttf")
FONT_FAMILY = "Roboto"
FONT_MONO_FAMILY = "JetBrains Mono"


def _setup_placeholder(text_widget, placeholder):
    text_widget._has_placeholder = False
    def _on_focus_in(_):
        if text_widget._has_placeholder:
            text_widget.delete("1.0", "end-1c")
            text_widget._has_placeholder = False
            text_widget.config(fg="#d4d4d4")
    def _on_focus_out(_):
        if not text_widget.get("1.0", "end-1c").strip():
            text_widget.delete("1.0", "end-1c")
            text_widget.insert("1.0", placeholder)
            text_widget._has_placeholder = True
            text_widget.config(fg="#666666")
    text_widget.bind("<FocusIn>", _on_focus_in, add="+")
    text_widget.bind("<FocusOut>", _on_focus_out, add="+")
    _on_focus_out(None)


def _register_fonts():
    for fp in (FONT_REGULAR, FONT_BOLD, FONT_MONO, FONT_MONO_BOLD):
        if not os.path.exists(fp):
            log.warning(f"Font file not found: {fp}")
            continue
        if sys.platform == "win32":
            r = ctypes.windll.gdi32.AddFontResourceExW(fp, 0x10, 0)
            if r:
                log.info(f"Registered font: {fp}")
            else:
                log.warning(f"Failed to register font: {fp}")
        elif sys.platform == "linux":
            methods = 0
            ok = 0
            try:
                lib = ctypes.CDLL("libfontconfig.so.1")
                if hasattr(lib, "FcConfigAppFontAddFile"):
                    r = lib.FcConfigAppFontAddFile(None, fp.encode("utf-8"))
                    ok += bool(r)
                methods += 1
            except Exception as e:
                log.info(f"fontconfig failed for {os.path.basename(fp)}: {e}")
            if not ok:
                try:
                    font_dir = os.path.expanduser("~/.local/share/fonts")
                    os.makedirs(font_dir, exist_ok=True)
                    dest = os.path.join(font_dir, os.path.basename(fp))
                    if not os.path.exists(dest):
                        import shutil
                        shutil.copy2(fp, dest)
                        ok += 1
                    methods += 1
                except Exception as e:
                    log.warning(f"Font copy fallback failed for {os.path.basename(fp)}: {e}")
            if ok:
                log.info(f"Registered Linux font: {fp}")
            elif methods:
                log.warning(f"All methods failed for: {fp}")


def _open(path, status_callback=None):
    if sys.platform == "win32":
        os.startfile(path)
    else:
        try:
            subprocess.Popen(["xdg-open", path])
        except OSError:
            if status_callback:
                status_callback("Could not open file: xdg-open not found")


def _generate_filename(solution, tps, time_v, movetimes, size_arg=None, index=0, speed_factor=1.0, scramble=None):
    moves = count_moves(solution)
    if isinstance(movetimes, list) and len(movetimes) > 1:
        time_s = movetimes[-1] / 1000.0 if movetimes[-1] > 0 else 0
        is_movetimes_accurate = True
        display_tps = tps if tps and tps > 0 else (moves / time_s if time_s > 0 else None)
    elif time_v and time_v > 0:
        time_s = time_v
        is_movetimes_accurate = False
        display_tps = moves / time_s
    elif tps and tps > 0:
        time_s = moves / tps
        is_movetimes_accurate = False
        display_tps = tps
    else:
        time_s = 0
        is_movetimes_accurate = False
        display_tps = None
    if size_arg:
        if isinstance(size_arg, tuple):
            w, h = size_arg
        else:
            parts = str(size_arg).lower().split("x")
            w, h = parts[0], parts[1]
    elif scramble:
        rows = scramble.split("/")
        h = len(rows)
        w = len(rows[0].split()) if rows else 0
    else:
        size = _quick_infer_size(solution, scramble)
        w, h = size if size else ("?", "?")
    parts = [f"{w}x{h}"]
    if time_s:
        parts.append(f"{time_s:.3f}")
    parts.append(str(moves))
    if display_tps:
        parts.append(f"{display_tps:.3f}")
    if is_movetimes_accurate:
        parts.append("movetimes")
    if speed_factor is not None and speed_factor != 1.0:
        parts.append(f"{speed_factor}x")
    name = "_".join(parts)
    name = name.translate(str.maketrans("", "", '\\/:*?\"<>|'))
    if index:
        name = f"{index:03d}_{name}"
    return f"{name}.mp4"


def _pick_output_filename(output_dir, base_name):
    path = os.path.join(output_dir, base_name)
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(base_name)
    n = 1
    while os.path.exists(os.path.join(output_dir, f"{stem}_{n}{ext}")):
        n += 1
    return os.path.join(output_dir, f"{stem}_{n}{ext}")


class ReplayGUI(tb.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.withdraw()
        self.title("Replay Video Generator")
        self._base_w, self._base_h = 1280, 720
        self.minsize(self._base_w, self._base_h)

        self.generated_files = []
        self.render_queue = []
        self._executor = None
        self._batch_futures = []
        self._item_progress = {}
        self._start_time = 0.0
        self._last_poll_time = 0.0
        self._last_poll_pct = 0.0
        self._rolling_rate = 0.0
        self.cancel_flag = False
        self.pb_overall_text = tk.StringVar(value="0 / 0")
        self.pb_detail_text = tk.StringVar(value="")

        self.fps_var = tk.IntVar(value=60)
        self.main_scheme_var = tk.StringVar(value="fringe")
        self.force_main_var = tk.BooleanVar(value=False)
        self.hue_start_var = tk.DoubleVar(value=0)
        self.hue_end_var = tk.DoubleVar(value=360)
        self.saturation_var = tk.IntVar(value=78)
        self.brightness_var = tk.IntVar(value=60)
        self._preview_job = None
        self._preview_photo = None
        self._preview_sel_idx = -1
        self.quality_preset_var = tk.StringVar(value="1080p")
        self._quality_presets = {"720p": 720, "1080p": 1080, "1440p (2K)": 1440, "2160p (4K)": 2160}
        self.compression_var = tk.IntVar(value=18)
        self.slow_render_var = tk.BooleanVar(value=False)
        self.speed_factor_var = tk.StringVar(value="1")

        self.tps_var = tk.StringVar()
        self.time_var = tk.StringVar()
        self.size_var = tk.StringVar()
        self.scramble_var = tk.StringVar()
        self.movetimes_var = tk.StringVar()
        self.out_folder_var = tk.StringVar(value=os.path.join(script_dir, "replays"))
        self.file_path_var = tk.StringVar()
        self.progress_text = self.pb_overall_text
        self._gpu_info_var = tk.StringVar(value="")
        self.no_layout_var = tk.BooleanVar(value=False)
        self.no_border_var = tk.BooleanVar(value=False)
        self.no_secondary_border_var = tk.BooleanVar(value=False)
        self.no_grid_bars_var = tk.BooleanVar(value=False)
        self.no_numbers_var = tk.BooleanVar(value=False)
        self.no_header_var = tk.BooleanVar(value=False)
        self.no_details_var = tk.BooleanVar(value=False)
        self.dynamic_md_var = tk.BooleanVar(value=False)
        self.upscale_var = tk.BooleanVar(value=False)
        self.cycles_detection_var = tk.BooleanVar(value=False)
        self.adjust_height_var = tk.BooleanVar(value=False)
        self.animate_moves_var = tk.BooleanVar(value=False)

        _register_fonts()
        os.makedirs(self.out_folder_var.get(), exist_ok=True)
        self._build_ui()
        self._center_window()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.deiconify()
        self.after(0, self._set_icon)

    def _set_icon(self):
        icon_path = os.path.join(base, "assets", "15PUZZLE_ICON.png")
        if os.path.exists(icon_path):
            try:
                from PIL import Image, ImageTk
                self._icon = ImageTk.PhotoImage(Image.open(icon_path))
                self.iconphoto(True, self._icon)
            except Exception:
                pass

    def _build_ui(self):
        style = tb.Style()
        style.configure("TCheckbutton", font=(FONT_FAMILY, 9))
        style.configure("Round.Toggle", font=(FONT_FAMILY, 10))
        root = tb.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        # ── Three-column layout using grid ──
        root.grid_columnconfigure(0, weight=0, minsize=300)
        root.grid_columnconfigure(1, weight=1, minsize=400)
        root.grid_columnconfigure(2, weight=1, minsize=300)
        root.grid_rowconfigure(0, weight=1)

        # ======== COLUMN 0: SETTINGS ========
        settings = tb.LabelFrame(root, text="Settings")
        settings.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        settings.grid_columnconfigure(0, weight=1)

        r = 0
        _sec_font = (FONT_FAMILY, 9, "bold")
        _tog_font = (FONT_FAMILY, 9)

        # ════════ OUTPUT ════════
        tb.Label(settings, text="OUTPUT", font=_sec_font, bootstyle="secondary").grid(
            row=r, column=0, sticky="w", padx=12, pady=(6, 0))
        r += 1
        tb.Separator(settings, bootstyle="secondary").grid(
            row=r, column=0, sticky="ew", pady=(1, 4), padx=12)
        r += 1

        out_row = tb.Frame(settings)
        out_row.grid(row=r, column=0, sticky="ew", pady=(2, 4), padx=12)
        out_row.grid_columnconfigure(1, weight=1)
        tb.Label(out_row, text="Output folder", font=(FONT_FAMILY, 9)).grid(
            row=0, column=0, sticky="w", padx=(0, 6))
        self.out_entry = tb.Entry(out_row, textvariable=self.out_folder_var)
        self.out_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        tb.Button(out_row, text="Browse...", command=self._browse_output,
                  bootstyle="secondary-outline", width=9).grid(row=0, column=2)
        r += 1

        # ════════ QUALITY ════════
        tb.Label(settings, text="QUALITY", font=_sec_font, bootstyle="secondary").grid(
            row=r, column=0, sticky="w", padx=12, pady=(6, 0))
        r += 1
        tb.Separator(settings, bootstyle="secondary").grid(
            row=r, column=0, sticky="ew", pady=(1, 4), padx=12)
        r += 1

        # Quality preset
        qual_row = tb.Frame(settings)
        qual_row.grid(row=r, column=0, sticky="ew", pady=(2, 4), padx=12)
        qual_row.grid_columnconfigure(1, weight=1)
        tb.Label(qual_row, text="Quality:", font=(FONT_FAMILY, 9)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        quality_combo = ttk.Combobox(qual_row, textvariable=self.quality_preset_var,
                                     values=list(self._quality_presets.keys()),
                                     state="readonly", width=14)
        quality_combo.grid(row=0, column=1, sticky="w")
        def _on_quality_change(*_):
            self._update_quality_warning()
        self.quality_preset_var.trace_add("write", _on_quality_change)
        r += 1

        # FPS
        fps_row = tb.Frame(settings)
        fps_row.grid(row=r, column=0, sticky="ew", pady=(2, 4), padx=12)
        fps_row.grid_columnconfigure(1, weight=1)
        tb.Label(fps_row, text="FPS:", font=(FONT_FAMILY, 9)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.fps_scale = tb.Scale(fps_row, from_=5, to=240,
                                  variable=self.fps_var, orient="horizontal")
        self.fps_scale.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.fps_label = tb.Label(fps_row, text="60", width=4, font=(FONT_FAMILY, 9))
        self.fps_label.grid(row=0, column=2)
        def _snap_fps(*_):
            v = self.fps_var.get()
            snapped = round(v / 5) * 5
            if snapped != v:
                self.fps_var.set(snapped)
            self.fps_label.config(text=f"{snapped:d}")
        self.fps_var.trace_add("write", _snap_fps)
        r += 1

        # Compression + range desc (side by side)
        comp_row = tb.Frame(settings)
        comp_row.grid(row=r, column=0, sticky="ew", pady=(2, 4), padx=12)
        comp_row.grid_columnconfigure(1, weight=1)
        tb.Label(comp_row, text="Compression:", font=(FONT_FAMILY, 9)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        compression_scale = tb.Scale(comp_row, from_=10, to=40, variable=self.compression_var,
                                     orient="horizontal", bootstyle="primary")
        compression_scale.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        self.compression_value_lbl = tb.Label(comp_row, text=str(self.compression_var.get()),
                                               font=(FONT_FAMILY, 9, "bold"), width=3)
        self.compression_value_lbl.grid(row=0, column=2, sticky="w", padx=(0, 8))
        tb.Label(comp_row, text="10 (best) – 40 (smallest)", font=(FONT_FAMILY, 8),
                 foreground="#888").grid(row=0, column=3, sticky="w")
        def _on_compression_change(*_):
            self.compression_value_lbl.config(text=str(self.compression_var.get()))
        self.compression_var.trace_add("write", _on_compression_change)
        r += 1

        # Speed
        spd_row = tb.Frame(settings)
        spd_row.grid(row=r, column=0, sticky="ew", pady=(2, 4), padx=12)
        spd_row.grid_columnconfigure(1, weight=1)
        tb.Label(spd_row, text="Speed:", font=(FONT_FAMILY, 9)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.speed_entry = tb.Entry(spd_row, textvariable=self.speed_factor_var, width=8)
        self.speed_entry.grid(row=0, column=1, sticky="w")
        tb.Label(spd_row, text="2.0 = 2× faster, 0.5 = half speed", font=(FONT_FAMILY, 8),
                 foreground="#888").grid(row=0, column=2, sticky="w", padx=(6, 0))
        r += 1

        # Quality RAM warning (conditional — no upscale fluff)
        self.quality_warning = tb.Label(settings, text="", font=(FONT_FAMILY, 8),
                                        foreground="#ffa500", anchor="w")
        self.quality_warning.grid(row=r, column=0, sticky="ew", pady=(0, 4), padx=12)
        self.quality_warning.grid_remove()
        r += 1

        # ════════ RENDER ════════
        tb.Label(settings, text="RENDER", font=_sec_font, bootstyle="secondary").grid(
            row=r, column=0, sticky="w", padx=12, pady=(6, 0))
        r += 1
        tb.Separator(settings, bootstyle="secondary").grid(
            row=r, column=0, sticky="ew", pady=(1, 4), padx=12)
        r += 1

        self._gpu_available = False
        self._gpu_name = ""
        try:
            import torch
            self._gpu_available = torch.cuda.is_available()
            if self._gpu_available:
                self._gpu_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        self.use_gpu_var = tk.BooleanVar(value=self._gpu_available)

        # GPU row: toggle + status text inline
        gpu_row = tb.Frame(settings)
        gpu_row.grid(row=r, column=0, sticky="ew", pady=(2, 4), padx=12)
        self.gpu_toggle = tb.Checkbutton(gpu_row, text="GPU acceleration",
                                          variable=self.use_gpu_var,
                                          bootstyle="round-toggle success")
        self.gpu_toggle.pack(side="left", padx=(0, 8))
        self.gpu_status = tb.Label(gpu_row, text="", font=(FONT_FAMILY, 9), foreground="#888")
        self.gpu_status.pack(side="left")
        r += 1

        # Slow render row
        slow_row = tb.Frame(settings)
        slow_row.grid(row=r, column=0, sticky="ew", pady=(2, 4), padx=12)
        self.slow_render_cb = tb.Checkbutton(slow_row, text="Slow render",
                                              variable=self.slow_render_var,
                                              bootstyle="round-toggle")
        self.slow_render_cb.pack(side="left")
        self.slow_render_desc = tb.Label(slow_row, text="~33% smaller file, ~33% longer",
                                          font=(FONT_FAMILY, 9), foreground="#888")
        self.slow_render_desc.pack(side="left", padx=(8, 0))
        r += 1

        # Upscale
        up_row = tb.Frame(settings)
        up_row.grid(row=r, column=0, sticky="ew", pady=(2, 4), padx=12)
        self.upscale_cb = tb.Checkbutton(up_row, text="Upscale to 2K",
                                          variable=self.upscale_var,
                                          bootstyle="round-toggle")
        self.upscale_cb.pack(side="left")
        tb.Label(up_row, text="Re-encode to 2560×1440 for best YouTube quality",
                 font=(FONT_FAMILY, 9), foreground="#888").pack(side="left", padx=(8, 0))
        self.upscale_var.trace_add("write", lambda *_: self._update_quality_warning())
        r += 1

        # Encoder
        enc_row = tb.Frame(settings)
        enc_row.grid(row=r, column=0, sticky="ew", pady=(2, 4), padx=12)
        tb.Label(enc_row, text="Encoder:", font=(FONT_FAMILY, 9)).pack(side="left", padx=(0, 6))
        self.encoder_var = tk.StringVar(value="Auto")
        available = _get_available_encoders()
        enc_values = ["Auto"] + available
        enc_combo = ttk.Combobox(enc_row, textvariable=self.encoder_var,
                                 values=enc_values, state="readonly", width=16)
        enc_combo.pack(side="left")
        r += 1

        def _update_gpu_status():
            if self.use_gpu_var.get():
                if self._gpu_available:
                    self.gpu_status.config(text=f"ON — {self._gpu_name}", bootstyle="success")
                else:
                    self.gpu_status.config(text="Not available — install CUDA", bootstyle="secondary")
            else:
                self.gpu_status.config(text="OFF (CPU)", bootstyle="secondary")
                self.slow_render_var.set(True)
            _update_slow_render_desc()

        def _update_slow_render_desc():
            if not self.use_gpu_var.get():
                self.slow_render_desc.config(text="auto-enabled for CPU (free smaller files)")
            elif self.slow_render_var.get():
                self.slow_render_desc.config(text="~33% smaller file, ~33% longer")
            else:
                self.slow_render_desc.config(text="")
        self.slow_render_var.trace_add("write", lambda *_: _update_slow_render_desc())
        self.gpu_toggle.config(command=_update_gpu_status)
        self.after(10, _update_gpu_status)

        # ════════ DISPLAY ════════
        tb.Label(settings, text="DISPLAY", font=_sec_font, bootstyle="secondary").grid(
            row=r, column=0, sticky="w", padx=12, pady=(6, 0))
        r += 1
        tb.Separator(settings, bootstyle="secondary").grid(
            row=r, column=0, sticky="ew", pady=(1, 4), padx=12)
        r += 1

        d_grid = tb.Frame(settings)
        d_grid.grid(row=r, column=0, sticky="ew", pady=(2, 4), padx=12)
        d_grid.grid_columnconfigure(0, weight=1)
        d_grid.grid_columnconfigure(1, weight=1)
        d_grid.grid_columnconfigure(2, weight=1)
        r += 1

        _col_hdr = {"font": (FONT_FAMILY, 9), "foreground": "#aaa", "anchor": "w"}

        # Column 0: Puzzle
        tb.Label(d_grid, text="Puzzle", **_col_hdr).grid(row=0, column=0, sticky="w", pady=(0, 2))
        tb.Checkbutton(d_grid, text="No border", variable=self.no_border_var,
                       bootstyle="round-toggle").grid(row=1, column=0, sticky="w")
        tb.Checkbutton(d_grid, text="No sec border", variable=self.no_secondary_border_var,
                       bootstyle="round-toggle").grid(row=2, column=0, sticky="w")
        tb.Checkbutton(d_grid, text="No numbers", variable=self.no_numbers_var,
                       bootstyle="round-toggle").grid(row=3, column=0, sticky="w")
        tb.Checkbutton(d_grid, text="No grid bars", variable=self.no_grid_bars_var,
                       bootstyle="round-toggle").grid(row=4, column=0, sticky="w")
 
        # Column 1: Layout
        tb.Label(d_grid, text="Layout", **_col_hdr).grid(row=0, column=1, sticky="w", pady=(0, 2))
        tb.Checkbutton(d_grid, text="No header", variable=self.no_header_var,
                       bootstyle="round-toggle").grid(row=1, column=1, sticky="w")
        tb.Checkbutton(d_grid, text="No details", variable=self.no_details_var,
                       bootstyle="round-toggle").grid(row=2, column=1, sticky="w")
        tb.Checkbutton(d_grid, text="No layout", variable=self.no_layout_var,
                       bootstyle="round-toggle").grid(row=3, column=1, sticky="w")
        tb.Checkbutton(d_grid, text="Adjust height", variable=self.adjust_height_var,
                       bootstyle="round-toggle").grid(row=4, column=1, sticky="w")

        # Column 2: Extra
        tb.Label(d_grid, text="Extra", **_col_hdr).grid(row=0, column=2, sticky="w", pady=(0, 2))
        tb.Checkbutton(d_grid, text="Dynamic MD", variable=self.dynamic_md_var,
                       bootstyle="round-toggle").grid(row=1, column=2, sticky="w")
        tb.Checkbutton(d_grid, text="Force main", variable=self.force_main_var,
                       bootstyle="round-toggle").grid(row=2, column=2, sticky="w")
        tb.Checkbutton(d_grid, text="Cycle detect", variable=self.cycles_detection_var,
                       bootstyle="round-toggle").grid(row=3, column=2, sticky="w")
        tb.Checkbutton(d_grid, text="Animate moves", variable=self.animate_moves_var,
                       bootstyle="round-toggle").grid(row=4, column=2, sticky="w")

        # ════════ COLORS ════════
        r += 1
        tb.Label(settings, text="COLORS", font=_sec_font, bootstyle="secondary").grid(
            row=r, column=0, sticky="w", padx=12, pady=(6, 0))
        r += 1
        tb.Separator(settings, bootstyle="secondary").grid(
            row=r, column=0, sticky="ew", pady=(1, 4), padx=12)
        r += 1

        self._color_vars = {
            "grid1": tk.StringVar(value="C86767"),
            "grid2": tk.StringVar(value="8DB3FF"),
            "tile_bg": tk.StringVar(value="454545"),
        }
        self._color_previews = {}

        def _pick_color(name: str, label: str):
            initial = self._color_vars[name].get()
            rgb = colorchooser.askcolor(
                color=f"#{initial}", title=f"Choose {label}",
                parent=self)
            if rgb[0] is not None:
                r_, g_, b_ = [int(x) for x in rgb[0]]
                hex_str = f"{r_:02X}{g_:02X}{b_:02X}"
                self._color_vars[name].set(hex_str)

        _color_labels = [
            ("grid1", "Grid 1 (red grids):"),
            ("grid2", "Grid 2 (blue grids):"),
            ("tile_bg", "Tile background:"),
        ]
        for name, label in _color_labels:
            row_f = tb.Frame(settings)
            row_f.grid(row=r, column=0, sticky="ew", pady=1, padx=12)
            row_f.grid_columnconfigure(1, weight=1)
            preview = tb.Label(row_f, text="    ", background=f"#{self._color_vars[name].get()}")
            preview.grid(row=0, column=0, padx=(0, 6))
            self._color_previews[name] = preview
            tb.Label(row_f, text=label, font=(FONT_FAMILY, 9)).grid(row=0, column=1, sticky="w")
            tb.Button(row_f, text="Pick", bootstyle="secondary-outline",
                      command=lambda n=name, lbl=label: _pick_color(n, lbl),
                      width=5).grid(row=0, column=2, padx=(4, 0))
            self._color_vars[name].trace_add("write",
                lambda *_, n=name: self._color_previews[n].config(background=f"#{self._color_vars[n].get()}"))
            r += 1

        scheme_row = tb.Frame(settings)
        scheme_row.grid(row=r, column=0, sticky="ew", pady=(4, 2), padx=12)
        scheme_row.grid_columnconfigure(1, weight=1)
        tb.Label(scheme_row, text="Main scheme:", font=(FONT_FAMILY, 9)).grid(row=0, column=0, padx=(0, 6))
        main_scheme_combo = ttk.Combobox(scheme_row, textvariable=self.main_scheme_var,
                                         values=["fringe", "rows", "columns"],
                                         state="readonly", width=9)
        main_scheme_combo.grid(row=0, column=1, sticky="w")
        r += 1

        # ======== COLUMN 1: UNIFIED INPUT + OVERRIDES ========
        self.mid = tb.Frame(root)
        self.mid.grid(row=0, column=1, sticky="nsew", padx=(3, 3))
        self.mid.grid_propagate(False)
        self.mid.grid_columnconfigure(0, weight=1)
        mid = self.mid

        # Preview frame
        self.preview_frame = tb.LabelFrame(mid, text="Preview")
        self.preview_frame.pack(fill="x", pady=(0, 4))
        preview_inner = tb.Frame(self.preview_frame)
        preview_inner.pack(pady=4)
        self._preview_label = tb.Label(preview_inner)
        self._preview_label.pack()
        self._preview_info = tb.Label(self.preview_frame, text="4x4 · Fringe",
                                       font=(FONT_FAMILY, 8), foreground="#888")
        self._preview_info.pack(anchor="w", padx=4, pady=(0, 2))
        tb.Label(self.preview_frame, text="Select a replay in the queue to preview",
                 font=(FONT_FAMILY, 7), foreground="#666").pack(anchor="w", padx=4, pady=(0, 2))

        def _add_slider(parent, label, var, from_, to, fmt="{:.0f}"):
            row = tb.Frame(parent)
            row.pack(fill="x", padx=8, pady=(0, 2))
            row.grid_columnconfigure(1, weight=1)
            tb.Label(row, text=label, font=(FONT_FAMILY, 8)).grid(row=0, column=0, sticky="w", padx=(0, 4))
            tb.Scale(row, from_=from_, to=to, variable=var, orient="horizontal", bootstyle="primary").grid(
                row=0, column=1, sticky="ew", padx=(0, 4))
            val_lbl = tb.Label(row, text=fmt.format(var.get()), width=4, font=(FONT_FAMILY, 8, "bold"))
            val_lbl.grid(row=0, column=2, sticky="w")
            def _on_change(*_, v=var, lbl=val_lbl, f=fmt):
                lbl.config(text=f.format(v.get()))
                self._schedule_preview()
            var.trace_add("write", _on_change)

        _add_slider(self.preview_frame, "Hue start:", self.hue_start_var, 0, 360, "{:.0f}")
        _add_slider(self.preview_frame, "Hue end:", self.hue_end_var, 0, 360, "{:.0f}")
        _add_slider(self.preview_frame, "Saturation:", self.saturation_var, 0, 100, "{}%")
        _add_slider(self.preview_frame, "Brightness:", self.brightness_var, 0, 100, "{}%")

        # File selection
        tb.Label(mid, text="File:", font=(FONT_FAMILY, 9)).pack(anchor="w")
        self.file_row = tb.Frame(mid)
        self.file_row.pack(fill="x", pady=(0, 4))
        self.file_row.grid_columnconfigure(0, weight=1)
        self.file_entry = tb.Entry(self.file_row, textvariable=self.file_path_var)
        self.file_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        tb.Button(self.file_row, text="Browse...", command=self._browse_file,
                  bootstyle="secondary-outline", width=9).grid(row=0, column=1)
        self.add_btn = tb.Button(self.file_row, text="Add to Queue", command=self._add_to_queue,
                                  bootstyle="primary", width=14)
        self.add_btn.grid(row=0, column=2, padx=(4, 0))
        self.clear_input_btn = tb.Button(self.file_row, text="Clear Input", command=self._clear_input,
                                          bootstyle="secondary-outline", width=12)
        self.clear_input_btn.grid(row=0, column=3, padx=(4, 0))

        # Main text area for URLs / solutions
        tb.Label(mid, text="URLs / Solution strings (one per line):",
                 font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", pady=(4, 0))
        self.input_text = scrolledtext.ScrolledText(
            mid, font=(FONT_MONO_FAMILY, 10), height=6,
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="#fff",
            relief="flat", borderwidth=0, highlightthickness=1,
            highlightbackground="#3a3a3a", highlightcolor="#3a3a3a")
        self.input_text.pack(fill="x", pady=(2, 0))
        _setup_placeholder(self.input_text,
                           "# paste URLs or solution strings here, one per line")

        # Override params (always visible, always editable) — inline
        self.ov_frame = tb.LabelFrame(mid, text="Override (optional — applied on add to queue)",
                                 font=(FONT_FAMILY, 9, "bold"))
        self.ov_frame.pack(fill="x", pady=(4, 0))

        ov_row1 = tb.Frame(self.ov_frame)
        ov_row1.pack(fill="x", padx=4, pady=(2, 1))
        def _ov_inline(parent, label, var, w, col):
            tb.Label(parent, text=label, font=(FONT_FAMILY, 9)).grid(row=0, column=col*2, padx=(2, 2))
            e = tb.Entry(parent, width=w, textvariable=var)
            e.grid(row=0, column=col*2+1, padx=(0, 6))
            return e
        _ov_inline(ov_row1, "TPS:", self.tps_var, 8, 0)
        _ov_inline(ov_row1, "Time (s):", self.time_var, 8, 1)
        _ov_inline(ov_row1, "Size:", self.size_var, 8, 2)

        ov_row2 = tb.Frame(self.ov_frame)
        ov_row2.pack(fill="x", padx=4, pady=(0, 2))
        tb.Label(ov_row2, text="Scramble:", font=(FONT_FAMILY, 9)).pack(side="left", padx=(2, 2))
        self.scramble_entry = tb.Entry(ov_row2, width=18, textvariable=self.scramble_var)
        self.scramble_entry.pack(side="left", padx=(0, 6))
        tb.Label(ov_row2, text="Movetimes:", font=(FONT_FAMILY, 9)).pack(side="left", padx=(2, 2))
        self.movetimes_entry = tb.Entry(ov_row2, width=18, textvariable=self.movetimes_var)
        self.movetimes_entry.pack(side="left")

        # ======== COLUMN 2: PROGRESS + QUEUE + OUTPUTS ========
        right = tb.Frame(root)
        right.grid(row=0, column=2, sticky="nsew", padx=(3, 0))
        right.grid_rowconfigure(1, weight=0)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # ── Progress (overall batch bar) ──
        prog_frame = tb.LabelFrame(right, text="Progress")
        prog_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        prog_frame.grid_columnconfigure(0, weight=1)

        tb.Label(prog_frame, text="Overall:", font=(FONT_FAMILY, 9)).grid(
            row=0, column=0, sticky="w", padx=(8, 0), pady=(4, 0))
        self.overall_bar = tb.Progressbar(prog_frame, mode="determinate",
                                           bootstyle="success-striped")
        self.overall_bar.grid(row=1, column=0, sticky="ew", padx=8, pady=(1, 0))
        ov_lbl = tb.Label(prog_frame, textvariable=self.pb_overall_text,
                          font=(FONT_FAMILY, 9), anchor="w")
        ov_lbl.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 2))

        tb.Label(prog_frame, text="Current:", font=(FONT_FAMILY, 9)).grid(
            row=3, column=0, sticky="w", padx=(8, 0), pady=(4, 0))
        self.detail_bar = tb.Progressbar(prog_frame, mode="determinate",
                                          bootstyle="info-striped")
        self.detail_bar.grid(row=4, column=0, sticky="ew", padx=8, pady=(1, 0))
        dt_lbl = tb.Label(prog_frame, textvariable=self.pb_detail_text,
                          font=(FONT_FAMILY, 9), anchor="w")
        dt_lbl.grid(row=5, column=0, sticky="ew", padx=8, pady=(0, 4))

        gpu_lbl = tb.Label(prog_frame, textvariable=self._gpu_info_var,
                           font=(FONT_FAMILY, 8), anchor="w", foreground="#aaaaaa")
        gpu_lbl.grid(row=6, column=0, sticky="ew", padx=8, pady=(0, 4))

        # ── Render Queue (collapsible — row 1, fixed height) ──
        q_frame = tb.LabelFrame(right, text="Render Queue")
        q_frame.grid(row=1, column=0, sticky="ew", pady=(0, 4))
        q_frame.grid_columnconfigure(0, weight=1)
        q_frame.grid_columnconfigure(1, weight=0)

        self.queue_listbox = tk.Listbox(
            q_frame, font=(FONT_MONO_FAMILY, 8), activestyle="none",
            selectbackground="#2a6d9c", selectforeground="white",
            height=6,
            bg="#1a1a1a", fg="#cccccc", relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground="#333")
        self.queue_listbox.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))

        q_scroll = tb.Scrollbar(q_frame, orient="vertical", command=self.queue_listbox.yview)
        q_scroll.grid(row=0, column=1, sticky="ns", pady=(6, 2))
        self.queue_listbox.configure(yscrollcommand=q_scroll.set)

        q_actions = tb.Frame(q_frame)
        q_actions.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 2))
        tb.Button(q_actions, text="Remove Selected", command=self._remove_selected_from_queue,
                  bootstyle="secondary-outline", width=16).pack(side="left", padx=(0, 4))
        tb.Button(q_actions, text="Clear Queue", command=self._clear_queue,
                  bootstyle="secondary-outline", width=12).pack(side="left")

        self.queue_count_var = tk.StringVar(value="0 items")
        q_count = tb.Label(q_actions, textvariable=self.queue_count_var,
                           font=(FONT_FAMILY, 8), foreground="#888")
        q_count.pack(side="right")

        q_render = tb.Frame(q_frame)
        q_render.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
        self.gen_btn = tb.Button(q_render, text="Render", command=self._generate,
                                 bootstyle="success", width=12)
        self.gen_btn.pack(side="left", padx=(0, 6))
        self.cancel_btn = tb.Button(q_render, text="Cancel", command=self._cancel,
                                    bootstyle="secondary", state="disabled")
        self.cancel_btn.pack(side="left")

        # ── Generated replays ──
        lst_frame = tb.LabelFrame(right, text="Generated Replays")
        lst_frame.grid(row=2, column=0, sticky="nsew")
        lst_frame.grid_columnconfigure(0, weight=1)
        lst_frame.grid_rowconfigure(0, weight=1)

        self.replay_listbox = tk.Listbox(
            lst_frame, font=(FONT_MONO_FAMILY, 8), activestyle="none",
            selectbackground="#2a6d9c", selectforeground="white",
            bg="#1a1a1a", fg="#cccccc", relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground="#333")
        self.replay_listbox.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 2))
        self.replay_listbox.bind("<Double-Button-1>", self._open_selected)

        scroll = tb.Scrollbar(lst_frame, orient="vertical", command=self.replay_listbox.yview)
        scroll.grid(row=0, column=1, sticky="ns", pady=(6, 2))
        self.replay_listbox.configure(yscrollcommand=scroll.set)

        lst_actions = tb.Frame(lst_frame)
        lst_actions.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
        tb.Button(lst_actions, text="Open", command=lambda: self._open_selected(),
                  bootstyle="info-outline", width=8).pack(side="left", padx=(0, 4))
        tb.Button(lst_actions, text="Folder", command=self._open_folder,
                  bootstyle="secondary-outline", width=8).pack(side="left", padx=(0, 4))
        tb.Button(lst_actions, text="Clear", command=self._clear_list,
                  bootstyle="secondary-outline", width=8).pack(side="left")

        # ── Trigger preview on any relevant change ──
        self.main_scheme_var.trace_add("write", lambda *_: self._schedule_preview())
        self.size_var.trace_add("write", lambda *_: self._schedule_preview())
        self.queue_listbox.bind("<<ListboxSelect>>", lambda e: self._schedule_preview())
        for name in ("grid1", "grid2", "tile_bg"):
            self._color_vars[name].trace_add("write", lambda *_: self._schedule_preview())
        for v in (self.no_border_var, self.no_numbers_var,
                  self.no_grid_bars_var, self.no_secondary_border_var):
            v.trace_add("write", lambda *_: self._schedule_preview())
        self.force_main_var.trace_add("write", lambda *_: self._schedule_preview())

        # Initial preview
        self.after(100, self._render_preview)
        self.bind("<Configure>", self._on_window_configure)


    def _on_window_configure(self, event):
        if event.widget is not self:
            return
        state = self.state()
        prev = getattr(self, '_prev_win_state', None)
        self._prev_win_state = state
        if state == "zoomed":
            if prev != "zoomed":
                self._prev_root_h = event.height
                self._schedule_preview()
        elif state == "iconic":
            pass
        elif prev in ("zoomed", "iconic"):
            self._prev_root_h = event.height
            self._schedule_preview()
        elif (event.width, event.height) != (self._base_w, self._base_h):
            self.geometry(f"{self._base_w}x{self._base_h}")

    def _get_speed_factor(self) -> float:
        try:
            return float(self.speed_factor_var.get().strip() or "1.0")
        except ValueError:
            return 1.0

    def _get_quality(self) -> int:
        return self._quality_presets.get(self.quality_preset_var.get(), 1080)

    def _update_quality_warning(self):
        h = self._get_quality()
        text = ""
        if h >= 2160:
            text = "⚠ 4K uses very high RAM during atlas prerender"
        elif h >= 1440:
            text = "⚠ 2K uses high RAM during atlas prerender"
        if text:
            self.quality_warning.config(text=text)
            self.quality_warning.grid()
        else:
            self.quality_warning.grid_remove()

    def _center_window(self):
        self.update_idletasks()
        w, h = 1280, 720
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _browse_output(self):
        path = filedialog.askdirectory(title="Output folder")
        if path:
            self.out_folder_var.set(path)

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select input file",
            filetypes=[("All files", "*.*")]
        )
        if path:
            self.file_path_var.set(path)


    def _schedule_preview(self):
        if self._preview_job:
            self.after_cancel(self._preview_job)
        self._preview_job = self.after(30, self._render_preview)

    def _render_preview(self):
        try:
            w = h = None
            item = None
            sel = self.queue_listbox.curselection()
            if sel and sel[0] < len(self.render_queue):
                item = self.render_queue[sel[0]]
                self._preview_sel_idx = sel[0]
            elif self._preview_sel_idx >= 0 and self._preview_sel_idx < len(self.render_queue):
                item = self.render_queue[self._preview_sel_idx]

            if item is None and not self.size_var.get().strip():
                if self._preview_sel_idx >= 0:
                    return
                w = h = 4
                info_text = "4x4 · Fringe"
                sat = self.saturation_var.get() / 100.0
                light = self.brightness_var.get() / 100.0
                hue_start = self.hue_start_var.get()
                hue_end = self.hue_end_var.get()
                scheme = self.main_scheme_var.get()
                grid_data = {"enableGridsStatus": -1, "width": 4, "height": 4, "offsetW": 0, "offsetH": 0}
                grid_states = generate_grids_stats(grid_data)
                first_state = list(grid_states.values())[0]
                matrix = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
            elif item is not None:
                sol = item.get("solution", "")
                sc = item.get("scramble")
                from replay_generator import expand_solution, scramble_to_puzzle, create_puzzle as cp
                from grids_analysis import analyse_grids_initial

                if sc:
                    init_matrix = scramble_to_puzzle(sc)
                    h = len(init_matrix)
                    w = len(init_matrix[0])
                else:
                    sz = _quick_infer_size(
                        item.get("solution", ""),
                        item.get("scramble"),
                        item.get("size"),
                    )
                    if sz:
                        w, h = sz
                    if w is None:
                        w = h = 4
                    init_matrix = cp(w, h)
                    if sol:
                        from replay_generator import reverse_solution, apply_moves
                        init_matrix = apply_moves(init_matrix, reverse_solution(expand_solution(sol)))

                if w > 16 or h > 16:
                    info_text = f"{w}x{h} · {self.main_scheme_var.get().capitalize()} ({w}x{h} puzzle too large for a dynamic preview)"
                    w = h = 4
                    sat = self.saturation_var.get() / 100.0
                    light = self.brightness_var.get() / 100.0
                    hue_start = self.hue_start_var.get()
                    hue_end = self.hue_end_var.get()
                    scheme = self.main_scheme_var.get()
                    grid_data = {"enableGridsStatus": -1, "width": w, "height": h, "offsetW": 0, "offsetH": 0}
                    grid_states = generate_grids_stats(grid_data)
                    first_state = list(grid_states.values())[0]
                    matrix = [[r * w + c + 1 for c in range(w)] for r in range(h)]
                    matrix[h-1][w-1] = 0
                else:
                    info_text = f"{w}x{h} · {self.main_scheme_var.get().capitalize()}"
                    sat = self.saturation_var.get() / 100.0
                    light = self.brightness_var.get() / 100.0
                    hue_start = self.hue_start_var.get()
                    hue_end = self.hue_end_var.get()
                    scheme = self.main_scheme_var.get()

                    exp_sol = expand_solution(sol)
                    if self.force_main_var.get():
                        grids_data = {"enableGridsStatus": -1, "width": w, "height": h, "offsetW": 0, "offsetH": 0}
                    else:
                        grids_data = analyse_grids_initial(init_matrix, exp_sol, cycles_detection=True)
                    grid_states = generate_grids_stats(grids_data)

                    keys = sorted([k for k in grid_states.keys() if isinstance(k, (int, float))])
                    first_state = grid_states[keys[0]] if keys else list(grid_states.values())[0]
                    matrix = cp(w, h)
            else:
                raw = self.size_var.get().strip()
                if raw and 'x' in raw:
                    parts = raw.lower().split('x')
                    w = int(parts[0]); h = int(parts[1])
                if w is None:
                    w = h = 4
                if w > 16 or h > 16:
                    info_text = f"{w}x{h} · {self.main_scheme_var.get().capitalize()} ({w}x{h} puzzle too large for a dynamic preview)"
                    w = h = 4
                else:
                    info_text = f"{w}x{h} · {self.main_scheme_var.get().capitalize()}"

                sat = self.saturation_var.get() / 100.0
                light = self.brightness_var.get() / 100.0
                hue_start = self.hue_start_var.get()
                hue_end = self.hue_end_var.get()
                scheme = self.main_scheme_var.get()

                grid_data = {"enableGridsStatus": -1, "width": w, "height": h, "offsetW": 0, "offsetH": 0}
                grid_states = generate_grids_stats(grid_data)
                first_state = list(grid_states.values())[0]
                matrix = [[r * w + c + 1 for c in range(w)] for r in range(h)]
                matrix[h-1][w-1] = 0

            all_fringe_schemes = get_all_fringe_schemes(grid_states, scheme, sat, light, hue_start, hue_end)

            opts = RenderOptions(
                grid_only=True,
                no_border=self.no_border_var.get(),
                no_numbers=self.no_numbers_var.get(),
                no_grid_bars=self.no_grid_bars_var.get(),
                no_secondary_border=self.no_secondary_border_var.get(),
                tile_bg_color=parse_hex_color(self._color_vars["tile_bg"].get()),
                grid1_color=parse_hex_color(self._color_vars["grid1"].get()),
                grid2_color=parse_hex_color(self._color_vars["grid2"].get()),
            )

            try:
                self.update_idletasks()
                avail_w = self.preview_frame.winfo_width() - 20
                total_h = self._prev_root_h if self._prev_root_h > 0 else self.winfo_height()
                file_row_h = max(1, self.file_row.winfo_height())
                text_h = max(1, self.input_text.winfo_height())
                ov_h = max(1, self.ov_frame.winfo_height())
                other_h = 210
                avail_h = max(50, total_h - file_row_h - text_h - ov_h - other_h)
                avail = max(50, min(avail_w, avail_h))
            except Exception:
                avail = 200
            tile_size = max(1, avail // max(w, h))
            font_size = max(1, tile_size // 3)

            stats_data = {
                "moves": [], "current_time": 0,
                "total_moves": 0, "total_time_ms": 0, "total_tps": 0,
                "is_movetimes_accurate": False,
                "score_title": "", "timer_text": "",
            }

            img = render_frame(
                matrix=matrix,
                grid_state=first_state,
                all_fringe_schemes=all_fringe_schemes,
                tile_size=tile_size,
                font_size=font_size,
                stats_data=stats_data,
                score_title_text="", timer_text="",
                is_movetimes_accurate=False,
                total_moves=0, total_time_ms=0, total_tps=0,
                opts=opts,
            )

            img.thumbnail((avail, avail), Image.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(img)
            self._preview_label.config(image=self._preview_photo)
            self._preview_info.config(text=info_text)
        except Exception as e:
            log.warning(f"Preview update failed: {e}")


    def _add_to_queue(self):
        """Parse input text and overrides, add entries to render queue."""
        raw = self.input_text.get("1.0", "end-1c").strip()

        # Capture current override values
        override_tps = self.tps_var.get().strip()
        override_time = self.time_var.get().strip()
        override_size = self.size_var.get().strip()
        override_scramble = self.scramble_var.get().strip()
        override_movetimes = self.movetimes_var.get().strip()

        # Also read file if set
        file_path = self.file_path_var.get().strip()
        file_raw = None
        if file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                file_raw = f.read().strip()

        count = 0
        # Process file content (treat each line as a separate input)
        if file_raw:
            for fline in file_raw.splitlines():
                fline = fline.strip()
                if not fline or fline.startswith("#"):
                    continue
                new_entries = self._parse_input_to_entries(
                    fline, override_tps, override_time, override_size,
                    override_scramble, override_movetimes)
                self.render_queue.extend(new_entries)
                count += len(new_entries)

        # Process each line from the text area
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            new_entries = self._parse_input_to_entries(
                line, override_tps, override_time, override_size,
                override_scramble, override_movetimes)
            self.render_queue.extend(new_entries)
            count += len(new_entries)

        if count:
            self._refresh_queue_display()
            last_idx = len(self.render_queue) - 1
            self.queue_listbox.selection_clear(0, "end")
            self.queue_listbox.selection_set(last_idx)
            self.queue_listbox.activate(last_idx)
            self._schedule_preview()
            self.input_text.delete("1.0", "end")

    def _parse_input_to_entries(self, text, override_tps, override_time,
                                 override_size, override_scramble, override_movetimes):
        """Parse a single input string (URL or solution) and return queue entry dicts."""
        if text.startswith(("http://", "https://")):
            try:
                solution, tps, scramble, movetimes = parse_replay_url(text)
            except Exception as e:
                self.after(0, lambda m=f"Parse error for URL: {e}": self.pb_overall_text.set(m))
                return []
            # Apply overrides on top of URL-parsed values
            if override_tps:
                tps = float(override_tps)
            if override_scramble:
                scramble = override_scramble
            if override_movetimes:
                movetimes = [int(x.strip()) for x in override_movetimes.split(",")]
            mode = "url"
        else:
            solution = text
            tps = float(override_tps) if override_tps else None
            scramble = override_scramble if override_scramble else None
            movetimes = [int(x.strip()) for x in override_movetimes.split(",")] if override_movetimes else -1
            mode = "manual"

        time_v = float(override_time) if override_time else None
        size_s = override_size if override_size else None

        if time_v and tps:
            tps = None

        if (override_tps or override_time) and isinstance(movetimes, list):
            movetimes = -1

        moves = count_moves(solution)
        if time_v and tps is None:
            tps = moves / time_v
        size = _quick_infer_size(solution, scramble, size_s)

        # Build display name (like filename would be)
        if isinstance(movetimes, list) and len(movetimes) > 1 and movetimes[-1] > 0:
            time_s = movetimes[-1] / 1000.0
            is_movetimes_accurate = True
            raw_tps = float(override_tps) if override_tps else (moves / time_s)
        elif time_v and time_v > 0:
            time_s = time_v
            is_movetimes_accurate = False
            raw_tps = float(override_tps) if override_tps else (moves / time_s)
        elif tps and tps > 0:
            time_s = moves / tps
            is_movetimes_accurate = False
            raw_tps = tps
        else:
            time_s = 0
            is_movetimes_accurate = False
            raw_tps = None

        size_str = f"{size[0]}x{size[1]}" if size and size[0] and size[1] else "?"
        parts = [size_str]
        if time_s:
            parts.append(f"{time_s:.3f}")
        parts.append(str(moves))
        if raw_tps:
            parts.append(f"{raw_tps:.3f}")
        if is_movetimes_accurate:
            parts.append("movetimes")
        display_name = "_".join(parts)

        return [{
            "mode": mode,
            "input_str": text,
            "solution": solution,
            "tps": tps,
            "time": time_v if time_v else 0,
            "size": size_s,
            "scramble": scramble,
            "movetimes": movetimes if isinstance(movetimes, list) else -1,
            "display_name": display_name,
            "tps_was_overridden": bool(override_tps or override_time),
        }]

    def _remove_selected_from_queue(self):
        sel = self.queue_listbox.curselection()
        if not sel:
            return
        for i in reversed(sel):
            del self.render_queue[i]
        self._refresh_queue_display()

    def _clear_queue(self):
        self.render_queue.clear()
        self._refresh_queue_display()

    def _clear_input(self):
        self.input_text.delete("1.0", "end")

    def _refresh_queue_display(self):
        self.queue_listbox.delete(0, "end")
        for item in self.render_queue:
            self.queue_listbox.insert("end", item["display_name"])
        n = len(self.render_queue)
        self.queue_count_var.set(f"{n} item{'s' if n != 1 else ''}")

    def _generate(self):
        if not self.render_queue:
            self.pb_overall_text.set("Queue is empty. Add items first.")
            return
        if self._executor and not all(f.done() for f in self._batch_futures):
            log.info("_generate: skipped — previous render still running")
            return

        self.cancel_flag = False
        self._set_ui_busy(True)
        self._start_time = time.time()
        self._last_poll_time = time.time()
        self._last_poll_pct = 0.0
        self._rolling_rate = 0.0
        self.overall_bar["value"] = 0
        self.detail_bar["value"] = 0
        self.pb_overall_text.set("0 / 0")
        self.pb_detail_text.set("")
        self._gpu_info_var.set("")
        self.replay_listbox.delete(0, "end")
        self.generated_files.clear()

        raw_folder = self.out_folder_var.get().strip()
        output_dir = os.path.abspath(raw_folder) if raw_folder else os.path.join(script_dir, "replays")
        os.makedirs(output_dir, exist_ok=True)

        total = len(self.render_queue)
        self._item_progress = {}
        for idx in range(total):
            self._item_progress[idx] = {"adjusted_cur": 0, "adjusted_tot": 1,
                                        "done": False, "path": None, "error": None,
                                        "cancelled": False}

        self._batch_futures = []
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._current_item_idx = -1
        fut = self._executor.submit(self._process_queue, output_dir)
        self._batch_futures = [fut]
        self.after(500, self._poll_queue)

    def _process_queue(self, output_dir):
        """Iterate all queued items and render each sequentially."""
        import copy
        total = len(self.render_queue)
        gen = ReplayVideoGenerator(cleanup_frames=False)

        for idx, entry in enumerate(self.render_queue):
            if self.cancel_flag:
                raise CancelError()

            self._current_item_idx = idx
            log.info(f"_process_queue[{idx}]: {entry['display_name']}")

            try:
                # Build params from current GUI settings + queue entry overrides
                from geometry import parse_hex_color
                opts = RenderOptions(
                    grid_only=self.no_layout_var.get(),
                    no_border=self.no_border_var.get(),
                    no_secondary_border=self.no_secondary_border_var.get(),
                    no_grid_bars=self.no_grid_bars_var.get(),
                    no_numbers=self.no_numbers_var.get(),
                    no_header=self.no_header_var.get(),
                    no_details=self.no_details_var.get(),
                    dynamic_md=self.dynamic_md_var.get(),
                    cycles_detection=self.cycles_detection_var.get(),
                    adjust_height=self.adjust_height_var.get(),
                    grid1_color=parse_hex_color(self._color_vars["grid1"].get()),
                    grid2_color=parse_hex_color(self._color_vars["grid2"].get()),
                    tile_bg_color=parse_hex_color(self._color_vars["tile_bg"].get()),
                    animate_moves=self.animate_moves_var.get(),
                    hue_start=self.hue_start_var.get(),
                    hue_end=self.hue_end_var.get(),
                    saturation=self.saturation_var.get() / 100.0,
                    brightness=self.brightness_var.get() / 100.0,
                )
                params = {
                    "main_scheme": self.main_scheme_var.get(),
                    "force_main": self.force_main_var.get(),
                    "quality": self._get_quality(),
                    "fps": self.fps_var.get(),
                    "compression": self.compression_var.get(),
                    "slow_render": self.slow_render_var.get(),
                    "speed_factor": self._get_speed_factor(),
                    "upscale": self.upscale_var.get(),
                    "encoder_override": "" if self.encoder_var.get() == "Auto" else self.encoder_var.get(),
                    "opts": opts,
                }

                solution = entry["solution"]
                if entry["tps"] is not None:
                    params["tps"] = entry["tps"]
                if entry["scramble"]:
                    params["scramble"] = entry["scramble"]
                if isinstance(entry["movetimes"], list) and len(entry["movetimes"]) > 0:
                    params["movetimes"] = entry["movetimes"]
                if entry["time"]:
                    params["time"] = entry["time"]
                if entry["size"]:
                    params["size"] = entry["size"]

                if isinstance(entry["movetimes"], list) and len(entry["movetimes"]) > 1:
                    tps_from_movetimes = count_moves(solution) / (entry["movetimes"][-1] / 1000.0) if entry["movetimes"][-1] > 0 else None
                    filename_tps = entry["tps"] if entry.get("tps_was_overridden") else (tps_from_movetimes or entry["tps"])
                else:
                    filename_tps = entry["tps"]
                filename_time = entry["time"] if entry["time"] else None
                base_name = _generate_filename(
                    solution, filename_tps, filename_time,
                    entry["movetimes"], entry["size"],
                    index=idx + 1, speed_factor=self._get_speed_factor(),
                    scramble=entry["scramble"] if entry["scramble"] else None)
                out_path = _pick_output_filename(output_dir, base_name)

                def on_progress(adjusted_cur, adjusted_tot, desc=None, gpu_stats=None, use_gpu=False):
                    self._on_item_progress(idx, adjusted_cur, adjusted_tot,
                                           desc=desc, gpu_stats=gpu_stats, use_gpu=use_gpu)

                log.info(f"_process_queue[{idx}]: rendering {entry['display_name']}")
                gen.generate_simple_replay(
                    solution=solution, output_path=out_path,
                    show_progress=False, external_progress_cb=on_progress,
                    use_gpu=self.use_gpu_var.get(),
                    cancel_check=lambda: self.cancel_flag, **params)
                log.info(f"_process_queue[{idx}]: completed")

                if not self.cancel_flag:
                    self._item_progress[idx]["path"] = out_path
                    self.after(0, lambda p=out_path: self._add_to_list(p))
                    if self.upscale_var.get() and self._get_quality() < 1440:
                        stem, ext = os.path.splitext(out_path)
                        upscaled_path = f"{stem}_1440p60{ext}"
                        self.after(0, lambda p=upscaled_path: os.path.exists(p) and self._add_to_list(p))
            except CancelError:
                self._item_progress[idx]["cancelled"] = True
                raise
            except Exception as e:
                log.error(f"_process_queue[{idx}]: FAILED: {e}", exc_info=True)
                self._item_progress[idx]["error"] = str(e)

        # All items processed
        log.info("_process_queue: all items done")

    @staticmethod
    def _time_str(t: float) -> str:
        total_sec = int(round(t))
        if total_sec >= 3600:
            return f"{total_sec // 3600}h{total_sec % 3600 // 60}m"
        if total_sec >= 60:
            return f"{total_sec // 60}m{total_sec % 60}s"
        return f"{t:.1f}s"

    def _on_item_progress(self, idx, adjusted_cur, adjusted_tot, desc=None, gpu_stats=None, use_gpu=False):
        item = self._item_progress[idx]
        if "start_time" not in item:
            item["start_time"] = time.time()
            item["last_poll_cur"] = adjusted_cur
            item["last_poll_time"] = item["start_time"]
        item["adjusted_cur"] = adjusted_cur
        item["adjusted_tot"] = adjusted_tot
        if gpu_stats:
            item["gpu_stats"] = gpu_stats
        if use_gpu:
            item["is_gpu"] = True
        if desc:
            item["desc"] = desc

    def _cancel(self):
        self.cancel_flag = True
        if self._executor:
            self._executor.shutdown(wait=False)
        self.pb_overall_text.set("Cancelling...")

    def _poll_queue(self):
        if not self._batch_futures:
            return
        fut = self._batch_futures[0]
        total = len(self.render_queue)

        if not fut.done():
            # Update overall bar
            done_count = sum(1 for p in self._item_progress.values() if p.get("path"))
            elapsed = time.time() - self._start_time

            overall_pct = (done_count * 100) // total if total else 0
            if overall_pct > 0 and not self.cancel_flag:
                self.overall_bar["value"] = overall_pct

            # Simple ETA: avg time per completed item × remaining items
            remaining = total - done_count
            total_expected = elapsed
            if done_count > 0 and remaining > 0:
                avg_per_item = elapsed / done_count
                total_expected = avg_per_item * total
            self.pb_overall_text.set(
                f"{done_count}/{total} — {self._time_str(elapsed)}/{self._time_str(total_expected)}"
            )

            # Update detail bar from current item
            cur_idx = self._current_item_idx
            if cur_idx is not None and cur_idx >= 0 and cur_idx < total:
                p = self._item_progress[cur_idx]
                cur = p["adjusted_cur"]
                tot = p["adjusted_tot"]
                detail_pct = 0
                if tot > 0:
                    detail_pct = min(cur * 100 // tot, 100)
                    self.detail_bar["value"] = detail_pct
                desc = p.get("desc", "")
                gs = p.get("gpu_stats")
                detail_parts = []
                if desc:
                    detail_parts.append(f"{desc}:")
                detail_parts.append(f"{detail_pct}%")

                st = p.get("start_time")
                if st and tot > 0 and cur > 0:
                    item_elapsed = time.time() - st
                    rate = cur / item_elapsed if item_elapsed > 0 else 0
                    remaining = tot - cur
                    eta = remaining / rate if rate > 0 else 0
                    total_expected = item_elapsed + eta
                    detail_parts.append(f"{rate:.0f}/s")
                    detail_parts.append(f"{self._time_str(item_elapsed)}/{self._time_str(total_expected) if total_expected < 1e8 else '?'}")

                if gs and gs.get("batch_size"):
                    detail_parts.append(f"GPU: {gs.get('mem_used_mb', 0)}/{gs.get('total_mem_mb', 0)} MB")
                self.pb_detail_text.set(" | ".join(detail_parts))

                gpu_str = ""
                gs = p.get("gpu_stats")
                if gs and gs.get("batch_size"):
                    gpu_str = f"GPU: {gs.get('gpu_name', '?')} | VRAM: {gs.get('mem_used_mb', 0)}/{gs.get('total_mem_mb', 0)} MB | Batch: {gs.get('batch_size', 0)} frames"
                self._gpu_info_var.set(gpu_str)

            self.after(500, self._poll_queue)
            return

        # Future done — final summary
        elapsed = time.time() - self._start_time
        elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed//60)}m {elapsed%60:.0f}s"
        try:
            fut.result()
            done_count = sum(1 for p in self._item_progress.values() if p.get("path"))
            errors = sum(1 for p in self._item_progress.values() if p["error"])
            cancelled = sum(1 for p in self._item_progress.values() if p.get("cancelled"))
            ok_count = done_count - errors - cancelled
            parts = []
            if ok_count:
                parts.append(f"{ok_count} replay(s) generated")
            if cancelled:
                parts.append(f"{cancelled} cancelled")
            if errors:
                parts.append(f"{errors} failed")
            msg = " — ".join(parts) + f". {elapsed_str}" if parts else f"Done. {elapsed_str}"
            self.pb_overall_text.set(msg)
            # Clear queue only on successful render
            self.render_queue.clear()
            self._refresh_queue_display()
        except CancelError:
            done_count = sum(1 for p in self._item_progress.values() if p.get("path"))
            self.pb_overall_text.set(f"Cancelled. {done_count} completed. {elapsed_str}")
        except Exception as e:
            done_count = sum(1 for p in self._item_progress.values() if p.get("path"))
            self.pb_overall_text.set(f"Render failed: {e}. {done_count} completed. {elapsed_str}")

        self.overall_bar["value"] = 100
        self.detail_bar["value"] = 100
        self._gpu_info_var.set("")
        self._set_ui_busy(False)
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._batch_futures = []
        self._current_item_idx = -1

    def _add_to_list(self, path):
        self.generated_files.append(path)
        self.replay_listbox.insert("end", os.path.basename(path))

    def _open_selected(self, event=None):
        sel = self.replay_listbox.curselection()
        if sel and sel[0] < len(self.generated_files):
            path = self.generated_files[sel[0]]
            if os.path.exists(path):
                _open(path, self.progress_text.set)
            else:
                self.progress_text.set(f"File not found: {path}")

    def _open_folder(self):
        if self.generated_files:
            _open(os.path.dirname(self.generated_files[-1]), self.progress_text.set)

    def _clear_list(self):
        self.generated_files.clear()
        self.replay_listbox.delete(0, "end")

    def _set_ui_busy(self, busy):
        st = "disabled" if busy else "normal"
        self.gen_btn.config(state=st)
        self.add_btn.config(state=st)
        self.clear_input_btn.config(state=st)
        self.cancel_btn.config(state="normal" if busy else "disabled")

    def _on_close(self):
        log.info("=== GUI CLOSING ===")
        self.cancel_flag = True
        if self._executor:
            self._executor.shutdown(wait=False)
        self.destroy()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Strip multiprocessing internal args before argparse sees them
    import sys as _sys
    _sys.argv = [a for a in _sys.argv if not a.startswith('--multiprocessing-')]

    import argparse

    parser = argparse.ArgumentParser(
        description="Sliding Puzzle Replay Video Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --solution R2D2L2U2 --size 3x3 --tps 10 -o replay.mp4
  python main.py -u "https://slidysim.github.io/?replay=..." -o replay.mp4
  python main.py -b urls.txt
  python main.py                    # launch GUI
        """
    )
    parser.add_argument("--solution", help="Solution string (e.g. R2D2L2U2)")
    parser.add_argument("--url", "-u", help="Slidysim replay URL")
    parser.add_argument("--file", "-f", help="File containing a replay URL or solution string (bypasses CLI length limit)")
    parser.add_argument("--tps", type=float, help="Tiles per second")
    parser.add_argument("--time", type=float, help="Total time in seconds")
    parser.add_argument("--size", help="Puzzle size (e.g. 3x3, 5x5)")
    parser.add_argument("--scramble", help="Scramble string")
    parser.add_argument("--output", "-o", default=None, help="Output file path (default: auto-generated name in replays/ folder)")
    parser.add_argument("--quality", "-q", type=int, default=1080, help="Target video quality (720, 1080, 1440, 2160)")
    parser.add_argument("--compression", "-c", type=int, default=18, help="Video encoder quality (10-40, lower = fewer artifacts but larger file, default: 18)")
    parser.add_argument("--slow-render", action="store_true", default=False, help="Slower encode, ~33%% smaller file (p7 for NVENC, slow for libx264)")
    parser.add_argument("--encoder-preset", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--fps", type=int, default=60, help="Output video frame rate (default: 60)")
    parser.add_argument("--no-gpu", "-g", action="store_true", default=None,
                        help="Disable GPU acceleration")
    parser.add_argument("--batch", "-b", help="File with solutions/URLs (one per line)")
    parser.add_argument("--movetimes", help="Comma-separated move timings (overrides --tps/--time)")
    parser.add_argument("--speedup", "-s", type=float, default=1.0,
                        help="Speed multiplier (e.g. 2.0 = 2x faster video, 0.5 = half speed)")
    parser.add_argument("--force-main", action="store_true", default=False,
                        help="Force main scheme everywhere (disable grids detection)")
    parser.add_argument("--main-scheme", type=str, default='fringe', choices=['fringe', 'rows', 'columns'],
                        help="Color scheme: fringe, rows, or columns (default: fringe)")
    parser.add_argument("--force-fringe", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--force-rows", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--force-columns", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--log", "-l", action="store_true", default=False,
                        help="Enable debug logging to file (logs/debug_<timestamp>.log)")
    parser.add_argument("--no-layout", action="store_true",
                        help="Only render the puzzle grid on dark background — no timer bar, no stats panel (shortcut for --no-header --no-details)")
    parser.add_argument("--grid-only", action="store_true", dest="no_layout",
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-header", action="store_true", default=False,
                        help="Hide the timer header bar (time/moves/tps and MD display)")
    parser.add_argument("--no-details", action="store_true", default=False,
                        help="Hide the stats panel on the right side of the puzzle")
    parser.add_argument("--dynamic-md", action="store_true", default=False,
                        help="Show MD/predicted/MMD timer on the right side of the header bar (disabled by default)")
    parser.add_argument("--cycles-detection", action="store_true", default=False,
                        help="EXPERIMENTAL: detect and display cycling tiles in grid stats (may increase analysis time)")
    parser.add_argument("--no-border", action="store_true",
                        help="Suppress tile border outlines")
    parser.add_argument("--no-secondary-border", action="store_true",
                        help="Suppress secondary color bar borders")
    parser.add_argument("--no-grid-bars", action="store_true", default=False,
                        help="Suppress secondary grid bar indicators inside tiles")
    parser.add_argument("--no-numbers", action="store_true",
                        help="Suppress tile number text")
    parser.add_argument("--upscale", action="store_true", default=False,
                        help="After rendering, upscale video to 2K (2560x1440) for best YouTube quality. "
                             "Only beneficial for qualities below 1440p. Keeps both original and upscaled versions.")
    parser.add_argument("--encoder", type=str, default="",
                        choices=["hevc_nvenc", "hevc_amf", "hevc_qsv", "libx265", "h264_nvenc", "h264_amf", "h264_qsv", "libx264"],
                        help="Force video encoder. Auto-detected from available hardware if not set.")
    parser.add_argument("--adjust-height", action="store_true", default=False,
                        help="Crop canvas height to puzzle content instead of fixed quality preset. "
                             "Aligns puzzle to the top (no centering) and removes bottom gap.")
    parser.add_argument("--animate-moves", action="store_true", default=False,
                        help="Animate tile sliding between moves (smooth transitions)")
    parser.add_argument("--grid1-color", type=str, default=None, help="Grid 1 color as hex (e.g. FF0000)")
    parser.add_argument("--grid2-color", type=str, default=None, help="Grid 2 color as hex (e.g. 0000FF)")
    parser.add_argument("--tile-bg-color", type=str, default=None, help="Tile background color as hex")
    parser.add_argument("--hue-start", type=float, default=0, help="Hue range start (0-360, default: 0)")
    parser.add_argument("--hue-end", type=float, default=360, help="Hue range end (0-360, default: 360)")
    parser.add_argument("--saturation", type=float, default=0.78, help="Color saturation (0-1, default: 0.78)")
    parser.add_argument("--brightness", type=float, default=0.6, help="Color brightness (0-1, default: 0.6)")

    args = parser.parse_args()

    main_scheme = args.main_scheme
    force_main = args.force_main
    if args.force_fringe:
        main_scheme = 'fringe'
        force_main = True
    if args.force_rows:
        main_scheme = 'rows'
        force_main = True
    if args.force_columns:
        main_scheme = 'columns'
        force_main = True

    from geometry import parse_hex_color
    opts = RenderOptions(
        grid_only=args.no_layout,
        no_border=args.no_border,
        no_secondary_border=args.no_secondary_border,
        no_grid_bars=args.no_grid_bars,
        no_numbers=args.no_numbers,
        no_header=args.no_header,
        no_details=args.no_details,
        dynamic_md=args.dynamic_md,
        cycles_detection=args.cycles_detection,
        adjust_height=args.adjust_height,
        grid1_color=parse_hex_color(args.grid1_color),
        grid2_color=parse_hex_color(args.grid2_color),
        tile_bg_color=parse_hex_color(args.tile_bg_color),
        animate_moves=args.animate_moves,
        hue_start=args.hue_start,
        hue_end=args.hue_end,
        saturation=args.saturation,
        brightness=args.brightness,
    )

    if args.speedup <= 0:
        parser.error("--speedup must be > 0")

    if args.quality < 720:
        parser.error("minimum quality is 720")

    if args.upscale and args.quality >= 1440:
        print("[Note] --upscale: source quality already >=1440p, no upscaling performed.", file=sys.stderr)

    movetimes = None
    if args.movetimes:
        movetimes = [float(x) for x in args.movetimes.split(",")]

    log_path = None
    if args.log:
        log_path = init_logfile()

    if not any([args.solution, args.url, args.file, args.batch]):
        if getattr(sys, 'frozen', False):
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
        gui = ReplayGUI()
        if log_path:
            log.info(f"=== GUI STARTED === log_path={log_path}")
        gui.mainloop()
        sys.exit(0)
    elif log_path:
        log.info(f"=== CLI STARTED === log_path={log_path}")

    use_gpu = True
    try:
        import torch
        torch_avail = torch.cuda.is_available()
    except Exception:
        torch_avail = False
    if args.no_gpu:
        use_gpu = False
    else:
        use_gpu = torch_avail

    slow_render = args.slow_render or not use_gpu

    if use_gpu and torch_avail:
        try:
            gpu_str = f"GPU ON ({torch.cuda.get_device_name(0)})"
        except Exception:
            gpu_str = "GPU ON (unknown)"
    else:
        gpu_str = "GPU OFF (CPU fallback)"
    print(f"[ReplayVideoGenerator] {gpu_str}")

    def run_single(solution, output, opts=RenderOptions(), **kwargs):
        try:
            gen = ReplayVideoGenerator(cleanup_frames=True)
            gen.generate_simple_replay(
                solution=solution, output_path=output,
                show_progress=True, use_gpu=use_gpu,
                fps=kwargs.pop("fps", 60), opts=opts, **kwargs
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[CRITICAL ERROR] {e}", file=sys.stderr)
            sys.exit(1)

    try:
        items = []
        if args.batch:
            with open(args.batch, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        items.append(("batch", line))
        elif args.file:
            with open(args.file, "r") as f:
                items.append(("url", f.read().strip()))
        elif args.url:
            items.append(("url", args.url))
        elif args.solution:
            items.append(("manual", args.solution))
        else:
            parser.print_help()
            sys.exit(1)

        replays_dir = os.path.join(script_dir, "replays")
        if args.output is None:
            os.makedirs(replays_dir, exist_ok=True)

        if len(items) > 1 and args.batch:
            # Batch mode: collect all items and render via batch_render
            batch_items = []
            for idx, item in enumerate(items):
                mode, val = item if isinstance(item, tuple) else ("manual", item)

                kwargs = dict(quality=args.quality, fps=args.fps, compression=args.compression,
                              slow_render=slow_render, encoder_preset=args.encoder_preset, speed_factor=args.speedup, main_scheme=main_scheme, force_main=force_main, upscale=args.upscale, encoder_override=args.encoder)
                try:
                    sol, tps, scramble, movetimes = parse_replay_url(val)
                    kwargs["tps"] = tps or args.tps
                    if scramble:
                        kwargs["scramble"] = scramble
                    if isinstance(movetimes, list) and movetimes:
                        kwargs["movetimes"] = movetimes
                except Exception:
                    sol = val
                    if args.tps is not None:
                        kwargs["tps"] = args.tps
                    if args.time is not None:
                        kwargs["time"] = args.time
                    if args.scramble:
                        kwargs["scramble"] = args.scramble
                    if args.size:
                        kwargs["size"] = args.size
                    if movetimes:
                        kwargs["movetimes"] = movetimes

                if args.output is not None:
                    root, ext = os.path.splitext(args.output)
                    output_path = f"{root}_{idx+1:03d}{ext}"
                else:
                    base_name = _generate_filename(
                        sol, kwargs.get("tps"), kwargs.get("time"),
                        kwargs.get("movetimes", -1), kwargs.get("size"),
                        index=idx + 1, speed_factor=args.speedup,
                        scramble=kwargs.get("scramble"))
                    output_path = _pick_output_filename(replays_dir, base_name)

                batch_items.append({"solution": sol, "output_path": output_path, "opts": opts, **kwargs})

            gen = ReplayVideoGenerator()
            gen.batch_render(batch_items, use_gpu=use_gpu, show_progress=True)
        else:
            # Single item: existing sequential path
            for idx, item in enumerate(items):
                mode, val = item if isinstance(item, tuple) else ("manual", item)

                if mode in ("url", "batch"):
                    try:
                        sol, tps, scramble, movetimes = parse_replay_url(val)
                    except Exception:
                        sol, tps, scramble, movetimes = val, args.tps, args.scramble, None

                    if args.output is not None:
                        if len(items) > 1:
                            root, ext = os.path.splitext(args.output)
                            output_path = f"{root}_{idx+1:03d}{ext}"
                        else:
                            output_path = args.output
                    else:
                        base_name = _generate_filename(
                            sol, tps or args.tps, None,
                            movetimes if movetimes else -1, None,
                            index=idx + 1 if len(items) > 1 else 0,
                            speed_factor=args.speedup, scramble=scramble)
                        output_path = _pick_output_filename(replays_dir, base_name)

                    run_single(sol, output_path, opts=opts,
                               tps=tps or args.tps, scramble=scramble,
                               movetimes=movetimes, quality=args.quality,
                               fps=args.fps, compression=args.compression,
                               slow_render=slow_render, encoder_preset=args.encoder_preset,
                               speed_factor=args.speedup, main_scheme=main_scheme,
                               force_main=force_main,
                               upscale=args.upscale, encoder_override=args.encoder)
                else:
                    if args.output is not None:
                        output_path = args.output
                    else:
                        base_name = _generate_filename(
                            val, args.tps, args.time,
                            movetimes if movetimes else -1, args.size,
                            index=idx + 1 if len(items) > 1 else 0,
                            speed_factor=args.speedup, scramble=args.scramble)
                        output_path = _pick_output_filename(replays_dir, base_name)

                    run_single(val, output_path, opts=opts,
                               tps=None if movetimes else args.tps, time=args.time,
                               scramble=args.scramble, size=args.size,
                               quality=args.quality, movetimes=movetimes,
                               fps=args.fps, compression=args.compression,
                               slow_render=slow_render, encoder_preset=args.encoder_preset,
                                speed_factor=args.speedup, main_scheme=main_scheme,
                                force_main=force_main,
                                upscale=args.upscale, encoder_override=args.encoder)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[CRITICAL ERROR] {e}", file=sys.stderr)
        sys.exit(1)

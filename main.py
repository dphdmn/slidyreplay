import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import threading
import time
import os
import subprocess
import sys
import ctypes
from concurrent.futures import ThreadPoolExecutor

import ttkbootstrap as tb
from ttkbootstrap.constants import *

from replay_video import ReplayVideoGenerator, CancelError, _quick_infer_size, _get_available_encoders
from sliding_puzzles import parse_replay_url
from replay_generator import count_moves
from geometry import RenderOptions
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
    if tps and tps > 0:
        display_tps = tps
    else:
        display_tps = None
    if isinstance(movetimes, list) and len(movetimes) > 1:
        time_s = movetimes[-1] / 1000.0 if movetimes[-1] > 0 else 0
        is_movetimes_accurate = True
    elif time_v and time_v > 0:
        time_s = time_v
        is_movetimes_accurate = False
    elif display_tps and display_tps > 0:
        time_s = moves / display_tps
        is_movetimes_accurate = False
    else:
        time_s = 0
        is_movetimes_accurate = False
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
        self.minsize(960, 640)

        self.generated_files = []
        self._executor = None
        self._batch_futures = []
        self._item_progress = {}
        self._start_time = 0.0
        self._last_poll_time = 0.0
        self._last_poll_pct = 0.0
        self._rolling_rate = 0.0
        self.cancel_flag = False

        self.fps_var = tk.IntVar(value=60)
        self.force_fringe_var = tk.BooleanVar(value=False)
        self.quality_preset_var = tk.StringVar(value="1080p")
        self._quality_presets = {"720p": 720, "1080p": 1080, "1440p (2K)": 1440, "2160p (4K)": 2160}
        self.compression_var = tk.IntVar(value=18)
        self.slow_render_var = tk.BooleanVar(value=False)
        self.speed_factor_var = tk.StringVar(value="1.0")

        self.tps_var = tk.StringVar()
        self.time_var = tk.StringVar()
        self.size_var = tk.StringVar()
        self.scramble_var = tk.StringVar()
        self.movetimes_var = tk.StringVar()
        self.out_folder_var = tk.StringVar(value=os.path.join(script_dir, "replays"))
        self.file_path_var = tk.StringVar()
        self.progress_text = tk.StringVar(value="Ready")
        self._gpu_info_var = tk.StringVar(value="")
        self.no_layout_var = tk.BooleanVar(value=False)
        self.no_border_var = tk.BooleanVar(value=False)
        self.no_secondary_border_var = tk.BooleanVar(value=False)
        self.no_numbers_var = tk.BooleanVar(value=False)
        self.upscale_var = tk.BooleanVar(value=False)

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

        # ── Two-column layout using grid ──
        root.grid_columnconfigure(0, weight=1, minsize=400)
        root.grid_columnconfigure(1, weight=1, minsize=440)
        root.grid_rowconfigure(1, weight=1)

        # ======== LEFT COLUMN ========
        left = tb.Frame(root)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 4))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # ── Settings (top of left) ──
        settings = tb.LabelFrame(left, text="Settings")
        settings.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        settings.grid_columnconfigure(0, weight=1)

        r = 0

        # ── OUTPUT ──
        tb.Separator(settings, bootstyle="secondary").grid(row=r, column=0, sticky="ew", pady=(2, 4), padx=12)
        r += 1
        out_row = tb.Frame(settings)
        out_row.grid(row=r, column=0, sticky="ew", pady=(4, 4), padx=12)
        out_row.grid_columnconfigure(1, weight=1)
        tb.Label(out_row, text="Output folder", font=(FONT_FAMILY, 9)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.out_entry = tb.Entry(out_row, textvariable=self.out_folder_var)
        self.out_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        tb.Button(out_row, text="Browse...", command=self._browse_output,
                  bootstyle="secondary-outline", width=9).grid(row=0, column=2)
        r += 1

        # ── VIDEO ──
        tb.Separator(settings, bootstyle="secondary").grid(row=r, column=0, sticky="ew", pady=(6, 4), padx=12)
        r += 1

        fps_row = tb.Frame(settings)
        fps_row.grid(row=r, column=0, sticky="ew", pady=(4, 4), padx=12)
        fps_row.grid_columnconfigure(1, weight=1)
        tb.Label(fps_row, text="FPS", font=(FONT_FAMILY, 9)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.fps_scale = tb.Scale(fps_row, from_=5, to=240,
                                  variable=self.fps_var, orient="horizontal")
        self.fps_scale.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.fps_label = tb.Label(fps_row, text="60", width=5, font=(FONT_FAMILY, 9))
        self.fps_label.grid(row=0, column=2)
        def _snap_fps(*_):
            v = self.fps_var.get()
            snapped = round(v / 5) * 5
            if snapped != v:
                self.fps_var.set(snapped)
            self.fps_label.config(text=f"{snapped:d}")
        self.fps_var.trace_add("write", _snap_fps)
        r += 1

        speed_row = tb.Frame(settings)
        speed_row.grid(row=r, column=0, sticky="ew", pady=(4, 4), padx=12)
        speed_row.grid_columnconfigure(1, weight=1)
        tb.Label(speed_row, text="Speed (×)", font=(FONT_FAMILY, 9)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.speed_entry = tb.Entry(speed_row, textvariable=self.speed_factor_var, width=10)
        self.speed_entry.grid(row=0, column=1, sticky="w", padx=(0, 6))
        r += 1

        compression_row = tb.Frame(settings)
        compression_row.grid(row=r, column=0, sticky="ew", pady=(4, 4), padx=12)
        compression_row.grid_columnconfigure(1, weight=1)
        tb.Label(compression_row, text="Compression (lower = fewer artifacts, larger file)", font=(FONT_FAMILY, 9)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        compression_scale = tb.Scale(compression_row, from_=10, to=40, variable=self.compression_var, orient="horizontal",
                              bootstyle="primary", length=200)
        compression_scale.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.compression_value_lbl = tb.Label(compression_row, text=str(self.compression_var.get()), font=(FONT_FAMILY, 9, "bold"), width=3)
        self.compression_value_lbl.grid(row=0, column=2, sticky="w")
        def _on_compression_change(*_):
            self.compression_value_lbl.config(text=str(self.compression_var.get()))
        self.compression_var.trace_add("write", _on_compression_change)
        r += 1

        # ── RENDERING ──
        tb.Separator(settings, bootstyle="secondary").grid(row=r, column=0, sticky="ew", pady=(6, 4), padx=12)
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

        # GPU + toggles row
        chk_row = tb.Frame(settings)
        chk_row.grid(row=r, column=0, sticky="ew", pady=(4, 4), padx=12)
        self.gpu_toggle = tb.Checkbutton(
            chk_row, text="GPU acceleration", variable=self.use_gpu_var,
            bootstyle="round-toggle success"
        )
        self.gpu_toggle.pack(side="left", padx=(0, 12))
        tb.Checkbutton(chk_row, text="Force fringe", variable=self.force_fringe_var,
                       bootstyle="round-toggle").pack(side="left", padx=(0, 12))
        tb.Label(chk_row, text="Quality:", font=(FONT_FAMILY, 9)).pack(side="left", padx=(0, 4))
        quality_combo = ttk.Combobox(chk_row, textvariable=self.quality_preset_var,
                                    values=list(self._quality_presets.keys()),
                                    state="readonly", width=12)
        quality_combo.pack(side="left")
        def _on_quality_change(*_):
            self._update_quality_warning()
        self.quality_preset_var.trace_add("write", _on_quality_change)
        r += 1

        # GPU info
        self.gpu_info_lbl = tb.Label(settings, font=(FONT_FAMILY, 9), anchor="w")
        self.gpu_info_lbl.grid(row=r, column=0, sticky="ew", pady=(0, 4), padx=12)
        if self._gpu_available:
            self.gpu_info_lbl.config(text=f"GPU ON ({self._gpu_name})", bootstyle="success")
        else:
            self.gpu_info_lbl.config(text="Not available — install CUDA (see README)", bootstyle="secondary")

        # Quality RAM warning (atlas prerender uses significant RAM)
        self.quality_warning = tb.Label(settings, text="", font=(FONT_FAMILY, 8), foreground="#ffa500", anchor="w")
        self.quality_warning.grid(row=r, column=0, sticky="ew", pady=(0, 4), padx=12)
        self.quality_warning.grid_remove()

        def _on_gpu_toggle():
            if self.use_gpu_var.get():
                if self._gpu_available:
                    self.gpu_info_lbl.config(text=f"GPU ON ({self._gpu_name})", bootstyle="success")
                else:
                    self.gpu_info_lbl.config(text="GPU not available — install CUDA (see README)", bootstyle="secondary")
            else:
                self.gpu_info_lbl.config(text="GPU OFF (CPU)", bootstyle="secondary")
                self.slow_render_var.set(True)
            _update_slow_render_desc()
        self.gpu_toggle.config(command=_on_gpu_toggle)
        r += 1

        # Slow render (single line: checkbox + inline description)
        slow_row = tb.Frame(settings)
        slow_row.grid(row=r, column=0, sticky="ew", pady=(4, 4), padx=12)
        self.slow_render_cb = tb.Checkbutton(slow_row, text="Slow render", variable=self.slow_render_var,
                                              bootstyle="round-toggle")
        self.slow_render_cb.pack(side="left")
        self.slow_render_desc = tb.Label(slow_row, text="~33% smaller file, ~33% longer",
                                          font=(FONT_FAMILY, 9), foreground="#888")
        self.slow_render_desc.pack(side="left", padx=(8, 0))

        def _update_slow_render_desc():
            if not self.use_gpu_var.get():
                self.slow_render_desc.config(text="auto-enabled for CPU (free smaller files)")
            elif self.slow_render_var.get():
                self.slow_render_desc.config(text="~33% smaller file, ~33% longer")
            else:
                self.slow_render_desc.config(text="")
        self.slow_render_var.trace_add("write", lambda *_: _update_slow_render_desc())
        self.after(10, _update_slow_render_desc)
        r += 1

        # Upscale to 2K
        upscale_row = tb.Frame(settings)
        upscale_row.grid(row=r, column=0, sticky="ew", pady=(4, 4), padx=12)
        self.upscale_cb = tb.Checkbutton(upscale_row, text="Upscale to 2K", variable=self.upscale_var,
                                          bootstyle="round-toggle")
        self.upscale_cb.pack(side="left")
        self.upscale_desc = tb.Label(upscale_row, text="Re-encode to 2560×1440 for best YouTube quality",
                                      font=(FONT_FAMILY, 9), foreground="#888")
        self.upscale_desc.pack(side="left", padx=(8, 0))
        self.upscale_var.trace_add("write", lambda *_: self._update_quality_warning())
        r += 1

        # Encoder override
        enc_row = tb.Frame(settings)
        enc_row.grid(row=r, column=0, sticky="ew", pady=(4, 4), padx=12)
        tb.Label(enc_row, text="Encoder:", font=(FONT_FAMILY, 9)).pack(side="left", padx=(0, 6))
        self.encoder_var = tk.StringVar(value="Auto")
        available = _get_available_encoders()
        enc_values = ["Auto"] + available
        enc_combo = ttk.Combobox(enc_row, textvariable=self.encoder_var,
                                 values=enc_values,
                                 state="readonly", width=18)
        enc_combo.pack(side="left")
        r += 1

        # Render toggles
        render_opts_row = tb.Frame(settings)
        render_opts_row.grid(row=r, column=0, sticky="ew", pady=(4, 4), padx=12)
        tb.Checkbutton(render_opts_row, text="No layout", variable=self.no_layout_var,
                       bootstyle="round-toggle").pack(side="left", padx=(0, 10))
        tb.Checkbutton(render_opts_row, text="No border", variable=self.no_border_var,
                       bootstyle="round-toggle").pack(side="left", padx=(0, 10))
        tb.Checkbutton(render_opts_row, text="No sec border", variable=self.no_secondary_border_var,
                       bootstyle="round-toggle").pack(side="left", padx=(0, 10))
        tb.Checkbutton(render_opts_row, text="No numbers", variable=self.no_numbers_var,
                       bootstyle="round-toggle").pack(side="left")
        r += 1

        # ── Notebook (below settings) ──
        nb = tb.Notebook(left, bootstyle="dark")
        nb.grid(row=1, column=0, sticky="nsew")
        self.nb = nb

        url_tab = tb.Frame(nb, padding=8)
        file_tab = tb.Frame(nb, padding=8)
        manual_tab = tb.Frame(nb, padding=8)
        nb.add(url_tab, text="URL")
        nb.add(file_tab, text="File")
        nb.add(manual_tab, text="Manual")

        # -- File tab --
        tb.Label(file_tab, text="Single input file (solution or replay URL):",
                 font=(FONT_FAMILY, 10, "bold")).pack(anchor="w", pady=(0, 4))
        file_row = tb.Frame(file_tab)
        file_row.pack(fill="x", pady=(0, 4))
        file_row.grid_columnconfigure(0, weight=1)
        self.file_entry = tb.Entry(file_row, textvariable=self.file_path_var)
        self.file_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        tb.Button(file_row, text="Browse...", command=self._browse_file,
                  bootstyle="secondary-outline", width=9).grid(row=0, column=1)
        self.file_meta_var = tk.StringVar(value="No file selected.")
        self.file_meta_label = tb.Label(file_tab, textvariable=self.file_meta_var,
                                        font=(FONT_FAMILY, 9), foreground="#aaaaaa",
                                        anchor="w", wraplength=500)
        self.file_meta_label.pack(fill="x", anchor="w")

        # -- URL tab --
        tb.Label(url_tab, text="Replay URLs (one per line):",
                 font=(FONT_FAMILY, 10, "bold")).pack(anchor="w")
        self.url_text = scrolledtext.ScrolledText(
            url_tab, height=8, font=(FONT_MONO_FAMILY, 10),
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="#fff",
            relief="flat", borderwidth=0, highlightthickness=1,
            highlightbackground="#3a3a3a", highlightcolor="#3a3a3a")
        self.url_text.pack(fill="both", expand=True, pady=(4, 0))
        _setup_placeholder(self.url_text, "# paste URLs here, one per line")

        # -- Manual tab --
        manual_tab.grid_rowconfigure(3, weight=1)
        manual_tab.grid_columnconfigure(0, weight=1)

        params = tb.Frame(manual_tab)
        params.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        c = 0
        tb.Label(params, text="TPS:", font=(FONT_FAMILY, 9)).grid(row=0, column=c, sticky="w", padx=(0, 2))
        c += 1
        self.tps_entry = tb.Entry(params, textvariable=self.tps_var, width=10)
        self.tps_entry.grid(row=0, column=c, padx=(0, 8))
        c += 1
        tb.Label(params, text="Time (s):", font=(FONT_FAMILY, 9)).grid(row=0, column=c, sticky="w", padx=(0, 2))
        c += 1
        self.time_entry = tb.Entry(params, textvariable=self.time_var, width=10)
        self.time_entry.grid(row=0, column=c, padx=(0, 8))
        c += 1
        tb.Label(params, text="Size:", font=(FONT_FAMILY, 9)).grid(row=0, column=c, sticky="w", padx=(0, 2))
        c += 1
        self.size_entry = tb.Entry(params, textvariable=self.size_var, width=10)
        self.size_entry.grid(row=0, column=c)

        params2 = tb.Frame(manual_tab)
        params2.grid(row=1, column=0, sticky="ew")
        tb.Label(params2, text="Scramble:", font=(FONT_FAMILY, 9)).pack(side="left")
        self.scramble_entry = tb.Entry(params2, textvariable=self.scramble_var, width=22)
        self.scramble_entry.pack(side="left", padx=(4, 8))
        tb.Label(params2, text="Movetimes:", font=(FONT_FAMILY, 9)).pack(side="left")
        self.movetimes_entry = tb.Entry(params2, textvariable=self.movetimes_var, width=22)
        self.movetimes_entry.pack(side="left", padx=(4, 0))

        tb.Label(manual_tab, text="Solution strings (one per line):",
                 font=(FONT_FAMILY, 10, "bold")).grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.solution_text = scrolledtext.ScrolledText(
            manual_tab, font=(FONT_MONO_FAMILY, 10),
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="#fff",
            relief="flat", borderwidth=0, highlightthickness=1,
            highlightbackground="#3a3a3a", highlightcolor="#3a3a3a")
        self.solution_text.grid(row=3, column=0, sticky="nsew", pady=(2, 0))
        _setup_placeholder(self.solution_text, "# solutions here, one per line")
        self.solution_text.bind("<<Modified>>", self._on_solution_change)

        # ── Action buttons ──
        act = tb.Frame(left)
        act.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.gen_btn = tb.Button(act, text="Generate All", command=self._generate,
                                 bootstyle="success", width=16)
        self.gen_btn.pack(side="left", padx=(0, 6))
        self.cancel_btn = tb.Button(act, text="Cancel", command=self._cancel,
                                    bootstyle="secondary", state="disabled")
        self.cancel_btn.pack(side="left")

        # ======== RIGHT COLUMN ========
        right = tb.Frame(root)
        right.grid(row=0, column=1, rowspan=2, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # ── Progress ──
        prog_frame = tb.LabelFrame(right, text="Progress")
        prog_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        prog_frame.pack_propagate(False)
        prog_frame.configure(height=120)

        self.progress_bar = tb.Progressbar(prog_frame, mode="determinate",
                                           bootstyle="success-striped")
        self.progress_bar.pack(fill="x", padx=8, pady=(8, 2))

        prog_label = tb.Label(prog_frame, textvariable=self.progress_text,
                              font=(FONT_FAMILY, 9), anchor="w")
        prog_label.pack(fill="x", padx=8, pady=(0, 0))

        self._gpu_info_var = tk.StringVar(value="")
        gpu_label = tb.Label(prog_frame, textvariable=self._gpu_info_var,
                             font=(FONT_FAMILY, 8), anchor="w", foreground="#aaaaaa")
        gpu_label.pack(fill="x", padx=8, pady=(0, 6))

        # ── Generated replays ──
        lst_frame = tb.LabelFrame(right, text="Generated Replays")
        lst_frame.grid(row=1, column=0, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        lst_frame.grid_columnconfigure(0, weight=1)
        lst_frame.grid_rowconfigure(0, weight=1)

        self.replay_listbox = tk.Listbox(
            lst_frame, font=(FONT_MONO_FAMILY, 9), activestyle="none",
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
            text = "⚠ 4K (2160p) uses very high RAM during atlas prerender — reduce puzzle size if out of memory"
        elif h >= 1440:
            text = "⚠ 2K (1440p) uses high RAM during atlas prerender — may cause slowdown or OOM on large puzzles (>=50×50)"
        if h >= 1440 and self.upscale_var.get():
            text += "  |  Upscale not needed (already ≥1440p)"
        elif h < 1440 and self.upscale_var.get():
            text += "  |  ✓ Upscale to 2K after render"
        if text:
            self.quality_warning.config(text=text)
            self.quality_warning.grid()
        else:
            self.quality_warning.grid_remove()

    def _center_window(self):
        self.update_idletasks()
        w, h = 960, 680
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _count_solutions(self):
        raw = self.solution_text.get("1.0", "end-1c").strip()
        lines = [l.strip() for l in raw.splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        return len(lines)

    def _on_solution_change(self, event=None):
        n = self._count_solutions()
        state = "disabled" if n > 1 else "normal"
        fg = "#555" if n > 1 else "#d4d4d4"
        self.scramble_entry.config(state=state)
        self.movetimes_entry.config(state=state)
        self.scramble_entry.configure(foreground=fg)
        self.movetimes_entry.configure(foreground=fg)
        self.solution_text.edit_modified(False)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Output folder")
        if path:
            self.out_folder_var.set(path)

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select input file",
            filetypes=[("All files", "*.*")]
        )
        if not path:
            return
        self.file_path_var.set(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw:
                self.file_meta_var.set("Empty file.")
                return
            if raw.startswith(("http://", "https://")):
                solution, tps, scramble, movetimes = parse_replay_url(raw)
                moves = count_moves(solution)
                if isinstance(movetimes, list) and movetimes[-1] > 0:
                    time_s = movetimes[-1] / 1000.0
                    tps = moves / time_s
                else:
                    time_s = 0
                size = _quick_infer_size(solution, scramble)
                size_str = f"{size[0]}x{size[1]}" if size else "?"
                accurate = isinstance(movetimes, list) and len(movetimes) > 1
                if time_s and tps:
                    meta = f"{size_str} | {time_s:.3f} ({moves} / {tps:.3f})"
                elif time_s:
                    meta = f"{size_str} | {time_s:.3f} ({moves})"
                else:
                    meta = f"{size_str} | {moves} moves"
                if accurate:
                    meta += " | movetimes accurate"
            else:
                solution = raw
                moves = count_moves(solution)
                size = _quick_infer_size(solution)
                size_str = f"{size[0]}x{size[1]}" if size else "?"
                meta = f"{size_str} | {moves} moves"
            self.file_meta_var.set(meta)
        except Exception as e:
            self.file_meta_var.set(f"Parse error: {e}")

    def _active_tab(self):
        return self.nb.index(self.nb.select())

    def _generate(self):
        if self._executor and not all(f.done() for f in self._batch_futures):
            log.info("_generate: skipped — previous batch still running")
            return

        tab = self._active_tab()
        items = []
        log.info(f"_generate: tab={tab} ('URL' if tab==0 else 'File' if tab==1 else 'Manual')")

        if tab == 0:
            raw = self.url_text.get("1.0", "end-1c").strip()
            log.info(f"  URL tab raw len={len(raw)}, first_100={repr(raw[:100])}")
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    log.info(f"  SKIP line: {repr(line[:80])}")
                    continue
                if line.startswith(("http://", "https://")):
                    log.info(f"  ADD url item: len={len(line)}, preview={repr(line[:100])}")
                    items.append(("url", line))
                else:
                    log.info(f"  SKIP (not http): {repr(line[:80])}")
        elif tab == 1:
            path = self.file_path_var.get().strip()
            log.info(f"  File tab path={repr(path)} exists={os.path.exists(path)}")
            if not path or not os.path.exists(path):
                self.progress_text.set("No file selected.")
                return
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            log.info(f"  File tab raw len={len(raw)} starts_http={raw.startswith('http')}")
            if raw.startswith(("http://", "https://")):
                log.info(f"  ADD url item from file: len={len(raw)}")
                items.append(("url", raw))
            else:
                log.info(f"  ADD manual item from file: len={len(raw)}")
                items.append(("manual", raw))
        else:
            raw = self.solution_text.get("1.0", "end-1c").strip()
            n_lines = 0
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                items.append(("manual", line))
                n_lines += 1
            log.info(f"  Manual tab: {n_lines} solution lines")

        log.info(f"_generate: total items={len(items)}")
        if not items:
            self.progress_text.set("No valid entries.")
            return

        self.cancel_flag = False
        self._set_ui_busy(True)
        self._start_time = time.time()
        self.progress_bar["value"] = 0
        self.progress_text.set("")
        self.replay_listbox.delete(0, "end")
        self.generated_files.clear()

        raw_folder = self.out_folder_var.get().strip()
        output_dir = os.path.abspath(raw_folder) if raw_folder else os.path.join(script_dir, "replays")
        log.info(f"_generate: output_dir={output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        total = len(items)
        self._item_progress = {}
        for idx in range(total):
            self._item_progress[idx] = {"adjusted_cur": 0, "adjusted_tot": 1,
                                        "done": False, "path": None, "error": None,
                                        "cancelled": False}

        self._batch_futures = []
        self._executor = ThreadPoolExecutor(max_workers=1)

        if total == 1:
            # Single item: existing per-item path (preserves detailed progress)
            self._is_batch = False
            def on_done(idx, fut):
                try:
                    fut.result()
                except CancelError:
                    self._item_progress[idx]["cancelled"] = True
                except Exception as e:
                    self._item_progress[idx]["error"] = str(e)
                self._item_progress[idx]["done"] = True

            for idx, (mode, input_str) in enumerate(items):
                fut = self._executor.submit(
                    self._process_item, idx, mode, input_str, output_dir, total)
                fut.add_done_callback(lambda f, i=idx: on_done(i, f))
                self._batch_futures.append(fut)
        else:
            # Batch: build all items, submit once via batch_render
            self._is_batch = True
            self._batch_done = 0
            self._batch_total = total
            self._batch_cancelled = False
            fut = self._executor.submit(self._process_batch, items, output_dir)
            self._batch_futures = [fut]

        self.after(1000, self._poll_batch)

    def _process_item(self, idx, mode, input_str, output_dir, total):
        log.info(f"_process_item[{idx}]: mode={mode}, input_str_len={len(input_str)}")
        try:
            opts = RenderOptions(
                grid_only=self.no_layout_var.get(),
                no_border=self.no_border_var.get(),
                no_secondary_border=self.no_secondary_border_var.get(),
                no_numbers=self.no_numbers_var.get(),
            )
            params = {
                "force_fringe": self.force_fringe_var.get(),
                "quality": self._get_quality(),
                "fps": self.fps_var.get(),
                "compression": self.compression_var.get(),
                "slow_render": self.slow_render_var.get(),
                "speed_factor": self._get_speed_factor(),
                "upscale": self.upscale_var.get(),
                "encoder_override": "" if self.encoder_var.get() == "Auto" else self.encoder_var.get(),
                "opts": opts,
            }
            log.info(f"_process_item[{idx}]: base_params={params}")

            if mode == "url":
                solution, tps, scramble, movetimes = parse_replay_url(input_str)
                sol_len = count_moves(solution)
                log.info(f"_process_item[{idx}]: parsed URL -> sol_len={sol_len}, tps={tps}, scramble={'yes' if scramble else 'no'}, movetimes_type={type(movetimes).__name__}")
                if isinstance(movetimes, list):
                    log.info(f"_process_item[{idx}]: movetimes len={len(movetimes)}, first={movetimes[0]}, last={movetimes[-1]}")
                if tps is not None:
                    params["tps"] = tps
                    log.info(f"_process_item[{idx}]: set tps={tps}")
                if scramble:
                    params["scramble"] = scramble
                    log.info(f"_process_item[{idx}]: set scramble (len={len(scramble)})")
                if isinstance(movetimes, list) and len(movetimes) > 0:
                    params["movetimes"] = movetimes
                    log.info(f"_process_item[{idx}]: set movetimes (len={len(movetimes)})")
            else:
                solution = input_str
                sol_len = count_moves(solution)
                log.info(f"_process_item[{idx}]: manual mode, sol_len={sol_len}")
                tps_s = self.tps_var.get().strip()
                tps = float(tps_s) if tps_s else None
                time_s = self.time_var.get().strip()
                time_v = float(time_s) if time_s else None
                if time_v and tps:
                    tps = None
                if tps:
                    params["tps"] = tps
                    log.info(f"_process_item[{idx}]: manual set tps={tps}")
                if time_v:
                    params["time"] = time_v
                    log.info(f"_process_item[{idx}]: manual set time={time_v}")
                scramble_s = self.scramble_var.get().strip()
                if scramble_s:
                    params["scramble"] = scramble_s
                    log.info(f"_process_item[{idx}]: manual set scramble (len={len(scramble_s)})")
                size_s = self.size_var.get().strip()
                if size_s:
                    params["size"] = size_s
                    log.info(f"_process_item[{idx}]: manual set size={size_s}")
                movetimes_s = self.movetimes_var.get().strip()
                if movetimes_s:
                    params["movetimes"] = [int(x.strip()) for x in movetimes_s.split(",")]
                    log.info(f"_process_item[{idx}]: manual set movetimes (from text)")

            log.info(f"_process_item[{idx}]: final params={ {k: v if not isinstance(v, list) or len(repr(v)) < 200 else f'<list len={len(v)}>' for k, v in params.items()} }")
            log.info(f"_process_item[{idx}]: use_gpu={self.use_gpu_var.get()}")

            filename_tps = params.get("tps", tps if mode == "url" else None)
            filename_time = params.get("time", None)
            base_name = _generate_filename(
                solution, filename_tps, filename_time,
                params.get("movetimes", -1), params.get("size"),
                index=idx + 1, speed_factor=self._get_speed_factor(),
                scramble=params.get("scramble"))
            out_path = _pick_output_filename(output_dir, base_name)
            log.info(f"_process_item[{idx}]: output={out_path}")

            def on_progress(adjusted_cur, adjusted_tot, gpu_stats=None, use_gpu=False):
                log.info(f"progress[{idx}]: adjusted_cur={adjusted_cur} adjusted_tot={adjusted_tot}")
                if gpu_stats:
                    gs = gpu_stats
                    log.info(f"progress[{idx}]: gpu_stats: name={gs.get('gpu_name','?')}, mem={gs.get('mem_used_mb',0)}/{gs.get('total_mem_mb',0)}MB, batch={gs.get('batch_size',0)}, batch_idx={gs.get('batch_idx',0)}/{gs.get('num_batches',0)}")
                self._on_item_progress(idx, adjusted_cur, adjusted_tot, gpu_stats=gpu_stats, use_gpu=use_gpu)

            log.info(f"_process_item[{idx}]: calling generate_simple_replay with params={ {k: v if not isinstance(v, list) or len(repr(v)) < 200 else f'<list len={len(v)}>' for k, v in params.items()} }")
            gen = ReplayVideoGenerator(cleanup_frames=False)
            gen.generate_simple_replay(
                solution=solution, output_path=out_path,
                show_progress=False, external_progress_cb=on_progress,
                use_gpu=self.use_gpu_var.get(),
                cancel_check=lambda: self.cancel_flag, **params)
            log.info(f"_process_item[{idx}]: generate_simple_replay completed")

            if not self.cancel_flag:
                self._item_progress[idx]["path"] = out_path
                self.after(0, lambda p=out_path: self._add_to_list(p))
                if self.upscale_var.get() and self._get_quality() < 1440:
                    stem, ext = os.path.splitext(out_path)
                    upscaled_path = f"{stem}_1440p60{ext}"
                    self.after(0, lambda p=upscaled_path: os.path.exists(p) and self._add_to_list(p))
        except CancelError:
            log.info(f"_process_item[{idx}]: CANCELLED")
            raise
        except Exception as e:
            log.error(f"_process_item[{idx}]: FAILED: {e}", exc_info=True)
            self._item_progress[idx]["error"] = str(e)
            self.after(0, lambda m=f"Item {idx+1} failed: {e}": self.progress_text.set(m))
            raise

    def _build_batch_items(self, items, output_dir):
        """Convert GUI (mode, input_str) pairs into batch_render item dicts."""
        batch_items = []
        for idx, (mode, input_str) in enumerate(items):
            params = {
                "force_fringe": self.force_fringe_var.get(),
                "quality": self._get_quality(),
                "fps": self.fps_var.get(),
                "compression": self.compression_var.get(),
                "slow_render": self.slow_render_var.get(),
                "speed_factor": self._get_speed_factor(),
                "upscale": self.upscale_var.get(),
                "encoder_override": "" if self.encoder_var.get() == "Auto" else self.encoder_var.get(),
                "opts": RenderOptions(
                    grid_only=self.no_layout_var.get(),
                    no_border=self.no_border_var.get(),
                    no_secondary_border=self.no_secondary_border_var.get(),
                    no_numbers=self.no_numbers_var.get(),
                ),
            }

            if mode == "url":
                solution, tps, scramble, movetimes = parse_replay_url(input_str)
                if tps is not None:
                    params["tps"] = tps
                if scramble:
                    params["scramble"] = scramble
                if isinstance(movetimes, list) and len(movetimes) > 0:
                    params["movetimes"] = movetimes
            else:
                solution = input_str
                tps_s = self.tps_var.get().strip()
                tps = float(tps_s) if tps_s else None
                time_s = self.time_var.get().strip()
                time_v = float(time_s) if time_s else None
                if time_v and tps:
                    tps = None
                if tps:
                    params["tps"] = tps
                if time_v:
                    params["time"] = time_v
                scramble_s = self.scramble_var.get().strip()
                if scramble_s:
                    params["scramble"] = scramble_s
                size_s = self.size_var.get().strip()
                if size_s:
                    params["size"] = size_s
                movetimes_s = self.movetimes_var.get().strip()
                if movetimes_s:
                    params["movetimes"] = [int(x.strip()) for x in movetimes_s.split(",")]

            out_path = _pick_output_filename(output_dir, _generate_filename(
                solution, params.get("tps"), params.get("time"),
                params.get("movetimes", -1), params.get("size"),
                index=idx + 1, speed_factor=self._get_speed_factor(),
                scramble=params.get("scramble")))

            batch_items.append({"solution": solution, "output_path": out_path, **params})
        return batch_items

    def _process_batch(self, items, output_dir):
        """Run batch_render on all items in a single background thread."""
        batch_items = self._build_batch_items(items, output_dir)
        total = len(batch_items)
        log.info(f"_process_batch: {total} items prepared")
        _pb_start = time.time()
        _pb_last = [0.0]
        _pb_prev = [0]

        def _fmt(t):
            return f"{t:.1f}s" if t < 60 else f"{int(t//60)}m {t%60:.0f}s"

        def on_progress(cur, _tot, **_):
            if self.cancel_flag:
                return
            now = time.time()
            dt = now - _pb_last[0]
            dc = cur - _pb_prev[0]
            _pb_last[0] = now
            _pb_prev[0] = cur
            elapsed = now - _pb_start
            rate = dc / dt if dt > 0 else 0
            remaining = total - cur
            eta = remaining / rate if rate > 0 else 0
            expected = elapsed + eta
            pct = cur * 100 / total
            exp_str = _fmt(expected) if rate > 0 and expected < elapsed * 100 else "?"
            label = f"{cur}/{total} — {_fmt(elapsed)}/{exp_str}"
            self.after(0, lambda v=pct: self.progress_bar.configure(value=v))
            self.after(0, lambda t=f"{label}": self.progress_text.set(t))

        try:
            gen = ReplayVideoGenerator()
            paths = gen.batch_render(
                batch_items,
                use_gpu=self.use_gpu_var.get(),
                show_progress=False,
                external_progress_cb=on_progress,
                cancel_check=lambda: self.cancel_flag,
            )
            log.info(f"_process_batch: completed {len(paths)} items")
            for p in paths:
                self.after(0, lambda p=p: self._add_to_list(p))
            return paths
        except CancelError:
            log.info("_process_batch: CANCELLED")
            raise
        except Exception as e:
            log.error(f"_process_batch: FAILED: {e}", exc_info=True)
            raise

    def _on_item_progress(self, idx, adjusted_cur, adjusted_tot, gpu_stats=None, use_gpu=False):
        item = self._item_progress[idx]
        item["adjusted_cur"] = adjusted_cur
        item["adjusted_tot"] = adjusted_tot
        if gpu_stats:
            item["gpu_stats"] = gpu_stats
        if use_gpu:
            item["is_gpu"] = True

    def _cancel(self):
        self.cancel_flag = True
        if self._executor:
            self._executor.shutdown(wait=False)
        self.progress_text.set("Cancelling...")

    def _poll_batch(self):
        if not self._batch_futures:
            return

        total = len(self._batch_futures)
        done_count = sum(1 for f in self._batch_futures if f.done())

        # Batch mode (single future from _process_batch)
        if getattr(self, '_is_batch', False):
            fut = self._batch_futures[0]
            if not fut.done():
                self.after(500, self._poll_batch)
                return
            elapsed = time.time() - self._start_time
            elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed//60)}m {elapsed%60:.0f}s"
            try:
                paths = fut.result()
                self.progress_text.set(f"{len(paths)} replay(s) generated. {elapsed_str}")
            except CancelError:
                self.progress_text.set(f"Cancelled. {elapsed_str}")
            except Exception as e:
                self.progress_text.set(f"Batch failed: {e}. {elapsed_str}")
            self.progress_bar["value"] = 100
            self._set_ui_busy(False)
            if self._executor:
                self._executor.shutdown(wait=False)
                self._executor = None
            self._batch_futures = []
            self._is_batch = False
            return

        # Single-item mode (per-item futures)
        log.info(f"_poll_batch: done={done_count}/{total}")

        overall_pct = 0.0
        running = 0
        for p in self._item_progress.values():
            share = 1.0 / total
            completion = min(p["adjusted_cur"] / p["adjusted_tot"], 1.0) if p["adjusted_tot"] > 0 else 0
            overall_pct += share * completion
            if not p["done"]:
                running += 1

        overall_pct *= 100

        now = time.time()
        dt = now - self._last_poll_time
        dp = overall_pct - self._last_poll_pct
        if dt > 0.5 and dp >= 0:
            inst = dp / dt
            self._rolling_rate = inst if self._rolling_rate <= 0 else self._rolling_rate * 0.7 + inst * 0.3
        self._last_poll_time = now
        self._last_poll_pct = overall_pct

        elapsed = time.time() - self._start_time
        elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed//60)}m {elapsed%60:.0f}s"

        if not self.cancel_flag:
            display_pct = min(overall_pct, 99.0) if running else overall_pct
            self.progress_bar["value"] = display_pct

            expected_str = ""
            if overall_pct > 1 and running:
                eta = (100 - overall_pct) / self._rolling_rate if self._rolling_rate > 0 else 0
                expected = elapsed + eta
                if expected < elapsed * 100:
                    exp_s = f"{expected:.1f}s" if expected < 60 else f"{int(expected//60)}m {expected%60:.0f}s"
                    expected_str = f"/{exp_s}"
                else:
                    expected_str = "/?"

            gpu_str = ""
            for p in self._item_progress.values():
                gs = p.get("gpu_stats")
                if gs and gs.get("batch_size"):
                    gpu_str = f"GPU: {gs.get('gpu_name', '?')} | VRAM: {gs.get('mem_used_mb', 0)}/{gs.get('total_mem_mb', 0)} MB | Batch: {gs.get('batch_size', 0)} frames"
                    break
            self._gpu_info_var.set(gpu_str)
            self.progress_text.set(f"{elapsed_str}{expected_str} ({display_pct:.0f}%)")

        if done_count == total:
            errors = sum(1 for p in self._item_progress.values() if p["error"])
            cancelled = sum(1 for p in self._item_progress.values() if p.get("cancelled"))
            ok_count = done_count - errors - cancelled
            took = f"took {elapsed_str}"
            parts = []
            if ok_count:
                parts.append(f"{ok_count} replay(s) generated")
            if cancelled:
                parts.append(f"{cancelled} cancelled")
            if errors:
                parts.append(f"{errors} failed")
            msg = " — ".join(parts) + f". {took}" if parts else f"Cancelled. {took}"
            self.progress_text.set(msg)
            self.progress_bar["value"] = 100
            self._set_ui_busy(False)
            if self._executor:
                self._executor.shutdown(wait=False)
                self._executor = None
            self._batch_futures = []
            return

        self.after(1000, self._poll_batch)

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
    parser.add_argument("--force-fringe", action="store_true", default=False,
                        help="Force fringe colors (disable grids detection)")
    parser.add_argument("--log", "-l", action="store_true", default=False,
                        help="Enable debug logging to file (logs/debug_<timestamp>.log)")
    parser.add_argument("--no-layout", action="store_true",
                        help="Only render the puzzle grid on dark background — no timer bar, no stats panel")
    parser.add_argument("--grid-only", action="store_true", dest="no_layout",
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-border", action="store_true",
                        help="Suppress tile border outlines")
    parser.add_argument("--no-secondary-border", action="store_true",
                        help="Suppress secondary color bar borders")
    parser.add_argument("--no-numbers", action="store_true",
                        help="Suppress tile number text")
    parser.add_argument("--upscale", action="store_true", default=False,
                        help="After rendering, upscale video to 2K (2560x1440) for best YouTube quality. "
                             "Only beneficial for qualities below 1440p. Keeps both original and upscaled versions.")
    parser.add_argument("--encoder", type=str, default="",
                        choices=["hevc_nvenc", "hevc_amf", "hevc_qsv", "libx265", "h264_nvenc", "h264_amf", "h264_qsv", "libx264"],
                        help="Force video encoder. Auto-detected from available hardware if not set.")

    args = parser.parse_args()

    opts = RenderOptions(
        grid_only=args.no_layout,
        no_border=args.no_border,
        no_secondary_border=args.no_secondary_border,
        no_numbers=args.no_numbers,
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
                              slow_render=slow_render, encoder_preset=args.encoder_preset, speed_factor=args.speedup, force_fringe=args.force_fringe, upscale=args.upscale, encoder_override=args.encoder)
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
                               speed_factor=args.speedup, force_fringe=args.force_fringe,
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
                                speed_factor=args.speedup, force_fringe=args.force_fringe,
                                upscale=args.upscale, encoder_override=args.encoder)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[CRITICAL ERROR] {e}", file=sys.stderr)
        sys.exit(1)

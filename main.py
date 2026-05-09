import tkinter as tk
from tkinter import filedialog, scrolledtext
import threading
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import ttkbootstrap as tb
from ttkbootstrap.constants import *

from replay_video import ReplayVideoGenerator, parse_replay_url
from replay_generator import expand_solution, parse_scramble_guess

if getattr(sys, 'frozen', False):
    base = sys._MEIPASS
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    os.environ["PATH"] = base + os.pathsep + os.environ.get("PATH", "")
else:
    base = os.path.dirname(os.path.abspath(__file__))
    script_dir = base


def _generate_filename(solution, tps, time_v, movetimes, size_arg=None, index=0):
    moves = len(expand_solution(solution))
    if tps and tps > 0:
        display_tps = tps / 1000.0 if tps >= 1000 else tps
    else:
        display_tps = None
    if isinstance(movetimes, list) and len(movetimes) > 1:
        time_s = movetimes[-1] / 1000.0 if movetimes[-1] > 0 else 0
    elif time_v and time_v > 0:
        time_s = time_v
    elif display_tps and display_tps > 0:
        time_s = moves / display_tps
    else:
        time_s = 0
    if size_arg:
        if isinstance(size_arg, tuple):
            w, h = size_arg
        else:
            parts = str(size_arg).lower().split("x")
            w, h = parts[0], parts[1]
    else:
        try:
            matrix = parse_scramble_guess(solution)
            w, h = len(matrix[0]), len(matrix)
        except Exception:
            w, h = "?", "?"
    tps_str = f"{display_tps:.3f}tps" if display_tps else ""
    time_str = f"{time_s:.2f}s" if time_s else ""
    parts = [f"{w}x{h}", tps_str, f"{moves}mov", time_str]
    name = "_".join(p for p in parts if p).rstrip("_") or "replay"
    name = name.translate(str.maketrans("", "", '\\/:*?\"<>|'))
    if index:
        name = f"{name}_{index}"
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

        ico = os.path.join(base, "assets", "15PUZZLE_ICON.ico")
        if os.path.exists(ico):
            try:
                self.iconbitmap(ico)
            except Exception:
                pass

        self.generated_files = []
        self._executor = None
        self._batch_futures = []
        self._item_progress = {}
        self._start_time = 0.0
        self._last_poll_time = 0.0
        self._last_poll_pct = 0.0
        self._rolling_rate = 0.0
        self.cancel_flag = False

        self.quality_var = tk.DoubleVar(value=2.0)
        self.force_fringe_var = tk.BooleanVar(value=False)
        self.out_folder_var = tk.StringVar(
            value=os.path.join(script_dir, "replays"))
        self.status_var = tk.StringVar(value="Ready")

        self.tps_var = tk.StringVar()
        self.time_var = tk.StringVar()
        self.size_var = tk.StringVar()
        self.scramble_var = tk.StringVar()
        self.movetimes_var = tk.StringVar()

        os.makedirs(self.out_folder_var.get(), exist_ok=True)

        self._build_ui()
        self._center_window()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.deiconify()

    def _build_ui(self):
        root = tb.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        # ── Two-column layout using grid ──
        root.grid_columnconfigure(0, weight=1, minsize=400)
        root.grid_columnconfigure(1, weight=0, minsize=300)
        root.grid_rowconfigure(1, weight=1)

        # ======== LEFT COLUMN ========
        left = tb.Frame(root)
        left.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 4))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # ── Settings (top of left) ──
        settings = tb.LabelFrame(left, text="Settings")
        settings.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        sinner = tb.Frame(settings)
        sinner.pack(fill="both", expand=True, padx=8, pady=6)

        r = 0
        tb.Label(sinner, text="Quality", font=("Segoe UI", 9)).grid(row=r, column=0, sticky="w")
        qf = tb.Frame(sinner)
        qf.grid(row=r, column=1, sticky="ew", padx=(6, 0))
        self.quality_scale = tb.Scale(qf, from_=1.0, to=4.0,
                                      variable=self.quality_var, orient="horizontal", length=180)
        self.quality_scale.pack(side="left")
        self.quality_label = tb.Label(qf, text="2.0x", width=5, font=("Segoe UI", 9))
        self.quality_label.pack(side="left", padx=(6, 0))
        self.quality_var.trace_add("write",
                                   lambda *a: self.quality_label.config(text=f"{self.quality_var.get():.1f}x"))
        r += 1

        fringe_row = tb.Frame(sinner)
        fringe_row.grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 0))
        tb.Checkbutton(fringe_row, text="Force fringe", variable=self.force_fringe_var,
                       bootstyle="round-toggle").pack(side="left")
        r += 1

        out_row = tb.Frame(sinner)
        out_row.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        out_row.grid_columnconfigure(1, weight=1)
        tb.Label(out_row, text="Output folder", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.out_entry = tb.Entry(out_row, textvariable=self.out_folder_var)
        self.out_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        tb.Button(out_row, text="Browse...", command=self._browse_output,
                  bootstyle="secondary-outline", width=9).grid(row=0, column=2)

        # ── Notebook (below settings) ──
        nb = tb.Notebook(left, bootstyle="dark")
        nb.grid(row=1, column=0, sticky="nsew")
        self.nb = nb

        url_tab = tb.Frame(nb, padding=8)
        manual_tab = tb.Frame(nb, padding=8)
        nb.add(url_tab, text="URL")
        nb.add(manual_tab, text="Manual")

        # -- URL tab --
        tb.Label(url_tab, text="Replay URLs (one per line):",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.url_text = scrolledtext.ScrolledText(
            url_tab, height=8, font=("Consolas", 10),
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="#fff",
            relief="flat", borderwidth=0, highlightthickness=1,
            highlightbackground="#3a3a3a", highlightcolor="#3a3a3a")
        self.url_text.pack(fill="both", expand=True, pady=(4, 0))
        self.url_text.insert("1.0", "# paste URLs here, one per line\n")

        # -- Manual tab --
        manual_tab.grid_rowconfigure(3, weight=1)
        manual_tab.grid_columnconfigure(0, weight=1)

        params = tb.Frame(manual_tab)
        params.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        c = 0
        tb.Label(params, text="TPS:", font=("Segoe UI", 9)).grid(row=0, column=c, sticky="w", padx=(0, 2))
        c += 1
        self.tps_entry = tb.Entry(params, textvariable=self.tps_var, width=10)
        self.tps_entry.grid(row=0, column=c, padx=(0, 8))
        c += 1
        tb.Label(params, text="Time (s):", font=("Segoe UI", 9)).grid(row=0, column=c, sticky="w", padx=(0, 2))
        c += 1
        self.time_entry = tb.Entry(params, textvariable=self.time_var, width=10)
        self.time_entry.grid(row=0, column=c, padx=(0, 8))
        c += 1
        tb.Label(params, text="Size:", font=("Segoe UI", 9)).grid(row=0, column=c, sticky="w", padx=(0, 2))
        c += 1
        self.size_entry = tb.Entry(params, textvariable=self.size_var, width=10)
        self.size_entry.grid(row=0, column=c)

        params2 = tb.Frame(manual_tab)
        params2.grid(row=1, column=0, sticky="ew")
        tb.Label(params2, text="Scramble:", font=("Segoe UI", 9)).pack(side="left")
        self.scramble_entry = tb.Entry(params2, textvariable=self.scramble_var, width=22)
        self.scramble_entry.pack(side="left", padx=(4, 8))
        tb.Label(params2, text="Movetimes:", font=("Segoe UI", 9)).pack(side="left")
        self.movetimes_entry = tb.Entry(params2, textvariable=self.movetimes_var, width=22)
        self.movetimes_entry.pack(side="left", padx=(4, 0))

        tb.Label(manual_tab, text="Solution strings (one per line):",
                 font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.solution_text = scrolledtext.ScrolledText(
            manual_tab, font=("Consolas", 10),
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="#fff",
            relief="flat", borderwidth=0, highlightthickness=1,
            highlightbackground="#3a3a3a", highlightcolor="#3a3a3a")
        self.solution_text.grid(row=3, column=0, sticky="nsew", pady=(2, 0))
        self.solution_text.insert("1.0", "# solutions here, one per line\n")
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
        prog_frame.configure(height=100)

        self.progress_bar = tb.Progressbar(prog_frame, mode="determinate",
                                           bootstyle="success-striped")
        self.progress_bar.pack(fill="x", padx=8, pady=(8, 2))

        self.progress_text = tk.StringVar(value="")
        prog_label = tb.Label(prog_frame, textvariable=self.progress_text,
                              font=("Segoe UI", 9), anchor="w")
        prog_label.pack(fill="x", padx=8, pady=(0, 6))

        # ── Generated replays ──
        lst_frame = tb.LabelFrame(right, text="Generated Replays")
        lst_frame.grid(row=1, column=0, sticky="nsew")
        lst_frame.grid_rowconfigure(0, weight=1)
        lst_frame.grid_columnconfigure(0, weight=1)

        self.replay_listbox = tk.Listbox(
            lst_frame, font=("Consolas", 9), activestyle="none",
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

        # ── Separator + status at bottom of right column ──
        tb.Separator(right).grid(row=2, column=0, sticky="ew", pady=(6, 4))
        tb.Label(right, textvariable=self.status_var,
                 font=("Segoe UI", 9, "bold"), bootstyle="secondary").grid(row=3, column=0, sticky="w")

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

    def _active_tab(self):
        return self.nb.index(self.nb.select())

    def _generate(self):
        if self._executor and not all(f.done() for f in self._batch_futures):
            return

        tab = self._active_tab()
        items = []

        if tab == 0:
            raw = self.url_text.get("1.0", "end-1c").strip()
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith(("http://", "https://")):
                    items.append(("url", line))
        else:
            raw = self.solution_text.get("1.0", "end-1c").strip()
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                items.append(("manual", line))

        if not items:
            self.status_var.set("No valid entries.")
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
        os.makedirs(output_dir, exist_ok=True)

        total = len(items)
        self._item_progress = {}
        for idx in range(total):
            self._item_progress[idx] = {"phase": 0, "prev_cur": 0,
                                        "adjusted_cur": 0, "adjusted_tot": 1,
                                        "done": False, "path": None, "error": None}

        self._batch_futures = []
        self._executor = ThreadPoolExecutor(max_workers=4)

        def on_done(idx, fut):
            try:
                fut.result()
            except Exception as e:
                self._item_progress[idx]["error"] = str(e)
            self._item_progress[idx]["done"] = True

        for idx, (mode, input_str) in enumerate(items):
            fut = self._executor.submit(
                self._process_item, idx, mode, input_str, output_dir, total)
            fut.add_done_callback(lambda f, i=idx: on_done(i, f))
            self._batch_futures.append(fut)

        self.after(200, self._poll_batch)

    def _process_item(self, idx, mode, input_str, output_dir, total):
        try:
            params = {
                "force_fringe": self.force_fringe_var.get(),
                "quality": self.quality_var.get(),
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

            filename_tps = params.get("tps", tps if mode == "url" else None)
            filename_time = params.get("time", None)
            base_name = _generate_filename(
                solution, filename_tps, filename_time,
                params.get("movetimes", -1), params.get("size"))
            out_path = _pick_output_filename(output_dir, base_name)

            def on_progress(cur, tot):
                self._on_item_progress(idx, cur, tot)

            gen = ReplayVideoGenerator(cleanup_frames=False)
            gen.generate_simple_replay(
                solution=solution, output_path=out_path,
                show_progress=False, external_progress_cb=on_progress, **params)

            if not self.cancel_flag:
                self._item_progress[idx]["path"] = out_path
                self.after(0, lambda p=out_path: self._add_to_list(p))
        except Exception as e:
            self._item_progress[idx]["error"] = str(e)
            self.after(0, lambda m=f"Item {idx+1} failed: {e}": self.status_var.set(m))

    def _on_item_progress(self, idx, raw_cur, raw_tot):
        item = self._item_progress[idx]
        if raw_cur < item["prev_cur"]:
            item["phase"] += 1
        item["prev_cur"] = raw_cur
        # fixed denominator: always 2 phases (frames + encode)
        item["adjusted_cur"] = raw_cur + item["phase"] * raw_tot
        item["adjusted_tot"] = raw_tot * 2

    def _cancel(self):
        self.cancel_flag = True
        if self._executor:
            self._executor.shutdown(wait=False)
        self.status_var.set("Cancelling...")

    def _poll_batch(self):
        if not self._batch_futures:
            return

        total = len(self._batch_futures)
        done_count = sum(1 for f in self._batch_futures if f.done())

        # item-weighted overall progress (no snapping)
        overall_pct = 0.0
        running = 0
        for p in self._item_progress.values():
            share = 1.0 / total
            completion = min(p["adjusted_cur"] / p["adjusted_tot"], 1.0) if p["adjusted_tot"] > 0 else 0
            overall_pct += share * completion
            if not p["done"]:
                running += 1

        overall_pct *= 100

        # Rolling rate (EMA) for accurate ETA across fast/slow phases
        now = time.time()
        dt = now - self._last_poll_time
        dp = overall_pct - self._last_poll_pct
        if dt > 0.05 and dp >= 0:
            inst = dp / dt
            self._rolling_rate = inst if self._rolling_rate <= 0 else self._rolling_rate * 0.7 + inst * 0.3
        self._last_poll_time = now
        self._last_poll_pct = overall_pct

        elapsed = time.time() - self._start_time
        elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{int(elapsed//60)}m {elapsed%60:.0f}s"

        # Cap at 99% while items are still finalising (ffmpeg post-processing)
        display_pct = min(overall_pct, 99.0) if running else overall_pct
        self.progress_bar["value"] = display_pct

        eta_str = ""
        if overall_pct > 1 and running:
            eta = (100 - overall_pct) / self._rolling_rate if self._rolling_rate > 0 else 0
            if eta < 3600:
                eta_str = f" — ETA {eta:.1f}s" if eta < 60 else f" — ETA {int(eta//60)}m {eta%60:.0f}s"

        parts = []
        if running:
            parts.append(f"{running} active")
        if done_count:
            parts.append(f"{done_count}/{total} completed")
        label = " — ".join(parts) + f" ({display_pct:.0f}%)" if parts else ""
        self.progress_text.set(f"{label}{eta_str}")
        self.status_var.set(f"{elapsed_str} {label}")

        if done_count == total:
            errors = sum(1 for p in self._item_progress.values() if p["error"])
            ok_count = done_count - errors
            took = f"took {elapsed_str}"
            msg = f"Done — {ok_count} replay(s) generated. {took}" + (f" {errors} failed." if errors else "")
            self.progress_text.set(msg)
            self.status_var.set(msg)
            self.progress_bar["value"] = 100
            self._set_ui_busy(False)
            if self._executor:
                self._executor.shutdown(wait=False)
                self._executor = None
            self._batch_futures = []
            return

        self.after(200, self._poll_batch)

    def _add_to_list(self, path):
        self.generated_files.append(path)
        self.replay_listbox.insert("end", os.path.basename(path))

    def _open_selected(self, event=None):
        sel = self.replay_listbox.curselection()
        if sel and sel[0] < len(self.generated_files):
            path = self.generated_files[sel[0]]
            if os.path.exists(path):
                os.startfile(path)
            else:
                self.status_var.set(f"File not found: {path}")

    def _open_folder(self):
        if self.generated_files:
            os.startfile(os.path.dirname(self.generated_files[-1]))

    def _clear_list(self):
        self.generated_files.clear()
        self.replay_listbox.delete(0, "end")

    def _set_ui_busy(self, busy):
        st = "disabled" if busy else "normal"
        self.gen_btn.config(state=st)
        self.cancel_btn.config(state=st if busy else "disabled")

    def _on_close(self):
        self.cancel_flag = True
        if self._executor:
            self._executor.shutdown(wait=False)
        self.destroy()


if __name__ == "__main__":
    ReplayGUI().mainloop()

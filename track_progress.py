"""
track_progress.py - Unified progress tracking across all render paths.

Single source of truth for phase weights and progress bar rendering.
Used by CLI (terminal bar), GUI (external callback), and batch modes.
"""

import sys
import shutil
import time as time_module
from typing import Optional, Callable

# ─── Phase weight constants ──────────────────────────────────────

CPU_PHASE_WEIGHTS = [2, 3, 33, 62]       # Analysis, Precompute, Render, Encode
GPU_PHASE_WEIGHTS = [2, 3, 95]            # Analysis, Precompute, GPU-Render
BATCH_PHASE_WEIGHTS = [100]                # single-phase item-level progress


class ProgressTracker:
    """Unified progress tracker.

    Tracks progress through a sequence of *phases*, each with a
    configurable weight.  Phase transitions are detected either by a
    *description change* (e.g. ``desc="Precompute"`` vs ``desc="Render"``)
    or by the raw‑current value rolling over (decreasing).  All progress
    is normalised to 0–100 so callers never need phase details.

    Optionally renders a terminal progress bar (``show_terminal=True``)
    and/or forwards normalised values to an *external callback* (GUI).

    Usage::

        tracker = ProgressTracker(
            total=100, desc="Render",
            phase_weights=CPU_PHASE_WEIGHTS,
            external_cb=gui_callback,
            show_terminal=True,
        )

        tracker.update(1, 1, desc="Analysis")
        tracker.update(1, N, desc="Precompute")
        # … more precompute ticks …
        tracker.update(N, N)
        tracker.update(1, M, desc="Render")
        # … etc …
        tracker.finish()
    """

    def __init__(
        self,
        total: int,
        desc: str = "Generating frames",
        phase_weights: Optional[list] = None,
        external_cb: Optional[Callable] = None,
        hide_rate: bool = False,
        show_terminal: bool = True,
    ):
        self.total = total
        self.desc = desc
        self.phase_weights = phase_weights if phase_weights is not None else CPU_PHASE_WEIGHTS
        self.external_cb = external_cb
        self.hide_rate = hide_rate
        self.show_terminal = show_terminal

        # Timing / rate
        self.start_time = time_module.time()
        self.last_update_time = self.start_time
        self.last_current = 0
        self.window_rate = 0.0
        self._last_print_time = -999.0

        # Phase state
        self._prev_cur: Optional[int] = None
        self.phase_idx = 0
        self._phase_start = 0
        self._phase_weight = self.phase_weights[0] if self.phase_weights else 100

        # GPU stats passthrough
        self.gpu_stats: Optional[dict] = None

        # Terminal rendering
        self._max_line_width = 0
        self._term_width = 120
        self._is_tty = False
        if show_terminal:
            self._is_tty = getattr(sys.stdout, 'isatty', lambda: False)()
            if self._is_tty:
                try:
                    self._term_width = shutil.get_terminal_size().columns
                except Exception:
                    pass

    # ── Public API ────────────────────────────────────────────────

    def update(
        self,
        current: int,
        total: int,
        desc: Optional[str] = None,
        use_gpu: bool = False,
        gpu_stats: Optional[dict] = None,
        **extra,
    ):
        """Report one progress tick.

        Parameters
        ----------
        current : int
            Raw progress within the current phase (0‑ or 1‑based).
        total : int
            Total for the current phase.
        desc : str or None
            Phase description.  When set and different from the previous
            description, a phase transition is triggered (the desc also
            updates the terminal bar label).
        use_gpu : bool
            Whether the current phase runs on GPU (forwarded to external
            callback).
        gpu_stats : dict or None
            GPU statistics dict (forwarded to external callback).
        """
        desc_changed = bool(desc is not None and desc != self.desc)
        phase_transition = self._prev_cur is not None and current < self._prev_cur
        is_new_phase = phase_transition or desc_changed

        if desc_changed:
            self.desc = desc

        if is_new_phase:
            if self._prev_cur is not None:
                self.phase_idx += 1
            w = (
                self.phase_weights[self.phase_idx]
                if self.phase_idx < len(self.phase_weights)
                else 100 - sum(self.phase_weights)
            )
            self._phase_start = sum(self.phase_weights[: self.phase_idx])
            self._phase_weight = w
            self.last_current = self._phase_start
            self.last_update_time = time_module.time()

        self._prev_cur = current

        # Normalise to 0‑100
        frac = current / total if total > 0 else 0
        overall_pct = self._phase_start + self._phase_weight * frac
        out_cur = int(round(overall_pct))
        self.total = 100

        if gpu_stats is not None:
            self.gpu_stats = gpu_stats

        # Terminal bar
        if self.show_terminal:
            self._render_terminal(out_cur, force=is_new_phase)

        # External callback (GUI)
        if self.external_cb:
            self.external_cb(out_cur, 100, gpu_stats=gpu_stats, use_gpu=use_gpu, **extra)

    __call__ = update

    def finish(self):
        """Force progress to 100 % and, in terminal mode, print a final newline."""
        self.total = 100
        self._render_terminal(100, force=True)
        if self.show_terminal:
            print()

    # ── Terminal rendering helpers ────────────────────────────────

    def _render_terminal(self, out_cur: int, force: bool = False):
        now = time_module.time()
        if out_cur < 100 and now - self._last_print_time < 1.0 and not force:
            return
        elapsed = now - self.start_time
        window_elapsed = now - self.last_update_time
        if window_elapsed > 0.5 and out_cur > self.last_current:
            instant = (out_cur - self.last_current) / window_elapsed
            if self.window_rate <= 0:
                self.window_rate = instant
            else:
                self.window_rate = self.window_rate * 0.5 + instant * 0.5
            self.last_update_time = now
            self.last_current = out_cur

        rate = self.window_rate if self.window_rate > 0 else (out_cur / elapsed if elapsed > 0 else 0)
        remaining = 100 - out_cur
        eta = remaining / rate if rate > 0 else 0
        line = self._build_line(out_cur, elapsed, rate, eta)

        if self._is_tty:
            self._max_line_width = max(self._max_line_width, len(line))
            print(f"\r{line.ljust(self._max_line_width)}", end="", flush=True)
        else:
            print(line, flush=True)
        self._last_print_time = now

    def _build_line(self, current: int, elapsed: float, rate: float, eta: float) -> str:
        frac = current / self.total if self.total > 0 else 0
        pct = frac * 100
        total_t = elapsed + eta
        if self.hide_rate:
            suffix = f" {pct:.0f}% | {self._time_str(elapsed)}/{self._time_str(total_t)}"
        else:
            suffix = f" {pct:.0f}% | {rate:.0f}/s | {self._time_str(elapsed)}/{self._time_str(total_t)}"

        if self.gpu_stats and self.gpu_stats.get("batch_size"):
            s = self.gpu_stats
            mb = s.get("mem_used_mb", 0) / 1024
            gb = s.get("total_mem_mb", 0) / 1024
            suffix += f" | {mb:.1f}/{gb:.1f}GB | Batch: {s.get('batch_size', 0)}"

        prefix = f"{self.desc}: ["
        fixed_len = len(prefix) + 1 + len(suffix)
        bar_w = max(2, min(40, self._term_width - fixed_len))
        filled = int(bar_w * frac)
        bar = "#" * filled + "-" * (bar_w - filled)
        return f"{prefix}{bar}]{suffix}"

    @staticmethod
    def _time_str(t: float) -> str:
        total_sec = int(round(t))
        if total_sec >= 3600:
            return f"{total_sec // 3600}h{total_sec % 3600 // 60}m"
        if total_sec >= 60:
            return f"{total_sec // 60}m{total_sec % 60}s"
        return f"{t:.1f}s"

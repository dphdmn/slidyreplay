"""Debug logging for replay video generator GUI."""
import logging
import os
import sys
import time
import psutil as _psutil
from datetime import datetime

_LOG = None
_BASELINE_RAM = None


class CancelError(Exception):
    pass


def get_logger():
    global _LOG
    if _LOG is not None:
        return _LOG
    _LOG = logging.getLogger("replay_debug")
    _LOG.setLevel(logging.DEBUG)
    return _LOG


def init_logfile():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    fname = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    path = os.path.join(log_dir, fname)
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d [%(threadName)s] %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    get_logger().addHandler(fh)
    return path


def reset_ram_baseline():
    global _BASELINE_RAM
    _BASELINE_RAM = _psutil.Process().memory_info().rss


def log_ram(label: str = "") -> int:
    global _BASELINE_RAM
    if _BASELINE_RAM is None:
        reset_ram_baseline()
    cur = _psutil.Process().memory_info().rss
    delta = cur - _BASELINE_RAM
    get_logger().info(f"  RAM [{label}]: {cur // (1024*1024)}MB ({delta // (1024*1024):+d}MB vs baseline)")
    return delta

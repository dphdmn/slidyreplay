"""Debug logging for replay video generator GUI."""
import logging
import os
import sys
import time
from datetime import datetime

_LOG = None

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

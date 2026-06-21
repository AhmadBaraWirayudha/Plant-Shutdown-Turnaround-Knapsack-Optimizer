"""
helpers.py — Logging, formatting, and timing utilities.
"""

from __future__ import annotations
import logging
import time
import functools
from pathlib import Path
from datetime import datetime


# ─── Logger factory ──────────────────────────────────────────────────────────
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = get_logger("helpers")


# ─── Timing decorator ─────────────────────────────────────────────────────────
def timed(fn):
    """Decorator: log elapsed time of any function call."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        log.info("⏱  %s completed in %.3f s", fn.__qualname__, elapsed)
        return result

    return wrapper


# ─── Formatting helpers ───────────────────────────────────────────────────────
def fmt_usd(value: float) -> str:
    """Format a dollar amount as  $1,234,567."""
    return f"${value:>12,.0f}"


def fmt_pct(value: float) -> str:
    """Format a fraction as percentage string."""
    return f"{value * 100:.1f} %"


def fmt_hours(value: float) -> str:
    """Format craft-hours."""
    return f"{value:>8,.0f} h"


# ─── Run-log writer ───────────────────────────────────────────────────────────
def write_run_log(output_dir: Path, metadata: dict) -> Path:
    """Persist a JSON audit trail of every optimizer run."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"run_log_{ts}.json"
    with open(log_path, "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    log.info("📋  Audit log saved → %s", log_path)
    return log_path


# ─── Banner printer ───────────────────────────────────────────────────────────
BANNER = r"""
╔══════════════════════════════════════════════════════════════════════╗
║   ██████╗ ██╗      █████╗ ███╗   ██╗████████╗                       ║
║   ██╔══██╗██║     ██╔══██╗████╗  ██║╚══██╔══╝                       ║
║   ██████╔╝██║     ███████║██╔██╗ ██║   ██║                          ║
║   ██╔═══╝ ██║     ██╔══██║██║╚██╗██║   ██║                          ║
║   ██║     ███████╗██║  ██║██║ ╚████║   ██║                          ║
║   ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝                          ║
║                                                                      ║
║   SHUTDOWN TURNAROUND KNAPSACK OPTIMIZER  v1.0                       ║
║   Google OR-Tools CP-SAT  ·  Weibull Analysis  ·  ILP Scheduling    ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def print_banner():
    print(BANNER)

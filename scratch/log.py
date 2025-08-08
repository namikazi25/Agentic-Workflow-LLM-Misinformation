"""
scratch/log.py – unified logging **plus** lightweight metrics reporting
======================================================================

Key additions
-------------
1. **Exit-time metrics dump**
   When `log.setup()` is called, we register an `atexit` hook that prints a
   one-line summary of counters collected via `scratch.metrics` (see the new
   module).  This makes it trivial to see, e.g., *% of branches with empty
   snippets* without changing existing pipeline code.

2. **Convenient `log.metric(name, delta=1)` helper**
   Modules can simply call `log.metric("empty_snippet_branch")` to increment
   the counter.  Internally this proxies to `scratch.metrics.inc()` so there
   is no circular import at module top-level.

Backward compatibility
----------------------
The basic `setup()` behaviour and coloured console/file logging are unchanged.
Other modules that previously imported `scratch.log` continue to work as
before.
"""

from __future__ import annotations

import atexit
import logging
import sys
from types import MappingProxyType
from typing import Optional

# --------------------------------------------------------------------------- #
# ANSI colours (only applied if sys.stderr.isatty())
# --------------------------------------------------------------------------- #

_COLOUR = MappingProxyType(
    {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[41m",  # red bg
    }
)
_RESET = "\033[0m"


def _colourise(level_name: str, message: str) -> str:
    if not sys.stderr.isatty():
        return message
    colour = _COLOUR.get(level_name, "")
    return f"{colour}{message}{_RESET}" if colour else message


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def metric(name: str, delta: int = 1):
    """Increment *name* counter via `scratch.metrics` when available."""
    try:
        from . import metrics as _m  # local import to avoid hard dependency

        _m.inc(name, delta)
    except Exception:  # noqa: BLE001
        # Metrics are best-effort; never crash the caller
        pass


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

def setup(level: str | int = "INFO", file: str | None = None) -> None:
    """Initialise root logger and register the exit-time metrics reporter."""
    level_int = logging.getLevelName(level) if isinstance(level, str) else level

    # -------------------------- handlers -------------------------- #
    handlers: list[logging.Handler] = []

    console = logging.StreamHandler(sys.stderr)
    handlers.append(console)

    if file:
        handlers.append(logging.FileHandler(file, encoding="utf-8"))

    # -------------------------- formatter ------------------------- #
    class _Formatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # noqa: D401
            base = super().format(record)
            if record.levelno >= logging.WARNING and isinstance(
                record.__dict__.get("handler", record), logging.StreamHandler
            ):
                base = _colourise(record.levelname, base)
            return base

    fmt = "%(asctime)s  %(levelname)8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = _Formatter(fmt, datefmt=datefmt)

    for h in handlers:
        h.setFormatter(formatter)

    logging.basicConfig(level=level_int, handlers=handlers, force=True)

    # Exit-hook prints metrics summary if available
    def _report_metrics():
        try:
            from . import metrics as _m

            summary = _m.summary()
            if not summary:
                return
            logging.info("==== Pipeline metrics ====\n%s", "  ".join(f"{k}={v}" for k, v in summary.items()))
            # Optionally flush traces
            _m.flush_traces()
        except Exception:  # noqa: BLE001
            pass  # never break exit path

    atexit.register(_report_metrics)

    logging.getLogger(__name__).debug("Logger initialised (level=%s)", level)

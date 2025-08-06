"""
scratch/log.py
==============

Tiny helper around Python's `logging` that:

  • sets a uniform, timestamped format for *all* project modules  
  • colours log-levels when running in a TTY (falls back gracefully)  
  • allows optional file logging alongside console output

Usage
-----

>>> from scratch import log
>>> log.setup(level="DEBUG", file="run.log")
>>> import logging
>>> logging.info("Hello")

Call `log.setup()` **once**, ideally at the top of `main_async.py`
(or `__main__.py`).  After that, plain `import logging` in any module
will inherit the configuration.
"""

from __future__ import annotations
import logging
import sys
from types import MappingProxyType

# --------------------------------------------------------------------------- #
# ANSI colours (only applied if sys.stderr.isatty())
# --------------------------------------------------------------------------- #

_COLOUR = MappingProxyType({
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[41m",   # red bg
})
_RESET = "\033[0m"

def _colourise(level_name: str, message: str) -> str:
    if not sys.stderr.isatty():
        return message
    colour = _COLOUR.get(level_name, "")
    return f"{colour}{message}{_RESET}" if colour else message

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def setup(level: str | int = "INFO", file: str | None = None) -> None:
    """
    Initialise root logger.

    Parameters
    ----------
    level : str | int
        Logging level (e.g. "INFO", "DEBUG").
    file : str | None
        Optional path to log-file (appends if exists).
    """
    # Convert level string → int
    level_int = logging.getLevelName(level) if isinstance(level, str) else level

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #
    handlers: list[logging.Handler] = []

    # Console
    console = logging.StreamHandler(sys.stderr)
    handlers.append(console)

    # File
    if file:
        handlers.append(logging.FileHandler(file, encoding="utf-8"))

    # ------------------------------------------------------------------ #
    # Formatter
    # ------------------------------------------------------------------ #
    class _Formatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # noqa: D401
            # Base message
            base = super().format(record)
            # Colourise level-name for console only
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

    # ------------------------------------------------------------------ #
    # Configure root
    # ------------------------------------------------------------------ #
    logging.basicConfig(
        level=level_int,
        handlers=handlers,
        force=True,        # override any prior configuration
    )

    logging.getLogger(__name__).debug("Logger initialised (level=%s)", level)

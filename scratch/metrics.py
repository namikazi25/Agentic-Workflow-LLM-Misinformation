"""
scratch/metrics.py – ultra-light metrics & traces
=================================================

What it provides:
- inc(name, delta=1)        → bump counters safely
- summary()                 → snapshot of all counters (dict)
- trace(event, payload=None)→ append a small JSON-serialisable trace
- flush_traces(path=None)   → write traces to JSONL (default: metrics_traces.jsonl)

Notes:
- Designed to be *best-effort*; failures never crash the app.
- Thread/async-safe via a simple Lock.
- Traces are capped (deque) to avoid unbounded growth.
- `log.setup()` already calls summary() and flush_traces() at exit.
"""

from __future__ import annotations

import json
import os
from collections import Counter, deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Optional

# ------------------------------ config knobs ------------------------------ #

# Max number of recent traces kept in memory (can be overridden by env)
_TRACE_MAX = int(os.getenv("METRICS_TRACE_MAX", "2000"))

# Default output file for traces when flush_traces() has no explicit path
_DEFAULT_TRACE_PATH = os.getenv("METRICS_TRACE_PATH", "metrics_traces.jsonl")

# ------------------------------ state ------------------------------------- #

_COUNTS: Counter[str] = Counter()
_TRACES = deque(maxlen=_TRACE_MAX)
_LOCK = Lock()

# If we ever need to avoid double-write on multiple flush calls
# we just clear the deque after a flush; no need for extra flags.


# ------------------------------ helpers ----------------------------------- #

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------ API --------------------------------------- #

def inc(name: str, delta: int = 1) -> None:
    """
    Increment a metric counter by *delta* (default 1).
    Safe to call from anywhere; never raises.
    """
    if not name:
        return
    try:
        with _LOCK:
            _COUNTS[name] += int(delta)
    except Exception:
        # Best-effort: never crash caller
        pass


def get(name: str) -> int:
    """Return the current value of a counter (0 if missing)."""
    try:
        with _LOCK:
            return int(_COUNTS.get(name, 0))
    except Exception:
        return 0


def summary() -> Dict[str, int]:
    """
    Return a **copy** of all counters as a plain dict.
    Safe to print or JSON-serialise.
    """
    try:
        with _LOCK:
            # shallow copy to avoid exposing internal Counter
            return dict(_COUNTS)
    except Exception:
        return {}


def trace(event: str, payload: Any | None = None) -> None:
    """
    Append a lightweight trace event.
    `payload` should be JSON-serialisable (dict/str/number/etc).
    """
    if not event:
        return
    try:
        item = {
            "ts": _now_iso(),
            "event": str(event),
            "payload": payload,
        }
        with _LOCK:
            _TRACES.append(item)
    except Exception:
        pass


def flush_traces(path: Optional[str] = None) -> Optional[str]:
    """
    Write all buffered traces to a JSONL file and clear the buffer.
    Returns the path written, or None on failure.
    """
    target = path or _DEFAULT_TRACE_PATH
    try:
        # Grab a snapshot then clear, under lock
        with _LOCK:
            if not _TRACES:
                return target
            snapshot = list(_TRACES)
            _TRACES.clear()

        # Ensure directory exists
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)

        # Append JSON lines
        with open(target, "a", encoding="utf-8") as fp:
            for item in snapshot:
                fp.write(json.dumps(item, ensure_ascii=False) + "\n")

        return target
    except Exception:
        # Swallow errors silently; metrics are best-effort
        return None


# ------------------------------ optional test hooks ------------------------ #

def _reset_all_for_tests() -> None:  # pragma: no cover
    """Clear counters and traces (intended for tests)."""
    try:
        with _LOCK:
            _COUNTS.clear()
            _TRACES.clear()
    except Exception:
        pass

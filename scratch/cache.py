"""
scratch/cache.py
================

Tiny utilities for in-memory caching used across the rewritten pipeline.

* ``NET_CACHE`` – a shared **TTLCache** for network-bound data
  (e.g. Brave search snippets).  Expiry is governed by
  ``config.CACHE_TTL_SEC``.

* ``memo`` – decorator factory for **LRU memoisation** of pure functions
  (useful for expensive but deterministic helpers).

* ``ttl_get / ttl_set`` – wrappers that hide ``KeyError`` so callers don’t
  need try/except every time.

These helpers deliberately have **no third-party dependencies** beyond
``cachetools``, which is already in ``requirements.txt``.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from cachetools import TTLCache

from . import config as C

# --------------------------------------------------------------------------- #
# Shared TTL cache for network results
# --------------------------------------------------------------------------- #

NET_CACHE: TTLCache[str, Any] = TTLCache(
    maxsize=20_000,          # arbitrary high number; evicts LRU on overflow
    ttl=C.CACHE_TTL_SEC,     # seconds until entry expires
)

# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #


def ttl_get(cache: TTLCache, key: str) -> Any | None:
    """Return cached value or *None* (swallows KeyError)."""
    try:
        return cache[key]
    except KeyError:
        return None


def ttl_set(cache: TTLCache, key: str, value: Any) -> None:
    """Store *value* under *key* in the given cache."""
    cache[key] = value


# --------------------------------------------------------------------------- #
# LRU memoisation decorator
# --------------------------------------------------------------------------- #

T = TypeVar("T")


def memo(maxsize: int = 512) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator factory that applies :pyfunc:`functools.lru_cache`
    with the chosen *maxsize*.

    Example
    -------
    >>> from scratch.cache import memo
    >>>
    >>> @memo(256)
    ... def expensive(x, y):
    ...     ...

    The returned wrapper exposes ``.cache_info()`` and ``.cache_clear()``.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        cached_fn = functools.lru_cache(maxsize=maxsize)(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):  # type: ignore[override]
            return cached_fn(*args, **kwargs)

        # re-export useful methods
        wrapper.cache_info = cached_fn.cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = cached_fn.cache_clear  # type: ignore[attr-defined]

        return wrapper

    return decorator

"""
scratch/brave_client.py
=======================

Asynchronous thin-wrapper around the **Brave Search API** that returns a list
of result snippets (`str`).  Designed to be called from coroutines in
`webqa.py`, but a synchronous helper is provided for quick debugging /
unit-tests.

Key features
------------

* **Async I/O** (`aiohttp`)  – dozens of queries in flight.
* **Retry with back-off** (`tenacity`) on network / 429 errors.
* **TTL LRU cache**         – avoids duplicate queries within the same run.
* **Pure snippets**         – no LLM summarisation to prevent hallucinations.

Environment
-----------

You must set **one** of:

* ``BRAVE_SEARCH_API_KEY``   – official env‐var name  
* ``SEARCH_API``             – fallback key expected by older code

If both are absent, an informative `RuntimeError` is raised.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List

import aiohttp
import async_timeout
from tenacity import retry, stop_after_attempt, wait_exponential

from . import config as C
from .cache import NET_CACHE, ttl_get, ttl_set

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_KEY: str | None = os.getenv("BRAVE_SEARCH_API_KEY") or os.getenv("SEARCH_API")

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Async snippet fetcher
# --------------------------------------------------------------------------- #


@retry(
    stop=stop_after_attempt(3),            # max 3 tries
    wait=wait_exponential(multiplier=1.2), # 1.2, 2.4, 4.8 sec …
    reraise=True,
)
async def brave_snippets(query: str, *, k: int = C.BRAVE_K) -> List[str]:
    """
    Coroutine that returns up to *k* snippets for *query*.

    Results are cached for `config.CACHE_TTL_SEC` seconds in memory.
    """
    # ----------------- cache hit ----------------- #
    hit = ttl_get(NET_CACHE, query)
    if hit is not None:
        return hit[:k]

    # ----------------- API call ------------------ #
    if not BRAVE_KEY:
        raise RuntimeError(
            "Brave API key not found.  "
            "Set BRAVE_SEARCH_API_KEY or SEARCH_API in your environment."
        )

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_KEY,
    }
    params = {"q": query, "count": k}

    async with async_timeout.timeout(10):            # network timeout
        async with aiohttp.ClientSession() as sess:
            async with sess.get(BRAVE_ENDPOINT, params=params, headers=headers) as r:
                r.raise_for_status()
                data = await r.json()

    # ----------------- parse --------------------- #
    snippets: List[str] = [
        (res.get("description") or res.get("title") or "").strip()
        for res in data.get("web", {}).get("results", [])
        if (res.get("description") or res.get("title"))
    ]

    # ----------------- cache store --------------- #
    ttl_set(NET_CACHE, query, snippets)

    # Log if no results (helps debugging selector failures)
    if not snippets:
        logger.debug("Brave returned 0 snippets for query %r", query)

    return snippets[:k]


# --------------------------------------------------------------------------- #
# Synchronous convenience wrapper
# --------------------------------------------------------------------------- #


def brave_snippets_sync(query: str, *, k: int = C.BRAVE_K) -> List[str]:
    """
    Blocking helper for REPL / tests:

    >>> brave_snippets_sync("What is AI?", k=5)
    """
    return asyncio.run(brave_snippets(query, k=k))

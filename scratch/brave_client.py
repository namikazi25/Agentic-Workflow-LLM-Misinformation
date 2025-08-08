"""
scratch/brave_client.py – improved Brave Search wrapper
=======================================================

Changes versus the first rewrite
--------------------------------
1. **Quality‑aware post‑filtering**
   • After fetching raw results we compute a simple keyword‑overlap score
     between the *question* and each *snippet* (stop‑words removed).
   • Snippets that score **0** or look like obvious shopping / spam ads are
     discarded.
   • The top‑scoring *k* snippets are returned.

2. **Adaptive widening**
   • If the first call (count = 20) yields **< k usable snippets**, we retry
     once with `count = 50` to fish for long‑tail hits.

3. **Structured return for debugging**
   • When `DEBUG=True` in `config.py`, we log how many snippets were fetched,
     filtered, and returned (helps monitor search health).

Public contract remains **identical**:

>>> snippets = await brave_snippets("What is AI?", k=8)

The synchronous helper `brave_snippets_sync` is preserved.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
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
# Minimal stop‑word list (hard‑coded to avoid extra deps)
# --------------------------------------------------------------------------- #
_STOP = {
    "the", "a", "an", "and", "or", "but", "on", "in", "at", "to", "of", "for",
    "with", "by", "about", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "from", "into", "than",
}

_SPAM_PAT = re.compile(r"\b(?:buy|shop|sale|price|discount|free shipping)\b", re.I)


# --------------------------------------------------------------------------- #
# Helper: score snippet quality given the question tokens
# --------------------------------------------------------------------------- #

def _tokenise(text: str) -> set[str]:
    return {tok.lower() for tok in re.findall(r"[A-Za-z0-9]+", text) if tok.lower() not in _STOP}


def _score_snippet(q_tokens: set[str], snippet: str) -> int:
    if _SPAM_PAT.search(snippet):
        return 0  # auto‑discard spammy looking ads
    s_tokens = _tokenise(snippet)
    return len(q_tokens & s_tokens)


# --------------------------------------------------------------------------- #
# Async snippet fetcher with quality filtering
# --------------------------------------------------------------------------- #

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.2), reraise=True)
async def brave_snippets(query: str, *, k: int = C.BRAVE_K) -> List[str]:
    """Return up to *k* **high‑quality** Brave snippets for *query*."""
    if not BRAVE_KEY:
        raise RuntimeError("Brave API key not found – set BRAVE_SEARCH_API_KEY or SEARCH_API.")

    # --- cache hit -------------------------------------------------- #
    hit = ttl_get(NET_CACHE, query)
    if hit is not None:
        return hit[:k]

    q_tokens = _tokenise(query)

    async def _fetch(count: int) -> List[dict]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_KEY,
        }
        params = {"q": query, "count": count}
        async with async_timeout.timeout(10):
            async with aiohttp.ClientSession() as sess:
                async with sess.get(BRAVE_ENDPOINT, params=params, headers=headers) as r:
                    r.raise_for_status()
                    data = await r.json()
                    return data.get("web", {}).get("results", [])

    # 1️⃣  first attempt (count=20) ---------------------------------- #
    raw_results = await _fetch(20)

    # 2️⃣  optional widen (count=50) if needed ----------------------- #
    if len(raw_results) < k:
        raw_results += await _fetch(50)

    # 3️⃣  extract + score ------------------------------------------ #
    scored: List[tuple[int, str]] = []
    for res in raw_results:
        text = (res.get("description") or res.get("title") or "").strip()
        if not text:
            continue
        score = _score_snippet(q_tokens, text)
        if score == 0:
            continue  # drop uninformative or spam
        scored.append((score, text))

    # Sort high‑to‑low score then by original order stability
    scored.sort(key=lambda t: (-t[0], raw_results.index(next(r for r in raw_results if (r.get("description") or r.get("title") or "").strip() == t[1]))))

    snippets = [txt for _, txt in scored][:k]

    # 4️⃣  cache + debug log ----------------------------------------- #
    ttl_set(NET_CACHE, query, snippets)

    if C.DEBUG:
        logger.info(
            "Brave: fetched %d, kept %d (requested %d) for query=%r",
            len(raw_results), len(snippets), k, query[:80],
        )

    return snippets


# --------------------------------------------------------------------------- #
# Sync wrapper for tests / REPL
# --------------------------------------------------------------------------- #

def brave_snippets_sync(query: str, *, k: int = C.BRAVE_K) -> List[str]:
    """Blocking helper (runs the async coroutine internally)."""
    return asyncio.run(brave_snippets(query, k=k))

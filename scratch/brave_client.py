"""
scratch/brave_client.py – improved Brave Search wrapper
=======================================================

Adds headline-aware scoring:
• A snippet must overlap tokens from the **question** and, if provided, the
  **headline**. We score each snippet as min(overlap_q, overlap_h). If no
  headline is supplied we fall back to question-only overlap.

scratch/brave_client.py – Brave Search wrapper (headline-aware + anti-spam + adaptive freshness)
===============================================================================================

Adds:
• Headline-aware scoring: snippet must overlap with BOTH question and headline.
• Expanded anti-spam filter: blocks obvious shopping/stock-image/linkbait.
• Adaptive freshness: caller can pass freshness_days=None to disable; we also
  support an optional target_year to slightly prefer temporally-matching snippets.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import List, Optional, Tuple

import aiohttp
import async_timeout
from tenacity import retry, stop_after_attempt, wait_exponential

from . import config as C
from .cache import NET_CACHE, ttl_get, ttl_set

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_KEY: str | None = os.getenv("BRAVE_SEARCH_API_KEY") or os.getenv("SEARCH_API")

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Stop-words & anti-spam
# --------------------------------------------------------------------------- #
_STOP = {
    "the", "a", "an", "and", "or", "but", "on", "in", "at", "to", "of", "for",
    "with", "by", "about", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "from", "into", "than",
}

_SPAM_PAT = re.compile(
    r"\b("
    r"buy|shop|sale|price|discount|coupon|deal|promo|free shipping|"
    r"template|wallpaper|vector|svg|png|jpeg|jpg|"
    r"pinterest|etsy|aliexpress|amazon|ebay|"
    r"shutterstock|istock|getty|depositphotos|adobe stock|freepik|dreamstime|123rf|"
    r"stock (?:photo|image|images)"
    r")\b",
    re.I,
)

_DOMAIN_SPAM_PAT = re.compile(
    r"(?:^|://)(?:www\.)?(?:pinterest|etsy|aliexpress|amazon|ebay|"
    r"shutterstock|istockphoto|gettyimages|depositphotos|adobestock|freepik|dreamstime|123rf)\.", re.I
)

def _tokenise(text: str) -> set[str]:
    return {tok.lower() for tok in re.findall(r"[A-Za-z0-9]+", text) if tok.lower() not in _STOP}


def _score_snippet(
    text: str,
    url: str | None,
    q_tokens: set[str],
    h_tokens: Optional[set[str]],
    *,
    target_year: Optional[int] = None,
) -> int:
    """
    Return min-overlap score across (question, headline), with a small bonus
    if the snippet text contains the target_year. Spammy snippets score 0.
    """
    if _SPAM_PAT.search(text or ""):
        return 0
    if url and _DOMAIN_SPAM_PAT.search(url or ""):
        return 0
    s_tokens = _tokenise(text)
    overlap_q = len(q_tokens & s_tokens)
    if h_tokens is None:
        score = overlap_q
    else:
        overlap_h = len(h_tokens & s_tokens)
        score = min(overlap_q, overlap_h)

    if target_year is not None and re.search(rf"\b{target_year}\b", text or ""):
        score += max(0, int(C.TEMPORAL_MATCH_BONUS))

    return score


def _freshness_code(days: Optional[int]) -> Optional[str]:
    """
    Map days → Brave freshness code (best-effort).
      ≤1→'pd' (past day), ≤7→'pw' (past week), ≤31→'pm' (past month), ≤365→'py' (past year)
    Returns None if no freshness constraint.
    """
    if not days or days <= 0:
        return None
    if days <= 1:
        return "pd"
    if days <= 7:
        return "pw"
    if days <= 31:
        return "pm"
    return "py"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.2), reraise=True)
async def brave_snippets(
    query: str,
    *,
    k: int = C.BRAVE_K,
    headline: str | None = None,
    freshness_days: Optional[int] = C.FRESHNESS_DAYS,
    target_year: Optional[int] = None,               # ✅ NEW
) -> List[str]:
    """
    Return up to *k* high-quality Brave snippets for *query*, filtered via
    question+headline overlap, anti-spam, and optional freshness.
    """
    if not BRAVE_KEY:
        raise RuntimeError("Brave API key not found – set BRAVE_SEARCH_API_KEY or SEARCH_API.")

    cache_key = f"{query} ||| {headline or ''} ||| {freshness_days or 'none'} ||| {target_year or 'na'}"
    hit = ttl_get(NET_CACHE, cache_key)
    if hit is not None:
        return hit[:k]

    q_tokens = _tokenise(query)
    h_tokens = _tokenise(headline) if headline else None
    fresh_code = _freshness_code(freshness_days)

    async def _fetch(count: int, use_fresh: bool) -> List[dict]:
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": BRAVE_KEY,
        }
        params = {"q": query, "count": count}
        if use_fresh and fresh_code:
            params["freshness"] = fresh_code
        async with async_timeout.timeout(10):
            async with aiohttp.ClientSession() as sess:
                async with sess.get(BRAVE_ENDPOINT, params=params, headers=headers) as r:
                    r.raise_for_status()
                    data = await r.json()
                    return data.get("web", {}).get("results", [])

    # First try with requested freshness (if any)
    raw_results = await _fetch(20, use_fresh=bool(fresh_code))
    if len(raw_results) < k:
        raw_results += await _fetch(50, use_fresh=bool(fresh_code))

    scored: List[Tuple[int, str, int]] = []
    for idx, res in enumerate(raw_results):
        text = (res.get("description") or res.get("title") or "").strip()
        if not text:
            continue
        score = _score_snippet(text, res.get("url"), q_tokens, h_tokens, target_year=target_year)
        if score == 0:
            continue
        scored.append((score, text, idx))

    # If freshness is too restrictive (few results) and freshness was used, widen without it
    if fresh_code and len(scored) < max(3, min(k, 5)):
        raw_results_wide = await _fetch(50, use_fresh=False)
        for idx, res in enumerate(raw_results_wide, 10_000):
            text = (res.get("description") or res.get("title") or "").strip()
            if not text:
                continue
            score = _score_snippet(text, res.get("url"), q_tokens, h_tokens, target_year=target_year)
            if score == 0:
                continue
            scored.append((score, text, idx))

    scored.sort(key=lambda t: (-t[0], t[2]))
    snippets = [txt for _, txt, _ in scored][:k]
    ttl_set(NET_CACHE, cache_key, snippets)

    if C.DEBUG:
        logger.info(
            "Brave: fetched %d, kept %d (req=%d) | fresh=%s | year=%s | query=%r | headline=%r",
            len(raw_results), len(snippets), k, _freshness_code(freshness_days) or "none",
            target_year if target_year is not None else "na",
            query[:80], (headline or "")[:80],
        )

    return snippets


def brave_snippets_sync(
    query: str,
    *,
    k: int = C.BRAVE_K,
    headline: str | None = None,
    freshness_days: Optional[int] = C.FRESHNESS_DAYS,
    target_year: Optional[int] = None,
) -> List[str]:
    """Blocking helper (runs the async coroutine internally)."""
    return asyncio.run(brave_snippets(query, k=k, headline=headline, freshness_days=freshness_days, target_year=target_year))


"""
Brave Search API: Pure-function snippet retriever for web Q&A.
"""

from __future__ import annotations
import os
import time
from typing import List

import requests

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_KEY = os.getenv("BRAVE_SEARCH_API_KEY") or os.getenv("SEARCH_API")

def brave_search_answer(
    query: str,
    k: int = 10,
    max_retries: int = 3,
    retry_delay: float = 1.2,
) -> List[str]:
    """
    Return a list of cleaned snippets (str) for `query`.
    No LangChain, no LLM – just Brave.
    """
    if not BRAVE_KEY:
        raise RuntimeError("Brave API key not found in env (BRAVE_SEARCH_API_KEY or SEARCH_API).")

    params = {"q": query, "count": k}
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_KEY,
    }

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(BRAVE_ENDPOINT, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            snippets: List[str] = []
            for res in data.get("web", {}).get("results", []):
                txt = res.get("description") or res.get("title") or ""
                if txt:
                    snippets.append(txt.strip())
            return snippets[:k]
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(retry_delay)

    return []
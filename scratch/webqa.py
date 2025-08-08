"""
scratch/webqa.py – improved Web‑QA module
========================================

Improvements
------------
1. **Empty / thin evidence guard**
   • If Brave returns fewer than `min_snippets` (default = 2) we skip the LLM
     call and mark the pair as `ok = False` so the selector can down‑weight
     it.

2. **Auto‑flagging weak answers**
   • After `answer_from_evidence` we set `ok = False` when the LLM replies
     with the sentinel “No relevant answer found.”  This prevents such pairs
     from being picked as “best” unless nothing else exists.

3. **Configurable snippet cap**
   • The constructor now allows `k` and `min_snippets` overrides but keeps
     defaults identical to the previous implementation so downstream code
     remains compatible.

Public API is unchanged – `run()` returns a dict with the same keys
(`question`, `answer`, `snippets`, `ok`).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

from . import config as C
from .brave_client import brave_snippets
from .evidence_llm import answer_from_evidence

logger = logging.getLogger(__name__)

__all__ = ["WebQAModule"]


class WebQAModule:
    """Async wrapper that couples Brave search & evidence‑grounded answering."""

    def __init__(
        self,
        question: str,
        llm,
        k: int = C.BRAVE_K,
        *,
        min_snippets: int = 2,
    ) -> None:
        self.question = question.strip()
        self._llm = llm
        self.k = k
        self.min_snippets = min_snippets

    # ------------------------------------------------------------------ #
    # Async entry‑point
    # ------------------------------------------------------------------ #
    async def run(self) -> Dict[str, object]:
        """Return a dict: {question, answer, snippets, ok}."""
        # 1️⃣  Evidence retrieval ------------------------------------------------
        try:
            snippets: List[str] = await brave_snippets(self.question, k=self.k)
        except Exception as exc:  # noqa: BLE001
            logger.error("Brave API failed for %r: %s", self.question, exc)
            return {
                "question": self.question,
                "answer": f"Brave error: {exc}",
                "snippets": [],
                "ok": False,
            }

        # Guard: not enough evidence
        if len(snippets) < self.min_snippets:
            logger.debug("Insufficient evidence (got %d < %d) for %r", len(snippets), self.min_snippets, self.question)
            return {
                "question": self.question,
                "answer": "Insufficient evidence (too few snippets).",
                "snippets": snippets,
                "ok": False,
            }

        # 2️⃣  Evidence‑grounded answer ----------------------------------------
        answer, ok_llm = answer_from_evidence(self._llm, self.question, snippets)

        # Flag generic negative replies so selector can drop them
        if answer.strip().lower().startswith("no relevant answer found"):
            ok_llm = False

        return {
            "question": self.question,
            "answer": answer,
            "snippets": snippets,
            "ok": ok_llm,
        }

    # ------------------------------------------------------------------ #
    # Sync helper for REPL
    # ------------------------------------------------------------------ #
    def run_sync(self) -> Dict[str, object]:
        """Blocking wrapper around the coroutine (do **not** call inside an event loop)."""
        return asyncio.run(self.run())

"""
scratch/final_classifier.py – smarter final decision module
===========================================================

Enhancements
------------
1. **Heuristic short‑circuit**
   • If *no reliable Q‑A evidence* remains **and** the image‑headline
     relevancy equals `IMAGE REFUTES`, we immediately label
     **Misinformation** (and analogue for `IMAGE SUPPORTS`).

2. **Quality‑aware evidence pass‑through**
   • Only Q‑A pairs that pass `_is_good()` (same criteria as the selector)
     are sent to the LLM.  If the filtered list is empty, the short‑circuit
     logic above applies.

3. **Guideline tweak**
   • The system prompt now tells the model to *ignore* Q‑A pairs whose answer
     begins with the sentinel phrases, reinforcing the earlier filtering.

Public interface remains:

>>> dec, reason = FinalClassifier(headline, relevancy_text, qa_pairs).run(llm)
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List, Tuple

from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------------------------------- #
# Helper – same quality filter as qa_selector
# --------------------------------------------------------------------------- #

_BAD_SENTINELS = (
    "no relevant answer found",
    "insufficient evidence",
    "brave error",
)


def _is_good(qa: Dict[str, Any]) -> bool:
    if not qa or not qa.get("question") or not qa.get("answer"):
        return False
    if qa.get("ok") is False:
        return False
    ans = qa["answer"].strip().lower()
    if any(ans.startswith(s) for s in _BAD_SENTINELS):
        return False
    if len(ans) < 15:
        return False
    return True


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #

_SYSTEM_TEMPLATE = textwrap.dedent(
    """
    You are a senior misinformation‑detection expert.  Evaluate the headline
    below based strictly on the provided evidence.

    EVIDENCE
    --------
    • Image–headline relevancy: {relevancy}
    • Fact‑checking Q‑A pairs (only those marked GOOD):
      {qa_context}

    Guidelines
    ----------
    • If the GOOD Q‑A evidence **directly refutes** the headline → "Misinformation".
    • If it **supports** the headline                       → "Not Misinformation".
    • If evidence is **inconclusive**                       → "Uncertain".
    Ignore any Q‑A pair whose answer begins with "No relevant answer found" or
    "Insufficient evidence".
    Ignore the image relevancy unless it *clearly contradicts or outweighs* the
    GOOD Q‑A evidence.

    Reply in exactly this format:
    DECISION: <Misinformation|Not Misinformation|Uncertain>
    REASON:   <concise paragraph citing evidence>
    """
)

_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_TEMPLATE),
        ("human", "HEADLINE: {headline}"),
    ]
)

# Cache compiled prompt × llm
from cachetools import LRUCache

_CHAIN_CACHE: LRUCache = LRUCache(maxsize=32)


def _get_chain(prompt_obj, llm):
    key = (id(prompt_obj), id(llm))
    if key not in _CHAIN_CACHE:
        _CHAIN_CACHE[key] = prompt_obj | llm
    return _CHAIN_CACHE[key]


# Regex for parsing
_DECISION_RE = re.compile(
    r"DECISION:\s*(Misinformation|Not Misinformation|Uncertain)", re.I
)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.I | re.S)


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #

class FinalClassifier:
    """Veracity classifier with heuristic fallback for visual‑only cases."""

    def __init__(
        self,
        headline: str,
        relevancy_text: str,
        qa_pairs: List[Dict[str, Any]],
        *,
        extra_guidelines: str | None = None,
    ) -> None:
        self.headline = headline
        self.relevancy_text = relevancy_text
        self.qa_pairs_raw = qa_pairs
        self.extra_guidelines = extra_guidelines or ""

        # Pre‑filter QA for quality
        self.good_pairs = [qa for qa in qa_pairs if _is_good(qa)]

    # ------------------------------------------------------------------ #
    def _parse_relevancy(self) -> str | None:
        txt = self.relevancy_text.upper()
        if "IMAGE REFUTES" in txt:
            return "REFUTES"
        if "IMAGE SUPPORTS" in txt:
            return "SUPPORTS"
        return None

    # ------------------------------------------------------------------ #
    def _short_circuit(self):
        """Return (decision, reason) if heuristic applies, else None."""
        rel_flag = self._parse_relevancy()
        if self.good_pairs:
            # Only short‑circuit if we *lack* good Q‑A evidence
            return None

        if rel_flag == "REFUTES":
            return (
                "Misinformation",
                "The image‑headline relevancy analysis explicitly returns IMAGE REFUTES and no reliable Q‑A evidence is available.",
            )
        if rel_flag == "SUPPORTS":
            return (
                "Not Misinformation",
                "The image‑headline relevancy analysis returns IMAGE SUPPORTS and no reliable Q‑A evidence contradicts it.",
            )
        return ("Uncertain", "No reliable Q‑A evidence and the image relevancy is inconclusive.")

    # ------------------------------------------------------------------ #
    def run(self, llm) -> Tuple[str, str]:
        # 1️⃣  heuristic fast path
        heuristic = self._short_circuit()
        if heuristic is not None:
            return heuristic

        # 2️⃣  Build QA context for LLM
        qa_context = "\n".join(
            f"Q: {qa['question']}\nA: {qa['answer']}" for qa in self.good_pairs
        )

        # 3️⃣  Prompt & invoke
        chain = _get_chain(_PROMPT_TEMPLATE, llm)
        if self.extra_guidelines:
            chain = chain.partial(__system_override=self.extra_guidelines)

        resp = chain.invoke(
            {
                "headline": self.headline,
                "relevancy": self.relevancy_text,
                "qa_context": qa_context,
            }
        )
        raw = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

        # 4️⃣  Parse
        dec_match = _DECISION_RE.search(raw)
        reason_match = _REASON_RE.search(raw)
        decision = dec_match.group(1).strip() if dec_match else "Uncertain"
        reason = reason_match.group(1).strip() if reason_match else "No reason provided"
        return decision, reason

"""
scratch/qa_selector.py – quality‑aware Q‑A selection
====================================================

Key upgrades
------------
1. **Bad‑pair filtering**
   • Pairs whose `ok` flag is *False* **or** whose answer begins with
     “No relevant answer found.”/“Insufficient evidence” are excluded *before*
     we even ask the LLM to choose.  If a branch has zero good pairs we return
     an empty dict so the final classifier can recognise missing evidence.

2. **Length sanity check**
   • Answers shorter than 15 characters are considered low‑quality and
     filtered.

3. **Cleaner fallbacks**
   • When the selection prompt fails, we now fall back to the *first good*
     pair (if any) instead of the first raw pair.

Public API (`QASelector`, `batch_select`) remains unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

__all__ = ["QASelector", "batch_select"]

# --------------------------------------------------------------------------- #
# Utility: assess QA quality
# --------------------------------------------------------------------------- #

_BAD_SENTINELS = (
    "no relevant answer found",
    "insufficient evidence",
    "brave error",
)


def _is_good(qa: Dict[str, Any]) -> bool:
    if not qa or not qa.get("question") or not qa.get("answer"):
        return False
    if not qa.get("ok", True):
        return False
    ans = qa["answer"].strip().lower()
    if any(ans.startswith(s) for s in _BAD_SENTINELS):
        return False
    if len(ans) < 15:  # too short to be useful
        return False
    return True


# --------------------------------------------------------------------------- #
# Prompt templates (unchanged except variable names)
# --------------------------------------------------------------------------- #

_PROMPT_SINGLE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            textwrap.dedent(
                """
                You are a fact‑checking analyst.  Below are several candidate
                Q‑A pairs.  Choose the **single most informative** pair to help
                verify the headline.

                Return ONLY that pair in the format:
                Question: <text>
                Answer:   <text>
                """
            ),
        ),
        ("human", "{qa_block}"),
    ]
)

_PROMPT_MULTI = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            textwrap.dedent(
                """
                You are a fact‑checking analyst.

                For EACH branch, select the **single best** Q‑A pair.
                Output a one‑line JSON array where each element is:
                {"branch": <int>, "question": <text>, "answer": <text>}
                """
            ),
        ),
        ("human", "{branches_block}"),
    ]
)

# --------------------------------------------------------------------------- #
# Chain cache
# --------------------------------------------------------------------------- #

from cachetools import LRUCache

_CHAIN_CACHE: LRUCache = LRUCache(maxsize=32)


def _get_chain(prompt_obj, llm):
    key = (id(prompt_obj), id(llm))
    if key not in _CHAIN_CACHE:
        _CHAIN_CACHE[key] = prompt_obj | llm
    return _CHAIN_CACHE[key]


# --------------------------------------------------------------------------- #
# Single‑branch selector
# --------------------------------------------------------------------------- #

class QASelector:
    """Select the best Q‑A pair from *one* branch, with quality filtering."""

    def __init__(self, qa_pairs: List[Dict[str, Any]]):
        # Store original + also build filtered view
        self._orig_pairs = qa_pairs
        self.qa_pairs = [qa for qa in qa_pairs if _is_good(qa)]

    def _format_block(self) -> str:
        return "\n---\n".join(
            f"Q: {qa['question']}\nA: {qa['answer']}" for qa in self.qa_pairs
        )

    # -------------------------------------------------------------- #
    def run(self, llm) -> Tuple[Optional[Dict[str, Any]], bool]:
        if not self.qa_pairs:
            return None, False  # nothing usable

        prompt_vars = {"qa_block": self._format_block()}

        try:
            resp = _get_chain(_PROMPT_SINGLE, llm).invoke(prompt_vars)
            txt = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as exc:  # noqa: BLE001
            logger.error("Selector LLM failed: %s", exc, exc_info=False)
            return self.qa_pairs[0], False

        q_match = re.search(r"Question:\s*(.+)", txt, re.I)
        a_match = re.search(r"Answer:\s*(.+)", txt, re.I | re.S)
        if q_match and a_match:
            return {
                "question": q_match.group(1).strip(),
                "answer": a_match.group(1).strip(),
                "snippets": None,
            }, True

        return self.qa_pairs[0], False


# --------------------------------------------------------------------------- #
# Batch selector (multi‑branch)
# --------------------------------------------------------------------------- #

def _format_branches(branches: List[List[Dict[str, Any]]]) -> str:
    blocks = []
    for idx, branch in enumerate(branches, start=1):
        goods = [qa for qa in branch if _is_good(qa)]
        body = "\n".join(f"Q: {qa['question']}\nA: {qa['answer']}" for qa in goods)
        blocks.append(f"BRANCH {idx}:\n{body}")
    return "\n\n".join(blocks)


def batch_select(
    qa_branches: List[List[Dict[str, Any]]],
    llm,
) -> List[Dict[str, Any]]:
    if not qa_branches:
        return []

    # Filter each branch first; keep empty lists if all bad
    filtered = [[qa for qa in br if _is_good(qa)] for br in qa_branches]

    # Fast path: single branch
    if len(filtered) == 1:
        sel, _ = QASelector(filtered[0]).run(llm) if filtered[0] else (None, False)
        return [sel or {}]

    if all(not br for br in filtered):
        # All branches empty → no evidence
        return [{} for _ in qa_branches]

    prompt_vars = {"branches_block": _format_branches(filtered)}

    try:
        resp = _get_chain(_PROMPT_MULTI, llm).invoke(prompt_vars)
        txt = resp.content if hasattr(resp, "content") else str(resp)
        parsed = json.loads(txt.strip())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Batch selector failed, falling back: %s", exc)
        return [br[0] if br else {} for br in filtered]

    best: List[Dict[str, Any]] = []
    for idx, orig_branch in enumerate(filtered, start=1):
        rec = next((r for r in parsed if isinstance(r, dict) and r.get("branch") == idx), None)
        if rec and rec.get("question") and rec.get("answer"):
            best.append(
                {
                    "question": rec["question"].strip(),
                    "answer": rec["answer"].strip(),
                    "snippets": None,
                }
            )
        else:
            # Fallback: first good pair or empty dict
            best.append(orig_branch[0] if orig_branch else {})
    return best

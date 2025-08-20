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

scratch/qa_selector.py – quality-aware Q-A selection (tightened)
===============================================================

Changes (High-impact):
* "GOOD" now means ALL of:
  - qa.ok is True
  - snippet_count >= MIN_SNIPPETS
  - overlap_score >= MIN_OVERLAP
  - len(answer) >= MIN_ANSWER_CHARS
  - answer does NOT start with a bad sentinel

* Branches with zero GOOD pairs return {} so downstream can short-circuit.
* On LLM selection failure, fall back to the **first GOOD** pair.

scratch/qa_selector.py – quality-aware Q-A selection (deterministic)
===================================================================

Upgrades:
1) Deterministic selection: we temporarily set router temperature=0.0
   for the selection prompt, then restore it.
2) Tight GOOD filter (length, snippets, overlap, confidence).
3) Clean fallback: if the LLM selection fails, use the first GOOD pair.

Public API:
    - batch_select(branches, router) -> List[Dict]  (one best per branch)

scratch/qa_selector.py – quality-aware Q-A selection (deterministic) + diagnostics
==================================================================================

What's new in this revision (Step 15):
- Log branch-level diagnostics for pairs that are filtered out:
  • selector_drop_short_answer
  • selector_drop_low_overlap
  • selector_drop_low_conf

We still:
- Gate Q-A by multiple quality signals (sentinels, length, snippets, overlap, confidence).
- Force temperature=0.0 only during selection for determinism, then restore.
- Fall back cleanly to the first GOOD pair if the LLM selection fails.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from cachetools import LRUCache

from .qa_quality import is_good_qa

logger = logging.getLogger(__name__)

__all__ = ["batch_select"]



# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #

_PROMPT_SINGLE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            textwrap.dedent(
                """
                You are a fact-checking analyst. Below are several candidate
                Q-A pairs. Choose the **single most informative** pair to help
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

# Cache compiled prompt × llm
_CHAIN_CACHE: LRUCache = LRUCache(maxsize=32)

def _get_chain(prompt_obj, llm):
    key = (id(prompt_obj), id(llm))
    if key not in _CHAIN_CACHE:
        _CHAIN_CACHE[key] = prompt_obj | llm
    return _CHAIN_CACHE[key]

_Q_RE = re.compile(r"Question:\s*(.+)", re.I)
_A_RE = re.compile(r"Answer:\s*(.+)", re.I | re.S)


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #

def _select_one_for_branch(router, qa_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the best QA from a single branch deterministically."""
    good = [qa for qa in qa_list if is_good_qa(qa, emit_metrics=True, metrics_prefix="selector")]
    if not good:
        return {}  # nothing usable in this branch

    # Build block for the selector prompt
    block_items = [f"Q: {qa['question']}\nA: {qa['answer']}" for qa in good]
    qa_block = "\n---\n".join(block_items)

    # Temporarily force temperature=0.0
    orig_model = router.model_name
    orig_temp = router.temperature
    try:
        router.switch_model(orig_model, temperature=0.0)
        llm0 = router.get()
        chain = _get_chain(_PROMPT_SINGLE, llm0)
        resp = chain.invoke({"qa_block": qa_block})
        raw = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
    except Exception as exc:  # fall back to first GOOD
        logger.warning("QA selection LLM error: %s – falling back to first GOOD.", exc)
        return good[0]
    finally:
        # Restore original temperature
        router.switch_model(orig_model, temperature=orig_temp)

    # Parse "Question: ... / Answer: ..."
    qm = _Q_RE.search(raw)
    am = _A_RE.search(raw)
    if not (qm and am):
        # If parsing fails, return first GOOD
        return good[0]

    q_text = qm.group(1).strip()
    a_text = am.group(1).strip()

    # Find the matching GOOD pair (exact question match preferred)
    for qa in good:
        if qa["question"].strip() == q_text:
            return qa

    # Loose fallback: return the first GOOD whose answer prefix matches
    a_prefix = a_text[:50].lower()
    for qa in good:
        if qa["answer"].lower().startswith(a_prefix):
            return qa

    # Final fallback
    return good[0]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def batch_select(branches: List[List[Dict[str, Any]]], router) -> List[Dict[str, Any]]:
    """
    For each branch, select the best QA pair (with temp=0.0 during selection).
    Returns a list with at most one dict per branch (empty dicts are skipped).
    """
    best: List[Dict[str, Any]] = []
    for b_idx, branch in enumerate(branches):
        picked = _select_one_for_branch(router, branch)
        if picked:
            best.append(picked)
        else:
            logger.info("Branch %d has no GOOD Q-A pairs.", b_idx)
    return best

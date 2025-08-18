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

from . import config as C
from . import log

logger = logging.getLogger(__name__)

__all__ = ["batch_select"]

# --------------------------------------------------------------------------- #
# Utility: assess QA quality (with diagnostics)
# --------------------------------------------------------------------------- #

_BAD_SENTINELS = (
    "no relevant answer found",
    "insufficient evidence",
    "brave error",
)

def _is_good(qa: Dict[str, Any]) -> bool:
    """
    Return True if the QA pair passes all gates.
    Also increments diagnostic metrics for certain drop reasons:
      - selector_drop_short_answer
      - selector_drop_low_overlap
      - selector_drop_low_conf
    """
    if not qa or not qa.get("question") or not qa.get("answer"):
        return False

    if qa.get("ok") is not True:
        return False

    ans = qa["answer"].strip()
    if any(ans.lower().startswith(s) for s in _BAD_SENTINELS):
        return False

    # Collect reasons (we may log multiple reasons for one dropped pair)
    reasons: List[str] = []

    # Core gates used by selector/classifier
    if len(ans) < C.MIN_ANSWER_CHARS:
        reasons.append("short_answer")

    if qa.get("snippet_count", 0) < C.MIN_SNIPPETS:
        # We keep this gate but do not count it per the brief
        return False if not reasons else False  # still fail fast

    if qa.get("overlap_score", 0) < C.MIN_OVERLAP:
        reasons.append("low_overlap")

    try:
        conf_val = float(qa.get("answer_conf", 0.0))
    except Exception:
        conf_val = 0.0
    if conf_val < C.MIN_CONF:
        reasons.append("low_conf")

    if reasons:
        # Emit diagnostics (one metric per reason observed)
        for r in set(reasons):
            if r == "short_answer":
                log.metric("selector_drop_short_answer")
            elif r == "low_overlap":
                log.metric("selector_drop_low_overlap")
            elif r == "low_conf":
                log.metric("selector_drop_low_conf")
        return False

    return True


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
    good = [qa for qa in qa_list if _is_good(qa)]
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

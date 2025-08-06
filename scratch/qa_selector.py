
"""
scratch/qa_selector.py
======================

Step-05 logic: choose the *single most useful* Q-A pair from one or more
branches.

  • **QASelector** – keeps backward-compat signature (`run()` per branch).  
  • **batch_select** – NEW helper that selects the best pair *for every
    branch* with one LLM round-trip.

Both helpers parse the LLM response; if parsing fails they fall back to the
*first Q-A* in the branch.

Returned Q-A dict structure
---------------------------

{
"question": <str>,
"answer": <str>,
"snippets": <list[str]> | None # optional; propagated unmodified
}

perl
Copy
Edit
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate

from .cache import memo

logger = logging.getLogger(__name__)

__all__ = ["QASelector", "batch_select"]

# --------------------------------------------------------------------------- #
# Prompt templates
# --------------------------------------------------------------------------- #

_PROMPT_SINGLE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            textwrap.dedent(
                """\
                You are a fact-checking analyst.  Below are several Q-A pairs
                generated to verify a news headline.

                Select the **one** pair that best helps decide whether the
                headline is true or false.

                Return ONLY the chosen pair in exactly this format:

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
                """\
                You are a fact-checking analyst.

                For EACH branch shown below, pick the **single best** Q-A pair
                (the one most helpful for verifying the headline).

                Output a single-line JSON array; each element must have:

                  {{
                    "branch":   <int>,     // 1-based branch index
                    "question": <text>,
                    "answer":   <text>
                  }}

                Do NOT include any keys besides the three above.
                """
            ),
        ),
        ("human", "{branches_block}"),
    ]
)

# --------------------------------------------------------------------------- #
# Chain-level cache
# --------------------------------------------------------------------------- #


from cachetools import LRUCache
_CHAIN_CACHE = LRUCache(maxsize=32)

def _get_chain(prompt_obj, llm):
    """Return a compiled (prompt | llm) chain, caching on object IDs."""
    key = (id(prompt_obj), id(llm))
    if key not in _CHAIN_CACHE:
        _CHAIN_CACHE[key] = prompt_obj | llm
    return _CHAIN_CACHE[key]


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #


class QASelector:
    """
    Backward-compat wrapper (single branch).

    Example
    -------
    >>> sel = QASelector(branch_qa)
    >>> best, ok = sel.run(llm)
    """

    def __init__(self, qa_pairs: List[Dict[str, Any]]) -> None:
        self.qa_pairs = qa_pairs

    def _format_block(self) -> str:
        return "\n---\n".join(
            f"Q: {qa['question']}\nA: {qa['answer']}" for qa in self.qa_pairs
        )

    # -------------------------------------------------------------- #
    # Main
    # -------------------------------------------------------------- #
    def run(self, llm) -> Tuple[Optional[Dict[str, Any]], bool]:
        if not self.qa_pairs:
            return None, True

        prompt_vars = {"qa_block": self._format_block()}

        try:
            resp = _get_chain(_PROMPT_SINGLE, llm).invoke(prompt_vars)
            txt = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as exc:  # noqa: BLE001
            logger.error("Selector LLM failed: %s", exc, exc_info=False)
            return self.qa_pairs[0], False

        # crude extraction
        q_match = re.search(r"Question:\s*(.+)", txt, re.I)
        a_match = re.search(r"Answer:\s*(.+)", txt, re.I | re.S)
        if q_match and a_match:
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            # keep snippets if they exist (copy from first match)
            return {"question": question, "answer": answer, "snippets": None}, True

        return self.qa_pairs[0], False


# --------------------------------------------------------------------------- #
# Batch selection for all branches
# --------------------------------------------------------------------------- #


def _format_branches(branches: List[List[Dict[str, Any]]]) -> str:
    """Convert branches → text block for the prompt."""
    blocks: List[str] = []
    for idx, branch in enumerate(branches, start=1):
        body = "\n".join(f"Q: {qa['question']}\nA: {qa['answer']}" for qa in branch)
        blocks.append(f"BRANCH {idx}:\n{body}")
    return "\n\n".join(blocks)


def batch_select(
    qa_branches: List[List[Dict[str, Any]]],
    llm,
) -> List[Dict[str, Any]]:
    """
    Parameters
    ----------
    qa_branches
        List of branches; each branch is a list of Q-A dicts.
    llm
        LangChain LLM instance.

    Returns
    -------
    best_pairs
        List with one *selected* Q-A dict per branch (same order).
    """
    if not qa_branches:
        return []

    # fall back quickly if LLM unavailable
    if len(qa_branches) == 1:
        sel, _ = QASelector(qa_branches[0]).run(llm)
        return [sel]

    prompt_vars = {"branches_block": _format_branches(qa_branches)}

    try:
        resp = _get_chain(_PROMPT_MULTI, llm).invoke(prompt_vars)
        txt = resp.content if hasattr(resp, "content") else str(resp)
        # Expect single-line JSON
        parsed = json.loads(txt.strip())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Batch selector failed, falling back: %s", exc)
        return [branch[0] if branch else {} for branch in qa_branches]

    # Build output list; default to first if missing
    best: List[Dict[str, Any]] = []
    for idx, branch in enumerate(qa_branches, start=1):
        record = next(
            (x for x in parsed if isinstance(x, dict) and x.get("branch") == idx), None
        )
        if record and record.get("question") and record.get("answer"):
            best.append(
                {
                    "question": record["question"].strip(),
                    "answer": record["answer"].strip(),
                    "snippets": next(iter(branch or [{}]), {}).get("snippets"),
                }
            )
        else:
            best.append(branch[0] if branch else {})
    return best
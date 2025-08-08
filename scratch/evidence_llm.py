"""
scratch/evidence_llm.py – stricter evidence‑grounded answering
=============================================================

Enhancements
------------
1. **Hard cap on hallucinations**
   • The prompt now forces the model to answer in **≤ 3 sentences** and makes
     it explicitly drop any sentence that cannot be matched to a snippet.

2. **Early sentinel on empty evidence**
   • If the snippet list is empty the function immediately returns
     "No relevant answer found." with `ok = False` – we skip the LLM call.

3. **Answer length check**
   • Even with the prompt guard, we add a post‑generation check: if the reply
     exceeds 400 characters we mark `ok = False` so the selector will prefer
     cleaner answers.

The public function signature is **unchanged**:

>>> answer, ok = answer_from_evidence(llm, question, snippets)
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #

_PROMPT_TMPL = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an evidence‑grounded answer generator.  Follow ALL rules:

            1️⃣  Use ONLY the snippets provided below – do not add outside facts.
            2️⃣  Write **no more than 3 sentences**.
            3️⃣  If the evidence is insufficient or contradictory, reply EXACTLY:
                No relevant answer found.
            4️⃣  Each sentence must be traceable to at least one snippet; if not,
                omit the sentence.

            SNIPPETS:
            {evidence}
            """,
        ),
        ("human", "{question}"),
    ]
)

# --------------------------------------------------------------------------- #
# Chain cache (prompt × llm) – prevents recompilation overhead
# --------------------------------------------------------------------------- #

from cachetools import LRUCache

_CHAIN_CACHE: LRUCache = LRUCache(maxsize=32)


def _get_chain(prompt_obj, llm):
    key = (id(prompt_obj), id(llm))
    if key not in _CHAIN_CACHE:
        _CHAIN_CACHE[key] = prompt_obj | llm
    return _CHAIN_CACHE[key]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def answer_from_evidence(
    llm,
    question: str,
    snippets: List[str] | None,
    *,
    max_chars: int = 400,
) -> Tuple[str, bool]:
    """Return a concise, evidence‑grounded answer (answer, ok)."""

    if not snippets:  # empty list or None
        return "No relevant answer found.", False

    evidence_text = "\n".join(f"- {s}" for s in snippets)

    try:
        resp = _get_chain(_PROMPT_TMPL, llm).invoke(
            {"evidence": evidence_text, "question": question}
        )
    except Exception as exc:  # noqa: BLE001
        return f"answer-llm error: {exc}", False

    answer = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

    # Basic validation – sentinel already handled but keep safety net
    if (
        not answer
        or answer.lower().startswith("no relevant answer found")
        or len(answer) > max_chars
    ):
        return answer, False

    return answer, True

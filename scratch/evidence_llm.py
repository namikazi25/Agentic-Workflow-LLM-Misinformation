"""
scratch/evidence_llm.py
=======================

LLM wrapper that **converts raw web-search snippets into a single answer**,
explicitly limiting the model to the provided evidence to avoid hallucination.

It exposes a single public function:

    >>> answer, ok = answer_from_evidence(llm, question, snippets)

where

* *llm*       – a LangChain chat-model instance (from `ModelRouter.get()`).
* *question*  – the fact-checking question (str).
* *snippets*  – list[str] of evidence lines (each is a Brave snippet).

Returns
-------

``(answer_text: str, ok: bool)``

* ``ok`` is *False* when the LLM invocation fails or returns an empty string.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate

from .cache import memo

# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #

_PROMPT_TMPL = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a factual answer generator.  Using ONLY the evidence shown "
            "below, write a concise answer to the user's question. "
            "If the evidence is insufficient or conflicting, reply exactly:\n"
            "'No relevant answer found.'\n\n"
            "EVIDENCE:\n{evidence}",
        ),
        ("human", "{question}"),
    ]
)

# --------------------------------------------------------------------------- #
# Chain cache  (one compiled prompt per LLM instance)
# --------------------------------------------------------------------------- #


@memo(maxsize=32)
def _get_chain(llm):
    return _PROMPT_TMPL | llm


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def answer_from_evidence(
    llm,
    question: str,
    snippets: List[str] | None,
) -> Tuple[str, bool]:
    """
    Parameters
    ----------
    llm
        LangChain model obtained via `ModelRouter.get()`.
    question
        The question to be answered.
    snippets
        List of evidence strings (can be empty).

    Returns
    -------
    (answer, ok)
        *answer* – LLM response (trimmed str).
        *ok*     – True if non-empty answer returned, False otherwise.
    """
    evidence_text = "\n".join(f"- {s}" for s in snippets or []) or "No evidence."

    try:
        resp = _get_chain(llm).invoke({"evidence": evidence_text, "question": question})
    except Exception as exc:  # noqa: BLE001
        return f"answer-llm error: {exc}", False

    answer = (
        resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
    )

    return answer, bool(answer)

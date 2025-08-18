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

scratch/evidence_llm.py – evidence-grounded answering with confidence
=====================================================================

Changes:
- Use ModelRouter.call(...) so we get a confidence score.
- Return (answer, ok, confidence).
- Keep ≤3 sentence rule and sentinel handling.

scratch/evidence_llm.py – evidence-grounded answering with confidence + headline self-policing
=============================================================================================

Changes:
- Accept optional `headline` and post-validate the final answer: if it contains
  **no tokens** from the headline, mark ok=False (blocks pretty but generic summaries).
- Keep using ModelRouter.call(...) so we get a confidence score.
- Return (answer, ok, confidence).
"""

from __future__ import annotations

import re
from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------------------------------------- #
# Prompt template (unchanged)
# --------------------------------------------------------------------------- #

_PROMPT_TMPL = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an evidence-grounded answer generator.  Follow ALL rules:

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

_BAD_SENTINEL = "no relevant answer found"

# Simple tokenization (shared style with other modules)
_STOP = {
    "the","a","an","and","or","but","on","in","at","to","of","for",
    "with","by","about","is","are","was","were","be","been","being",
    "this","that","these","those","it","its","as","from","into","than",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOP}


def answer_from_evidence(
    router,                     # ModelRouter instance (not raw llm)
    question: str,
    snippets: List[str] | None,
    *,
    headline: str | None = None,    # ✅ NEW: for self-policing
    max_chars: int = 400,
) -> Tuple[str, bool, float]:
    """Return (answer, ok, confidence).  ok=False if sentinel/too long/empty or no headline overlap."""

    if not snippets:
        return "No relevant answer found.", False, 0.0

    evidence_text = "\n".join(f"- {s}" for s in snippets)

    # Prepare messages for ModelRouter.call(...)
    messages = _PROMPT_TMPL.format_prompt(
        evidence=evidence_text,
        question=question,
    ).to_messages()

    try:
        out = router.call(messages)  # {"raw": resp, "confidence": float}
    except Exception as exc:  # noqa: BLE001
        return f"answer-llm error: {exc}", False, 0.0

    resp = out.get("raw")
    conf = float(out.get("confidence", 0.5))

    answer = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

    ok = True
    if (
        not answer
        or answer.lower().startswith(_BAD_SENTINEL)
        or len(answer) > max_chars
    ):
        ok = False

    # ✅ Self-police: the final answer must include ≥1 token/entity from the headline
    if ok and headline:
        ans_t = _tokens(answer)
        head_t = _tokens(headline)
        if len(ans_t & head_t) < 1:
            ok = False  # fails entity/token anchoring to the claim

    return answer, ok, conf
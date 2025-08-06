"""
scratch/final_classifier.py
===========================

Final **veracity classifier** – makes the ultimate call:

    • "Misinformation"
    • "Not Misinformation"
    • "Uncertain"

based on

    1. Image–headline relevancy text
    2. Best Q-A evidence pairs (one per reasoning chain)

Typical usage
-------------

>>> clsf = FinalClassifier(headline, relevancy_text, qa_pairs)
>>> decision, reason = clsf.run(llm)

Return values
-------------

* `decision` – one of the three labels above  
* `reason`   – concise paragraph citing evidence
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List, Tuple

from langchain_core.prompts import ChatPromptTemplate

from .cache import memo

__all__ = ["FinalClassifier"]

# --------------------------------------------------------------------------- #
# Prompt template (defined once)
# --------------------------------------------------------------------------- #

_SYSTEM_TEMPLATE = textwrap.dedent(
    """\
    You are a senior misinformation-detection expert.  Evaluate the headline
    below based STRICTLY on the provided evidence.

    EVIDENCE
    --------
    • Image–headline relevancy: {relevancy}
    • Fact-checking Q-A pairs:
      {qa_context}

    Guidelines
    ----------
    • If the Q-A evidence **directly refutes** the headline → "Misinformation".
    • If it **supports** the headline                       → "Not Misinformation".
    • If evidence is **inconclusive**                       → "Uncertain".
    Ignore the image relevancy unless it *clearly contradicts* the Q-A.

    Reply in exactly this format:

    DECISION: <Misinformation|Not Misinformation|Uncertain>
    REASON:   <concise paragraph citing evidence>
    """
)

# --------------------------------------------------------------------------- #
# Chain cache (prompt × llm)
# --------------------------------------------------------------------------- #


@memo(maxsize=32)
def _get_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM_TEMPLATE),
            ("human", "HEADLINE: {headline}"),
        ]
    )
    return prompt | llm


# --------------------------------------------------------------------------- #
# Regex compiled once
# --------------------------------------------------------------------------- #

_DECISION_RE = re.compile(
    r"DECISION:\s*(Misinformation|Not Misinformation|Uncertain)", re.I
)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.I | re.S)

# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #


class FinalClassifier:
    """
    Parameters
    ----------
    headline
        News headline text (str).
    relevancy_text
        Output from `ImageHeadlineRelevancyChecker` (str).
    qa_pairs
        List of *selected* Q-A dicts (one per chain).
    extra_guidelines
        Optional free-form string appended to the Guidelines section.
    """

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
        self.qa_pairs = qa_pairs
        self.extra_guidelines = extra_guidelines or ""

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, llm) -> Tuple[str, str]:
        """Return `(decision, reason)`."""
        qa_context = "\n".join(
            f"Q: {qa['question']}\nA: {qa['answer']}" for qa in self.qa_pairs
        )

        # Append any custom instructions once
        prompt_chain = _get_chain(llm)
        if self.extra_guidelines:
            prompt_chain = prompt_chain.partial(
                __system_override=self.extra_guidelines
            )

        # Invoke
        resp = prompt_chain.invoke(
            {
                "headline": self.headline,
                "relevancy": self.relevancy_text,
                "qa_context": qa_context,
            }
        )

        raw = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()

        # ------------------ parse decision & reason ------------------ #
        decision_match = _DECISION_RE.search(raw)
        reason_match = _REASON_RE.search(raw)

        decision = (
            decision_match.group(1).strip() if decision_match else "Uncertain"
        )
        reason = reason_match.group(1).strip() if reason_match else "No reason provided"

        return decision, reason
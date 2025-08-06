"""
scratch/qa_gen.py
=================

Question-generation module (pipeline *Step-03*).

It produces exactly **ONE** concise, web-searchable question that helps
verify a news headline.  Behaviour can be switched between:

* **headline** – use headline text + previous branch Q-A as context.  
* **report**   – additionally leverage a multimodal *event report* produced
  by `EventContextGenerator`.  
* **auto**     – use *report* mode if an event report is supplied, otherwise
  fallback to *headline* mode.

Public usage
------------

>>> q_tool = QAGenerationTool(headline, previous_qa, event_report, strategy="auto")
>>> question, ok = q_tool.run(llm)

`ok` is *False* when the LLM call fails or returns an empty string.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Dict, List, Tuple

from langchain_core.prompts import ChatPromptTemplate

from .cache import memo

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt templates (defined once)
# --------------------------------------------------------------------------- #

_BASE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            textwrap.dedent(
                """\
                You are a fact-checking *question generator*.

                Given the HEADLINE below, write **one** concise, Google-style
                query that would help verify the claim.  Avoid duplicates of
                earlier questions shown as JSON.

                PREVIOUS_QA:
                {previous_qa}
                """
            ),
        ),
        ("human", "HEADLINE: {headline}"),
    ]
)

_ENRICHED_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            textwrap.dedent(
                """\
                You are a fact-checking *question generator*.

                Using the SUMMARY below, write **one** concise, Google-style
                query that would help verify a still-uncertain detail of the
                event.  Avoid duplicates of earlier questions.

                SUMMARY:
                {summary}

                PREVIOUS_QA:
                {previous_qa}
                """
            ),
        ),
        ("human", "HEADLINE: {headline}"),
    ]
)

# --------------------------------------------------------------------------- #
# Chain cache
# --------------------------------------------------------------------------- #


@memo(maxsize=64)  # (prompt_id, llm_instance) → compiled chain
def _get_chain(prompt_obj, llm):
    return prompt_obj | llm


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #


class QAGenerationTool:
    """
    Parameters
    ----------
    headline
        Original news headline.
    previous_qa
        List of Q-A dicts already asked in the current branch (avoids repetition).
    event_report
        Dict produced by `EventContextGenerator` (must contain `"summary"` key).
    strategy
        "headline" | "report" | "auto"
    """

    def __init__(
        self,
        headline: str,
        previous_qa: List[Dict[str, str]] | None = None,
        *,
        event_report: Dict[str, Any] | None = None,
        strategy: str = "auto",
    ) -> None:
        self.headline = headline
        self.previous_qa = previous_qa or []
        self.event_report = event_report or {}
        self.summary = self.event_report.get("summary", "")
        self.strategy = strategy.lower()

        # Resolve final strategy
        if self.strategy == "auto":
            self.strategy = "report" if self.event_report else "headline"
        if self.strategy == "report" and not self.summary:
            raise ValueError(
                "strategy='report' requires `event_report` with a 'summary' key."
            )

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(self, llm) -> Tuple[str, bool]:
        """
        Returns
        -------
        question : str
        ok       : bool   (False if generation failed)
        """
        prompt = _ENRICHED_PROMPT if self.strategy == "report" else _BASE_PROMPT

        variables = {
            "headline": self.headline,
            "previous_qa": json.dumps(self.previous_qa, indent=2, ensure_ascii=False),
        }
        if self.strategy == "report":
            variables["summary"] = self.summary

        try:
            resp = _get_chain(prompt, llm).invoke(variables)
            question = (
                resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            )
            return question, bool(question)

        except Exception as exc:  # noqa: BLE001
            logger.error("Q-gen failed: %s", exc, exc_info=False)
            return f"Q-gen error: {exc}", False

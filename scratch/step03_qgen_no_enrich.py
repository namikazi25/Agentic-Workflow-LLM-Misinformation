# scratch/step03_qgen.py
"""
Unified question-generation module.

• Baseline mode   → only the headline + previous Q-A.
• Enriched mode   → also receives an event SUMMARY paragraph produced
  by EventContextGenerator (image-aware).
You can flip behaviour per call via `strategy`.
"""

from __future__ import annotations
import json
from typing import List, Dict, Any, Tuple, Optional

from langchain_core.prompts import ChatPromptTemplate


class QAGenerationTool:
    """
    Parameters
    ----------
    headline : str
        Original news headline.
    previous_qa : list[dict] | None
        Q-A pairs already asked in this branch (for deduplication).
    event_report : dict | None
        Dict returned by EventContextGenerator.  Must contain "summary"
        when using strategy="report".
    strategy : {"auto", "headline", "report"}
        • auto      – use "report" when event_report is given, else "headline".
        • headline  – force baseline behaviour (ignore event_report).
        • report    – require event_report; raises if missing.
    """

    # -------- Prompt templates ---------------------------------------- #

    BASE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are a fact-checking question generator. "
         "Given the HEADLINE below, produce ONE concise, web-searchable "
         "question that will help verify it. Avoid repeating questions "
         "already asked.\n\n"
         "Previous Q-A (if any):\n{previous_qa}"),
        ("human", "HEADLINE: {headline}"),
    ])

    ENRICHED_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "You are a fact-checking question generator.\n"
         "Using the SUMMARY of the event below, ask **one** concise, "
         "web-searchable question that will help verify something still "
         "uncertain. Avoid duplicates.\n\n"
         "SUMMARY:\n{summary}\n"
         "Previous Q-A:\n{previous_qa}"),
        ("human", "HEADLINE: {headline}"),
    ])

    # ------------------------------------------------------------------ #

    def __init__(
        self,
        headline: str,
        previous_qa: Optional[List[Dict[str, str]]] = None,
        *,
        event_report: Optional[Dict[str, Any]] = None,
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
        if self.strategy == "report" and not self.event_report:
            raise ValueError("strategy='report' requires event_report with a 'summary' field.")

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(self, llm) -> Tuple[str, bool]:
        """Return (question, success_flag)."""

        prompt = (
            self.ENRICHED_PROMPT
            if self.strategy == "report"
            else self.BASE_PROMPT
        )

        variables = {
            "headline": self.headline,
            "previous_qa": json.dumps(self.previous_qa, indent=2),
        }

        if self.strategy == "report":
            variables["summary"] = self.summary

        try:
            resp = (prompt | llm).invoke(variables)
            question = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            return question, bool(question)
        except Exception as e:
            return f"Q-Gen Error: {e}", False

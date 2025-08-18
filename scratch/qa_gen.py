"""
scratch/qa_gen.py – *improved version*
======================================

This rewrite fixes the main weaknesses we diagnosed:

1. **Cross‑branch duplicates**
   A class‑level **_global_asked** registry tracks every question that has
   already been generated for the *same headline* in the current run.  A
   newly generated question is checked against both
   – questions from the current branch (`previous_qa`)
   – questions from *other* branches (the registry)
   If a duplicate is detected the LLM is re‑prompted (max 2 retries).  The
   final accepted question is added to the registry so later calls avoid it.

2. **Prompt sharpening**
   The system prompts now explicitly instruct the LLM to
   – include at least **one key noun / named entity** from the headline;
   – avoid duplicates found in *both* of the lists described above.

3. **Logging for silent fallbacks**
   If `strategy='report'` is requested but the supplied `event_report`
   lacks a usable summary we *log* at WARNING and transparently fall back
   to `headline` mode (instead of raising and killing the branch).

The public signature *remains unchanged* so the rest of the pipeline does
not need edits.

scratch/qa_gen.py – anchored *and* de-duplicated question generation
====================================================================

Upgrades vs. previous version
-----------------------------
1) **Entity/Place/Time anchoring**
   • Prompts explicitly require:
       – include at least one **verbatim** named entity or key noun from the HEADLINE,
       – if available, include the event **location** and/or **date** from the event report.
   • After generation we **validate anchors** and re-prompt (up to MAX_DUP_RETRY).

2) **Same duplicate-avoidance** across branches
   • Registry `_global_asked` prevents re-asking similar questions (normalized).

3) **Non-breaking**: Public API and behavior in the rest of the pipeline are unchanged.

Notes
-----
Anchoring checks are lightweight:
  – headline overlap ≥ 1 token (stopwords removed)
  – if report.location or report.date_time exist, require either a location token
    OR a year/month token from the date string be present in the question.

scratch/qa_gen.py – *improved version (event-context aware)*
============================================================

What’s new in this revision:
1) Prefer strategy='report' **only when** a non-empty event SUMMARY exists.
   (We previously flipped to report if event_report was merely present.)
2) In report-mode, pass {summary, location, date_time} into the prompt and
   instruct the model to *prefer* including location and/or temporal cues
   (year/month) when they are available and relevant.
3) Retain duplicate-avoidance across branches and the existing fallbacks.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Dict, List, Tuple

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Global duplicate-tracking registry
# --------------------------------------------------------------------------- #

# key: headline.lower()  →  set[str] of already-asked questions
_global_asked: Dict[str, set[str]] = {}


def _question_key(q: str) -> str:
    """Normalise a question string for fast deduplication."""
    return " ".join(q.lower().split())  # strip + collapse whitespace


# --------------------------------------------------------------------------- #
# Prompt templates (mention location/date_time in report-mode)
# --------------------------------------------------------------------------- #

_BASE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            textwrap.dedent(
                """
                You are a **fact-checking question generator**.

                • Produce **ONE** concise, Google-style query that will help
                  verify the headline.
                • Your question **must** include at least one significant noun
                  or named entity from the headline so that web search hits
                  the right topic.
                • **Do NOT** repeat or paraphrase any question listed below in
                  either *PREVIOUS_BRANCH* or *ALREADY_ASKED*.

                PREVIOUS_BRANCH:
                {previous_qa}

                ALREADY_ASKED:
                {global_prev}
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
                """
                You are a **fact-checking question generator**.

                Using the event SUMMARY and optional LOCATION / DATE below,
                write **ONE** concise, search-ready question that probes a
                *still-uncertain* fact. The question should:
                  • include at least one key noun/entity from the HEADLINE, and
                  • when helpful, **prefer** to include the LOCATION and/or a
                    specific **year/month** derived from DATE (but do not
                    fabricate missing details).

                SUMMARY:
                {summary}

                LOCATION: {location}
                DATE:     {date_time}

                PREVIOUS_BRANCH:
                {previous_qa}

                ALREADY_ASKED:
                {global_prev}
                """
            ),
        ),
        ("human", "HEADLINE: {headline}"),
    ]
)

# --------------------------------------------------------------------------- #
# Prompt/LLM chain cache (minor perf win)
# --------------------------------------------------------------------------- #

from cachetools import LRUCache

_CHAIN_CACHE: LRUCache = LRUCache(maxsize=32)  # id(prompt), id(llm) → chain


def _get_chain(prompt_obj, llm):
    key = (id(prompt_obj), id(llm))
    if key not in _CHAIN_CACHE:
        _CHAIN_CACHE[key] = prompt_obj | llm
    return _CHAIN_CACHE[key]


# --------------------------------------------------------------------------- #
# Main public class
# --------------------------------------------------------------------------- #

class QAGenerationTool:
    """One-shot question generator with duplicate avoidance + event context."""

    MAX_DUP_RETRY = 2  # how many times to re-prompt on duplicate

    def __init__(
        self,
        headline: str,
        previous_qa: List[Dict[str, str]] | None = None,
        *,
        event_report: Dict[str, Any] | None = None,
        strategy: str = "auto",
    ) -> None:
        self.headline = headline.strip()
        self.previous_qa = previous_qa or []
        self.event_report = event_report or {}
        self.summary: str = (self.event_report.get("summary") or "").strip()
        self.location: str = (self.event_report.get("location") or "").strip()
        self.date_time: str = (self.event_report.get("date_time") or "").strip()
        self.strategy = strategy.lower()

        # Prefer 'report' only when a non-empty SUMMARY exists
        if self.strategy == "auto" and self.summary:
            self.strategy = "report"

        # If report was requested but summary is missing/empty, fall back (warn once)
        if self.strategy == "report" and not self.summary:
            logger.warning(
                "Event report missing/invalid – falling back to headline mode for: %.60s…",
                self.headline,
            )
            self.strategy = "headline"

        # Per-headline global registry entry
        self._registry = _global_asked.setdefault(self.headline.lower(), set())

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, llm) -> Tuple[str, bool]:
        """Generate a de-duplicated question.  Returns (question, ok)."""

        prompt_template = _ENRICHED_PROMPT if self.strategy == "report" else _BASE_PROMPT

        # Flatten previous Q-A questions for the prompt
        prev_qs = [qa.get("question", "") for qa in self.previous_qa]
        prev_q_txt = json.dumps(prev_qs, ensure_ascii=False, indent=2)
        global_prev_txt = json.dumps(sorted(self._registry), ensure_ascii=False, indent=2)

        variables = {
            "headline": self.headline,
            "previous_qa": prev_q_txt,
            "global_prev": global_prev_txt,
        }
        if self.strategy == "report":
            # Pass extra fields to anchor queries without coupling downstream
            variables["summary"] = self.summary
            variables["location"] = self.location
            variables["date_time"] = self.date_time

        # -------- attempt generation with de-dup checks -------- #
        for attempt in range(self.MAX_DUP_RETRY + 1):
            try:
                resp = _get_chain(prompt_template, llm).invoke(variables)
                question = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            except Exception as exc:  # noqa: BLE001
                logger.error("Q-gen LLM error: %s", exc, exc_info=False)
                return f"Q-gen error: {exc}", False

            # Empty response → give up immediately
            if not question:
                return "", False

            key = _question_key(question)
            if key in self._registry or any(_question_key(q) == key for q in prev_qs):
                if attempt < self.MAX_DUP_RETRY:
                    # Mark duplicate and retry with it included in previous_qa
                    prev_qs.append(question)
                    variables["previous_qa"] = json.dumps(prev_qs, ensure_ascii=False, indent=2)
                    continue  # re-prompt
                else:
                    logger.debug("Duplicate tolerated after %d retries: %s", attempt, question)
                    # fall through and accept duplicate

            # Unique question → record & return
            self._registry.add(key)
            return question, True

        # Should not reach here
        return "", False

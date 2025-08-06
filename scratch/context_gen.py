"""
scratch/context_gen.py
======================

Multimodal **EventContextGenerator** – builds a concise *event report*
from a news *headline* **and** its *image*.

The LLM is instructed to:

1. Write a 3-4 sentence **SUMMARY** describing *what*, *who*, *when*,
   *where*, *why* (only using visual + headline evidence).
2. Output a compact **JSON** block with the same facts
   (keys: "headline_restated", "entities", "date_time", "location",
   "action", "cause", "open_questions").

Return value
------------

>>> raw_text, report_dict = generator.run()

* `raw_text`     – full LLM response (str)
* `report_dict`  – parsed JSON + `"summary"` key
                   (empty `{}` if parsing failed)

The method is **memoised** so any (headline, image) pair is processed
at most once per run.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Dict, Tuple

from .cache import memo
from .model_router import ModelRouter

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an investigative fact-checker.

    TASK 1 — SUMMARY (3-4 sentences)
      • Describe WHAT is happening, WHO is involved, WHEN, WHERE, and WHY
        using ONLY evidence visible in the IMAGE and the HEADLINE text.
        No external knowledge.

    TASK 2 — JSON (same facts but compact)
      Output immediately after the summary under a header 'JSON:'.

      Keys:
        "headline_restated"
        "entities"        (list)
        "date_time"
        "location"
        "action"
        "cause"
        "open_questions"  (list of 2-3 unverified points)

    FORMAT EXACTLY:
    SUMMARY:
    <paragraph>

    JSON:
    <single-line JSON>
    """
)

# --------------------------------------------------------------------------- #
# Generator class
# --------------------------------------------------------------------------- #


class EventContextGenerator:
    def __init__(self, headline: str, image_path: str, model_router: ModelRouter):
        self.headline = headline
        self.image_path = image_path
        self._mr = model_router

    # ------------------------------------------------------------------ #
    # Cached public entry
    # ------------------------------------------------------------------ #

    @memo(maxsize=2_048)  # (headline, image_path) ↔ result
    def run(self) -> Tuple[str, Dict[str, Any]]:
        """
        Returns
        -------
        raw_text : str
            Full LLM output (summary + JSON).
        report   : dict
            Parsed JSON plus always a `"summary"` key.  Empty dict if parsing
            failed.
        """
        logger.debug("Generating event context for: %s", self.headline[:60])

        # Build multimodal message via ModelRouter helper
        try:
            resp = self._mr.call_multimodal(
                system_prompt=_SYSTEM_PROMPT,
                text_prompt=self.headline,
                image_path=self.image_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Event context generation failed: %s", exc, exc_info=False)
            return f"API Error: {exc}", {}

        raw = getattr(resp["raw"], "content", "") if resp else ""
        raw = raw.strip()

        # ------------------ JSON parsing ------------------ #
        summary = raw
        report: Dict[str, Any] = {}

        if "JSON:" in raw:
            summary_part, json_part = raw.split("JSON:", 1)
            summary = summary_part.replace("SUMMARY:", "").strip()

            try:
                report = json.loads(json_part.strip())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse JSON block (%s)", exc)

        # Always include the narrative summary
        report["summary"] = summary

        return raw, report

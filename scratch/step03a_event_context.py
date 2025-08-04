# scratch/step03a_event_context.py  (multimodal version)
from __future__ import annotations
import json
from typing import Tuple, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from model_router import ModelRouter   # <-- reuse router for multimodal

class EventContextGenerator:
    """
    Build a structured event report from HEADLINE + IMAGE (multimodal).
    Returns (raw_llm_text, report_dict).  If LLM fails to return valid JSON,
    the second element is {} and `ok` flag in calling code should fallback.
    """

    SYSTEM_PROMPT = (
            "You are an investigative fact-checker.\n"
            "TASK 1  –  SUMMARY  (3-4 sentences)\n"
            "  • Describe WHAT is happening (action), WHO is involved, WHEN (date/weekday/season), WHERE, and WHY (cause / consequence)\n"
            "  • Use ONLY evidence visible in the IMAGE plus the HEADLINE text.  No external knowledge.\n\n"
            "TASK 2  –  JSON  (same facts, compact)\n"
            "  Output immediately after the summary under a header 'JSON:'\n"
            '  Keys: "headline_restated", "entities", "date_time", "location", '
            '"action", "cause", "open_questions" (list of 2-3 unverified points)\n\n"'
            "FORMAT EXACTLY:\n"
            "SUMMARY:\n"
            "<paragraph>\n"
            "JSON:\n"
            "<single-line JSON>"
                )

    def __init__(self, headline: str, image_path: str, model_router: ModelRouter) -> None:
        self.headline = headline
        self.image_path = image_path
        self.mr = model_router

    def run(self) -> Tuple[str, Dict[str, Any]]:
        messages = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", self.headline)   # goes into the text part of multimodal msg
        ])

        # Build full multimodal message (system+human+image)
        multimodal_msg = self.mr.create_multimodal_message(
            system_prompt=self.SYSTEM_PROMPT,
            text_prompt=self.headline,
            image_path=self.image_path,
        )

        try:
            resp = self.mr.call_model(multimodal_msg)
            raw = getattr(resp["raw"], "content", "").strip()

            if "JSON:" in raw:
                summary_part, json_part = raw.split("JSON:", 1)
                summary = summary_part.replace("SUMMARY:", "").strip()
                try:
                    report = json.loads(json_part.strip())
                except Exception:
                    report = {}
            else:
                summary, report = raw, {}
            report["summary"] = summary    # always store the paragraph
            return raw, report
        except Exception:
            # malformed JSON → caller can decide to fall back
            return raw, {}

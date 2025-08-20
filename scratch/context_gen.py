"""
scratch/context_gen.py – resilient multimodal event context generator
=====================================================================

Upgrades
--------
1. **Robust JSON enforcement**
   • First call uses the original prompt.
   • If JSON parsing fails, we **retry once** with a stricter variant of the
     system prompt and `temperature = 0` to encourage deterministic output.

2. **Schema normalization with defaults (this change)**
   • After parsing, normalize to a fixed schema so downstream code never
     needs to guard for missing keys. Defaults:
       - Strings: headline_restated, date_time, location, action, cause → ""
       - Lists:   entities, open_questions → []
       - Always includes "summary" (string)

3. **Transparent logging**
   • Parsing errors are logged at *WARNING* once per (headline, image).
   • Retry success/failure is logged at *INFO* when `config.DEBUG` is True.

4. **Same public API & memoisation**
   • `run() → (raw_text, report_dict)` signature unchanged.
   • Results are still cached via `@memo` so downstream callers see no change.
"""


from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Dict, Tuple, List

from .cache import memo
from .model_router import ModelRouter
from . import config as C

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Base and strict prompts
# --------------------------------------------------------------------------- #

_BASE_PROMPT = textwrap.dedent(
    """
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

# Strict prompt adds one extra instruction to maximise parseability
_STRICT_PROMPT = _BASE_PROMPT + textwrap.dedent(
    """

    IMPORTANT: The JSON must be valid, single-line, and parseable by Python's
    json.loads().  Do NOT wrap it in triple backticks or markdown fences.
    """
)

# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #

def _as_str(x: Any) -> str:
    """Coerce arbitrary input to a trimmed string; None → ''."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    # If it's a short list/tuple of strings, join; else fallback to str()
    if isinstance(x, (list, tuple, set)):
        try:
            items = [str(i).strip() for i in x if str(i).strip()]
            return ", ".join(items)
        except Exception:
            return str(x).strip()
    return str(x).strip()


def _as_list_of_str(x: Any) -> List[str]:
    """Coerce to a list[str]; strings → [string], None/invalid → []."""
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, (list, tuple, set)):
        out: List[str] = []
        for i in x:
            s = str(i).strip()
            if s:
                out.append(s)
        # optional: deduplicate preserving order
        seen = set()
        uniq: List[str] = []
        for s in out:
            if s not in seen:
                uniq.append(s); seen.add(s)
        return uniq
    # Fallback: single coerced string if meaningful
    s = str(x).strip()
    return [s] if s else []


def _normalize_schema(parsed: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Return a dict with all expected keys present and correctly typed.
    Does not set "summary" — caller appends it separately.
    """
    d = parsed or {}
    return {
        "headline_restated": _as_str(d.get("headline_restated")),
        "entities": _as_list_of_str(d.get("entities")),
        "date_time": _as_str(d.get("date_time")),
        "location": _as_str(d.get("location")),
        "action": _as_str(d.get("action")),
        "cause": _as_str(d.get("cause")),
        "open_questions": _as_list_of_str(d.get("open_questions")),
    }





class EventContextGenerator:
    def __init__(self, headline: str, image_path: str, model_router: ModelRouter):
        self.headline = headline.strip()
        self.image_path = image_path
        self._mr = model_router

    # ------------------------------------------------------------------ #
    @memo(maxsize=2_048)
    def run(self) -> Tuple[str, Dict[str, Any]]:
        """Return (raw_text, report_dict) with graceful retry on JSON fail."""
        raw, report = self._invoke(_BASE_PROMPT)

        # Retry only if the JSON block did not parse successfully
        if report.get("_parsed_ok"):
            report.pop("_parsed_ok", None)  # hide internal flag from downstream
            return raw, report

        # Retry once with stricter prompt + deterministic temperature
        logger.warning("EventContext: JSON parse failed – retrying with strict mode for headline: %.60s…", self.headline)
        # Temporarily switch temperature to 0 for determinism
        orig_temp = self._mr.temperature if hasattr(self._mr, "temperature") else C.TEMPERATURE
        try:
            self._mr.switch_model(self._mr.model_name, temperature=0.0)
            raw2, report2 = self._invoke(_STRICT_PROMPT)
        finally:
            # Restore original temperature regardless of success
            self._mr.switch_model(self._mr.model_name, temperature=orig_temp)

        if C.DEBUG:
            logger.info("EventContext retry success=%s for headline: %.60s…", bool(report2), self.headline)

        final_raw, final_report = (raw2, report2) if report2.get("_parsed_ok") else (raw, report)
        final_report.pop("_parsed_ok", None)  # hide internal flag
        return final_raw, final_report

    # ------------------------------------------------------------------ #
    def _invoke(self, system_prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Helper that calls the LLM once and attempts JSON extraction."""
        try:
            resp = self._mr.call_multimodal(
                system_prompt=system_prompt,
                text_prompt=self.headline,
                image_path=self.image_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Event context LLM error: %s", exc, exc_info=False)
            return f"API Error: {exc}", {}

        raw = getattr(resp["raw"], "content", "").strip() if resp else ""

        summary = raw
        report: Dict[str, Any] = {}
        parsed_ok: bool = False

        if "JSON:" in raw:
            summary_part, json_part = raw.split("JSON:", 1)
            summary = summary_part.replace("SUMMARY:", "").strip()
            try:
                parsed = json.loads(json_part.strip())
                report = _normalize_schema(parsed)
                parsed_ok = True
            except Exception as exc:  # noqa: BLE001
                if C.DEBUG:
                    logger.debug("EventContext JSON parse error: %s", exc)
                # Provide normalized defaults even on parse failure
                report = _normalize_schema({})

        # Always include summary narrative and parse status flag
        report["summary"] = _as_str(summary)
        report["_parsed_ok"] = parsed_ok
        return raw, report

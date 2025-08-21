"""
scratch/final_classifier.py – smarter final decision (image wins when text is weak)
==================================================================================

What's new:
* We compute an average evidence strength from GOOD Q-A (`overlap_score`).
* If the GOOD Q-A is present but **weak** (avg overlap < EVIDENCE_STRENGTH_MIN),
  we defer to the image-headline relevancy:
    - IMAGE REFUTES  → Misinformation
    - IMAGE SUPPORTS → Not Misinformation

GOOD Q-A definition stays aligned with qa_selector (strict gates).

scratch/final_classifier.py – smarter final decision (deterministic)
===================================================================

Enhancements:
1) Deterministic verdict: temporarily set router temperature=0.0 for the
   final classification prompt, then restore.
2) Uses the same GOOD Q-A filter (incl. confidence) as the selector.
3) Heuristic short-circuit: if no GOOD Q-A and image says REFUTES/SUPPORTS.

Public interface:
    FinalClassifier(headline, relevancy_text, qa_pairs).run(router) -> (decision, reason)
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List, Tuple
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate

from . import config as C
from .qa_quality import is_good_qa



# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #

_SYSTEM_TEMPLATE = textwrap.dedent(
    """
    You are a senior misinformation-detection expert. Evaluate the headline
    below based strictly on the provided evidence.

    EVIDENCE
    --------
    • Image–headline relevancy: {relevancy}
    • Fact-checking Q-A pairs (only those marked GOOD):
      {qa_context}

    Guidelines
    ----------
    • If the GOOD Q-A evidence **directly refutes** the headline → "Misinformation".
    • If it **supports** the headline                       → "Not Misinformation".
    • If evidence is **inconclusive**                       → "Uncertain".
    Ignore any Q-A pair whose answer begins with "No relevant answer found" or
    "Insufficient evidence".
    Ignore the image relevancy unless it *clearly contradicts or outweighs* the
    GOOD Q-A evidence.

    Reply in exactly this format:
    DECISION: <Misinformation|Not Misinformation|Uncertain>
    REASON:   <concise paragraph citing evidence>
    """
)

_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_TEMPLATE),
        ("human", "HEADLINE: {headline}"),
    ]
)

from cachetools import LRUCache
_CHAIN_CACHE: LRUCache = LRUCache(maxsize=32)

def _get_chain(prompt_obj, llm):
    key = (id(prompt_obj), id(llm))
    if key not in _CHAIN_CACHE:
        _CHAIN_CACHE[key] = prompt_obj | llm
    return _CHAIN_CACHE[key]

_DECISION_RE = re.compile(r"DECISION:\s*(Misinformation|Not Misinformation|Uncertain)", re.I)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.I | re.S)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_NEG_SUPPORT_RE = re.compile(
    r"\b(false|fake|fabricated|hoax|debunk(?:ed|ing)?|misleading|incorrect|inaccurate|"
    r"no evidence|not true|never happened|refute(?:s|d)?|denied)\b",
    re.I,
)


    # ------------------------------------------------------------------ #
    # NEW helpers
    # ------------------------------------------------------------------ #


# --------------------------------------------------------------------------- #
# Main class
# --------------------------------------------------------------------------- #

class FinalClassifier:
    """Veracity classifier with deterministic temp and heuristic fallback."""

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
        self.qa_pairs_raw = qa_pairs
        self.extra_guidelines = extra_guidelines or ""
        self.good_pairs = [qa for qa in qa_pairs if is_good_qa(qa, emit_metrics=False)]

    def _parse_relevancy(self) -> str | None:
        txt = (self.relevancy_text or "").upper()
        if "IMAGE REFUTES" in txt:
            return "REFUTES"
        if "IMAGE SUPPORTS" in txt:
            return "SUPPORTS"
        return None

    def _short_circuit(self):
        """
        Return (decision, reason) if heuristic applies, else None.
        Respects config.USE_IMAGE_FALLBACK.
        """
        # Only consider image fallback if we *lack* good Q-A evidence
        if self.good_pairs:
            return None

        if not C.USE_IMAGE_FALLBACK:
            return ("Uncertain", "Image fallback disabled and no reliable Q‑A evidence.")

        rel_flag = self._parse_relevancy()
        if rel_flag == "REFUTES":
            return (
                "Misinformation",
                "IMAGE REFUTES and no reliable Q‑A evidence is available.",
            )
        if rel_flag == "SUPPORTS":
            # CHANGED: previously returned Not Misinformation on SUPPORTS; now prefer Uncertain.
            return (
                "Uncertain",
                "IMAGE SUPPORTS but no reliable Q‑A evidence is available.",
            )
        return ("Uncertain", "No reliable Q‑A evidence and the image relevancy is inconclusive.")
    
    def _image_override_if_weak(self):
        """
        If GOOD Q-A exists but is *weak* (avg overlap < C.EVIDENCE_STRENGTH_MIN),
        defer to the image relevancy when enabled.
        Returns (decision, reason) or None if not applicable.
        """
        if not self.good_pairs:
            return None
        try:
            avg_overlap = sum(float(qa.get("overlap_score", 0) or 0.0) for qa in self.good_pairs) / max(1, len(self.good_pairs))
        except Exception:
            avg_overlap = 0.0

        if not C.USE_IMAGE_FALLBACK or avg_overlap >= float(C.EVIDENCE_STRENGTH_MIN):
            return None

        rel_flag = self._parse_relevancy()
        if rel_flag == "REFUTES":
            return (
                "Misinformation",
                f"GOOD Q‑A evidence is weak (avg overlap {avg_overlap:.2f} < {C.EVIDENCE_STRENGTH_MIN}); image analysis REFUTES.",
            )
        if rel_flag == "SUPPORTS":
            return (
                "Not Misinformation",
                f"GOOD Q‑A evidence is weak (avg overlap {avg_overlap:.2f} < {C.EVIDENCE_STRENGTH_MIN}); image analysis SUPPORTS.",
            )
        # If image is inconclusive, do not override; proceed to LLM verdict.
        return None

    @staticmethod
    def _extract_years(text: str | None) -> set[int]:
        if not text:
            return set()
        try:
            return {int(m.group(0)) for m in _YEAR_RE.finditer(text)}
        except Exception:
            return set()

    @staticmethod
    def _current_year() -> int:
        try:
            return datetime.utcnow().year
        except Exception:
            # Safe fallback if clock missing
            return 2025

    def _answer_supports_headline(self, answer: str) -> bool:
        """
        Heuristic: treat as supportive if the answer does NOT contain strong refutation/negation cues.
        We intentionally avoid plain 'not' to reduce false negatives.
        """
        if not answer:
            return False
        return _NEG_SUPPORT_RE.search(answer) is None

    def _has_time_aligned_text_refutation(self) -> bool:
        """
        Despite the name, this returns True when there exists at least one GOOD Q‑A that:
            • is strong (overlap >= EVIDENCE_STRENGTH_MIN and conf >= MIN_CONF), and
            • is time‑aligned (answer mentions the target year, or a year matching the headline,
            or a recent year within OLD_EVENT_YEARS), and
            • semantically SUPPORTS the headline claim (no strong negation cues).
        This means the text **contradicts** the IMAGE REFUTES signal.
        """
        # Gather headline years for alignment fallback
        head_years = self._extract_years(self.headline)
        now = self._current_year()
        recent_cutoff = now - max(1, int(getattr(C, "OLD_EVENT_YEARS", 3)))

        for qa in self.good_pairs:
            try:
                overlap = float(qa.get("overlap_score", 0) or 0.0)
                conf = float(qa.get("answer_conf", 0.0) or 0.0)
                if overlap < float(getattr(C, "EVIDENCE_STRENGTH_MIN", 2)):
                    continue
                if conf < float(getattr(C, "MIN_CONF", 0.3)):
                    continue

                ans = (qa.get("answer") or "").strip()
                if not self._answer_supports_headline(ans):
                    continue

                target_year = qa.get("target_year", None)
                years_in_ans = self._extract_years(ans)
                years_in_query = self._extract_years(qa.get("query_used", ""))

                aligned = False
                if isinstance(target_year, int):
                    aligned = (target_year in years_in_ans) or (target_year in years_in_query)
                if not aligned and head_years:
                    aligned = bool(years_in_ans & head_years) or bool(years_in_query & head_years)
                if not aligned and years_in_ans:
                    aligned = any(y >= recent_cutoff for y in years_in_ans)
                if not aligned and years_in_query:
                    aligned = any(y >= recent_cutoff for y in years_in_query)

                if aligned:
                    return True
            except Exception:
                # Best-effort; ignore malformed QA records
                continue
        return False

    def run(self, router) -> Tuple[str, str]:
        # Heuristic fast path
        # --- NEW pre-decision guards (make IMAGE REFUTES decisive by default) ---
        rel_flag = self._parse_relevancy()
        if rel_flag == "REFUTES":
            # If we do not have strong, time-aligned textual support for the headline,
            # call Misinformation immediately.
            if not self._has_time_aligned_text_refutation():
                return (
                    "Misinformation",
                    "IMAGE REFUTES and no time‑aligned text support for the headline was found in the GOOD Q‑A.",
                )
        if rel_flag == "SUPPORTS" and not self.good_pairs:
            # Do not claim Not‑Misinformation on image alone when no GOOD Q‑A exists
            return (
                "Uncertain",
                "IMAGE SUPPORTS but there is no reliable Q‑A evidence; marking Uncertain.",
            )

        # Heuristic fast path (after pre-guards)
        heuristic = self._short_circuit()
        if heuristic is not None:
            return heuristic

        # If GOOD Q-A is present but weak, allow image to override.
        weak_override = self._image_override_if_weak()
        if weak_override is not None:
            return weak_override


        qa_context = "\n".join(f"Q: {qa['question']}\nA: {qa['answer']}" for qa in self.good_pairs)

        # Deterministic temp for final verdict
        orig_model = router.model_name
        orig_temp = router.temperature
        try:
            router.switch_model(orig_model, temperature=0.0)
            llm0 = router.get()
            chain = _get_chain(_PROMPT_TEMPLATE, llm0)
            if self.extra_guidelines:
                chain = chain.partial(__system_override=self.extra_guidelines)
            resp = chain.invoke({"headline": self.headline, "relevancy": self.relevancy_text, "qa_context": qa_context})
            raw = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        except Exception as exc:
            # Deterministic fallback: if the LLM fails, degrade to simple rule
            rel_flag = self._parse_relevancy()
            if rel_flag == "REFUTES":
                return ("Misinformation", f"Classifier error: {exc}. Falling back to image REFUTES.")
            if rel_flag == "SUPPORTS":
                return ("Not Misinformation", f"Classifier error: {exc}. Falling back to image SUPPORTS.")
            return ("Uncertain", f"Classifier error: {exc}.")
        finally:
            router.switch_model(orig_model, temperature=orig_temp)

        dec_match = _DECISION_RE.search(raw)
        reason_match = _REASON_RE.search(raw)
        decision = dec_match.group(1).strip() if dec_match else "Uncertain"
        reason = reason_match.group(1).strip() if reason_match else "No reason provided"
        return decision, reason

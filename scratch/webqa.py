"""
scratch/webqa.py – Web Q&A with at-source quality gating
========================================================

Now headline-aware:
• We pass the headline into brave_snippets so snippets must match both.
• Our overlap_score mirrors that notion: per snippet score is
  min(question-overlap, headline-overlap); overall we take the max across snippets.

scratch/webqa.py – Web Q&A with adaptive freshness and temporal bias
====================================================================

What’s new:
• We look for a 4-digit year in the QUESTION, or in the event report's date.
• If we detect an older year (>= OLD_EVENT_YEARS ago) and policy is "auto",
  we disable freshness for this call.
• We pass `target_year` to Brave so snippets mentioning that year get a small bonus.


scratch/webqa.py – Web Q&A with adaptive freshness, temporal bias, and confidence gating
=======================================================================================

Adds:
• Stores answer_conf from ModelRouter and applies MIN_CONF gate.

scratch/webqa.py – Web Q&A with adaptive freshness, temporal bias, confidence gating,
and conditional widening when snippets are thin.
====================================================================================

Adds (Step 14):
• If the first retrieval yields fewer than MIN_SNIPPETS, we **auto-retry once**
  with a broadened query that appends a disambiguating token (LOCATION and/or YEAR).
• Records widening metadata: widen_attempted, widen_success, query_used.

Still includes earlier improvements:
• Headline-aware overlap gating, MIN_CONF gating, adaptive freshness/target_year.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from . import config as C
from . import log
from .brave_client import brave_snippets
from .evidence_llm import answer_from_evidence

logger = logging.getLogger(__name__)

_STOP = {
    "the", "a", "an", "and", "or", "but", "on", "in", "at", "to", "of", "for",
    "with", "by", "about", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "as", "from", "into", "than",
}
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


_PUNCT_RE = re.compile(r"[^\w\s-]")  # conservative: keep letters/digits/_ and hyphens
def _sanitize_query(q: str, max_len: int = 120) -> str:
    q0 = (q or "").strip()
    q1 = _PUNCT_RE.sub(" ", q0)
    q2 = " ".join(q1.split())
    return q2[:max_len].strip()

def _tokenise(text: str | None) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOP]


def _overlap(tokens: List[str], snippet: str) -> int:
    s = set(_tokenise(snippet))
    return sum(1 for t in tokens if t in s)


def _extract_year_from_text(s: str | None) -> Optional[int]:
    if not s:
        return None
    m = _YEAR_RE.search(s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _decide_freshness_and_year(
    question: str,
    event_report: Optional[Dict[str, object]],
) -> Tuple[Optional[int], Optional[int]]:
    """Return (freshness_days, target_year) based on policy and available dating."""
    if C.FRESHNESS_POLICY.lower() == "off":
        yr = _extract_year_from_text((event_report or {}).get("date_time") if event_report else None)
        return None, yr

    q_year = _extract_year_from_text(question)
    if q_year is not None:
        return None, q_year

    rep_year = _extract_year_from_text((event_report or {}).get("date_time") if event_report else None)
    if rep_year is not None:
        age = datetime.utcnow().year - rep_year
        if C.FRESHNESS_POLICY.lower() == "auto" and age >= max(0, int(C.OLD_EVENT_YEARS)):
            # Old event → don’t force recency; but we still anchor year
            return None, rep_year
        return C.FRESHNESS_DAYS, rep_year

    return (C.FRESHNESS_DAYS if C.FRESHNESS_POLICY.lower() in {"auto", "force"} else None, None)


def _build_widened_query(
    base_q: str,
    *,
    headline: Optional[str],
    event_report: Optional[Dict[str, object]],
) -> Tuple[str, bool, Dict[str, object]]:
    """
    Create a single broadened query by appending LOCATION and/or YEAR
    if they are not already present in the question.
    Returns (query, widened_flag, details).
    """
    base_lower = (base_q or "").lower()
    details: Dict[str, object] = {"location_used": None, "year_used": None}

    # Prefer location and year from event report
    loc = (event_report or {}).get("location") if event_report else None
    loc = (loc or "").strip()
    yr = _extract_year_from_text((event_report or {}).get("date_time") if event_report else None)

    # If year still missing, try the headline
    if yr is None:
        yr = _extract_year_from_text(headline)

    tokens_to_add: List[str] = []

    if loc and loc.lower() not in base_lower:
        tokens_to_add.append(loc)
        details["location_used"] = loc

    if yr is not None and str(yr) not in base_lower:
        tokens_to_add.append(str(yr))
        details["year_used"] = yr

    if not tokens_to_add:
        return base_q, False, details

    widened = f"{base_q} " + " ".join(tokens_to_add)
    return widened, True, details

def _backoff_query(base_q: str, *, headline: Optional[str], event_report: Optional[Dict[str, object]]) -> str:
    """
    Make an easier, verification-first query from the same content.
    Heuristics:
      - sanitize punctuation
      - keep top tokens from QUESTION + HEADLINE
      - append LOCATION/YEAR if available (not already present)
    """
    # reuse sanitizer
    q = _sanitize_query(base_q, max_len=140)
    # add loc/year with existing helper
    widened, _, _ = _build_widened_query(q, headline=headline, event_report=event_report)
    return widened

class WebQAModule:
    """One Web-QA round with headline-aware gating, adaptive freshness, confidence, and conditional widen."""

    def __init__(
        self,
        question: str,
        router,
        k: int = C.BRAVE_K,
        *,
        headline: Optional[str] = None,
        event_report: Optional[Dict[str, object]] = None,
    ) -> None:
        self.question = question.strip()
        self.router = router
        self.k = k
        self.headline = headline.strip() if headline else None
        self.event_report = event_report or {}

    async def _retrieve(self, query: str, freshness_days: Optional[int], target_year: Optional[int]) -> List[str]:
        return await brave_snippets(
            query,
            k=self.k,
            headline=self.headline,
            freshness_days=freshness_days,
            target_year=target_year,
        )

    async def run(self) -> Dict[str, object]:
        freshness_days, target_year = _decide_freshness_and_year(self.question, self.event_report)

        # 1) First retrieval
        try:
            snippets: List[str] = await self._retrieve(self.question, freshness_days, target_year)
        except Exception as exc:  # noqa: BLE001
            # Retry with sanitized query and no freshness
            logger.warning("Brave error on first try (%s). Retrying with sanitized query.", exc)
            log.metric("brave_error")
            s_q = _sanitize_query(self.question)
            try:
                snippets = await self._retrieve(s_q, None, target_year)
                if not snippets:
                    raise RuntimeError("No snippets after sanitize-retry")
                self_question_used = s_q
            except Exception as exc2:
                msg = f"Brave error (after sanitize): {exc2}"
                logger.warning(msg)
                return self._make_record(
                    answer=msg, answer_conf=0.0, snippets=[], ok=False, overlap_score=0,
                    freshness_days_used=None, target_year=target_year, query_used=s_q,
                    widen_attempted=False, widen_success=False, widen_details={},
                )
        else:
            self_question_used = self.question

        # 2) If thin, **auto-retry once** with widened query
        widen_attempted = False
        widen_success = False
        widen_details: Dict[str, object] = {}
        query_used = self_question_used

        if len(snippets) < C.MIN_SNIPPETS:
            widen_q, widened, details = _build_widened_query(
                self.question, headline=self.headline, event_report=self.event_report
            )
            widen_attempted = widened
            widen_details = details
            if widened:
                try:
                    wid_snips = await self._retrieve(widen_q, freshness_days, target_year)
                    # Prefer widened result if it improved coverage
                    if len(wid_snips) > len(snippets):
                        snippets = wid_snips
                        query_used = widen_q
                        widen_success = True
                except Exception as exc:  # noqa: BLE001
                    logger.info("Widened retrieval failed (%s); keeping original snippets.", exc)

        # 3) Answer from evidence (with confidence + headline self-policing)
        answer, ok_llm, conf = answer_from_evidence(
            self.router,
            self.question,
            snippets,
            headline=self.headline,
        )

        # 4) Quality signals (headline-aware min-overlap)
        q_tokens = _tokenise(self.question)
        h_tokens = _tokenise(self.headline) if self.headline else []
        per_snip_scores = []
        for s in snippets:
            oq = _overlap(q_tokens, s)
            if h_tokens:
                oh = _overlap(h_tokens, s)
                per_snip_scores.append(min(oq, oh))
            else:
                per_snip_scores.append(oq)
        overlap_score = max(per_snip_scores, default=0)
        snippet_count = len(snippets)
        answer_len = len(answer or "")

        # 5) Gates (incl. confidence)
        ok = bool(ok_llm)
        if snippet_count < C.MIN_SNIPPETS:
            ok = False
            log.metric("qa_gate_min_snippets")
        if overlap_score < C.MIN_OVERLAP:
            ok = False
            log.metric("qa_gate_min_overlap")
        if answer_len < C.MIN_ANSWER_CHARS:
            ok = False
            log.metric("qa_gate_min_answer_len")
            # Backoff once with a more general, verification-first query
            try:
                backoff_q = _backoff_query(self.question, headline=self.headline, event_report=self.event_report)
                if backoff_q and backoff_q != query_used:
                    wid_snips = await self._retrieve(backoff_q, None, target_year)
                    # Re-answer on backoff snippets
                    if wid_snips:
                        b_answer, b_ok_llm, b_conf = answer_from_evidence(
                            self.router, self.question, wid_snips, headline=self.headline
                        )
                        b_len = len((b_answer or "").strip())
                        # Adopt backoff result only if it materially improves length/ok/conf
                        if b_ok_llm and b_len >= C.MIN_ANSWER_CHARS and b_conf >= max(0.25, conf):
                            answer, conf = b_answer, b_conf
                            snippets = wid_snips
                            snippet_count = len(wid_snips)
                            # recompute overlap & flags
                            per_snip_scores = []
                            for s in wid_snips:
                                oq = _overlap(q_tokens, s)
                                oh = _overlap(h_tokens, s) if h_tokens else oq
                                per_snip_scores.append(min(oq, oh))
                            overlap_score = max(per_snip_scores, default=0)
                            ok = True
                            query_used = backoff_q
                            log.metric("webqa_backoff_success")
                        else:
                            log.metric("webqa_backoff_no_gain")
                    else:
                        log.metric("webqa_backoff_no_snips")
            except Exception:
                log.metric("webqa_backoff_error")
        if conf < C.MIN_CONF:
            ok = False
            log.metric("qa_gate_low_conf")

        rec = self._make_record(
            answer=answer,
            answer_conf=conf,
            snippets=snippets,
            ok=ok,
            overlap_score=overlap_score,
            freshness_days_used=freshness_days,
            target_year=target_year,
            query_used=query_used,
            widen_attempted=widen_attempted,
            widen_success=widen_success,
            widen_details=widen_details,
        )

        if C.DEBUG:
            logger.info(
                "WebQA ok=%s conf=%.2f snips=%d overlap=%d fresh=%s year=%s widened=%s q=%.80s",
                rec["ok"],
                rec["answer_conf"],
                rec["snippet_count"],
                rec["overlap_score"],
                rec["freshness_days_used"] if rec["freshness_days_used"] is not None else "none",
                rec["target_year"] if rec["target_year"] is not None else "na",
                f"{rec['widen_attempted']}/{rec['widen_success']}",
                rec["query_used"],
            )
        return rec

    def _make_record(
        self,
        *,
        answer: str,
        answer_conf: float,
        snippets: List[str],
        ok: bool,
        overlap_score: int,
        freshness_days_used: Optional[int],
        target_year: Optional[int],
        query_used: str,
        widen_attempted: bool,
        widen_success: bool,
        widen_details: Dict[str, object],
    ) -> Dict[str, object]:
        return {
            "question": self.question,
            "answer": (answer or "").strip(),
            "answer_conf": float(answer_conf),
            "snippets": snippets,
            "ok": bool(ok),
            "snippet_count": len(snippets),
            "overlap_score": int(overlap_score),
            "headline": self.headline,
            "freshness_days_used": freshness_days_used,
            "target_year": target_year,
            "query_used": query_used,
            "widen_attempted": widen_attempted,
            "widen_success": widen_success,
            "widen_details": widen_details,
        }

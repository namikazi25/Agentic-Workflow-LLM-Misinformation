"""
scratch/qa_quality.py
=====================

Centralised predicate for assessing whether a Q-A pair is "GOOD"
according to the project's quality gates. This avoids divergence
between the selector and the final classifier.

GOOD means all of:
  - qa.ok is True
  - answer does not start with a bad sentinel
  - snippet_count >= config.MIN_SNIPPETS
  - overlap_score >= config.MIN_OVERLAP
  - len(answer) >= config.MIN_ANSWER_CHARS
  - answer_conf >= config.MIN_CONF

Optional diagnostics:
  - If emit_metrics=True, increment counters for certain drop reasons
    using metric names "<metrics_prefix>_drop_short_answer",
    "<metrics_prefix>_drop_low_overlap", "<metrics_prefix>_drop_low_conf".
    (Per prior behavior, snippet_count failures are not counted.)
"""

from __future__ import annotations

from typing import Any, Dict, List

from . import config as C
from . import log

__all__ = ["is_good_qa"]

_BAD_SENTINELS = (
    "no relevant answer found",
    "insufficient evidence",
    "brave error",
)


def is_good_qa(
    qa: Dict[str, Any],
    *,
    emit_metrics: bool = False,
    metrics_prefix: str = "selector",
) -> bool:
    """
    Return True if the QA pair passes all gates. Optionally emit diagnostics.
    """
    if not qa or not qa.get("question") or not qa.get("answer"):
        return False

    if qa.get("ok") is not True:
        return False

    ans = (qa.get("answer") or "").strip()
    if any(ans.lower().startswith(s) for s in _BAD_SENTINELS):
        return False

    reasons: List[str] = []

    if len(ans) < C.MIN_ANSWER_CHARS:
        reasons.append("short_answer")

    # Keep snippet_count as a hard gate without emitting a metric, per prior brief.
    if qa.get("snippet_count", 0) < C.MIN_SNIPPETS:
        return False

    if qa.get("overlap_score", 0) < C.MIN_OVERLAP:
        reasons.append("low_overlap")

    try:
        conf_val = float(qa.get("answer_conf", 0.0))
    except Exception:
        conf_val = 0.0
    if conf_val < C.MIN_CONF:
        reasons.append("low_conf")

    if reasons:
        if emit_metrics and metrics_prefix:
            for r in set(reasons):
                if r == "short_answer":
                    log.metric(f"{metrics_prefix}_drop_short_answer")
                elif r == "low_overlap":
                    log.metric(f"{metrics_prefix}_drop_low_overlap")
                elif r == "low_conf":
                    log.metric(f"{metrics_prefix}_drop_low_conf")
        return False

    return True

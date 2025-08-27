# scratch/heads/consistency_head.py
from __future__ import annotations
from typing import Dict, Any
from ..relevancy import ImageHeadlineRelevancyChecker

def _parse(text: str) -> str:
    u = (text or "").upper()
    if "FINISH[IMAGE REFUTES]" in u:  return "REFUTE"
    if "FINISH[IMAGE SUPPORTS]" in u: return "SUPPORT"
    return "UNKNOWN"

def run_consistency_head(checker: ImageHeadlineRelevancyChecker, image_path: str, headline: str) -> Dict[str, Any]:
    out = checker.check_relevancy(image_path, headline)  # you already have this
    text = out.get("text", "")
    lbl = _parse(text)
    conf = float(out.get("confidence", 0.6))  # added in relevancy patch below
    # probabilities from verdict + confidence; soft mapping is fine for start
    if lbl == "REFUTE":
        return {"vl_support_prob": 0.2, "vl_refute_prob": 0.8, "consistency_conf": conf, "verdict": lbl, "rationale": text}
    if lbl == "SUPPORT":
        return {"vl_support_prob": 0.8, "vl_refute_prob": 0.2, "consistency_conf": conf, "verdict": lbl, "rationale": text}
    return {"vl_support_prob": 0.5, "vl_refute_prob": 0.5, "consistency_conf": conf, "verdict": "UNKNOWN", "rationale": text}

# scratch/heads/text_head.py
from __future__ import annotations
from typing import Dict, Any
import re
from .. import config as C

_NEG = re.compile(r"\b(false|fake|fabricated|hoax|misleading|incorrect|no evidence|never happened|refute(?:s|d)?|debunk(?:ed|ing)?)\b", re.I)

def from_webqa(record: Dict[str, Any]) -> Dict[str, float]:
    """
    record: output dict from WebQAModule.run()
    returns a compact text-head feature dict
    """
    ans = (record.get("answer") or "").strip()
    conf = float(record.get("answer_conf", 0.5))
    overlap = float(record.get("overlap_score", 0.0))
    # polarity: crude but effective; you can tighten later
    refute = 1.0 if _NEG.search(ans) else 0.0
    support = 1.0 - refute
    return {
        "text_support": support,
        "text_refute": refute,
        "overlap_max": overlap,     # keep integer; fuser will scale
        "answer_conf": conf,
    }

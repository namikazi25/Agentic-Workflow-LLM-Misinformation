from __future__ import annotations
import joblib, numpy as np
from . import config as C

_M = None
def _load():
    global _M
    if _M is None:
        _M = joblib.load(C.FUSER_MODEL_PATH)
    return _M["clf"], _M["features"]

def _get(d,k,default=0.0):
    try: return float(d.get(k,default))
    except: return float(default)

def _vectorize(record, feats):
    t, v, c = record["head_text"], record["head_visual"], record["head_consistency"]
    mapping = {
        "text_support": _get(t,"text_support",0.5),
        "text_refute": _get(t,"text_refute",0.5),
        "overlap_max": _get(t,"overlap_max",0.0)/5.0,
        "answer_conf": _get(t,"answer_conf",0.5),
        "ai_gen_prob": _get(v,"ai_gen_prob",0.5),
        "manip_prob": _get(v,"manip_prob",0.5),
        "photo_like": _get(v,"photo_like",0.5),
        "vl_support_prob": _get(c,"vl_support_prob",0.5),
        "vl_refute_prob": _get(c,"vl_refute_prob",0.5),
    }
    return np.array([mapping[k] for k in feats], dtype=float).reshape(1,-1)

def decide(record, tau=0.60):
    clf, feats = _load()
    x = _vectorize(record, feats)
    p = float(clf.predict_proba(x)[0,1])

    tconf = float(record["head_text"].get("answer_conf",0.5))
    vconf = float(record["head_visual"].get("auth_conf",0.5))
    cconf = float(record["head_consistency"].get("consistency_conf",0.5))
    if max(tconf, vconf, cconf) < tau:
        return "Uncertain", p, "Low confidence across heads."

    if record["head_text"].get("text_support",0.0) > 0.6 and record["head_consistency"].get("vl_refute_prob",0.0) > 0.6:
        return "Uncertain", p, "Strong text support but visual refutation (conflict)."

    return ("Misinformation" if p>=0.5 else "Not Misinformation"), p, "Learned fusion."

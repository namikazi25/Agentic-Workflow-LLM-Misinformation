# scratch/heads/visual_head.py
from __future__ import annotations
from typing import Dict, Any
from ..model_router import ModelRouter

_PROMPT = (
    "You are an image forensics analyst.\n"
    "Decide if the NEWS IMAGE is likely a real camera photo or AI-generated/manipulated.\n"
    "Use ONLY visible cues (textures, text rendering, anatomy, shadows). No outside facts.\n"
    "Respond with a one-line justification and EXACTLY one of:\n"
    "Finish[LIKELY_REAL] or Finish[LIKELY_FAKE] or Finish[UNKNOWN]"
)

def _parse_token(text: str) -> str:
    u = (text or "").upper()
    if "FINISH[LIKELY_FAKE]" in u: return "LIKELY_FAKE"
    if "FINISH[LIKELY_REAL]" in u: return "LIKELY_REAL"
    if "FINISH[UNKNOWN]" in u:     return "UNKNOWN"
    return "UNKNOWN"

def run_visual_head(router: ModelRouter, image_path: str) -> Dict[str, Any]:
    try:
        msgs = router.create_multimodal_message(_PROMPT, "Analyze this image.", image_path)
        out = router.call(msgs)
        raw = out.get("raw")
        text = getattr(raw, "content", "") or ""
        token = _parse_token(text)
        conf = float(out.get("confidence", 0.6))
    except Exception as exc:
        return {"ai_gen_prob": 0.5, "manip_prob": 0.5, "photo_like": 0.5, "auth_conf": 0.3, "auth_label": "UNKNOWN",
                "rationale": f"Error: {exc}"}

    # map token to a simple score; you can refine with ELA/EXIF later
    ai_gen_prob = 0.7 if token == "LIKELY_FAKE" else (0.3 if token == "LIKELY_REAL" else 0.5)
    photo_like  = 1.0 - abs(ai_gen_prob - 0.5) * 2.0  # 1 when far from 0.5
    return {
        "ai_gen_prob": float(ai_gen_prob),
        "manip_prob": 0.5,          # placeholder; add ELA later if you want
        "photo_like": float(photo_like),
        "auth_conf": conf,
        "auth_label": token,
        "rationale": text.strip(),
    }

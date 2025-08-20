"""
scratch/model_router.py
=======================

Single, reusable *ModelRouter* that supports **OpenAI GPT-models** and
**Google Gemini** via LangChain wrappers.

Key improvements over the legacy version
----------------------------------------
1. **Singleton LLMs** – an LLM is created once per (model-name, temperature)
   combination and cached in-memory.  Subsequent calls reuse it with *zero*
   overhead.

2. **Memoised `encode_image`** – converts each image file to base-64 at most
   once per run (LRU cache, 1 Ki images by default).

3. Reduced public surface:
   • `get()` ⇒ returns the cached LangChain LLM  
   • `call()` ⇒ retry wrapper, returns `{"raw": …, "confidence": float}`  
   • `create_multimodal_message()` ⇒ helper for Gemini vision

Environment variables
---------------------
* ``OPENAI_API_KEY``
* ``GEMINI_API_KEY``   (a/k/a ``GOOGLE_API_KEY``)

Both are read lazily; no import-time failure if missing.
"""

from __future__ import annotations

import os
import math
import time
import json
import re
from typing import Any, Dict, Optional, Tuple, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .cache import memo
from . import config as C

# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


@memo(maxsize=8)
def _make_llm(
    model_name: str,
    temperature: float,
    openai_key: str | None,
    google_key: str | None,
):
    """
    Factory that returns a *single* instance per unique argument tuple
    thanks to the @memo decorator.
    """
    mdl = model_name.lower()

    if "gpt" in mdl:
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not set.")
        return ChatOpenAI(
            api_key=openai_key,
            model=model_name,
            temperature=temperature,
            logprobs=1,
            model_kwargs={"return_full_text": False},
        )

    if "gemini" in mdl:
        if not google_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_key,
            temperature=temperature,
            candidate_count=1,
        )

    raise ValueError(f"Unsupported model: {model_name!r}")


@memo(maxsize=1024)  # encodes up to 1 024 unique image paths
def encode_image(image_path: str) -> Tuple[str | None, str | None]:
    """
    Convert *image_path* → ``(base64_str, mime_type)`` once per run.

    Returns ``(None, None)`` if the file does not exist OR Pillow fails
    to read it (caller decides next step).
    """
    import base64
    from io import BytesIO
    from pathlib import Path

    from PIL import Image  # heavy import deferred

    path = Path(image_path)
    if not path.is_file():
        return None, None

    try:
        with Image.open(path) as img:
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=90)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return b64, "image/jpeg"
    except Exception:
        return None, None


# --------------------------------------------------------------------------- #
# Public class
# --------------------------------------------------------------------------- #


class ModelRouter:
    """
    Wrapper that unifies interaction with LangChain models
    and provides retry / confidence-score convenience.
    """

    def __init__(
        self,
        model_name: str = C.MODEL_DEFAULT,
        temperature: float = C.TEMPERATURE,
        *,
        max_retries: int = 5,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.google_key = google_api_key or os.getenv("GEMINI_API_KEY")

        # Cached LLM instance (singleton behaviour)
        self._llm = _make_llm(
            self.model_name,
            self.temperature,
            self.openai_key,
            self.google_key,
        )

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def get(self):
        """Return the cached LangChain LLM instance."""
        return self._llm

    def switch_model(self, new_model_name: str, *, temperature: float | None = None):
        """Swap to a different model *without* reinstantiating existing ones."""
        self.model_name = new_model_name
        if temperature is not None:
            self.temperature = temperature
        self._llm = _make_llm(
            self.model_name,
            self.temperature,
            self.openai_key,
            self.google_key,
        )

    # ------------------------------------------------------------------ #
    # Multimodal helper
    # ------------------------------------------------------------------ #

    def create_multimodal_message(
        self,
        system_prompt: str,
        text_prompt: str,
        image_path: str,
    ):
        """
        Build a Gemini-compatible multimodal message list.

        Raises
        ------
        ValueError
            If the image cannot be encoded (non-existent or corrupt file).
        """
        b64, mime = encode_image(image_path)
        if not b64:
            raise ValueError(f"Cannot encode image: {image_path}")

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            ),
        ]

    # ------------------------------------------------------------------ #
    # Core invocation with retry & confidence
    # ------------------------------------------------------------------ #

    def call(self, messages) -> Dict[str, Any]:
        """
        Invoke the LLM with *messages* (either list[Message] or str prompt).

        Returns
        -------
        dict
            ``{"raw": <LangChain return>, "confidence": float}``

        Confidence is estimated heuristically with model-aware fallbacks:
        • OpenAI:   exp(mean token log-prob) over first N tokens, if available.
        • Gemini:   use candidate.probability when present; otherwise token logprobs if exposed.
        • Fallback: structure/format compliance score derived from the text.
        """
        tries = 0
        while tries < self.max_retries:
            try:
                resp = self._llm.invoke(messages)

                
                conf = self._infer_confidence(resp)
                return {"raw": resp, "confidence": conf}

            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if any(tok in msg for tok in ("rate limit", "429", "quota", "please try again")):
                    time.sleep(2**tries)  # exponential back-off: 1,2,4,8,…
                    tries += 1
                    continue
                raise  # non-retryable

        raise RuntimeError("Max retries exhausted.")

    # ------------------------------------------------------------------ #
    # Confidence helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _bound01(x: float) -> float:
        try:
            return float(min(max(x, 0.0), 1.0))
        except Exception:
            return 0.5

    def _logprobs_to_conf(self, logs: List[float]) -> float:
        if not logs:
            return 0.5
        try:
            mean_lp = sum(float(v) for v in logs) / max(1, len(logs))
            return self._bound01(math.exp(mean_lp))
        except Exception:
            return 0.5

    def _extract_openai_token_logprobs(self, resp) -> List[float]:
        """
        Try multiple LangChain/OpenAI shapes to harvest token logprobs.
        Returns a (possibly empty) list of floats.
        """
        logs: List[float] = []
        # Legacy/explicit path used earlier in this project
        try:
            llm_out = getattr(resp, "llm_output", {}) or {}
            cand = llm_out.get("token_logprobs")
            if isinstance(cand, list) and cand:
                return [float(x) for x in cand if isinstance(x, (int, float))]
        except Exception:
            pass

        # Newer LangChain response metadata variants
        try:
            meta = getattr(resp, "response_metadata", {}) or {}
            # 1) meta["token_logprobs"] → [floats]
            cand = meta.get("token_logprobs")
            if isinstance(cand, list) and cand:
                return [float(x) for x in cand if isinstance(x, (int, float))]
            # 2) meta["logprobs"]["content"] → [{"logprob": -0.1, ...}, ...]
            lp = meta.get("logprobs")
            if isinstance(lp, dict) and isinstance(lp.get("content"), list):
                for item in lp["content"]:
                    val = item.get("logprob") if isinstance(item, dict) else None
                    if isinstance(val, (int, float)):
                        logs.append(float(val))
                if logs:
                    return logs
            # 3) meta["choices"][0]["logprobs"]["content"][i]["logprob"]
            choices = meta.get("choices")
            if isinstance(choices, list) and choices:
                ch = choices[0] or {}
                ch_lp = ch.get("logprobs") if isinstance(ch, dict) else None
                if isinstance(ch_lp, dict) and isinstance(ch_lp.get("content"), list):
                    for item in ch_lp["content"]:
                        val = item.get("logprob") if isinstance(item, dict) else None
                        if isinstance(val, (int, float)):
                            logs.append(float(val))
                    if logs:
                        return logs
        except Exception:
            pass
        return logs

    def _extract_gemini_prob_or_logs(self, resp) -> Tuple[Optional[float], List[float]]:
        """
        Gemini sometimes exposes candidate.probability; otherwise token logprobs
        may be absent. Return (probability|None, logprobs_list).
        """
        prob: Optional[float] = None
        logs: List[float] = []
        try:
            if hasattr(resp, "candidates") and resp.candidates:
                cand = resp.candidates[0]
                # candidate.probability is not guaranteed across releases
                p = getattr(cand, "probability", None)
                if isinstance(p, (int, float)):
                    prob = float(p)
        except Exception:
            pass
        try:
            meta = getattr(resp, "response_metadata", {}) or {}
            cand = meta.get("token_logprobs")
            if isinstance(cand, list) and cand:
                logs = [float(x) for x in cand if isinstance(x, (int, float))]
        except Exception:
            pass
        return prob, logs

    def _structure_conf_from_text(self, resp) -> float:
        """
        Fallback: derive a soft confidence from output structure/length and
        formats expected elsewhere in this project.
        """
        try:
            text = getattr(resp, "content", None) or str(resp or "")
        except Exception:
            text = ""
        t = (text or "").strip()
        if not t:
            return 0.2

        # Length banding
        n = len(t)
        if n < 10:
            conf = 0.2
        elif n < 40:
            conf = 0.4
        elif n <= 1000:
            conf = 0.6
        elif n <= 3000:
            conf = 0.5
        else:
            conf = 0.4

        u = t.upper()
        # Known task patterns in this repo
        if "FINISH[IMAGE REFUTES]" in u or "FINISH[IMAGE SUPPORTS]" in u:
            conf += 0.2
        if re.search(r"\bDECISION:\b", t, re.I) and re.search(r"\bREASON:\b", t, re.I):
            conf += 0.15
        if re.search(r"\bQuestion:\b", t) and re.search(r"\bAnswer:\b", t):
            conf += 0.10
        # JSON-like (for context_gen)
        if "{" in t and "}" in t:
            try:
                start = t.find("{"); end = t.rfind("}")
                if 0 <= start < end:
                    json.loads(t[start : end + 1])
                    conf += 0.15
            except Exception:
                pass
        return self._bound01(conf)

    def _infer_confidence(self, resp) -> float:
        """
        Unified confidence adapter across providers with graceful degradation.
        """
        # 1) Try OpenAI-style token logprobs first
        logs = self._extract_openai_token_logprobs(resp)
        if logs:
            # Use a fixed prefix of tokens to reduce tail noise
            return self._logprobs_to_conf(logs[:20])

        # 2) Gemini: probability or logs (if exposed)
        prob, glogs = self._extract_gemini_prob_or_logs(resp)
        if prob is not None:
            return self._bound01(prob)
        if glogs:
            return self._logprobs_to_conf(glogs[:20])

        # 3) Fallback: structure/format compliance
        return self._structure_conf_from_text(resp)





    # ------------------------------------------------------------------ #
    # Convenience multimodal wrapper
    # ------------------------------------------------------------------ #

    def call_multimodal(
        self,
        system_prompt: str,
        text_prompt: str,
        image_path: str,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper:
        ``create_multimodal_message → call`` in one line.
        """
        messages = self.create_multimodal_message(system_prompt, text_prompt, image_path)
        return self.call(messages)

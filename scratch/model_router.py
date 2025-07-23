# src/model_router.py
"""
Single, reusable ModelRouter for OpenAI or Gemini.
"""

from __future__ import annotations
import os
import time
import math
from typing import Tuple, Optional, Any, Dict

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


def encode_image(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Encode an image file to base64 and return (base64_string, mime_type).
    """
    try:
        import base64
        from PIL import Image
        from io import BytesIO

        if not os.path.isfile(image_path):
            return None, None

        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            return b64, "image/jpeg"
    except Exception:
        return None, None


class ModelRouter:
    """
    Thin wrapper around LangChain models (OpenAI GPT or Google Gemini).
    Provides:
        - automatic model selection by name
        - retry/back-off logic
        - multimodal helpers
    """

    def __init__(
        self,
        model_name: str,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: int = 5,
    ) -> None:
        self.model_name = model_name.lower()
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.google_api_key = google_api_key or os.getenv("GEMINI_API_KEY")
        self.temperature = temperature
        self.max_retries = max_retries
        self._llm = self._init_llm()

    def _init_llm(self) -> Any:
        if "gpt" in self.model_name:
            if not self.openai_api_key:
                raise ValueError("OpenAI key missing.")
            return ChatOpenAI(
                api_key=self.openai_api_key,
                model=self.model_name,
                temperature=self.temperature,
                logprobs=1,
                model_kwargs={"return_full_text": False},
            )
        elif "gemini" in self.model_name:
            if not self.google_api_key:
                raise ValueError("Google API key missing.")
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.google_api_key,
                temperature=self.temperature,
                candidate_count=1,
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    # ------------------------------------------------------------------ #
    # Public helpers                                                     #
    # ------------------------------------------------------------------ #
    def get_model(self) -> Any:
        return self._llm

    def switch_model(self, new_model_name: str) -> None:
        self.model_name = new_model_name.lower()
        self._llm = self._init_llm()

    def create_multimodal_message(
        self,
        system_prompt: str,
        text_prompt: str,
        image_path: str,
    ) -> Any:
        b64, mime = encode_image(image_path)
        if not b64:
            raise ValueError(f"Cannot encode image: {image_path}")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ]
            ),
        ]
        return messages

    def call_model(self, messages: Any) -> Dict[str, Any]:
        """
        Invoke the LLM with retries and return:
            {"raw": <LangChain message>, "confidence": <float 0-1>}
        """
        tries = 0
        while tries < self.max_retries:
            try:
                resp = self._llm.invoke(messages)

                # default fallback
                conf = 0.5

                # OpenAI logprobs
                llm_out = getattr(resp, "llm_output", {})
                if llm_out and "token_logprobs" in llm_out:
                    lp = llm_out["token_logprobs"]
                    if lp:
                        avg = sum(lp) / len(lp)
                        conf = float(min(max(math.exp(avg), 0.0), 1.0))

                # Gemini candidate probability
                elif hasattr(resp, "candidates") and resp.candidates:
                    cand = resp.candidates[0]
                    prob = getattr(cand, "probability", None)
                    if prob is not None:
                        conf = float(min(max(prob, 0.0), 1.0))

                return {"raw": resp, "confidence": conf}

            except Exception as e:
                msg = str(e).lower()
                if any(x in msg for x in ("rate limit", "429", "quota", "please try again")):
                    sleep_time = 2 ** tries
                    time.sleep(sleep_time)
                    tries += 1
                    continue
                raise e

        raise RuntimeError("Max retries exhausted")

    def llm_multimodal(
        self,
        system_prompt: str,
        text: str,
        image_path: str,
        **extra_llm_kwargs: Any,
    ) -> Dict[str, Any]:
        messages = self.create_multimodal_message(system_prompt, text, image_path)
        if extra_llm_kwargs:
            # fallback if model refuses kwargs
            try:
                return self.call_model(messages)
            except TypeError:
                return self.call_model(messages)
        return self.call_model(messages)
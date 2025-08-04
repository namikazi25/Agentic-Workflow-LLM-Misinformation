# scratch/step02_relevancy.py
"""
Step-02: Image–Headline Relevancy Checker only.
Depends on ModelRouter (assumed to live in src/ or will be imported later).
"""

from __future__ import annotations
import os
from typing import Dict, Any
from model_router import ModelRouter

# We will import ModelRouter from a single shared file later;
# for now we forward-declare so the code runs standalone.
try:
    from model_router import ModelRouter, encode_image
except ModuleNotFoundError:
    # allow direct import for quick testing
    from step01_dataloader import encode_image  # noqa: F401  (only for stub below)
    ModelRouter = None  # type: ignore


class ImageHeadlineRelevancyChecker:
    """
    Visual-veracity-only checker:
    “Does the news image itself contradict objective facts?”
    """

    def __init__(self, model_router: ModelRouter) -> None:
        self.model_router = model_router

        self.system_prompt = (
            "According to the given news image, determine if the news image goes "
            "against objective facts.  Follow the instructions exactly:\n\n"
            "Thought 1: Please describe the content in the news image that goes "
            "against the objective fact.\n"
            "Observation: [Fact-conflicting Description]\n"
            "Action 1: Draw the conclusion based on the observation:\n"
            "- If there is any credible objective fact refuting the news image, "
            "answer exactly: Finish[IMAGE REFUTES].\n"
            "- Otherwise, answer exactly: Finish[IMAGE SUPPORTS]."
        )

    def check_relevancy(
        self,
        image_path: str,
        headline: str,  # kept for signature compatibility; not used in prompt
    ) -> Dict[str, Any]:
        if not image_path or not os.path.isfile(image_path):
            return {"text": "Image File Not Found", "success": False}

        try:
            response = self.model_router.llm_multimodal(
                system_prompt=self.system_prompt,
                text="Analyze the image for factual accuracy.",
                image_path=image_path,
            )
            raw = response.get("raw") if response else None
            text = getattr(raw, "content", None) or str(raw or "")
            text = text.strip()

            if not text:
                raise RuntimeError("Empty response")

            return {"text": text, "success": True}

        except Exception as e:
            return {"text": f"API Error: {e}", "success": False}
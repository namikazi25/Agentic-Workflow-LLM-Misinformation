"""
scratch/relevancy.py
====================

Multimodal **image–headline relevancy checker** (pipeline *Step-02*).

Given a *news image* and its *headline* the LLM answers **exactly** one of:

    • `Finish[IMAGE REFUTES]`
    • `Finish[IMAGE SUPPORTS]`

plus a short explanation.

Public API
----------

>>> from scratch.model_router import ModelRouter
>>> from scratch.relevancy import ImageHeadlineRelevancyChecker
>>> checker = ImageHeadlineRelevancyChecker(ModelRouter())
>>> result = checker.check_relevancy("/path/img.jpg", "Headline text …")

Returns
-------

{
"text": "<raw LLM answer>",
"success": bool
}

pgsql
Copy
Edit

If Pillow fails to open the image or the file is missing, ``success`` is False.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

from .cache import memo
from .model_router import ModelRouter

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Class definition
# --------------------------------------------------------------------------- #


class ImageHeadlineRelevancyChecker:
    """
    Very lightweight wrapper – all heavy work is done by the LLM via
    `ModelRouter.call_multimodal`.
    """

    # Frozen system prompt (class constant so it’s not re-created)
    _PROMPT: str = (
        "According to the given news image, determine if the news image goes "
        "against objective facts.  Follow the instructions exactly:\n\n"
        "Thought 1: Please describe the content in the news image that goes "
        "against the objective fact.\n"
        "Observation: [Fact‐conflicting Description]\n"
        "Action 1: Draw the conclusion based on the observation:\n"
        "- If there is any credible objective fact refuting the news image, "
        "answer exactly: Finish[IMAGE REFUTES].\n"
        "- Otherwise, answer exactly: Finish[IMAGE SUPPORTS]."
    )

    def __init__(self, model_router: ModelRouter):
        self._mr = model_router

    # ------------------------------------------------------------------ #
    # Cached public method
    # ------------------------------------------------------------------ #

    @memo(maxsize=2_048)  # (image_path, headline) → result caching
    def check_relevancy(self, image_path: str, headline: str) -> Dict[str, Any]:
        """
        Parameters
        ----------
        image_path
            Absolute path of the image.
        headline
            Headline text (currently *not used* in the prompt but kept for
            future extensions and signature compatibility).

        Returns
        -------
        dict
            ``{"text": str, "success": bool}``
        """
        if not image_path or not os.path.isfile(image_path):
            logger.warning("Image file not found: %s", image_path)
            return {"text": "Image File Not Found", "success": False}

        try:
            response = self._mr.call_multimodal(
                system_prompt=self._PROMPT,
                text_prompt="Analyze the image for factual accuracy.",
                image_path=image_path,
            )
            raw = response.get("raw") if response else None
            text = getattr(raw, "content", None) or str(raw or "")
            text = text.strip()

            if not text:
                raise RuntimeError("Empty response from LLM")

            return {"text": text, "success": True}

        except Exception as exc:  # noqa: BLE001
            logger.error("Relevancy check failed: %s", exc, exc_info=False)
            return {"text": f"API Error: {exc}", "success": False}
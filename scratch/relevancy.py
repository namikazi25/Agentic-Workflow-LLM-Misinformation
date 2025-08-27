"""
scratch/relevancy.py
====================

Multimodal **image–headline relevancy checker** (pipeline *Step-02*).

Given a *news image* and its *headline* the LLM must decide whether the image
**supports or refutes the specific claim in that headline**, and answer
**exactly** one of:

    • `Finish[IMAGE REFUTES]`
    • `Finish[IMAGE SUPPORTS]`

plus a short justification.

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
        "You will be given a **news HEADLINE** and its **IMAGE**. "
        "Decide whether the IMAGE contradicts the **specific claim** made in the HEADLINE.\n\n"
        "Follow the instructions exactly:\n"
        "Thought 1: Briefly describe only the visible evidence in the image that is relevant to the HEADLINE's claim.\n"
        "Observation: [Concise visual evidence]\n"
        "Action 1: Based on the Observation and the HEADLINE's claim:\n"
        "- If the image contradicts/refutes the claim, answer exactly: Finish[IMAGE REFUTES].\n"
        "- Otherwise, answer exactly: Finish[IMAGE SUPPORTS].\n"
        "Rules: Use only what is visible in the image. Do not add outside knowledge. "
        "Judge against the HEADLINE as written."
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
            # Include the HEADLINE in the human message so the model can compare
            # the specific textual claim against the visual evidence.
            hp = (headline or "").strip()
            text_prompt = (
                f"HEADLINE:\n{hp}\n\n"
                "Task: Compare the IMAGE to the exact HEADLINE claim above and decide per the instructions. "
                "Respond with a brief justification and the required Finish[...] token."
            )

            response = self._mr.call_multimodal(
                system_prompt=self._PROMPT,
                text_prompt=text_prompt,
                image_path=image_path,
            )
            raw = response.get("raw") if response else None
            conf = float(response.get("confidence", 0.6)) if response else 0.6
            text = getattr(raw, "content", None) or str(raw or "")
            text = text.strip()

            if not text:
                raise RuntimeError("Empty response from LLM")

            return {"text": text, "success": True, "confidence": conf}

        except Exception as exc:  # noqa: BLE001
            logger.error("Relevancy check failed: %s", exc, exc_info=False)
            return {"text": f"API Error: {exc}", "success": False}
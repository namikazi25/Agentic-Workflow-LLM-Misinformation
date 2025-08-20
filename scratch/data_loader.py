"""
scratch/data_loader.py
======================

Fast, lightweight loader for **MMFakeBench** metadata that:

* Filters out samples whose *image file is missing*.
* Supports deterministic random sampling via ``config.LIMIT`` + ``config.SEED``.
* Exposes a PyTorch-style ``Dataset`` interface (but works without torch).

Returned tuple per sample
-------------------------
(
image_path, # str – absolute path
headline_text, # str
label_binary, # str – "True" | "Fake"
label_multiclass, # str – original fake_cls field
text_source, # str | None
image_source, # str | None
)

pgsql
Copy
Edit
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging

try:  # make torch an optional dependency for users who don't need it
    from torch.utils.data import Dataset
except ImportError:  # minimal fallback
    class Dataset:  # type: ignore
        def __len__(self): ...
        def __getitem__(self, idx): ...


from . import config as C

__all__ = ["MMFakeBenchDataset"]

logger = logging.getLogger(__name__)
def _load_metadata() -> List[Dict[str, Any]]:
    """Read JSON once; raises FileNotFoundError if missing."""
    with open(C.DATA_JSON, encoding="utf-8") as fp:
        return json.load(fp)


class MMFakeBenchDataset(Dataset):
    """
    Minimal dataset wrapper that keeps data in Python lists (no Tensor
    conversion – we do not train models here).
    """

    def __init__(
        self,
        limit: int | None = C.LIMIT,
        seed: int = C.SEED,
        data: List[Dict[str, Any]] | None = None,
    ):
        """
        Parameters
        ----------
        limit
            If not ``None``, dataset is randomly trimmed to *limit* samples
            after filtering missing images.
        seed
            RNG seed used for sampling (only relevant when *limit* is set).
        data
            Pre-supplied metadata list (mainly for unit-tests).  When ``None``,
            the file at ``config.DATA_JSON`` is loaded.
        """
        # 1) load + basic validation
        meta: List[Dict[str, Any]] = data if data is not None else _load_metadata()
        total = len(meta)

        # 2) resolve image paths + filter out missing
        base_dir = Path(C.IMAGES_DIR)
        items: List[Dict[str, Any]] = []
        missing = 0
        for entry in meta:
            img_rel = entry["image_path"].lstrip("/")
            img_abs = base_dir / img_rel
            if not img_abs.is_file():
                missing += 1
                continue  # drop
            entry["_abs_path"] = str(img_abs)
            items.append(entry)

        # 3) deterministic sub-sampling
        if limit is not None and len(items) > limit:
            random.seed(seed)
            random.shuffle(items)
            items = items[:limit]

        kept = len(items)
        logger.info(
            "Dataset scan: meta=%d, kept=%d, missing_images=%d, limit=%s, seed=%s, images_dir=%s",
            total, kept, missing, str(limit), str(seed), str(base_dir),
        )

        if not items:
            logger.error(
                "No samples available after filtering. Check paths and structure.\n"
                "  DATA_JSON=%s\n  IMAGES_DIR=%s\n  meta_entries=%d, missing_images=%d",
                C.DATA_JSON, C.IMAGES_DIR, total, missing
            )
            raise RuntimeError("Dataset initialisation resulted in 0 samples.")

        self._items = items

    # --------------------------------------------------------------------- #
    # Standard Dataset interface
    # --------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        e = self._items[idx]
        return (
            e["_abs_path"],
            e["text"],
            e["gt_answers"],   # "True" | "Fake"
            e["fake_cls"],     # multi-class label
            e.get("text_source"),
            e.get("image_source"),
        )

    # --------------------------------------------------------------------- #
    # Extra helpers
    # --------------------------------------------------------------------- #

    def stats(self) -> Dict[str, int]:
        """Return simple counts for sanity-checks."""
        from collections import Counter
        return Counter(e["gt_answers"] for e in self._items)  # type: ignore[arg-type]
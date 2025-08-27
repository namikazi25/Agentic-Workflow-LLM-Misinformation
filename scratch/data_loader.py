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
import re
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

# --------------------------------------------------------------------------- #
# Distortion family inference
# --------------------------------------------------------------------------- #
# Heuristic regex buckets (override by editing here if your schema differs)
_PAT_TEXTUAL = re.compile(r"(textual|text|caption|headline|claim|article|quote|statement|context)", re.I)
_PAT_VISUAL  = re.compile(r"(visual|image|photo|picture|manipulat|deepfake|ai[- ]?generated|photoshop|cgi)", re.I)
_PAT_CROSS   = re.compile(r"(cross[ -]?modal|crossmodal|mismatch|inconsisten|caption[- ]image|out[- ]of[- ]context)", re.I)

# Canonical normalization for common dataset strings
_FAMILY_CANON = {
    # textual
    "textualveracitydistortion": "textual",
    "textual_veracity_distortion": "textual",
    "textual veracity distortion": "textual",
    "textual": "textual",
    # visual
    "visualveracitydistortion": "visual",
    "visual_veracity_distortion": "visual",
    "visual veracity distortion": "visual",
    "visual": "visual",
    # cross-modal
    "crossmodalconsistencydistortion": "crossmodal",
    "cross_modal_consistency_distortion": "crossmodal",
    "cross-modal consistency distortion": "crossmodal",
    "cross modal consistency distortion": "crossmodal",
    "crossmodal": "crossmodal",
    "cross-modal": "crossmodal",
}

def _canon_family(s: str | None) -> str | None:
    if not s:
        return None
    t = re.sub(r"[\W_]+", " ", str(s).strip().lower())      # normalize hyphens/underscores
    key = re.sub(r"\s+", "", t)                              # remove spaces for matching keys above
    return _FAMILY_CANON.get(key)

def _infer_group(entry: Dict[str, Any]) -> str:
    """
    Map metadata to one of: 'textual' | 'visual' | 'crossmodal' | 'unknown'
    Uses common fields; customize if your JSON has a dedicated 'veracity_type'.
    """
    fields = [
        entry.get("veracity_type", ""),
        entry.get("distortion_type", ""),
        entry.get("attack_type", ""),
        entry.get("fake_cls", ""),
        entry.get("category", ""),
    ]
    # 1) Try canonical mapping first (most reliable)
    for f in fields:
        fam = _canon_family(f)
        if fam:
            return fam
    # 2) Fall back to broad regex matching on joint blob
    blob = " ".join(str(x) for x in fields).lower()
    if _PAT_TEXTUAL.search(blob):
        return "textual"
    if _PAT_VISUAL.search(blob):
        return "visual"
    if _PAT_CROSS.search(blob):
        return "crossmodal"
    return "unknown"


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
            # annotate distortion group
            entry["_group"] = _infer_group(entry)
            items.append(entry)

        # 3) slice by requested distortion mode (if any)
        mode = (C.DISTORTION_MODE or "any").lower()
        if mode not in {"any", "textual", "visual", "crossmodal"}:
            logger.warning("Unknown DISTORTION_MODE=%r – falling back to 'any'", mode)
            mode = "any"

        # Partition by label and (optionally) group
        def _is_true(e: Dict[str, Any]) -> bool:
            return str(e.get("gt_answers", "")).strip().lower() == "true"

        def _is_fake(e: Dict[str, Any]) -> bool:
            return str(e.get("gt_answers", "")).strip().lower() == "fake"

        if mode == "any":
            pos_all = [e for e in items if _is_fake(e)]
            neg_all = [e for e in items if _is_true(e)]
        else:
            pos_all = [e for e in items if _is_fake(e) and e.get("_group") == mode]
            if C.APPLY_MODE_TO_TRUE:
                neg_all = [e for e in items if _is_true(e) and e.get("_group") == mode]
            else:
                neg_all = [e for e in items if _is_true(e)]

        # 4) build a 50/50 balanced sample (deterministic)
        random.seed(seed)
        random.shuffle(pos_all)
        random.shuffle(neg_all)

        if limit is None:
            # Use the maximum perfectly balanced size available
            half = min(len(pos_all), len(neg_all))
            pos_keep = pos_all[:half]
            neg_keep = neg_all[:half]
        else:
            # Force 50/50 split
            half = limit // 2
            pos_keep = pos_all[:half]
            neg_keep = neg_all[: (limit - half)]
            # Strictness: if not enough on either side, either raise or shrink
            if (len(pos_keep) < half or len(neg_keep) < (limit - half)):
                # Build helpful diagnostics before raising
                # Count per label for the *requested* mode (or 'any')
                avail_fake = len(pos_all); avail_true = len(neg_all)
                diag = {
                    "requested_mode": mode,
                    "available": {"Fake": avail_fake, "True": avail_true},
                    "required": {"Fake": half, "True": (limit - half)},
                }
                msg = (
                    f"Insufficient samples for 50/50 split.\n"
                    f"  mode='{mode}'  LIMIT={limit}\n"
                    f"  available(Fake={avail_fake}, True={avail_true})  "
                    f"required(Fake={half}, True={limit - half})\n"
                    f"  Hint: lower --limit, choose a different --mode, or ensure your dataset "
                    f"labels include canonical family names for TRUE items (e.g., 'Textual Veracity Distortion')."
                )
                if C.STRICT_BALANCE:
                    logger.error(msg)
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg + " – shrinking to available balanced size.")
                    half2 = min(len(pos_all), len(neg_all))
                    pos_keep = pos_all[:half2]
                    neg_keep = neg_all[:half2]

        items = pos_keep + neg_keep
        random.shuffle(items)  # avoid label-order bias in downstream loops
        kept = len(items)
        logger.info(
            "Dataset scan: meta=%d, kept=%d, missing_images=%d, limit=%s, seed=%s, images_dir=%s, mode=%s, "
            "counts(Fake=%d, True=%d)",
            total, kept, missing, str(limit), str(seed), str(base_dir), mode,
            sum(1 for e in items if _is_fake(e)),
            sum(1 for e in items if _is_true(e)),
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

    def stats_by_group(self) -> Dict[str, Dict[str, int]]:
        """
        Return nested counts by distortion group and label, e.g.:
        {"textual": {"Fake": 12, "True": 12}, "visual": {...}, ...}
        """
        out: Dict[str, Dict[str, int]] = {}
        for e in self._items:
            g = e.get("_group", "unknown")
            y = e.get("gt_answers", "NA")
            out.setdefault(g, {}).setdefault(y, 0)
            out[g][y] += 1
        return out
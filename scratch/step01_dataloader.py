# scratch/step01_dataloader.py
"""
Step-01: MMFakeBench dataset loader + image encoder.
No other pipeline code included.
"""

from __future__ import annotations
import os
import json
import base64
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset
from io import BytesIO
from model_router import encode_image


class MMFakeBenchDataset(Dataset):
    """
    Lightweight wrapper around MMFakeBench JSON that **only keeps samples whose
    image file actually exists** on disk.
    Returns (image_path, text, label_binary, label_multiclass, text_source, image_source).
    """

    def __init__(
        self,
        json_path: str,
        images_base_dir: str,
        limit: Optional[int] = None,
        balanced: bool = False,
        seed: int = 42,
    ) -> None:
        self.items: List[Dict[str, Any]] = []

        # 1. Load metadata
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # 2. Filter by image existence
        for entry in data:
            img_path = os.path.join(images_base_dir, entry["image_path"].lstrip("/"))
            if not os.path.isfile(img_path):
                continue
            self.items.append(
                {
                    "image_path": img_path,
                    "text": entry["text"],
                    "label_binary": entry["gt_answers"],   # original string: "True" or "Fake"
                    "label_multiclass": entry["fake_cls"], # original string, e.g. "original"
                    "text_source": entry.get("text_source"),
                    "image_source": entry.get("image_source"),
                }
            )

        # 3. Optional balancing / trimming
        if limit is not None and limit < len(self.items):
            import random

            random.seed(seed)
            if balanced:
                buckets: Dict[str, List[Dict[str, Any]]] = {}
                for it in self.items:
                    buckets.setdefault(it["label_binary"], []).append(it)
                for lst in buckets.values():
                    random.shuffle(lst)
                half = limit // 2
                extra = limit % 2
                sampled = (
                    buckets.get("Fake", [])[: half + extra]
                    + buckets.get("True", [])[:half]
                )
                random.shuffle(sampled)
                self.items = sampled
            else:
                self.items = self.items[:limit]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        return (
            item["image_path"],
            item["text"],
            item["label_binary"],      # "True" or "Fake"
            item["label_multiclass"],  # e.g. "original", "mismatch", ...
            item["text_source"],
            item["image_source"],
        )


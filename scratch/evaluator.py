"""
scratch/evaluator.py
====================

Lightweight evaluation harness for the pipeline's final outputs.

Features
--------
* Accepts **lists**, **iterators**, or **JSONL files** – pick what fits RAM.
* Produces accuracy, confusion matrix (counts + %), classification report.
* Saves CSV artefacts when `save_prefix` is provided.

Canonical labels
----------------
Raw MMFakeBench labels    →  Canonical decision used here
------------------------------------------------------------
"True"                    →  "Not Misinformation"
"Fake"                    →  "Misinformation"
"""

from __future__ import annotations

import json
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from . import config as C

warnings.filterwarnings("ignore", category=RuntimeWarning)

__all__ = [
    "evaluate",
    "evaluate_list",
    "evaluate_stream",
    "evaluate_jsonl",
]

# --------------------------------------------------------------------------- #
# Canonical mapping
# --------------------------------------------------------------------------- #

_LABEL_MAP = {
    "Fake": "Misinformation",
    "True": "Not Misinformation",
}

_CANONICAL = ["Not Misinformation", "Misinformation", "Uncertain"]


def _canon(label: str) -> str:
    """Map raw label to canonical label (idempotent)."""
    return _LABEL_MAP.get(label, label)


# --------------------------------------------------------------------------- #
# Core evaluators
# --------------------------------------------------------------------------- #


def _metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, object]:
    acc = accuracy_score(y_true, y_pred)
    cm_counts = confusion_matrix(y_true, y_pred, labels=_CANONICAL)
    cm_perc = (
        cm_counts.astype("float") / cm_counts.sum(axis=1, keepdims=True) * 100
    ).round(1)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=_CANONICAL,
        target_names=_CANONICAL,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "confusion_counts": pd.DataFrame(
            cm_counts,
            index=[f"True {l}" for l in _CANONICAL],
            columns=[f"Pred {l}" for l in _CANONICAL],
        ),
        "confusion_percent": pd.DataFrame(
            cm_perc,
            index=[f"True {l}" for l in _CANONICAL],
            columns=[f"Pred {l}" for l in _CANONICAL],
        ),
        "classification_report": pd.DataFrame(report_dict).transpose(),
    }


# --------------------------------------------------------------------------- #
# Public wrappers
# --------------------------------------------------------------------------- #


def evaluate_list(
    predictions: List[str],
    ground_truth: List[str],
    *,
    save_prefix: str | None = None,
) -> Dict[str, object]:
    """
    Evaluate from two parallel lists (in-memory).

    Parameters
    ----------
    predictions
        Sequence of model decisions.
    ground_truth
        Raw dataset labels ("True"/"Fake").
    save_prefix
        If given, artefacts are saved as `<prefix>_*.csv`.
    """
    y_pred = predictions
    y_true = [_canon(label) for label in ground_truth]

    result = _metrics(y_true, y_pred)
    _save(result, save_prefix)
    _pretty_print(result)
    return result


def evaluate_stream(
    record_iter: Iterable[Dict[str, object]],
    *,
    pred_key: str = "decision",
    gt_key: str = "label_binary",
    save_prefix: str | None = None,
):
    """
    Evaluate streaming records (JSON objects) to avoid high memory use.

    Only extracts `pred_key` and `gt_key` from each dict.
    """
    y_pred, y_true = [], []
    for rec in record_iter:
        y_pred.append(rec[pred_key])
        y_true.append(_canon(rec[gt_key]))

    result = _metrics(y_true, y_pred)
    _save(result, save_prefix)
    _pretty_print(result)
    return result


def evaluate_jsonl(
    jsonl_path: str,
    *,
    pred_key: str = "decision",
    gt_key: str = "label_binary",
    save_prefix: str | None = None,
):
    """
    Load results from a JSON-lines file written by `main_async.py`
    and run evaluation.
    """
    with open(jsonl_path, encoding="utf-8") as fp:
        records = (json.loads(line) for line in fp)
        return evaluate_stream(records, pred_key=pred_key, gt_key=gt_key, save_prefix=save_prefix)


# Alias for backward compatibility
evaluate = evaluate_list

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _save(result: Dict[str, object], prefix: str | None):
    """Dump DataFrames to CSV if prefix given."""
    if not prefix:
        return

    out_dir = Path(prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    result["confusion_counts"].to_csv(f"{prefix}_cm_counts.csv")
    result["confusion_percent"].to_csv(f"{prefix}_cm_percent.csv")
    result["classification_report"].to_csv(f"{prefix}_report.csv")

    print(f"Evaluation artefacts saved under '{out_dir}'")


def _pretty_print(result: Dict[str, object]):
    print("\nAccuracy :", round(result["accuracy"], 4))
    print("\nConfusion matrix (counts):")
    print(result["confusion_counts"])
    print("\nConfusion matrix (%):")
    print(result["confusion_percent"])
    print("\nClassification report:")
    print(result["classification_report"])

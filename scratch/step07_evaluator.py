# scratch/step07_evaluator.py
"""
Step-07: Evaluation harness for the whole framework.

Usage:
    from step07_evaluator import evaluate_results
    evaluate_results(predictions, ground_truths, save_path="results.csv")
"""

from __future__ import annotations
from typing import List, Dict, Any
import json
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


LABEL_MAP = {
    "Fake": "Misinformation",
    "True": "Not Misinformation",
}


def _safe_label(label: str) -> str:
    """Convert raw MMFakeBench label to canonical decision."""
    return LABEL_MAP.get(label, label)


def evaluate_results(
    predictions: List[str],
    ground_truth_raw: List[str],
    labels_order: List[str] = None,
    save_path: str = None,
) -> Dict[str, Any]:
    """
    predictions  : list of strings from FinalClassifier
                   ['Misinformation', 'Not Misinformation', 'Uncertain']
    ground_truth : list of raw labels from dataset ('True', 'Fake')
    returns      : dict with metrics + optional CSV
    """
    labels_order = labels_order or ["Not Misinformation", "Misinformation", "Uncertain"]

    y_true = [_safe_label(gt) for gt in ground_truth_raw]
    y_pred = predictions

    acc = accuracy_score(y_true, y_pred)

    cm_counts = confusion_matrix(y_true, y_pred, labels=labels_order)
    cm_perc = (
        cm_counts.astype("float") / cm_counts.sum(axis=1, keepdims=True) * 100
    ).round(1)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels_order,
        target_names=labels_order,
        output_dict=True,
        zero_division=0,  # dict for easy CSV
    )

    # Pretty DataFrames
    cm_df = pd.DataFrame(
        cm_counts,
        index=[f"True {l}" for l in labels_order],
        columns=[f"Pred {l}" for l in labels_order],
    )
    perc_df = pd.DataFrame(
        cm_perc,
        index=[f"True {l}" for l in labels_order],
        columns=[f"Pred {l}" for l in labels_order],
    )

    result = {
        "accuracy": acc,
        "confusion_counts": cm_df,
        "confusion_percent": perc_df,
        "classification_report": pd.DataFrame(report).transpose(),
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cm_df.to_csv(save_path.replace(".csv", "_cm_counts.csv"))
        perc_df.to_csv(save_path.replace(".csv", "_cm_percent.csv"))
        pd.DataFrame(report).transpose().to_csv(
            save_path.replace(".csv", "_report.csv")
        )
        print(f" Evaluation artefacts saved to {os.path.dirname(save_path)}")

    print("\n Accuracy:", round(acc, 4))
    print("\nConfusion matrix (counts):")
    print(cm_df)
    print("\nConfusion matrix (%):")
    print(perc_df)
    print("\nClassification report:")
    print(pd.DataFrame(report).transpose())

    return result
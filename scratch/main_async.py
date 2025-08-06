"""
scratch/main_async.py
=====================

End-to-end **async** pipeline (Step-08) that ties everything together:

    • dataset loading
    • image–headline relevancy
    • optional event-report context
    • NUM_CHAINS × NUM_Q_PER_CHAIN concurrent Web-QA rounds
    • batch Q-A selection
    • final classification
    • JSONL streaming output + progress bar
    • optional evaluation summary at the end (when small LIMIT is set)

Requirements
------------
• All rewritten modules (`scratch.*`) must be importable.
• `log.setup()` is called once here → unified logging everywhere.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List

from tqdm.asyncio import tqdm

from . import config as C, log
from .data_loader import MMFakeBenchDataset
from .model_router import ModelRouter
from .relevancy import ImageHeadlineRelevancyChecker
from .context_gen import EventContextGenerator
from .qa_gen import QAGenerationTool
from .webqa import WebQAModule
from .qa_selector import batch_select
from .final_classifier import FinalClassifier
from .evaluator import evaluate_list

# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #

log.setup(level="INFO", file="run.log")
logger = logging.getLogger(__name__)

mr = ModelRouter(C.MODEL_DEFAULT, C.TEMPERATURE)
dataset = MMFakeBenchDataset()

Path(C.RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
jsonl_fp = open(C.RESULTS_PATH, "w", encoding="utf-8")

# storage for optional evaluation (only if LIMIT is manageable)
_y_pred: List[str] = []
_y_true: List[str] = []

# --------------------------------------------------------------------------- #
# Async helpers
# --------------------------------------------------------------------------- #


async def generate_branch(
    headline: str,
    event_report: Dict[str, object] | None,
) -> List[Dict[str, object]]:
    """
    Generate one *branch* (NUM_Q_PER_CHAIN sequential Q-A pairs).
    """
    branch: List[Dict[str, object]] = []

    for _ in range(C.NUM_Q_PER_CHAIN):
        q_tool = QAGenerationTool(
            headline,
            branch,
            event_report=event_report,
            strategy=C.QGEN_STRATEGY,
        )
        question, ok_q = q_tool.run(mr.get())
        if not ok_q:
            continue

        record = await WebQAModule(question, mr.get(), C.BRAVE_K).run()
        branch.append(record)

    return branch


async def process_sample(sample_idx: int, sample) -> Dict[str, object]:
    """
    Full processing of one dataset sample.
    """
    img_path, headline, label_bin, label_multi, *_ = sample

    # 1️⃣  Relevancy (blocking ~ but single LLM call)
    rel_checker = ImageHeadlineRelevancyChecker(mr)
    relevancy = rel_checker.check_relevancy(img_path, headline)["text"]

    # 2️⃣  Optional event report
    event_report = None
    if C.QGEN_STRATEGY in {"report", "auto"}:
        _, event_report = EventContextGenerator(
            headline,
            img_path,
            mr,
        ).run()

    # 3️⃣  Concurrent chain generation
    branches = await asyncio.gather(
        *[generate_branch(headline, event_report) for _ in range(C.NUM_CHAINS)]
    )

    # 4️⃣  Best Q-A per branch
    best_pairs = batch_select(branches, mr.get())

    # 5️⃣  Final classification
    decision, reason = FinalClassifier(
        headline, relevancy, best_pairs
    ).run(mr.get())

    record = {
        "sample_idx": sample_idx,
        "image_path": img_path,
        "headline": headline,
        "label_binary": label_bin,
        "label_multiclass": label_multi,
        "relevancy": relevancy,
        "best_qa_pairs": best_pairs,
        "decision": decision,
        "explanation": reason,
    }

    # evaluation collection (optional)
    _y_pred.append(decision)
    _y_true.append(label_bin)

    return record


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #


async def main():
    logger.info(
        "Pipeline start – %d samples, model=%s, chains=%d, q_per_chain=%d",
        len(dataset),
        C.MODEL_DEFAULT,
        C.NUM_CHAINS,
        C.NUM_Q_PER_CHAIN,
    )

    # Sequential sample loop; inside each sample we run chains concurrently.
    async for idx, sample in tqdm(
        enumerate(dataset),
        total=len(dataset),
        desc="Samples",
        unit="sample",
    ):
        record = await process_sample(idx, sample)
        jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        jsonl_fp.flush()  # safe in case of crash

    jsonl_fp.close()
    logger.info("Results written to %s", C.RESULTS_PATH)

    # Quick evaluation if dataset is small (so RAM fine)
    if len(_y_true) <= 200:
        logger.info("Running quick in-memory evaluation …")
        evaluate_list(_y_pred, _y_true)


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user – partial results kept at %s", C.RESULTS_PATH)
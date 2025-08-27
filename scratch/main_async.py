"""
scratch/main_async.py
=====================

End-to-end **async** pipeline (Step-08+) that ties everything together:

    • dataset loading
    • image–headline relevancy
    • optional event-report context
    • NUM_CHAINS × NUM_Q_PER_CHAIN concurrent Web-QA rounds
    • batch Q-A selection (temp=0.0)
    • final classification (temp=0.0)
    • JSONL streaming output + progress bar
    • optional evaluation summary at the end
    • per-sample runtime guard (never lose a run to one bad sample)

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

from dotenv import load_dotenv
from tqdm import tqdm  # plain tqdm is fine; outer loop is synchronous

# Load env once (OPENAI_API_KEY, GEMINI_API_KEY, BRAVE_SEARCH_API_KEY, etc.)
load_dotenv()

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
from datetime import datetime
from pathlib import Path
import logging
from . import config as C, log

# Build a unique log file per run
ts = datetime.now().strftime(C.LOG_TS_FMT)
log_dir = Path(C.LOG_DIR)
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / C.LOG_FILE_TEMPLATE.format(ts=ts)

# Init logging (console + file)
log.setup(level="INFO", file=str(log_path))
logger = logging.getLogger(__name__)

# Best-effort "latest" pointer
try:
    latest = log_dir / C.LOG_LATEST_NAME
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    # Try symlink; if it fails (e.g., Windows without privileges), fall back silently
    try:
        latest.symlink_to(log_path.name)  # relative symlink inside logs/
    except Exception:
        with open(log_dir / "LATEST_LOG.txt", "w", encoding="utf-8") as fp:
            fp.write(str(log_path.resolve()))
except Exception:
    pass  # never block startup on log niceties

logger.info("Logging to %s", log_path)

# Router is initialised inside `main()` to respect CLI/env overrides
mr = None  # will be assigned in main()

# Storage for optional evaluation (only if LIMIT is manageable)
_y_pred: list[str] = []
_y_true: list[str] = []
# --------------------------------------------------------------------------- #
# Async helpers
# --------------------------------------------------------------------------- #


async def generate_branch(
    headline: str,
    event_report: dict | None,
) -> list[dict]:
    """
    Generate one *branch* (NUM_Q_PER_CHAIN sequential Q-A pairs).
    """
    branch: list[dict] = []

    for _ in range(C.NUM_Q_PER_CHAIN):
        # Let QAGenerationTool prefer 'report' when summary exists, and fall back
        q_tool = QAGenerationTool(
            headline,
            branch,
            event_report=event_report,
            strategy=C.QGEN_STRATEGY,
        )

        question, ok_q = q_tool.run(mr.get())
        if not ok_q:
            continue

        # One WebQA round (uses router for confidence; passes headline+report)
        record = await WebQAModule(
            question,
            mr,
            C.BRAVE_K,
            headline=headline,
            event_report=event_report,
        ).run()
        branch.append(record)

    return branch


async def process_sample(sample_idx: int, sample) -> dict:
    """
    Full processing of one dataset sample.
    """
    img_path, headline, label_bin, label_multi, *_ = sample

    # 1️⃣  Relevancy (single LLM call)
    rel_checker = ImageHeadlineRelevancyChecker(mr)
    relevancy = rel_checker.check_relevancy(img_path, headline)["text"]

    # 2️⃣  Optional event report (context generator already retries for JSON)
    event_report = None
    if C.QGEN_STRATEGY in {"report", "auto"}:
        _, event_report = EventContextGenerator(
            headline,
            img_path,
            mr,
        ).run()

    # 3️⃣  Single-shot WebQA
    from .qa_gen import QAGenerationTool
    q_tool = QAGenerationTool(headline, [], event_report=event_report, strategy=C.QGEN_STRATEGY)
    question, ok_q = q_tool.run(mr.get())
    if not ok_q: raise RuntimeError("Question generation failed.")
    qa_record = await WebQAModule(question, mr, C.BRAVE_K, headline=headline, event_report=event_report).run()
    best_pairs = batch_select(branches, mr)

    # 4️⃣  Three heads
    from .heads.text_head import from_webqa as text_head
    from .heads.visual_head import run_visual_head
    from .heads.consistency_head import run_consistency_head
    text_feat = text_head(qa_record)
    visual_feat = run_visual_head(mr, img_path)
    cons_feat = run_consistency_head(rel_checker, img_path, headline)

    # 5️⃣  Fuser inference
    from .fuser_infer import decide as fuser_decide
    tmp_record = {
        "head_text": text_feat,
        "head_visual": visual_feat,
        "head_consistency": cons_feat,
    }
    decision, prob, reason = fuser_decide(tmp_record, tau=C.ABSTAIN_TAU)

    record = {
        "sample_idx": sample_idx,
        "image_path": img_path,
        "headline": headline,
        "label_binary": label_bin,
        "label_multiclass": label_multi,
        "relevancy": relevancy,            # keep raw for debugging
        "qa_record": qa_record,            # single WebQA turn
        "head_text": text_feat,
        "head_visual": visual_feat,
        "head_consistency": cons_feat,
        "fuser_prob": prob,
        "decision": decision,
        "explanation": reason,
    }

    # evaluation collection (only on success path)
    _y_pred.append(decision)
    _y_true.append(label_bin)

    return record


# --------------------------------------------------------------------------- #
# Main driver
# --------------------------------------------------------------------------- #


def _apply_overrides(*, data_json: str | None, images_dir: str | None, limit: int | None, seed: int | None, mode: str | None = None):
    """Mutate config with CLI overrides (if provided) and log the effective values."""
    changed = []
    if data_json is not None and data_json != C.DATA_JSON:
        C.DATA_JSON = data_json; changed.append(f"DATA_JSON={data_json}")
    if images_dir is not None and images_dir != C.IMAGES_DIR:
        C.IMAGES_DIR = images_dir; changed.append(f"IMAGES_DIR={images_dir}")
    if limit is not None and limit != C.LIMIT:
        C.LIMIT = limit; changed.append(f"LIMIT={limit}")
    if seed is not None and seed != C.SEED:
        C.SEED = seed; changed.append(f"SEED={seed}")
    if mode is not None and mode != C.DISTORTION_MODE:
        C.DISTORTION_MODE = mode; changed.append(f"DISTORTION_MODE={mode}")
    if changed:
        logger.info("Applied CLI overrides: %s", ", ".join(changed))
    logger.info("Dataset paths: DATA_JSON=%s  IMAGES_DIR=%s  LIMIT=%s  SEED=%s",
                C.DATA_JSON, C.IMAGES_DIR, str(C.LIMIT), str(C.SEED))


async def main(*, data_json: str | None = None, images_dir: str | None = None, limit: int | None = None, seed: int | None = None, mode: str | None = None):
    # Apply CLI overrides before creating router/dataset
    _apply_overrides(data_json=data_json, images_dir=images_dir, limit=limit, seed=seed, mode=mode)

    # Single router (shared across the run). Downstream callers can temporarily
    # switch temperature to 0.0 for deterministic prompts and restore afterwards.
    global mr
    mr = ModelRouter(C.MODEL_DEFAULT, C.TEMPERATURE)

    # Dataset (create inside main so overrides/env are respected)
    try:
        dataset = MMFakeBenchDataset(limit=C.LIMIT, seed=C.SEED)
    except FileNotFoundError as e:
        logger.error("Dataset metadata not found: %s (DATA_JSON=%s)", e, C.DATA_JSON)
        return
    except RuntimeError as e:
        logger.error("Dataset initialisation error: %s (DATA_JSON=%s, IMAGES_DIR=%s)", e, C.DATA_JSON, C.IMAGES_DIR)
        return

    # Output
    Path(C.RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Pipeline start – %d samples, model=%s, chains=%d, q_per_chain=%d",
        len(dataset),
        C.MODEL_DEFAULT,
        C.NUM_CHAINS,
        C.NUM_Q_PER_CHAIN,
    )

    # Open results file inside main for safer lifetime management
    with open(C.RESULTS_PATH, "w", encoding="utf-8") as jsonl_fp:
        # Sequential sample loop; inside each sample we run chains concurrently.
        for idx, sample in tqdm(
            enumerate(dataset),
            total=len(dataset),
            desc="Samples",
            unit="sample",
        ):
            try:
                record = await process_sample(idx, sample)
            except Exception as exc:  # ✅ runtime guard: never die on one bad sample
                logger.error("Failed to process sample %d: %s", idx, exc, exc_info=True)
                log.metric("sample_failure")

                # Try to extract basic fields for a useful stub
                try:
                    img_path, headline, label_bin, label_multi, *_ = sample
                except Exception:
                    img_path = headline = label_bin = label_multi = None

                record = {
                    "sample_idx": idx,
                    "image_path": img_path,
                    "headline": headline,
                    "label_binary": label_bin,
                    "label_multiclass": label_multi,
                    "relevancy": "N/A",
                    "best_qa_pairs": [],
                    "decision": "Uncertain",
                    "explanation": f"Processing error: {exc}",
                    "error": str(exc),
                }
                # Note: do **not** append to _y_pred/_y_true on failure

            jsonl_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl_fp.flush()  # keep progress even if we crash later

    logger.info("Results written to %s", C.RESULTS_PATH)

    # Quick evaluation if dataset is small (so RAM fine)
    if len(_y_true) <= 200 and _y_true:
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

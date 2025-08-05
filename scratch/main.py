import os
import sys
import time
import json
import logging
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------- Arguments & Configuration ----------
parser = argparse.ArgumentParser(
    description="MMFakeBench multimodal misinformation pipeline"
)
parser.add_argument(
    "--step", type=int, default=8,
    help="Pipeline step to run (1-8)"
)
parser.add_argument(
    "--limit", type=int, default=50,
    help="Max number of samples to process"
)
parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed for sampling"
)
parser.add_argument(
    "--num_chains", type=int, default=3,
    help="Number of reasoning chains per sample"
)
parser.add_argument(
    "--num_q_per_chain", type=int, default=3,
    help="Number of questions per chain"
)
parser.add_argument(
    "--strategy", choices=["headline","report","auto"],
    default="report",
    help="Question-generation strategy"
)
args = parser.parse_args()

# Paths (cross-platform)
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR.parent / "data"
JSON_PATH = DATA_DIR / "MMFakeBench_test.json"
IMAGES_BASE_DIR = DATA_DIR / "MMFakeBench_test-001" / "MMFakeBench_test"

# Pipeline config
STEP = args.step
LIMIT = args.limit
SEED = args.seed
NUM_CHAINS = args.num_chains
NUM_Q_PER_CHAIN = args.num_q_per_chain
QGEN_STRATEGY = args.strategy

# Ensure project modules are importable
sys.path.insert(0, str(BASE_DIR))

# ---------- Helpers ----------
def escape_curly(text: str) -> str:
    """
    Escape braces to avoid template parsing errors.
    """
    return text.replace("{", "{{").replace("}", "}}").strip()

# ---------- Step Implementations ----------

def run_step1():
    from step01_dataloader import MMFakeBenchDataset
    from model_router import encode_image

    ds = MMFakeBenchDataset(str(JSON_PATH), str(IMAGES_BASE_DIR), limit=LIMIT, seed=SEED)
    logger.info(f"Loaded {len(ds)} samples")
    for idx, (img, text, bin_lbl, multi_lbl, *_ ) in enumerate(ds):
        b64, mime = encode_image(img)
        logger.info(f"%d: bin=%s | multi=%s | text=%.60s... | b64_len=%d", idx, bin_lbl, multi_lbl, text, len(b64 or ""))


def run_step2():
    from step01_dataloader import MMFakeBenchDataset
    from step02_relevancy import ImageHeadlineRelevancyChecker
    from model_router import ModelRouter

    mr = ModelRouter("gemini-2.5-flash")
    checker = ImageHeadlineRelevancyChecker(mr)
    ds = MMFakeBenchDataset(str(JSON_PATH), str(IMAGES_BASE_DIR), limit=LIMIT, seed=SEED)
    for img, head, *rest in ds:
        res = checker.check_relevancy(img, head)
        logger.info("Relevancy: %s", res)


def run_step3():
    from step01_dataloader import MMFakeBenchDataset
    from step02_relevancy import ImageHeadlineRelevancyChecker
    from step03_qgen_no_enrich import QAGenerationToolNoEnrich
    from step04_webqa_stub import WebQAModuleStub as WebQAModule
    from model_router import ModelRouter

    mr = ModelRouter("gemini-2.5-flash")
    ds = MMFakeBenchDataset(str(JSON_PATH), str(IMAGES_BASE_DIR), limit=LIMIT, seed=SEED)
    for idx, (img, head, bin_lbl, multi_lbl, *_ ) in enumerate(ds):
        logger.info("Sample %d/%d: %s", idx+1, len(ds), img)
        rel = ImageHeadlineRelevancyChecker(mr).check_relevancy(img, head)
        qa_global = []
        for c in range(NUM_CHAINS):
            branch = []
            for q in range(NUM_Q_PER_CHAIN):
                q_tool = QAGenerationToolNoEnrich(head, branch + qa_global)
                question, ok = q_tool.run(mr.get_model())
                ans = "Q-Gen failed" if not ok else WebQAModule(question, mr.get_model()).run()[0]
                branch.append({"question": question, "answer": ans})
                time.sleep(1)
            qa_global.extend(branch)
            logger.info("Chain %d made %d Q&A", c+1, len(branch))
        record = {"idx": idx, "img": img, "bin": bin_lbl, "multi": multi_lbl, "head": head, "rel": rel, "qa": qa_global}
        print(json.dumps(record, ensure_ascii=False, indent=2))


def run_step8():
    from step01_dataloader import MMFakeBenchDataset
    from step02_relevancy import ImageHeadlineRelevancyChecker
    from step03a_event_context import EventContextGenerator
    from step03_qgen_no_enrich import QAGenerationTool
    from step04_webqa_stub import WebQAModuleStub as WebQAModule
    from step05_qa_selector import QASelector
    from step06_final_classifier import FinalClassifier
    from brave_search import brave_search_answer
    from langchain_core.prompts import ChatPromptTemplate
    from model_router import ModelRouter

    mr = ModelRouter("gemini-2.5-flash")
    ds = MMFakeBenchDataset(str(JSON_PATH), str(IMAGES_BASE_DIR), limit=LIMIT, seed=SEED)

    results = []
    for idx, (img, head, bin_lbl, multi_lbl, *_ ) in enumerate(ds):
        logger.info("Processing %d/%d: %s", idx+1, len(ds), img)
        rel = ImageHeadlineRelevancyChecker(mr).check_relevancy(img, head)
        report = None
        if QGEN_STRATEGY in {"report","auto"}:
            raw, report = EventContextGenerator(head, img, mr).run()
            logger.info("Event summary: %s", report.get("summary", raw))

        best_pairs = []
        for c in range(NUM_CHAINS):
            branch = []
            for q in range(NUM_Q_PER_CHAIN):
                q_tool = QAGenerationTool(headline=head, previous_qa=branch, event_report=report or {}, strategy=QGEN_STRATEGY)
                question, ok = q_tool.run(mr.get_model())
                if not ok:
                    branch.append({"question": question, "answer": "Q-Gen failed"})
                    continue
                snippets = brave_search_answer(question, k=8)
                evidence = "\n".join(f"- {s}" for s in snippets) or "No relevant results."
                evidence = escape_curly(evidence)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"Using ONLY the evidence below...\nEvidence:\n{evidence}"),
                    ("human", question),
                ])
                try:
                    resp = (prompt | mr.get_model()).invoke({})
                    answer = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
                except Exception as e:
                    answer = f"Error: {e}"
                branch.append({"question": question, "answer": answer})
                time.sleep(1)
            selected, ok_sel = QASelector(branch).run(mr.get_model())
            best_pairs.append(selected or branch[0])

        decision, reason = FinalClassifier(head, rel["text"], best_pairs).run(mr.get_model())
        record = {"img": img, "bin": bin_lbl, "multi": multi_lbl, "head": head, "rel": rel["text"], "qa": best_pairs, "decision": decision, "reason": reason}
        results.append(record)
        print(json.dumps(record, ensure_ascii=False, indent=2))

    preds = [r["decision"] for r in results]
    gts = [r["bin"] for r in results]
    from step07_evaluator import evaluate_results
    evaluate_results(predictions=preds, ground_truth_raw=gts, save_path="results/step08_eval.csv")

# Map steps to functions
STEP_FUNCS = {1: run_step1, 2: run_step2, 3: run_step3, 8: run_step8}

if __name__ == "__main__":
    if STEP not in STEP_FUNCS:
        logger.error("Unknown STEP %d", STEP)
        sys.exit(1)
    logger.info("Starting STEP %d", STEP)
    STEP_FUNCS[STEP]()

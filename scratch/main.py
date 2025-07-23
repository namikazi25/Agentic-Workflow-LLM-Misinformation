# main.py
"""
Single-entry script for the whole MMFakeBench rewrite.
We move one step at a time by flipping the STEP constant.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# ---------- CONFIGURATION SECTION (edit here) ----------
STEP = 8            # 1,2,3,... – only change this line
JSON_PATH       = r"..\data\MMFakeBench_test.json"
IMAGES_BASE_DIR = r"..\data\MMFakeBench_test-001\MMFakeBench_test"
LIMIT           = 10        # small set while iterating
SEED            = 42
NUM_CHAINS      = 3        # 1..N
NUM_Q_PER_CHAIN = 3        # 1..N
# -------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent))

# ---------- STEP-SPECIFIC IMPORTS ----------
if STEP == 1:
    from step01_dataloader import MMFakeBenchDataset
    from model_router import encode_image
elif STEP == 2:
    from step01_dataloader import MMFakeBenchDataset, encode_image
    from step02_relevancy import ImageHeadlineRelevancyChecker
    from model_router import ModelRouter
elif STEP == 3:
    from step01_dataloader import MMFakeBenchDataset
    from step02_relevancy import ImageHeadlineRelevancyChecker
    from step03_qgen_no_enrich import QAGenerationToolNoEnrich
    from step04_webqa_stub import WebQAModuleStub as WebQAModule
    from model_router import ModelRouter
elif STEP == 4:
    from step01_dataloader import MMFakeBenchDataset
    from step02_relevancy import ImageHeadlineRelevancyChecker
    from step03_qgen_no_enrich import QAGenerationToolNoEnrich
    from step04_webqa_stub import WebQAModuleStub as WebQAModule
    from model_router import ModelRouter
elif STEP == 5:
    from step01_dataloader import MMFakeBenchDataset
    from step02_relevancy import ImageHeadlineRelevancyChecker
    from step03_qgen_no_enrich import QAGenerationToolNoEnrich
    from step04_webqa_stub import WebQAModuleStub as WebQAModule
    from model_router import ModelRouter
    from step05_qa_selector import QASelector
elif STEP == 6:
    from step01_dataloader import MMFakeBenchDataset
    from step02_relevancy import ImageHeadlineRelevancyChecker
    from step03_qgen_no_enrich import QAGenerationToolNoEnrich
    from step04_webqa_stub import WebQAModuleStub as WebQAModule
    from model_router import ModelRouter
    from step05_qa_selector import QASelector
elif STEP == 8:
    from step01_dataloader import MMFakeBenchDataset
    from step02_relevancy import ImageHeadlineRelevancyChecker
    from step03_qgen_no_enrich import QAGenerationToolNoEnrich
    from step04_webqa_stub import WebQAModuleStub as WebQAModule
    from model_router import ModelRouter
    from step05_qa_selector import QASelector
# ... add more elif branches as we proceed
else:
    raise ValueError(f"Unknown STEP {STEP}")

# ---------- STEP-SPECIFIC MAIN ----------
def run_step1():
    """Loader + encoder only."""
    dataset = MMFakeBenchDataset(JSON_PATH, IMAGES_BASE_DIR, LIMIT, seed=SEED)
    print(f"Loaded {len(dataset)} samples")
    for idx in range(len(dataset)):
        img_path, text, label_bin, label_multi, *_ = dataset[idx]
        b64, mime = encode_image(img_path)
        print(f"{idx}: bin={label_bin} | multi={label_multi} | {text[:60]}... | base64={len(b64 or '')}")

def run_step2():
    """Relevancy checker only."""
    mr = ModelRouter("gemini-2.5-flash")  # or any model you have keys for
    checker = ImageHeadlineRelevancyChecker(mr)
    dataset = MMFakeBenchDataset(JSON_PATH, IMAGES_BASE_DIR, LIMIT, seed=SEED)
    for img_path, text, *_ in dataset:
        res = checker.check_relevancy(img_path, text)
        print("RELEVANCY:", res)

def run_step3():
    """Full configurable mini-pipeline (no claim enrichment)."""
    mr = ModelRouter("gemini-2.5-flash")
    dataset = MMFakeBenchDataset(JSON_PATH, IMAGES_BASE_DIR, LIMIT, seed=SEED)

    for idx, (img_path, headline, label_bin, label_multi, *_rest) in enumerate(dataset):
        print(f"\n SAMPLE {idx+1}/{len(dataset)}")
        print(f"  Image : {img_path}")
        print(f"  Label : {label_bin} / {label_multi}")
        print(f"  Head  : {headline}")

        # 1) Image-headline relevancy
        checker = ImageHeadlineRelevancyChecker(mr)
        rel = checker.check_relevancy(img_path, headline)

        # 2) Iterative QA
        global_qa: List[Dict[str, Any]] = []
        for chain in range(NUM_CHAINS):
            qa_branch: List[Dict[str, Any]] = []
            for q_num in range(NUM_Q_PER_CHAIN):
                q_tool = QAGenerationToolNoEnrich(headline, qa_branch + global_qa)
                question, ok_q = q_tool.run(mr.get_model())
                if not ok_q:
                    qa_branch.append({"question": question, "answer": "Q-Gen failed"})
                    continue

                web = WebQAModule(question, mr.get_model())
                answer, ok_a = web.run()
                qa_branch.append({"question": question, "answer": answer})
                time.sleep(1)  # crude rate-limit

            global_qa.extend(qa_branch)
            print(f"  Chain {chain+1}: {len(qa_branch)} Q-A pairs")

        # 3) Bundle ready for classifier
        record = {
            "sample_idx": idx,
            "image_path": img_path,
            "label_binary": label_bin,
            "label_multiclass": label_multi,
            "headline": headline,
            "relevancy": rel,
            "qa_pairs": global_qa,
        }
        print(json.dumps(record, indent=2, ensure_ascii=False))

def run_step4():
    """
    STEP-4 FULL-TRACE RUN
    * one sample
    * N chains (configurable at top of file)
    * M questions per chain (configurable)
    * prints every Q + A live
    * prints best pair from each chain
    * prints final aggregated context
    """
    from step05_qa_selector import QASelector

    mr = ModelRouter("gemini-2.5-flash")
    dataset = MMFakeBenchDataset(JSON_PATH, IMAGES_BASE_DIR, 1, seed=SEED)
    img_path, headline, label_bin, label_multi, *_rest = dataset[0]

    print(f"\n SAMPLE\nImage : {img_path}\nLabel : {label_bin} / {label_multi}\nHead  : {headline}\n")

    # ---- 1) Relevancy check ----
    checker = ImageHeadlineRelevancyChecker(mr)
    rel = checker.check_relevancy(img_path, headline)
    print(" Relevancy result:", rel["text"], "\n")

    # ---- 2) Iterative Q-A per chain ----
    final_context: list[dict[str, Any]] = []   # best pair from each chain

    for chain_idx in range(NUM_CHAINS):
        print(f"----- CHAIN {chain_idx + 1} / {NUM_CHAINS} -----")

        branch_qa: list[dict[str, Any]] = []

        # ---- 2a) generate & answer M questions ----
        for q_num in range(NUM_Q_PER_CHAIN):
            # generate
            q_tool = QAGenerationToolNoEnrich(headline, branch_qa)
            question, ok_q = q_tool.run(mr.get_model())
            if not ok_q:
                print(f"   Q{q_num+1}: generation failed {question}")
                branch_qa.append({"question": question, "answer": "Q-Gen failed"})
                continue
            print(f"  Q{q_num+1}: {question}")

            # answer
            web = WebQAModule(question, mr.get_model())
            answer, ok_a = web.run()
            if not ok_a:
                print(f"   A{q_num+1}: answering failed → {answer}")
            else:
                print(f"   A{q_num+1}: {answer}")
            branch_qa.append({"question": question, "answer": answer})
            time.sleep(1)   # crude rate-limit

        # ---- 2b) select best pair from this chain ----
        selector = QASelector(branch_qa)
        best_pair, ok_sel = selector.run(mr.get_model())

        if ok_sel and best_pair:
            print(f"   Best pair selected:\n     Q: {best_pair['question']}\n     A: {best_pair['answer']}\n")
            final_context.append(best_pair)
        else:
            print("    Selector failed – using first pair")
            final_context.append(branch_qa[0])

    # ---- 3) Final aggregated context ----
    print("=" * 60)
    print(" FINAL CONTEXT (best Q-A from each chain)")
    print(json.dumps(final_context, indent=2, ensure_ascii=False))


def run_step5():
    """
    STEP-5 BATCH FULL-TRACE RUN
    * processes LIMIT samples in sequence
    * N chains per sample (configurable at top of file)
    * M questions per chain (configurable)
    * prints every Q + A live for each sample
    * prints best pair from each chain
    * prints final aggregated context for each sample
    """

    mr = ModelRouter("gemini-2.5-flash")
    dataset = MMFakeBenchDataset(JSON_PATH, IMAGES_BASE_DIR, LIMIT, seed=SEED)

    for sample_idx, (img_path, headline, label_bin, label_multi, *_rest) in enumerate(dataset):
        print("\n" + "="*80)
        print(f" SAMPLE {sample_idx + 1}/{len(dataset)}")
        print(f"Image : {img_path}\nLabel : {label_bin} / {label_multi}\nHead  : {headline}\n")

        # 1) Relevancy check
        checker = ImageHeadlineRelevancyChecker(mr)
        rel = checker.check_relevancy(img_path, headline)
        print(" Relevancy result:", rel["text"], "\n")

        # 2) Iterative Q-A per chain
        final_context: list[dict[str, Any]] = []

        for chain_idx in range(NUM_CHAINS):
            print(f"----- CHAIN {chain_idx + 1} / {NUM_CHAINS} -----")
            branch_qa: list[dict[str, Any]] = []

            for q_num in range(NUM_Q_PER_CHAIN):
                # generate question
                q_tool = QAGenerationToolNoEnrich(headline, branch_qa)
                question, ok_q = q_tool.run(mr.get_model())
                if not ok_q:
                    print(f"   Q{q_num+1}: generation failed {question}")
                    branch_qa.append({"question": question, "answer": "Q-Gen failed"})
                    continue
                print(f"  Q{q_num+1}: {question}")

                # answer question
                web = WebQAModule(question, mr.get_model())
                answer, ok_a = web.run()
                if not ok_a:
                    print(f"   A{q_num+1}: answering failed → {answer}")
                else:
                    print(f"   A{q_num+1}: {answer}")
                branch_qa.append({"question": question, "answer": answer})
                time.sleep(1)

            # select best Q-A pair
            selector = QASelector(branch_qa)
            best_pair, ok_sel = selector.run(mr.get_model())

            if ok_sel and best_pair:
                print(f"   Best pair selected:\n     Q: {best_pair['question']}\n     A: {best_pair['answer']}\n")
                final_context.append(best_pair)
            else:
                print("    Selector failed – using first pair")
                final_context.append(branch_qa[0])

        # Final aggregated context
        print("=" * 60)
        print(" FINAL CONTEXT (best Q-A from each chain)")
        print(json.dumps(final_context, indent=2, ensure_ascii=False))

def run_step6():
    """Full pipeline with final classification for all samples (batch)."""
    from step05_qa_selector import QASelector
    from step06_final_classifier import FinalClassifier

    mr = ModelRouter("gemini-2.5-flash")
    dataset = MMFakeBenchDataset(JSON_PATH, IMAGES_BASE_DIR, LIMIT, seed=SEED)

    results_list = []
    for sample_idx, (img_path, headline, label_bin, label_multi, *_rest) in enumerate(dataset):
        print(f"\nSAMPLE {sample_idx + 1}/{len(dataset)}\nHeadline: {headline}\n")

        # 1) Image–headline relevancy
        checker = ImageHeadlineRelevancyChecker(mr)
        rel = checker.check_relevancy(img_path, headline)
        print("Relevancy:", rel["text"], "\n")

        # 2) Per-chain Q-A + selection
        best_qa_pairs: list[dict[str, Any]] = []
        for chain_idx in range(NUM_CHAINS):
            qa_branch: list[dict[str, Any]] = []
            for q_num in range(NUM_Q_PER_CHAIN):
                q_tool = QAGenerationToolNoEnrich(headline, qa_branch)
                question, ok_q = q_tool.run(mr.get_model())
                if not ok_q:
                    qa_branch.append({"question": question, "answer": "Q-Gen failed"})
                    continue

                web = WebQAModule(question, mr.get_model())
                answer, ok_a = web.run()
                qa_branch.append({"question": question, "answer": answer})

            selector = QASelector(qa_branch)
            best, ok_sel = selector.run(mr.get_model())
            if ok_sel and best:
                best_qa_pairs.append(best)
            else:
                best_qa_pairs.append(qa_branch[0])

        print("Best Q-A per chain:")
        print(json.dumps(best_qa_pairs, indent=2, ensure_ascii=False))

        # 3) Final classification
        classifier = FinalClassifier(headline, rel["text"], best_qa_pairs)
        decision, reason = classifier.run(mr.get_model())

        print("\nFinal Decision")
        print("DECISION:", decision)
        print("REASON  :", reason)

        # 4) Bundle everything
        record = {
            "image_path": img_path,
            "label_binary": label_bin,
            "label_multiclass": label_multi,
            "headline": headline,
            "relevancy": rel["text"],
            "best_qa_pairs": best_qa_pairs,
            "final_decision": decision,
            "explanation": reason,
        }
        results_list.append(record)
        print("\nFull record")
        print(json.dumps(record, indent=2, ensure_ascii=False))

    predictions = [rec["final_decision"] for rec in results_list]
    ground_truth = [rec["label_binary"] for rec in results_list]  # True / Fake

    from step07_evaluator import evaluate_results
    evaluate_results(
        predictions=predictions,
        ground_truth_raw=ground_truth,
        save_path="results/step04_eval.csv",
    )

def run_step8():
    """Full pipeline using Brave Search for evidence, then LLM answers, batch mode."""
    from step05_qa_selector import QASelector
    from step06_final_classifier import FinalClassifier
    from brave_search import brave_search_answer

    mr = ModelRouter("gemini-2.5-flash")
    dataset = MMFakeBenchDataset(JSON_PATH, IMAGES_BASE_DIR, LIMIT, seed=SEED)

    results_list = []
    for sample_idx, (img_path, headline, label_bin, label_multi, *_rest) in enumerate(dataset):
        print(f"\nSAMPLE {sample_idx + 1}/{len(dataset)}\nHeadline: {headline}\n")

        # 1) Image–headline relevancy
        checker = ImageHeadlineRelevancyChecker(mr)
        rel = checker.check_relevancy(img_path, headline)
        print("Relevancy:", rel["text"], "\n")

        # 2) Per-chain Q-A + selection using Brave
        best_qa_pairs: list[dict[str, Any]] = []
        for chain_idx in range(NUM_CHAINS):
            qa_branch: list[dict[str, Any]] = []
            for q_num in range(NUM_Q_PER_CHAIN):
                q_tool = QAGenerationToolNoEnrich(headline, qa_branch)
                question, ok_q = q_tool.run(mr.get_model())
                if not ok_q:
                    qa_branch.append({"question": question, "answer": "Q-Gen failed"})
                    continue

                # --- Brave Search ---
                snippets = brave_search_answer(question, k=8)
                evidence = "\n".join(f"- {s}" for s in snippets) if snippets else "No relevant results found."

                # --- LLM answers grounded on Brave ---
                from langchain_core.prompts import ChatPromptTemplate
                prompt = ChatPromptTemplate.from_messages([
                    ("system", 
                     "Given these web search snippets, answer the user's question factually using only the provided evidence. "
                     "If the evidence does not contain the answer, say 'No relevant answer found.'\n"
                     f"Evidence:\n{evidence}"),
                    ("human", question)
                ])
                try:
                    chain = prompt | mr.get_model()
                    response = chain.invoke({})
                    answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
                    ok_a = bool(answer)
                except Exception as e:
                    answer = f"Brave-LLM Error: {e}"
                    ok_a = False

                qa_branch.append({"question": question, "answer": answer})

            selector = QASelector(qa_branch)
            best, ok_sel = selector.run(mr.get_model())
            if ok_sel and best:
                best_qa_pairs.append(best)
            else:
                best_qa_pairs.append(qa_branch[0])

        print("Best Q-A per chain:")
        print(json.dumps(best_qa_pairs, indent=2, ensure_ascii=False))

        # 3) Final classification
        classifier = FinalClassifier(headline, rel["text"], best_qa_pairs)
        decision, reason = classifier.run(mr.get_model())

        print("\nFinal Decision")
        print("DECISION:", decision)
        print("REASON  :", reason)

        # 4) Bundle everything
        record = {
            "image_path": img_path,
            "label_binary": label_bin,
            "label_multiclass": label_multi,
            "headline": headline,
            "relevancy": rel["text"],
            "best_qa_pairs": best_qa_pairs,
            "final_decision": decision,
            "explanation": reason,
        }
        results_list.append(record)
        print("\nFull record")
        print(json.dumps(record, indent=2, ensure_ascii=False))

    predictions = [rec["final_decision"] for rec in results_list]
    ground_truth = [rec["label_binary"] for rec in results_list]  # True / Fake

    from step07_evaluator import evaluate_results
    evaluate_results(
        predictions=predictions,
        ground_truth_raw=ground_truth,
        save_path="results/step08_eval.csv",
    )


# ---------- DISPATCH ----------
if __name__ == "__main__":
    print(f"======== STEP {STEP} ========")
    globals()[f"run_step{STEP}"]()
"""
scratch/config.py
=================

Config with sensible defaults; certain fields can be overridden via
environment variables (see below) or via CLI flags handled in main_async.py.

Env overrides (optional):
  - DATA_JSON      → path to metadata json (default: data/MMFakeBench_test.json)
  - IMAGES_DIR     → root dir for images     (default: data/MMFakeBench_test-001/MMFakeBench_test)
  - DATA_LIMIT     → int or "None"           (default: 10)
  - DATA_SEED      → int                     (default: 71)
"""

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

import os

def _env_int(name: str, default):
    val = os.getenv(name)
    if val is None:
        return default
    try:
        if isinstance(default, type(None)) and val.lower() == "none":
            return None
        return int(val)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

DATA_JSON: str = os.getenv("DATA_JSON", "data/MMFakeBench_test.json")
IMAGES_DIR: str = os.getenv("IMAGES_DIR", "data/MMFakeBench_test-001/MMFakeBench_test")
LIMIT: int | None = _env_int("DATA_LIMIT", 10)
SEED: int = _env_int("DATA_SEED", 71)

# --------------------------------------------------------------------------- #
# Distortion-family sampling (dataset slice control)
# --------------------------------------------------------------------------- #
# Choose ONE of: "any", "textual", "visual", "crossmodal"
DISTORTION_MODE: str = os.getenv("DISTORTION_MODE", "any").lower()
# Apply the same mode to TRUE (not-misinformation) items too (strict slice)
APPLY_MODE_TO_TRUE: bool = _env_bool("APPLY_MODE_TO_TRUE", True)
# Enforce exact 50/50 balance; if not enough items exist, raise
STRICT_BALANCE: bool = _env_bool("STRICT_BALANCE", True)


# --------------------------------------------------------------------------- #
# LLM / model routing
# --------------------------------------------------------------------------- #

MODEL_DEFAULT: str = "gemini-2.5-flash"
TEMPERATURE: float = 0.2

# --------------------------------------------------------------------------- #
# Pipeline hyper-parameters
# --------------------------------------------------------------------------- #

NUM_CHAINS: int = 3
NUM_Q_PER_CHAIN: int = 3
QGEN_STRATEGY: str = "auto"          # headline | report | auto

# --------------------------------------------------------------------------- #
# Brave search
# --------------------------------------------------------------------------- #

BRAVE_K: int = 15

# Prefer fresher results when claims are time-bound.
#  • None or 0  → no recency filter
#  • e.g., 365  → ~past year
FRESHNESS_DAYS: int | None = 365

# Adaptive freshness policy:
#  - "auto": use FRESHNESS_DAYS unless we detect an older target year (then disable)
#  - "off" : never apply freshness
#  - "force": always apply freshness regardless of target year
FRESHNESS_POLICY: str = "auto"    

# If event year is older than (current_year - OLD_EVENT_YEARS), disable freshness
OLD_EVENT_YEARS: int = 3          # (treat ≥3 years as “historical”)

# Small bonus to snippet score if it mentions the inferred target year
TEMPORAL_MATCH_BONUS: int = 2     


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #

CACHE_TTL_SEC: int = 60 * 60           # 1 hour

# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

RESULTS_PATH: str = "results/step08_async.jsonl"

# --------------------------------------------------------------------------- #
# Quality gates for WebQA  
# --------------------------------------------------------------------------- #

# Minimum number of non-spam snippets required for a question to be considered usable
MIN_SNIPPETS: int = 3

# Minimum token-overlap between the question and at least one snippet
# (Step #1 uses question-only overlap; Step #2 will add headline-aware overlap)
MIN_OVERLAP: int = 1

# Minimum answer length to avoid fluffy one-liners
MIN_ANSWER_CHARS: int = 35

MIN_CONF: float = 0.30         
# When average overlap of GOOD Q-A is below this, treat text evidence as weak
# and let image relevancy decide (if REFUTES/SUPPORTS present).
EVIDENCE_STRENGTH_MIN: int = 2   

# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #

DEBUG: bool = False


# --------------------------------------------------------------------------- #
# Ablation toggles
# --------------------------------------------------------------------------- #

# If True, when text evidence is weak/absent we allow image REFUTES/SUPPORTS
# to short-circuit the decision. If False, we won't fall back to the image.
USE_IMAGE_FALLBACK: bool = True  # ✅ NEW (requested)

# --------------------------------------------------------------------------- #
# Q/A generation – de‑duplication controls
# --------------------------------------------------------------------------- #
# Enable fuzzy (near‑duplicate) detection in addition to exact match checks.
QA_DEDUP_FUZZY: bool = True
# "trigram" (character trigrams) or "token" (word tokens) for Jaccard.
QA_DEDUP_SIM_MODE: str = "trigram"
# Jaccard similarity threshold to treat as duplicate (0..1).
QA_DEDUP_THRESHOLD: float = 0.82
# For trigram mode, the n in n‑grams.
QA_DEDUP_NGRAM: int = 3


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_DIR: str = "logs"
LOG_FILE_TEMPLATE: str = "run_{ts}.log"   # {ts} will be filled with timestamp
LOG_TS_FMT: str = "%Y%m%d-%H%M%S"
LOG_WRITE_LATEST_SYMLINK: bool = True     # best effort; falls back quietly on Windows
LOG_LATEST_NAME: str = "latest.log"
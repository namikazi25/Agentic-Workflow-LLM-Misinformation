"""
scratch/config.py
=================

Centralised configuration for the agentic-workflow rewrite.

All other modules import *only* from this file, so you can tweak
parameters (model, dataset size, chain counts, …) in one place.

Environment variables (API keys, etc.) are still read inside the
respective helper modules; here we keep pipeline-level knobs.
"""

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

# Path to the MMFakeBench JSON metadata file
DATA_JSON: str = "data/MMFakeBench_test.json"

# Directory with image sub-folders (real/, fake/, …)
IMAGES_DIR: str = "data/MMFakeBench_test-001/MMFakeBench_test"

# How many samples to load.
# • None  → load ALL (can be slow + expensive)
# • int   → random subset of that size (seed below)
LIMIT: int | None = 5

# Random seed used for sampling / shuffling
SEED: int = 71

# --------------------------------------------------------------------------- #
# LLM / model routing
# --------------------------------------------------------------------------- #

# Default model name recognised by ModelRouter
MODEL_DEFAULT: str = "gemini-2.5-flash"

# Temperature passed to the LLM (0-1)
TEMPERATURE: float = 0.2

# --------------------------------------------------------------------------- #
# Pipeline hyper-parameters
# --------------------------------------------------------------------------- #

# How many independent reasoning chains per sample
NUM_CHAINS: int = 3

# Questions generated inside each chain
NUM_Q_PER_CHAIN: int = 3

# Question-generation strategy:
#   "headline" – use headline text only
#   "report"   – require event report context (will raise if missing)
#   "auto"     – use "report" if context present, else "headline"
QGEN_STRATEGY: str = "auto"          # headline | report | auto

# --------------------------------------------------------------------------- #
# Brave search
# --------------------------------------------------------------------------- #

# Number of snippets to retrieve per query
BRAVE_K: int = 8

# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #

# Time-to-live (seconds) for network-level caches (Brave results, etc.)
CACHE_TTL_SEC: int = 60 * 60           # 1 hour

# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

# Where main_async.py writes JSONL results
RESULTS_PATH: str = "results/step08_async.jsonl"

# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #

# Toggle verbose debugging in individual modules if needed
DEBUG: bool = False

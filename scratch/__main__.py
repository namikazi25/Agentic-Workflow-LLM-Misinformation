"""
scratch.__main__
================

This file allows the package to be executed directly:

    $ python -m scratch

It merely delegates to `scratch.main_async.main()` and
passes through any keyboard interrupt cleanly.
"""

import asyncio
import sys
import argparse

from .main_async import main as _pipeline_main

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m scratch",
        description="Run the agentic misinformation pipeline."
    )
    p.add_argument("--data-json", dest="data_json", default=None,
                   help="Path to dataset metadata JSON (overrides env DATA_JSON).")
    p.add_argument("--images-dir", dest="images_dir", default=None,
                   help="Root directory for images (overrides env IMAGES_DIR).")
    p.add_argument("--limit", dest="limit", type=str, default=None,
                   help='Sample limit (int or "None") to override config/DATA_LIMIT.')
    p.add_argument("--seed", dest="seed", type=int, default=None,
                   help="Sampling seed to override config/DATA_SEED.")
    return p.parse_args(argv)

if __name__ == "__main__":
    try:
        args = _parse_args()
        # Normalize limit: allow "None"
        lim = None
        if args.limit is not None:
            lim = None if str(args.limit).lower() == "none" else int(args.limit)
        asyncio.run(_pipeline_main(
            data_json=args.data_json,
            images_dir=args.images_dir,
            limit=lim,
            seed=args.seed,
        ))
    except KeyboardInterrupt:
        # Ensure a clean exit status instead of a traceback
        sys.exit("\nInterrupted by user.")

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

from .main_async import main as _pipeline_main

if __name__ == "__main__":
    try:
        asyncio.run(_pipeline_main())
    except KeyboardInterrupt:
        # Ensure a clean exit status instead of a traceback
        sys.exit("\nInterrupted by user.")

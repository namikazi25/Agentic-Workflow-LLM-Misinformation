"""
scratch package
===============

Convenience re-exports so downstream code can write:

>>> from scratch import config as C, log

instead of importing each sub-module directly.
"""

from importlib.metadata import version, PackageNotFoundError

# Public sub-modules
from . import config              # noqa: F401
from . import log                 # noqa: F401

__all__ = ["config", "log", "__version__"]

# --------------------------------------------------------------------------- #
# Version helper (optional, shows 0.0.0 if package not installed)
# --------------------------------------------------------------------------- #
try:
    __version__: str = version("namikazi25_agentic_workflow")  # pip install -e .
except PackageNotFoundError:
    __version__ = "0.0.0"

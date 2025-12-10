"""
Alias wrapper to keep legacy `model: autogluon` templates working.

Delegates to the standard AutoGluon baseline implementation while
allowing import via a bare module file (not a Python package).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the models directory is importable when loaded via importlib
MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from autogluon_baseline import train  # type: ignore  # noqa: E402,F401

__all__ = ["train"]

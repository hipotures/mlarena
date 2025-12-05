#!/usr/bin/env python
"""
Thin wrapper for MLArena CLI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlarena.cli.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

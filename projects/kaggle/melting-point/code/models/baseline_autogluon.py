"""Thin wrapper delegating to the shared AutoGluon runner."""

from pathlib import Path
import sys

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from autogluon_runner import cli_entry  # noqa: E402


if __name__ == "__main__":
    cli_entry(default_project="melting-point")

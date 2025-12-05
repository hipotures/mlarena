"""Basic formatting helpers for CLI output."""

from typing import Dict

from mlarena.core.module import ModuleResult


def format_results(results: Dict[str, ModuleResult]) -> str:
    lines = []
    for name, res in results.items():
        status = "ok" if res.success else "fail"
        lines.append(f"[{status}] {name}")
    return "\n".join(lines)

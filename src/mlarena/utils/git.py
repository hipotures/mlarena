"""Lightweight git helpers used for experiment metadata."""

import subprocess
from pathlib import Path
from typing import Dict, List


def _run_git(args: List[str], repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def get_git_info(repo_root: Path) -> Dict[str, object]:
    info: Dict[str, object] = {}
    try:
        info["commit"] = _run_git(["rev-parse", "HEAD"], repo_root)
        info["branch"] = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
        status = _run_git(["status", "--short"], repo_root)
        info["is_dirty"] = bool(status)
        info["dirty_files"] = status.splitlines()
    except Exception:
        # Repository may not be initialized; return empty info gracefully.
        return {}
    return info

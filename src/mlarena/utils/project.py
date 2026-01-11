"""Project-level utilities for MLArena modules."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Tuple


def load_project_config(project_root: Path):
    """
    Attempt to import project-specific config from code/utils/config.py.
    Returns a SimpleNamespace fallback with sane defaults when missing.
    """
    code_dir = project_root / "code"
    if code_dir.exists() and str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    try:
        return importlib.import_module("utils.config")
    except Exception:
        # Minimal fallback for tests or incomplete projects
        data_dir = project_root / "data"
        return SimpleNamespace(
            PROJECT_ROOT=project_root,
            DATA_DIR=data_dir,
            TRAIN_PATH=data_dir / "train.csv.gz",
            TEST_PATH=data_dir / "test.csv.gz",
            TARGET_COLUMN="target",
            ID_COLUMN="id",
            AUTOGLUON_PROBLEM_TYPE=None,
            AUTOGLUON_EVAL_METRIC=None,
            AUTOGLUON_PRESET="medium",
            AUTOGLUON_TIME_LIMIT=300,
            SUBMISSION_PROBAS=False,
            COMPETITION_NAME=project_root.name,
            IGNORED_COLUMNS=[],
        )


def load_submission_module(project_root: Path):
    """
    Import project-level submission helper if present, otherwise fall back
    to global kaggle_tools submission utilities.
    """
    code_dir = project_root / "code"
    if code_dir.exists() and str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    try:
        return importlib.import_module("utils.submission")
    except Exception:
        from kaggle_tools import submission

        return submission


def data_paths(config_module) -> Tuple[Path, Path]:
    train = getattr(config_module, "TRAIN_PATH", None)
    test = getattr(config_module, "TEST_PATH", None)

    # If not defined or defined but doesn't exist, try to resolve/fallback
    if train is None or not Path(train).exists():
        data_dir = getattr(config_module, "DATA_DIR", None)
        if data_dir is None and train is not None:
            data_dir = Path(train).parent
        
        if data_dir:
            # Try extensions in priority order
            for ext in [".parquet", ".csv.gz", ".csv"]:
                p = Path(data_dir) / f"train{ext}"
                if p.exists():
                    train = p
                    break
            if train is None:
                train = Path(data_dir) / "train.csv.gz" # Default fallback

    if test is None or not Path(test).exists():
        data_dir = getattr(config_module, "DATA_DIR", None)
        if data_dir is None and test is not None:
            data_dir = Path(test).parent
            
        if data_dir:
            for ext in [".parquet", ".csv.gz", ".csv"]:
                p = Path(data_dir) / f"test{ext}"
                if p.exists():
                    test = p
                    break
            if test is None:
                test = Path(data_dir) / "test.csv.gz" # Default fallback

    return Path(train), Path(test)

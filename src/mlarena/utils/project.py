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
    if train is None or test is None:
        data_dir = getattr(config_module, "DATA_DIR", Path("."))  # type: ignore

        # Priorytet: .csv.gz (compressed)
        train = data_dir / "train.csv.gz"
        test = data_dir / "test.csv.gz"

        # Fallback do .csv jeśli .gz nie istnieje (kompatybilność wsteczna)
        if not train.exists():
            train_csv = data_dir / "train.csv"
            if train_csv.exists():
                train = train_csv

        if not test.exists():
            test_csv = data_dir / "test.csv"
            if test_csv.exists():
                test = test_csv

    return Path(train), Path(test)

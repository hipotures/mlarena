"""Artifact management utilities for preprocessing sub-modules."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict


def save_fitted_object(obj: Any, artifact_dir: Path, name: str) -> Path:
    """
    Pickle and save fitted sklearn/custom object.

    Args:
        obj: Object to save (must be picklable)
        artifact_dir: Directory to save artifact to
        name: Filename (should end with .pkl)

    Returns:
        Path to saved file
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not name.endswith(".pkl"):
        name = f"{name}.pkl"

    filepath = artifact_dir / name

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

    return filepath


def load_fitted_object(artifact_dir: Path, name: str) -> Any:
    """
    Load pickled object.

    Args:
        artifact_dir: Directory containing artifact
        name: Filename (should end with .pkl)

    Returns:
        Loaded object

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    artifact_dir = Path(artifact_dir)

    if not name.endswith(".pkl"):
        name = f"{name}.pkl"

    filepath = artifact_dir / name

    if not filepath.exists():
        raise FileNotFoundError(
            f"Artifact not found: {filepath}"
        )

    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_report(data: Dict, artifact_dir: Path, name: str) -> Path:
    """
    Save JSON report.

    Args:
        data: Dictionary to save as JSON
        artifact_dir: Directory to save report to
        name: Filename (should end with .json)

    Returns:
        Path to saved file
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not name.endswith(".json"):
        name = f"{name}.json"

    filepath = artifact_dir / name

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return filepath


def load_report(artifact_dir: Path, name: str) -> Dict:
    """
    Load JSON report.

    Args:
        artifact_dir: Directory containing report
        name: Filename (should end with .json)

    Returns:
        Loaded dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    artifact_dir = Path(artifact_dir)

    if not name.endswith(".json"):
        name = f"{name}.json"

    filepath = artifact_dir / name

    if not filepath.exists():
        raise FileNotFoundError(
            f"Report not found: {filepath}"
        )

    with open(filepath, "r") as f:
        return json.load(f)


def get_submodule_artifact_dir(artifact_dir: Path, submodule_name: str) -> Path:
    """
    Create and return submodule-specific artifact directory.

    Creates structure: {artifact_dir}/submodules/{submodule_name}/

    Args:
        artifact_dir: Base artifact directory (e.g., experiments/pre-{template}/artifacts/preprocess)
        submodule_name: Name of the sub-module

    Returns:
        Path to sub-module artifact directory
    """
    artifact_dir = Path(artifact_dir)
    submodule_dir = artifact_dir / "submodules" / submodule_name
    submodule_dir.mkdir(parents=True, exist_ok=True)
    return submodule_dir

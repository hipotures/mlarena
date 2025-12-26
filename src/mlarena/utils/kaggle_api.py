"""Kaggle API utilities for submission management."""

from __future__ import annotations

import csv
import subprocess
from typing import Optional


def fetch_submissions(competition: str) -> list[dict]:
    """
    Fetch submissions for a competition via Kaggle CLI.

    Args:
        competition: Kaggle competition name/slug

    Returns:
        List of dicts with keys: id/submissionId, fileName/file_name,
        publicScore, description, etc.
        Returns empty list if command fails.
    """
    try:
        out = subprocess.check_output(
            ["kaggle", "competitions", "submissions", "-c", competition, "--csv"],
            text=True,
        )
        reader = csv.DictReader(out.splitlines())
        return list(reader)
    except Exception:
        return []


def find_submission_by_message(
    submissions: list[dict],
    experiment_id: str
) -> Optional[dict]:
    """
    Find submission by searching for experiment_id in description field.

    Args:
        submissions: List of submission dicts from fetch_submissions()
        experiment_id: Experiment ID to search for (e.g., "exp-20251225-155517")

    Returns:
        Matching submission dict or None if not found
    """
    for sub in submissions:
        description = sub.get("description", "")
        if experiment_id in description:
            return sub
    return None


def find_submission_by_filename(
    submissions: list[dict],
    filename: str
) -> Optional[dict]:
    """
    Find submission by exact filename match.

    Args:
        submissions: List of submission dicts from fetch_submissions()
        filename: Filename to search for (e.g., "submission-20251225155530.csv")

    Returns:
        Matching submission dict or None if not found
    """
    for sub in submissions:
        # Kaggle API uses both 'fileName' and 'file_name' in different contexts
        if sub.get("fileName") == filename or sub.get("file_name") == filename:
            return sub
    return None

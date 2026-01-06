#!/usr/bin/env python
"""Fetch Kaggle submission scores and map them to local experiments.

- Reads Kaggle submissions CSV
- Extracts exp-YYYYMMDD-HHMMSS from description
- Runs `mla fetch-score` per experiment (latest submission per exp)
- Skips experiments missing locally
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

EXP_PATTERN = re.compile(r"exp-\d{8}-\d{6}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fetch_submissions(competition: str) -> List[Dict[str, str]]:
    try:
        out = subprocess.check_output(
            ["kaggle", "competitions", "submissions", "-c", competition, "--csv"],
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[error] Kaggle CLI failed: {exc}")
        return []
    except FileNotFoundError:
        print("[error] Kaggle CLI not found. Is `kaggle` installed and on PATH?")
        return []

    reader = csv.DictReader(out.splitlines())
    return list(reader)


def _extract_exp_id(description: str) -> Optional[str]:
    if not description:
        return None
    match = EXP_PATTERN.search(description)
    return match.group(0) if match else None


def _file_name(row: Dict[str, str]) -> Optional[str]:
    for key in ("fileName", "file_name", "filename"):
        value = row.get(key)
        if value:
            return value
    return None


def _iter_targets(rows: Iterable[Dict[str, str]]) -> Iterable[Dict[str, str]]:
    seen = set()
    for row in rows:
        exp_id = _extract_exp_id(row.get("description", ""))
        if not exp_id or exp_id in seen:
            continue
        seen.add(exp_id)
        filename = _file_name(row)
        yield {"exp_id": exp_id, "file_name": filename or ""}


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill MLA fetch-score from Kaggle submissions CSV")
    parser.add_argument("--project", required=True, help="Project slug (e.g., playground-series-s5e12)")
    parser.add_argument(
        "--competition",
        default=None,
        help="Kaggle competition name (defaults to project slug)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of experiments processed")

    args = parser.parse_args()
    competition = args.competition or args.project

    rows = _fetch_submissions(competition)
    if not rows:
        print("[error] No submissions returned from Kaggle")
        return 1

    repo_root = _repo_root()
    exp_root = repo_root / "projects" / "kaggle" / args.project / "experiments"

    if not exp_root.exists():
        print(f"[error] Experiments dir not found: {exp_root}")
        return 1

    missing = []
    processed = 0
    first_run = True

    for target in _iter_targets(rows):
        exp_id = target["exp_id"]
        file_name = target["file_name"]

        if args.limit is not None and processed >= args.limit:
            break

        exp_dir = exp_root / exp_id
        if not exp_dir.exists():
            missing.append(exp_id)
            continue

        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "mla.py"),
            "fetch-score",
            "--project",
            args.project,
            "--exp-id",
            exp_id,
            "--force",
        ]
        if file_name:
            cmd.append(f"fetch-score.submission_file={file_name}")

        processed += 1
        if args.dry_run:
            print(" ".join(cmd))
        else:
            if not first_run:
                time.sleep(1)
            print(f"[info] {exp_id} -> {file_name}")
            subprocess.run(cmd, check=False)
            first_run = False

    print(f"\nDone. Processed: {processed}")
    if missing:
        print(f"Missing experiments ({len(missing)}):")
        for exp_id in missing:
            print(f"  - {exp_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

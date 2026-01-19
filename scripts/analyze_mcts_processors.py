#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Any


def _find_mcts_db(root: Path, project: str) -> Path:
    candidates = [
        root / "projects" / "kaggle" / project / "experiments" / "db" / "mcts.db",
        root / "projects" / "kaggle" / project / "experiments" / "mcts.db",
        root / "projects" / "kaggle" / project / "mcts.db",
    ]
    for path in candidates:
        if path.exists():
            return path

    project_root = root / "projects" / "kaggle" / project
    if project_root.exists():
        matches = list(project_root.rglob("mcts.db"))
        if matches:
            # Prefer the largest DB (likely most complete)
            matches.sort(key=lambda p: p.stat().st_size, reverse=True)
            return matches[0]

    raise FileNotFoundError(f"mcts.db not found under {project_root}")


def _get_study_sizes(conn: sqlite3.Connection) -> list[tuple[int, str, int]]:
    rows = conn.execute(
        """
        SELECT s.study_id, s.study_name, COUNT(t.trial_id) AS n_trials
        FROM studies s
        LEFT JOIN trials t ON t.study_id = s.study_id
        GROUP BY s.study_id
        """
    ).fetchall()
    return [(int(r[0]), str(r[1]), int(r[2])) for r in rows]


def _get_direction(conn: sqlite3.Connection, study_id: int) -> str:
    row = conn.execute(
        "SELECT direction FROM study_directions WHERE study_id=? AND objective=0",
        (study_id,),
    ).fetchone()
    if not row:
        return "unknown"
    direction = int(row[0])
    if direction == 1:
        return "minimize"
    if direction == 2:
        return "maximize"
    return "unknown"


def _analyze(
    conn: sqlite3.Connection, study_id: int, direction: str, eps: float
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    counters = {
        "edges": 0,
        "skipped_missing_values": 0,
        "skipped_missing_action": 0,
    }

    rows = conn.execute(
        """
        SELECT e.action_json,
               pv_parent.value AS parent_value,
               pv_child.value AS child_value
        FROM mcts_edges e
        JOIN trials tchild ON tchild.trial_id = e.child_trial_id
        LEFT JOIN trial_values pv_parent ON pv_parent.trial_id = e.parent_trial_id AND pv_parent.objective = 0
        LEFT JOIN trial_values pv_child ON pv_child.trial_id = e.child_trial_id AND pv_child.objective = 0
        WHERE tchild.study_id = ?
        """,
        (study_id,),
    ).fetchall()

    for action_json, parent_value, child_value in rows:
        counters["edges"] += 1
        if parent_value is None or child_value is None:
            counters["skipped_missing_values"] += 1
            continue

        step_name = None
        try:
            action = json.loads(action_json) if action_json else None
            if isinstance(action, dict):
                step_name = action.get("step_name")
        except json.JSONDecodeError:
            step_name = None

        if not step_name:
            counters["skipped_missing_action"] += 1
            continue

        delta = float(child_value) - float(parent_value)
        if abs(delta) <= eps:
            bucket = "no_change"
        else:
            if direction == "minimize":
                bucket = "improved" if delta < 0 else "worsened"
            elif direction == "maximize":
                bucket = "improved" if delta > 0 else "worsened"
            else:
                bucket = "worsened" if delta > 0 else "improved"

        if step_name not in stats:
            stats[step_name] = {"no_change": 0, "improved": 0, "worsened": 0, "total": 0}
        stats[step_name][bucket] += 1
        stats[step_name]["total"] += 1

    return stats, counters


def _print_table(stats: dict[str, dict[str, int]], counters: dict[str, int], *, study_id: int, study_name: str, direction: str, db_path: Path) -> None:
    rows = sorted(stats.items(), key=lambda kv: (-kv[1]["total"], kv[0]))

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print(f"DB: [cyan]{db_path}[/cyan]")
        console.print(f"Study: [bold]{study_name}[/bold] (id={study_id}, direction={direction})")
        console.print(
            f"Edges: {counters['edges']} | Skipped (values): {counters['skipped_missing_values']} | Skipped (action): {counters['skipped_missing_action']}"
        )

        table = Table(title="Processor impact (by step)")
        table.add_column("step")
        table.add_column("no_change", justify="right")
        table.add_column("improved", justify="right")
        table.add_column("worsened", justify="right")
        table.add_column("total", justify="right")
        for step, counts in rows:
            table.add_row(
                step,
                str(counts["no_change"]),
                str(counts["improved"]),
                str(counts["worsened"]),
                str(counts["total"]),
            )
        console.print(table)

        only_no_change = [
            step
            for step, counts in rows
            if counts["improved"] == 0 and counts["worsened"] == 0 and counts["no_change"] > 0
        ]
        if only_no_change:
            console.print("Steps with only no_change:")
            console.print(", ".join(only_no_change))
    except Exception:
        print(f"DB: {db_path}")
        print(f"Study: {study_name} (id={study_id}, direction={direction})")
        print(
            f"Edges: {counters['edges']} | Skipped (values): {counters['skipped_missing_values']} | Skipped (action): {counters['skipped_missing_action']}"
        )
        print("step,no_change,improved,worsened,total")
        for step, counts in rows:
            print(
                f"{step},{counts['no_change']},{counts['improved']},{counts['worsened']},{counts['total']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze MCTS processor impact by step")
    parser.add_argument("--root", default="/mnt/mlarena", help="NFS root (default: /mnt/mlarena)")
    parser.add_argument("--project", default="playground-series-s6e1", help="Project slug")
    parser.add_argument("--epsilon", type=float, default=1e-9, help="Delta threshold for no_change")
    parser.add_argument(
        "--study-id",
        type=int,
        default=None,
        help="Optional explicit study_id (overrides largest study selection)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    db_path = _find_mcts_db(root, args.project)

    conn = sqlite3.connect(db_path)
    try:
        studies = _get_study_sizes(conn)
        if not studies:
            raise RuntimeError("No studies found in mcts.db")

        if args.study_id is not None:
            matching = [s for s in studies if s[0] == args.study_id]
            if not matching:
                raise RuntimeError(f"study_id {args.study_id} not found in mcts.db")
            study_id, study_name, _ = matching[0]
        else:
            # Pick the largest study by trial count, tie-break by study_id
            studies.sort(key=lambda s: (s[2], s[0]), reverse=True)
            study_id, study_name, _ = studies[0]

        direction = _get_direction(conn, study_id)
        stats, counters = _analyze(conn, study_id, direction, args.epsilon)
    finally:
        conn.close()

    _print_table(stats, counters, study_id=study_id, study_name=study_name, direction=direction, db_path=db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

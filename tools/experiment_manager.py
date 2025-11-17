"""
Experiment Manager - track modular pipeline state (EDA, model, submit, etc.).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_experiment_id() -> str:
    return datetime.utcnow().strftime("exp-%Y%m%d-%H%M%S")


def run_git_command(args, cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()


def get_git_info(project_root: Path) -> Dict:
    repo_root = project_root.parent
    git_info = {
        "hash": None,
        "branch": None,
        "has_uncommitted_changes": True,
    }
    try:
        git_info["hash"] = run_git_command(["git", "rev-parse", "HEAD"], repo_root)
        git_info["branch"] = run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
        status = run_git_command(["git", "status", "--porcelain"], repo_root)
        git_info["has_uncommitted_changes"] = bool(status)
    except Exception:
        pass
    return git_info


def load_project_config(project_name: str):
    project_root = REPO_ROOT / project_name
    code_dir = project_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    return __import__("utils.config", fromlist=["dummy"])


@dataclass
class ExperimentManager:
    project_name: str
    experiment_id: str
    project_root: Path
    base_dir: Path
    artifact_dir: Path
    json_path: Path
    data: Dict = field(default_factory=dict)

    @classmethod
    def load_or_create(cls, project_name: str, experiment_id: Optional[str] = None) -> "ExperimentManager":
        project_root = REPO_ROOT / project_name
        base_dir = project_root / "experiments"
        if experiment_id is None:
            experiment_id = generate_experiment_id()
        json_path = base_dir / f"{experiment_id}.json"
        artifact_dir = base_dir / experiment_id
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
        else:
            data = {
                "experiment_id": experiment_id,
                "project": project_name,
                "created_at": utc_now(),
                "git": get_git_info(project_root),
                "modules": {},
            }
            base_dir.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
        return cls(
            project_name=project_name,
            experiment_id=experiment_id,
            project_root=project_root,
            base_dir=base_dir,
            artifact_dir=artifact_dir,
            json_path=json_path,
            data=data,
        )

    def save(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.json_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def modules(self) -> Dict:
        return self.data.setdefault("modules", {})

    def get_module(self, module: str) -> Optional[Dict]:
        return self.modules().get(module)

    def require(self, module: str):
        entry = self.get_module(module)
        if not entry or entry.get("status") != "completed":
            raise RuntimeError(f"Module '{module}' must complete before continuing.")

    def start_module(self, module: str, extra: Optional[Dict] = None):
        entry = self.modules().setdefault(module, {})
        entry.update(extra or {})
        entry["status"] = "running"
        entry["started_at"] = utc_now()
        entry["updated_at"] = utc_now()
        self.save()

    def complete_module(self, module: str, payload: Dict):
        entry = self.modules().setdefault(module, {})
        entry.update(payload)
        entry["status"] = "completed"
        entry["finished_at"] = utc_now()
        entry["updated_at"] = utc_now()
        self.save()

    def artifact_path(self, filename: str) -> Path:
        path = self.artifact_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def run_eda(args):
    manager = ExperimentManager.load_or_create(args.project, args.experiment_id)
    manager.start_module("eda", {"notes": args.notes or ""})
    config = load_project_config(args.project)
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df = pd.read_csv(config.TEST_PATH)
    summary = {
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "columns": train_df.columns.tolist(),
    }
    if hasattr(config, "TARGET_COLUMN") and config.TARGET_COLUMN in train_df.columns:
        summary["target_distribution"] = train_df[config.TARGET_COLUMN].value_counts(normalize=True).to_dict()
    stats_path = manager.artifact_path("eda_summary.json")
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)
    manager.complete_module(
        "eda",
        {
            "summary_file": str(stats_path.relative_to(manager.project_root)),
            "train_shape": train_df.shape,
            "test_shape": test_df.shape,
        },
    )
    print(f"[EDA] Experiment {manager.experiment_id} summary saved to {stats_path}")


def run_list(args):
    project_root = REPO_ROOT / args.project
    base_dir = project_root / "experiments"
    if not base_dir.exists():
        print("No experiments found.")
        return
    for path in sorted(base_dir.glob("exp-*.json")):
        with open(path) as f:
            data = json.load(f)
        modules = data.get("modules", {})
        statuses = ", ".join(f"{k}:{v.get('status')}" for k, v in modules.items())
        print(f"{data['experiment_id']} - {statuses}")


def build_parser():
    parser = argparse.ArgumentParser(description="Experiment workflow manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eda_parser = subparsers.add_parser("eda", help="Run EDA module")
    eda_parser.add_argument("--project", required=True)
    eda_parser.add_argument("--experiment-id")
    eda_parser.add_argument("--notes")
    eda_parser.set_defaults(func=run_eda)

    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--project", required=True)
    list_parser.set_defaults(func=run_list)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

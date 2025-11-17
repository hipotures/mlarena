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
TOOLS_ROOT = Path(__file__).resolve().parent
MODULES = ["eda", "model", "submit", "fetch-score"]


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
        artifact_dir = base_dir / experiment_id
        json_path = artifact_dir / "state.json"
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
            artifact_dir.mkdir(parents=True, exist_ok=True)
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

    @classmethod
    def load_existing(cls, project_name: str, experiment_id: str) -> "ExperimentManager":
        if not experiment_id:
            raise ValueError("experiment-id is required")
        project_root = REPO_ROOT / project_name
        base_dir = project_root / "experiments"
        artifact_dir = base_dir / experiment_id
        json_path = artifact_dir / "state.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Experiment '{experiment_id}' not found in {artifact_dir}")
        with open(json_path) as f:
            data = json.load(f)
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
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
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
        modules = self.modules()
        entry = modules.get(module)
        if entry and entry.get("status") == "completed":
            raise RuntimeError(f"Module '{module}' is already completed for {self.experiment_id}")
        entry = modules.setdefault(module, {})
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
    for dir_path in sorted(base_dir.glob("exp-*")):
        state_path = dir_path / "state.json"
        if not state_path.exists():
            continue
        with open(state_path) as f:
            data = json.load(f)
        modules = data.get("modules", {})
        statuses = ", ".join(f"{k}:{v.get('status')}" for k, v in modules.items())
        print(f"{data['experiment_id']} - {statuses}")


def run_modules(args):
    print("Available modules:")
    for module in MODULES:
        print(f"- {module}")


def run_model(args):
    manager = ExperimentManager.load_or_create(args.project, args.experiment_id)
    if args.experiment_id is None:
        print(f"[Model] Using new experiment ID: {manager.experiment_id}")
    script = TOOLS_ROOT / "autogluon_runner.py"
    cmd = [
        sys.executable,
        str(script),
        "--project",
        args.project,
        "--template",
        args.template,
        "--experiment-id",
        manager.experiment_id,
    ]
    if args.skip_submit:
        cmd.append("--skip-submit")
    if args.auto_submit:
        cmd.append("--auto-submit")
    if args.skip_score_fetch:
        cmd.append("--skip-score-fetch")
    if args.skip_git:
        cmd.append("--skip-git")
    if args.force_extreme:
        cmd.append("--force-extreme")
    if args.skip_eda_check:
        cmd.append("--skip-eda-check")
    if args.time_limit is not None:
        cmd += ["--time-limit", str(args.time_limit)]
    if args.preset:
        cmd += ["--preset", args.preset]
    if args.use_gpu is not None:
        cmd += ["--use-gpu", str(args.use_gpu)]
    cmd += ["--wait-seconds", str(args.wait_seconds)]
    cmd += ["--cdp-url", args.cdp_url]
    if args.kaggle_message:
        cmd += ["--kaggle-message", args.kaggle_message]
    subprocess.run(cmd, check=True)


def _resolve_submission_filename(project: str, experiment_id: Optional[str], explicit_filename: Optional[str]) -> str:
    if explicit_filename:
        return explicit_filename
    if not experiment_id:
        raise ValueError("Provide --experiment-id or --filename")
    manager = ExperimentManager.load_existing(project, experiment_id)
    model_module = manager.get_module("model")
    if not model_module or not model_module.get("submission_file"):
        raise RuntimeError(f"Experiment {experiment_id} has no recorded submission file")
    submission_path = Path(model_module["submission_file"])
    return submission_path.name


def run_submit(args):
    filename = _resolve_submission_filename(args.project, args.experiment_id, args.filename)
    script = TOOLS_ROOT / "submission_workflow.py"
    cmd = [
        sys.executable,
        str(script),
        "submit",
        "--project",
        args.project,
        "--filename",
        filename,
        "--wait-seconds",
        str(args.wait_seconds),
        "--cdp-url",
        args.cdp_url,
    ]
    if args.experiment_id:
        cmd += ["--experiment-id", args.experiment_id]
    if args.kaggle_message:
        cmd += ["--kaggle-message", args.kaggle_message]
    if args.skip_score_fetch:
        cmd.append("--skip-score-fetch")
    if args.skip_git:
        cmd.append("--skip-git")
    subprocess.run(cmd, check=True)


def run_fetch_score(args):
    filename = _resolve_submission_filename(args.project, args.experiment_id, args.filename)
    script = TOOLS_ROOT / "submission_workflow.py"
    cmd = [
        sys.executable,
        str(script),
        "fetch",
        "--project",
        args.project,
        "--filename",
        filename,
        "--wait-seconds",
        str(args.wait_seconds),
        "--cdp-url",
        args.cdp_url,
    ]
    if args.experiment_id:
        cmd += ["--experiment-id", args.experiment_id]
    if args.kaggle_message:
        cmd += ["--kaggle-message", args.kaggle_message]
    if args.skip_score_fetch:
        cmd.append("--skip-score-fetch")
    if args.skip_git:
        cmd.append("--skip-git")
    subprocess.run(cmd, check=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Experiment workflow manager",
        epilog=(
            "Examples:\n"
            "  # Start EDA (auto experiment_id)\n"
            "  uv run python tools/experiment_manager.py eda --project playground-series-s5e11\n\n"
            "  # Train model using template fast-cpu and resume ID\n"
            "  uv run python tools/experiment_manager.py model --project playground-series-s5e11 "
            "--experiment-id exp-20250101-010101 --template fast-cpu --skip-submit\n\n"
            "  # Submit existing CSV and fetch score\n"
            "  uv run python tools/experiment_manager.py submit --project playground-series-s5e11 "
            "--experiment-id exp-20250101-010101\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    eda_parser = subparsers.add_parser("eda", help="Run EDA module")
    eda_parser.add_argument("--project", required=True)
    eda_parser.add_argument("--experiment-id")
    eda_parser.add_argument("--notes")
    eda_parser.set_defaults(func=run_eda)

    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--project", required=True)
    list_parser.set_defaults(func=run_list)

    modules_parser = subparsers.add_parser("modules", help="List available modules")
    modules_parser.set_defaults(func=run_modules)

    model_parser = subparsers.add_parser("model", help="Run modeling module")
    model_parser.add_argument("--project", required=True)
    model_parser.add_argument("--experiment-id")
    model_parser.add_argument("--template", required=True)
    model_parser.add_argument("--time-limit", type=int)
    model_parser.add_argument("--preset")
    model_parser.add_argument("--use-gpu", type=int, choices=[0, 1])
    model_parser.add_argument("--force-extreme", action="store_true")
    model_parser.add_argument("--skip-submit", action="store_true")
    model_parser.add_argument("--auto-submit", action="store_true")
    model_parser.add_argument("--skip-score-fetch", action="store_true")
    model_parser.add_argument("--skip-git", action="store_true")
    model_parser.add_argument("--skip-eda-check", action="store_true")
    model_parser.add_argument("--wait-seconds", type=int, default=30)
    model_parser.add_argument("--cdp-url", default="http://localhost:9222")
    model_parser.add_argument("--kaggle-message")
    model_parser.set_defaults(func=run_model)

    submit_parser = subparsers.add_parser("submit", help="Submit an existing CSV and update experiment")
    submit_parser.add_argument("--project", required=True)
    submit_parser.add_argument("--experiment-id")
    submit_parser.add_argument("--filename")
    submit_parser.add_argument("--wait-seconds", type=int, default=30)
    submit_parser.add_argument("--cdp-url", default="http://localhost:9222")
    submit_parser.add_argument("--skip-score-fetch", action="store_true")
    submit_parser.add_argument("--skip-git", action="store_true")
    submit_parser.add_argument("--kaggle-message")
    submit_parser.set_defaults(func=run_submit)

    fetch_parser = subparsers.add_parser("fetch-score", help="Fetch leaderboard score for an existing submission")
    fetch_parser.add_argument("--project", required=True)
    fetch_parser.add_argument("--experiment-id")
    fetch_parser.add_argument("--filename")
    fetch_parser.add_argument("--wait-seconds", type=int, default=0)
    fetch_parser.add_argument("--cdp-url", default="http://localhost:9222")
    fetch_parser.add_argument("--skip-score-fetch", action="store_true")
    fetch_parser.add_argument("--skip-git", action="store_true")
    fetch_parser.add_argument("--kaggle-message")
    fetch_parser.set_defaults(func=run_fetch_score)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""
Experiment Manager - track modular pipeline state (EDA, model, submit, etc.).
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_ROOT = Path(__file__).resolve().parent
MODULES = ["eda", "model", "submit", "fetch-score"]


class ModuleStateError(RuntimeError):
    """Raised when pipeline module state prevents the requested action."""


class ModuleAlreadyCompleted(ModuleStateError):
    pass


class ModuleAlreadyRunning(ModuleStateError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_experiment_id() -> str:
    return datetime.now(timezone.utc).strftime("exp-%Y%m%d-%H%M%S")


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
    project_root = REPO_ROOT / "projects" / "kaggle" / project_name
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
        project_root = REPO_ROOT / "projects" / "kaggle" / project_name
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
        project_root = REPO_ROOT / "projects" / "kaggle" / project_name
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
            raise ModuleStateError(f"Module '{module}' must complete before continuing.")

    def _is_pid_active(self, pid: Optional[int]) -> bool:
        if not pid:
            return False
        try:
            os.kill(pid, 0)
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return False
            if exc.errno == errno.EPERM:
                return True
            return False
        else:
            return True

    def start_module(self, module: str, extra: Optional[Dict] = None, allow_restart: bool = False):
        modules = self.modules()
        entry = modules.get(module)
        if entry:
            status = entry.get("status")
            if status == "completed":
                raise ModuleAlreadyCompleted(
                    f"Module '{module}' is already completed for experiment {self.experiment_id}. "
                    "Create a new experiment ID if you need to rerun it."
                )
            if status == "running":
                pid = entry.get("pid")
                finished = entry.get("finished_at")
                pid_active = self._is_pid_active(pid)
                if allow_restart and (finished or not pid_active):
                    self.fail_module(module, "Detected stale running entry", {"previous_pid": pid})
                    entry = modules.get(module)
                else:
                    raise ModuleAlreadyRunning(
                        f"Module '{module}' is already running for experiment {self.experiment_id}."
                    )
        new_entry = dict(extra or {})
        new_entry["status"] = "running"
        new_entry["started_at"] = utc_now()
        new_entry["updated_at"] = utc_now()
        new_entry["pid"] = os.getpid()
        modules[module] = new_entry
        self.save()

    def complete_module(self, module: str, payload: Dict):
        entry = self.modules().setdefault(module, {})
        entry.update(payload)
        entry["status"] = "completed"
        entry["finished_at"] = utc_now()
        entry["updated_at"] = utc_now()
        entry["pid"] = None
        self.save()

    def fail_module(self, module: str, reason: str, payload: Optional[Dict] = None):
        entry = self.modules().setdefault(module, {})
        entry.update(payload or {})
        entry["status"] = "failed"
        entry["error"] = reason
        entry["pid"] = None
        entry["finished_at"] = utc_now()
        entry["updated_at"] = utc_now()
        self.save()

    def artifact_path(self, filename: str) -> Path:
        path = self.artifact_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def run_eda(args):
    manager = ExperimentManager.load_or_create(args.project, args.experiment_id)
    manager.start_module("eda", {"notes": args.notes or ""}, allow_restart=True)
    try:
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
    except Exception as exc:
        manager.fail_module("eda", str(exc))
        raise


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
    existing = manager.get_module("model")
    if existing and existing.get("status") == "completed":
        print(
            f"[Model] Module already completed for experiment {manager.experiment_id}; "
            "create a new experiment to retrain."
        )
        return
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
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[Model] Training command failed (exit {exc.returncode}).")
        raise SystemExit(exc.returncode)


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


def fetch_kaggle_evaluation(competition_slug: str) -> str:
    """
    Fetch Evaluation info from Kaggle competition.

    Simple approach: Use sample_submission.csv analysis + competition category.
    Most Kaggle competitions follow standard patterns:
    - Binary classification (0/1 predictions) → ROC AUC
    - Regression (continuous values) → RMSE/MAE
    - Multiclass (3+ classes) → Log Loss/Accuracy

    Args:
        competition_slug: Kaggle competition identifier (e.g., 'titanic')

    Returns:
        Evaluation hint text based on sample_submission pattern

    """
    try:
        from pathlib import Path
        import pandas as pd

        # Try to read any *submission*.csv file
        project_root = REPO_ROOT / "projects" / "kaggle" / competition_slug
        data_dir = project_root / "data"

        sample_path = None
        if data_dir.exists():
            submission_files = list(data_dir.glob("*submission*.csv"))
            if submission_files:
                sample_path = submission_files[0]

        if sample_path:
            sample = pd.read_csv(sample_path, nrows=5)
            if len(sample.columns) >= 2:
                target_col = sample.columns[1]
                target_values = sample[target_col]

                # Analyze target pattern
                if target_values.dtype in ['float64', 'float32']:
                    # Check if probabilities (0-1 range) or continuous
                    if (target_values >= 0).all() and (target_values <= 1).all():
                        return "Submissions evaluated on probability predictions (0-1 range). Likely uses ROC AUC or Log Loss metric."
                    else:
                        return "Submissions evaluated on continuous value predictions. Likely uses RMSE or MAE metric."
                elif target_values.dtype in ['int64', 'int32']:
                    unique_vals = target_values.nunique()
                    if unique_vals == 2:
                        return "Submissions evaluated on binary classification (0/1). Likely uses ROC AUC metric."
                    else:
                        return f"Submissions evaluated on multiclass classification ({unique_vals} classes). Likely uses Log Loss or Accuracy metric."

        return ""

    except Exception:
        return ""  # Graceful fallback


def run_init_project(args):
    """Initialize a new competition project with standard structure."""
    import shutil
    import zipfile
    from rich.console import Console
    from rich.table import Table

    console = Console()
    project_name = args.project
    project_root = REPO_ROOT / "projects" / "kaggle" / project_name
    template_project = REPO_ROOT / "config" / "templates" / "kaggle_competition"

    # Check if project already exists
    if project_root.exists() and not args.migrate:
        console.print(f"[red]Error: Project '{project_name}' already exists. Use --migrate to migrate old project.[/red]")
        sys.exit(1)

    # Handle migration
    if args.migrate:
        if not project_root.exists():
            console.print(f"[red]Error: Project '{project_name}' does not exist. Cannot migrate.[/red]")
            sys.exit(1)

        console.print(f"[yellow]Migrating old project '{project_name}' to new structure...[/yellow]")
        old_backup = project_root / ".old"
        old_backup.mkdir(exist_ok=True)

        # Move all existing files/dirs to .old/ (except .old itself)
        for item in project_root.iterdir():
            if item.name != ".old":
                dest = old_backup / item.name
                console.print(f"  Moving {item.name} → .old/{item.name}")
                shutil.move(str(item), str(dest))

        console.print(f"[green]✓ Old project backed up to {old_backup}[/green]\n")

    # Create project directory
    project_root.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]Creating project structure for '{project_name}'...[/cyan]")

    # Create directory structure
    dirs_to_create = [
        "data",
        "code/exploration",
        "code/models",
        "code/utils",
        "docs",
        "experiments",
        "submissions",
    ]

    for dir_path in dirs_to_create:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        console.print(f"  [green]✓[/green] {dir_path}/")

    # Create .gitkeep files
    for keep_dir in ["data", "docs", "experiments", "submissions"]:
        (project_root / keep_dir / ".gitkeep").touch()

    # Copy template files
    console.print("\n[cyan]Copying template files...[/cyan]")

    # .gitignore
    shutil.copy(template_project / ".gitignore", project_root / ".gitignore")
    console.print("  [green]✓[/green] .gitignore")

    # README.md (will customize later)
    shutil.copy(template_project / "README.md", project_root / "README.md")
    console.print("  [green]✓[/green] README.md")

    # code/exploration/01_initial_eda.py
    shutil.copy(
        template_project / "code/exploration/01_initial_eda.py",
        project_root / "code/exploration/01_initial_eda.py"
    )
    console.print("  [green]✓[/green] code/exploration/01_initial_eda.py")

    # code/models/baseline_autogluon.py (will customize later)
    shutil.copy(
        template_project / "code/models/baseline_autogluon.py",
        project_root / "code/models/baseline_autogluon.py"
    )
    console.print("  [green]✓[/green] code/models/baseline_autogluon.py")

    # code/utils/submission.py (wrapper - use as-is)
    shutil.copy(
        template_project / "code/utils/submission.py",
        project_root / "code/utils/submission.py"
    )
    console.print("  [green]✓[/green] code/utils/submission.py")

    # code/utils/config.py (will customize)
    console.print("  [green]✓[/green] code/utils/config.py (will customize)")

    # Download data from Kaggle if not skipped
    if not args.skip_download:
        console.print(f"\n[cyan]Downloading data from Kaggle for '{project_name}'...[/cyan]")
        data_dir = project_root / "data"

        try:
            result = subprocess.run(
                ["kaggle", "competitions", "download", "-c", project_name, "-p", str(data_dir)],
                capture_output=True,
                text=True,
                check=True
            )
            console.print(f"  [green]✓[/green] Data downloaded to data/")

            # Find and unzip
            zip_files = list(data_dir.glob("*.zip"))
            if zip_files:
                zip_file = zip_files[0]
                console.print(f"  [cyan]Extracting {zip_file.name}...[/cyan]")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
                console.print(f"  [green]✓[/green] Data extracted")

                # Optionally remove zip
                if not args.keep_zip:
                    zip_file.unlink()
                    console.print(f"  [dim]Removed {zip_file.name}[/dim]")

        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error downloading data: {e.stderr}[/red]")
            console.print("[yellow]You can download data manually later with:[/yellow]")
            console.print(f"  cd {project_name}/data && kaggle competitions download -c {project_name}")
    else:
        console.print("\n[yellow]Skipping data download (--skip-download)[/yellow]")

    # Detect problem type and target from data if possible
    target_column = args.target_column
    problem_type = args.problem_type
    metric = args.metric

    # Try to detect from any *submission*.csv file
    sample_path = None
    submission_files = list((project_root / "data").glob("*submission*.csv"))
    if submission_files:
        sample_path = submission_files[0]

    if sample_path and not target_column:
        try:
            sample = pd.read_csv(sample_path, nrows=1)
            if len(sample.columns) >= 2:
                detected_target = sample.columns[1]
                console.print(f"\n[cyan]Detected target column: '{detected_target}' from {sample_path.name}[/cyan]")
                target_column = detected_target
        except Exception:
            pass

    # Try AI-based detection first
    if not problem_type or not metric:
        try:
            console.print(f"\n[cyan]Fetching competition details from Kaggle...[/cyan]")
            eval_text = fetch_kaggle_evaluation(project_name)

            if eval_text:
                console.print(f"[dim]Evaluation section: {eval_text[:100]}...[/dim]")
                console.print(f"[cyan]Asking AI to detect problem type and metric...[/cyan]")

                # Import AI helper
                sys.path.insert(0, str(Path(__file__).parent))
                from ai_helper import call_ai_json, AIError

                # Build prompt
                prompt = f"""You are a Kaggle competition expert analyzing evaluation metrics.

Given the Evaluation section from a Kaggle competition, determine:
1. problem_type: "binary", "regression", or "multiclass"
2. metric: AutoGluon-compatible metric name

EVALUATION SECTION:
{eval_text}

AUTOGLUON METRIC MAPPING (use exact names):
- AUC/ROC/Area Under Curve → "roc_auc"
- RMSE/Root Mean Squared Error → "root_mean_squared_error"
- MAE/Mean Absolute Error → "mean_absolute_error"
- Accuracy → "accuracy"
- Log Loss/Logarithmic Loss → "log_loss"
- F1 Score → "f1"
- Precision → "precision"
- Recall → "recall"

PROBLEM TYPE RULES:
- If predicting 0/1, True/False, or probability → "binary"
- If predicting continuous number → "regression"
- If predicting one of 3+ categories → "multiclass"

Return ONLY valid JSON (no markdown, no explanation):
{{"problem_type": "binary|regression|multiclass", "metric": "autogluon_metric_name"}}"""

                ai_result, model = call_ai_json(prompt, primary="gemini", retries=2)

                # Validate response
                if "problem_type" in ai_result and "metric" in ai_result:
                    detected_type = ai_result["problem_type"]
                    detected_metric = ai_result["metric"]

                    if detected_type in ["binary", "regression", "multiclass"]:
                        problem_type = problem_type or detected_type
                        metric = metric or detected_metric
                        console.print(f"[green]✓ AI detected ({model}): {problem_type} / {metric}[/green]")
                    else:
                        console.print(f"[yellow]AI returned invalid problem_type: {detected_type}[/yellow]")
                else:
                    console.print(f"[yellow]AI response missing required fields[/yellow]")

            else:
                console.print(f"[yellow]Could not fetch Evaluation section from Kaggle[/yellow]")

        except Exception as e:
            console.print(f"[yellow]AI detection failed: {e}[/yellow]")
            # Continue to interactive prompts

    # Interactive prompts if not provided (fallback)
    if not target_column:
        target_column = input("Target column name: ").strip() or "target"

    if not problem_type:
        console.print("\nProblem type:")
        console.print("  1. binary (binary classification)")
        console.print("  2. regression")
        console.print("  3. multiclass (multiclass classification)")
        choice = input("Choose (1/2/3): ").strip()
        problem_type = {"1": "binary", "2": "regression", "3": "multiclass"}.get(choice, "binary")

    if not metric:
        default_metrics = {
            "binary": "roc_auc",
            "regression": "mean_absolute_error",
            "multiclass": "accuracy"
        }
        metric = default_metrics.get(problem_type, "roc_auc")
        console.print(f"Using default metric for {problem_type}: {metric}")

    # Create config.py with customized values
    config_content = f'''"""
Configuration and constants for the competition
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CODE_DIR = PROJECT_ROOT / "code"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Data paths
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

# Model settings
RANDOM_SEED = 42
N_FOLDS = 5

# Target column
TARGET_COLUMN = "{target_column}"

# AutoGluon settings
AUTOGLUON_TIME_LIMIT = 600  # seconds (10 minutes)
AUTOGLUON_PRESET = "medium_quality"  # best_quality, high_quality, medium_quality, optimize_for_deployment
AUTOGLUON_PROBLEM_TYPE = "{problem_type}"  # binary, regression, multiclass
AUTOGLUON_EVAL_METRIC = "{metric}"  # evaluation metric

# Competition details
COMPETITION_NAME = "{project_name}"
METRIC = "{metric.replace('mean_absolute_error', 'mae').replace('root_mean_squared_error', 'rmse')}"
'''

    config_path = project_root / "code/utils/config.py"
    config_path.write_text(config_content)
    console.print(f"\n[green]✓[/green] Customized code/utils/config.py")

    # Customize baseline_autogluon.py
    baseline_path = project_root / "code/models/baseline_autogluon.py"
    baseline_content = baseline_path.read_text()
    baseline_content = baseline_content.replace(
        'cli_entry(default_project="playground-series-s5e11")',
        f'cli_entry(default_project="{project_name}")'
    )
    baseline_path.write_text(baseline_content)
    console.print(f"[green]✓[/green] Customized code/models/baseline_autogluon.py")

    # Customize README.md
    readme_path = project_root / "README.md"
    readme_content = readme_path.read_text()
    readme_content = readme_content.replace("playground-series-s5e11", project_name)
    readme_content = readme_content.replace(
        "Area under the ROC curve",
        f"{metric} ({'lower is better' if 'error' in metric or 'loss' in metric else 'higher is better'})"
    )
    readme_path.write_text(readme_content)
    console.print(f"[green]✓[/green] Customized README.md")

    # Print summary
    console.print("\n" + "="*60)
    console.print(f"[green bold]✓ Project '{project_name}' initialized successfully![/green bold]")
    console.print("="*60)

    table = Table(title="Project Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Project Name", project_name)
    table.add_row("Target Column", target_column)
    table.add_row("Problem Type", problem_type)
    table.add_row("Metric", metric)
    table.add_row("Location", str(project_root))

    console.print(table)

    console.print("\n[cyan]Next steps:[/cyan]")
    console.print(f"  1. Review configuration: {project_root}/code/utils/config.py")
    console.print(f"  2. Run EDA: uv run python scripts/experiment_manager.py eda --project {project_name}")
    console.print(f"  3. Train baseline: uv run python scripts/experiment_manager.py model --project {project_name} --experiment-id <EXP_ID> --template dev-gpu")


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

    init_parser = subparsers.add_parser("init-project", help="Initialize new competition project")
    init_parser.add_argument("--project", required=True, help="Competition slug from Kaggle")
    init_parser.add_argument("--migrate", action="store_true", help="Migrate old project to new structure")
    init_parser.add_argument("--target-column", help="Target column name")
    init_parser.add_argument("--problem-type", choices=["binary", "regression", "multiclass"], help="Problem type")
    init_parser.add_argument("--metric", help="Evaluation metric")
    init_parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    init_parser.add_argument("--keep-zip", action="store_true", help="Keep zip file after extraction")
    init_parser.set_defaults(func=run_init_project)

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
    try:
        args.func(args)
    except ModuleStateError as exc:
        print(f"[Experiment] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Project bootstrap helper for MLArena `init` command.

Replicates the legacy `init-project` flow: creates competition folder
structure, copies starter code/templates, optional Kaggle data download,
and leaves `.gitkeep` breadcrumbs. Designed to be idempotent with
`--force` when re-running.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_ROOT = REPO_ROOT / "config" / "templates"
CODE_TEMPLATE_ROOT = REPO_ROOT / "config" / "code"
CONFIG_TEMPLATE_ROOT = REPO_ROOT / "config" / "configs"
GITIGNORE_TEMPLATE = REPO_ROOT / "config" / ".gitignore"


def _log(msg: str) -> None:
    print(msg)


def _copy_file(src: Path, dest: Path, *, force: bool) -> bool:
    """Copy a single file if it does not exist or when force=True."""
    if not src.exists():
        return False
    if dest.exists() and not force:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _copy_tree(src: Path, dest: Path, *, force: bool) -> bool:
    """Copy a directory tree with ignore rules; idempotent with dirs_exist_ok."""
    if not src.exists():
        return False
    if dest.exists() and not force:
        return True
    shutil.copytree(
        src,
        dest,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    return True


def _render_placeholders(text: str, replacements: Dict[str, str]) -> str:
    rendered = text
    for key, val in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(val))
    return rendered


def _csv_header(path: Path) -> Optional[list[str]]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        return None


def _csv_shape(path: Path) -> Optional[tuple[int, int]]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df.shape
    except Exception:
        return None


def _analyze_target(train_path: Path, target_col: str) -> Tuple[str, str, str, str, bool, str]:
    """Infer problem type, metric, direction, class balance, submission mode."""
    if not train_path.exists():
        return "binary", "roc_auc", "ROC AUC", "maximize", True, "TBD"

    df = pd.read_csv(train_path)
    if target_col not in df.columns:
        return "binary", "roc_auc", "ROC AUC", "maximize", True, "TBD"

    target_series = df[target_col]
    nunique = target_series.nunique(dropna=True)
    is_numeric = pd.api.types.is_numeric_dtype(target_series)

    if not is_numeric:
        problem = "multiclass" if nunique > 2 else "binary"
        metric_name = "accuracy"
        metric_label = "Accuracy"
        direction = "maximize"
        submission_probas = False
    else:
        if nunique <= 2:
            problem = "binary"
            metric_name = "roc_auc"
            metric_label = "ROC AUC"
            direction = "maximize"
            submission_probas = True
        elif nunique <= 50:
            problem = "multiclass"
            metric_name = "accuracy"
            metric_label = "Accuracy"
            direction = "maximize"
            submission_probas = False
        else:
            problem = "regression"
            metric_name = "rmse"
            metric_label = "RMSE"
            direction = "minimize"
            submission_probas = False

    vc = target_series.value_counts().head(5)
    balance = " | ".join([f"{idx} {cnt}" for idx, cnt in vc.items()]) if not vc.empty else "TBD"
    return problem, metric_name, metric_label, direction, submission_probas, balance


def _detect_dataset_stats(project_root: Path, competition_slug: str) -> Dict[str, str]:
    data_dir = project_root / "data"
    sample_path = next((p for p in data_dir.glob("*submission*.csv")), None)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"

    sample_cols = _csv_header(sample_path) if sample_path else None
    train_shape = _csv_shape(train_path)
    test_shape = _csv_shape(test_path)

    id_col = sample_cols[0] if sample_cols else "id"
    target_col = sample_cols[1] if sample_cols and len(sample_cols) > 1 else "target"

    problem, metric_name, metric_label, metric_dir, submission_probas, balance = _analyze_target(train_path, target_col)

    stats = {
        "COMPETITION_NAME": competition_slug,
        "METRIC_LABEL": metric_label,
        "METRIC_NAME": metric_name,
        "METRIC_DIRECTION": metric_dir,
        "TARGET_COLUMN": target_col,
        "ID_COLUMN": id_col,
        "TRAIN_ROWS": str(train_shape[0]) if train_shape else "?",
        "TRAIN_COLS": str(train_shape[1]) if train_shape else "?",
        "TEST_ROWS": str(test_shape[0]) if test_shape else "?",
        "TEST_COLS": str(test_shape[1]) if test_shape else "?",
        "SAMPLE_COLUMNS": ", ".join(sample_cols) if sample_cols else "?",
        "CLASS_SUMMARY": balance,
        "PROBLEM_TYPE": problem,
        "SUBMISSION_PROBAS": submission_probas,
        "METRIC_NAME_RAW": metric_name,
    }
    return stats


def _render_config_py(
    template_path: Path,
    competition_slug: str,
    id_column: str,
    target_column: str,
    problem_type: str,
    metric_name: str,
    submission_probas: bool,
) -> str:
    """Render config.py using detected dataset settings when available."""
    lines = template_path.read_text().splitlines()
    rendered: list[str] = []

    for line in lines:
        if line.startswith("TARGET_COLUMN"):
            rendered.append(f'TARGET_COLUMN = "{target_column}"')
        elif line.startswith("ID_COLUMN"):
            rendered.append(f'ID_COLUMN = "{id_column}"')
        elif line.startswith("COMPETITION_NAME"):
            rendered.append(f'COMPETITION_NAME = "{competition_slug}"')
        elif line.startswith("DEADLINE"):
            rendered.append('DEADLINE = ""')
        elif line.startswith("AUTOGLUON_PROBLEM_TYPE"):
            rendered.append(f'AUTOGLUON_PROBLEM_TYPE = "{problem_type}"')
        elif line.startswith("AUTOGLUON_EVAL_METRIC"):
            rendered.append(f'AUTOGLUON_EVAL_METRIC = "{metric_name}"')
        elif line.startswith("METRIC"):
            rendered.append(f'METRIC = "{metric_name}"')
        elif line.startswith("SUBMISSION_PROBAS"):
            rendered.append(f"SUBMISSION_PROBAS = {str(submission_probas)}")
        else:
            rendered.append(line)

    return "\n".join(rendered) + "\n"


def _download_competition(slug: str, data_dir: Path) -> bool:
    """Download and unzip Kaggle competition data; returns True on success."""
    if shutil.which("kaggle") is None:
        _log("[error] kaggle CLI not found. Install `pip install kaggle` or use --skip-download.")
        return False

    data_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[info] Downloading competition data: {slug}")
    download = subprocess.run(
        ["kaggle", "competitions", "download", "-c", slug, "-p", str(data_dir)],
        capture_output=True,
        text=True,
    )
    if download.returncode != 0:
        _log(f"[error] kaggle download failed: {download.stderr.strip()}")
        return False

    zip_path = data_dir / f"{slug}.zip"
    if not zip_path.exists():
        _log(f"[warn] Expected archive {zip_path.name} not found; skipping unzip")
        return True

    _log(f"[info] Unzipping {zip_path.name}")
    unzip = subprocess.run(["unzip", "-o", zip_path.name], cwd=data_dir, capture_output=True, text=True)
    if unzip.returncode != 0:
        _log(f"[error] unzip failed: {unzip.stderr.strip()}")
        return False

    return True


def init_project(
    *,
    project_root: Path,
    competition_slug: str,
    skip_download: bool = False,
    force: bool = False,
) -> Dict[str, Any]:
    """Initialize a Kaggle competition project (legacy init-project parity).

    Returns dict with keys:
      - success: bool
      - stats: detected dataset stats (if available)
      - error: optional error message
    """

    if project_root.exists() and any(project_root.iterdir()) and not force:
        msg = f"Project {project_root} already exists. Use --force/--migrate to overwrite."
        _log(f"[error] {msg}")
        return {"success": False, "error": msg}

    _log(f"[info] Creating project at {project_root}")
    project_root.mkdir(parents=True, exist_ok=True)

    # Core directories
    dirs = [
        project_root / "data",
        project_root / "submissions",
        project_root / "experiments",
        project_root / "code" / "utils",
        project_root / "code" / "models",
        project_root / "code" / "preprocessing",
        project_root / "code" / "exploration",
        project_root / "templates",
        project_root / "configs",
        project_root / "docs",
        project_root / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    for keep_dir in [project_root / "data", project_root / "submissions", project_root / "experiments"]:
        (keep_dir / ".gitkeep").touch(exist_ok=True)

    _copy_file(GITIGNORE_TEMPLATE, project_root / ".gitignore", force=force)

    # Ensure templates dir exists (user-owned overrides live here)
    tmpl_dir = project_root / "templates"
    tmpl_dir.mkdir(parents=True, exist_ok=True)

    # Leave code/{models,preprocessing,exploration} empty for user; add .gitkeep
    for empty_dir in [
        project_root / "code" / "models",
        project_root / "code" / "preprocessing",
        project_root / "code" / "exploration",
    ]:
        empty_dir.mkdir(parents=True, exist_ok=True)
        (empty_dir / ".gitkeep").touch(exist_ok=True)

    _copy_tree(CONFIG_TEMPLATE_ROOT, project_root / "configs", force=force)

    if not skip_download:
        ok = _download_competition(competition_slug, project_root / "data")
        if not ok:
            return {"success": False, "error": "kaggle_download_failed"}

    stats = _detect_dataset_stats(project_root, competition_slug)

    utils_template = CODE_TEMPLATE_ROOT / "utils"
    _copy_tree(utils_template, project_root / "code" / "utils", force=force)
    config_py = project_root / "code" / "utils" / "config.py"
    if not config_py.exists() or force:
        if (utils_template / "config.py").exists():
            config_py.write_text(
                _render_config_py(
                    utils_template / "config.py",
                    competition_slug,
                    stats["ID_COLUMN"],
                    stats["TARGET_COLUMN"],
                    stats.get("PROBLEM_TYPE", "binary"),
                    stats.get("METRIC_NAME_RAW", "roc_auc"),
                    stats.get("SUBMISSION_PROBAS", False),
                )
            )

    _copy_file(utils_template / "submission.py", project_root / "code" / "utils" / "submission.py", force=force)

    # README template (render with detected stats)
    readme_tpl = TEMPLATE_ROOT / "kaggle_competition" / "README.md"
    if readme_tpl.exists():
        readme_dst = project_root / "README.md"
        if not readme_dst.exists() or force:
            rendered_readme = _render_placeholders(readme_tpl.read_text(), stats)
            readme_dst.write_text(rendered_readme)

    _log("[done] Project initialized. Next steps: update code/utils/config.py and run mla eda/model.")
    return {"success": True, "stats": stats}


__all__ = ["init_project"]

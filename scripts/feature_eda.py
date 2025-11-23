"""Generate EDA reports on feature-engineered datasets."""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from ydata_profiling import ProfileReport

from kaggle_tools.config_models import DatasetConfig, Hyperparameters, ModelConfig, SystemConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_experiment_id() -> str:
    return datetime.now(timezone.utc).strftime("exp-%Y%m%d-%H%M%S-fe")


def _sanitize_profile(payload: Any) -> Any:
    remove = {
        "value_counts_without_nan",
        "value_counts_index_sorted",
        "histogram",
        "histogram_length",
        "character_counts",
        "package",
        "analysis",
    }
    if isinstance(payload, dict):
        cleaned = {}
        for key, value in payload.items():
            if key in remove:
                continue
            cleaned[key] = _sanitize_profile(value)
        return cleaned
    if isinstance(payload, list):
        return [_sanitize_profile(item) for item in payload]
    return payload


def load_model_module(project_root: Path, model_name: str):
    code_dir = project_root / "code"
    import sys

    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    module = importlib.import_module(f"models.{model_name}")
    return module


def build_model_config(project_root: Path, cfg_module, args) -> ModelConfig:
    dataset = DatasetConfig(
        train_path=cfg_module.TRAIN_PATH,
        test_path=cfg_module.TEST_PATH,
        target=cfg_module.TARGET_COLUMN,
        id_column=getattr(cfg_module, "ID_COLUMN", "id"),
        ignored_columns=list(getattr(cfg_module, "IGNORED_COLUMNS", [])),
        sample_submission_path=cfg_module.SAMPLE_SUBMISSION_PATH,
        problem_type=getattr(cfg_module, "AUTOGLUON_PROBLEM_TYPE", None),
        metric=getattr(cfg_module, "AUTOGLUON_EVAL_METRIC", None),
        submission_probas=getattr(cfg_module, "SUBMISSION_PROBAS", False),
    )
    system = SystemConfig(
        project_root=project_root,
        code_dir=project_root / "code",
        experiment_dir=project_root / "experiments" / args.experiment_id,
        artifact_dir=project_root / "experiments" / args.experiment_id / "artifacts",
        model_path=project_root / "experiments" / args.experiment_id / "artifacts" / args.model,
        template=args.template,
        experiment_id=args.experiment_id,
        random_seed=getattr(cfg_module, "RANDOM_SEED", 42),
        use_gpu=False,
    )
    hyper = Hyperparameters(presets="medium_quality", time_limit=0, use_gpu=False)
    return ModelConfig(system=system, dataset=dataset, hyperparameters=hyper)


def run_feature_eda(args: argparse.Namespace):
    project_root = REPO_ROOT / "projects" / "kaggle" / args.project
    project_root.mkdir(parents=True, exist_ok=True)

    code_dir = project_root / "code"
    import sys

    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    cfg_module = importlib.import_module("utils.config")

    module = load_model_module(project_root, args.model)
    experiment_id = args.experiment_id or generate_experiment_id()
    args.experiment_id = experiment_id
    out_dir = project_root / "experiments" / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = build_model_config(project_root, cfg_module, args)

    train_df = pd.read_csv(cfg_module.TRAIN_PATH)
    test_df = pd.read_csv(cfg_module.TEST_PATH)

    if hasattr(module, "preprocess"):
        train_processed = module.preprocess(train_df, config, is_train=True)
        test_processed = module.preprocess(test_df, config, is_train=False)
    else:
        train_processed, test_processed = train_df, test_df

    feature_eda_dir = out_dir / "feature_eda"
    feature_eda_dir.mkdir(parents=True, exist_ok=True)

    def write_report(df: pd.DataFrame, name: str):
        report = ProfileReport(
            df,
            title=f"{args.project} - {name} (feature engineered)",
            minimal=False,
            infer_dtypes=True,
            progress_bar=False,
        )
        html_path = feature_eda_dir / f"{name}.html"
        json_path = feature_eda_dir / f"{name}.json"
        report.to_file(str(html_path))
        report_json = json.loads(report.to_json())
        json_path.write_text(json.dumps(report_json, indent=2))
        trimmed = _sanitize_profile(report_json)
        (feature_eda_dir / f"{name}_min.json").write_text(json.dumps(trimmed, indent=2))
        return {
            "columns": list(df.columns),
            "shape": list(df.shape),
            "html": str(html_path.relative_to(project_root)),
            "json": str(json_path.relative_to(project_root)),
            "json_min": str((feature_eda_dir / f"{name}_min.json").relative_to(project_root)),
        }

    train_meta = write_report(train_processed, "train")
    test_meta = write_report(test_processed, "test")

    summary = {
        "experiment_id": experiment_id,
        "project": args.project,
        "model": args.model,
        "created_at": utc_now(),
        "profiles": {"train": train_meta, "test": test_meta},
    }
    (feature_eda_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Feature EDA stored under {feature_eda_dir.relative_to(project_root)}")


def main():
    parser = argparse.ArgumentParser(description="Run feature-engineered EDA")
    parser.add_argument("--project", required=True)
    parser.add_argument("--model", default="autogluon_eda_features")
    parser.add_argument("--experiment-id")
    parser.add_argument("--template", default="feature-eda")
    args = parser.parse_args()
    run_feature_eda(args)


if __name__ == "__main__":
    main()

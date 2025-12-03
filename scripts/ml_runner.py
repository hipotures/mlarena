"""
Generic ML runner that separates modeling code from infrastructure.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml
from rich.console import Console
from rich.panel import Panel

from experiment_manager import ExperimentManager, ModuleStateError
from kaggle_tools.config_models import DatasetConfig, Hyperparameters, ModelConfig, SystemConfig
from submission_workflow import SubmissionRunner

REPO_ROOT = Path(__file__).resolve().parent.parent
console = Console()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ProjectContext:
    name: str
    root: Path
    config_module: Any
    submission_module: Any


def load_project_context(project_name: str) -> ProjectContext:
    project_root = (REPO_ROOT / "projects" / "kaggle" / project_name).resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"Project directory '{project_name}' not found at {project_root}")

    code_dir = project_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    config_module = importlib.import_module("utils.config")
    submission_module = importlib.import_module("utils.submission")
    return ProjectContext(
        name=project_name,
        root=project_root,
        config_module=config_module,
        submission_module=submission_module,
    )


def parse_args(default_project: Optional[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run template-driven ML models")
    parser.add_argument("--project", default=default_project, required=True)
    parser.add_argument("--model-template", help="Model template name defined in templates/model.yaml")
    parser.add_argument("--preprocess-template", default="identity", help="Preprocess template name defined in templates/preprocess.yaml")
    parser.add_argument("--use-preprocessed", action="store_true", help="(Deprecated) use cached preprocessing if available")
    parser.add_argument("--recompute-preprocess", action="store_true", help="Force rebuild preprocessing cache, ignoring existing cache")
    parser.add_argument("--template", help="Deprecated: use --model-template instead")
    parser.add_argument(
        "--stage",
        choices=["all", "train", "predict", "preprocess"],
        default="all",
        help="Pipeline stage to run: preprocess only, train, predict, or both (default).",
    )
    parser.add_argument("--experiment-id")
    parser.add_argument("--time-limit", type=int)
    parser.add_argument("--preset")
    parser.add_argument("--use-gpu", type=int, choices=[0, 1])
    parser.add_argument("--force-extreme", action="store_true", help="Deprecated compatibility flag")
    parser.add_argument("--ag-smoke", action="store_true", help="AutoGluon smoke test mode")
    parser.add_argument("--skip-submit", action="store_true")
    parser.add_argument("--auto-submit", action="store_true")
    parser.add_argument("--skip-score-fetch", action="store_true")
    parser.add_argument("--skip-git", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=30)
    parser.add_argument("--cdp-url", default="http://localhost:9222")
    parser.add_argument("--require-eda", action="store_true")
    parser.add_argument("--skip-eda-check", action="store_true")
    parser.add_argument("--kaggle-message")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun of modules for the selected stage (clears module state if completed).",
    )
    parser.add_argument(
        "--force-predict",
        action="store_true",
        help="Allow rerunning predict module even if already completed for this experiment.",
    )
    return parser.parse_args()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _load_template_file(path: Path, kind: str) -> Dict[str, Any]:
    data = _load_yaml(path)
    templates = data.get("templates", {})
    if not templates:
        raise RuntimeError(f"No {kind} templates defined in {path}")
    return templates


def _apply_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    hyper = overrides.setdefault("hyperparameters", {})

    # --ag-smoke FORCE overrides (highest priority)
    if args.ag_smoke:
        hyper["time_limit"] = 300
        hyper["presets"] = "medium"
        # Ignore other CLI args when --ag-smoke is active
    else:
        # Normal CLI overrides
        if args.time_limit is not None:
            hyper["time_limit"] = args.time_limit
        if args.preset is not None:
            hyper["presets"] = args.preset

    # GPU setting applies regardless of --ag-smoke
    if args.use_gpu is not None:
        hyper["use_gpu"] = bool(args.use_gpu)

    return overrides


class MLRunner:
    def __init__(
        self,
        project: ProjectContext,
        manager: ExperimentManager,
        model_template_name: Optional[str],
        model_template_payload: Optional[Dict[str, Any]],
        args: argparse.Namespace,
        *,
        stage: str = "all",
        config_override: Optional[ModelConfig] = None,
        model_name_override: Optional[str] = None,
        training_summary_override: Optional[Dict[str, Any]] = None,
        snapshot_override: Optional[Path] = None,
        preprocess_template_name: str = "identity",
        preprocess_template_payload: Optional[Dict[str, Any]] = None,
        preprocess_config_override: Optional[Dict[str, Any]] = None,
    ):
        self.project = project
        self.manager = manager
        self.template_name = model_template_name
        self.template_payload = model_template_payload or {}
        self.args = args
        self.stage = stage
        self.model_name = model_name_override or self.template_payload.get("model")

        self.experiment_dir = self.project.root / "experiments" / self.manager.experiment_id
        self.artifact_dir = self.experiment_dir / "artifacts"
        self.dataset_config = self._build_dataset_config()

        self.config = None
        self.model_module = None
        if self.stage != "preprocess":
            if not self.model_name:
                raise ValueError("Model name is required to run the ML pipeline.")
            self.model_module = self._load_model_module()
            if config_override is not None:
                self.config = config_override
                self.dataset_config = self.config.dataset
                self.experiment_dir = self.config.system.experiment_dir
                self.artifact_dir = self.config.system.artifact_dir
            else:
                self.config = self._build_model_config()

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.training_summary_override = training_summary_override or {}
        self.snapshot_override = snapshot_override

        self.preprocess_template_name = preprocess_template_name
        self.preprocess_payload = preprocess_template_payload or {"module": "identity", "config": {}}
        self.preprocess_config = preprocess_config_override or (self.preprocess_payload.get("config") or {})
        self.preprocess_module_name = self.preprocess_payload.get("module", "identity")
        self.features_dir: Optional[Path] = None
        self.preprocess_cache_sig: Optional[str] = None
        self.preprocess_module = self._load_preprocess_module()
        self.preprocess_record: Dict[str, Any] = {}

        # Validate --ag-smoke is used only with AutoGluon
        if args.ag_smoke:
            if not self.model_name.startswith("autogluon"):
                raise ValueError(
                    f"--ag-smoke can only be used with AutoGluon models. "
                    f"Template '{self.template_name}' uses model '{self.model_name}'. "
                    f"Use --ag-smoke only with autogluon_* models."
                )
            console.print("[yellow]--ag-smoke mode activated for AutoGluon[/yellow]")

    def _load_model_module(self):
        model_path = self.project.root / "code" / "models" / f"{self.model_name}.py"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        spec = importlib.util.spec_from_file_location(self.model_name, model_path)
        module = importlib.util.module_from_spec(spec)
        if not spec.loader:
            raise RuntimeError(f"Unable to load module spec for {model_path}")
        spec.loader.exec_module(module)
        return module

    def _load_preprocess_module(self):
        module_path = self.project.root / "code" / "preprocessing" / f"{self.preprocess_module_name}.py"
        if not module_path.exists():
            raise FileNotFoundError(f"Preprocess file not found: {module_path}")
        spec = importlib.util.spec_from_file_location(self.preprocess_module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        if not spec.loader:
            raise RuntimeError(f"Unable to load module spec for {module_path}")
        spec.loader.exec_module(module)
        return module

    def _build_dataset_config(self) -> DatasetConfig:
        cfg = self.project.config_module
        ignored_columns = list(getattr(cfg, "IGNORED_COLUMNS", []))
        id_column = getattr(cfg, "ID_COLUMN", None)
        if not id_column:
            if ignored_columns:
                id_column = ignored_columns[0]
            else:
                id_column = "id"
        return DatasetConfig(
            train_path=cfg.TRAIN_PATH,
            test_path=cfg.TEST_PATH,
            target=getattr(cfg, "TARGET_COLUMN"),
            id_column=id_column,
            ignored_columns=ignored_columns,
            sample_submission_path=cfg.SAMPLE_SUBMISSION_PATH,
            problem_type=getattr(cfg, "AUTOGLUON_PROBLEM_TYPE", None),
            metric=getattr(cfg, "AUTOGLUON_EVAL_METRIC", None),
            submission_probas=getattr(cfg, "SUBMISSION_PROBAS", True),
        )

    def _build_model_config(self) -> ModelConfig:
        defaults = {}
        default_fn = getattr(self.model_module, "get_default_config", None)
        if callable(default_fn):
            defaults = default_fn() or {}
        project_config = _load_yaml(self.project.root / "configs" / "project.yaml")
        template_config = self.template_payload.get("config", {})
        cli_overrides = _apply_cli_overrides(self.args)

        merged = _deep_merge(defaults, project_config)
        merged = _deep_merge(merged, template_config)
        merged = _deep_merge(merged, cli_overrides)

        # Avoid passing duplicate keys that conflict with explicit ModelConfig args
        merged.pop("system", None)
        merged.pop("dataset", None)

        hyper_payload = merged.pop("hyperparameters", {})
        model_payload = merged.pop("model", {})

        cfg = self.project.config_module
        system = SystemConfig(
            project_root=self.project.root,
            code_dir=self.project.root / "code",
            experiment_dir=self.experiment_dir,
            artifact_dir=self.artifact_dir,
            model_path=self.artifact_dir / f"{self.template_name}-{self.model_name}",
            template=self.template_name,
            experiment_id=self.manager.experiment_id,
            random_seed=getattr(cfg, "RANDOM_SEED", 42),
            use_gpu=bool(hyper_payload.get("use_gpu")),
        )

        hyperparameters = Hyperparameters(**hyper_payload)
        return ModelConfig(
            system=system,
            dataset=self.dataset_config,
            hyperparameters=hyperparameters,
            model=model_payload,
            **merged,
        )

    def execute(self) -> Dict[str, Any]:
        console.print(
            Panel.fit(
                f"[bold magenta]{self.project.name}[/bold magenta]\n"
                f"Template: {self.template_name}\n"
                f"Preprocess: {self.preprocess_template_name}\n"
                f"Model: {self.model_name}\n"
                f"Stage: {self.stage}",
                title="ML Runner",
            )
        )

        train_df, val_df, test_df = self._load_data()
        if self.stage in {"all", "train", "preprocess"}:
            try:
                self.manager.start_module(
                    "preprocess",
                    {"template": self.preprocess_template_name, "module": self.preprocess_module_name},
                    allow_restart=True,
                )
            except ModuleStateError as exc:
                console.print(f"[yellow]{exc}[/yellow]")
                if self.stage == "preprocess":
                    return {}
                if self.stage in {"all", "train"}:
                    raise
            try:
                train_df, val_df, test_df = self._run_preprocess(train_df, val_df, test_df)
                if self.preprocess_record:
                    self.manager.complete_module("preprocess", self.preprocess_record)
            except Exception as exc:
                self.manager.fail_module("preprocess", str(exc))
                raise
            if self.stage == "preprocess":
                return {"preprocess": self.preprocess_record}
        else:
            train_df, val_df, test_df = self._run_preprocess(train_df, val_df, test_df)
        artifacts = self._prepare_artifacts(train_df)

        model = None
        training_summary: Dict[str, Any] = {}
        snapshot_path: Optional[Path] = None
        submission = None

        if self.stage in {"all", "train"}:
            model, training_summary, snapshot_path = self._run_training(train_df, val_df, artifacts)
        elif self.stage == "predict":
            training_summary = self._load_training_summary_from_state()
            snapshot_path = self.snapshot_override
            model = self._load_trained_model()
        else:  # preprocess-only
            console.print("[yellow]Stage set to 'preprocess' → skipping model/prediction.[/yellow]")
            return {"preprocess": self.preprocess_record}

        if self.stage in {"all", "predict"}:
            submission = self._run_predict(model, test_df, artifacts, training_summary)
        else:
            console.print("[yellow]Stage set to 'train' → skipping prediction/submission.[/yellow]")

        return {
            "submission": submission,
            "training_summary": training_summary,
            "snapshot_path": snapshot_path,
            "train_rows": len(train_df),
        }

    def _load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        train_df = pd.read_csv(self.dataset_config.train_path)
        test_df = pd.read_csv(self.dataset_config.test_path)

        val_path = getattr(self.project.config_module, "VALIDATION_PATH", None)
        if val_path:
            val_df = pd.read_csv(val_path)
        else:
            val_df = None

        console.print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        if self.args.ag_smoke:
            console.print("[yellow]--ag-smoke active: medium preset, 300s time limit[/yellow]")

        return train_df, val_df, test_df

    def _run_preprocess(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        """
        Run preprocessing template with caching support.
        """
        id_col = self.dataset_config.id_column

        def _ensure_id(df: Optional[pd.DataFrame], raw_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None or not id_col:
                return df
            if id_col in df.columns:
                return df
            if raw_df is None or id_col not in raw_df.columns:
                return df
            out = df.copy()
            out[id_col] = raw_df[id_col].reset_index(drop=True)
            return out

        preprocess_config = self._prepare_preprocess_config()
        self.features_dir, self.preprocess_cache_sig = self._compute_features_dir(preprocess_config)
        paths = self._feature_paths()
        use_cache = not self.args.recompute_preprocess

        if use_cache:
            cached = self._load_preprocess_cache(paths)
            if cached:
                train_fe, val_fe, test_fe, state_dict, meta = cached
                train_fe = _ensure_id(train_fe, train_df)
                val_fe = _ensure_id(val_fe, val_df)
                test_fe = _ensure_id(test_fe, test_df)
                self.preprocess_record = self._build_preprocess_record(
                    paths, state_dict, meta, used_cached=True, resolved_config=preprocess_config, cache_sig=self.preprocess_cache_sig or ""
                )
                console.print("[cyan]Loaded cached preprocessed features.[/cyan]")
                return train_fe, val_fe, test_fe
            if self.stage == "predict":
                raise RuntimeError("Preprocessed features not found; cannot run predict without cache. Rerun training first.")

        fit_fn = getattr(self.preprocess_module, "fit_transform", None)
        if not callable(fit_fn):
            raise RuntimeError(f"Preprocess module '{self.preprocess_module_name}' missing fit_transform()")

        console.print(f"[cyan]Running preprocessing template: {self.preprocess_template_name}[/cyan]")
        train_fe, val_fe, test_fe, state_dict = fit_fn(train_df, val_df, test_df, preprocess_config)
        train_fe = _ensure_id(train_fe, train_df)
        val_fe = _ensure_id(val_fe, val_df)
        test_fe = _ensure_id(test_fe, test_df)
        meta = {
            "template": self.preprocess_template_name,
            "module": self.preprocess_module_name,
            "config": self.preprocess_config,
            "saved_at": utc_now(),
            "shapes": {
                "train": list(train_fe.shape),
                "val": list(val_fe.shape) if val_fe is not None else None,
                "test": list(test_fe.shape),
            },
            "state_keys": sorted(list(state_dict.keys())) if isinstance(state_dict, dict) else [],
            "drop_summary": state_dict.get("drop_summary", {}) if isinstance(state_dict, dict) else {},
        }
        self._save_preprocess_cache(paths, train_fe, val_fe, test_fe, state_dict, meta)
        self.preprocess_record = self._build_preprocess_record(
            paths, state_dict, meta, used_cached=False, resolved_config=preprocess_config, cache_sig=self.preprocess_cache_sig or ""
        )
        drop_info = meta.get("drop_summary") or state_dict.get("drop_summary") if isinstance(state_dict, dict) else {}
        console.print(
            f"[cyan]Preprocessed shapes[/cyan]: "
            f"train={meta['shapes']['train']}, "
            f"val={meta['shapes']['val']}, "
            f"test={meta['shapes']['test']}"
        )
        if drop_info:
            console.print(
                f"[cyan]Feature selection[/cyan]: "
                f"before={drop_info.get('before','?')} "
                f"const_dropped={drop_info.get('const_dropped','?')} "
                f"corr_dropped={drop_info.get('corr_dropped','?')}"
            )
        return train_fe, val_fe, test_fe

    def _feature_paths(self) -> Dict[str, Path]:
        if not self.features_dir:
            raise RuntimeError("features_dir not initialized; call _compute_features_dir first")
        return {
            "train": self.features_dir / "train_fe.parquet",
            "val": self.features_dir / "val_fe.parquet",
            "test": self.features_dir / "test_fe.parquet",
            "state": self.features_dir / "state.pkl",
            "meta": self.features_dir / "state.meta.json",
        }

    def _prepare_preprocess_config(self) -> Dict[str, Any]:
        cfg = json.loads(json.dumps(self.preprocess_config)) if self.preprocess_config else {}
        cfg["_dataset"] = {
            "target": self.dataset_config.target,
            "id_column": self.dataset_config.id_column,
            "ignored_columns": self.dataset_config.ignored_columns,
        }
        return cfg

    def _compute_features_dir(self, resolved_config: Dict[str, Any]) -> Tuple[Path, str]:
        payload = {"template": self.preprocess_template_name, "config": resolved_config}
        sig = sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
        path = self.project.root / "preprocess_cache" / self.preprocess_template_name / sig
        path.mkdir(parents=True, exist_ok=True)
        return path, sig

    def _save_preprocess_cache(
        self,
        paths: Dict[str, Path],
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        test_df: pd.DataFrame,
        state_dict: Dict[str, Any],
        meta: Dict[str, Any],
    ):
        train_df.to_parquet(paths["train"])
        if val_df is not None:
            val_df.to_parquet(paths["val"])
        test_df.to_parquet(paths["test"])
        with open(paths["state"], "wb") as f:
            pickle.dump(state_dict, f)
        paths["meta"].write_text(json.dumps(meta, indent=2))

    def _load_preprocess_cache(
        self, paths: Dict[str, Path]
    ) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, Dict[str, Any], Dict[str, Any]]]:
        required = [paths["train"], paths["test"], paths["state"]]
        if not all(p.exists() for p in required):
            return None
        train_df = pd.read_parquet(paths["train"])
        val_df = pd.read_parquet(paths["val"]) if paths["val"].exists() else None
        test_df = pd.read_parquet(paths["test"])
        with open(paths["state"], "rb") as f:
            state = pickle.load(f)
        meta = json.loads(paths["meta"].read_text()) if paths["meta"].exists() else {}
        return train_df, val_df, test_df, state, meta

    def _build_preprocess_record(
        self,
        paths: Dict[str, Path],
        state: Dict[str, Any],
        meta: Dict[str, Any],
        used_cached: bool,
        resolved_config: Dict[str, Any],
        cache_sig: str,
    ):
        return {
            "template": self.preprocess_template_name,
            "module": self.preprocess_module_name,
            "config": self.preprocess_config,
            "resolved_config": resolved_config,
            "used_cached": used_cached,
            "cache": {
                "train": self._relative_to_project(paths["train"]),
                "val": self._relative_to_project(paths["val"]) if paths["val"].exists() else None,
                "test": self._relative_to_project(paths["test"]),
                "state": self._relative_to_project(paths["state"]),
                "meta": self._relative_to_project(paths["meta"]),
            },
            "cache_sig": cache_sig,
            "state_keys": sorted(list(state.keys())) if isinstance(state, dict) else [],
            "meta": meta,
        }

    def _prepare_artifacts(self, train_df: pd.DataFrame):
        prepare_fn = getattr(self.model_module, "prepare_artifacts", None)
        if callable(prepare_fn):
            console.print("[cyan]Preparing artifacts...[/cyan]")
            return prepare_fn(train_df, self.config)
        return None

    def _train_model(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        artifacts,
    ) -> Tuple[Any, Dict[str, Any]]:
        console.print("[green]Training model...[/green]")
        train_result = self.model_module.train(train_df, val_df, self.config, artifacts=artifacts)
        if isinstance(train_result, tuple) and len(train_result) == 2 and isinstance(train_result[1], dict):
            model, summary = train_result
        else:
            model, summary = train_result, {}
        return model, summary

    def _run_predictions(self, model: Any, test_df: pd.DataFrame, artifacts) -> pd.DataFrame:
        console.print("[green]Generating predictions...[/green]")
        predictions = self.model_module.predict(model, test_df, self.config, artifacts=artifacts)
        if not isinstance(predictions, pd.DataFrame):
            raise TypeError("predict() must return a pandas DataFrame with submission columns.")
        id_col = self.dataset_config.id_column
        if id_col not in predictions.columns:
            raise ValueError(f"Predictions DataFrame must include ID column '{id_col}'.")
        return predictions

    def _save_submission(self, predictions: pd.DataFrame, training_summary: Dict[str, Any]):
        submission = self.project.submission_module.create_submission(
            predictions=predictions,
            test_ids=None,
            model_name=f"{self.model_name}-{self.template_name}",
            local_cv_score=training_summary.get("local_cv"),
            notes=f"template={self.template_name}; preprocess={self.preprocess_template_name}",
            config=self.config.as_plain_dict(),
            track=False,
        )
        console.print(f"[bold green]Submission saved:[/bold green] {submission.path}")
        return submission

    def _snapshot_code(self) -> Path:
        snapshot_dir = self.experiment_dir / "code_snapshot"
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        shutil.copytree(self.project.root / "code", snapshot_dir)
        console.print(f"[dim]Code snapshot at {snapshot_dir.relative_to(self.project.root)}[/dim]")
        return snapshot_dir

    def _relative_to_project(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project.root))
        except ValueError:
            return str(path)

    def _load_training_summary_from_state(self) -> Dict[str, Any]:
        if self.training_summary_override:
            return dict(self.training_summary_override)
        model_entry = self.manager.get_module("model") or {}
        return dict(model_entry.get("training_summary") or {})

    def _load_trained_model(self):
        load_fn = getattr(self.model_module, "load_model", None)
        if callable(load_fn):
            return load_fn(self.config)
        try:
            from autogluon.tabular import TabularPredictor

            model_path = self.config.system.model_path
            if model_path.exists():
                console.print(f"[cyan]Loading model from {model_path}...[/cyan]")
                return TabularPredictor.load(str(model_path))
        except Exception as exc:  # pragma: no cover - fallback logging only
            console.print(f"[yellow]Autogluon loader fallback failed: {exc}[/yellow]")
        raise RuntimeError(
            "Model reload not implemented for this model. "
            "Provide a load_model(config) helper in the model module."
        )

    def _run_training(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        artifacts,
    ) -> Tuple[Any, Dict[str, Any], Path]:
        try:
            self.manager.start_module(
                "model",
                {
                    "template": self.template_name,
                    "model": self.model_name,
                },
                allow_restart=True,
            )
        except ModuleStateError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            raise

        try:
            model, training_summary = self._train_model(train_df, val_df, artifacts)
            snapshot_path = self._snapshot_code()
            payload = {
                "template": self.template_name,
                "model": self.model_name,
                "local_cv": training_summary.get("local_cv"),
                "training_summary": training_summary,
                "config": self.config.as_plain_dict(),
                "code_snapshot": str(self._relative_to_project(snapshot_path)),
                "model_path": str(self._relative_to_project(self.config.system.model_path)),
                "preprocess": self.preprocess_record,
            }
            self.manager.complete_module("model", payload)
            return model, training_summary, snapshot_path
        except Exception as exc:
            self.manager.fail_module("model", str(exc))
            raise

    def _run_predict(
        self,
        model: Any,
        test_df: pd.DataFrame,
        artifacts,
        training_summary: Dict[str, Any],
    ):
        try:
            self.manager.require("model")
        except ModuleStateError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            raise

        try:
            self.manager.start_module(
                "predict",
                {
                    "template": self.template_name,
                    "model": self.model_name,
                },
                allow_restart=True,
            )
        except ModuleStateError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            return None

        try:
            predictions = self._run_predictions(model, test_df, artifacts)
            submission = self._save_submission(predictions, training_summary)
            payload = {
                "template": self.template_name,
                "model": self.model_name,
                "local_cv": training_summary.get("local_cv"),
                "submission_file": str(self._relative_to_project(submission.path)),
                "model_path": str(self._relative_to_project(self.config.system.model_path)),
            }
            self.manager.complete_module("predict", payload)
            return submission
        except Exception as exc:
            self.manager.fail_module("predict", str(exc))
            raise


def run(args: argparse.Namespace):
    context = load_project_context(args.project)
    templates_dir = context.root / "templates"
    preprocess_templates = _load_template_file(templates_dir / "preprocess.yaml", "preprocess")
    model_templates = _load_template_file(templates_dir / "model.yaml", "model")
    stage = args.stage or "all"

    if stage == "predict" and not args.experiment_id:
        raise RuntimeError("--experiment-id is required when running stage=predict")
    if stage == "preprocess":
        # Model template optional; skip model validation
        pass

    if stage == "predict":
        manager = ExperimentManager.load_existing(args.project, args.experiment_id)
    else:
        manager = ExperimentManager.load_or_create(args.project, args.experiment_id)
        if args.experiment_id is None:
            console.print(f"[bold blue]Using experiment ID:[/bold blue] {manager.experiment_id}")

    require_eda = stage in {"all", "train"} and args.require_eda and not args.skip_eda_check
    if require_eda:
        try:
            manager.require("eda")
        except ModuleStateError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
            return

    force_predict_flag = bool(args.force or getattr(args, "force_predict", False))

    if args.force:
        modules_to_clear = []
        if stage in {"all", "train"}:
            modules_to_clear.extend(["preprocess", "model", "predict"])
        elif stage == "predict":
            modules_to_clear.append("predict")
        elif stage == "preprocess":
            modules_to_clear.append("preprocess")

        cleared = []
        for mod in modules_to_clear:
            if manager.modules().pop(mod, None):
                cleared.append(mod)
        if cleared:
            console.print(
                f"[yellow]--force: cleared modules {', '.join(cleared)} for experiment {manager.experiment_id}[/yellow]"
            )
            manager.save()
    elif force_predict_flag and stage in {"all", "predict"}:
        if manager.modules().pop("predict", None):
            console.print(
                f"[yellow]--force-predict: cleared 'predict' for experiment {manager.experiment_id}[/yellow]"
            )
            manager.save()

    model_entry = manager.get_module("model") or {}
    preprocess_entry = model_entry.get("preprocess") or {}

    preprocess_template_name = args.preprocess_template or preprocess_entry.get("template") or "identity"
    model_template_name = args.model_template or args.template or model_entry.get("template")
    if stage != "preprocess" and not model_template_name:
        raise RuntimeError("Model template is required (pass --model-template).")

    preprocess_payload = preprocess_templates.get(preprocess_template_name)
    if preprocess_payload is None and stage != "predict":
        raise RuntimeError(f"Preprocess template '{preprocess_template_name}' not defined for project {args.project}")
    if preprocess_payload is None:
        preprocess_payload = {
            "module": preprocess_entry.get("module", "identity"),
            "config": preprocess_entry.get("config", {}),
        }

    template_payload = model_templates.get(model_template_name) if model_template_name else None
    if template_payload is None and stage not in {"predict", "preprocess"}:
        raise RuntimeError(f"Model template '{model_template_name}' not defined for project {args.project}")
    if template_payload is None:
        template_payload = {"model": model_entry.get("model", "autogluon_baseline"), "config": {}}

    config_override = None
    training_summary_override = None
    snapshot_override = None
    preprocess_config_override = None
    if stage == "predict":
        config_dict = model_entry.get("config")
        if not config_dict:
            raise RuntimeError(f"Experiment {manager.experiment_id} has no saved config; cannot run predict.")
        config_override = ModelConfig.model_validate(config_dict)
        training_summary_override = model_entry.get("training_summary") or {}
        snapshot_str = model_entry.get("code_snapshot")
        if snapshot_str:
            snapshot_path = Path(snapshot_str)
            snapshot_override = snapshot_path if snapshot_path.is_absolute() else (context.root / snapshot_path)
        preprocess_config_override = preprocess_entry.get("config") or {}

    runner = MLRunner(
        context,
        manager,
        model_template_name if stage != "preprocess" else None,
        template_payload if stage != "preprocess" else None,
        args,
        stage=stage,
        config_override=config_override,
        model_name_override=model_entry.get("model"),
        training_summary_override=training_summary_override,
        snapshot_override=snapshot_override,
        preprocess_template_name=preprocess_template_name,
        preprocess_template_payload=preprocess_payload,
        preprocess_config_override=preprocess_config_override,
    )

    if stage != "preprocess" and any([args.time_limit is not None, args.preset, args.use_gpu is not None, args.ag_smoke]):
        hp = runner.config.hyperparameters
        console.print(
            f"[dim]Overrides → preset={hp.presets or '<template>'} "
            f"time_limit={hp.time_limit or '<template>'} "
            f"use_gpu={int(bool(hp.use_gpu))}[/dim]"
        )

    try:
        result = runner.execute()
    except ModuleStateError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return

    submission_artifact = result.get("submission")
    training_summary = result.get("training_summary") or {}
    local_cv = training_summary.get("local_cv")

    if stage not in {"all", "predict"} or submission_artifact is None:
        return

    if args.skip_submit or os.environ.get("KAGGLE_SKIP_SUBMIT"):
        console.print("[yellow]Skipping Kaggle submission workflow (--skip-submit or KAGGLE_SKIP_SUBMIT).[/yellow]")
        return

    if args.kaggle_message:
        description = args.kaggle_message
    else:
        description = SubmissionRunner.build_message(
            submission_artifact,
            experiment_id=manager.experiment_id,
            local_score=local_cv,
            model_label=runner.template_name,
            feature_count=getattr(submission_artifact, "feature_count", None),
            smoke=bool(args.ag_smoke),
        )

    submission_runner = SubmissionRunner(
        artifact=submission_artifact,
        kaggle_message=description,
        wait_seconds=args.wait_seconds,
        cdp_url=args.cdp_url,
        auto_submit=args.auto_submit,
        prompt=not args.auto_submit,
        skip_browser=args.skip_score_fetch,
        skip_git=args.skip_git,
        experiment_id=manager.experiment_id,
    )
    submission_runner.execute()


def cli_entry(default_project: Optional[str] = None):
    args = parse_args(default_project)
    run(args)


if __name__ == "__main__":
    cli_entry()

"""Optuna-driven preprocessing tuning (FAST evaluation)."""

from __future__ import annotations

import json
import shutil
import os
import signal
import time
import traceback
from dataclasses import dataclass
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rich.console import Console
import re

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.core.experiment import ExperimentState
from mlarena.core.module import ModuleContext
from mlarena.core.pipeline import PipelineExecutor
from mlarena.core.config import load_pipeline_def
from mlarena.utils.project import load_project_config


REPO_ROOT = Path(__file__).resolve().parents[3]
SEARCH_SPACE_DIR = REPO_ROOT / "src" / "mlarena" / "search_spaces" / "preprocess"
DEFAULT_SUPER_CHAIN = REPO_ROOT / "conf" / "preprocess" / "super_chain_optuna.yaml"


@dataclass
class EDAInfo:
    numeric_cols: set[str]
    categorical_cols: set[str]
    datetime_cols: set[str]
    all_cols: set[str]


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return data


def _safe_name(value: str) -> str:
    value = value.strip().replace("/", "-").replace("\\", "-").replace(" ", "-")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "unnamed"


def _write_best_templates(
    project_root: Path,
    study_name: str,
    best_trial: int,
    pipeline_path: Path,
) -> Dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(pipeline_path).read_text()) or {}
    pipeline = payload.get("pipeline", []) or []

    templates_dir = project_root / "templates" / "preprocess"
    templates_dir.mkdir(parents=True, exist_ok=True)

    study_safe = _safe_name(study_name)
    trial_suffix = f"{best_trial:04d}"
    base_name = f"optuna-{study_safe}-{trial_suffix}"

    chain_list: List[str] = []
    step_files: List[str] = []
    for step in pipeline:
        step_name = _safe_name(step.get("name", "step"))
        tpl_name = f"{base_name}-{step_name}"
        tpl_path = templates_dir / f"{tpl_name}.yaml"
        module_name = step.get("module") or step.get("template")
        config = step.get("config") or {}

        tpl_payload = {"module": module_name, "config": config}
        tpl_path.write_text(yaml.safe_dump(tpl_payload, sort_keys=False))

        chain_list.append(tpl_name)
        step_files.append(str(tpl_path))

    chain_path = templates_dir / f"{base_name}.yaml"
    chain_path.write_text(yaml.safe_dump({"chain": chain_list}, sort_keys=False))

    return {
        "chain_template": str(chain_path),
        "step_templates": step_files,
        "base_name": base_name,
    }


def _load_search_spaces() -> Dict[str, Dict[str, Any]]:
    spaces: Dict[str, Dict[str, Any]] = {}
    for path in SEARCH_SPACE_DIR.glob("*.yaml"):
        name = path.stem
        spaces[name] = _load_yaml(path)
    return spaces


def _extract_eda_info(eda_path: Path) -> EDAInfo:
    if not eda_path.exists():
        raise FileNotFoundError(f"EDA summary missing: {eda_path}")
    data = json.loads(eda_path.read_text())
    summary = data.get("train", {}).get("summary", {})
    variables = summary.get("variables", {})
    numeric = set()
    categorical = set()
    datetime = set()
    for col, info in variables.items():
        vtype = (info or {}).get("type")
        if vtype == "Numeric":
            numeric.add(col)
        elif vtype == "Categorical":
            categorical.add(col)
        elif vtype in {"DateTime", "Datetime", "Date"}:
            datetime.add(col)
    all_cols = set(variables.keys())
    return EDAInfo(
        numeric_cols=numeric,
        categorical_cols=categorical,
        datetime_cols=datetime,
        all_cols=all_cols,
    )


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    if not override:
        return dict(base)
    merged = dict(base)
    merged.update(override)
    return merged


def _sample_subset(trial, name: str, values: List[Any], min_items: int, max_items: int, sort: bool) -> List[Any]:
    combos: List[Tuple[Any, ...]] = []
    for k in range(min_items, max_items + 1):
        combos.extend(combinations(values, k))
    if not combos:
        return []

    # Use JSON-encoded strings to avoid Optuna warnings about tuple/list choices
    encoded = [json.dumps(list(combo)) for combo in combos]
    chosen = trial.suggest_categorical(name, encoded)
    try:
        result = json.loads(chosen)
    except Exception:
        result = []
    if sort:
        try:
            result = sorted(result)
        except Exception:
            pass
    return result


def _sample_param(trial, name: str, spec: Dict[str, Any]) -> Any:
    ptype = spec.get("type")
    if ptype == "choice":
        values = spec.get("values", [])
        # Encode complex values to keep Optuna storage happy
        encoded: List[Any] = []
        needs_decode = False
        for v in values:
            if isinstance(v, (list, tuple, dict)):
                encoded.append(f"__json__:{json.dumps(v)}")
                needs_decode = True
            else:
                encoded.append(v)
        chosen = trial.suggest_categorical(name, encoded)
        if needs_decode and isinstance(chosen, str) and chosen.startswith("__json__:"):
            try:
                return json.loads(chosen[len("__json__:"):])
            except Exception:
                return chosen
        return chosen
    if ptype == "int_range":
        step = spec.get("step")
        return trial.suggest_int(name, int(spec.get("min", 0)), int(spec.get("max", 0)), step=step or 1)
    if ptype == "float_range":
        step = spec.get("step")
        log = bool(spec.get("log", False))
        val = trial.suggest_float(name, float(spec.get("min", 0.0)), float(spec.get("max", 0.0)), step=step if not log else None, log=log)
        round_to = spec.get("round")
        if round_to is not None:
            try:
                val = round(val, int(round_to))
            except Exception:
                pass
        return val
    if ptype == "bool":
        return trial.suggest_categorical(name, [True, False])
    if ptype == "fixed":
        return spec.get("value")
    if ptype == "subset":
        return _sample_subset(
            trial,
            name,
            spec.get("values", []),
            int(spec.get("min_items", 1)),
            int(spec.get("max_items", 1)),
            bool(spec.get("sort", False)),
        )
    raise ValueError(f"Unsupported param type: {ptype}")


def _filter_variants(
    variants: List[Dict[str, Any]],
    *,
    heavy_variants: List[str],
    allow_heavy_variants: bool,
    allow_group_mode: bool,
    force_global_only: bool,
) -> List[Dict[str, Any]]:
    filtered = []
    for variant in variants:
        vname = variant.get("name")
        if not allow_heavy_variants and vname in heavy_variants:
            continue
        if force_global_only:
            params = variant.get("params", {})
            mode_spec = params.get("mode") if isinstance(params, dict) else None
            if mode_spec and isinstance(mode_spec, dict) and mode_spec.get("values") == ["by_group"]:
                continue
        if not allow_group_mode:
            params = variant.get("params", {})
            mode_spec = params.get("mode") if isinstance(params, dict) else None
            if mode_spec and isinstance(mode_spec, dict) and "by_group" in (mode_spec.get("values") or []):
                continue
        filtered.append(variant)
    return filtered




def _should_disable_step(
    step: Dict[str, Any],
    eda: EDAInfo,
    problem_type: Optional[str],
    allow_heavy_steps: bool,
    merged_cfg: Dict[str, Any],
    filter_by_eda: bool,
) -> bool:
    meta = step.get("meta", {}) or {}
    if meta.get("heavy_step") and not allow_heavy_steps:
        return True

    template = step.get("template")
    if filter_by_eda:
        if template == "datetime_handler" and not eda.datetime_cols:
            return True
        if template == "time_series_features" and not eda.datetime_cols:
            return True
        if template == "encoder" and not eda.categorical_cols:
            return True
        if template == "categorical_cross" and len(eda.categorical_cols) < 2:
            return True
        if template == "numeric_binner" and not eda.numeric_cols:
            return True

    if template == "imbalance_handler" and (problem_type or "").lower() == "regression":
        return True
    if template == "target_transformer" and (problem_type or "").lower() != "regression":
        return True
    # Group-dependent steps: disable if keys/values empty
    if template in {"feature_group_agg", "groupwise_normalizer"}:
        group_keys = merged_cfg.get("group_keys") or []
        value_cols = merged_cfg.get("group_value_cols") or merged_cfg.get("value_cols") or []
        if not group_keys or not value_cols:
            return True

    return False


def _build_trial_pipeline(
    trial,
    super_chain: Dict[str, Any],
    search_spaces: Dict[str, Dict[str, Any]],
    eda: EDAInfo,
    *,
    allow_heavy_steps: bool,
    allow_heavy_variants: bool,
    problem_type: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    steps = super_chain.get("preprocessors", []) or []
    enabled_groups: set[str] = set()
    pipeline: List[Dict[str, Any]] = []
    enabled_steps: List[str] = []
    variant_map: Dict[str, str] = {}

    filter_by_eda = bool(super_chain.get("filter_by_eda", True))

    for step in steps:
        step_name = step.get("name")
        template = step.get("template")
        group = step.get("group") or step_name
        if not template:
            continue

        fixed_config = step.get("fixed_config") or {}
        overrides = step.get("overrides") or {}
        override_cfg = overrides.get("config") if isinstance(overrides, dict) else None
        if override_cfg is None and isinstance(overrides, dict):
            # Strip meta-only override keys if present
            override_cfg = {k: v for k, v in overrides.items() if k not in {"allow_group_mode", "force_global_only"}}
        meta = step.get("meta") or {}

        # Determine base config
        base_cfg = {}
        if template in search_spaces:
            base_cfg = search_spaces[template].get("base_config", {}) or {}

        merged_base = _merge_dicts(base_cfg, override_cfg or {})
        merged_base = _merge_dicts(merged_base, fixed_config)

        if _should_disable_step(step, eda, problem_type, allow_heavy_steps, merged_base, filter_by_eda):
            continue

        # Decide enablement (allow toggle unless explicitly fixed)
        is_fixed = bool(meta.get("fixed", False))
        if is_fixed:
            enabled = bool(step.get("enabled", True))
        else:
            enabled = trial.suggest_categorical(f"{step_name}.enabled", [True, False])

        if not enabled:
            continue

        # Build config
        variant_name = "fixed"
        variant_params: Dict[str, Any] = {}

        if not fixed_config:
            space = search_spaces.get(template, {})
            variants = space.get("variants", []) or []
            heavy_variants = meta.get("heavy_variants", []) or []
            allow_group_mode = bool(overrides.get("allow_group_mode", True))
            force_global_only = bool(overrides.get("force_global_only", False))
            if template == "rank_features":
                group_keys = merged_base.get("group_keys") or []
                valid_group = bool(group_keys) and all(k in eda.all_cols for k in group_keys)
                if not valid_group:
                    allow_group_mode = False

            variants = _filter_variants(
                variants,
                heavy_variants=heavy_variants,
                allow_heavy_variants=allow_heavy_variants,
                allow_group_mode=allow_group_mode,
                force_global_only=force_global_only,
            )
            if not variants:
                continue

            variant_name = trial.suggest_categorical(
                f"{step_name}.variant",
                [v.get("name") for v in variants],
            )
            variant = next(v for v in variants if v.get("name") == variant_name)
            requires = variant.get("requires_preproc") or []
            missing_req = False
            for req in requires:
                group = req.get("group")
                if group and group not in enabled_groups:
                    missing_req = True
                    break
            if missing_req:
                # Keep search space stable, but skip step when prerequisites aren't met
                continue
            params = variant.get("params", {}) or {}
            for param_name, spec in params.items():
                if not isinstance(spec, dict):
                    continue
                full_name = f"{step_name}.{variant_name}.{param_name}"
                variant_params[param_name] = _sample_param(trial, full_name, spec)

        # Special handling for rank_features
        if template == "rank_features":
            group_keys = merged_base.get("group_keys") or []
            valid_group = bool(group_keys) and all(k in eda.all_cols for k in group_keys)
            if not valid_group and variant_params.get("mode") == "by_group":
                variant_params["mode"] = "global"

        final_cfg = _merge_dicts(merged_base, variant_params)

        pipeline.append(
            {
                "name": step_name,
                "group": group,
                "template": template,
                "module": template,
                "variant": variant_name,
                "config": final_cfg,
                "meta": meta,
            }
        )
        enabled_groups.add(group)
        enabled_steps.append(step_name)
        variant_map[step_name] = variant_name

    meta_out = {
        "enabled_steps": enabled_steps,
        "variant_map": variant_map,
    }
    return pipeline, meta_out


def _build_baseline_pipeline(
    super_chain: Dict[str, Any],
    search_spaces: Dict[str, Dict[str, Any]],
    eda: EDAInfo,
    *,
    allow_heavy_steps: bool,
    problem_type: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    steps = super_chain.get("preprocessors", []) or []
    pipeline: List[Dict[str, Any]] = []
    enabled_steps: List[str] = []
    variant_map: Dict[str, str] = {}
    enabled_groups: set[str] = set()
    filter_by_eda = bool(super_chain.get("filter_by_eda", True))

    for step in steps:
        meta = step.get("meta", {}) or {}
        if not meta.get("fixed"):
            continue
        if not bool(step.get("enabled", True)):
            continue
        template = step.get("template")
        step_name = step.get("name")
        group = step.get("group") or step_name
        if not template:
            continue

        fixed_config = step.get("fixed_config") or {}
        overrides = step.get("overrides") or {}
        override_cfg = overrides.get("config") if isinstance(overrides, dict) else None
        if override_cfg is None and isinstance(overrides, dict):
            override_cfg = {k: v for k, v in overrides.items() if k not in {"allow_group_mode", "force_global_only"}}

        base_cfg = search_spaces.get(template, {}).get("base_config", {}) or {}
        merged_base = _merge_dicts(base_cfg, override_cfg or {})
        merged_base = _merge_dicts(merged_base, fixed_config)

        if _should_disable_step(step, eda, problem_type, allow_heavy_steps, merged_base, filter_by_eda):
            continue

        pipeline.append(
            {
                "name": step_name,
                "group": group,
                "template": template,
                "module": template,
                "variant": "fixed",
                "config": merged_base,
                "meta": meta,
            }
        )
        enabled_groups.add(group)
        enabled_steps.append(step_name)
        variant_map[step_name] = "fixed"

    return pipeline, {"enabled_steps": enabled_steps, "variant_map": variant_map}


def _build_context(
    project_root: Path,
    project_name: str,
    module_name: str,
    experiment_id: str,
    config,
    config_module,
    pipeline_def: Dict[str, Any],
) -> ModuleContext:
    state = ExperimentState.load_or_create(
        project_root=project_root,
        project_name=project_name,
        experiment_id=experiment_id,
        pipeline=pipeline_def or {},
        run_invocation={"argv": [], "module": module_name},
        create_dirs=True,
        setup_module_name=None,
    )
    artifact_dir = state.experiment_dir / "artifacts" / module_name
    return ModuleContext(
        project_name=project_name,
        project_root=project_root,
        experiment_id=state.experiment_id,
        experiment_dir=state.experiment_dir,
        artifact_dir=artifact_dir,
        cli_args={},
        state=state,
        config_module=config_module,
        config=config,
    )


def _cleanup_trial_artifacts(
    trial_dir: Path,
    project_root: Path,
    *,
    cleanup_model: bool,
    cleanup_processed: bool,
) -> Dict[str, Any]:
    """Remove large trial artifacts safely (model_fast dir and *_processed.csv.gz files)."""
    if not cleanup_model and not cleanup_processed:
        return {"skipped": "cleanup disabled"}

    try:
        trial_dir_resolved = trial_dir.resolve()
        experiments_root = (project_root / "experiments").resolve()
        if experiments_root not in trial_dir_resolved.parents:
            return {"skipped": "outside experiments"}
        if not trial_dir_resolved.name.startswith("trial_"):
            return {"skipped": "not a trial dir"}
    except Exception:
        return {"skipped": "resolve failed"}

    removed = {"processed_files": 0, "model_dir_removed": False}

    if cleanup_model:
        model_dir = trial_dir_resolved / "model_fast"
        if model_dir.exists() and model_dir.is_dir() and model_dir.parent == trial_dir_resolved:
            shutil.rmtree(model_dir, ignore_errors=True)
            removed["model_dir_removed"] = True

    if cleanup_processed:
        for path in trial_dir_resolved.rglob("*_processed.csv.gz"):
            try:
                path_resolved = path.resolve()
                if not path_resolved.is_relative_to(trial_dir_resolved):
                    continue
                path_resolved.unlink(missing_ok=True)
                removed["processed_files"] += 1
            except Exception:
                continue

    return removed


def _run_trial_worker(
    payload_path: Path,
    project_root: Path,
    project_name: str,
    chain_exp_id: str,
    model_template: str,
    max_features_out: int,
    config,
) -> None:
    try:
        # Make this process a session leader so parent can kill the group on timeout.
        try:
            os.setsid()
        except Exception:
            pass

        payload = json.loads(payload_path.read_text())
        pipeline = payload["pipeline"]
        trial_dir = Path(payload["trial_dir"])
        trial_dir.mkdir(parents=True, exist_ok=True)

        config_module = load_project_config(project_root)
        pipeline_def, _ = load_pipeline_def("default", project_root=project_root)

        preprocess_start = time.time()
        last_step_dir = None
        last_shapes = None

        from mlarena.modules.preprocess import PreprocessModule

        for idx, step in enumerate(pipeline):
            step_name = step["name"]
            exp_id = f"{chain_exp_id}/{idx}-{step_name}"
            input_source = f"{idx - 1}-{pipeline[idx - 1]['name']}" if idx > 0 else None

            context = _build_context(
                project_root=project_root,
                project_name=project_name,
                module_name="preprocess",
                experiment_id=exp_id,
                config=config,
                config_module=config_module,
                pipeline_def=pipeline_def,
            )

            module = PreprocessModule(context)
            module.set_invocation_params(
                {
                    "preprocess_template": step_name,
                    "preprocess_template_config": {
                        "module": step["module"],
                        "config": step["config"],
                    },
                    "input_source": input_source,
                    "chain_exp_id": chain_exp_id,
                    "is_last_in_chain": idx == (len(pipeline) - 1),
                    "force": True,
                    "lock": False,
                    "quiet_preprocess_panel": bool(payload.get("quiet_preprocess_panel")),
                    "quiet_model_panel": bool(payload.get("quiet_model_panel")),
                }
            )

            executor = PipelineExecutor({"preprocess": module})
            results = executor.run_module("preprocess", force=True, skip_deps=False)
            result = results.get("preprocess")
            if not result or not result.success:
                raise RuntimeError(f"Preprocess step failed: {step_name}")

            last_step_dir = Path(context.experiment_dir)
            last_shapes = (result.payload or {}).get("shapes")

        preprocess_sec = time.time() - preprocess_start

        if last_step_dir is None:
            raise RuntimeError("No preprocessing steps executed")

        n_features_out = None
        if last_shapes and "train_after" in last_shapes:
            try:
                n_features_out = int(last_shapes["train_after"][1])
            except Exception:
                n_features_out = None

        if n_features_out is not None and n_features_out > max_features_out:
            raise RuntimeError(f"Too many features: {n_features_out} > {max_features_out}")

        # Run FAST model
        from mlarena.modules.model import ModelModule

        model_exp_id = f"{chain_exp_id}/model_fast"
        model_ctx = _build_context(
            project_root=project_root,
            project_name=project_name,
            module_name="model",
            experiment_id=model_exp_id,
            config=config,
            config_module=config_module,
            pipeline_def=pipeline_def,
        )

        model_module = ModelModule(model_ctx)
        model_params = {
            "model_template": model_template,
            "preprocess_template": "optuna_trial",
            "preprocess_exp_dir": str(last_step_dir) if last_step_dir else None,
            "force": True,
            "quiet_model_panel": bool(payload.get("quiet_model_panel")),
        }
        if payload.get("model_verbosity") is not None:
            try:
                model_params["verbosity"] = int(payload.get("model_verbosity"))
            except Exception:
                model_params["verbosity"] = payload.get("model_verbosity")
        # Enable weight_evaluation only if preprocessing produced weights
        weight_eval = False
        if last_step_dir:
            state_path = last_step_dir / "state.json"
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text())
                    modules = state.get("modules", {})
                    preprocess_entry = modules.get("preprocess") or (next(iter(modules.values())) if modules else {})
                    custom_state = preprocess_entry.get("payload", {}).get("custom_module_state", {})
                    weights_path = custom_state.get("weights_path")
                    if weights_path:
                        wp = Path(weights_path)
                        if not wp.is_absolute():
                            wp = project_root / wp
                        if wp.exists():
                            weight_eval = True
                except Exception:
                    weight_eval = False
        model_params["weight_evaluation"] = weight_eval
        try:
            for k, v in (config.common.model_dump() if hasattr(config, "common") else {}).items():
                if v is not None:
                    model_params[k] = v
        except Exception:
            pass
        model_module.set_invocation_params(model_params)

        model_executor = PipelineExecutor({"model": model_module})
        model_results = model_executor.run_module("model", force=True, skip_deps=False)
        model_result = model_results.get("model")
        if not model_result or not model_result.success:
            detail = None
            if model_result is not None:
                detail = model_result.error or (model_result.payload or {}).get("error")
            raise RuntimeError(f"Model step failed{f': {detail}' if detail else ''}")

        score_fast = None
        payload = model_result.payload or {}
        if payload.get("local_cv_score") is not None:
            score_fast = float(payload["local_cv_score"])
        elif payload.get("leaderboard"):
            try:
                import pandas as pd
                lb = pd.read_csv(payload["leaderboard"], compression="infer")
                if "score" in lb.columns:
                    score_fast = float(lb.iloc[0]["score"])
            except Exception:
                score_fast = None

        metrics = {
            "score_fast": score_fast,
            "preprocess_sec": preprocess_sec,
            "n_features_out": n_features_out,
        }

        (trial_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    except Exception as exc:
        error_path = Path(payload_path).parent / "trial_error.json"
        error_path.write_text(json.dumps({"error": str(exc), "traceback": traceback.format_exc()}, indent=2))
        raise


@ModuleRegistry.register
class PreprocessTuneModule(BaseModule):
    """Tune preprocessing via Optuna (FAST evaluation only)."""

    name = "preprocess-tune"
    description = "Optuna preprocessing tuning (FAST)"

    def execute(self) -> ModuleResult:
        console = Console(force_terminal=True)
        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

        try:
            import optuna
        except Exception as exc:
            marker = artifact_dir / "tune_failed.txt"
            marker.write_text(f"Optuna missing: {exc}")
            return ModuleResult(success=False, error="optuna missing", artifacts=[marker])

        config_module = self.context.config_module or load_project_config(self.context.project_root)
        problem_type = getattr(config_module, "AUTOGLUON_PROBLEM_TYPE", None)

        # Resolve parameters (invocation params override config)
        params = self.invocation_params or {}
        cfg = self.context.config

        def _param(name: str, default: Any) -> Any:
            if name in params:
                return params[name]
            if hasattr(cfg, "preprocess_tune"):
                section = getattr(cfg, "preprocess_tune") or {}
                if name in section:
                    return section[name]
            if hasattr(cfg, name):
                return getattr(cfg, name)
            return default

        super_chain_path = Path(_param("super_chain", DEFAULT_SUPER_CHAIN))
        super_chain = _load_yaml(super_chain_path)

        tune_cfg = super_chain.get("tune", {}) or {}
        study_name = _param("study_name", tune_cfg.get("study_name") or super_chain.get("experiment_prefix") or "optuna_preprocess")
        n_trials = int(_param("n_trials", tune_cfg.get("n_trials", 10)))
        optuna_workers = int(_param("optuna_workers", tune_cfg.get("optuna_workers", 1)))
        model_cleanup = bool(
            _param("model_cleanup", tune_cfg.get("model_cleanup", False))
            or _param("ag_cleanup", tune_cfg.get("ag_cleanup", False))
        )
        cleanup_processed = bool(
            _param("cleanup_processed", tune_cfg.get("cleanup_processed", False))
            or _param("model_cleanup", tune_cfg.get("model_cleanup", False))
            or _param("ag_cleanup", tune_cfg.get("ag_cleanup", False))
        )
        max_trial_sec = int(_param("max_trial_sec", tune_cfg.get("max_trial_sec", 1800)))
        allow_heavy_steps = bool(_param("allow_heavy_steps", tune_cfg.get("allow_heavy_steps", False)))
        allow_heavy_variants = bool(_param("allow_heavy_variants", tune_cfg.get("allow_heavy_variants", False)))
        max_features_out = int(_param("max_features_out", tune_cfg.get("max_features_out", 50000)))
        seed = int(_param("seed", getattr(cfg.common, "seed", 42)))
        quiet_preprocess_panel = bool(
            _param("quiet_preprocess_panel", tune_cfg.get("quiet_preprocess_panel", False))
            or _param("quiet_preprocess_panels", tune_cfg.get("quiet_preprocess_panels", False))
        )
        quiet_model_panel = bool(
            _param("quiet_model_panel", tune_cfg.get("quiet_model_panel", False))
            or _param("quiet_model_panels", tune_cfg.get("quiet_model_panels", False))
        )
        preprocess_cfg = cfg.preprocess if isinstance(cfg.preprocess, dict) else {}
        if not quiet_preprocess_panel:
            quiet_preprocess_panel = bool(
                preprocess_cfg.get("quiet_preprocess_panel")
                or preprocess_cfg.get("quiet_preprocess_panels")
                or preprocess_cfg.get("quiet_preprocess")
            )
        if not quiet_model_panel:
            model_cfg = cfg.model if isinstance(cfg.model, dict) else {}
            quiet_model_panel = bool(
                model_cfg.get("quiet_model_panel")
                or model_cfg.get("quiet_model_panels")
            )
        model_verbosity = _param("model_verbosity", tune_cfg.get("model_verbosity"))
        if model_verbosity is None:
            model_cfg = cfg.model if isinstance(cfg.model, dict) else {}
            model_verbosity = model_cfg.get("verbosity")

        model_template = _param("model_template", None)
        if not model_template:
            eval_cfg = super_chain.get("evaluation", {}) or {}
            if isinstance(eval_cfg, dict):
                model_template = eval_cfg.get("model")
        if not model_template:
            model_template = getattr(cfg, "model_template", None)
        if not model_template:
            raise ValueError("model_template is required for preprocess tuning")

        # EDA summary is required
        eda_path = self.context.project_root / "experiments" / "eda" / "artifacts" / "eda" / "eda_summary.json"
        eda_info = _extract_eda_info(eda_path)

        # Storage path
        db_dir = self.context.project_root / "experiments" / "db"
        db_dir.mkdir(parents=True, exist_ok=True)
        storage_url = _param("storage_url", f"sqlite:///{db_dir / f'optuna_{study_name}.sqlite'}")
        optuna_storage_timeout = _param("optuna_storage_timeout", tune_cfg.get("optuna_storage_timeout"))

        search_spaces = _load_search_spaces()

        sampler = optuna.samplers.TPESampler(seed=seed)
        if optuna_storage_timeout and str(storage_url).startswith("sqlite:///"):
            storage = optuna.storages.RDBStorage(
                url=storage_url,
                engine_kwargs={"connect_args": {"timeout": int(optuna_storage_timeout)}},
            )
        else:
            storage = storage_url
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=storage,
            load_if_exists=True,
            sampler=sampler,
        )
        stop_state = {"requested": False}
        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_stop(signum, frame):  # type: ignore[override]
            if stop_state["requested"]:
                raise KeyboardInterrupt
            stop_state["requested"] = True
            try:
                study.stop()
            except Exception:
                pass
            console.print(
                "[yellow]⚠ Stop requested (Ctrl-C). No new trials will be started; waiting for running trials to finish.[/yellow]"
            )
            console.print("[yellow]⚠ Press Ctrl-C again to force stop.[/yellow]")

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        def objective(trial: optuna.Trial) -> float:
            if stop_state["requested"]:
                raise optuna.TrialPruned("stop requested")
            if trial.number == 0:
                pipeline, meta = _build_baseline_pipeline(
                    super_chain,
                    search_spaces,
                    eda_info,
                    allow_heavy_steps=allow_heavy_steps,
                    problem_type=problem_type,
                )
            else:
                pipeline, meta = _build_trial_pipeline(
                    trial,
                    super_chain,
                    search_spaces,
                    eda_info,
                    allow_heavy_steps=allow_heavy_steps,
                    allow_heavy_variants=allow_heavy_variants,
                    problem_type=problem_type,
                )

            trial_id = trial.number
            trial_dir = self.context.project_root / "experiments" / f"optuna_{study_name}" / f"trial_{trial_id:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            # Save trial pipeline for reproducibility
            pipeline_path = trial_dir / "trial_pipeline.yaml"
            pipeline_payload = {
                "pipeline": pipeline,
                "meta": meta,
                "study": study_name,
                "trial_id": trial_id,
            }
            import yaml
            pipeline_path.write_text(yaml.safe_dump(pipeline_payload, sort_keys=False))

            # Prepare worker payload
            payload = {
                "pipeline": pipeline,
                "trial_dir": str(trial_dir),
                "config": self.context.config.model_dump(),
                "quiet_preprocess_panel": quiet_preprocess_panel,
                "quiet_model_panel": quiet_model_panel,
                "model_verbosity": model_verbosity,
            }
            payload_path = trial_dir / "trial_payload.json"
            payload_path.write_text(json.dumps(payload, indent=2, default=str))

            chain_exp_id = f"optuna_{study_name}/trial_{trial_id:04d}"
            split_id = seed
            for step in pipeline:
                if step.get("template") == "train_fraction":
                    split_id = step.get("config", {}).get("random_state", split_id)
                    break

            ctx = get_context("spawn")
            proc = ctx.Process(
                target=_run_trial_worker,
                args=(
                    payload_path,
                    self.context.project_root,
                    self.context.project_name,
                    chain_exp_id,
                    model_template,
                    max_features_out,
                    self.context.config,
                ),
            )
            start_time = time.time()
            proc.start()
            proc.join(timeout=max_trial_sec)

            if proc.is_alive():
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except Exception:
                    proc.terminate()
                proc.join(timeout=10)
                metrics_path = trial_dir / "metrics.json"
                if not metrics_path.exists():
                    metrics_path.write_text(
                        json.dumps(
                            {
                                "score_fast": None,
                                "total_sec": max_trial_sec,
                                "preprocess_sec": None,
                                "n_features_out": None,
                                "seed": seed,
                                "split_id": split_id,
                                "enabled_steps": meta.get("enabled_steps"),
                                "variant_map": meta.get("variant_map"),
                                "error": f"timeout after {max_trial_sec}s",
                            },
                            indent=2,
                        )
                    )
                _cleanup_trial_artifacts(
                    trial_dir,
                    self.context.project_root,
                    cleanup_model=model_cleanup,
                    cleanup_processed=cleanup_processed,
                )
                raise optuna.TrialPruned(f"timeout after {max_trial_sec}s")

            metrics_path = trial_dir / "metrics.json"
            if not metrics_path.exists():
                metrics_path.write_text(
                    json.dumps(
                        {
                            "score_fast": None,
                            "total_sec": None,
                            "preprocess_sec": None,
                            "n_features_out": None,
                            "seed": seed,
                            "split_id": split_id,
                            "enabled_steps": meta.get("enabled_steps"),
                            "variant_map": meta.get("variant_map"),
                            "error": "trial failed (no metrics)",
                        },
                        indent=2,
                    )
                )
                _cleanup_trial_artifacts(
                    trial_dir,
                    self.context.project_root,
                    cleanup_model=model_cleanup,
                    cleanup_processed=cleanup_processed,
                )
                raise optuna.TrialPruned("trial failed (no metrics)")

            metrics = json.loads(metrics_path.read_text())
            score_fast = metrics.get("score_fast")
            total_sec = time.time() - start_time

            # Enrich metrics.json with additional metadata
            metrics.update(
                {
                    "total_sec": total_sec,
                    "seed": seed,
                    "split_id": split_id,
                    "enabled_steps": meta.get("enabled_steps"),
                    "variant_map": meta.get("variant_map"),
                }
            )
            metrics_path.write_text(json.dumps(metrics, indent=2))

            trial.set_user_attr("total_sec", total_sec)
            trial.set_user_attr("preprocess_sec", metrics.get("preprocess_sec"))
            trial.set_user_attr("n_features_out", metrics.get("n_features_out"))
            trial.set_user_attr("enabled_steps", meta.get("enabled_steps"))
            trial.set_user_attr("variant_map", meta.get("variant_map"))
            trial.set_user_attr("pipeline_path", str(pipeline_path))

            if score_fast is None:
                _cleanup_trial_artifacts(
                    trial_dir,
                    self.context.project_root,
                    cleanup_model=model_cleanup,
                    cleanup_processed=cleanup_processed,
                )
                raise optuna.TrialPruned("no score")
            _cleanup_trial_artifacts(
                trial_dir,
                self.context.project_root,
                cleanup_model=model_cleanup,
                cleanup_processed=cleanup_processed,
            )
            return float(score_fast)

        best_written: set[int] = set()

        def _on_trial_complete(study_obj, trial_obj):
            try:
                import optuna
                if trial_obj.state != optuna.trial.TrialState.COMPLETE:
                    return
                if study_obj.best_trial.number != trial_obj.number:
                    return
                if trial_obj.number in best_written:
                    return
                pipeline_path = trial_obj.user_attrs.get("pipeline_path")
                if not pipeline_path:
                    return
                template_info = _write_best_templates(
                    self.context.project_root,
                    study_name,
                    trial_obj.number,
                    Path(pipeline_path),
                )
                trial_obj.set_user_attr("best_chain_template", template_info.get("chain_template"))
                best_written.add(trial_obj.number)
                console.print(
                    f"[green]✓ New best trial {trial_obj.number} -> templates written ({template_info.get('base_name')})[/green]"
                )
            except Exception as exc:
                console.print(f"[yellow]⚠ Failed to write best templates: {exc}[/yellow]")

        if optuna_workers < 1:
            optuna_workers = 1
        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                n_jobs=optuna_workers,
                show_progress_bar=False,
                callbacks=[_on_trial_complete],
            )
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)

        best_payload = {}
        try:
            best_payload = {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_trial": study.best_trial.number,
            }
        except Exception:
            best_payload = {"best_value": None, "best_params": {}, "best_trial": None}

        # Ensure best templates exist at end
        try:
            if study.best_trial and study.best_trial.user_attrs.get("pipeline_path"):
                info = _write_best_templates(
                    self.context.project_root,
                    study_name,
                    study.best_trial.number,
                    Path(study.best_trial.user_attrs["pipeline_path"]),
                )
                best_payload["best_chain_template"] = info.get("chain_template")
        except Exception:
            pass
        (artifact_dir / "study_best.json").write_text(json.dumps(best_payload, indent=2))
        (artifact_dir / "study_config.json").write_text(
            json.dumps(
                {
                    "study_name": study_name,
                    "n_trials": n_trials,
                    "optuna_workers": optuna_workers,
                    "max_trial_sec": max_trial_sec,
                    "allow_heavy_steps": allow_heavy_steps,
                    "allow_heavy_variants": allow_heavy_variants,
                    "quiet_preprocess_panel": quiet_preprocess_panel,
                    "quiet_model_panel": quiet_model_panel,
                    "model_verbosity": model_verbosity,
                    "storage_url": storage_url,
                    "optuna_storage_timeout": optuna_storage_timeout,
                    "model_cleanup": model_cleanup,
                    "cleanup_processed": cleanup_processed,
                    "model_template": model_template,
                    "best_chain_template": best_payload.get("best_chain_template"),
                },
                indent=2,
            )
        )

        if best_payload.get("best_chain_template"):
            try:
                from rich.panel import Panel

                best_chain_path = Path(best_payload["best_chain_template"])
                best_chain_name = best_chain_path.stem
                project_name = self.context.project_name
                next_steps = (
                    f"[bold]1.[/] Run preprocess chain:\n"
                    f"   • [cyan]uv run python scripts/mla.py preprocess --project {project_name} preprocess_template={best_chain_name}[/cyan]\n"
                    f"[bold]2.[/] Run FAST model on best chain:\n"
                    f"   • [cyan]uv run python scripts/mla.py model --project {project_name} model_template={model_template} preprocess_template={best_chain_name}[/cyan]\n"
                    f"[bold]3.[/] Templates saved:\n"
                    f"   • [dim]{best_chain_path}[/dim]"
                )
                console.print(Panel(next_steps, title="Next Steps", border_style="yellow"))
            except Exception:
                pass

        return ModuleResult(
            success=True,
            payload={
                "study_name": study_name,
                "best_value": best_payload.get("best_value"),
                "best_chain_template": best_payload.get("best_chain_template"),
            },
            artifacts=[artifact_dir / "study_best.json", artifact_dir / "study_config.json"],
        )

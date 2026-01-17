"""Data preprocessing module.

Executes a single preprocessing step defined by a template, supports chains, caching, and custom modules.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import warnings
import contextlib
import io
import time
import sys
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.core.config import TemplateLoader
from mlarena.utils.project import data_paths, load_project_config
from mlarena.utils.time import utc_now_iso
from mlarena.core.experiment import PipelineProgress, PipelineStep

try:
    from sklearn.pipeline import Pipeline
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:
    # Optional dependency, but required for pipeline mode
    # Use distinct placeholders to avoid TypeError: duplicate base class object
    class Pipeline: pass
    class BaseEstimator: pass
    class TransformerMixin: pass

from dataclasses import dataclass, field

@dataclass
class MLArenaDataContainer:
    """Holds dataframes and state for in-memory pipeline execution."""
    train: pd.DataFrame
    test: pd.DataFrame
    val: Optional[pd.DataFrame] = None
    eval: Optional[pd.DataFrame] = None
    orig: Optional[pd.DataFrame] = None
    state: Dict[str, Any] = field(default_factory=dict)
    
    # Track shapes for reporting
    initial_shapes: Dict[str, Tuple] = field(default_factory=dict)
    
    def update_data(self, train, val, test, eval_df, orig):
        self.train = train
        self.val = val
        self.test = test
        self.eval = eval_df
        self.orig = orig


class MLArenaStepWrapper(BaseEstimator, TransformerMixin):
    """Wraps ML Arena preprocessing modules as Sklearn transformers."""
    
    def __init__(self, step_name: str, step_idx: int, config: Dict, context, loader_instance):
        self.step_name = step_name
        self.step_idx = step_idx
        self.config = config
        self.context = context
        self.loader_instance = loader_instance  # Access to _load_preprocessing_module
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: MLArenaDataContainer) -> MLArenaDataContainer:
        start_time = time.time()
        started_at = utc_now_iso()
        from rich.text import Text
        
        # 0. Update Central State (Started)
        if self.context.state and self.context.state.pipeline_progress:
            progress = self.context.state.pipeline_progress
            progress.current_step_idx = self.step_idx
            if self.step_idx < len(progress.steps):
                step_state = progress.steps[self.step_idx]
                step_state.status = "running"
                step_state.started_at = started_at
            self.context.state.last_heartbeat = started_at
            self.context.state.save()

        # Capture shapes BEFORE transformation
        shape_before_train = list(X.train.shape)
        shape_before_test = list(X.test.shape)
        shape_before_val = list(X.val.shape) if X.val is not None else None
        shape_before_eval = list(X.eval.shape) if X.eval is not None else None
        shape_before_orig = list(X.orig.shape) if X.orig is not None else None
        
        # Always show progress on stdout (unless json_output is enabled)
        is_quiet = False
        use_stderr = False
        if self.context.config and getattr(self.context.config, "json_output", False):
            is_quiet = True
            invocation_params = getattr(self.loader_instance, "invocation_params", {}) or {}
            for key in ("quiet_preprocess_panel", "quiet_preprocess_panels", "quiet_preprocess"):
                if key in invocation_params and invocation_params.get(key) is False:
                    is_quiet = False
                    use_stderr = True
                    break
        console = Console(
            file=sys.stderr if use_stderr else sys.stdout,
            force_terminal=True,
            highlight=False,
            quiet=is_quiet,
        )
        
        # Build progress line: gray for the whole info
        msg = Text(f"Pipeline Step: {self.step_name}...", style="gray")
        console.print(msg, end="")
        
        # 1. Resolve configuration
        # Support inline config or template reference
        template_name = self.config.get("template")
        template_cfg = self.config
        
        # OPTIMIZATION: If we already have 'module', we don't need any YAML files
        if template_name and not template_cfg.get("module"):
            # Load ONLY the specific template file instead of scanning everything
            from mlarena.core.config import TemplateLoader
            loader = TemplateLoader(self.context.project_root, template_type="preprocess")
            loaded_cfg = loader.load(template_name) or {}
            
            # Merge: config overrides template
            template_cfg = loaded_cfg.copy()
            template_cfg.update(self.config)
        
        module_name = template_cfg.get("module")
        if not module_name:
            # Fallback to basic template ops
            X.train = self.loader_instance._apply_template(X.train, template_cfg)
            X.test = self.loader_instance._apply_template(X.test, template_cfg)
            if X.val is not None:
                X.val = self.loader_instance._apply_template(X.val, template_cfg)
            if X.eval is not None:
                X.eval = self.loader_instance._apply_template(X.eval, template_cfg)
            
            # Update State for basic ops
            duration = time.time() - start_time
            if self.context.state and self.context.state.pipeline_progress:
                progress = self.context.state.pipeline_progress
                if self.step_idx < len(progress.steps):
                    step_state = progress.steps[self.step_idx]
                    step_state.status = "completed"
                    step_state.finished_at = utc_now_iso()
                    step_state.duration = round(duration, 3)
                    step_state.shapes = {
                        "train_before": shape_before_train,
                        "train_after": list(X.train.shape),
                        "test_before": shape_before_test,
                        "test_after": list(X.test.shape),
                    }
                self.context.state.last_heartbeat = utc_now_iso()
                self.context.state.save()
            return X

        # 2. Load module
        try:
            preprocess_module = self.loader_instance._load_preprocessing_module(module_name)
            
            # Prepare standard directory structure for this step
            # If we are in unified pipeline mode, self.context.experiment_dir points to the LAST step.
            # We want all step folders to be siblings.
            step_dir = self.context.experiment_dir.parent / self.step_name
            step_artifact_dir = step_dir / "artifacts" / "preprocess"
            step_artifact_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle both nested (legacy) and flattened (TemplateLoader) configs
            if "config" in template_cfg and isinstance(template_cfg["config"], dict):
                preprocess_config = template_cfg["config"].copy()
            else:
                # Configuration is already flattened
                preprocess_config = template_cfg.copy()

            if self.loader_instance.invocation_params.get("quiet_preprocess_panel"):
                preprocess_config["quiet"] = True
            preprocess_config["_artifact_dir"] = str(step_artifact_dir)
            preprocess_config["_dataset"] = {
                "id_column": getattr(self.context.config_module, "ID_COLUMN", "id"),
                "target": getattr(self.context.config_module, "TARGET_COLUMN", None),
                "ignored_columns": getattr(self.context.config_module, "IGNORED_COLUMNS", []),
                "problem_type": getattr(self.context.config_module, "AUTOGLUON_PROBLEM_TYPE", "binary"),
            }
            # Pass aggregated original features - Ensure correct lookup
            orig_feat_list = X.state.get("custom_module_state", {}).get("original_features")
            preprocess_config["_original_features"] = orig_feat_list

            # 3. Execute fit_transform
            import inspect
            fit_sig = inspect.signature(preprocess_module.fit_transform)
            params = fit_sig.parameters
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            supports_eval_df = "eval_df" in params or has_kwargs
            supports_orig_df = "orig_df" in params or has_kwargs

            if supports_eval_df:
                result = preprocess_module.fit_transform(
                    train_df=X.train,
                    val_df=X.val,
                    test_df=X.test,
                    eval_df=X.eval,
                    config=preprocess_config,
                    orig_df=X.orig,
                )
            else:
                combined_test = X.test
                test_len = len(X.test)
                if X.eval is not None:
                    combined_test = pd.concat([X.test, X.eval], axis=0).reset_index(drop=True)
                
                if supports_orig_df:
                    result = preprocess_module.fit_transform(
                        train_df=X.train,
                        val_df=X.val,
                        test_df=combined_test,
                        config=preprocess_config,
                        orig_df=X.orig,
                    )
                else:
                    result = preprocess_module.fit_transform(
                        train_df=X.train,
                        val_df=X.val,
                        test_df=combined_test,
                        config=preprocess_config,
                    )

            # 4. Unpack result
            custom_preprocess_state = {}
            new_train, new_val, new_test, new_eval, new_orig = None, None, None, None, None

            if len(result) == 6:
                new_train, new_val, new_test, new_eval, new_orig, custom_preprocess_state = result
            elif len(result) == 5:
                new_train, new_val, combined_res, new_orig, custom_preprocess_state = result
                if not supports_eval_df and X.eval is not None:
                    new_test = combined_res.iloc[:test_len].copy()
                    new_eval = combined_res.iloc[test_len:].copy()
                else:
                    new_test = combined_res
            elif len(result) == 4:
                new_train, _, combined_res, custom_preprocess_state = result
                new_orig = None
                new_val = None
                if not supports_eval_df and X.eval is not None:
                    new_test = combined_res.iloc[:test_len].copy()
                    new_eval = combined_res.iloc[test_len:].copy()
                else:
                    new_test = combined_res

            # Update container with index reset to prevent alignment spikes
            if new_train is not None: X.train = new_train.reset_index(drop=True)
            if new_test is not None: X.test = new_test.reset_index(drop=True)
            if new_val is not None: X.val = new_val.reset_index(drop=True)
            if new_eval is not None: X.eval = new_eval.reset_index(drop=True)
            if new_orig is not None: X.orig = new_orig.reset_index(drop=True)
            
            # Feature Guard: Check for explosion immediately
            max_feats = getattr(self.context.config, "max_features_out", 5000)
            current_feats = X.train.shape[1]
            if current_feats > max_feats:
                raise RuntimeError(
                    f"[FEATURE_EXPLOSION] Step '{self.step_name}' generated {current_feats} features, "
                    f"exceeding the safety limit of {max_feats}."
                )

            # Explicitly free memory
            del new_train, new_test, new_val, new_eval, new_orig, result
            import gc
            gc.collect()
            
            # 5. Merge State
            if "custom_module_state" not in X.state:
                X.state["custom_module_state"] = {}
                
            for key in ["weights_path", "eval_path", "original_features"]:
                if key in custom_preprocess_state:
                    X.state["custom_module_state"][key] = custom_preprocess_state[key]
            
            step_info = {
                "step": self.step_name,
                "module": module_name,
                "state": custom_preprocess_state
            }
            if "pipeline_steps" not in X.state:
                X.state["pipeline_steps"] = []
            X.state["pipeline_steps"].append(step_info)

            # 6. Save Step-Specific state.json (for trackers/compatibility)
            # This satisfies the requirement that scripts depend on state.json in step folders
            step_payload = {
                "template": template_name or self.step_name,
                "input_source": None, # In-memory, no disk source
                "cached": False,
                "shapes": {
                    "train_before": shape_before_train,
                    "train_after": list(X.train.shape),
                    "test_before": shape_before_test,
                    "test_after": list(X.test.shape),
                    "val_before": shape_before_val,
                    "val_after": list(X.val.shape) if X.val is not None else None,
                    "eval_before": shape_before_eval,
                    "eval_after": list(X.eval.shape) if X.eval is not None else None,
                    "orig_before": shape_before_orig,
                    "orig_after": list(X.orig.shape) if X.orig is not None else None,
                },
                "custom_module_state": custom_preprocess_state
            }
            
            step_state_data = {
                "experiment_id": self.context.experiment_id,
                "project": self.context.project_name,
                "status": "completed",
                "modules": {
                    "preprocess": {
                        "status": "completed",
                        "payload": step_payload
                    }
                }
            }
            with open(step_dir / "state.json", "w") as f:
                json.dump(step_state_data, f, indent=2)

            # 7. Update Central State (Completed)
            duration = time.time() - start_time
            if self.context.state and self.context.state.pipeline_progress:
                progress = self.context.state.pipeline_progress
                if self.step_idx < len(progress.steps):
                    step_state = progress.steps[self.step_idx]
                    step_state.status = "completed"
                    step_state.finished_at = utc_now_iso()
                    step_state.duration = round(duration, 3)
                    step_state.shapes = step_payload["shapes"]
                    step_state.custom_module_state = custom_preprocess_state
                self.context.state.last_heartbeat = utc_now_iso()
                self.context.state.save()

            res_msg = Text()
            res_msg.append(" done", style="dim green")
            res_msg.append(f" ({duration:.1f}s)", style="gray")
            # Always use original stdout (unless json_output is enabled)
            is_quiet = False
            if self.context.config and getattr(self.context.config, "json_output", False):
                is_quiet = True
            console = Console(file=sys.__stdout__, force_terminal=True, highlight=False, quiet=is_quiet)
            console.print(res_msg)

        except Exception as e:
            if self.context.state and self.context.state.pipeline_progress:
                progress = self.context.state.pipeline_progress
                if self.step_idx < len(progress.steps):
                    step_state = progress.steps[self.step_idx]
                    step_state.status = "failed"
                    step_state.finished_at = utc_now_iso()
                self.context.state.status = "failed"
                self.context.state.save()
            console.print(f"[red]Error in step {self.step_name}:[/red] {e}")
            raise
            
        return X


@ModuleRegistry.register
class PreprocessModule(BaseModule):
    """Apply preprocessing defined by a template (supports chaining)."""

    name = "preprocess"
    description = "Data preprocessing"

    def can_run(self) -> tuple[bool, str]:
        """
        Validate that the referenced template exists before execution.

        Returns:
            Tuple of (is_valid, message) where ``is_valid`` is False when validation fails.
        """
        # Allow direct template config injection (used by Optuna/preprocess tuning)
        if self.invocation_params.get("preprocess_template_config") or self.invocation_params.get("preprocess_template_path"):
            return True, ""

        template_name = self.invocation_params.get("preprocess_template")
        if not template_name:
            return False, "Missing preprocess_template=<name> argument"

        # template_name should be a string, not a list
        if isinstance(template_name, list):
            return False, f"preprocess_template should be string, got list: {template_name}"

        # Check if template exists by trying to load it
        from pathlib import Path as P
        REPO_ROOT = P(__file__).resolve().parents[3]
        sys.path.insert(0, str(REPO_ROOT / "scripts"))

        try:
            from mlarena.core.config import TemplateLoader
            loader = TemplateLoader(self.context.project_root, template_type="preprocess")
            template_cfg = loader.load(template_name)
            if not template_cfg:
                return False, f"Template '{template_name}' not found."
        except Exception as e:
            import traceback
            return False, f"Failed to load template: {e}\n{traceback.format_exc()}"

        return True, ""

    def _drop_columns(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

    def _apply_template(self, df: pd.DataFrame, template: Dict[str, Any]) -> pd.DataFrame:
        # Basic operations: drop_columns, fillna
        drop_cols = template.get("drop_columns", [])
        if drop_cols:
            df = self._drop_columns(df, drop_cols)
        fills = template.get("fillna", {})
        if fills:
            df = df.fillna(fills)
        return df

    def _load_preprocessing_module(self, module_name: str):
        """
        Dynamically load preprocessing module.

        Search order:
        1. Project-local: {project}/code/preprocessing/{module_name}.py
        2. Global: src/mlarena/defaults/preprocessing/{module_name}.py
        """
        # Ensure project + global helper modules are importable inside dynamically loaded files.
        # Many project preprocess modules do `import adversarial_validation` (from code/utils)
        # or `import categorical_utils` (from mlarena/defaults/preprocessing). Without these paths,
        # imports fail because modules are loaded via spec_from_file_location (not a package).

        # Repository root (resolve from this file's location)
        from pathlib import Path as P
        repo_root = P(__file__).resolve().parents[3]  # src/mlarena/modules/preprocess.py -> ../../.. -> repo root

        extra_sys_paths = [
            # Project-local helpers (highest priority)
            self.context.project_root / "code" / "utils",
            self.context.project_root / "code" / "preprocessing",
            self.context.project_root / "code",
            # Global helpers
            repo_root / "src" / "mlarena" / "defaults" / "preprocessing",
            repo_root / "scripts",
        ]
        # Insert in reverse so the first item ends up with highest precedence.
        for path in reversed(extra_sys_paths):
            path_str = str(path)
            if path.exists() and path_str not in sys.path:
                sys.path.insert(0, path_str)

        local_path = self.context.project_root / "code" / "preprocessing" / f"{module_name}.py"
        global_path = repo_root / "src" / "mlarena" / "defaults" / "preprocessing" / f"{module_name}.py"

        local_exists = local_path.exists()
        global_exists = global_path.exists()

        # Ambiguity detection
        if local_exists and global_exists:
            raise RuntimeError(
                f"Ambiguous preprocessing module '{module_name}': exists in both\n"
                f"  - project: {local_path}\n"
                f"  - global:  {global_path}\n"
                f"Remove one to resolve ambiguity."
            )

        # Select path
        if local_exists:
            module_path = local_path
        elif global_exists:
            module_path = global_path
        else:
            raise FileNotFoundError(
                f"Preprocessing module '{module_name}' not found in:\n"
                f"  - project: {local_path}\n"
                f"  - global:  {global_path}"
            )

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if not spec or not spec.loader:
            raise RuntimeError(f"Unable to load module spec for {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _resolve_original_features(
        self,
        train_df: "pd.DataFrame",
        config_module,
        prev_custom_state: Dict[str, Any],
    ) -> List[str]:
        """Resolve original feature list from prior state, init/EDA, or current train_df."""
        orig_features = None
        if isinstance(prev_custom_state, dict):
            orig_features = prev_custom_state.get("original_features")

        id_column = getattr(config_module, "ID_COLUMN", "id")
        target_column = getattr(config_module, "TARGET_COLUMN", None)
        ignored_columns = getattr(config_module, "IGNORED_COLUMNS", []) or []
        exclude = {id_column, target_column, *ignored_columns}
        exclude = {c for c in exclude if c}

        if not orig_features:
            init_state = self.context.project_root / "experiments" / "init" / "state.json"
            if init_state.exists():
                try:
                    payload = json.loads(init_state.read_text())
                    cols = (
                        payload.get("modules", {})
                        .get("eda", {})
                        .get("profiles", {})
                        .get("train", {})
                        .get("columns")
                    )
                    if isinstance(cols, list) and cols:
                        orig_features = [c for c in cols if c not in exclude]
                except Exception:
                    orig_features = None

        if not orig_features:
            orig_features = [c for c in train_df.columns if c not in exclude]

        return orig_features

    def execute(self) -> ModuleResult:
        """
        Execute preprocessing for the provided template.

        Returns:
            ModuleResult capturing processed file locations and shape metadata.

        Raises:
            FileNotFoundError: When input data or referenced modules are missing.
            RuntimeError: When a custom preprocessing module fails to import or run.
        """
        import pandas as pd
        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        config = self.context.config_module or load_project_config(self.context.project_root)
        template_name = self.invocation_params.get("preprocess_template")
        if not template_name:
            # Allow dynamic templates; keep a stable label for logging/state
            if self.invocation_params.get("preprocess_template_config") or self.invocation_params.get("preprocess_template_path"):
                template_name = "inline"
            else:
                raise ValueError("preprocess_template is required")
        cache_ok = bool(self.invocation_params.get("cache"))
        input_source = self.invocation_params.get("input_source", None)
        chain_exp_id = self.invocation_params.get("chain_exp_id", f"pre-{template_name}")

        processed_train = artifact_dir / "train_processed.csv.gz"
        processed_test = artifact_dir / "test_processed.csv.gz"
        processed_tuning = artifact_dir / "tuning_processed.csv.gz"

        json_output = self.invocation_params.get("json_output", False)
        quiet_console = json_output
        use_stderr = False
        for key in ("quiet_preprocess_panel", "quiet_preprocess_panels", "quiet_preprocess"):
            if key in self.invocation_params and self.invocation_params.get(key) is False:
                quiet_console = False
                if json_output:
                    use_stderr = True
                break
        console = Console(
            force_terminal=True,
            quiet=quiet_console,
            file=sys.stderr if use_stderr else sys.stdout,
        )

        if cache_ok and processed_train.exists() and processed_test.exists():
            console.print(f"\n[bold yellow]Using cached preprocessed data[/bold yellow]")

            # Check if tuning and eval files exist in cache
            tuning_exists = processed_tuning.exists()
            processed_eval = artifact_dir / "eval_processed.csv.gz"
            eval_exists = processed_eval.exists()

            artifacts_list = [processed_train, processed_test]
            if tuning_exists:
                artifacts_list.append(processed_tuning)
            if eval_exists:
                artifacts_list.append(processed_eval)

            return ModuleResult(
                success=True,
                payload={
                    "train_processed": str(processed_train),
                    "test_processed": str(processed_test),
                    "tuning_processed": str(processed_tuning) if tuning_exists else None,
                    "cached": True,
                    "template": template_name,
                    "input_source": input_source,
                },
                artifacts=artifacts_list,
            )

        # Load input data (from previous preprocessing step or raw data)
        prev_custom_state = {}
        eval_path = None
        if input_source:
            # Load from previous preprocessing step (within same chain)
            # input_source already contains index (e.g., "0-noop")
            prev_exp_dir = self.context.project_root / "experiments" / chain_exp_id / input_source
            
            def _find_path(base_name: str) -> Optional[Path]:
                base = prev_exp_dir / "artifacts" / "preprocess" / base_name
                for ext in [".parquet", ".csv.gz", ".csv"]:
                    p = base.with_suffix(ext)
                    if p.exists():
                        return p
                return None
    
            train_path = _find_path("train_processed")
            test_path = _find_path("test_processed")
            orig_path = _find_path("orig_processed")
            tuning_path = _find_path("tuning_processed")
            eval_path = _find_path("eval_processed")
    
            if not train_path:
                raise FileNotFoundError(
                    f"Previous preprocessing output not found in: {prev_exp_dir}\n"
                    f"Chain broken: {input_source} must complete before {template_name}\n"
                    f"Chain experiment: {chain_exp_id}"
                )
            
            # Load previous state to propagate custom_module_state (e.g., weights_path)
            prev_state_path = prev_exp_dir / "state.json"
            if prev_state_path.exists():
                try:
                    with open(prev_state_path) as f:
                        prev_state = json.load(f)
                    
                    # Modules in chain steps are always named "preprocess"
                    prev_payload = prev_state.get("modules", {}).get("preprocess", {}).get("payload", {})
                    prev_custom_state = prev_payload.get("custom_module_state", {})
                except Exception:
                    pass
        else:
            # First step: load raw data
            train_path, test_path = data_paths(config)
            orig_path = None  # No orig at start of chain
            tuning_path = None  # No tuning at start of chain
            eval_path = None  # No eval at start of chain

            if not train_path.exists() or not test_path.exists():
                marker = artifact_dir / "preprocess_skipped.txt"
                marker.write_text("Missing train/test; preprocess skipped.")
                return ModuleResult(success=True, payload={"skipped": True, "input_source": input_source}, artifacts=[marker])

        def _read_df(p: Optional[Path]):
            if p is None: return None
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p, compression='infer')

        train_df = _read_df(train_path)
        test_df = _read_df(test_path)
        orig_df = _read_df(orig_path)
        tuning_df = _read_df(tuning_path)
        eval_df = _read_df(eval_path)

        # Store original shapes
        orig_train_shape = train_df.shape
        orig_test_shape = test_df.shape
        orig_orig_shape = orig_df.shape if orig_df is not None else None
        orig_tuning_shape = tuning_df.shape if tuning_df is not None else None
        orig_eval_shape = eval_df.shape if eval_df is not None else None

        # If eval_df exists, temporarily merge it with test_df to ensure it receives 
        # identical transformations (feature engineering, scaling, encoding, etc.)
        combined_test_df = test_df
        test_len = len(test_df)
        if eval_df is not None:
            combined_test_df = pd.concat([test_df, eval_df], axis=0).reset_index(drop=True)

        # Get ignored columns for later use
        ignored = getattr(config, "IGNORED_COLUMNS", []) or []

        # Resolve original feature set (used to avoid feature cascades)
        original_features = self._resolve_original_features(train_df, config, prev_custom_state)

        # Always suppress common preprocessing noise
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Load template config (supports override via invocation params)
        template_cfg = None
        template_cfg_override = self.invocation_params.get("preprocess_template_config")
        template_path = self.invocation_params.get("preprocess_template_path")

        if template_cfg_override:
            if not isinstance(template_cfg_override, dict):
                raise ValueError("preprocess_template_config must be a dict")
            template_cfg = template_cfg_override
        elif template_path:
            try:
                import yaml
                template_cfg = yaml.safe_load(Path(template_path).read_text()) or {}
            except Exception as e:
                raise RuntimeError(f"Failed to load preprocess_template_path: {template_path} ({e})")

        if template_cfg is None:
            from mlarena.core.config import TemplateLoader
            loader = TemplateLoader(self.context.project_root, template_type="preprocess")
            template_cfg = loader.load(template_name)
            if not template_cfg:
                raise ValueError(f"Template '{template_name}' not found.")

        # >>>>>> UNIFIED PIPELINE EXECUTION BLOCK START <<<<<<
        pipeline_definition = template_cfg.get("steps")
        is_classic = self.invocation_params.get("classic", False)
        quiet_mode = self.invocation_params.get("quiet_preprocess_panel")
        
        # Default behavior: Treat 'chain' as 'steps' for pipeline execution
        # unless classic mode is explicitly requested.
        if pipeline_definition is None and template_cfg.get("chain") and not is_classic:
            pipeline_definition = template_cfg.get("chain")

        # If steps is provided, we execute the whole chain in-memory
        if pipeline_definition:
            try:
                # 1. Initialize Container
                # Capture original shapes for reporting
                orig_train_shape = train_df.shape
                orig_test_shape = test_df.shape
                orig_orig_shape = orig_df.shape if orig_df is not None else None
                orig_tuning_shape = tuning_df.shape if tuning_df is not None else None
                orig_eval_shape = eval_df.shape if eval_df is not None else None

                container = MLArenaDataContainer(
                    train=train_df,
                    test=test_df,
                    val=tuning_df,
                    eval=eval_df,
                    orig=orig_df,
                    state={"custom_module_state": prev_custom_state.copy()},
                    initial_shapes={
                        "train": orig_train_shape,
                        "test": orig_test_shape,
                        "val": orig_tuning_shape,
                        "eval": orig_eval_shape,
                        "orig": orig_orig_shape,
                    }
                )
                
                # Critical: Propagate original features list
                if original_features:
                    if "custom_module_state" not in container.state:
                        container.state["custom_module_state"] = {}
                    container.state["custom_module_state"]["original_features"] = original_features
                
                # 2. Build Pipeline & Initialize State
                pipeline_steps = []
                state_steps = []
                for idx, step_item in enumerate(pipeline_definition):
                    if isinstance(step_item, str):
                        step_cfg = {"template": step_item}
                        step_name = f"{idx}-{step_item}"
                        module_name_for_step = step_item # Fallback
                    elif isinstance(step_item, dict):
                        step_cfg = step_item
                        raw_name = step_cfg.get("name") or step_cfg.get("template") or f"step_{idx}"
                        # Robust check for prefix (e.g. "0-", "10-")
                        if not (raw_name and re.match(r"^\d+-", str(raw_name))):
                            step_name = f"{idx}-{raw_name}"
                        else:
                            step_name = raw_name
                        module_name_for_step = step_cfg.get("module") or step_cfg.get("template") or "unknown"
                    else:
                        raise ValueError(f"Invalid step format: {step_item}")

                    # Pass loader instance for module loading
                    wrapper = MLArenaStepWrapper(step_name, idx, step_cfg, self.context, self)
                    pipeline_steps.append((step_name, wrapper))
                    state_steps.append(PipelineStep(name=step_name, module=module_name_for_step))
                
                if self.context.state:
                    # IMPORTANT: Move state.json to parent directory (hash level) for unified pipeline
                    # as per docs/preprocessing_state_schema.md
                    original_exp_dir = self.context.state.experiment_dir
                    self.context.state.experiment_dir = original_exp_dir.parent
                    
                    # Update experiment_id to reflect hash level (remove the last component)
                    exp_id_parts = self.context.state.experiment_id.split("/")
                    if len(exp_id_parts) > 1:
                        self.context.state.experiment_id = "/".join(exp_id_parts[:-1])

                    self.context.state.pipeline_progress = PipelineProgress(
                        status="running",
                        total_steps=len(pipeline_definition),
                        current_step_idx=0,
                        start_time=utc_now_iso(),
                        steps=state_steps
                    )
                    self.context.state.status = "running"
                    self.context.state.save()

                pipeline = Pipeline(pipeline_steps)
                
                # 3. Execute
                if quiet_mode:
                    with contextlib.redirect_stdout(io.StringIO()):
                        final_container = pipeline.fit_transform(container)
                else:
                    final_container = pipeline.fit_transform(container)
                
                # 4. Finalize State
                if self.context.state and self.context.state.pipeline_progress:
                    self.context.state.pipeline_progress.status = "completed"
                    self.context.state.pipeline_progress.end_time = utc_now_iso()
                    self.context.state.status = "completed"
                    self.context.state.save()

                # 5. Determine Target Directory
                target_dir = self.context.artifact_dir
                target_dir.mkdir(parents=True, exist_ok=True)
                
                last_step_full_name = pipeline_steps[-1][0]
                
                if not quiet_mode:
                    console.print(f"\n[bold green]Pipeline Completed.[/bold green] Saving final artifacts to [cyan]{last_step_full_name}[/cyan]...")
                
                # Use Parquet for performance
                final_container.train.to_parquet(target_dir / "train_processed.parquet", index=False)
                final_container.test.to_parquet(target_dir / "test_processed.parquet", index=False)
                
                processed_eval = None
                if final_container.eval is not None:
                    processed_eval = target_dir / "eval_processed.parquet"
                    final_container.eval.to_parquet(processed_eval, index=False)
                    if "custom_module_state" not in final_container.state:
                        final_container.state["custom_module_state"] = {}
                    final_container.state["custom_module_state"]["eval_path"] = str(processed_eval)
                    
                processed_orig = None
                if final_container.orig is not None:
                    processed_orig = target_dir / "orig_processed.parquet"
                    final_container.orig.to_parquet(processed_orig, index=False)
                    
                processed_tuning_final = None
                if final_container.val is not None:
                    processed_tuning_final = target_dir / "tuning_processed.parquet"
                    final_container.val.to_parquet(processed_tuning_final, index=False)
                    
                shapes_dict = {
                    "train_before": final_container.initial_shapes.get("train"),
                    "train_after": final_container.train.shape,
                    "test_before": final_container.initial_shapes.get("test"),
                    "test_after": final_container.test.shape,
                    "val_before": final_container.initial_shapes.get("val"),
                    "val_after": final_container.val.shape if final_container.val is not None else None,
                    "eval_before": final_container.initial_shapes.get("eval"),
                    "eval_after": final_container.eval.shape if final_container.eval is not None else None,
                    "orig_before": final_container.initial_shapes.get("orig"),
                    "orig_after": final_container.orig.shape if final_container.orig is not None else None,
                    "pipeline_mode": True,
                    "last_step": last_step_full_name,
                }
                
                payload = {
                    "train_processed": str((target_dir / "train_processed.parquet").resolve()),
                    "test_processed": str((target_dir / "test_processed.parquet").resolve()),
                    "orig_processed": str((target_dir / "orig_processed.parquet").resolve()) if final_container.orig is not None else None,
                    "tuning_processed": str((target_dir / "tuning_processed.parquet").resolve()) if final_container.val is not None else None,
                    "eval_processed": str((target_dir / "eval_processed.parquet").resolve()) if final_container.eval is not None else None,
                    "ignored_columns": ignored,
                    "template": template_name,
                    "input_source": input_source,
                    "cached": False,
                    "shapes": shapes_dict,
                    "custom_module_state": final_container.state.get("custom_module_state", {}),
                    "pipeline_steps": final_container.state.get("pipeline_steps", [])
                }
                
                artifacts_list = [target_dir / "train_processed.parquet", target_dir / "test_processed.parquet"]
                if final_container.orig is not None: artifacts_list.append(target_dir / "orig_processed.parquet")
                if final_container.val is not None: artifacts_list.append(target_dir / "tuning_processed.parquet")
                if final_container.eval is not None: artifacts_list.append(target_dir / "eval_processed.parquet")
                
                return ModuleResult(success=True, payload=payload, artifacts=artifacts_list)

            except Exception as e:
                # CATCH ALL ERRRORS (including Feature Explosion)
                error_msg = str(e)
                # Fallback shapes if container exists
                shapes_dict = {}
                if 'container' in locals():
                    shapes_dict = {
                        "train_before": container.initial_shapes.get("train"),
                        "train_after": container.train.shape if hasattr(container, 'train') else None,
                    }
                
                return ModuleResult(
                    success=False, 
                    error=error_msg, 
                    payload={"shapes": shapes_dict, "exception": type(e).__name__}
                )
        # >>>>>> UNIFIED PIPELINE EXECUTION BLOCK END <<<<<<

        custom_module_name = template_cfg.get("module") if template_cfg else None

        # Use custom preprocessing module if specified (not None and not empty string)
        custom_preprocess_state = {}
        is_pass_through = False
        if custom_module_name:
            try:
                preprocess_module = self._load_preprocessing_module(custom_module_name)

                # Check if module is pass-through (does not modify train/test)
                is_pass_through = getattr(preprocess_module, "PASS_THROUGH", False)

                # Prepare config with artifact_dir
                preprocess_config = template_cfg.get("config", {}).copy()
                preprocess_config["_artifact_dir"] = str(artifact_dir)
                preprocess_config["_dataset"] = {
                    "id_column": getattr(config, "ID_COLUMN", "id"),
                    "target": getattr(config, "TARGET_COLUMN", None),
                    "ignored_columns": getattr(config, "IGNORED_COLUMNS", []),
                    "problem_type": getattr(config, "AUTOGLUON_PROBLEM_TYPE", "binary"),
                }
                preprocess_config["_original_features"] = original_features

                # Inspect signature to determine supported arguments
                import inspect
                fit_sig = inspect.signature(preprocess_module.fit_transform)
                params = fit_sig.parameters
                has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                
                supports_eval_df = "eval_df" in params or has_kwargs
                supports_orig_df = "orig_df" in params or has_kwargs

                # Execute fit_transform
                if supports_eval_df:
                    # New interface: pass eval_df explicitly
                    result = preprocess_module.fit_transform(
                        train_df=train_df,
                        val_df=tuning_df,
                        test_df=test_df,
                        eval_df=eval_df,
                        config=preprocess_config,
                        orig_df=orig_df,
                    )
                else:
                    # Legacy interface: combined test+eval
                    # Create combined_test_df only if needed for legacy modules
                    combined_test_df = test_df
                    if eval_df is not None:
                        combined_test_df = pd.concat([test_df, eval_df], axis=0).reset_index(drop=True)

                    if supports_orig_df:
                        result = preprocess_module.fit_transform(
                            train_df=train_df,
                            val_df=tuning_df,
                            test_df=combined_test_df,
                            config=preprocess_config,
                            orig_df=orig_df,
                        )
                    else:
                        result = preprocess_module.fit_transform(
                            train_df=train_df,
                            val_df=tuning_df,
                            test_df=combined_test_df,
                            config=preprocess_config,
                        )

                # Handle return values
                # Expected signatures:
                # 6-tuple: (train, val, test, eval, orig, state)  [New Standard]
                # 5-tuple: (train, val, test, orig, state)        [Previous Standard]
                # 4-tuple: (train, val, test, state)              [Old Legacy]

                if len(result) == 6:
                    train_df, tuning_df, test_df, eval_returned, orig_df, custom_preprocess_state = result
                    if eval_returned is not None:
                        eval_df = eval_returned
                elif len(result) == 5:
                    # Check if this is (train, val, test, orig, state) or (train, val, test, eval, state)?
                    # Standard assumption for 5-tuple is the previous standard: (train, val, test, orig, state)
                    train_df, tuning_df, combined_result, orig_df, custom_preprocess_state = result
                    
                    # If we used legacy mode (concat), we must split combined_result
                    if not supports_eval_df and eval_df is not None:
                        test_df = combined_result.iloc[:test_len].copy()
                        eval_df = combined_result.iloc[test_len:].copy()
                    else:
                        test_df = combined_result
                elif len(result) == 4:
                    train_df, _, combined_result, custom_preprocess_state = result
                    orig_df = None
                    tuning_df = None
                    
                    if not supports_eval_df and eval_df is not None:
                        test_df = combined_result.iloc[:test_len].copy()
                        eval_df = combined_result.iloc[test_len:].copy()
                    else:
                        test_df = combined_result
                else:
                    raise ValueError(
                        f"Invalid fit_transform return signature: expected 4, 5, or 6 values, got {len(result)}"
                    )

            except Exception as e:
                console.print(f"[red]Error in custom preprocessing:[/red] {e}")
                raise
        else:
            # Fallback: basic template operations (no custom module)
            # Combine for basic ops if needed, or handle separately?
            # Basic ops use internal methods _drop_columns etc.
            # We can apply to eval_df directly if it exists.
            
            # Drop IGNORED_COLUMNS first for basic preprocessing
            if ignored:
                train_df = self._drop_columns(train_df, ignored)
                test_df = self._drop_columns(test_df, ignored)
                if eval_df is not None:
                    eval_df = self._drop_columns(eval_df, ignored)

            if template_cfg:
                operations = []
                if template_cfg.get("drop_columns"):
                    operations.append(f"drop {len(template_cfg['drop_columns'])} columns")
                if template_cfg.get("fillna"):
                    operations.append(f"fillna ({len(template_cfg['fillna'])} columns)")
                if operations:
                    console.print(f"[bold]Template operations:[/bold] {', '.join(operations)}")

                train_df = self._apply_template(train_df, template_cfg)
                test_df = self._apply_template(test_df, template_cfg)
                if eval_df is not None:
                    eval_df = self._apply_template(eval_df, template_cfg)


        train_df.to_csv(processed_train, index=False, compression='infer')
        test_df.to_csv(processed_test, index=False, compression='infer')

        # Save eval if present
        processed_eval = None
        if eval_df is not None:
            processed_eval = artifact_dir / "eval_processed.csv.gz"
            eval_df.to_csv(processed_eval, index=False, compression='infer')
            # Important: update eval_path in custom_preprocess_state so model.py knows where to find the TRANSFORMED file
            if custom_preprocess_state is None:
                custom_preprocess_state = {}
            custom_preprocess_state["eval_path"] = str(processed_eval)
            # Keep eval metadata in sync for display/debugging
            custom_preprocess_state["eval_rows"] = int(eval_df.shape[0])
            custom_preprocess_state["eval_cols"] = int(eval_df.shape[1])

        # Save orig if present
        processed_orig = None
        if orig_df is not None:
            processed_orig = artifact_dir / "orig_processed.csv.gz"
            orig_df.to_csv(processed_orig, index=False, compression='infer')

        # Save tuning if present
        processed_tuning_final = None
        if tuning_df is not None:
            processed_tuning_final = artifact_dir / "tuning_processed.csv.gz"
            tuning_df.to_csv(processed_tuning_final, index=False, compression='infer')

        # Prepare payload with custom preprocessing state
        # For pass-through modules, shapes before/after are the same (no modification)
        if is_pass_through:
            shapes_dict = {
                "train_before": orig_train_shape,
                "train_after": orig_train_shape,  # Same as input
                "test_before": orig_test_shape,
                "test_after": orig_test_shape,  # Same as input
                "orig_before": orig_orig_shape,
                "orig_after": orig_orig_shape,  # Same as input
                "tuning_before": orig_tuning_shape,
                "tuning_after": orig_tuning_shape,  # Same as input
                "eval_before": orig_eval_shape,
                "eval_after": orig_eval_shape,
                "pass_through": True,  # Flag for display
            }
        else:
            shapes_dict = {
                "train_before": orig_train_shape,
                "train_after": train_df.shape,
                "test_before": orig_test_shape,
                "test_after": test_df.shape,
                "orig_before": orig_orig_shape,
                "orig_after": orig_df.shape if orig_df is not None else None,
                "tuning_before": orig_tuning_shape,
                "tuning_after": tuning_df.shape if tuning_df is not None else None,
                "eval_before": orig_eval_shape,
                "eval_after": eval_df.shape if eval_df is not None else None,
                "pass_through": False,
            }

        payload = {
            "train_processed": str(processed_train),
            "test_processed": str(processed_test),
            "orig_processed": str(processed_orig) if processed_orig else None,
            "tuning_processed": str(processed_tuning_final) if processed_tuning_final else None,
            "eval_processed": str(processed_eval) if processed_eval else None,
            "ignored_columns": ignored,
            "template": template_name,
            "input_source": input_source,  # Track previous step in chain
            "cached": False,
            "shapes": shapes_dict,
        }

        # Add custom module state (e.g., av_weights_path)
        # Merge previous custom state with current (current takes precedence)
        final_custom_state = prev_custom_state.copy()
        if original_features and "original_features" not in final_custom_state:
            final_custom_state["original_features"] = original_features
        if custom_preprocess_state:
            final_custom_state.update(custom_preprocess_state)

        if final_custom_state:
            payload["custom_module_state"] = final_custom_state

        # Next steps will be printed by pipeline after footer (only for last in chain)

        # Build artifacts list (include orig, tuning, and eval if present)
        artifacts_list = [processed_train, processed_test]
        if processed_orig:
            artifacts_list.append(processed_orig)
        if processed_tuning_final:
            artifacts_list.append(processed_tuning_final)
        if processed_eval:
            artifacts_list.append(processed_eval)

        return ModuleResult(
            success=True,
            payload=payload,
            artifacts=artifacts_list,
        )

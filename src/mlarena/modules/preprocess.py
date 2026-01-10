"""Data preprocessing module.

Executes a single preprocessing step defined by a template, supports chains, caching, and custom modules.
"""

from __future__ import annotations

import importlib.util
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from mlarena.core.module import BaseModule, ModuleResult
from mlarena.core.registry import ModuleRegistry
from mlarena.core.config import TemplateLoader
from mlarena.utils.project import data_paths, load_project_config


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
            return False, "Missing --preprocess-template argument"

        # template_name should be a string, not a list
        if isinstance(template_name, list):
            return False, f"preprocess_template should be string, got list: {template_name}"

        # Check if template exists by trying to load it
        import sys
        from pathlib import Path as P
        REPO_ROOT = P(__file__).resolve().parents[3]
        sys.path.insert(0, str(REPO_ROOT / "scripts"))

        try:
            from template_loader import load_templates
            templates, _ = load_templates("preprocess", self.context.project_root, suppress_warnings=True)
            if template_name not in templates:
                available = ", ".join(list(templates.keys())[:10]) if templates else "none"
                return False, f"Template '{template_name}' not found. Available: {available}"
        except Exception as e:
            import traceback
            return False, f"Failed to load templates: {e}\n{traceback.format_exc()}"

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
        import sys

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
                raise ValueError("--preprocess-template is required")
        cache_ok = bool(self.invocation_params.get("cache"))
        input_source = self.invocation_params.get("input_source", None)
        chain_exp_id = self.invocation_params.get("chain_exp_id", f"pre-{template_name}")

        processed_train = artifact_dir / "train_processed.csv.gz"
        processed_test = artifact_dir / "test_processed.csv.gz"
        processed_tuning = artifact_dir / "tuning_processed.csv.gz"

        console = Console(force_terminal=True)

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
            train_path = prev_exp_dir / "artifacts" / "preprocess" / "train_processed.csv.gz"
            test_path = prev_exp_dir / "artifacts" / "preprocess" / "test_processed.csv.gz"
            orig_path = prev_exp_dir / "artifacts" / "preprocess" / "orig_processed.csv.gz"
            tuning_path = prev_exp_dir / "artifacts" / "preprocess" / "tuning_processed.csv.gz"
            eval_path = prev_exp_dir / "artifacts" / "preprocess" / "eval_processed.csv.gz"

            if not train_path.exists():
                raise FileNotFoundError(
                    f"Previous preprocessing output not found: {train_path}\n"
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

        train_df = pd.read_csv(train_path, compression='infer')
        test_df = pd.read_csv(test_path, compression='infer')
        orig_df = pd.read_csv(orig_path, compression='infer') if orig_path and orig_path.exists() else None
        tuning_df = pd.read_csv(tuning_path, compression='infer') if tuning_path and tuning_path.exists() else None
        eval_df = pd.read_csv(eval_path, compression='infer') if eval_path and eval_path.exists() else None

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
            import sys
            from pathlib import Path as P
            REPO_ROOT = P(__file__).resolve().parents[3]
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
            from template_loader import load_templates

            templates, _ = load_templates("preprocess", self.context.project_root, suppress_warnings=True)
            template_cfg = templates.get(template_name, {})

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
                console.print(f"\n[bold]Dropping columns:[/bold] {', '.join(f'[yellow]{c}[/yellow]' for c in ignored)}")
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


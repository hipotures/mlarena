"""Data preprocessing module."""

from __future__ import annotations

import importlib.util
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
    name = "preprocess"
    description = "Data preprocessing"

    @classmethod
    def register_cli_args(cls, parser) -> None:
        parser.add_argument("--preprocess-template", type=str, required=True, help="Name of preprocessing template (required to avoid mistakes).")
        parser.add_argument("--cache", action="store_true", help="Reuse existing processed files if present.")

    def can_run(self) -> tuple[bool, str]:
        """Validate that template exists before running."""
        template_name = self.invocation_params.get("preprocess_template")
        if not template_name:
            return False, "Missing --preprocess-template argument"

        # Check if template exists by trying to load it
        import sys
        from pathlib import Path as P
        REPO_ROOT = P(__file__).resolve().parents[3]
        sys.path.insert(0, str(REPO_ROOT / "scripts"))

        try:
            from template_loader import load_templates
            templates, _ = load_templates("preprocess", self.context.project_root, suppress_warnings=True)
            if template_name not in templates:
                available = ", ".join(templates.keys()) if templates else "none"
                return False, f"Template '{template_name}' not found. Available: {available}"
        except Exception as e:
            return False, f"Failed to load templates: {e}"

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
        2. Global: config/code/preprocessing/{module_name}.py
        """
        # Repository root (resolve from this file's location)
        from pathlib import Path as P
        repo_root = P(__file__).resolve().parents[3]  # src/mlarena/modules/preprocess.py -> ../../.. -> repo root

        local_path = self.context.project_root / "code" / "preprocessing" / f"{module_name}.py"
        global_path = repo_root / "config" / "code" / "preprocessing" / f"{module_name}.py"

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

    def execute(self) -> ModuleResult:
        import pandas as pd
        artifact_dir: Path = self.context.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        config = self.context.config_module or load_project_config(self.context.project_root)
        template_name = self.invocation_params.get("preprocess_template")
        if not template_name:
            raise ValueError("--preprocess-template is required")
        cache_ok = bool(self.invocation_params.get("cache"))
        input_source = self.invocation_params.get("input_source", None)

        processed_train = artifact_dir / "train_processed.csv"
        processed_test = artifact_dir / "test_processed.csv"

        console = Console(force_terminal=True)

        if cache_ok and processed_train.exists() and processed_test.exists():
            console.print(f"\n[bold yellow]Using cached preprocessed data[/bold yellow]")

            return ModuleResult(
                success=True,
                payload={
                    "train_processed": str(processed_train),
                    "test_processed": str(processed_test),
                    "cached": True,
                    "template": template_name,
                    "input_source": input_source,
                },
                artifacts=[processed_train, processed_test],
            )

        # Load input data (from previous preprocessing step or raw data)
        if input_source:
            # Load from previous preprocessing step
            prev_exp_dir = self.context.project_root / "experiments" / f"pre-{input_source}"
            train_path = prev_exp_dir / "artifacts" / "preprocess" / "train_processed.csv"
            test_path = prev_exp_dir / "artifacts" / "preprocess" / "test_processed.csv"

            if not train_path.exists():
                raise FileNotFoundError(
                    f"Previous preprocessing output not found: {train_path}\n"
                    f"Chain broken: pre-{input_source} must complete before pre-{template_name}"
                )
        else:
            # First step: load raw data
            train_path, test_path = data_paths(config)

            if not train_path.exists() or not test_path.exists():
                marker = artifact_dir / "preprocess_skipped.txt"
                marker.write_text("Missing train/test; preprocess skipped.")
                return ModuleResult(success=True, payload={"skipped": True, "input_source": input_source}, artifacts=[marker])

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Store original shapes
        orig_train_shape = train_df.shape
        orig_test_shape = test_df.shape

        # Get ignored columns for later use
        ignored = getattr(config, "IGNORED_COLUMNS", []) or []

        # Load template config using template_loader (same as can_run)
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
        if custom_module_name:
            try:
                preprocess_module = self._load_preprocessing_module(custom_module_name)

                # Prepare config with artifact_dir
                preprocess_config = template_cfg.get("config", {}).copy()
                preprocess_config["_artifact_dir"] = str(artifact_dir)
                preprocess_config["_dataset"] = {
                    "id_column": getattr(config, "ID_COLUMN", "id"),
                    "target": getattr(config, "TARGET_COLUMN", None),
                    "ignored_columns": getattr(config, "IGNORED_COLUMNS", []),
                }

                # Call fit_transform(train, val, test, config)
                train_df, val_df, test_df, custom_preprocess_state = preprocess_module.fit_transform(
                    train_df=train_df,
                    val_df=None,
                    test_df=test_df,
                    config=preprocess_config
                )
            except Exception as e:
                console.print(f"[red]Error in custom preprocessing:[/red] {e}")
                raise
        else:
            # Fallback: basic template operations (no custom module)
            # Drop IGNORED_COLUMNS first for basic preprocessing
            if ignored:
                console.print(f"\n[bold]Dropping columns:[/bold] {', '.join(f'[yellow]{c}[/yellow]' for c in ignored)}")
                train_df = self._drop_columns(train_df, ignored)
                test_df = self._drop_columns(test_df, ignored)

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

        train_df.to_csv(processed_train, index=False)
        test_df.to_csv(processed_test, index=False)

        # Prepare payload with custom preprocessing state
        payload = {
            "train_processed": str(processed_train),
            "test_processed": str(processed_test),
            "ignored_columns": ignored,
            "template": template_name,
            "input_source": input_source,  # Track previous step in chain
            "cached": False,
            "shapes": {
                "train_before": orig_train_shape,
                "train_after": train_df.shape,
                "test_before": orig_test_shape,
                "test_after": test_df.shape,
            }
        }

        # Add custom module state (e.g., av_weights_path)
        if custom_preprocess_state:
            payload["custom_module_state"] = custom_preprocess_state

        # Print next steps only if last in chain (footer is handled by pipeline)
        is_last_in_chain = self.invocation_params.get("is_last_in_chain", True)  # Default True for backwards compatibility
        if is_last_in_chain:
            from mlarena.core.module import print_next_steps
            print_next_steps("preprocess", self.context.project_name, self.context.experiment_id, console)

        return ModuleResult(
            success=True,
            payload=payload,
            artifacts=[processed_train, processed_test],
        )

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
        """Dynamically load preprocessing module from code/preprocessing/{module_name}.py"""
        module_path = self.context.project_root / "code" / "preprocessing" / f"{module_name}.py"
        if not module_path.exists():
            raise FileNotFoundError(f"Preprocessing module not found: {module_path}")

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
        train_path, test_path = data_paths(config)
        template_name = self.invocation_params.get("preprocess_template")
        if not template_name:
            raise ValueError("--preprocess-template is required")
        cache_ok = bool(self.invocation_params.get("cache"))

        processed_train = artifact_dir / "train_processed.csv"
        processed_test = artifact_dir / "test_processed.csv"

        console = Console(force_terminal=True)

        if cache_ok and processed_train.exists() and processed_test.exists():
            console.print(f"\n[bold yellow]Using cached preprocessed data[/bold yellow]")
            console.print(f"  Template: [cyan]{template_name}[/cyan]")
            console.print(f"  Train: [dim]{processed_train.relative_to(self.context.project_root)}[/dim]")
            console.print(f"  Test:  [dim]{processed_test.relative_to(self.context.project_root)}[/dim]")

            return ModuleResult(
                success=True,
                payload={
                    "train_processed": str(processed_train),
                    "test_processed": str(processed_test),
                    "cached": True,
                    "template": template_name,
                },
                artifacts=[processed_train, processed_test],
            )

        if not train_path.exists() or not test_path.exists():
            marker = artifact_dir / "preprocess_skipped.txt"
            marker.write_text("Missing train/test; preprocess skipped.")
            return ModuleResult(success=True, payload={"skipped": True}, artifacts=[marker])

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Store original shapes
        orig_train_shape = train_df.shape
        orig_test_shape = test_df.shape

        # Show preprocessing info
        console.print(f"\n[bold]Template:[/bold] [cyan]{template_name}[/cyan]")

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
            console.print(f"[bold]Custom module:[/bold] [cyan]{custom_module_name}[/cyan]")

            try:
                preprocess_module = self._load_preprocessing_module(custom_module_name)
                console.print(f"[green]✓[/green] Loaded preprocessing module: {custom_module_name}")

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

                console.print(f"[green]✓[/green] Custom preprocessing completed")
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

        # Show shape changes
        shape_table = Table(title="Dataset Shapes", show_header=True)
        shape_table.add_column("Dataset", style="cyan")
        shape_table.add_column("Before", style="dim")
        shape_table.add_column("After", style="green")
        shape_table.add_row(
            "Train",
            f"{orig_train_shape[0]:,} × {orig_train_shape[1]}",
            f"{train_df.shape[0]:,} × {train_df.shape[1]}"
        )
        shape_table.add_row(
            "Test",
            f"{orig_test_shape[0]:,} × {orig_test_shape[1]}",
            f"{test_df.shape[0]:,} × {test_df.shape[1]}"
        )
        console.print(shape_table)

        train_df.to_csv(processed_train, index=False)
        test_df.to_csv(processed_test, index=False)

        # Show output files
        console.print(f"\n[bold green]✓[/bold green] Preprocessed files saved:")
        console.print(f"  Train: [cyan]{processed_train.relative_to(self.context.project_root)}[/cyan]")
        console.print(f"  Test:  [cyan]{processed_test.relative_to(self.context.project_root)}[/cyan]")

        # Prepare payload with custom preprocessing state
        payload = {
            "train_processed": str(processed_train),
            "test_processed": str(processed_test),
            "ignored_columns": ignored,
            "template": template_name,
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

        return ModuleResult(
            success=True,
            payload=payload,
            artifacts=[processed_train, processed_test],
        )

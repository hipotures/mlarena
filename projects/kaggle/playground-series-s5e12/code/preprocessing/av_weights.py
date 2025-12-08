"""
Preprocessing hook that computes adversarial validation weights and saves them.

Features are returned unchanged; weights are written to CSV so they can be used
as sample weights or for CV design. Uses compute_adversarial_weights().

Reads categorical_columns from previous preprocessing step (if available) to ensure
correct dtype='category' before AV training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json

import pandas as pd
from uuid import uuid4
import sys
import yaml

from adversarial_validation import compute_adversarial_weights


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "train_av_weights.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "preprocess_cache" / "av_model"
AV_PROB_COL = "av_prob"


def _resolve_path(path: str | Path | None, default: Path, base: Path) -> Path:
    """
    Resolve a path allowing per-run artifact dir overrides.

    - Absolute paths are returned as-is.
    - Relative paths are resolved against `base` (artifact_dir from MLArena),
      not the project root, so temporary models land under experiments/<id>/.
    - When `path` is None, `default` is returned.
    """
    if path is None:
        return default
    p = Path(path)
    if p.is_absolute():
        return p
    return base / p


def _unique_subdir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return base / f"run_{uuid4().hex[:8]}"


def _parse_cli_overrides() -> Dict[str, Any]:
    """Best-effort parse of model CLI flags to mirror main model settings."""
    args = sys.argv[1:]
    out: Dict[str, Any] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--preset", "--presets") and i + 1 < len(args):
            out["preset"] = args[i + 1]
            i += 1
        elif arg == "--time-limit" and i + 1 < len(args):
            try:
                out["time_limit"] = int(args[i + 1])
            except ValueError:
                pass
            i += 1
        elif arg in ("--model-template", "--template") and i + 1 < len(args):
            out["model_template"] = args[i + 1]
            i += 1
        i += 1
    return out


def _load_template_params(template_name: Optional[str]) -> Dict[str, Any]:
    if not template_name:
        return {}
    path = PROJECT_ROOT / "templates" / "model.yaml"
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
        templates = data.get("templates", {})
        tmpl = templates.get(template_name, {})
        hyper = (tmpl.get("config") or {}).get("hyperparameters") or {}
        return {
            "preset": hyper.get("presets"),
            "time_limit": hyper.get("time_limit"),
        }
    except Exception:
        return {}


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict[str, Any]]:
    cfg = config or {}
    artifact_base = Path(cfg.get("_artifact_dir") or PROJECT_ROOT)
    ds_cfg = cfg.get("_dataset", {}) or {}
    id_column: str = ds_cfg.get("id_column") or cfg.get("id_column") or "id"
    target_column: Optional[str] = ds_cfg.get("target") or cfg.get("target_column")
    cli_overrides = _parse_cli_overrides()
    tmpl_overrides = _load_template_params(cli_overrides.get("model_template"))

    drop_columns: List[str] = []
    if "drop_columns" in cfg and cfg["drop_columns"] is not None:
        drop_columns = list(cfg["drop_columns"])
    else:
        ignored = ds_cfg.get("ignored_columns") or []
        drop_columns = list(ignored)
        drop_columns.append(id_column)

    presets = (
        cli_overrides.get("preset")
        or cfg.get("presets")
        or tmpl_overrides.get("preset")
        or "medium_quality_faster_train"
    )
    time_limit = int(
        cli_overrides.get("time_limit")
        or cfg.get("time_limit")
        or tmpl_overrides.get("time_limit")
        or 600
    )
    included_model_types = cfg.get("included_model_types")
    output_path = _resolve_path(cfg.get("output_path"), DEFAULT_OUTPUT, artifact_base)
    base_model_dir = _resolve_path(cfg.get("model_output_dir"), DEFAULT_MODEL_DIR, artifact_base)
    model_output_dir = _unique_subdir(base_model_dir)

    av_df = compute_adversarial_weights(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        target_column=target_column,
        drop_columns=drop_columns,
        presets=presets,
        time_limit=time_limit,
        included_model_types=included_model_types,
        output_dir=model_output_dir,
    )

    # Transform weights: p/(1-p) -> clip to 2.0 -> normalize to mean 1.0
    p = av_df[AV_PROB_COL].clip(lower=1e-9, upper=1 - 1e-9)
    weights = p / (1 - p)
    weights = weights.clip(upper=2.0)
    weights = weights / weights.mean()
    av_df[AV_PROB_COL] = weights

    output_path.parent.mkdir(parents=True, exist_ok=True)
    av_df.to_csv(output_path, index=False)

    state: Dict[str, Any] = {
        "version": "1.0",
        "presets": presets,
        "time_limit": time_limit,
        "included_model_types": included_model_types,
        "drop_columns": drop_columns,
        "id_column": id_column,
        "target_column": target_column,
        "weights_path": str(output_path),
        "model_output_dir": str(model_output_dir),
    }
    return train_df.copy(), None if val_df is None else val_df.copy(), test_df.copy(), state


def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    return df.copy()

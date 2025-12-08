"""
Preprocessing hook that computes adversarial validation weights and saves them.

Features are returned unchanged; weights are written to CSV so they can be used
as sample weights or for CV design. Uses compute_adversarial_weights().
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import sys

# Add code directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.adversarial_validation import compute_adversarial_weights


AV_PROB_COL = "av_prob"


def _resolve_path(path: str | Path | None, default_name: str, artifact_dir: Path) -> Path:
    """Resolve path relative to artifact_dir unless absolute."""
    if path is None:
        return artifact_dir / default_name
    p = Path(path)
    return p if p.is_absolute() else artifact_dir / p


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict[str, Any]]:
    cfg = config or {}
    ds_cfg = cfg.get("_dataset", {}) or {}
    artifact_dir = Path(cfg.get("_artifact_dir", "."))

    id_column: str = ds_cfg.get("id_column") or cfg.get("id_column") or "id"
    target_column: Optional[str] = ds_cfg.get("target") or cfg.get("target_column")

    drop_columns: List[str] = []
    if "drop_columns" in cfg and cfg["drop_columns"] is not None:
        drop_columns = list(cfg["drop_columns"])
    else:
        ignored = ds_cfg.get("ignored_columns") or []
        drop_columns = list(ignored)
        drop_columns.append(id_column)

    presets = cfg.get("presets", "best")
    time_limit = int(cfg.get("time_limit", 600))

    output_path = _resolve_path(cfg.get("output_path"), "train_av_weights.csv", artifact_dir)
    model_output_dir = _resolve_path(cfg.get("model_output_dir"), "av_model", artifact_dir)

    av_df = compute_adversarial_weights(
        train_df=train_df,
        test_df=test_df,
        id_column=id_column,
        target_column=target_column,
        drop_columns=drop_columns,
        presets=presets,
        time_limit=time_limit,
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
        "drop_columns": drop_columns,
        "id_column": id_column,
        "target_column": target_column,
        "weights_path": str(output_path),
        "model_output_dir": str(model_output_dir),
    }
    return train_df.copy(), None if val_df is None else val_df.copy(), test_df.copy(), state


def transform(df: pd.DataFrame, state_dict: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
    return df.copy()

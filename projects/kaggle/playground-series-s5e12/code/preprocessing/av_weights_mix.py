"""
Adversarial validation weights with original dataset merging.

Two modes:
- align (default): Keep only competition columns; extras from original dropped,
  missing filled with NA. Weights returned only for original train (clipped).
- union: Take union of all columns; missing filled with NA; optionally add
  source flag column (0=synthetic, 1=original) to mark missing fields.

Integrates with MLArena preprocessing pipeline via fit_transform/transform interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from adversarial_validation import compute_adversarial_weights


AV_PROB_COL = "av_prob"


def resolve_path(p: str | Path, base: Path) -> Path:
    """Resolve path: absolute paths as-is, relative paths against base."""
    p = Path(p)
    return p if p.is_absolute() else (base / p)


def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict[str, Any]]:
    """
    Merge original dataset with synthetic train/test, compute AV weights.

    Args:
        train_df: Synthetic competition train data
        val_df: Optional validation split (unused, returned as-is)
        test_df: Synthetic competition test data
        config: Configuration dict with:
            - _artifact_dir: Path to experiment artifacts (injected by PreprocessModule)
            - _dataset: Dataset metadata (id_column, target, ignored_columns)
            - _project_root: Project root path (optional, defaults to cwd)
            - orig_path: Path to original dataset CSV (relative to project root)
            - mode: "align" (default) or "union"
            - source_flag: Optional column name for source flag (0=synth, 1=orig)
            - time_limit: AV model training time limit (seconds)
            - presets: AutoGluon preset for AV model
            - included_model_types: Optional list of model types (e.g., ["GBM", "XGB"])
            - drop_columns: Additional columns to drop before AV training
            - output_filename: Weights CSV filename (default: train_av_weights.csv)
            - model_dir: Directory for AV model (default: av_model_mix)
            - keep_extra_weights: If True, return weights for all rows (default: False)

    Returns:
        - train_df (unchanged)
        - val_df (unchanged)
        - test_df (unchanged)
        - state_dict: Metadata including weights_path, mode, row counts, stats
    """
    # Extract config
    artifact_dir = Path(config["_artifact_dir"])
    dataset_info = config.get("_dataset", {}) or {}

    # Infer project root from artifact_dir
    # Path structure: {project_root}/experiments/pre-{template}/artifacts/preprocess
    # So: preprocess -> artifacts -> pre-{template} -> experiments -> project_root
    project_root = artifact_dir.parent.parent.parent.parent

    id_column = dataset_info.get("id_column") or config.get("id_column", "id")
    target_column = dataset_info.get("target") or config.get("target_column")

    # User config
    orig_path = resolve_path(config["orig_path"], project_root)
    mode = config.get("mode", "align")
    source_flag = config.get("source_flag", None)
    time_limit = int(config.get("time_limit", 300))
    presets = config.get("presets", "medium_quality_faster_train")
    included_model_types = config.get("included_model_types", None)
    drop_columns_extra = config.get("drop_columns", []) or []
    output_filename = config.get("output_filename", "train_av_weights.csv")
    model_dir_name = config.get("model_dir", "av_model_mix")
    keep_extra_weights = config.get("keep_extra_weights", False)

    # Validate original dataset exists
    if not orig_path.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {orig_path}\n"
            f"Please ensure the file exists or update 'orig_path' in template config."
        )

    print(f"[av_weights_mix] Loading original dataset: {orig_path}")
    orig_df = pd.read_csv(orig_path)

    # Build extra_df with IDs if missing
    if id_column not in orig_df.columns:
        start_id = train_df[id_column].max() + 1 if id_column in train_df.columns else 0
        orig_df[id_column] = range(start_id, start_id + len(orig_df))
        print(f"[av_weights_mix] Added ID column to original dataset: {id_column}")

    # Mode handling: align or union
    if mode == "align":
        print(f"[av_weights_mix] Mode: align (keep competition columns only)")
        # Keep only competition columns, fill missing with NA
        keep_cols: List[str] = [
            c for c in train_df.columns
            if c != target_column or (target_column and target_column in orig_df.columns)
        ]
        extra_df = orig_df.reindex(columns=keep_cols, fill_value=pd.NA)
    elif mode == "union":
        print(f"[av_weights_mix] Mode: union (merge all columns)")
        # Union of all columns
        union_cols = sorted(set(train_df.columns) | set(orig_df.columns) | set(test_df.columns))
        train_df = train_df.reindex(columns=union_cols, fill_value=pd.NA)
        test_df = test_df.reindex(columns=union_cols, fill_value=pd.NA)
        extra_df = orig_df.reindex(columns=union_cols, fill_value=pd.NA)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'align' or 'union'.")

    # Add source flag if requested
    def add_source(df: pd.DataFrame, flag: int) -> pd.DataFrame:
        if source_flag:
            df = df.copy()
            df[source_flag] = flag
        return df

    train_len = len(train_df)
    train_concat = pd.concat(
        [add_source(train_df, 0), add_source(extra_df, 1)],
        ignore_index=True
    )
    test_df = add_source(test_df, 0)

    # Build drop columns list
    drop_cols = list(drop_columns_extra)
    ignored = dataset_info.get("ignored_columns", []) or []
    drop_cols.extend(ignored)
    if id_column not in drop_cols:
        drop_cols.append(id_column)

    # Setup output paths
    model_dir = artifact_dir / model_dir_name
    weights_output = artifact_dir / output_filename
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[av_weights_mix] Training AV model...")
    print(f"[av_weights_mix]   Train rows: {len(train_concat)} (synthetic: {train_len}, original: {len(extra_df)})")
    print(f"[av_weights_mix]   Test rows: {len(test_df)}")
    print(f"[av_weights_mix]   Preset: {presets}, Time limit: {time_limit}s")

    # Compute adversarial weights
    av_df = compute_adversarial_weights(
        train_df=train_concat,
        test_df=test_df,
        id_column=id_column,
        target_column=target_column,
        drop_columns=drop_cols,
        presets=presets,
        time_limit=time_limit,
        included_model_types=included_model_types,
        output_dir=model_dir,
    )

    # Transform weights: p/(1-p) -> clip to 2.0 -> normalize to mean 1.0
    p = av_df[AV_PROB_COL].clip(lower=1e-9, upper=1 - 1e-9)
    weights = p / (1 - p)
    weights = weights.clip(upper=2.0)
    weights = weights / weights.mean()
    av_df[AV_PROB_COL] = weights

    # Clip to original train length if requested
    if not keep_extra_weights:
        av_df = av_df.iloc[:train_len].reset_index(drop=True)
        print(f"[av_weights_mix] Clipped weights to original train length: {len(av_df)} rows")

    # Save weights
    av_df.to_csv(weights_output, index=False)
    print(f"[av_weights_mix] Saved weights to: {weights_output}")

    # Build state dict
    state: Dict[str, Any] = {
        "version": "1.0",
        "weights_path": str(weights_output.relative_to(project_root)),
        "mode": mode,
        "source_flag": source_flag,
        "orig_rows": len(orig_df),
        "train_rows": train_len,
        "total_av_rows": len(train_concat),
        "output_rows": len(av_df),
        "keep_extra_weights": keep_extra_weights,
        "av_stats": {
            "mean": float(av_df[AV_PROB_COL].mean()),
            "min": float(av_df[AV_PROB_COL].min()),
            "max": float(av_df[AV_PROB_COL].max()),
            "std": float(av_df[AV_PROB_COL].std()),
        },
        "config": {
            "presets": presets,
            "time_limit": time_limit,
            "included_model_types": included_model_types,
            "drop_columns": drop_cols,
        },
    }

    print(f"[av_weights_mix] AV stats: mean={state['av_stats']['mean']:.4f}, "
          f"min={state['av_stats']['min']:.4f}, max={state['av_stats']['max']:.4f}")

    # Return unchanged data + state
    return train_df.copy(), None if val_df is None else val_df.copy(), test_df.copy(), state


def transform(
    df: pd.DataFrame,
    state_dict: Dict[str, Any],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Inference-time preprocessing (no weights needed for new data).

    Returns:
        df unchanged (weights only apply to training data)
    """
    return df.copy()

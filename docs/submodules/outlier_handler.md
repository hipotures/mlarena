# Outlier Handler Sub-Module

## Overview

The **outlier_handler** sub-module detects and handles outliers in numeric features with configurable strategies. It can clip, set to NA, or just flag outliers. Supports quantile-, IQR-, z-score-based rules and IsolationForest.

**Module Name**: `outlier_handler`  
**Location**: `src/mlarena/defaults/preprocessing/outlier_handler.py`

## Capabilities
- Methods: `quantile`, `iqr`, `zscore`, `isolation_forest`, or `none`.
- Actions: `clip` (default), `set_na`, `flag_only` (adds `_outlier_flag` columns).
- Column selection via `include_cols` / `exclude_cols`; auto-excludes id/target/ignored.
- IsolationForest option for more flexible detection (uses sklearn).

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `outlier_method` | str | `"iqr"` | `none`, `quantile`, `iqr`, `zscore`, `isolation_forest` |
| `action` | str | `"clip"` | `clip`, `set_na`, `flag_only` |
| `lower_quantile` | float \| null | `0.01` | Lower bound (quantile method) |
| `upper_quantile` | float \| null | `0.99` | Upper bound (quantile method) |
| `iqr_factor` | float | `1.5` | IQR multiplier for bounds |
| `zscore_threshold` | float | `3.0` | |z| threshold for z-score method |
| `isoforest_contamination` | float | `0.05` | Contamination for IsolationForest |
| `include_cols` | List[str] \| null | `null` | Specific numeric columns to process (null = auto-detect) |
| `exclude_cols` | List[str] | `[]` | Additional columns to skip |
| `random_state` | int | `42` | Seed (IsolationForest) |

## Examples

### IQR Clipping (default)
```yaml
outlier_iqr_clip:
  module: outlier_handler
  cache: true
  config:
    outlier_method: "iqr"
    iqr_factor: 1.5
    action: "clip"
```

### Quantile Clipping
```yaml
outlier_quantile:
  module: outlier_handler
  cache: true
  config:
    outlier_method: "quantile"
    lower_quantile: 0.01
    upper_quantile: 0.99
    action: "clip"
```

### Z-Score Flag Only
```yaml
outlier_zscore_flag:
  module: outlier_handler
  cache: true
  config:
    outlier_method: "zscore"
    zscore_threshold: 3.5
    action: "flag_only"
```

### IsolationForest with NA Setting
```yaml
outlier_isoforest:
  module: outlier_handler
  cache: true
  config:
    outlier_method: "isolation_forest"
    isoforest_contamination: 0.03
    action: "set_na"
    random_state: 123
```

## Artifacts
- `outlier_report.json`: method/action, columns processed, bounds (where applicable), outlier counts, flag columns, sanitized config.
- `summary.json`: standard preprocessing summary (shape/column changes).

## State Dictionary (`fit_transform` return)
```python
{
    "version": "1.0",
    "method": "iqr",
    "action": "clip",
    "columns_processed": ["Age", "Fare", ...],
    "flag_columns": [],
    "stats": {
        "Fare": {
            "method": "iqr",
            "action": "clip",
            "bounds": {"lower": -10.5, "upper": 120.7},
            "train_outliers": 15,
            "val_outliers": 3,
            "test_outliers": 8
        }
    },
    "config": {...}
}
```

## Notes & Tips
- Only numeric columns are processed; run encoders first if you need encoded features handled.
- Quantile/IQR/Z-score use bounds computed from train and applied consistently to val/test.
- `flag_only` adds `{col}_outlier_flag` columns and leaves values untouched.
- IsolationForest uses median of inliers for `clip`; `set_na` will introduce NaNs—impute later if needed.
***

# Feature Selector Sub-Module

## Overview

The **feature_selector** sub-module systematically reduces feature dimensionality using multiple selection strategies (filter, embedded, wrapper). It keeps one implementation with behavior controlled by YAML, so you can quickly compare selection methods without changing code.

**Module Name**: `feature_selector`  
**Location**: `config/code/preprocessing/feature_selector.py`

## Capabilities
- Multiple selection modes: variance filtering, mutual information, correlation with target, model-based importances, L1 sparsity, and RFE
- Supports both classification and regression (`_dataset.problem_type`)
- Safety guard via `max_drop_fraction` to avoid over-pruning
- Produces detailed importance/score report for analysis

## Parameters

### Core
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `selection_method` | str | `"variance"` | Selection strategy: `variance`, `mi`, `correlation`, `model_importance`, `l1`, `rfe`, `none` |
| `k_features` | int \| null | `null` | Absolute number of features to keep (caps at available) |
| `keep_fraction` | float \| null | `0.8` | Fraction of features to keep (0-1). Ignored if `k_features` is set |
| `max_drop_fraction` | float | `0.5` | Max fraction of features that can be removed in one run |

### Thresholds / Filters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_variance` | float | `0.01` | Variance cutoff for `variance` method |
| `min_importance` | float | `0.001` | Importance cutoff for `model_importance` |

### Model-Based Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `importance_model_type` | str | `"lgbm"` | Base model for `model_importance`: `lgbm`, `xgb`, `rf` (falls back to RF if missing) |
| `n_estimators` | int | `100` | Number of trees/estimators for model-based methods and RFE |
| `max_depth` | int | `5` | Tree depth for model-based methods and RFE |
| `random_state` | int | `42` | Random seed for reproducibility |

## Selection Methods (What They Do)
- **`variance`**: Drop low-variance numeric features (`min_variance`); backfills to target `k_features`/`keep_fraction` if too many would be removed.
- **`mi`**: Mutual information vs. target (`mutual_info_classif`/`regression`); keeps top `k_features`/fraction.
- **`correlation`**: Absolute Pearson correlation with target (numeric only); keeps highest correlations.
- **`model_importance`**: Train tree model (LGBM/XGB/RF) and keep features above `min_importance` or top-K/fraction.
- **`l1`**: L1-regularized LogisticRegression (classification) or Lasso (regression); keeps features with non-zero coefficients (top-K/fraction if needed).
- **`rfe`**: Recursive Feature Elimination with tree estimator; selects target number of features.
- **`none`**: Skip selection (pass-through), still produces summary report.

## Examples

### Variance Threshold (default)
```yaml
feature_selection_variance:
  module: feature_selector
  cache: true
  config:
    selection_method: "variance"
    min_variance: 0.01
    keep_fraction: 0.8
```

### Mutual Information Top-K
```yaml
feature_selection_mi:
  module: feature_selector
  cache: true
  config:
    selection_method: "mi"
    k_features: 100
    random_state: 42
```

### Model Importances (LightGBM) with Drop Cap
```yaml
feature_selection_lgbm:
  module: feature_selector
  cache: true
  config:
    selection_method: "model_importance"
    importance_model_type: "lgbm"
    n_estimators: 300
    max_depth: 8
    min_importance: 0.0
    keep_fraction: 0.5
    max_drop_fraction: 0.4
```

### L1 Sparsity (Logistic Regression)
```yaml
feature_selection_l1:
  module: feature_selector
  cache: true
  config:
    selection_method: "l1"
    keep_fraction: 0.6
    random_state: 7
```

### RFE with Random Forest
```yaml
feature_selection_rfe:
  module: feature_selector
  cache: true
  config:
    selection_method: "rfe"
    k_features: 50
    importance_model_type: "rf"
    n_estimators: 200
    max_depth: 6
```

## Artifacts
- `feature_selection_report.json`: Detailed results (method, before/after counts, selected/dropped lists, per-feature scores).
- `summary.json`: Standard preprocessing summary (shape/columns changes, config snapshot).

## State Dictionary (returned by `fit_transform`)
```python
{
    "version": "1.0",
    "method": "mi",
    "config": {...},              # sanitized (no internal _ keys)
    "features_before": 500,
    "features_after": 200,
    "selected_features": [...],
    "feature_scores_summary": {
        "mean": 0.012,
        "std": 0.031,
        "min": 0.0,
        "max": 0.34,
    },
    "message": "..."  # present for skip/edge cases
}
```

## Notes & Tips
- Uses `_dataset.target` and `_dataset.problem_type` to choose correct scorers/models; ensure they are set.
- Only numeric columns are considered; categorical encoders should run before this module if you need encoded features selected.
- Guardrail: `max_drop_fraction` prevents over-pruning—keep it < 0.8 unless you are sure.
- For tree-only models downstream, `selection_method: "none"` is often fine; use selection to speed training or remove noise.
- If LightGBM/XGBoost are unavailable, the module falls back to RandomForest with a warning.

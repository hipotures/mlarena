# Feature Selector Sub-Module

## Overview

The **feature_selector** sub-module systematically reduces feature dimensionality using multiple selection strategies (filter, embedded, wrapper). It keeps one implementation with behavior controlled by YAML, so you can quickly compare selection methods without changing code.

**Module Name**: `feature_selector`  
**Location**: `src/mlarena/defaults/preprocessing/feature_selector.py`

## Capabilities
- Multiple selection modes: variance filtering, mutual information, correlation with target, model-based importances, permutation/null importances, L1 sparsity, and RFE
- Supports both classification and regression (`_dataset.problem_type`)
- Safety guard via `max_drop_fraction` to avoid over-pruning
- Produces detailed importance/score report for analysis

## Parameters

### Core
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `selection_method` | str | `"variance"` | Selection strategy: `variance`, `mi`, `correlation`, `model_importance`, `permutation_importance`, `null_importance`, `l1`, `rfe`, `none` |
| `n_features` | int \| float \| null | `null` | **Universal feature selector** (see modes below) |
| `max_drop_fraction` | float | `0.5` | Max fraction of features that can be removed in one run (safety constraint) |
| `protect_cb_features` | bool | `true` | Keep numeric columns ending with `_cb` regardless of selection |

#### `n_features` Modes
| Value | Behavior | Example |
|-------|----------|---------|
| `0` | **Pass-through** (no selection) | Keep all N features |
| `0 < n < 1` | **Keep fraction** | `0.8` = keep 80% best features |
| `n >= 1` | **Keep exact count** | `25` = keep exactly 25 best features |
| `n < 0` | **Drop N worst** | `-1` = drop 1 worst feature, `-3` = drop 3 worst |
| `null` | **Threshold-only** | Use `min_importance`/`min_variance`, no hard count limit |

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
| `perm_importance_repeats` | int | `5` | Repeats for permutation importance |
| `perm_importance_scoring` | str \| null | `null` | Scoring name for permutation importance |
| `perm_importance_max_samples` | int \| float \| null | `null` | Subsample rows for permutation importance |
| `null_importance_rounds` | int | `10` | Number of target shuffles for null importances |
| `null_importance_quantile` | float | `0.95` | Quantile cutoff for null importances |

## Selection Methods (What They Do)
- **`variance`**: Drop low-variance numeric features (`min_variance`); applies top-K cap based on `n_features`. **Bugfix**: Now correctly trims to exact count when threshold passes too many features.
- **`mi`**: Mutual information vs. target (`mutual_info_classif`/`regression`); keeps top-K based on `n_features`.
- **`correlation`**: Absolute Pearson correlation with target (numeric only); keeps highest correlations based on `n_features`.
- **`model_importance`**: Train tree model (LGBM/XGB/RF) and select features. **Bugfix**: Now enforces exact `n_features` count regardless of `min_importance` threshold (uses hybrid threshold + top-K cap).
- **`permutation_importance`**: Train model and rank features by permutation importance (supports optional subsampling).
- **`null_importance`**: Compare real importances to shuffled-target importances and keep features above a quantile threshold.
- **`l1`**: L1-regularized LogisticRegression (classification) or Lasso (regression); keeps top-K features by coefficient magnitude.
- **`rfe`**: Recursive Feature Elimination with tree estimator; selects exact number specified by `n_features`.
- **`none`**: Skip selection (pass-through), still produces summary report.

## Examples

### Variance Threshold with Fraction
```yaml
feature_selection_variance:
  module: feature_selector
  cache: true
  config:
    selection_method: "variance"
    min_variance: 0.01
    n_features: 0.8  # Keep 80% best features by variance
```

### Mutual Information - Keep Exact Count
```yaml
feature_selection_mi:
  module: feature_selector
  cache: true
  config:
    selection_method: "mi"
    n_features: 100  # Keep exactly 100 best features by MI
    random_state: 42
```

### Model Importances - Drop N Worst
```yaml
feature_selection_drop_worst:
  module: feature_selector
  cache: true
  config:
    selection_method: "model_importance"
    importance_model_type: "lgbm"
    n_estimators: 300
    max_depth: 8
    n_features: -3  # Drop 3 worst features
    max_drop_fraction: 0.5
    min_importance: 0.0
    protect_cb_features: true
```

### Model Importances - Keep Fraction
```yaml
feature_selection_lgbm:
  module: feature_selector
  cache: true
  config:
    selection_method: "model_importance"
    importance_model_type: "lgbm"
    n_estimators: 300
    max_depth: 8
    n_features: 0.5  # Keep 50% best features
    max_drop_fraction: 0.4
    min_importance: 0.0
```

### L1 Sparsity (Logistic Regression)
```yaml
feature_selection_l1:
  module: feature_selector
  cache: true
  config:
    selection_method: "l1"
    n_features: 0.6  # Keep 60% best features
    random_state: 7
```

### Permutation Importance
```yaml
feature_selection_perm:
  module: feature_selector
  cache: true
  config:
    selection_method: "permutation_importance"
    perm_importance_repeats: 5
    n_features: 0.6
```

### Null Importances
```yaml
feature_selection_null:
  module: feature_selector
  cache: true
  config:
    selection_method: "null_importance"
    null_importance_rounds: 10
    null_importance_quantile: 0.95
    n_features: 0.6
```

### RFE with Random Forest
```yaml
feature_selection_rfe:
  module: feature_selector
  cache: true
  config:
    selection_method: "rfe"
    n_features: 50  # Keep exactly 50 features
    importance_model_type: "rf"
    n_estimators: 200
    max_depth: 6
```

### Threshold-Only Mode (No Hard Limit)
```yaml
feature_selection_threshold:
  module: feature_selector
  cache: true
  config:
    selection_method: "model_importance"
    importance_model_type: "lgbm"
    n_features: null  # No hard limit, use min_importance only
    min_importance: 0.01  # Keep all features with importance >= 0.01
    n_estimators: 200
```

### Pass-Through (No Selection)
```yaml
feature_selection_none:
  module: feature_selector
  cache: true
  config:
    selection_method: "model_importance"
    n_features: 0  # Pass-through mode: keep all features
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
- **Protected columns**: By default, numeric columns ending with `_cb` are protected (set `protect_cb_features: false` to disable).
- **Unified API**: Single `n_features` parameter replaces old `k_features` and `keep_fraction` - no more parameter conflicts!
- **Bugfix applied**: `model_importance` and `variance` methods now enforce exact count even when `min_importance`/`min_variance` thresholds pass too many features.
- Guardrail: `max_drop_fraction` prevents over-pruning—keep it < 0.8 unless you are sure. Acts as safety constraint that overrides `n_features` if needed.
- For tree-only models downstream, `n_features: 0` (pass-through) or `selection_method: "none"` is often fine; use selection to speed training or remove noise.
- If LightGBM/XGBoost are unavailable, the module falls back to RandomForest with a warning.

## Migration from Old API
If you have old templates using `k_features` or `keep_fraction`:
- `k_features: 100` → `n_features: 100` (exact count)
- `keep_fraction: 0.8` → `n_features: 0.8` (80% best)
- Drop N worst (new): `n_features: -3` (drop 3 worst)
- Pass-through (new): `n_features: 0` (keep all)

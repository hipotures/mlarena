# Feature Group Aggregations Sub-Module

## Overview

The **feature_group_agg** sub-module creates group-based aggregations (e.g., mean price per category). It computes statistics on value columns grouped by one or more key columns from the training set and merges them back into all datasets.

**Module Name**: `feature_group_agg`  
**Location**: `src/mlarena/defaults/preprocessing/feature_group_agg.py`

## Capabilities
- **Group Statistics**: Supports `mean`, `std`, `min`, `max`, `count`, `nunique`, and optional quantiles.
- **Multi-key Grouping**: Can group by multiple columns simultaneously.
- **Consistent Merging**: Aggregates are computed on `train_df` only and merged into `val_df`, `test_df`, and `orig_df` to prevent leakage.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_keys` | List[str] | `[]` | Columns to group by |
| `group_value_cols` | List[str] | `[]` | Numeric columns to aggregate |
| `aggs` | List[str] | `[]` | Aggregations to compute (pandas compatible) |
| `quantiles` | List[float] | `[]` | Quantiles to compute per group (e.g., `[0.25, 0.5, 0.75]`) |
| `max_generated_features` | int | `200` | Hard cap on total new columns created |

## Examples

### Mean and Std per Category
```yaml
feature_group_category:
  module: feature_group_agg
  cache: true
  config:
    group_keys: ["category_id"]
    group_value_cols: ["price", "amount"]
    aggs: ["mean", "std"]
```

### Multi-key Grouping
```yaml
feature_group_multi:
  module: feature_group_agg
  cache: true
  config:
    group_keys: ["user_id", "day_of_week"]
    group_value_cols: ["session_duration"]
    aggs: ["mean", "max", "count"]
```

### Quantiles per Group
```yaml
feature_group_quantiles:
  module: feature_group_agg
  cache: true
  config:
    group_keys: ["segment"]
    group_value_cols: ["price"]
    aggs: ["mean"]
    quantiles: [0.25, 0.5, 0.75]
```

## Artifacts
- `feature_group_agg_report.json`: Lists grouped keys, value columns, and result features.
- `summary.json`: Standard preprocessing summary.

## Notes & Tips
- **Leakage Prevention**: Statistics are strictly calculated on the training set. If a key exists in test but not in train, it will receive `NaN` for aggregated columns.
- **Target Leakage**: **NEVER** include the target column in `group_value_cols` unless using specific out-of-fold logic (not supported by this basic module).
- **Naming**: Generated columns follow the pattern `key1__key2__value__agg` or `...__quantile_q0_5`.

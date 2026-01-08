# Row Aggregates Sub-Module

## Overview

Computes row-wise summary statistics for numeric columns (sum, mean, std, etc.).

**Module Name**: `row_aggregates`
**Location**: `src/mlarena/defaults/preprocessing/row_aggregates.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_cols` | list\|null | `null` | Numeric columns to include (null = all numeric). |
| `exclude_cols` | list | `[]` | Columns to exclude. |
| `stats` | list | `['mean','std','sum']` | Row-wise stats to compute. |
| `prefix` | str | `row_` | Prefix for generated feature names. |
| `nan_policy` | str | `omit` | `omit` = ignore NaNs, `fill_zero` = fill with 0 before computing. |

## Examples

### Basic Stats
```yaml
row_aggregates:
  module: row_aggregates
  config:
    stats: [sum, mean, std]
    prefix: row_
```

### Extended Stats
```yaml
row_aggregates:
  module: row_aggregates
  config:
    stats: [sum, mean, std, min, max, range, mad, skew, kurt]
    nan_policy: fill_zero
```

## Artifacts
- `row_aggregates_report.json`: List of new features and config used.
- `summary.json`: Standard preprocessing summary.

## Notes & Tips
- Use `nan_policy: fill_zero` if many missing values are expected in numeric inputs.
- `mad`, `skew`, and `kurt` can be more expensive on wide datasets.

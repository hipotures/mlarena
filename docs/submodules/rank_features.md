# Rank Features Sub-Module

## Overview

Transforms numeric columns into ranks or percentiles, optionally within groups.

**Module Name**: `rank_features`
**Location**: `src/mlarena/defaults/preprocessing/rank_features.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numeric_include` | list\|null | `null` | Numeric columns to include (null = all numeric). |
| `numeric_exclude` | list | `[]` | Columns to exclude. |
| `group_keys` | list | `[]` | Grouping keys for `by_group` mode. |
| `mode` | str | `global` | `global` or `by_group`. |
| `method` | str | `percentile` | `rank` or `percentile`. |
| `tie_method` | str | `average` | Rank tie handling: `average`, `min`, `max`, `first`, `dense`. |
| `add_original` | bool | `true` | Keep original numeric columns. |

## Examples

### Global Percentiles
```yaml
rank_features:
  module: rank_features
  config:
    mode: global
    method: percentile
    add_original: true
```

### Grouped Ranking
```yaml
rank_features:
  module: rank_features
  config:
    mode: by_group
    group_keys: [category]
    method: rank
```

## Artifacts
- `rank_features_report.json`: List of generated features.
- `summary.json`: Standard preprocessing summary.

## Notes & Tips
- Current implementation ranks each dataset independently (train/val/test).
- If you need strict train-fitted ranking, implement ECDF-based mapping.

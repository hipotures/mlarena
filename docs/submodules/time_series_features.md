# Time Series Features Sub-Module

## Overview

Generates lag and rolling-window features for time-ordered data, optionally grouped by entity.

**Module Name**: `time_series_features`
**Location**: `src/mlarena/defaults/preprocessing/time_series_features.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `entity_id_col` | str\|list\|null | `null` | Entity/grouping column(s). |
| `timestamp_col` | str\|null | `null` | Timestamp column used for sorting. |
| `sort_ascending` | bool | `true` | Sort order for time. |
| `value_cols` | list | `[]` | Columns to lag/roll. |
| `lags` | list | `[]` | Lag steps (e.g., `[1, 7, 28]`). |
| `windows` | list | `[]` | Rolling window sizes. |
| `rolling_aggs` | list | `['mean']` | Rolling aggregations: `mean`, `std`, `sum`, `min`, `max`. |
| `fill_method` | str | `none` | `none`, `ffill`, `bfill`, or `zero`. |
| `drop_original_value_cols` | bool | `false` | Drop original value columns after feature creation. |

## Examples

### Lags Only
```yaml
time_series_features:
  module: time_series_features
  config:
    entity_id_col: [user_id]
    timestamp_col: event_time
    value_cols: [target]
    lags: [1, 7, 28]
    fill_method: ffill
```

### Rolling Windows
```yaml
time_series_features:
  module: time_series_features
  config:
    entity_id_col: [user_id]
    timestamp_col: event_time
    value_cols: [target]
    windows: [7, 14]
    rolling_aggs: [mean, std]
    fill_method: zero
```

## Artifacts
- `summary.json`: Standard preprocessing summary.

## Notes & Tips
- This module concatenates train/val/test to build consistent lag features.
- Requires `value_cols` to be provided; missing columns will cause a skip/error.

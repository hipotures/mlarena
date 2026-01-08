# Missingness Features Sub-Module

## Overview

Creates features that capture missing-value patterns at both column and row level.

**Module Name**: `missingness_features`
**Location**: `src/mlarena/defaults/preprocessing/missingness_features.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_cols` | list\|null | `null` | Columns to check (null = all columns except system/excluded). |
| `exclude_cols` | list | `[]` | Columns to exclude from missingness checks. |
| `add_per_column_indicators` | bool | `true` | Add `{col}_na` indicator for columns with missing values in train. |
| `add_row_missing_count` | bool | `true` | Add `row_missing_count` feature. |
| `add_row_missing_ratio` | bool | `false` | Add `row_missing_ratio` feature. |
| `cap_row_missing_count` | int\|null | `null` | Cap the row count feature (outlier protection). |

## Examples

### Column Indicators Only
```yaml
missingness_features:
  module: missingness_features
  config:
    add_per_column_indicators: true
    add_row_missing_count: false
    add_row_missing_ratio: false
```

### Row Statistics Only
```yaml
missingness_features:
  module: missingness_features
  config:
    add_per_column_indicators: false
    add_row_missing_count: true
    add_row_missing_ratio: true
    cap_row_missing_count: 100
```

## Artifacts
- `missingness_report.json`: Summary of new features and missingness stats.
- `summary.json`: Standard preprocessing summary.

## Notes & Tips
- Indicators are created only for columns that have missing values in train.
- Row-level features are useful for models that benefit from missingness patterns.

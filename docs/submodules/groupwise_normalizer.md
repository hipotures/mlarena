# Groupwise Normalizer Sub-Module

## Overview

Normalizes numeric values relative to group-level statistics (mean/std), creating centered, z-score, and ratio features.

**Module Name**: `groupwise_normalizer`
**Location**: `src/mlarena/defaults/preprocessing/groupwise_normalizer.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_keys` | list | `[]` | Grouping columns (e.g., category). |
| `value_cols` | list | `[]` | Numeric columns to normalize. |
| `add_group_mean` | bool | `true` | Add group mean as feature. |
| `add_centered` | bool | `true` | Add value - group mean. |
| `add_zscore` | bool | `true` | Add z-score within group. |
| `add_ratio` | bool | `false` | Add value / group mean. |
| `eps` | float | `1.0e-6` | Epsilon for division stability. |

## Examples

### Centered + Z-Score
```yaml
groupwise_normalizer:
  module: groupwise_normalizer
  config:
    group_keys: [category]
    value_cols: [price]
    add_centered: true
    add_zscore: true
```

### Ratio Only
```yaml
groupwise_normalizer:
  module: groupwise_normalizer
  config:
    group_keys: [category]
    value_cols: [price]
    add_group_mean: false
    add_ratio: true
```

## Artifacts
- `summary.json`: Standard preprocessing summary.

## Notes & Tips
- Requires `group_keys` and `value_cols` to be provided; otherwise the module is skipped.
- If a group is unseen in inference data, global stats are used as fallback.

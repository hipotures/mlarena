# Categorical Cross Sub-Module

## Overview

Creates cross-product features for pairs of categorical columns (e.g., `A__B`).

**Module Name**: `categorical_cross`
**Location**: `src/mlarena/defaults/preprocessing/categorical_cross.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cross_pairs` | list | `[]` | Explicit column pairs. Empty = auto-generate pairs. |
| `max_pair_cardinality` | int | `5000` | Cardinality limit for auto-generated pairs. |
| `separator` | str | `"__"` | Separator between category values. |
| `output` | str | `hashed` | `hashed`, `onehot`, or `target_mean_oof`. |
| `hash_dim` | int | `12` | Hash space size for `hashed` output. |
| `oof_folds` | int | `5` | Folds for `target_mean_oof`. |
| `oof_random_state` | int | `42` | Random seed for OOF. |

## Examples

### Hashed Crosses
```yaml
categorical_cross:
  module: categorical_cross
  config:
    output: hashed
    hash_dim: 12
```

### One-Hot Crosses (Small Cardinality)
```yaml
categorical_cross:
  module: categorical_cross
  config:
    output: onehot
    max_pair_cardinality: 2000
```

## Artifacts
- `cross_onehot.pkl` (onehot only)
- `summary.json`: Standard preprocessing summary.

## Notes & Tips
- For `target_mean_oof`, target column must be present in train.
- Auto-generated pairs are limited by `max_pair_cardinality` to avoid explosion.

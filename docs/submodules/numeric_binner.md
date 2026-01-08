# Numeric Binner Sub-Module

## Overview

Discretizes continuous numeric features into bins using uniform, quantile, or k-means strategies.

**Module Name**: `numeric_binner`
**Location**: `src/mlarena/defaults/preprocessing/numeric_binner.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numeric_include` | list\|null | `null` | Numeric columns to include (null = all numeric). |
| `numeric_exclude` | list | `[]` | Columns to exclude. |
| `strategy` | str | `quantile` | `uniform`, `quantile`, or `kmeans`. |
| `n_bins` | int | `5` | Number of bins per feature. |
| `encode` | str | `ordinal` | `ordinal` or `onehot`. |
| `drop_original` | bool | `false` | Drop original numeric columns after binning. |

## Examples

### Quantile Ordinal Binning
```yaml
numeric_binner:
  module: numeric_binner
  config:
    strategy: quantile
    n_bins: 10
    encode: ordinal
```

### Uniform One-Hot Binning
```yaml
numeric_binner:
  module: numeric_binner
  config:
    strategy: uniform
    n_bins: 5
    encode: onehot
```

## Artifacts
- `discretizer.pkl`: Fitted `KBinsDiscretizer` object.
- `summary.json`: Standard preprocessing summary.

## Notes & Tips
- Requires imputation upstream when NaNs are present.
- `encode: onehot` can significantly expand feature count.

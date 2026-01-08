# Dimensionality Reducer Sub-Module

## Overview

Adds low-dimensional components using PCA or TruncatedSVD.

**Module Name**: `dimensionality_reducer`
**Location**: `src/mlarena/defaults/preprocessing/dimensionality_reducer.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | `pca` | `pca` or `svd`. |
| `n_components` | int | `10` | Number of components to generate. |
| `include_sparse` | bool | `false` | Reserved for future sparse handling. |
| `whiten` | bool | `false` | PCA whitening. |
| `random_state` | int | `42` | Random seed. |

## Examples

### PCA Components
```yaml
dimensionality_reducer:
  module: dimensionality_reducer
  config:
    method: pca
    n_components: 20
    whiten: false
```

### Truncated SVD
```yaml
dimensionality_reducer:
  module: dimensionality_reducer
  config:
    method: svd
    n_components: 50
```

## Artifacts
- `reducer.pkl`: Fitted PCA/SVD model.
- `reducer_report.json`: Explained variance details.
- `summary.json`: Standard preprocessing summary.

## Notes & Tips
- Requires numeric inputs without NaNs (use imputer/scaler beforehand).
- SVD is useful for sparse one-hot features.

# Clustering Features Sub-Module

## Overview

Generates clustering-based features using KMeans (cluster IDs and/or distances).

**Module Name**: `clustering_features`
**Location**: `src/mlarena/defaults/preprocessing/clustering_features.py`

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numeric_include` | list\|null | `null` | Numeric columns to include (null = all numeric). |
| `numeric_exclude` | list | `[]` | Columns to exclude. |
| `n_clusters` | int | `10` | Number of clusters. |
| `add_cluster_id` | bool | `true` | Add cluster ID column. |
| `add_distances` | bool | `false` | Add distances to each cluster center. |
| `random_state` | int | `42` | Random seed. |
| `n_init` | int | `10` | KMeans initializations. |
| `algorithm` | str | `kmeans` | Reserved for future methods. |

## Examples

### Cluster ID Only
```yaml
clustering_features:
  module: clustering_features
  config:
    n_clusters: 20
    add_cluster_id: true
    add_distances: false
```

### Cluster Distances
```yaml
clustering_features:
  module: clustering_features
  config:
    n_clusters: 10
    add_cluster_id: true
    add_distances: true
```

## Artifacts
- `kmeans.pkl`: Fitted KMeans model.
- `summary.json`: Standard preprocessing summary.

## Notes & Tips
- Requires numeric inputs without NaNs (impute first).
- Distance features can add many columns (`n_clusters`).

# knn_graph_features

## Overview
Creates KNN distance-based features for each row. Distances are computed in numeric
feature space using `sklearn.neighbors.NearestNeighbors`.

## Parameters
- `include_cols` (list[str] | null): Explicit numeric columns to use.
- `exclude_cols` (list[str]): Columns to exclude.
- `use_original_features_only` (bool): Restrict to original features.
- `k` (int): Number of neighbors.
- `metric` (str): Distance metric (e.g., `euclidean`, `manhattan`).
- `fit_on` (str): `train`, `train_val`, `train_test`, `train_val_test`.
- `scale` (bool): Standardize numeric columns before KNN.
- `missing_strategy` (str): `mean`, `median`, `zero`.
- `include_self` (bool): Keep self-distance in neighbor list.
- `add_density` (bool): Add `1/(mean_dist + eps)` feature.
- `prefix` (str): Prefix for generated columns.
- `random_state` (int): RNG seed (for consistent preprocessing decisions).

## Example
```yaml
module: knn_graph_features
config:
  k: 5
  metric: euclidean
  fit_on: train
  scale: true
```

## Artifacts
- `knn.pkl`: Fitted NearestNeighbors model.
- `scaler.pkl`: Optional StandardScaler.
- `summary.json`: Transformation summary.

## Notes
- Requires numeric input; missing values are imputed before KNN.
- Use after encoding/scaling for stable distances.
- Large datasets may need smaller `k` or reduced feature sets.

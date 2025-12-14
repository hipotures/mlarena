# Imbalance Handler Sub-Module

## Overview

The **imbalance_handler** sub-module addresses class imbalance for classification tasks. It can compute class weights (for models that consume sample weights) or resample the training data via random over/under sampling. SMOTE/ADASYN/SMOTENC are supported when `imbalanced-learn` is installed; otherwise the module raises an informative error.

**Module Name**: `imbalance_handler`  
**Location**: `src/mlarena/defaults/preprocessing/imbalance_handler.py`

## Capabilities
- `class_weight`: Adds a `sample_weight` column using balanced weights (total / (n_classes * count)).
- `random_over`: Upsamples minority classes to match the majority count (train only).
- `random_under`: Downsamples majority classes to match the minority count (train only).
- `none`: Pass-through.
- `smote` / `smotenc` / `adasyn`: Not active without `imbalanced-learn`; the module raises an informative ImportError if chosen.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `imbalance_method` | str | `"none"` | Strategy: `none`, `class_weight`, `random_over`, `random_under`, `smote`, `smotenc`, `adasyn` |
| `sampling_strategy` | str | `"auto"` | Currently only `auto` is supported (full balance to majority/minority) |
| `use_sample_weights` | bool | `true` | When `class_weight`, add `sample_weight` column (also to val if present) |
| `categorical_features` | List[str] | `[]` | Required for `smotenc` (column names treated as categorical) |
| `random_state` | int | `42` | Seed for resampling |

## Examples

### Class Weights (recommended baseline)
```yaml
imbalance_class_weight:
  module: imbalance_handler
  cache: true
  config:
    imbalance_method: "class_weight"
    use_sample_weights: true
```

### Random Over-Sampling
```yaml
imbalance_over:
  module: imbalance_handler
  cache: true
  config:
    imbalance_method: "random_over"
    random_state: 123
```

### Random Under-Sampling
```yaml
imbalance_under:
  module: imbalance_handler
  cache: true
  config:
    imbalance_method: "random_under"
    random_state: 123
```

### SMOTE / SMOTENC / ADASYN (requires imbalanced-learn)
```yaml
imbalance_smote:
  module: imbalance_handler
  cache: true
  config:
    imbalance_method: "smote"
    random_state: 123

imbalance_smotenc:
  module: imbalance_handler
  cache: true
  config:
    imbalance_method: "smotenc"
    categorical_features: ["cat_col1", "cat_col2"]
    random_state: 123
```
If `imbalanced-learn` is not installed, the module raises an ImportError with guidance.

## Artifacts
- `imbalance_report.json`: method, class counts before/after, sampling strategy, class weights (if used), sample weight column name.
- `summary.json`: standard preprocessing summary (shapes/columns, config snapshot).

## State Dictionary (`fit_transform` return)
```python
{
    "version": "1.0",
    "method": "class_weight",
    "class_counts_before": {"0": 549, "1": 342},
    "class_counts_after": {"0": 549, "1": 342},  # weights-only keeps counts
    "class_weights": {"0": 0.810", "1": 1.300},
    "sample_weight_column": "sample_weight",
    "config": {...}
}
```

## Notes & Tips
- Intended for classification only; skips if `problem_type` is not `binary`/`multiclass`.
- Target must be present and non-null.
- Resampling alters only the training set; validation/test remain untouched.
- For SMOTE/ADASYN/SMOTENC, install `imbalanced-learn` or stick to class weights / random sampling.***

# Drift Detector Sub-Module

## Overview

Detects features with significantly different distributions between train and test sets. This is crucial in Kaggle competitions where train-test drift can severely impact model performance.

## Purpose

- **Problem**: Features may have different distributions in train vs test, indicating data shift or leakage
- **Solution**: Calculate drift metrics (PSI, KS, Chi2, Model AUC) and optionally remove high-drift features
- **Use case**: Improve model stability by removing unstable features before training

## Parameters

### Required Parameters

None - all parameters have sensible defaults.

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `drift_metric` | str | `"psi"` | Method to detect drift: `psi`, `ks`, `chi2`, `model_auc` |
| `max_psi` | float | `0.25` | Maximum allowed PSI (Population Stability Index). Typical thresholds: <0.1 (no drift), 0.1-0.25 (moderate), >0.25 (high drift) |
| `max_ks` | float | `0.1` | Maximum allowed KS statistic for `ks` metric (numeric features only) |
| `max_pvalue` | float | `0.01` | Maximum p-value for statistical tests (`ks` and `chi2` metrics). Lower = more strict |
| `min_auc` | float | `0.6` | Minimum AUC for train-vs-test discrimination model. Higher = more drift detected |
| `action` | str | `"flag_only"` | What to do with drifting features: `none`, `drop`, `flag_only` |
| `max_drop_fraction` | float | `0.2` | Maximum fraction of features to drop (safety limit) |
| `exclude_cols` | list | `[]` | Additional columns to exclude from drift detection |
| `random_state` | int | `42` | Random state for reproducibility (used in `model_auc` metric) |

## Drift Metrics Explained

### 1. PSI (Population Stability Index)

**Best for**: All feature types (numeric and categorical)

**How it works**:
- Bins the feature into buckets
- Compares distribution of train vs test in each bucket
- PSI = Σ(test% - train%) × ln(test% / train%)

**Interpretation**:
- PSI < 0.1: No drift
- 0.1 ≤ PSI < 0.25: Moderate drift (monitor)
- PSI ≥ 0.25: High drift (consider removing)

**Recommended use**: Default choice, works universally

### 2. KS (Kolmogorov-Smirnov)

**Best for**: Numeric features only

**How it works**:
- Statistical test comparing cumulative distributions
- Returns KS statistic and p-value

**Interpretation**:
- KS > `max_ks` OR p-value < `max_pvalue` → drift detected

**Recommended use**: When you want statistical rigor for numeric features

### 3. Chi-Square

**Best for**: Categorical features only

**How it works**:
- Chi-square test of independence between feature values and train/test label
- Returns χ² statistic and p-value

**Interpretation**:
- p-value < `max_pvalue` → drift detected

**Recommended use**: Categorical features with low cardinality

### 4. Model AUC

**Best for**: Universal, but computationally expensive

**How it works**:
- Trains a simple Random Forest to predict "is this row from train or test?"
- Higher AUC = easier to distinguish train from test = more drift

**Interpretation**:
- AUC ≈ 0.5: No drift (random guess)
- AUC > `min_auc`: Drift detected
- AUC ≈ 1.0: Perfect drift (very bad)

**Recommended use**: When PSI/KS/Chi2 are inconclusive, or for complex multimodal distributions

## Actions

### `action: "none"`
- Only compute drift metrics, don't modify data
- Use when you want to analyze drift without removing features

### `action: "flag_only"` (default)
- Compute metrics and flag drifting features in the report
- Don't modify the data
- **Recommended for exploration**: See what drifts before deciding to drop

### `action: "drop"`
- Remove features with detected drift
- Subject to `max_drop_fraction` safety limit
- **Use with caution**: May remove informative features

## Examples

### Example 1: Basic Drift Detection (PSI)

```yaml
drift_detector:
  module: drift_detector
  cache: true
  config:
    drift_metric: "psi"
    max_psi: 0.25
    action: "flag_only"  # Just report, don't drop
```

**Use case**: Exploratory analysis - check which features drift

### Example 2: Aggressive Drift Removal

```yaml
drift_detector_aggressive:
  module: drift_detector
  cache: true
  config:
    drift_metric: "psi"
    max_psi: 0.15  # Stricter threshold
    action: "drop"
    max_drop_fraction: 0.3  # Allow dropping up to 30% of features
```

**Use case**: Competition with known severe train-test drift

### Example 3: Statistical Tests for Numeric Features

```yaml
drift_detector_ks:
  module: drift_detector
  cache: true
  config:
    drift_metric: "ks"
    max_ks: 0.1
    max_pvalue: 0.01
    action: "drop"
```

**Use case**: Numeric features where you want statistical confidence

### Example 4: Model-Based Drift Detection

```yaml
drift_detector_model:
  module: drift_detector
  cache: true
  config:
    drift_metric: "model_auc"
    min_auc: 0.65  # AUC > 0.65 means drift
    action: "drop"
    max_drop_fraction: 0.2
    random_state: 42
```

**Use case**: Complex features where PSI might miss subtle patterns

### Example 5: Conservative Drift Flagging

```yaml
drift_detector_conservative:
  module: drift_detector
  cache: true
  config:
    drift_metric: "psi"
    max_psi: 0.4  # Very lenient
    action: "flag_only"
    exclude_cols: ["important_feature_1", "important_feature_2"]
```

**Use case**: You don't want to risk dropping important features

## Artifacts Generated

### 1. `drift_report.json`

Detailed drift analysis:

```json
{
  "drift_metric": "psi",
  "columns_analyzed": 50,
  "columns_with_drift": 5,
  "columns_dropped": ["feature_1", "feature_3"],
  "columns_flagged": ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
  "drift_details": {
    "feature_1": {
      "column": "feature_1",
      "dtype": "float64",
      "psi": 0.456,
      "drifted": true
    },
    "feature_2": {
      "column": "feature_2",
      "dtype": "int64",
      "psi": 0.023,
      "drifted": false
    }
  },
  "config": {...}
}
```

### 2. `summary.json`

Standard preprocessing report with shape changes.

## Workflow Integration

### Typical Pipeline Position

**Recommended**: After feature engineering, before feature selection

```yaml
pipeline:
  chain: [
    sanity_check,
    imputer,
    encoder,
    feature_interactions, # Create new features
    feature_polynomial,
    drift_detector,      # Remove unstable features ← HERE
    feature_selector,    # Select best remaining features
    scaler
  ]
```

**Rationale**:
- Run after feature engineering so new features are also checked
- Run before feature selection to avoid wasting computation on drifted features

### Alternative: Early Drift Detection

```yaml
early_drift_removal:
  chain: [
    sanity_check,
    drift_detector,      # Remove drifted raw features early ← HERE
    imputer,
    encoder,
    feature_selector
  ]
```

**Rationale**: Remove problematic features early to speed up pipeline

## Tuning Guide

### 1. Choose the Right Metric

| Scenario | Recommended Metric | Reason |
|----------|-------------------|--------|
| Mixed feature types | `psi` | Works for both numeric and categorical |
| Only numeric features | `ks` | Statistical rigor, interpretable p-values |
| Only categorical | `chi2` | Proper statistical test for categories |
| Complex distributions | `model_auc` | Catches subtle patterns |
| Quick check | `psi` | Fast, no model training |

### 2. Set Appropriate Thresholds

**Conservative (keep more features)**:
```yaml
max_psi: 0.4
max_ks: 0.15
min_auc: 0.7
```

**Balanced (default)**:
```yaml
max_psi: 0.25
max_ks: 0.1
min_auc: 0.6
```

**Aggressive (remove more features)**:
```yaml
max_psi: 0.15
max_ks: 0.05
min_auc: 0.55
```

### 3. Start with `action: "flag_only"`

Always explore first:
1. Run with `flag_only` to see what drifts
2. Analyze `drift_report.json`
3. If many features drift, consider:
   - Adjusting thresholds
   - Using `exclude_cols` for important features
   - Switching to `drop` action

### 4. Protect Important Features

```yaml
drift_detector:
  config:
    exclude_cols:
      - "critical_feature_from_eda"
      - "known_strong_predictor"
```

## Edge Cases & Notes

### 1. Empty DataFrame After Drift Removal

If `action: "drop"` removes all features:
- **Cause**: Thresholds too strict or severe dataset shift
- **Solution**: Relax thresholds or use `max_drop_fraction` limit
- **Prevention**: Always use `max_drop_fraction < 1.0`

### 2. Different Column Sets in Train/Test

- Only columns present in **both** train and test are analyzed
- Columns unique to train or test are automatically excluded

### 3. High Missing Values

- Drift metrics handle missing values by dropping NaNs before calculation
- If a column is all NaN, drift = NaN (not flagged)

### 4. Categorical with High Cardinality

- PSI may be unstable with many categories
- Consider running `rare_category_handler` before `drift_detector`

### 5. Small Test Set

- Statistical tests (KS, Chi2) may have low power with small test sets
- PSI and model AUC are more robust to small sample sizes

## Performance Considerations

### Speed

| Metric | Speed | Notes |
|--------|-------|-------|
| `psi` | Fast | Simple distribution comparison |
| `ks` | Fast | Efficient statistical test |
| `chi2` | Fast | Simple contingency table |
| `model_auc` | **Slow** | Trains RF model per feature |

**Tip**: For large datasets with many features, use `psi` for speed.

### Memory

- All metrics: Low memory footprint
- `model_auc`: Subsamples to 10,000 rows if data is larger

## Real-World Example

### Scenario: Time-Series Competition

```yaml
# Competition where train is from 2020, test from 2021
# Many features likely drifted due to COVID-19

drift_detector_covid:
  module: drift_detector
  cache: true
  config:
    drift_metric: "psi"
    max_psi: 0.2  # Stricter due to known drift
    action: "drop"
    max_drop_fraction: 0.4  # Allow removing many features
    exclude_cols:
      - "user_id"  # Keep user features
      - "product_category"  # Domain knowledge: stable
```

**Expected outcome**: Remove 20-40% of features showing temporal drift, keeping model focused on stable patterns.

## Common Issues

### Issue 1: "No columns to check for drift"

**Cause**: All columns excluded (ID, target, ignored)

**Solution**: Check `exclude_cols` and `_dataset.ignored_columns`

### Issue 2: Drift report shows all NaN

**Cause**: Columns have incompatible types or all missing values

**Solution**: Run `sanity_check` before `drift_detector`

### Issue 3: Too many features dropped

**Cause**: `max_drop_fraction` too high or thresholds too strict

**Solution**:
- Reduce `max_drop_fraction`
- Relax thresholds (increase `max_psi`, decrease `min_auc`)

## References

- **PSI**: [Population Stability Index](https://www.listendata.com/2015/05/population-stability-index.html)
- **KS Test**: [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
- **Drift Detection**: [Concept Drift in Machine Learning](https://machinelearningmastery.com/gentle-introduction-concept-drift-machine-learning/)

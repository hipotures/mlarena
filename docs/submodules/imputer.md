# Imputer Sub-Module

## Overview

Universal missing value imputation with configurable strategies per column type and per individual column. Supports multiple imputation methods from simple (mean/median) to advanced (KNN, iterative). Optional outlier treatment before imputation.

**Module Name**: `imputer`

**Location**: `src/mlarena/defaults/preprocessing/imputer.py`

## Features

- **Multiple strategies**: mean, median, most_frequent, constant, KNN, iterative
- **Type-aware**: Different default strategies for numeric vs categorical columns
- **Column-specific overrides**: Set custom strategy for individual columns
- **Outlier treatment**: Optional conversion of outliers to NA before imputation
- **Comprehensive reporting**: Track missing values before/after, imputation statistics

## Parameters

### Global Strategy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `numeric_strategy` | str | `"mean"` | Default strategy for numeric columns. Options: `mean`, `median`, `most_frequent`, `constant`, `knn`, `iterative` |
| `categorical_strategy` | str | `"most_frequent"` | Default strategy for categorical columns. Options: `most_frequent`, `constant` |

### Column-Specific Overrides

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `column_strategies` | dict | `{}` | Mapping of column names to specific strategies. Example: `{"age": "median", "income": "knn"}` |

### Strategy-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fill_value` | any | `0` (numeric), `"__MISSING__"` (categorical) | Value used for `constant` strategy |
| `knn_n_neighbors` | int | `5` | Number of neighbors for KNN imputation |
| `iterative_estimator` | str | `"bayesian_ridge"` | Estimator for IterativeImputer (currently only supports BayesianRidge) |
| `iterative_max_iter` | int | `10` | Maximum iterations for IterativeImputer |

### Outlier Treatment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `treat_outliers_as_na` | bool | `false` | Convert outliers to NA before imputation |
| `outlier_method` | str | `"iqr"` | Outlier detection method. Options: `iqr`, `zscore` |
| `outlier_threshold` | float | `1.5` (IQR), `3.0` (zscore) | Threshold for outlier detection |

## Imputation Strategies

### Numeric Strategies

1. **`mean`**: Replace missing values with column mean
   - Fast, simple
   - Affected by outliers
   - Good for normally distributed data

2. **`median`**: Replace missing values with column median
   - Robust to outliers
   - Good default choice
   - Preserves distribution shape

3. **`most_frequent`**: Replace with most common value
   - Works for discrete numeric columns
   - Can create mode bias

4. **`constant`**: Replace with specified value
   - Full control over fill value
   - Use for domain-specific defaults

5. **`knn`**: K-Nearest Neighbors imputation
   - Uses similar rows to impute
   - Captures relationships between features
   - Slower, but more accurate
   - Good for structured missing patterns

6. **`iterative`**: Iterative imputation (MICE)
   - Models each feature as function of others
   - Most sophisticated method
   - Slowest, but handles complex patterns
   - Good for multivariate missing data

### Categorical Strategies

1. **`most_frequent`**: Replace with mode
   - Standard approach
   - Preserves distribution

2. **`constant`**: Replace with custom value
   - Create explicit "Missing" category
   - Good for models that can use this signal

## Examples

### Example 1: Basic Usage (Default Strategies)

```yaml
imputer:
  module: imputer
  cache: true
  config:
    numeric_strategy: "median"
    categorical_strategy: "most_frequent"
```

**Use case**: Quick, robust imputation for most datasets

### Example 2: Column-Specific Strategies

```yaml
imputer:
  module: imputer
  cache: true
  config:
    numeric_strategy: "median"
    categorical_strategy: "most_frequent"
    column_strategies:
      age: "median"
      income: "knn"
      education: "constant"
      city: "most_frequent"
    fill_value: "__UNKNOWN__"
    knn_n_neighbors: 7
```

**Use case**: Fine-tuned imputation where different columns need different treatment

### Example 3: Advanced - KNN Imputation

```yaml
imputer:
  module: imputer
  cache: true
  config:
    numeric_strategy: "knn"
    categorical_strategy: "most_frequent"
    knn_n_neighbors: 5
```

**Use case**: When features are correlated and you want to leverage relationships

### Example 4: Iterative Imputation (MICE)

```yaml
imputer:
  module: imputer
  cache: true
  config:
    numeric_strategy: "iterative"
    categorical_strategy: "most_frequent"
    iterative_max_iter: 10
    iterative_estimator: "bayesian_ridge"
```

**Use case**: Complex missing patterns with feature dependencies

### Example 5: Outlier Treatment Before Imputation

```yaml
imputer:
  module: imputer
  cache: true
  config:
    numeric_strategy: "median"
    categorical_strategy: "most_frequent"
    treat_outliers_as_na: true
    outlier_method: "iqr"
    outlier_threshold: 1.5
```

**Use case**: When outliers should be treated as missing values (e.g., data quality issues)

### Example 6: Constant Fill Values

```yaml
imputer:
  module: imputer
  cache: true
  config:
    numeric_strategy: "constant"
    categorical_strategy: "constant"
    fill_value: 0  # For numeric
    column_strategies:
      category_col: "constant"  # Uses "__MISSING__" for categorical
```

**Use case**: Domain-specific defaults (e.g., 0 for counts, specific category for missing)

## Artifacts

### Saved Files

1. **`imputer_{column}.pkl`**: Fitted imputer for each column
   - Can be loaded for inference-time imputation
   - Preserves exact transformation

2. **`imputation_report.json`**: Detailed imputation statistics
   ```json
   {
     "missing_before": {"age": 150, "income": 50},
     "missing_after": {"age": 0, "income": 0},
     "column_strategies": {"age": "median", "income": "knn"},
     "outlier_treatment": {
       "age": {
         "method": "iqr",
         "threshold": 1.5,
         "train_outliers": 23,
         "test_outliers": 12
       }
     },
     "imputed_columns": {
       "numeric": ["age", "income"],
       "categorical": ["city", "education"]
     }
   }
   ```

3. **`summary.json`**: Standard preprocessing report
   - Shape changes
   - Column changes
   - Metadata

## Performance Considerations

### Speed

| Strategy | Speed | Memory |
|----------|-------|--------|
| mean/median/most_frequent | ⚡⚡⚡ Fast | Low |
| constant | ⚡⚡⚡ Fast | Low |
| knn | ⚡ Slow | Medium-High |
| iterative | ⚡ Slowest | Medium |

### Recommendations

- **Small datasets (<10K rows)**: Any strategy works
- **Medium datasets (10K-100K)**: Prefer mean/median/constant for speed
- **Large datasets (>100K)**: Avoid KNN/iterative unless necessary
- **High-dimensional**: Consider simple strategies first

## Edge Cases & Handling

### All Values Missing

If a column has all missing values:
- Simple strategies (mean/median): Imputation fails gracefully
- KNN/Iterative: May fail, consider dropping column in sanity_check first

### No Missing Values

If a column has no missing values:
- Imputer still fits but transform is no-op
- No artifacts saved for that column

### New Categories in Test

For categorical imputation:
- SimpleImputer handles unknown values gracefully
- Falls back to fill_value for constant strategy

### Outliers as Missing

When `treat_outliers_as_na: true`:
1. Detect outliers on **train only** (IQR or Z-score)
2. Apply same bounds to test/val
3. Convert outliers to NaN
4. Then impute as normal missing values

This prevents outliers from biasing mean/median calculations.

## Integration with Other Sub-Modules

### Before Imputer

**Recommended**:
- `sanity_check`: Clean obvious issues first
  ```yaml
  chain: [sanity_check, imputer]
  ```

### After Imputer

**Common patterns**:
- Encoding: No missing values makes encoding easier
  ```yaml
  chain: [imputer, encoder]
  ```

- Scaling: Imputation before scaling prevents NaN propagation
  ```yaml
  chain: [imputer, scaler]
  ```

- Feature Selection: Complete data needed for variance/correlation
  ```yaml
  chain: [imputer, feature_selector]
  ```

## Troubleshooting

### High Memory Usage with KNN

**Problem**: KNN requires computing distances for all rows

**Solution**:
- Reduce `knn_n_neighbors`
- Use sampling for large datasets
- Switch to iterative or simple strategy

### IterativeImputer Not Converging

**Problem**: Max iterations reached without convergence

**Solution**:
- Increase `iterative_max_iter` (e.g., 20)
- Check for multicollinearity
- Consider simpler strategy for problematic columns

### Categorical Columns Treated as Numeric

**Problem**: Numeric IDs incorrectly identified as numeric

**Solution**:
- Use `sanity_check` first to fix column types
- Or add to `column_strategies` with categorical strategy

## Tips & Best Practices

1. **Start simple**: Try mean/median first, then move to KNN/iterative if needed
2. **Check missingness pattern**: Use EDA to understand why values are missing
3. **Domain knowledge**: Use column_strategies for domain-specific imputation
4. **Outlier treatment**: Only use if outliers are likely data errors
5. **Test impact**: Compare model performance with/without imputation
6. **Monitor artifacts**: Check imputation_report.json for imputation quality

## Example Chain: Complete Pipeline

```yaml
# Basic cleaning + robust imputation + encoding
data_preparation:
  chain: [sanity_check, imputer, encoder]

# With configs
sanity_check:
  module: sanity_check
  config:
    drop_duplicates: true
    max_missing_fraction: 0.95

imputer:
  module: imputer
  config:
    numeric_strategy: "median"
    categorical_strategy: "most_frequent"
    treat_outliers_as_na: false

encoder:
  module: encoder
  config:
    encoding_method: "one_hot"
```

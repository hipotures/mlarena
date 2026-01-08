# Feature Polynomial Sub-Module

## Overview

The **feature_polynomial** sub-module generates polynomial and interaction features using `sklearn.preprocessing.PolynomialFeatures`. It is designed for models that benefit from non-linear feature expansions.

**Module Name**: `feature_polynomial`  
**Location**: `src/mlarena/defaults/preprocessing/feature_polynomial.py`

## Capabilities
- **Polynomial Expansion**: Create features of degree 2 to 5.
- **Interaction Only**: Option to only create interaction terms (e.g., $x_1 \cdot x_2$) without powers ($x_1^2$).
- **NaN Protection**: Automatically skips execution if input contains `NaN` values (PolynomialFeatures requirement).
- **OOM Protection**: Estimates the number of output features and skips if it exceeds 10,000 to prevent Out-Of-Memory errors.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `poly_degree` | int \| null | `null` | Degree (2–5). `null` disables polynomial features |
| `poly_columns` | List[str] \| null | `null` | Columns to expand (null = all numeric) |
| `poly_include_bias` | bool | `false` | Include bias column (1.0) in output |
| `poly_interaction_only` | bool | `false` | If true, only interaction terms (no powers) |
| `max_generated_features` | int | `200` | Hard cap on total new columns created (truncates if exceeded) |

## Examples

### Polynomial Degree 2
```yaml
feature_poly_2:
  module: feature_polynomial
  cache: true
  config:
    poly_degree: 2
    poly_interaction_only: false
```

### Degree 3 Interactions Only
```yaml
feature_poly_3_inter:
  module: feature_polynomial
  cache: true
  config:
    poly_degree: 3
    poly_interaction_only: true
    max_generated_features: 500
```

## Artifacts
- `feature_polynomial_report.json`: Details about degree, columns, and generated names.
- `summary.json`: Standard preprocessing summary.

## Notes & Tips
- **Pre-requisite**: **Requires no missing values.** Always place an `imputer` before this module in the chain.
- **Complexity**: Polynomial expansions grow very quickly. Degree 3 on 50 columns produces ~20,000 features. Use `poly_columns` to limit the scope or `poly_interaction_only: true`.
- **Scaling**: Polynomial features often result in very different scales. Placing a `scaler` after this module is highly recommended.

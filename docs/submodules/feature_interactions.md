# Feature Interactions Sub-Module

## Overview

The **feature_interactions** sub-module creates simple arithmetic interaction features (add, sub, mul, div) between numeric column pairs. It supports both explicit pair definitions and automatic pairing of numeric columns.

**Module Name**: `feature_interactions`  
**Location**: `src/mlarena/defaults/preprocessing/feature_interactions.py`

## Capabilities
- **Numeric Interactions**: Performs basic arithmetic (+, -, *, /) on pairs of columns.
- **Explicit Pairs**: Manually specify which columns to interact.
- **Auto-Pairing**: Automatically find and interact numeric columns (limited by `max_auto_pairs`).
- **Safety Guards**: `max_generated_features` prevents explosion of the feature space.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `interaction_types` | List[str] | `[]` | Operations to create: `add`, `sub`, `mul`, `div` |
| `numeric_pairs` | List[List[str]] | `[]` | Explicit pairs e.g., `[["col1", "col2"], ["a", "b"]]` |
| `auto_pair_numeric` | bool | `false` | Auto-generate pairs from all numeric columns |
| `max_auto_pairs` | int | `30` | Max auto-generated pairs (applied to combinations in order) |
| `max_generated_features` | int | `200` | Hard cap on total new columns created |

## Examples

### Basic Interactions (Auto-Pairs)
```yaml
feature_interactions_auto:
  module: feature_interactions
  cache: true
  config:
    interaction_types: ["add", "mul"]
    auto_pair_numeric: true
    max_auto_pairs: 5
    max_generated_features: 50
```

### Explicit Pairs
```yaml
feature_interactions_manual:
  module: feature_interactions
  cache: true
  config:
    interaction_types: ["sub", "div"]
    numeric_pairs:
      - ["price", "discount"]
      - ["clicks", "views"]
```

## Artifacts
- `feature_interactions_report.json`: Lists new columns and interaction details.
- `summary.json`: Standard preprocessing summary.

## Notes & Tips
- **Division**: Sets division-by-zero results to `NaN`. Handle these downstream using an `imputer` if necessary.
- **Unique Names**: Automatically generates unique names (e.g., `colA_mul_colB`) and handles collisions.
- **Dtypes**: Only works on numeric columns. Ensure categories are encoded before this step if you need them included.

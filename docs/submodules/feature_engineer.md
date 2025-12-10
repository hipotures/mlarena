# Feature Engineer Sub-Module

## Overview

The **feature_engineer** sub-module creates interaction, polynomial, and group-aggregation features in one configurable step. It is designed to stay generic (one code path) while letting YAML pick which families of features to generate and how many. A safety cap (`max_generated_features`) prevents explosion.

**Module Name**: `feature_engineer`  
**Location**: `config/code/preprocessing/feature_engineer.py`

## Capabilities
- Numeric interactions: add/subtract/multiply/divide across selected pairs (explicit or auto-generated).
- Polynomial features: configurable degree, interaction-only toggle, optional bias term.
- Group aggregations: group-by keys with multiple value columns and aggregations (mean/std/min/max/count/nunique, etc.).
- Guards: `max_generated_features` limits total new columns; `max_auto_pairs` limits auto pairing.

## Parameters

### Interaction Features
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `interaction_types` | List[str] | `[]` | Operations to create: `add`, `sub`, `mul`, `div` |
| `numeric_pairs` | List[List[str]] | `[]` | Explicit pairs e.g., `[[\"col1\", \"col2\"], [\"a\", \"b\"]]` |
| `auto_pair_numeric` | bool | `false` | Auto-generate pairs from all numeric columns |
| `max_auto_pairs` | int | `30` | Max auto-generated pairs (applied to combinations in order) |

### Polynomial Features
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `poly_degree` | int \| null | `null` | Degree (2–5). `null` disables polynomial features |
| `poly_columns` | List[str] \| null | `null` | Columns to expand (null = all numeric) |
| `poly_include_bias` | bool | `false` | Include bias column in polynomial output |
| `poly_interaction_only` | bool | `false` | If true, only interaction terms (no powers) |

### Group Aggregations
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_keys` | List[str] | `[]` | Columns to group by |
| `group_value_cols` | List[str] | `[]` | Value columns to aggregate |
| `aggs` | List[str] | `[]` | Aggregations to compute (`mean`, `std`, `min`, `max`, `count`, `nunique`, etc.) |

### Safety / Limits
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_generated_features` | int | `200` | Hard cap on total new columns created across all steps |

## Examples

### Basic Interactions (Auto-Pairs)
```yaml
feature_engineer_auto_pairs:
  module: feature_engineer
  cache: true
  config:
    interaction_types: ["add", "mul"]
    auto_pair_numeric: true
    max_auto_pairs: 5       # first 5 numeric pairs
    max_generated_features: 50
```

### Explicit Pairs + Polynomial Degree 2
```yaml
feature_engineer_poly2:
  module: feature_engineer
  cache: true
  config:
    interaction_types: ["add", "div"]
    numeric_pairs:
      - ["price", "discount"]
      - ["clicks", "views"]
    poly_degree: 2
    poly_columns: ["price", "discount", "views"]
    poly_interaction_only: false
    max_generated_features: 120
```

### Group Aggregations Only
```yaml
feature_engineer_group:
  module: feature_engineer
  cache: true
  config:
    group_keys: ["customer_id"]
    group_value_cols: ["purchase_amount", "visit_duration"]
    aggs: ["mean", "std", "max", "nunique"]
    max_generated_features: 40
```

### Strict Cap (e.g., only 3 new columns total)
```yaml
feature_engineer_small:
  module: feature_engineer
  cache: true
  config:
    interaction_types: ["mul"]
    auto_pair_numeric: true
    max_auto_pairs: 3
    max_generated_features: 3
```

## Artifacts
- `feature_engineering_report.json`: Lists new columns, interaction details, polynomial settings, group aggregation info, and total generated features.
- `summary.json`: Standard preprocessing summary (shape/column changes, config snapshot).

## State Dictionary (`fit_transform` return)
```python
{
    "version": "1.0",
    "new_columns": [...],
    "interactions": [
        {"type": "interaction", "operation": "mul", "columns": ["a", "b"], "new_column": "a_mul_b"},
        ...
    ],
    "polynomial": { "type": "polynomial", "degree": 2, "generated_columns": [...] },
    "group_aggregations": {
        "type": "group_aggregation",
        "group_keys": ["customer_id"],
        "value_columns": ["amount"],
        "aggs": ["mean", "std"],
        "generated_columns": [...]
    },
    "config": {...}  # user config without internal keys
}
```

## Notes & Tips
- Works on numeric columns for interactions/polynomials; ensure encoders run earlier if you need encoded categorical features included.
- `max_generated_features` guards against runaway expansion; lower it for small experiments.
- Avoid leakage: don’t include target in `group_value_cols`.
- Division sets division-by-zero results to NaN; handle downstream as needed.
- Order in chains: typically after encoding and before drift detection/feature selection.

# Categorical Encoder Sub-Module

## Overview

The **categorical_encoder** sub-module detects and converts columns to the pandas `category` dtype. It is specifically designed to leverage EDA metadata and auto-detection of numeric-encoded categorical features (like binary flags or ordinal scales), making it ideal for GBDT models (XGBoost, LightGBM, CatBoost) that have native categorical support.

**Module Name**: `categorical_encoder`  
**Location**: `src/mlarena/defaults/preprocessing/categorical_encoder.py`

## Capabilities
- **EDA Integration**: Automatically uses metadata from `ydata-profiling` (if `mla eda` was run) to identify `Categorical` and `Text` types.
- **Auto-Detection**: Scans numeric columns for low-cardinality patterns (e.g., columns with < 25 unique values and < 1% unique ratio).
- **Type Classification**: Identifies and logs whether a numeric column is `Binary`, `Ordinal`, or `Nominal`.
- **Comprehensive Summary**: Generates a rich table in the console showing every feature's detected type, cardinality, and how it was converted.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_cardinality` | int | `50` | Maximum distinct values for EDA-detected categorical columns |
| `exclude_text_type` | bool | `false` | Whether to skip columns identified as "Text" by EDA |
| `include_numeric_categories` | bool | `true` | Include low-cardinality numeric columns identified by EDA |
| `enable_auto_detect` | bool | `true` | Enable auto-detection of numeric categorical columns |
| `auto_detect_threshold` | int | `25` | Maximum unique values for numeric auto-detection |

## Examples

### Basic Auto-Detection
```yaml
categorical_auto:
  module: categorical_encoder
  cache: true
  config:
    enable_auto_detect: true
    auto_detect_threshold: 25
```

### Strict EDA-based
```yaml
categorical_strict:
  module: categorical_encoder
  cache: true
  config:
    max_cardinality: 20
    include_numeric_categories: false
    enable_auto_detect: false
```

### Full Boost Support
```yaml
categorical_boost:
  module: categorical_encoder
  cache: true
  config:
    max_cardinality: 50
    exclude_text_type: false
    include_numeric_categories: true
    enable_auto_detect: true
    auto_detect_threshold: 25
```

## Artifacts
- `summary.json`: standard preprocessing summary (shape/column changes).
- Console Table: A detailed "ALL FEATURES TYPE SUMMARY" is printed to the console during execution.

## State Dictionary (`fit_transform` return)
```python
{
    "categorical_columns": ["Sex", "Pclass", "Embarked", "is_gold_member"],
    "eda_metadata": {...},
    "auto_detect_metadata": {...},
    "all_categorical_metadata": {...},
    "conversion_summary": {
        "train_converted": 12,
        "test_converted": 12,
        ...
    }
}
```

## Notes & Tips
- **Prerequisite**: Running `mla eda project=<project>` first is highly recommended as this module uses the generated `eda_summary.json`.
- **Difference from `encoder`**: Unlike the `encoder` module which performs transformations like One-Hot or Target Encoding, this module only changes the pandas **dtype** to `category`. This is what GBDT models need for their internal categorical handling.
- **Alignment**: It ensures that categorical levels are consistent across train, validation, and test sets.
- **System Columns**: Automatically excludes `id` and the target column from conversion.

# feat Module - Feature Engineering

The `feat` module applies lightweight, template-driven feature transformations to raw train and test data. Unlike the preprocessing pipeline, `feat` operates independently and is designed for quick feature engineering experiments.

## Overview

- **Module name**: `feat`
- **Dependencies**: None
- **Part of auto-flow**: No (manual use only)
- **Source**: `src/mlarena/modules/feat.py`

## Purpose

The `feat` module provides a simple way to apply common feature transformations without creating custom preprocessing modules. It's useful for:

- Quick feature engineering experiments
- Testing impact of log transformations
- Creating ratio features between numeric columns
- Dropping columns that don't improve model performance

## Usage

### Basic Usage

```bash
# Apply feature template to a project
uv run python scripts/mla.py feat --project <competition-slug> --feat-template <template-name>
```

### With Specific Template

```bash
# Use custom feature template
uv run python scripts/mla.py feat --project titanic --feat-template log_ratios
```

## Template Configuration

Feature templates are YAML files that define transformations to apply.

### Template Structure

```yaml
# projects/kaggle/<slug>/templates/feat/my_features.yaml
log1p:
  - Age
  - Fare

ratios:
  - numerator: Fare
    denominator: Age
    name: fare_per_age
  - numerator: SibSp
    denominator: Parch
    name: sibling_parent_ratio

drop_columns:
  - Ticket
  - Cabin
```

### Supported Operations

#### 1. log1p

Apply natural logarithm plus one (log(x + 1)) to specified columns. Useful for normalizing skewed distributions.

```yaml
log1p:
  - column_name_1
  - column_name_2
```

- **Input**: List of column names
- **Effect**: Transforms `x` to `log(x + 1)` for each specified column
- **Note**: Handles None/NA values gracefully

#### 2. ratios

Create ratio features from pairs of numeric columns. Useful for capturing relationships between features.

```yaml
ratios:
  - numerator: column_a
    denominator: column_b
    name: a_over_b  # optional, defaults to "column_a_over_column_b"
  - numerator: column_c
    denominator: column_d
```

- **Input**: List of ratio definitions
- **Required fields**: `numerator`, `denominator`
- **Optional fields**: `name` (output column name)
- **Effect**: Creates new column with `numerator / denominator`
- **Note**: Division by zero handled with `pd.NA`

#### 3. drop_columns

Remove specified columns from the dataset. Useful for excluding low-value or problematic features.

```yaml
drop_columns:
  - column_to_remove_1
  - column_to_remove_2
```

- **Input**: List of column names
- **Effect**: Drops listed columns if they exist
- **Note**: Ignores columns that don't exist (no error)

## Outputs

The module creates the following artifacts in `experiments/<exp_id>/artifacts/feat/`:

1. **train_features.csv**: Transformed training data
2. **test_features.csv**: Transformed test data
3. **features_meta.json**: Metadata including:
   - Template name used
   - List of columns in transformed data

## Examples

### Example 1: Log Transformations

```yaml
# templates/feat/log_transform.yaml
log1p:
  - Age
  - Fare
  - Income
```

```bash
uv run python scripts/mla.py feat --project titanic --feat-template log_transform
```

### Example 2: Ratio Features

```yaml
# templates/feat/ratios.yaml
ratios:
  - numerator: Fare
    denominator: Age
    name: fare_per_age
  - numerator: Cabin_Count
    denominator: Family_Size
    name: cabin_per_person
```

### Example 3: Combined Transformations

```yaml
# templates/feat/combined.yaml
log1p:
  - Fare
  - Income

ratios:
  - numerator: Fare
    denominator: Age
    name: fare_per_age

drop_columns:
  - Ticket
  - Cabin
  - Name
```

## Integration with Pipeline

The `feat` module operates independently of the preprocessing pipeline. If you want to use feature engineering in a full experiment:

1. Run `feat` first to create transformed data
2. Use the output files as input to your model training
3. Or incorporate transformations into a preprocessing template instead

**Note**: For production pipelines, prefer using preprocessing templates over `feat`, as they integrate better with the caching system.

## Template Locations

- **Global templates**: `src/mlarena/templates/feat/*.yaml`
- **Project templates**: `projects/kaggle/<slug>/templates/feat/*.yaml`

Project templates override global templates when names collide.

## Default Template

The default template is `identity`, which returns data unchanged:

```bash
# No transformations applied
uv run python scripts/mla.py feat --project titanic
# Equivalent to:
uv run python scripts/mla.py feat --project titanic --feat-template identity
```

## Tips and Best Practices

1. **Start simple**: Test one transformation at a time to measure impact
2. **Check distributions**: Use EDA to verify log transforms improve normality
3. **Avoid leakage**: Don't create ratios with target-dependent columns
4. **Handle missing values**: `feat` preserves NA values; use preprocessing imputation first if needed
5. **Consider preprocessing instead**: For production pipelines, move transformations to preprocessing templates for better caching

## Error Handling

- Missing columns are skipped silently (no error)
- Division by zero results in `pd.NA` (not NaN)
- Template not found raises clear error with available templates listed
- Invalid YAML syntax shows parsing error with line number

## See Also

- [Preprocessing Guide](submodules/README.md) - For production-ready feature engineering
- [Model Templates](model_templates.md) - How to pair features with models
- [Feature Engineer Submodule](submodules/feature_engineer.md) - Advanced feature creation in preprocessing

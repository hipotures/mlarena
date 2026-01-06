# MLArena Terminology Guide

This guide explains naming conventions used throughout MLArena documentation and code to avoid confusion.

---

## Parameter Naming Conventions

MLArena uses different naming conventions depending on context:

| Context | Convention | Example | When to Use |
|:--------|:-----------|:--------|:------------|
| **Python code** | `snake_case` | `preprocess_template` | In Python files, function parameters, variable names |
| **YAML keys** | `kebab-case` | `preprocess-template` | In template YAML files, config.yaml keys |
| **CLI dotted overrides** | `snake_case` with dots | `preprocess_template=value` | Command-line parameter overrides (recommended) |
| **CLI flags (legacy)** | `kebab-case` with `--` | `--preprocess-template value` | Older flag format (still works, but dotted preferred) |
| **Module names** | `kebab-case` | `fetch-score` | Module registration and subcommand names |

### Examples

**Python code:**
```python
def train(preprocess_template: str, model_template: str):
    experiment_id = "exp-20251217-152730"
```

**YAML template:**
```yaml
# src/mlarena/templates/model/baseline.yaml
model: autogluon_baseline
preprocess-template: baseline
config:
  time-limit: 300
```

**CLI dotted override (recommended):**
```bash
uv run python scripts/mla.py --project titanic \
  preprocess_template=baseline \
  model_template=cpu-dev-5m \
  common.time_limit=600
```

**CLI flag format (legacy, still works):**
```bash
uv run python scripts/mla.py --project titanic \
  --preprocess-template baseline \
  --model-template cpu-dev-5m \
  --time-limit 600
```

---

## Product and Package Names

| Name | Usage | Example |
|:-----|:------|:--------|
| **MLArena** | Product name in prose, marketing | "MLArena is a framework for Kaggle competitions" |
| **mlarena** | Python package name, imports | `from mlarena.core import pipeline` |
| **mla** | CLI command name | `uv run python scripts/mla.py` |
| **mla.py** | Entry point script | `scripts/mla.py` |

**Correct usage:**
```python
# Import
from mlarena.modules import ModelModule

# Documentation
"MLArena provides a unified workflow..."

# Command line
uv run python scripts/mla.py --project titanic
```

---

## Common Parameter Variations

### experiment_id vs exp-id

Both formats work in CLI due to automatic conversion:

```bash
# Both are equivalent:
--exp-id eda
experiment_id=eda
```

**Rule:** Use `experiment_id` in Python code, either format in CLI.

### fetch-score vs fetch_score

This is a module name that appears in different contexts:

```bash
# CLI module name (kebab-case):
uv run python scripts/mla.py fetch-score --project titanic

# Python module (underscore):
from mlarena.modules.fetch_score import FetchScoreModule

# State file (kebab-case):
"modules": {
  "fetch-score": {
    "status": "completed"
  }
}
```

**Rule:** CLI and state files use `fetch-score`, Python imports use `fetch_score`.

---

## Special Cases

### AutoGluon vs autogluon

| Context | Format | Example |
|:--------|:-------|:--------|
| Product name | `AutoGluon` | "AutoGluon is an AutoML framework" |
| Python import | `autogluon` | `from autogluon.tabular import TabularPredictor` |
| Template value | `autogluon_baseline` | `model: autogluon_baseline` |

### Preprocess vs Preprocessing

| Context | Format |
|:--------|:-------|
| Module name | `preprocess` |
| Directory name | `preprocessing/` |
| Documentation | Both "preprocessing" and "preprocess" used interchangeably |
| State file | `"preprocess"` (module name) |

**Recommendation:** Use "preprocessing" when referring to the concept, "preprocess" when referring to the module or command.

---

## Dash vs Underscore: Quick Reference

**Use dashes (`-`):**
- YAML keys: `time-limit`, `model-template`
- CLI flags (legacy): `--preprocess-template`
- Module names: `fetch-score`, `init`, `eda`

**Use underscores (`_`):**
- Python code: `time_limit`, `model_template`
- CLI dotted overrides: `model_template=`, `common.seed=`
- Python imports: `fetch_score`, `__init__`

---

## Template Naming

Templates use `kebab-case` with descriptive names:

**Model templates:**
```
cpu-fast-1m           # cpu, fast training, 1 minute
gpu-dev-5m            # gpu, development, 5 minutes
cpu-best-8h           # cpu, best quality, 8 hours
hpo_boost_medium      # HPO preset, medium trials
```

**Preprocess templates:**
```
baseline              # Basic preprocessing
full-pipeline         # Complete preprocessing chain
imputer_median        # Specific imputation strategy
categorical_boost     # Categorical encoding for boosting models
```

**Naming pattern:**
```
{hardware}-{quality}-{time}     # For model templates
{method}_{variant}              # For preprocess templates
```

---

## CLI Parsing: Flag vs Override

MLArena supports two parameter formats that are internally equivalent:

### Flag Format (Legacy)

```bash
--parameter-name value
```

**Behavior:**
- Converted to `parameter_name=value` internally
- Common parameters (`time_limit`, `use_gpu`, `preset`, `seed`) automatically prefixed with `common.`
- Works but less explicit

**Example:**
```bash
uv run python scripts/mla.py model --project titanic --time-limit 600
# Internally becomes: common.time_limit=600
```

### Dotted Override Format (Recommended)

```bash
key.subkey=value
```

**Behavior:**
- Explicit path to parameter
- No automatic prefixing
- Shows exact config structure
- Recommended for clarity

**Example:**
```bash
uv run python scripts/mla.py model --project titanic common.time_limit=600
```

### Best Practice

**Use flags for:**
- Core CLI arguments: `--project`, `--exp-id`, `--profile`, `--force`

**Use dotted overrides for:**
- All configuration parameters: `model_template=`, `common.seed=`, `model.hyperparameters.GBM.max_depth=`

**Example (recommended style):**
```bash
uv run python scripts/mla.py model \
  --project titanic \
  --exp-id eda \
  model_template=cpu-best-1h \
  common.time_limit=3600 \
  common.seed=42
```

---

## Common Mistakes

### ❌ Incorrect

```bash
# Mixing conventions inconsistently
uv run python scripts/mla.py --project titanic model-template=cpu-fast-1m

# Using Python names in YAML
model_template: baseline  # Should be: model or preprocess-template

# Using dashes in dotted overrides
common.time-limit=600  # Should be: common.time_limit=600
```

### ✅ Correct

```bash
# Consistent use of dotted overrides
uv run python scripts/mla.py --project titanic model_template=cpu-fast-1m

# Proper YAML format
model: autogluon_baseline
preprocess-template: baseline

# Proper dotted override
common.time_limit=600
```

---

## Cross-Reference

For more details on configuration:
- **Configuration system**: See [configs.md](configs.md)
- **CLI parsing behavior**: See [README.md](../README.md#cli-parsing-behavior)
- **Template system**: See [architecture.md](architecture.md#templates-and-resolution)

---

## Summary

**Quick Decision Guide:**

| If you're... | Use... | Example |
|:-------------|:-------|:--------|
| Writing Python code | `snake_case` | `preprocess_template = "baseline"` |
| Writing YAML config | `kebab-case` | `preprocess-template: baseline` |
| Using CLI | `dotted.snake_case=value` | `preprocess_template=baseline` |
| Naming a module | `kebab-case` | `fetch-score` |
| Referring to product | `MLArena` | "MLArena framework" |

**When in doubt:** Check existing code in the relevant context (Python file, YAML file, or CLI example).

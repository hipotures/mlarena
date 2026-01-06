# Configuration System

## Complete Parameter Reference

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `lock` | bool | `false` | Create `overwrite.lock` after successful completion to prevent re-runs |
| `skip_deps` | bool | `false` | Skip dependency resolution (run only target module) |
| `show_payload` | bool | `false` | Display module output payload in console |
| `model.mla_retention` | bool | `false` | Clean up AutoGluon intermediate models after training (saves disk space). Can be set via CLI as `model.mla_retention=true`. |

## Built-in Profile Fallbacks

If profile YAML files don't exist, the system provides hardcoded fallbacks for:
- **smoke**: `{common: {time_limit: 60, preset: "medium", use_gpu: false}}`
- **dev**: `{common: {time_limit: 300, preset: "high", use_gpu: false}}`

**For naming conventions and parameter format, see:** [Terminology Guide](TERMINOLOGY.md)

---

## Configuration Levels

## Core Concepts

The configuration is structured as a tree (the `GlobalConfig` class) which includes sections for shared settings and individual module configurations.

### Hierarchy & Merging Priority

Settings are merged in the following order (from lowest to highest priority):

1.  **Hardcoded Defaults**: Base values defined in the Python code (`src/mlarena/core/conf.py`).
2.  **Profiles**: Reusable sets of parameters (e.g., `smoke` for fast testing).
3.  **Project Config**: The `config.yaml` file inside your project directory.
4.  **Template Config**: Config defined within model or preprocessing YAML templates.
5.  **CLI Overrides**: Any `key=value` pair provided in the command line (e.g., `common.seed=123`).

## Configuration Structure

The root config tree contains several key sections:

### 1. Core Metadata
- `project`: (Required) The competition slug.
- `experiment_id`: Current experiment identifier.
- `profile`: Name of the active profile.

### 2. Auto-flow & Global Settings
- `model_template`: Default model template to use (default: `baseline`).
- `preprocess_template`: Preprocessing template or chain.
- `wait_seconds`: (Integer, default: 30) Seconds to wait for Kaggle processing before fetching score.
- `skip_submit`: (Boolean, default: `false`) Generate predictions but don't upload to Kaggle.
- `skip_git`: (Boolean, default: `false`) Skip automatic git commit after auto-flow.
- `force`: (Boolean, default: `false`) Force re-execution of completed modules.
- `lock`: (Boolean, default: `false`) Create an `overwrite.lock` file in the experiment directory after successful completion to prevent accidental re-runs.
- `skip_deps`: (Boolean, default: `false`) Skip dependency resolution (run only the target module).
- `show_payload`: (Boolean, default: `false`) Show module output payload in the console.
- `model.mla_retention`: (Boolean, default: `false`) If `true`, cleans up AutoGluon intermediate models to save disk space after training.

### 3. Common Section (`common`)
Parameters used as fallbacks by multiple modules:
- `seed`: Global random seed (default: 42).
- `time_limit`: Global time limit for model training.
- `use_gpu`: Whether to use GPU acceleration.
- `preset`: AutoGluon quality preset.

### 4. Module Sections
Each module (init, eda, preprocess, model, etc.) has its own dictionary for custom parameters.
Example: `model.hyperparameters.GBM.max_depth=5`

---

## Profiles

Profiles allow you to switch between different execution modes quickly.

### Using a Profile
```bash
uv run python scripts/mla.py model --project titanic --profile smoke
```

### Predefined Profiles
- **`smoke`**: Designed for fast verification. Sets short time limits and medium quality presets.
- **`dev`**: Standard development settings with moderate time limits.

### Custom Profiles
You can create custom profiles in `src/mlarena/templates/profiles/<name>.yaml` or project-local `projects/kaggle/<slug>/templates/profiles/<name>.yaml`.

---

## CLI Overrides

You can override any configuration value using dotted path notation.

### Simple Overrides
```bash
uv run python scripts/mla.py --project titanic common.seed=777 force=true
```

### Module-Specific Overrides
```bash
uv run python scripts/mla.py model --project titanic \
  model.time_limit=600 \
  model.included_model_types=["GBM", "CAT"]
```

### Nested Dictionary Overrides
```bash
uv run python scripts/mla.py model --project titanic \
  model.hyperparameters.GBM.num_boost_round=1000
```

---

## Project Config (`config.yaml`)

Each project can have a `config.yaml` at its root (`projects/kaggle/<slug>/config.yaml`) to set permanent defaults for that competition.

```yaml
# projects/kaggle/titanic/config.yaml
common:
  time_limit: 3600
  use_gpu: true

model_template: "cpu-best-1h"
```

## Validation

All configurations are validated using **Pydantic** before execution. This ensures that:
- Required fields are present.
- Data types are correct (e.g., `seed` must be an integer).
- Unknown fields (outside module-specific dicts) are caught early.

```
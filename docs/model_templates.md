# Model Templates

Model templates are YAML files that describe which model implementation to load and the training hyperparameters to apply. Templates keep experiments reproducible and make it easy to switch between fast smoke runs and longer production-grade training.

## Where templates live

- **Global defaults**: `src/mlarena/templates/model/*.yaml`
- **Project overrides**: `projects/kaggle/<slug>/templates/model/*.yaml`

Project-local files override global templates when names collide.

## Template format

Each file contains a single template (no `templates:` wrapper). Example:

```yaml
# projects/kaggle/titanic/templates/model/cpu-dev-5m.yaml
model: autogluon_baseline
preprocess_template: baseline
config:
  preset: medium
  time_limit: 300          # seconds
  use_gpu: false
  excluded_model_types: [] # optional AutoGluon exclusions
```

Common keys:

- `model`: Python module to load (project module preferred over `src/mlarena/defaults/models/` when both exist).
- `preprocess_template`: Name of the preprocessing template to pair with this model (optional).
- `mla_retention`: Boolean (default: `false`). If `true`, cleans up intermediate AutoGluon models after training to save disk space while keeping the best model for predictions.
- `config`: Passed through to the model implementation; typically includes `preset`, `time_limit`, `use_gpu`, `hyperparameters`, and optional HPO fields such as `hpo_preset`, `search_space`, and `hyperparameter_tune_kwargs`.

## AutoGluon HPO layout

When using AutoGluon native HPO in templates:

- Put **tunable ranges** under `search_space` (lists/ranges only).
- Put **static model params** under `hyperparameters.<MODEL>` (for example `ag_args_fit`, `task_type`, `devices`).
- `hpo_preset` can live at top-level or inside `config` (the loader flattens `config` keys).

Example:

```yaml
model: autogluon_baseline
hpo_preset: hpo_boost_medium
config:
  preset: high_quality
  use_gpu: true
  included_model_types: [CAT]

  hyperparameters:
    CAT:
      task_type: GPU
      devices: "0"
      ag_args_fit:
        num_gpus: 1

  search_space:
    CAT:
      depth: [4, 10, int]
      learning_rate: [0.01, 0.3, log]
```

Note: `search_space` is parsed by `mlarena.utils.hpo_space` and non-list values are passed through, but keeping static params under `hyperparameters` is clearer.

## Listing templates

```bash
uv run python scripts/mla.py model model_template=list --project <slug>
```

The CLI shows whether templates come from global defaults or the project folder.

## Overriding configuration at runtime

- Overrides like `model.preset=high`, `model.time_limit=600`, `common.use_gpu=true` override the template’s `config` fields.
- Convenience profiles apply presets without editing YAML:
  - `--profile dev`: `preset=high`, `time_limit=300`, `use_gpu=0`
  - `--profile smoke`: `preset=medium`, `time_limit=60`, `use_gpu=0`
- Additional template parameters (for example, `excluded_model_types`, `hyperparameters`) remain intact unless explicitly overridden.

## Custom models

Implement project-specific models in `projects/kaggle/<slug>/code/models/<name>.py`. The loader checks the project path first, then falls back to `src/mlarena/defaults/models/`. Custom modules must expose:

```python
def train(train_df, val_df, config, artifacts=None):
    ...
```

Use `model: <name>` inside the template to reference your implementation. Pair with `preprocess_template` to ensure the correct processed datasets are used.

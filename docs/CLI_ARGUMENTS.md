# CLI Arguments

MLArena uses `key=value` overrides only. Global `--flags` are not supported.

## Basics

```bash
uv run python scripts/mla.py <module?> project=<slug> [overrides...]
```

Examples:

```bash
# Auto-flow
uv run python scripts/mla.py project=titanic model_template=cpu-fast-1m common.time_limit=600

# Single module
uv run python scripts/mla.py model project=titanic experiment_id=eda model_template=cpu-best-1h force=true
```

## Core overrides

- `project=<slug>` (required)
- `experiment_id=<name>`
- `profile=smoke|dev|...`
- `model_template=<name>`
- `preprocess_template=<name>` or `preprocess_template=imputer,scaler`
- `force=true`
- `skip_submit=true`
- `skip_git=true`
- `json_output=true`
- `wait_seconds=<int>`

## Dotted paths

Use dotted paths to reach nested config:

```bash
# Common defaults
common.seed=123
common.time_limit=600

# Model tuning
model.time_limit=1200
model.preset=high
model.hyperparameters.GBM.max_depth=6
```

## Lists

Use list syntax for multi-value params:

```bash
stack.prediction_files='[exp1/submission.csv,exp2/submission.csv]'
```

Tip: Quote list values to avoid shell globbing (especially in zsh).

## preprocess-tune: Optuna

Prefer the `optuna.*` namespace for Optuna settings:

```bash
uv run python scripts/mla.py preprocess-tune \
  project=titanic \
  optuna.study_name=test_optuna_titanic_01 \
  optuna.n_trials=20 \
  preprocess_tune.model_template=cpu-fast-1m
```

Legacy compatibility: `preprocess_tune.study_name`, `preprocess_tune.n_trials`, etc still work, but `optuna.*` is preferred.

## preprocess-tune: MCTS

Enable MCTS and pass any `mcts.*` config keys directly:

```bash
uv run python scripts/mla.py preprocess-tune \
  project=titanic \
  mcts.enabled=true \
  mcts.study_name=test_mcts_titanic_01 \
  mcts.budget=2 \
  mcts.parallelism.workers=1
```

Short list of supported MCTS keys:

- `mcts.study_name`, `mcts.direction`, `mcts.storage_url`, `mcts.budget`, `mcts.max_depth`, `mcts.seed`
- `mcts.selection_policy`, `mcts.exploration_weight`, `mcts.prior_policy`
- `mcts.expansion_width`, `mcts.expansion_alpha`
- `mcts.param_expansion_width`, `mcts.param_expansion_alpha`, `mcts.param_expansion_max_samples`
- `mcts.model_verbosity`, `mcts.model_cleanup`, `mcts.cleanup_processed`, `mcts.debug`
- Nested sections: `mcts.penalties.*`, `mcts.parallelism.*`, `mcts.multi_fidelity.*`, `mcts.pruning.*`, `mcts.templates.*`, `mcts.dedupe.*`

## Module-specific flags

Some modules expose their own flags that are not part of config (these are allowed only for those modules):

- `mla experiments`: `--status`, `--sort-by`, `--reverse`, `--show-table`
- `mla submissions`: `--limit`, `--sort-by`, `--public`, `--private`, etc
- `mla queue`: passes through to `scripts/task_queue.py` (e.g., `--priority`, `--max-tasks`)

Use `--help` with a module to see its supported flags.

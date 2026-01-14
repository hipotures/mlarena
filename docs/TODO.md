# Development TODO

> **Note:** This file tracks development tasks and system enhancement ideas.

## Performance Optimization

### Completed
- [x] Remove toplevel pandas imports from modules (moved to execute() functions).
- [x] Remove toplevel pandas import from utils/init/core.py.
- [x] Skip ModuleRegistry.clear() on every run (use cached imports).

### Planned Optimizations
- [ ] **State caching** - Don't reload state.json when module status is already `completed`.
- [ ] **Lazy config loading** - Defer project config.py loading until needed in `execute()`.
- [ ] **Parallel module discovery** - Use ThreadPoolExecutor for importing modules.
- [ ] **Bytecode compilation** - Pre-compile modules to .pyc for faster startup.
- [ ] **Preprocess timing audit** - Analyze 100–200 Optuna trials’ `state.json` to rank slow steps (avg/median/p95), then decide if targeted Polars migration is worth it.

## Documentation (In Progress)

- [x] Create comprehensive sub-module documentation (`docs/submodules/*.md`).
- [x] Document the hierarchical configuration system (`docs/configs.md`).
- [x] Document the `submissions` and `experiments` listing modules.
- [ ] Add architecture diagrams to `docs/architecture.md`.
- [ ] Create a "Troubleshooting" section in the main guide.

## System Enhancements

### Experiment Management
- [ ] Add explicit `aborted` status for user-interrupted runs.
- [ ] Implement `admin clean` command to prune old experiments/artifacts.
- [ ] **Relative Path Portability**: Convert all absolute paths in `state.json` to relative (project root) to support cross-machine syncing.

### Feature Selection & Reporting
- [ ] Add `score_direction` metadata to `feature_selection_report.json`.
- [ ] Allow CLI overrides for individual steps in a preprocessing chain.

### Sample Weight & Drift Optimization (AutoGluon)
- [x] T1: Implement `weight_evaluation` support.
- [ ] T2: Weight Normalization (mean ≈ 1.0) in `adversarial_validation.py`.
- [ ] T3: Weight Clipping (percentile or fixed) to limit importance-weighting variance.
- [ ] T4: Drift-aware bagging via groups (generate `__grp__` in AV and pass to AG).
- [ ] T5: External dataset weight variants (strategies for weighting merged external data).

## CLI & UX
- [ ] Default pipeline mode: `mla --project X` starts the auto-flow without specifying a module.
- [ ] Implement `--from <module>` to resume auto-flow from a specific stage.
- [ ] Add `mla status` for a quick summary of recent runs.

## MCTS Preprocessing Search

### Visualization & Monitoring
- [ ] **Focused Live View**: Update `--live` to only render the 'Best Path' and 'Active Path' to maintain readability when tree exceeds 100+ nodes.
- [ ] **Rich Layout Dashboard**: Create a multi-pane CLI dashboard showing the tree on top and a "Top-K Leaderboard" at the bottom.
- [ ] **Interactive HTML Export**: Implement a script to export the SQLite search tree to a D3.js or Plotly based interactive HTML for deep offline analysis.
- [ ] **Surrogate Value Model**: Integrate a simple regressor to predict node values and guide search (surrogate-assisted MCTS).
- [ ] **Dynamic Time Penalty**: Replace fixed `time_lambda` with a dynamic penalty based on running average/std-dev of trial durations.

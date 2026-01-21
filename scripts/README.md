# MLArena Scripts

Central tools for managing Kaggle competition workflows.

## 🛠️ Utility Scripts

Standalone helper tools for project maintenance and advanced workflows are located in **[`utils/`](utils/)**:

- **[`clean.py`](utils/README.md#-cleanpy---artifact-cleanup)** - Remove AutoGluon artifacts to save disk space
- **[`sync.py`](utils/README.md#-syncpy---project-synchronization)** - Sync projects between machines using rsync
- **[`av_weights_mix.py`](utils/README.md#-av_weights_mixpy---adversarial-validation-with-external-data)** - Advanced adversarial validation with external datasets

See [utils/README.md](utils/README.md) for detailed usage instructions.

---

## Core Scripts

### 🎯 Main Entry Point

#### `mla.py`
MLArena CLI orchestrator. This is the primary interface for all pipeline operations.

```bash
# Run full auto-flow (init → eda → preprocess → model → predict → submit)
uv run python scripts/mla.py project=Titanic

# Run specific module
uv run python scripts/mla.py model project=Titanic model_template=cpu-dev-5m

# Use profile for quick testing
uv run python scripts/mla.py project=Titanic profile=smoke

# List available modules
uv run python scripts/mla.py modules
```

See [README.md](../README.md) and [docs/MLA_WORKFLOW_GUIDE.md](../docs/MLA_WORKFLOW_GUIDE.md) for detailed usage.

---

## Task Queue

### `task_queue.py`
Manage and execute queued experiments/tasks.

**CLI Usage:**
```bash
# List queued tasks
python scripts/task_queue.py --project Titanic list

# Add a task to the queue
python scripts/task_queue.py --project Titanic add "model model_template=lgbm skip_submit=true" --priority 5

# Run tasks from the queue
python scripts/task_queue.py --project Titanic run

# Run specific number of tasks
python scripts/task_queue.py --project Titanic run --max-tasks 3

# Remove a task
python scripts/task_queue.py --project Titanic remove 1
```

---

## Experiment Tracking & Analysis

### `experiment_logger.py`
Git-based experiment tracking with code snapshots for reproducibility.

**CLI Usage:**
```bash
# List experiments
uv run python scripts/experiment_logger.py --project Titanic list

# Show experiment details
uv run python scripts/experiment_logger.py --project Titanic show exp-20251217-182230

# Restore code snapshot
uv run python scripts/experiment_logger.py --project Titanic restore exp-20251217-182230
```

### `analyze_mcts_processors.py`
Analyze MCTS step impact (no change / improved / worsened) from `mcts.db`.

**CLI Usage:**
```bash
# Analyze largest study by trial count
python scripts/analyze_mcts_processors.py

# Point to a specific study id
python scripts/analyze_mcts_processors.py --study-id 12
```

### `analyze_mcts_trend.py`
Analyze optimization trends (maximize/minimize) over time from `mcts.db`.

**CLI Usage:**
```bash
# Analyze trend for a specific study (optional)
python scripts/analyze_mcts_trend.py --project Titanic --window 50
```

### `analyze_top_models.py`
Identify top-performing models and their training times from experiment artifacts.

**CLI Usage:**
```bash
# Run analysis (paths currently hardcoded to specific project structure, check source)
python scripts/analyze_top_models.py
```

---

## Submission Management

### `submissions_tracker.py`
Track submission history with local CV, public leaderboard, and private leaderboard scores.

**CLI Usage:**
```bash
# List all submissions for a project
uv run python scripts/submissions_tracker.py --project Titanic list

# Sort by public score
uv run python scripts/submissions_tracker.py --project Titanic list --sort-by public_score

# Update scores manually
uv run python scripts/submissions_tracker.py --project Titanic update 5 --public 0.8123

# Export to CSV for analysis
uv run python scripts/submissions_tracker.py --project Titanic export
```

### `submission_queue.py`
Manage submission queue for batch processing with duplicate detection.

**CLI Usage:**
```bash
# List queued submissions
python scripts/submission_queue.py --project Titanic list

# Submit from queue (by queue number, experiment-id, or filename)
python scripts/submission_queue.py --project Titanic submit 1

# Submit with auto fetch-score
python scripts/submission_queue.py --project Titanic submit 1 --continue-flow
```

### `blend_submissions.py`
Blend top-N Kaggle submissions using public scores as weights.

**CLI Usage:**
```bash
# Blend top 5 submissions
python scripts/blend_submissions.py --project Titanic --top-n 5 --weighting public --output-name blend.csv

# Include ensemble submissions (skipped by default)
python scripts/blend_submissions.py --project Titanic --top-n 5 --include-ensembles
```

### `fetch_scores_from_kaggle.py`
Fetch latest public scores from Kaggle and update local records.

**CLI Usage:**
```bash
# Update scores for a project
python scripts/fetch_scores_from_kaggle.py --project Titanic
```

---

## Optuna Optimization

### `optuna_dashboard.py`
Textual-based TUI dashboard for monitoring Optuna trials in real-time.

**CLI Usage:**
```bash
python scripts/optuna_dashboard.py --db projects/kaggle/Titanic/experiments/db/optuna.db --project-root projects/kaggle/Titanic
```

### `optuna_viz_explorer.py`
CLI tool to generate various Optuna visualizations (history, importance, contour, etc.).

**CLI Usage:**
```bash
python scripts/optuna_viz_explorer.py --db projects/kaggle/Titanic/experiments/db/optuna.db
```

---

## Internal Utilities

These scripts are used internally by the pipeline. You typically don't call them directly:

### `template_loader.py`
Loads and merges YAML templates (model/*.yaml, preprocess/*.yaml).
- Resolves project overrides
- Handles meta-templates (preprocessing chains)
- Used by `mla.py` and preprocessing modules

### `ai_helper.py`
AI-powered code generation for config.py and project setup.
- Used during `mla.py init`
- Generates competition-specific configurations
- Integrates with Claude API

---

## File Structure

```
scripts/
├── mla.py                      # Main CLI entry point ⭐
├── task_queue.py               # Task/Experiment queue management
├── submissions_tracker.py      # Submission tracking
├── submission_queue.py         # Submission queue management
├── blend_submissions.py        # Submission blending
├── fetch_scores_from_kaggle.py # Kaggle score syncing
├── experiment_logger.py        # Experiment tracking
├── analyze_mcts_processors.py  # MCTS analysis
├── analyze_mcts_trend.py       # MCTS trend analysis
├── optuna_dashboard.py         # Optuna TUI Dashboard
├── optuna_viz_explorer.py      # Optuna visualization generator
├── template_loader.py          # YAML template loader (internal)
├── ai_helper.py                # AI code generation (internal)
├── utils/                      # Standalone utilities
│   ├── clean.py
│   ├── sync.py
│   └── av_weights_mix.py
└── README.md                   # This file
```

## See Also

- [Main README](../README.md) - Repository overview
- [MLA Workflow Guide](../docs/MLA_WORKFLOW_GUIDE.md) - Pipeline documentation
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [Utility Scripts](utils/README.md) - Standalone tools documentation
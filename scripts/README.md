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

## Supporting Libraries

These scripts provide functionality used by the main pipeline and can also be used standalone:

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

**Python API:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from submissions_tracker import SubmissionsTracker

tracker = SubmissionsTracker(project_root)
tracker.add_submission(
    filename="submission.csv",
    model_name="autogluon_baseline",
    local_cv_score=0.8234,
    public_score=0.7987
)
tracker.display_submissions()
```

### `submission_queue.py`
Manage submission queue for batch processing with duplicate detection.

**CLI Usage:**
```bash
# List queued submissions
python scripts/submission_queue.py --project Titanic list

# Submit from queue (by queue number, experiment-id, or filename)
python scripts/submission_queue.py --project Titanic submit 1
python scripts/submission_queue.py --project Titanic submit exp-20251226-103504
python scripts/submission_queue.py --project Titanic submit submission.csv

# Submit with auto fetch-score (waits 30s, fetches score, removes on success)
python scripts/submission_queue.py --project Titanic submit 1 --continue-flow

# Remove from queue
python scripts/submission_queue.py --project Titanic remove 1
```

**Features:**
- Duplicate detection via Kaggle API (prevents re-submission)
- Error tracking with timestamps
- Status tracking: pending → submitted → completed (or failed)
- Auto-cleanup on successful fetch-score (with `--continue-flow`)
- Thread-safe operations using file locking

**Queue file:** `projects/kaggle/{project}/submissions/queue.json`

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
# Analyze largest study by trial count (default: /mnt/mlarena, playground-series-s6e1)
python scripts/analyze_mcts_processors.py

# Point to a specific study id
python scripts/analyze_mcts_processors.py --study-id 12

# Use a custom NFS root or project slug
python scripts/analyze_mcts_processors.py --root /mnt/mlarena --project playground-series-s6e1
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
├── submissions_tracker.py      # Submission tracking (CLI + library)
├── submission_queue.py         # Submission queue management (CLI)
├── experiment_logger.py        # Experiment tracking (CLI + library)
├── analyze_mcts_processors.py  # MCTS step impact analysis (CLI)
├── template_loader.py          # YAML template loader (internal)
├── ai_helper.py                # AI code generation (internal)
├── utils/                      # Standalone utilities
│   ├── clean.py               # Cleanup AutoGluon artifacts
│   ├── sync.py                # Project synchronization
│   ├── av_weights_mix.py      # Adversarial validation
│   └── README.md              # Detailed utils documentation
└── README.md                   # This file
```

---

## See Also

- [Main README](../README.md) - Repository overview
- [MLA Workflow Guide](../docs/MLA_WORKFLOW_GUIDE.md) - Pipeline documentation
- [CLAUDE.md](../CLAUDE.md) - Development guidelines
- [Utility Scripts](utils/README.md) - Standalone tools documentation

# Frequently Asked Questions (FAQ)

Common questions and troubleshooting for MLArena.

**For more help, see:**
- **[Terminology Guide](TERMINOLOGY.md)** - Naming conventions
- **[MLA Workflow Guide](MLA_WORKFLOW_GUIDE.md)** - Complete examples
- **[Submission Queue](submission_queue.md)** - Queue management

---

## Getting Started

### How do I install MLArena?

```bash
# Clone repository
git clone https://github.com/hipotures/mlarena.git
cd mlarena

# Install dependencies
uv sync

# Install Playwright for score fetching (optional)
uv run playwright install chromium
```

See [Quick Start Guide](quick_start.md) for details.

---

### What Python version is required?

Python 3.10+ is required. MLArena uses modern Python features including type hints and pattern matching.

---

### How do I set up Kaggle credentials?

1. Create API token at: https://www.kaggle.com/settings/account
2. Download `kaggle.json`
3. Move to `~/.kaggle/kaggle.json`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

---

## Naming Conventions

### How do I select a preprocessing template?

Use dotted overrides:
```bash
uv run python scripts/mla.py project=titanic preprocess_template=baseline
```

You can also chain multiple steps with a comma-separated list (e.g., `preprocess_template=imputer,scaler`).

---

### What's the difference between `mlarena`, `MLArena`, and `mla`?

- **MLArena**: Product name in prose and documentation
- **mlarena**: Python package name (`from mlarena.core import ...`)
- **mla**: CLI command name (`uv run python scripts/mla.py`)

**See:** [Terminology Guide - Product Names](TERMINOLOGY.md#product-and-package-names)

---

## Execution Issues

### "Module already completed" - how do I re-run?

Use `force=true` to re-execute completed modules:

```bash
# Force re-run single module
uv run python scripts/mla.py model project=titanic force=true

# Force re-run entire auto-flow
uv run python scripts/mla.py project=titanic force=true
```

**Note:** This overwrites existing results unless `lock=true` was used.

---

### "Kaggle API errors about credentials"

**Checklist:**
1. ✅ File exists: `~/.kaggle/kaggle.json`
2. ✅ Correct permissions: `chmod 600 ~/.kaggle/kaggle.json`
3. ✅ Valid JSON format
4. ✅ Kaggle CLI installed: `pip install kaggle`

**Test credentials:**
```bash
kaggle competitions list
```

---

### "No such preprocess/model template"

**List available templates:**
```bash
# List modules
uv run python scripts/mla.py modules

# Check specific template location
ls src/mlarena/templates/model/
ls src/mlarena/templates/preprocess/
```

**Common mistakes:**
- ❌ Using `.yaml` extension: `model_template=baseline.yaml`
- ✅ Correct format: `model_template=baseline`

---

### "Ambiguous preprocessing/model (project vs global)"

**Cause:** Same filename exists in both:
- `src/mlarena/defaults/preprocessing/encoder.py`
- `projects/kaggle/titanic/code/preprocessing/encoder.py`

**Fix:** Rename or remove one file to resolve ambiguity.

---

### State shows "running" after crash

**Fix:** Re-run the same module - stale "running" status is automatically marked as failed before execution.

```bash
uv run python scripts/mla.py model project=titanic experiment_id=exp-20251217-152730
```

---

## Templates and Configuration

### How do I create a custom model template?

1. Create YAML file in `projects/kaggle/<slug>/templates/model/my_template.yaml`
2. Define model and config:
   ```yaml
   model: autogluon_baseline
   config:
     time_limit: 600
     preset: medium_quality
   ```
3. Use template:
   ```bash
   uv run python scripts/mla.py model project=<slug> model_template=my_template
   ```

**See:** [MLA Workflow Guide - Custom Templates](MLA_WORKFLOW_GUIDE.md#advanced-topics)

---

### What's the difference between `cpu-fast-1m` and `cpu-dev-5m`?

Template naming pattern: `{hardware}-{quality}-{time}`

- `cpu-fast-1m`: CPU, fast quality, 1 minute time limit
- `cpu-dev-5m`: CPU, development quality, 5 minutes
- `gpu-best-8h`: GPU, best quality, 8 hours

**See:** [Terminology Guide - Template Naming](TERMINOLOGY.md#template-naming)

---

### How do I override configuration parameters?

Use dotted paths for any parameter:

```bash
# Override model time limit
uv run python scripts/mla.py model project=titanic model.time_limit=3600

# Override multiple parameters
uv run python scripts/mla.py project=titanic \
  common.seed=42 \
  common.time_limit=600 \
  model.preset=best_quality
```

**See:** [Configuration System](configs.md)

---

## Preprocessing

### How do I add a custom preprocessing step?

1. Create file: `projects/kaggle/<slug>/code/preprocessing/my_step.py`
2. Implement `fit_transform(train, val, test, config, orig_df=None)`:
   ```python
   def fit_transform(train_df, val_df, test_df, config, orig_df=None):
       # Your preprocessing logic
       return train_df, val_df, test_df, orig_df, state_dict
   ```
3. Create template: `templates/preprocess/my_step.yaml`
4. Use in chain: `preprocess_template=my_step`

**See:** [Preprocessing Submodules Guide](submodules/README.md)

---

### Can I run just preprocessing chains?

Yes! Run preprocessing standalone:

```bash
# Single template
uv run python scripts/mla.py preprocess project=titanic preprocess_template=baseline

# Chain (comma-separated)
uv run python scripts/mla.py preprocess project=titanic preprocess_template=sanity_check,imputer,scaler
```

Completed steps are cached unless `force=true` is used.

**See:** [MLA Workflow Guide - Manual Workflow](MLA_WORKFLOW_GUIDE.md#manual-workflow-example)

---

### Where are preprocessed files saved?

**Single-step:**
```
experiments/pre-{template}/artifacts/preprocess/
├── train_processed.csv.gz
├── test_processed.csv.gz
└── submodules/
    └── {submodule}/
        ├── summary.json
        └── fitted_objects.pkl
```

**Chain:**
```
experiments/pre-{template}/{hash}/{step}-{name}/artifacts/preprocess/
```

**See:** [State Payload Formats](state_payload_formats.md#preprocessing-payload-variations)

---

## Submission and Scoring

### What's the difference between submission queue and task queue?

**Two separate systems:**

| Feature | Submission Queue | Task Queue |
|:--------|:----------------|:-----------|
| Purpose | Upload to Kaggle | Run experiments |
| Command | `submission_queue.py` | `mla queue` |
| Scope | Submit/fetch only | Full pipeline |

**See:** [Submission Queue Guide](submission_queue.md#submission-queue-vs-task-queue)

---

### How do I queue multiple submissions?

```bash
# During experiments - queue instead of upload
uv run python scripts/mla.py submit project=titanic experiment_id=exp1 submit.queue_submit=true

# Later - batch upload
python scripts/submission_queue.py --project titanic submit 1 --continue-flow
```

**See:** [Submission Queue Guide](submission_queue.md)

---

### Fetch-score fails or hangs

**Setup Chrome with remote debugging:**

```bash
# Start Chrome
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug

# Log into Kaggle in that window
# Keep window open

# Run fetch-score
uv run python scripts/mla.py fetch-score project=titanic
```

**For non-default port:**
```bash
uv run python scripts/mla.py fetch-score project=titanic init.cdp_url=http://localhost:9223
```

---

### Why is my submission rejected for bad format?

**Validation checks:**
- ✅ Column names match `sample_submission.csv` exactly
- ✅ Row count matches test.csv
- ✅ No missing values in prediction column

**Debug:**
```python
import pandas as pd

# Check your submission
sub = pd.read_csv("submission.csv")
sample = pd.read_csv("data/sample_submission.csv")

print("Columns match:", set(sub.columns) == set(sample.columns))
print("Row count match:", len(sub) == len(sample))
print("Missing values:", sub.isnull().sum())
```

---

## Experiment Management

### How do I rerun predict/submit on existing experiment?

Pass `experiment_id=` to reuse artifacts:

```bash
# Re-run predict
uv run python scripts/mla.py predict project=titanic experiment_id=exp-20251217-152730

# Re-run submit
uv run python scripts/mla.py submit project=titanic experiment_id=exp-20251217-152730
```

Pipeline automatically reuses model artifacts and skips dependencies.

---

### How do I protect experiments from accidental overwrite?

Use `lock=true` to create `overwrite.lock`:

```bash
uv run python scripts/mla.py model project=titanic lock=true
```

**To delete locked experiment:**
```bash
rm experiments/exp-20251217-152730/overwrite.lock
```

**See:** [MLA Workflow Guide - Lock Files](MLA_WORKFLOW_GUIDE.md#experiment-locking)

---

### Where are experiment results stored?

```
projects/kaggle/<slug>/experiments/
├── init/                    # Project initialization
├── eda/                     # EDA reports
├── pre-{template}/          # Preprocessing
└── exp-{timestamp}/         # Model experiments
    ├── state.json          # Execution status
    └── artifacts/
        ├── model/          # Trained models
        ├── predict/        # Predictions
        └── preprocess/     # Preprocessed data
```

**See:** [Architecture - Experiment Structure](architecture.md#experiment-state-snapshot)

---

## Git and Reproducibility

### Git commit fails at end of auto-flow

**Cause:** Nothing to commit or git hooks blocking

**Fix:**
```bash
# Skip auto-commit
uv run python scripts/mla.py project=titanic skip_git=true

# Or commit manually
git add projects/kaggle/titanic
git commit -m "your message"
```

**See:** [README - Auto-Flow Git Commits](../README.md#auto-flow-git-commits)

---

### What does auto-flow commit message mean?

**Format:**
```
auto-flow(project): module1→module2→... | local CV_SCORE | public PUB_SCORE
```

**Example:**
```
auto-flow(titanic): preprocess→model→predict→submit→fetch-score | local 0.834 | public 0.798
```

Shows complete pipeline execution with validation and public scores.

**See:** [README - Auto-Flow Git Commits](../README.md#auto-flow-git-commits)

---

## Performance and Optimization

### How do I speed up AutoGluon training?

**Use profiles:**
```bash
# Smoke test (fast, low quality)
uv run python scripts/mla.py project=titanic profile=smoke

# Development (medium quality, 5 minutes)
uv run python scripts/mla.py project=titanic profile=dev
```

**Or override time limit:**
```bash
uv run python scripts/mla.py model project=titanic common.time_limit=60
```

**See:** [Configuration System - Profiles](configs.md#built-in-profile-fallbacks)

---

### How do I use HPO (hyperparameter optimization)?

```bash
# Quick HPO (50 trials, 1-2h)
uv run python scripts/mla.py model project=titanic model_template=hpo_boost_medium

# Advanced HPO (100 trials, 4-6h)
uv run python scripts/mla.py model project=titanic model_template=hpo_boost_high

# Full HPO (200 trials, 8-12h)
uv run python scripts/mla.py model project=titanic model_template=hpo_boost_best
```

**See:** [MLA Workflow Guide - HPO](MLA_WORKFLOW_GUIDE.md#hyperparameter-optimization-hpo)

---

### How do I save disk space with AutoGluon?

Use `mla_retention=true` to delete intermediate models:

```bash
uv run python scripts/mla.py model project=titanic model.mla_retention=true
```

Keeps only the best ensemble, removes individual models.

**See:** [Configuration System - Parameters](configs.md#complete-parameter-reference)

---

## Advanced Topics

### How do I understand state.json format?

**See comprehensive guide:** [State Payload Formats](state_payload_formats.md)

Covers:
- Single-step vs chain preprocessing
- Custom module state (weights, eval data)
- Model payload structure
- Backward compatibility

---

### How do I use adversarial validation?

1. Add to preprocessing chain:
   ```yaml
   # templates/preprocess/with_av.yaml
   chain: [sanity_check, adversarial_validation, imputer, encoder]
   ```

2. Run preprocessing:
   ```bash
   uv run python scripts/mla.py preprocess project=titanic preprocess_template=with_av
   ```

3. Model automatically uses sample weights from `custom_module_state`

**See:** [State Payload Formats - AV Weights](state_payload_formats.md#adversarial-validation-weights)

---

### Can I run multiple experiments in parallel?

**No** - MLArena uses file locking to prevent conflicts.

**Use Task Queue instead:**
```bash
# Queue multiple experiments
uv run python scripts/mla.py queue project=titanic add model_template=cpu-fast-1m --priority 1
uv run python scripts/mla.py queue project=titanic add model_template=cpu-dev-5m --priority 2

# Run sequentially
uv run python scripts/mla.py queue project=titanic run
```

**See:** [README - Task Queue Management](../README.md#task-queue-management)

---

## Troubleshooting

### ModuleNotFoundError when importing mlarena

**Fix:** Install in development mode:
```bash
uv sync
```

Or add to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

---

### Playwright browser not found

**Fix:** Install Chromium:
```bash
uv run playwright install chromium
```

---

### "Template not found" despite file existing

**Checklist:**
- ✅ File in correct location (`src/mlarena/templates/` or `projects/kaggle/<slug>/templates/`)
- ✅ Correct naming (no `.yaml` in template name)
- ✅ Valid YAML syntax

**Debug:**
```bash
# Check template exists
cat src/mlarena/templates/model/baseline.yaml

# Verify YAML is valid
python -c "import yaml; yaml.safe_load(open('src/mlarena/templates/model/baseline.yaml'))"
```

---

### Import errors in custom preprocessing

**Cause:** Preprocessing modules loaded dynamically need imports in sys.path

**Fix:** Use relative imports or ensure paths are correct:
```python
# In projects/kaggle/titanic/code/preprocessing/my_module.py

# Import from code/utils (works automatically)
from config import TARGET_COLUMN

# Import from other preprocessing module
import encoder_utils  # Must be in code/preprocessing/ or defaults/preprocessing/
```

---

## Still Need Help?

1. **Check documentation:**
   - [Documentation Index](../README.md#documentation-index)
   - [AGENTS.md](../AGENTS.md) - AI agent guide

2. **Report issues:**
   - GitHub: https://github.com/hipotures/mlarena/issues

3. **Review logs:**
   ```bash
   # Check experiment state
   cat projects/kaggle/titanic/experiments/exp-*/state.json | jq .

   # Check queue logs
   cat projects/kaggle/titanic/queue/logs/task-*.log
   ```

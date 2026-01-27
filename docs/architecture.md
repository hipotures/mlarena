# Architecture Overview

MLArena is organized into four layers: a thin CLI wrapper, a core orchestrator, project-specific code and templates, and a tracking layer that persists state and submissions.

```
                +----------------------+
                |   scripts/mla.py     |
                +----------+-----------+
                           |
                           v
      +--------------------+---------------------+
      |   Core (registry, pipeline, experiment)  |
      +--------------------+---------------------+
                           |
                           v
      +--------------------+---------------------+
      | Project layer (projects/kaggle/<slug>)   |
      +--------------------+---------------------+
                           |
                           v
      +--------------------+---------------------+
      | Tracking (experiments/, submissions/)    |
      +------------------------------------------+
```

## Layer breakdown

- **Scripts**: `scripts/mla.py` wires the repository root into `sys.path` and calls `mlarena.cli.main:main`.
- **Core**: Registry, pipeline executor, experiment state, console display utilities, and CLI parsing live in `src/mlarena/core/` and `src/mlarena/cli/`.
- **Project**: Competition-specific assets under `projects/kaggle/<slug>/` — data, custom preprocessing/model code, and project-local templates.
- **Tracking**: `experiments/<id>/state.json` keeps module status/payloads with a git hash snapshot, while `submissions/submissions.json` tracks leaderboard entries.

## Execution flow

1. **Registry discovery** (`ModuleRegistry.discover`) loads modules from `src/mlarena/modules/` and project overrides.
2. **CLI parsing** builds dynamic subcommands for each registered module plus the `modules` listing helper.
3. **Pipeline orchestration** (`PipelineExecutor`) topo-sorts dependencies, prints standardized headers/footers, and updates state with file locking.
4. **State handling** (`ExperimentState`) merges fixed setup runs (`init`, `eda`) into subsequent experiments and marks interrupted runs as failed on restart.
5. **Auto-flow** (`run_auto_flow`) executes the sequence `preprocess (chains) → model → predict → submit → fetch-score`. Note: `init` and `eda` must be completed manually beforehand as prerequisites; the auto-flow validates their completion before starting. Successful runs optionally create a git commit summarizing local CV and public score.

## Templates and resolution

- **Precedence**: project templates in `projects/kaggle/<slug>/templates/{model|preprocess}` override global templates in `src/mlarena/templates/`.
- **Preprocess meta-templates**: YAML files may define a `chain` key to express sequential preprocessing steps. Chains are executed step-by-step with cached outputs when unchanged.
- **Model templates**: Map a `model` implementation (project or default) and config, optionally referencing a preprocessing template via `preprocess_template`.

## Template Resolution Details

### Precedence Order (highest to lowest):
1. CLI dotted overrides (`key=value`)
2. Template config (within YAML)
3. Project config (`projects/kaggle/<slug>/config.yaml`)
4. Profile (`templates/profiles/<name>.yaml`)
5. Hardcoded defaults (`src/mlarena/core/conf.py`)

### Search Order for Templates:
1. Project-local: `projects/kaggle/<slug>/templates/{model|preprocess}/<name>.yaml`
2. Global: `src/mlarena/templates/{model|preprocess}/<name>.yaml`

### Chain Resolution Algorithm:
1. Parse template argument (single name or comma-separated list)
2. Load template config from precedence order
3. Check for `chain` key in config
4. If `chain` exists: expand to list of template names
5. If comma-separated: use as-is
6. For each template in list:
   - Load individual config
   - Compute semantic hash
7. Combine hashes to generate chain experiment ID
8. Create directory: `pre-{chain_id}/{combined_hash}/{idx}-{template}`

## Experiment state snapshot

`state.json` tracks module status, payload, invocation parameters, git metadata, and artifacts such as processed datasets and submission files. The file is guarded by a lock to avoid concurrent writes and is reused when resuming modules with `experiment_id=...`.

## Distributed Computing Architecture (Optional)

For users running heavy computations on remote servers, mlarena supports a split environment architecture with automatic path compatibility:

### Environment Setup

- **Local Dev Server** (`~/ml/kaggle`): Development workspace
  - Edit code, templates, and scripts here
  - Manage task queue and git commits
  - Run experiments locally when needed

- **Remote Computing Server**: Execution environment for heavy workloads
  - Physical location: `/home/xai/ml/mlarena`
  - Symlink: `~/ml/kaggle` → `/home/xai/ml/mlarena` (for path compatibility)
  - Heavy model training runs here
  - Exports `/home/xai/ml/mlarena` via NFS

- **NFS Mount** (`/mnt/mlarena` on local): Remote filesystem access
  - Mounts remote `/home/xai/ml/mlarena`
  - View experiment results locally without SSH
  - Access state.json and artifacts directly

### Path Compatibility

The symlink `~/ml/kaggle` on remote ensures that paths work identically on both machines:

```bash
# Local development
cd ~/ml/kaggle/projects/kaggle/<slug>
vim code/models/my_model.py

# Remote execution (same path!)
cd ~/ml/kaggle/projects/kaggle/<slug>
uv run python scripts/mla.py project=<slug>
```

### Synchronization

Code is synchronized from local to remote, but **configuration stays local**:

```bash
# Sync code changes (scripts/, src/, project code/)
rsync -av ~/ml/kaggle/ remote:~/ml/kaggle/

# Configuration files are NOT synced automatically
# This allows different settings per environment
```

### Viewing Results

Access remote experiment results via NFS mount:

```bash
# View remote experiments locally
cat /mnt/mlarena/projects/kaggle/<slug>/experiments/<id>/state.json

# Or use the compatible path (if remote symlink exists)
cat ~/ml/kaggle/projects/kaggle/<slug>/experiments/<id>/state.json
```

**Note**: This architecture is optional. Single-machine users can ignore this section.

# Repository Guidelines

## Project Structure & Module Organization
Competition workspaces sit in directories such as `playground-series-s5e11/` or `melting-point/`. Within each, raw Kaggle files belong to `data/`, runnable code to `code/` (split into `exploration/`, `models/`, `utils/`), generated files to `submissions/`, experiment metadata to `experiments/`, and notes to `docs/`. Shared tooling for experiment tracking and Kaggle automation lives in `tools/`, while dependency metadata stays in `pyproject.toml` plus `uv.lock`; do not commit bulky outputs outside these locations.

## Build, Test, and Development Commands
`uv sync` installs the Python 3.12 environment defined for the entire repo. Re-run experiments with `uv run python playground-series-s5e11/code/models/baseline.py` (adjust the path per project) to rebuild submissions deterministically. Use `uv run python tools/submissions_tracker.py --project playground-series-s5e11 list` to audit historical scores and `uv run python tools/experiment_logger.py --project playground-series-s5e11 list --limit 10` to inspect experiment metadata. Pull fresh data via `kaggle competitions download -c <competition>` from the relevant project directory; the CLI reads `~/.kaggle/kaggle.json` outside the repo.

## Coding Style & Naming Conventions
Use 4-space indentation, `snake_case` for modules/functions, and `CamelCase` for classes. Zero-pad exploration scripts (`01_initial_eda.py`) so chronological ordering is preserved. Name outputs `submission-YYYYMMDDHHMM-model.csv` for automatic ingestion by `submissions_tracker.py`, and centralize constants plus random seeds inside `code/utils/config.py`. Keep logging helpers and datapath utilities in `code/utils` so every project script shares the same conventions.

## Testing Guidelines
Every model script must expose a quick `main()` call that runs on a sample or stratified subset before launching the full training job. Use smoke tests under `WORKSPACE/scripts/*test*.py` (for example, `uv run python playground-series-s5e7/WORKSPACE/scripts/20250703_2320_test_lightgbm_simple.py`) to spot regressions in preprocessing logic. Capture CV metrics and important parameters in `experiments/*.params`, and block merges when newly reported metrics deviate by more than ±0.002 RMSE or the leaderboard trend in `submissions/submissions.json` declines.

## Commit & Pull Request Guidelines
Follow the lightweight conventional style noted in `README.md`: prefix messages with `feat:`, `fix:`, or `experiment:` and describe the observable change ("feat: add Autogluon medium baseline"). Each PR should link the tracked Kaggle issue or discussion, include reproduction commands, mention the relevant tracker entry, and attach leaderboard evidence for any new submission. Never include dataset files, and refresh `uv.lock` whenever dependencies change.

## Security & Configuration Tips
Keep `~/.kaggle/kaggle.json` local and chmod 600; reference it from scripts via environment variables, not hard-coded paths. Document any sensitive configs in project-level READMEs, scrub notebooks before committing, and share only the minimum per-competition folder when collaborating with external agents or runners.

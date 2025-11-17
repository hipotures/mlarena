# Repository Guidelines

## Project Structure & Module Organization
Competition workspaces sit in directories such as `playground-series-s5e11/` or `melting-point/`. Within each, raw Kaggle files belong to `data/`, runnable code to `code/` (split into `exploration/`, `models/`, `utils/`), generated files to `submissions/`, experiment metadata to `experiments/`, and notes to `docs/`. Shared tooling for experiment tracking and Kaggle automation lives in `tools/`, while dependency metadata stays in `pyproject.toml` plus `uv.lock`; do not commit bulky outputs outside these locations.

## Build, Test, and Development Commands
`uv sync` installs the Python 3.12 environment defined for the entire repo. Re-run experiments with the shared runner (`uv run python tools/autogluon_runner.py --project playground-series-s5e11 --template dev-gpu`) or, equivalently, via the thin wrappers in each project (`uv run python playground-series-s5e11/code/models/baseline_autogluon.py`). Templates replace raw numeric flags:

| Template     | Time limit | Preset           | GPU | Notes |
|--------------|-----------:|------------------|-----|-------|
| `fast-cpu`   | 60 s       | `medium_quality` | ❌  | XGBoost only, sanity checks |
| `dev-cpu`    | 300 s      | `medium_quality` | ❌  | default stack |
| `dev-gpu`    | 300 s      | `medium_quality` | ✅  | default stack |
| `best-cpu`   | 3600 s     | `best_quality`   | ❌  | high-quality ensemble |
| `best-gpu`   | 3600 s     | `best_quality`   | ✅  | high-quality ensemble |
| `extreme-gpu`| 24 h       | `extreme_quality`| ✅  | ≤30k rows, prompts if higher |

Overrides (`--time-limit`, `--preset`, `--use-gpu`) remain available when a template needs tweaking. Reach for `fast-cpu` when chcesz 60-sekundowy smoke test XGBoost-em – nie traktuj go jako finalnego treningu. Use `uv run python tools/submissions_tracker.py --project playground-series-s5e11 list` to audit historical scores and `uv run python tools/experiment_logger.py --project playground-series-s5e11 list --limit 10` to inspect experiment metadata. Pull fresh data via `kaggle competitions download -c <competition>` from the relevant project directory; the CLI reads `~/.kaggle/kaggle.json` outside the repo.

### Experiment Workflow

`experiments/<experiment_id>.json` trzyma stan modułów. Standardowa ścieżka:

1. **EDA:** `uv run python tools/experiment_manager.py eda --project playground-series-s5e11` (wydrukuje ID, np. `exp-20251117-011230`).
2. **Model:** `uv run python tools/autogluon_runner.py --project playground-series-s5e11 --experiment-id exp-20251117-011230 --template dev-gpu ...`
3. **Submit/Resume:** `uv run python tools/submission_workflow.py resume --project playground-series-s5e11 --filename submission-YYYYMMDDHHMMSS.csv --experiment-id exp-20251117-011230 --cdp-url http://127.0.0.1:9222`

Każdy moduł dopisuje swoją sekcję (status, artefakty, wyniki). `uv run python tools/experiment_manager.py list --project playground-series-s5e11` wypisze stan wszystkich eksperymentów. Dzięki temu możesz odpalać moduły osobno, wznawiać pipeline od dowolnego kroku lub od razu zobaczyć, które submission odpowiada któremu kodowi.

Spis kroków znajdziesz pod `uv run python tools/experiment_manager.py modules`.

Submission automation is built into the runner: append `--auto-submit --wait-seconds 45` to push the CSV to Kaggle, wait for scoring, scrape the latest score via Playwright, update the tracker, and commit the code/state (omit `--auto-submit` to keep the interactive “Submit? [y/N]” prompt, or pass `--skip-submit` to opt out entirely). Customize the Kaggle description with `--kaggle-message "autogluon medium exp-2"` and tweak the CDP endpoint (`--cdp-url http://localhost:9222`) when connecting to an existing Chrome session.

## Coding Style & Naming Conventions
Use 4-space indentation, `snake_case` for modules/functions, and `CamelCase` for classes. Zero-pad exploration scripts (`01_initial_eda.py`) so chronological ordering is preserved. Name outputs `submission-YYYYMMDDHHMM-model.csv` for automatic ingestion by `submissions_tracker.py`, and centralize constants plus random seeds inside `code/utils/config.py`. Keep logging helpers and datapath utilities in `code/utils` so every project script shares the same conventions.

## Testing Guidelines
Rapid validation now relies on the shared templates: run `uv run python tools/experiment_manager.py model --project <proj> --template fast-cpu --skip-submit` to exercise the full pipeline in ~60 s (XGBoost only) before committing heavier compute. For longer jobs, stick to `dev-*` or `best-*` templates and compare local CV in the experiment’s `state.json`. Block merges when newly reported metrics deviate by more than ±0.002 ROC-AUC/RMSE or when the leaderboard trend stored in `submissions/submissions.json` regresses. All preprocessing/modeling scripts should remain deterministic (respect the seeds defined in each `config.py`).

## Commit & Pull Request Guidelines
Follow the lightweight conventional style noted in `README.md`: prefix messages with `feat:`, `fix:`, or `experiment:` and describe the observable change ("feat: add Autogluon medium baseline"). Each PR should link the tracked Kaggle issue or discussion, include reproduction commands, mention the relevant tracker entry, and attach leaderboard evidence for any new submission. Never include dataset files, and refresh `uv.lock` whenever dependencies change.

## Security & Configuration Tips
Keep `~/.kaggle/kaggle.json` local and chmod 600; reference it from scripts via environment variables, not hard-coded paths. Document any sensitive configs in project-level READMEs, scrub notebooks before committing, and share only the minimum per-competition folder when collaborating with external agents or runners.

## Submission Automation Workflow
`tools/submission_workflow.py` orchestrates Kaggle submissions: it runs the CLI upload, waits (`--wait-seconds`, default 30s), connects to an already running Chrome (`google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug`) via Playwright, reads the latest score from `/submissions`, updates `submissions/submissions.json`, and creates a git commit like `submission(playground-series-s5e11): autogluon-medium | local 0.92379 | public 0.92227`. Install Playwright with `uv run playwright install chromium` before first use. Pass `--skip-score-fetch` or `--skip-git` to the model scripts if the browser isn't available or you want to review changes manually. The runner only stages the active competition directory, so adjust manually if other folders changed.

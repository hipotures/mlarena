# Quick Start

This guide gets you from a fresh checkout to a full MLArena run in a few commands.

## Prerequisites

- Python 3.10+ with [uv](https://github.com/astral-sh/uv) available (`pip install uv` if missing).
- Kaggle API credentials in `~/.kaggle/kaggle.json` (`chmod 600` on the file).
- Chromium installed for score scraping: `uv run playwright install chromium` (one-time).

## Install dependencies

```bash
uv sync
```

If you plan to use automated score fetching, install the Playwright browser helper:

```bash
uv run playwright install chromium
```

## Initialize a project

```bash
# Downloads competition data and scaffolds projects/kaggle/<slug>
uv run python scripts/mla.py init --project <competition-slug> --competition <kaggle-id>
```

The `--competition` flag defaults to the project name; override it when the Kaggle slug differs.

## Run the full pipeline

```bash
# Auto-flow: init → eda → preprocess → model → predict → submit → fetch-score
uv run python scripts/mla.py --project <competition-slug>
```

Useful flags:

- `--model-template <name>`: pick a model template (default: `baseline`).
- `--preprocess-template <name>`: override preprocessing chain (defaults to the model template’s choice).
- `--smoke` or `--dev`: fast presets for quick iterations.
- `--skip-submit`: build the submission file without uploading to Kaggle.
- `--force`: re-run completed modules.

Example (Titanic baseline with cached setup):

```bash
uv run python scripts/mla.py -p Titanic --model-template cpu-dev-5m --skip-submit
```

Recommended end-to-end check on Titanic (no smoke mode needed):

```bash
uv run python scripts/mla.py -p Titanic
```

## Run individual modules

```bash
uv run python scripts/mla.py preprocess --project <slug> --preprocess-template baseline
uv run python scripts/mla.py model --project <slug> --model-template cpu-fast-1m --skip-submit
```

List available modules:

```bash
uv run python scripts/mla.py modules
```

## Verify score fetching

Start Chrome with remote debugging before using `fetch-score`:

```bash
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
# Log into Kaggle in that window and keep it open.
```

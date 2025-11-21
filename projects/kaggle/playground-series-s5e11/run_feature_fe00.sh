#!/usr/bin/env bash
# Run the baseline EDA feature set (fe00) with 1h CPU best_quality, auto-submit.

set -euo pipefail

PROJECT="playground-series-s5e11"
WAIT_SECONDS="${WAIT_SECONDS:-45}"
CDP_URL="${CDP_URL:-http://localhost:9222}"

echo ">>> Running best-cpu-fe00 (wait=${WAIT_SECONDS}s, CDP=${CDP_URL})"
uv run python scripts/experiment_manager.py model \
  --project "${PROJECT}" \
  --template "best-cpu-fe00" \
  --auto-submit \
  --wait-seconds "${WAIT_SECONDS}" \
  --cdp-url "${CDP_URL}"

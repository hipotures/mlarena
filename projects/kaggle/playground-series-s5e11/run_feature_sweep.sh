#!/usr/bin/env bash
# Sequentially run all 10 feature variants with 1h CPU (best_quality) and auto-submit.

set -euo pipefail

PROJECT="playground-series-s5e11"
WAIT_SECONDS="${WAIT_SECONDS:-45}"
CDP_URL="${CDP_URL:-http://localhost:9222}"

TEMPLATES=(
  best-cpu-fe01
  best-cpu-fe02
  best-cpu-fe03
  best-cpu-fe04
  best-cpu-fe05
  best-cpu-fe06
  best-cpu-fe07
  best-cpu-fe08
  best-cpu-fe09
  best-cpu-fe10
)

for tmpl in "${TEMPLATES[@]}"; do
  echo ""
  echo ">>> Running ${tmpl} (wait=${WAIT_SECONDS}s, CDP=${CDP_URL})"
  uv run python scripts/experiment_manager.py model \
    --project "${PROJECT}" \
    --template "${tmpl}" \
    --auto-submit \
    --wait-seconds "${WAIT_SECONDS}" \
    --cdp-url "${CDP_URL}"
done

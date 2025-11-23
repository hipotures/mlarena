#!/usr/bin/env bash
# Sequentially run the rich baseline and all 10 new feature variants (11-20).
# Uses 1h CPU (best_quality) and auto-submits.

set -euo pipefail

PROJECT="playground-series-s5e11"
WAIT_SECONDS="${WAIT_SECONDS:-45}"
CDP_URL="${CDP_URL:-http://localhost:9222}"

# First, run the rich baseline experiment
#echo ">>> Running rich baseline (wait=${WAIT_SECONDS}s, CDP=${CDP_URL})"
#uv run python scripts/experiment_manager.py model \
  #--project "${PROJECT}" \
  #--template "best-cpu-rich-baseline" \
  #--auto-submit \
  #--wait-seconds "${WAIT_SECONDS}" \
  #--cdp-url "${CDP_URL}"

# Next, run the new feature variants from 11 to 20
TEMPLATES=(
#  best-cpu-fe11
#  best-cpu-fe12
#  best-cpu-fe13
#  best-cpu-fe14
#  best-cpu-fe15
#  best-cpu-fe16
#  best-cpu-fe17
#  best-cpu-fe18
#  best-cpu-fe19
  best-cpu-fe20
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

echo ""
echo "All 11 experiments (rich baseline + 10 variants) have been launched."

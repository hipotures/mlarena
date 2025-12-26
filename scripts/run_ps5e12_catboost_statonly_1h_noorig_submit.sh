#!/usr/bin/env bash
set -euo pipefail

PROJECT=playground-series-s5e12
EXPS=(
  exp-20251224-160559  # statonly-linear-v2 (best local CV)
  exp-20251224-213642  # statonly-uncorr
  exp-20251224-203542  # statonly-mi90
  exp-20251224-190645  # statonly-clean-novar
  exp-20251224-180626  # statonly-strict
)

# Submit each experiment (no confirmation prompt), sleep 5s between submissions
for exp_id in "${EXPS[@]}"; do
  uv run python scripts/mla.py submit -p "$PROJECT" -e "$exp_id" \
    submit.confirm_timeout=0 \
    skip_submit=false
  sleep 5
done

# Wait for Kaggle to process submissions
sleep 60

# Fetch scores for the same experiments
for exp_id in "${EXPS[@]}"; do
  uv run python scripts/mla.py fetch-score -p "$PROJECT" -e "$exp_id" --force
done

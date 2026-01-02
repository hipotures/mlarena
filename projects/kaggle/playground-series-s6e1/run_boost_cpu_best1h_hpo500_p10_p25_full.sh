#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-playground-series-s6e1}"
SCRIPT="uv run python scripts/mla.py"

PCTS=(10 25)

for pct in "${PCTS[@]}"; do
  PP_TEMPLATE="train_p${pct}"
  MODEL_TEMPLATE="boost_cpu_best1h_hpo500_p${pct}"

  echo "==> Full flow: ${MODEL_TEMPLATE} (train ${pct}%)"
  ${SCRIPT} \
    --project "${PROJECT}" \
    --model-template "${MODEL_TEMPLATE}" \
    --preprocess-template "${PP_TEMPLATE}"
done

echo "All runs completed."

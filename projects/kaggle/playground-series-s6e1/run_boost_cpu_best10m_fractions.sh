#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-playground-series-s6e1}"
SCRIPT="uv run python scripts/mla.py"

PCTS=(10 20 30 40 50 60 70 80 90 100)

for pct in "${PCTS[@]}"; do
  PP_TEMPLATE="train_p${pct}"
  MODEL_TEMPLATE="boost_cpu_best10m_p${pct}"
  EXP_ID="exp-boost-cpu-best10m-p${pct}"

  echo "==> Preprocess: ${PP_TEMPLATE} (${pct}%)"
  ${SCRIPT} preprocess \
    --project "${PROJECT}" \
    --preprocess-template "${PP_TEMPLATE}"

  echo "==> Model: ${MODEL_TEMPLATE} (${pct}%)"
  ${SCRIPT} model \
    --project "${PROJECT}" \
    --model-template "${MODEL_TEMPLATE}" \
    --preprocess-template "${PP_TEMPLATE}" \
    --experiment-id "${EXP_ID}"
done

echo "All runs completed."

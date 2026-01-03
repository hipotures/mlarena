#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-playground-series-s6e1}"
SCRIPT="uv run python scripts/mla.py"

PP_TEMPLATE="train_p50"
MODEL_TEMPLATE="gbm_xgb_medq_fast10m_p50_seed42"
EXP_ID="exp-gbm-xgb-medq-fast10m-p50-s42"

echo "==> Preprocess: ${PP_TEMPLATE} (50%)"
${SCRIPT} preprocess \
  --project "${PROJECT}" \
  --preprocess-template "${PP_TEMPLATE}"

echo "==> Model: ${MODEL_TEMPLATE} (50%)"
${SCRIPT} model \
  --project "${PROJECT}" \
  --model-template "${MODEL_TEMPLATE}" \
  --preprocess-template "${PP_TEMPLATE}" \
  --experiment-id "${EXP_ID}"

echo "Done."

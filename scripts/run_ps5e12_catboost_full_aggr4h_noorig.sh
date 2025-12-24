#!/usr/bin/env bash
set -euo pipefail

PROJECT=playground-series-s5e12
MODEL_TEMPLATE=catboost-binned-hpo-aggressive-4h-noorig
COMMON_ARGS=(common.use_gpu=true common.time_limit=14400 common.preset=best skip_submit=true)

PRE_TEMPLATES=(
  catboost-full-linear-v1
  catboost-full-binq20
  catboost-full-statonly
)

for tpl in "${PRE_TEMPLATES[@]}"; do
  uv run python scripts/mla.py -p "$PROJECT" \
    preprocess_template="$tpl" \
    model_template="$MODEL_TEMPLATE" \
    "${COMMON_ARGS[@]}"
done

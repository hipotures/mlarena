#!/usr/bin/env bash
set -euo pipefail

PROJECT=playground-series-s5e12
MODEL_TEMPLATE=catboost-binned-hpo
COMMON_ARGS=(common.use_gpu=true common.time_limit=600 common.preset=best skip_submit=true)

PRE_TEMPLATES=(
  catboost-full-clean-rfe-linear-v1-rfe50
  catboost-full-clean-linear-v1-mi90
  catboost-full-clean-linear-v1-binq20
)

for tpl in "${PRE_TEMPLATES[@]}"; do
  uv run python scripts/mla.py project="$PROJECT" \
    preprocess_template="$tpl" \
    model_template="$MODEL_TEMPLATE" \
    "${COMMON_ARGS[@]}"
done

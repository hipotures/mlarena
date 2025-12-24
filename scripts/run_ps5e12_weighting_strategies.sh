#!/usr/bin/env bash
set -euo pipefail

PROJECT=playground-series-s5e12
MODEL_TEMPLATE=catboost-binned-hpo
COMMON_ARGS=(common.use_gpu=true common.time_limit=600 common.preset=best skip_submit=true)

PRE_TEMPLATES=(
  catboost-full-clean-rfe-percentile
  catboost-full-clean-rfe-tail
  catboost-full-clean-rfe-linear
)

for tpl in "${PRE_TEMPLATES[@]}"; do
  uv run python scripts/mla.py -p "$PROJECT" \
    preprocess_template="$tpl" \
    model_template="$MODEL_TEMPLATE" \
    "${COMMON_ARGS[@]}"
done

#!/usr/bin/env bash
set -euo pipefail

PROJECT=playground-series-s5e12
MODEL_TEMPLATE=catboost-binned-hpo-gap
COMMON_ARGS=(common.use_gpu=true common.time_limit=3600 common.preset=best skip_submit=true)

PRE_TEMPLATES=(
  catboost-full-clean-rfe-gap-p90
  catboost-full-clean-rfe-gap-p91
  catboost-full-clean-rfe-gap-p92
  catboost-full-clean-rfe-gap-p93
  catboost-full-clean-rfe-gap-p94
  catboost-full-clean-rfe-gap-p95
  catboost-full-clean-rfe-gap-p96
  catboost-full-clean-rfe-gap-p97
  catboost-full-clean-rfe-gap-p98
  catboost-full-clean-rfe-gap-p99
)

for tpl in "${PRE_TEMPLATES[@]}"; do
  uv run python scripts/mla.py project="$PROJECT" \
    preprocess_template="$tpl" \
    model_template="$MODEL_TEMPLATE" \
    "${COMMON_ARGS[@]}"
done

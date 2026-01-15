#!/usr/bin/env bash
set -euo pipefail

PROJECT=playground-series-s5e12
MODEL_TEMPLATE=catboost-binned-hpo-gap
COMMON_ARGS=(common.use_gpu=true common.time_limit=3600 common.preset=best skip_submit=true)

PRE_TEMPLATES=(
  catboost-full-clean-rfe-gap-p99-wm02
  catboost-full-clean-rfe-gap-p99-wm03
  catboost-full-clean-rfe-gap-p99-wm04
  catboost-full-clean-rfe-gap-p99-wm05
  catboost-full-clean-rfe-gap-p99-wm06
  catboost-full-clean-rfe-gap-p99-wm08
  catboost-full-clean-rfe-gap-p99-wm10
)

for tpl in "${PRE_TEMPLATES[@]}"; do
  uv run python scripts/mla.py project="$PROJECT" \
    preprocess_template="$tpl" \
    model_template="$MODEL_TEMPLATE" \
    "${COMMON_ARGS[@]}"
done

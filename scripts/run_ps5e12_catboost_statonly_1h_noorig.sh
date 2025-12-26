#!/usr/bin/env bash
set -euo pipefail

PROJECT=playground-series-s5e12
MODEL_TEMPLATE=catboost-binned-hpo-aggressive-4h-noorig
COMMON_ARGS=(common.use_gpu=true common.time_limit=3600 common.preset=best skip_submit=true)

PRE_TEMPLATES=(
  # catboost-full-statonly-linear-v2  # DONE: completed
  # catboost-full-statonly-tail       # DONE: completed
  # catboost-full-statonly-strict     # DONE: completed
  # catboost-full-statonly-clean-novar # DONE: completed
  catboost-full-statonly-mi90
  catboost-full-statonly-uncorr
)

for tpl in "${PRE_TEMPLATES[@]}"; do
  uv run python scripts/mla.py -p "$PROJECT" \
    preprocess_template="$tpl" \
    model_template="$MODEL_TEMPLATE" \
    "${COMMON_ARGS[@]}"
done

#!/usr/bin/env bash
set -euo pipefail

PROJECT=playground-series-s5e12
MODEL_TEMPLATE=catboost-binned-hpo
COMMON_ARGS=(common.use_gpu=true common.time_limit=600 common.preset=best skip_submit=true)

uv run python scripts/mla.py -p "$PROJECT" \
  preprocess_template=catboost-full-clean-rfe-linear-v1-rfe \
  model_template="$MODEL_TEMPLATE" \
  "${COMMON_ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/all-algos.sh <project> [preset] [time_limit_seconds]

Runs all single-model CPU templates sequentially for the given project.
- project (required): Kaggle competition directory (e.g., playground-series-s5e12)
- preset (optional): AutoGluon preset override (e.g., best, medium)
- time_limit_seconds (optional): Time limit override applied to each template

By default submissions are NOT uploaded (--skip-submit); add your own submit step later.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

PROJECT="$1"
PRESET="${2:-}"
TIME_LIMIT="${3:-}"

TEMPLATES=(
  cpu-gbm-8h
  cpu-cat-8h
  cpu-xgb-8h
  cpu-rf-8h
  cpu-xt-8h
  cpu-knn-8h
  cpu-ebm-8h
)

for tpl in "${TEMPLATES[@]}"; do
  echo "=== Running template: ${tpl} ==="
  cmd=(python scripts/experiment_manager.py model --project "${PROJECT}" --template "${tpl}" --skip-submit)
  if [[ -n "${PRESET}" ]]; then
    cmd+=(--preset "${PRESET}")
  fi
  if [[ -n "${TIME_LIMIT}" ]]; then
    cmd+=(--time-limit "${TIME_LIMIT}")
  fi
  "${cmd[@]}"
done

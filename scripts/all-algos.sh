#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/all-algos.sh [--without-nn|--only-nn] [--use-gpu] <project> [preset] [time_limit_seconds]

Runs all single-model templates sequentially for the given project.
Modes:
  (default) all      - CPU stacks + NN (torch/fastai) at the end
  --without-nn       - Only CPU stacks (no NN)
  --only-nn          - Only NN templates

Flags:
  --use-gpu          - Allow GPU for templates (NN gets --use-gpu 1; others stay as configured)

Args:
  project            - Competition directory (e.g., playground-series-s5e12)
  preset             - AutoGluon preset override (e.g., best, medium)
  time_limit_seconds - Override time_limit for every template

Submissions are NOT uploaded (--skip-submit). Submit separately when ready.
EOF
}

MODE="all"        # all | without-nn | only-nn
GPU_ENABLED=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --without-nn) MODE="without-nn"; shift ;;
    --only-nn) MODE="only-nn"; shift ;;
    --use-gpu) GPU_ENABLED=true; shift ;;
    --help|-h) usage; exit 0 ;;
    --*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    *) break ;;
  esac
done

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

PROJECT="$1"
PRESET="${2:-}"
TIME_LIMIT="${3:-}"

BASE_TEMPLATES=(
  cpu-gbm-8h
  cpu-cat-8h
  cpu-xgb-8h
  cpu-rf-8h
  cpu-xt-8h
  cpu-knn-8h
  cpu-ebm-8h
)

NN_TEMPLATES=(
  gpu-torch-8h
  gpu-fastai-8h
)

case "${MODE}" in
  all)       TEMPLATES=("${BASE_TEMPLATES[@]}" "${NN_TEMPLATES[@]}") ;;
  without-nn)TEMPLATES=("${BASE_TEMPLATES[@]}") ;;
  only-nn)   TEMPLATES=("${NN_TEMPLATES[@]}") ;;
  *) echo "Invalid mode: ${MODE}" >&2; exit 1 ;;
esac

for tpl in "${TEMPLATES[@]}"; do
  echo "=== Running template: ${tpl} ==="
  cmd=(python scripts/experiment_manager.py model --project "${PROJECT}" --template "${tpl}" --skip-submit)
  if [[ -n "${PRESET}" ]]; then
    cmd+=(--preset "${PRESET}")
  fi
  if [[ -n "${TIME_LIMIT}" ]]; then
    cmd+=(--time-limit "${TIME_LIMIT}")
  fi

  if [[ "${GPU_ENABLED}" == true ]]; then
    # Force GPU for NN templates; leave CPU templates as-is.
    if [[ " ${NN_TEMPLATES[*]} " == *" ${tpl} "* ]]; then
      cmd+=(--use-gpu 1)
    fi
  else
    # Force CPU everywhere (even for gpu-* templates).
    cmd+=(--use-gpu 0)
  fi

  "${cmd[@]}"
done

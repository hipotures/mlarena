#!/usr/bin/env bash
# Test new medical-domain and diabetes-ratios preprocessing
# Based on best pipeline: ID#97 (Local 0.726, Public 0.703)

set -euo pipefail

PROJECT=playground-series-s5e12
MODEL=catboost-binned-hpo-aggressive-4h-noorig-cpu
COMMON_ARGS=(common.time_limit=14400 common.preset=best skip_submit=false)

# Baseline best pipeline (for comparison): external-diabetes → diabetes-binning → diabetes-orig-stats → diabetes-tail-weights-linear-v1

PRE_TEMPLATES=(
  medical-domain-test           # + medical-domain features
  diabetes-ratios-test          # + diabetes-ratios features
  medical-ratios-combined-test  # + both
)

for tpl in "${PRE_TEMPLATES[@]}"; do
  uv run python scripts/mla.py -p "$PROJECT" \
    preprocess_template="$tpl" \
    model_template="$MODEL" \
    "${COMMON_ARGS[@]}"
done

echo ""
echo "===== Experiments Complete ====="
echo "Baseline: #97 with Local 0.726, Public 0.703"
echo "Test 1: medical-domain-test (+ medical-domain)"
echo "Test 2: diabetes-ratios-test (+ diabetes-ratios)"
echo "Test 3: medical-ratios-combined-test (+ both)"
echo ""
echo "Check results: python scripts/submissions_tracker.py --project playground-series-s5e12 list"

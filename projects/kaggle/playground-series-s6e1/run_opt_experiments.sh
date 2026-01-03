#!/bin/bash
set -euo pipefail

PROJECT="playground-series-s6e1"
SCRIPT="uv run python scripts/mla.py"

#echo "=== Running Opt 1: Smooth Operator (Rare -> Log1p -> TargetSmooth) ==="
# Note: Submit is enabled by default (no skip_submit=true)
#${SCRIPT} --project "${PROJECT}" --model-template "opt_cpu_best_30m_1"

echo ""
echo "=== Running Opt 2: CatBoost Hybrid (Log1p -> CatBoost) ==="
${SCRIPT} --project "${PROJECT}" --model-template "opt_cpu_best_30m_2"

echo ""
echo "Optimization experiments completed."

#!/usr/bin/env bash
# Run optimized experiments: fe21 (ultimate), fe22 (target encoding), and final ensemble
# Expected improvement: 0.92356 → 0.927+ → 0.930+

set -euo pipefail

PROJECT="playground-series-s5e11"
WAIT_SECONDS="${WAIT_SECONDS:-45}"
CDP_URL="${CDP_URL:-http://localhost:9222}"

echo "=========================================="
echo "Starting Optimized Experiment Series"
echo "=========================================="
echo ""

# Step 1: Run fe21 - Ultimate feature combination (3 hours)
echo ">>> Step 1: Running fe21 (Ultimate Features)"
echo "    Combines best features from fe17, fe20, fe13, fe15"
echo "    Expected: 0.926+ public score"
echo ""
uv run python scripts/experiment_manager.py model \
  --project "${PROJECT}" \
  --template "best-cpu-fe21" \
  --auto-submit \
  --wait-seconds "${WAIT_SECONDS}" \
  --cdp-url "${CDP_URL}"

echo ""
echo "Waiting 60s before next experiment..."
sleep 60

# Step 2: Run fe22 - Target encoding with CV (3 hours)
echo ">>> Step 2: Running fe22 (Target Encoding)"
echo "    fe21 features + CV-protected target encoding"
echo "    Expected: 0.927+ public score"
echo ""
uv run python scripts/experiment_manager.py model \
  --project "${PROJECT}" \
  --template "best-cpu-fe22" \
  --auto-submit \
  --wait-seconds "${WAIT_SECONDS}" \
  --cdp-url "${CDP_URL}"

echo ""
echo "Waiting 60s before ensemble creation..."
sleep 60

# Step 3: Create final ensemble
echo ">>> Step 3: Creating Final Ensemble"
echo "    Weighted average of best models"
echo "    Expected: 0.930+ public score"
echo ""

# First, ensure we have the paths to the models
PROJECT_ROOT="/mnt/ml/kaggle-fork1/projects/kaggle/${PROJECT}"
TEST_DATA="${PROJECT_ROOT}/data/test.csv"
OUTPUT_DIR="${PROJECT_ROOT}/submissions"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ENSEMBLE_OUTPUT="${OUTPUT_DIR}/ensemble_${TIMESTAMP}.csv"

# Check if validation data exists (optional)
VAL_DATA=""
if [ -f "${PROJECT_ROOT}/data/validation.csv" ]; then
    VAL_DATA="--validation-data ${PROJECT_ROOT}/data/validation.csv"
fi

# Run ensemble creation
python /home/claude/ensemble_best_models.py \
  --project-root "${PROJECT_ROOT}" \
  --test-data "${TEST_DATA}" \
  ${VAL_DATA} \
  --output "${ENSEMBLE_OUTPUT}" \
  --analyze-diversity

echo ""
echo ">>> Submitting ensemble to Kaggle"
kaggle competitions submit \
  -c "playground-series-s5e11" \
  -f "${ENSEMBLE_OUTPUT}" \
  -m "Final ensemble: fe17+fe20+fe21+fe22+original"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - fe21 (Ultimate): Combines best features"
echo "  - fe22 (Target): Adds CV-protected encoding"
echo "  - Ensemble: Weighted combination of top models"
echo ""
echo "Check submissions at:"
echo "  ${OUTPUT_DIR}"
echo ""
echo "Monitor progress at:"
echo "  https://www.kaggle.com/competitions/playground-series-s5e11/leaderboard"

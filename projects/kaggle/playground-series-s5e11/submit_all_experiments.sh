#!/bin/bash
# Submit all trained experiments that don't have submissions yet
# Usage: ./submit_all_experiments.sh

set -e

PROJECT="playground-series-s5e11"
WAIT_TIME=45

echo "=========================================="
echo "SUBMIT ALL TRAINED EXPERIMENTS"
echo "Project: ${PROJECT}"
echo "=========================================="
echo ""

cd /mnt/ml/kaggle-fork1

# Find all experiment directories with completed models
EXPERIMENT_DIRS=$(find "projects/kaggle/${PROJECT}/experiments/" -maxdepth 1 -type d -name "exp-*" | sort)

if [ -z "$EXPERIMENT_DIRS" ]; then
    echo "No experiments found!"
    exit 1
fi

TOTAL=0
SUBMITTED=0
SKIPPED=0

for exp_dir in $EXPERIMENT_DIRS; do
    EXPERIMENT_ID=$(basename "$exp_dir")
    STATE_FILE="${exp_dir}/state.json"

    # Check if state.json exists
    if [ ! -f "$STATE_FILE" ]; then
        echo "⊘ ${EXPERIMENT_ID}: No state.json, skipping"
        ((SKIPPED++))
        continue
    fi

    # Check if model module completed
    MODEL_STATUS=$(grep -oP '"model":\s*\{[^}]*"status":\s*"\K[^"]+' "$STATE_FILE" 2>/dev/null || echo "not_found")

    if [ "$MODEL_STATUS" != "completed" ]; then
        echo "⊘ ${EXPERIMENT_ID}: Model not completed (status: ${MODEL_STATUS}), skipping"
        ((SKIPPED++))
        continue
    fi

    # Find the template used
    TEMPLATE=$(grep -oP '"template":\s*"\K[^"]+' "$STATE_FILE" | head -1)

    if [ -z "$TEMPLATE" ]; then
        echo "⊘ ${EXPERIMENT_ID}: Could not detect template, skipping"
        ((SKIPPED++))
        continue
    fi

    # Check if AutogluonModels directory exists
    MODEL_PATH="projects/kaggle/${PROJECT}/AutogluonModels"

    # Try to find model directory by template or experiment name
    POSSIBLE_MODEL_DIRS=(
        "${MODEL_PATH}/exp01_tier1_features"
        "${MODEL_PATH}/exp02_tier2_encoding"
        "${MODEL_PATH}/exp03_lgbm_optuna"
        "${MODEL_PATH}/exp04_stacking_ensemble"
        "${MODEL_PATH}/exp05_transfer_learning"
    )

    MODEL_FOUND=false
    for model_dir in "${POSSIBLE_MODEL_DIRS[@]}"; do
        if [ -d "$model_dir" ]; then
            MODEL_FOUND=true
            break
        fi
    done

    if [ "$MODEL_FOUND" = false ]; then
        echo "⊘ ${EXPERIMENT_ID}: No trained model found, skipping"
        ((SKIPPED++))
        continue
    fi

    ((TOTAL++))

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Experiment ${TOTAL}: ${EXPERIMENT_ID}"
    echo "   Template: ${TEMPLATE}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Run prediction + submission
    uv run python scripts/ml_runner.py \
        --project ${PROJECT} \
        --template ${TEMPLATE} \
        --experiment-id ${EXPERIMENT_ID} \
        --auto-submit \
        --wait-seconds ${WAIT_TIME} || {
        echo "ERROR: Failed to submit ${EXPERIMENT_ID}"
        echo "Continuing with next experiment..."
        continue
    }

    ((SUBMITTED++))

    echo ""
    echo "✓ Submitted ${EXPERIMENT_ID}"
    echo ""

    # Small delay between submissions
    sleep 5
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo "Total experiments found: ${TOTAL}"
echo "Successfully submitted: ${SUBMITTED}"
echo "Skipped: ${SKIPPED}"
echo ""

if [ $SUBMITTED -gt 0 ]; then
    echo "Check results:"
    echo "  uv run python scripts/submissions_tracker.py --project ${PROJECT} list | head -20"
fi

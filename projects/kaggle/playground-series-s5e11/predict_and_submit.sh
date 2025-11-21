#!/bin/bash
# Generate predictions and submit for already trained models
# Usage: ./predict_and_submit.sh [experiment_id]

set -e

PROJECT="playground-series-s5e11"
WAIT_TIME=45

# Check if experiment_id provided
if [ -z "$1" ]; then
    echo "ERROR: Experiment ID required"
    echo "Usage: ./predict_and_submit.sh [experiment_id]"
    echo ""
    echo "Available experiments:"
    ls -1 experiments/ | grep "^exp-" | head -10
    exit 1
fi

EXPERIMENT_ID=$1

echo "=========================================="
echo "PREDICT & SUBMIT"
echo "Experiment: ${EXPERIMENT_ID}"
echo "=========================================="
echo ""

# Change to repo root
cd /mnt/ml/kaggle-fork1

# Find which template was used for this experiment
TEMPLATE=$(grep -r "template" "projects/kaggle/${PROJECT}/experiments/${EXPERIMENT_ID}/state.json" 2>/dev/null | grep -oP 'exp\d+-\w+' | head -1)

if [ -z "$TEMPLATE" ]; then
    echo "ERROR: Could not determine template for experiment ${EXPERIMENT_ID}"
    echo "Please specify template manually:"
    echo "  uv run python scripts/ml_runner.py --project ${PROJECT} --template [TEMPLATE] --experiment-id ${EXPERIMENT_ID} --auto-submit"
    exit 1
fi

echo "Detected template: ${TEMPLATE}"
echo ""

# Run prediction + submission
echo "Generating predictions and submitting..."
uv run python scripts/ml_runner.py \
    --project ${PROJECT} \
    --template ${TEMPLATE} \
    --experiment-id ${EXPERIMENT_ID} \
    --auto-submit \
    --wait-seconds ${WAIT_TIME}

echo ""
echo "✓ Complete!"
echo ""

#!/bin/bash
# List all trained experiments
# Usage: ./list_trained_experiments.sh

PROJECT="playground-series-s5e11"

cd /mnt/ml/kaggle-fork1

echo "=========================================="
echo "TRAINED EXPERIMENTS"
echo "Project: ${PROJECT}"
echo "=========================================="
echo ""

# Find all experiment directories
EXPERIMENT_DIRS=$(find "projects/kaggle/${PROJECT}/experiments/" -maxdepth 1 -type d -name "exp-*" | sort)

if [ -z "$EXPERIMENT_DIRS" ]; then
    echo "No experiments found!"
    exit 1
fi

echo "┌────────────────────┬─────────────────┬──────────┬─────────────────────────┐"
echo "│ Experiment ID      │ Template        │ Status   │ Local CV                │"
echo "├────────────────────┼─────────────────┼──────────┼─────────────────────────┤"

for exp_dir in $EXPERIMENT_DIRS; do
    EXPERIMENT_ID=$(basename "$exp_dir")
    STATE_FILE="${exp_dir}/state.json"

    if [ ! -f "$STATE_FILE" ]; then
        printf "│ %-18s │ %-15s │ %-8s │ %-23s │\n" "$EXPERIMENT_ID" "N/A" "N/A" "No state.json"
        continue
    fi

    # Extract info from state.json
    TEMPLATE=$(grep -oP '"template":\s*"\K[^"]+' "$STATE_FILE" | head -1 || echo "unknown")
    MODEL_STATUS=$(grep -oP '"model":\s*\{[^}]*"status":\s*"\K[^"]+' "$STATE_FILE" 2>/dev/null || echo "unknown")
    LOCAL_CV=$(grep -oP '"local_cv":\s*\K[0-9.]+' "$STATE_FILE" 2>/dev/null || echo "N/A")

    # Truncate template name if too long
    TEMPLATE_SHORT=$(echo "$TEMPLATE" | cut -c1-15)

    printf "│ %-18s │ %-15s │ %-8s │ %-23s │\n" "$EXPERIMENT_ID" "$TEMPLATE_SHORT" "$MODEL_STATUS" "$LOCAL_CV"
done

echo "└────────────────────┴─────────────────┴──────────┴─────────────────────────┘"
echo ""

# Show AutoGluon model directories
echo "Trained model directories:"
MODEL_PATH="projects/kaggle/${PROJECT}/AutogluonModels"
if [ -d "$MODEL_PATH" ]; then
    ls -1 "$MODEL_PATH" | grep -E "^exp0[1-5]" | while read model_dir; do
        SIZE=$(du -sh "${MODEL_PATH}/${model_dir}" 2>/dev/null | cut -f1)
        echo "  📁 ${model_dir} (${SIZE})"
    done
else
    echo "  No model directories found"
fi

echo ""
echo "To submit an experiment:"
echo "  ./predict_and_submit.sh [experiment_id]"
echo ""
echo "To submit all at once:"
echo "  ./submit_all_experiments.sh"
echo ""

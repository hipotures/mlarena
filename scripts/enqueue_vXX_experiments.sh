#!/bin/bash
PROJECT="playground-series-s6e1"
BASE="test_c_01_0306"

for v in {01..10}; do
    TEMPLATE="${BASE}_v${v}"
    EXP_ID="exp-${TEMPLATE}"
    
    echo "Adding task for variant v${v}..."
    python scripts/task_queue.py --project $PROJECT add --command "model model_template=${TEMPLATE} experiment_id=${EXP_ID} skip_submit=true skip_git=true model.mla_retention=true"
done

echo "Done. Use 'python scripts/task_queue.py --project $PROJECT run' to start processing."

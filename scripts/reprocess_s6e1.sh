#!/bin/bash
# Batch preprocessing for 1000 experiments
PROJECT="playground-series-s6e1"
PREFIX="test_c_01_"
START=1
END=1000

echo "Starting batch preprocessing for $PROJECT (Prefix: $PREFIX)..."

for i in $(seq -f "%04g" $START $END); do
    TEMPLATE="${PREFIX}${i}"
    echo "----------------------------------------------------------"
    echo "Target: $TEMPLATE ($i/$END)"
    
    # Run mla preprocess. 
    # Note: without --force, it will use cache if train/test exist.
    uv run python scripts/mla.py preprocess --project "$PROJECT" --preprocess-template "$TEMPLATE"
done

echo "Done."

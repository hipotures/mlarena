#!/bin/bash
# Run all weight_evaluation tests for playground-series-s5e12
# Total runtime: ~34h (sorted by duration: shortest first)

PROJECT="playground-series-s5e12"
PREPROCESS="top0_te_scale_ext_m"

# Boosting + NN tests (8h each, 30 HPO trials, GPU required)
uv run python scripts/mla.py -p $PROJECT --model-template test_with_nn_wt --preprocess-template $PREPROCESS
uv run python scripts/mla.py -p $PROJECT --model-template test_with_nn_unwt --preprocess-template $PREPROCESS

echo "All tests completed at $(date)"

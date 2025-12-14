#!/bin/bash
# Run all weight_evaluation tests for playground-series-s5e12

PROJECT="playground-series-s5e12"

# Quick tests (30min each)
uv run python scripts/mla.py -p $PROJECT --model-template test_xgb_only_wt
uv run python scripts/mla.py -p $PROJECT --model-template test_xgb_only_unwt

# Full tests (3h each)
uv run python scripts/mla.py -p $PROJECT --model-template test_wt_eval_true
uv run python scripts/mla.py -p $PROJECT --model-template test_wt_eval_false

# NN tests (4h each, GPU)
uv run python scripts/mla.py -p $PROJECT --model-template test_with_nn_wt
uv run python scripts/mla.py -p $PROJECT --model-template test_with_nn_unwt

echo "All tests completed at $(date)"

#!/usr/bin/env bash
set -euo pipefail

# Run AV weights preprocess (best boost) and then four models in order.

PROJECT="${PROJECT:-playground-series-s5e12}"

echo "==> Preprocess: av_weights_best_boost (project=${PROJECT})"
uv run python scripts/mla.py preprocess \
  project="${PROJECT}" \
  preprocess_template=av_weights_best_boost \
  force=true

for TEMPLATE in best-1h-av-gbm best-1h-av-xgb best-3h-av-cat best-8h-av-boost; do
  echo "==> Model: ${TEMPLATE} (project=${PROJECT})"
  uv run python scripts/mla.py model \
    project="${PROJECT}" \
    model_template="${TEMPLATE}"
done

echo "All runs completed."

#!/usr/bin/env bash
set -euo pipefail

PROJECT=playground-series-s5e12
# Wspólne ustawienia modelu (bez submitu, bo uruchamiamy tylko trening)
MODEL_FLAGS=(--preset best --time-limit 600 --use-gpu 0)
MODEL_TEMPLATE="cpu-best-1h-gbm-drift-mi"

# Helper to log local CV to /tmp/scores.log if available in state.json
log_score() {
  local exp_id="$1"
  local state_file="projects/kaggle/$PROJECT/experiments/$exp_id/state.json"
  if [ -f "$state_file" ]; then
    local score
    score=$(jq -r '.modules.model.payload.local_cv_score // empty' "$state_file" 2>/dev/null)
    if [ -n "$score" ] && [ "$score" != "null" ]; then
      echo "$exp_id local_cv_score=$score" >> /tmp/scores.log
      echo "Logged score for $exp_id -> $score"
    else
      echo "No local_cv_score found in $state_file"
    fi
  else
    echo "State file not found: $state_file"
  fi
}

# 0) Model bez preprocessu (surowe dane) – odkomentuj, jeśli chcesz ponowić baseline
# uv run python scripts/mla.py model \
#   --project "$PROJECT" \
#   --experiment-id 000.raw_best10m \
#   "${MODEL_FLAGS[@]}"
# log_score "000.raw_best10m"

# 1) Jeden moduł preprocess na surowych danych, osobny model dla każdego
PRE_TEMPLATES=(
  "sanity_check_ps5e12"
  "imputer"
  "rare_category_handler"
  "encoder_catboost"
  "drift_detector_moderate"
  "feature_selector_modelimp"
  "feature_selector_mi"
  "outlier_handler_light"
  "imbalance_handler"
)

for tpl in "${PRE_TEMPLATES[@]}"; do
  uv run python scripts/mla.py preprocess \
    --project "$PROJECT" \
    --preprocess-template "$tpl"

  uv run python scripts/mla.py model \
    --project "$PROJECT" \
    --preprocess-template "$tpl" \
    --model-template "$MODEL_TEMPLATE" \
    --experiment-id "solo.$tpl" \
    "${MODEL_FLAGS[@]}"

  log_score "solo.$tpl"
done

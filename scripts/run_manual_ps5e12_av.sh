#!/usr/bin/env bash
set -euo pipefail

PROJECT=playground-series-s5e12
# Wspólne ustawienia modelu (bez submitu, bo uruchamiamy tylko trening)
MODEL_FLAGS=(--preset best --time-limit 600 --use-gpu 0)
MODEL_TEMPLATE_AV="cpu-dev-5m-av"  # używa autogluon_av_weights

# Helper to log local CV to /tmp/scores.log if available in state.json
log_score() {
  local exp_id="$1"
  local state_file="projects/kaggle/$PROJECT/experiments/$exp_id/state.json"
  if [ -f "$state_file" ]; then
    local score
    score=$(jq -r '.modules.model.payload.local_cv // empty' "$state_file" 2>/dev/null)
    if [ -n "$score" ] && [ "$score" != "null" ]; then
      echo "$exp_id local_cv=$score" >> /tmp/scores.log
      echo "Logged score for $exp_id -> $score"
    else
      echo "No local_cv found in $state_file"
    fi
  else
    echo "State file not found: $state_file"
  fi
}

#0) Model bez preprocessu (surowe dane) – odkomentuj, jeśli chcesz ponowić baseline z AV
uv run python scripts/mla.py model \
  --project "$PROJECT" \
  --experiment-id 000.raw_best10m_av \
  --model-template "$MODEL_TEMPLATE_AV" \
  "${MODEL_FLAGS[@]}"
log_score "000.raw_best10m_av"

# 1) Jeden moduł preprocess na surowych danych, osobny model AV dla każdego
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
    --model-template "$MODEL_TEMPLATE_AV" \
    --experiment-id "av.$tpl" \
    "${MODEL_FLAGS[@]}"

  log_score "av.$tpl"
done

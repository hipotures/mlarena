#!/bin/bash
#
# Batch execution script for 48 DOE Phase 2 experiments (E36-E83)
# Usage: ./run_experiments_e2.sh [phase]
#
# Phases:
#   transforms - Run E36-E42 (Target Transform Combinations)
#   rare       - Run E43-E50 (Rare Category Combinations)
#   outliers   - Run E51-E54 (Outlier Handling)
#   hybrid     - Run E55-E60 (Hybrid Chains)
#   tuning     - Run E61-E70 (New Encoders & Parameter Tuning)
#   advanced   - Run E71-E83 (Advanced Combinations)
#   all        - Run all experiments (default)
#

set -euo pipefail

# Parse phase argument
PHASE="${1:-all}"

PROJECT="playground-series-s6e1"
SCRIPT="uv run python scripts/mla.py"

# Counters for summary
TOTAL_RUN=0
TOTAL_SKIP=0
TOTAL_FAIL=0

# Results for summary table
declare -a EXP_IDS=()
declare -a EXP_NAMES=()
declare -a EXP_STATUS=()
declare -a EXP_SCORES=()

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

get_local_cv_score() {
    local exp_name=$1
    local state_path="projects/kaggle/${PROJECT}/experiments/${exp_name}/state.json"

    if [ ! -f "$state_path" ]; then
        echo "n/a"
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "n/a"
        return 0
    fi

    python3 << EOF 2>/dev/null || true
import json
try:
    with open("${state_path}") as f:
        state = json.load(f)
    score = state.get("modules", {}).get("model", {}).get("payload", {}).get("local_cv_score")
    if score is None:
        print("n/a")
    else:
        print(f"{score:.6f}")
except Exception:
    print("n/a")
EOF
}

register_result() {
    local exp_id=$1
    local exp_name=$2
    local status=$3
    local score
    score=$(get_local_cv_score "$exp_name")

    EXP_IDS+=("$exp_id")
    EXP_NAMES+=("$exp_name")
    EXP_STATUS+=("$status")
    EXP_SCORES+=("$score")
}

print_summary() {
    if [ "${#EXP_IDS[@]}" -eq 0 ]; then
        return 0
    fi

    echo -e "\n${BLUE}[INFO]${NC} === Summary Phase 2 ==="
    printf "% -6s % -32s % -10s % -12s\n" "ID" "Experiment" "Status" "LocalCV"
    local i
    for i in "${!EXP_IDS[@]}"; do
        printf "% -6s % -32s % -10s % -12s\n" "${EXP_IDS[$i]}" "${EXP_NAMES[$i]}" "${EXP_STATUS[$i]}" "${EXP_SCORES[$i]}"
    done
    echo -e "${BLUE}[INFO]${NC} Totals: run=${TOTAL_RUN} skipped=${TOTAL_SKIP} failed=${TOTAL_FAIL}"
}

trap print_summary EXIT

check_experiment_completed() {
    local exp_name=$1
    local project_dir="projects/kaggle/${PROJECT}"
    local exp_dir="${project_dir}/experiments/${exp_name}"

    if [ ! -d "$exp_dir" ]; then
        return 1
    fi

    if [ ! -f "${exp_dir}/state.json" ]; then
        return 1
    fi

    python3 << EOF 2>/dev/null
import json
import sys
try:
    with open("${exp_dir}/state.json") as f:
        state = json.load(f)
    # Check if submit module completed (since we want full flow with submission)
    # Or at least predict if submission is skipped
    modules = state.get("modules", {})
    if modules.get("submit", {}).get("status") == "completed":
        sys.exit(0)
    if modules.get("predict", {}).get("status") == "completed":
        # If submit wasn't run but predict was, consider it done for now
        sys.exit(0)
    sys.exit(1)
except:
    sys.exit(1)
EOF
    return $?
}

run_experiment() {
    local exp_id=$1
    local template=$2

    # Construct experiment name: exp-E36-log1p_smooth
    local exp_name="exp-${exp_id}-${template#*_}"

    if check_experiment_completed "$exp_name"; then
        log_success "${exp_id} already completed (skipping)"
        ((++TOTAL_SKIP))
        register_result "$exp_id" "$exp_name" "skipped"
        return 0
    fi

    log_info "Running ${exp_id}: ${template} → ${exp_name}"

    # Note: Removed skip_submit=true to enable submission as requested
    if $SCRIPT --project $PROJECT --experiment-id "$exp_name" --model-template "$template"; then
        log_success "${exp_id} completed successfully"
        ((++TOTAL_RUN))
        register_result "$exp_id" "$exp_name" "completed"
        return 0
    else
        log_error "${exp_id} failed"
        ((++TOTAL_FAIL))
        register_result "$exp_id" "$exp_name" "failed"
        return 1
    fi
}

run_transforms() {
    log_info "=== Phase 2.1: Target Transform Combinations (E36-E42) ==="
    run_experiment "E36" "20260103180000_log1p_smooth"
    run_experiment "E37" "20260103180100_log1p_cat"
    run_experiment "E38" "20260103180200_log1p_onehot"
    run_experiment "E39" "20260103180300_sqrt_smooth"
    run_experiment "E40" "20260103180400_sqrt_cat"
    run_experiment "E41" "20260103180500_sqrt_oof"
    run_experiment "E42" "20260103180600_sqrt_onehot"
}

run_rare() {
    log_info "=== Phase 2.2: Rare Category Combinations (E43-E50) ==="
    run_experiment "E43" "20260103180700_rare01_smooth"
    run_experiment "E44" "20260103180800_rare01_cat"
    run_experiment "E45" "20260103180900_rare01_oof"
    run_experiment "E46" "20260103181000_rare01_onehot"
    run_experiment "E47" "20260103181100_rare005_smooth"
    run_experiment "E48" "20260103181200_rare005_cat"
    run_experiment "E49" "20260103181300_rare005_oof"
    run_experiment "E50" "20260103181400_rare005_onehot"
}

run_outliers() {
    log_info "=== Phase 2.3: Outlier Handling (E51-E54) ==="
    run_experiment "E51" "20260103181500_outlier_smooth"
    run_experiment "E52" "20260103181600_outlier_cat"
    run_experiment "E53" "20260103181700_outlier_oof"
    run_experiment "E54" "20260103181800_outlier_onehot"
}

run_hybrid() {
    log_info "=== Phase 2.4: Hybrid Chains (E55-E60) ==="
    run_experiment "E55" "20260103181900_rare01_log_smooth"
    run_experiment "E56" "20260103182000_rare01_log_cat"
    run_experiment "E57" "20260103182100_out_log_smooth"
    run_experiment "E58" "20260103182200_out_log_cat"
    run_experiment "E59" "20260103182300_rare_out_log_smooth"
    run_experiment "E60" "20260103182400_rare_out_log_cat"
}

run_tuning() {
    log_info "=== Phase 2.5: Tuning & New Encoders (E61-E70) ==="
    run_experiment "E61" "20260103182500_glmm_enc"
    run_experiment "E62" "20260103182600_loo_enc"
    run_experiment "E63" "20260103182700_smooth_s5"
    run_experiment "E64" "20260103182800_smooth_s20"
    run_experiment "E65" "20260103182900_smooth_m50"
    run_experiment "E66" "20260103183000_smooth_m200"
    run_experiment "E67" "20260103183100_oof_s5"
    run_experiment "E68" "20260103183200_oof_m10"
    run_experiment "E69" "20260103183300_cat_a1"
    run_experiment "E70" "20260103183400_cat_a10"
}

run_advanced() {
    log_info "=== Phase 2.6: Advanced Combinations (E71-E83) ==="
    run_experiment "E71" "20260103183500_rare_log_glmm"
    run_experiment "E72" "20260103183600_rare_log_loo"
    run_experiment "E73" "20260103183700_rare005_log_smooth"
    run_experiment "E74" "20260103183800_rare005_log_cat"
    run_experiment "E75" "20260103183900_out_sqrt_smooth"
    run_experiment "E76" "20260103184000_out_sqrt_cat"
    run_experiment "E77" "20260103184100_rare_out_smooth"
    run_experiment "E78" "20260103184200_rare_out_cat"
    run_experiment "E79" "20260103184300_date_smooth"
    run_experiment "E80" "20260103184400_date_cat"
    run_experiment "E81" "20260103184500_date_log_smooth"
    run_experiment "E82" "20260103184600_date_log_cat"
    run_experiment "E83" "20260103184700_kitchen_sink_v2"
}

run_all() {
    run_transforms
    run_rare
    run_outliers
    run_hybrid
    run_tuning
    run_advanced
    log_success "All 48 Phase 2 experiments completed!"
}

# Main execution
case "$PHASE" in
    transforms) run_transforms ;; 
    rare)       run_rare ;; 
    outliers)   run_outliers ;; 
    hybrid)     run_hybrid ;; 
    tuning)     run_tuning ;; 
    advanced)   run_advanced ;; 
    all)        run_all ;; 
    *) 
        log_error "Unknown phase: $PHASE"
        echo "Usage: $0 [transforms|rare|outliers|hybrid|tuning|advanced|all]"
        exit 1 
        ;; 
esac

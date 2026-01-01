#!/bin/bash
#
# Batch execution script for 36 DOE experiments
# Usage: ./run_experiments.sh [phase]
#
# Phases:
#   baseline - Run E00 only (establish baseline)
#   single   - Run E01-E20 (single-step experiments)
#   chains   - Run E21-E35 (chain experiments)
#   all      - Run all experiments (default)
#
# NOTE: Script automatically skips already completed experiments
#       (those with predict status = completed)
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

    echo -e "\n${BLUE}[INFO]${NC} === Summary ==="
    printf "%-6s %-28s %-10s %-12s\n" "ID" "Experiment" "Status" "LocalCV"
    local i
    for i in "${!EXP_IDS[@]}"; do
        printf "%-6s %-28s %-10s %-12s\n" "${EXP_IDS[$i]}" "${EXP_NAMES[$i]}" "${EXP_STATUS[$i]}" "${EXP_SCORES[$i]}"
    done
    echo -e "${BLUE}[INFO]${NC} Totals: run=${TOTAL_RUN} skipped=${TOTAL_SKIP} failed=${TOTAL_FAIL}"
}

trap print_summary EXIT

check_experiment_completed() {
    local exp_name=$1

    # Sprawdź czy istnieje katalog z tym eksperymentem
    local project_dir="projects/kaggle/${PROJECT}"
    local exp_dir="${project_dir}/experiments/${exp_name}"

    if [ ! -d "$exp_dir" ]; then
        return 1  # Not found
    fi

    if [ ! -f "${exp_dir}/state.json" ]; then
        return 1  # No state file
    fi

    # Sprawdź status modułu predict (bo skip_submit=true kończy na predict)
    # JSON jest w wielu liniach, więc używamy python do parsowania
    python3 << EOF 2>/dev/null
import json
import sys
try:
    with open("${exp_dir}/state.json") as f:
        state = json.load(f)
    predict_status = state.get("modules", {}).get("predict", {}).get("status", "")
    if predict_status == "completed":
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

    # Utworz nazwę eksperymentu z template (np. exp-E00-baseline)
    local exp_name="exp-${exp_id}-${template#*_}"  # usuwa timestamp prefix

    # Sprawdź czy eksperyment już został ukończony
    if check_experiment_completed "$exp_name"; then
        log_success "${exp_id} already completed (skipping)"
        ((++TOTAL_SKIP))
        register_result "$exp_id" "$exp_name" "skipped"
        return 0
    fi

    log_info "Running ${exp_id}: ${template} → ${exp_name}"

    if $SCRIPT model --project $PROJECT --experiment-id "$exp_name" --model-template "$template" skip_submit=true; then
        if $SCRIPT predict --project $PROJECT --experiment-id "$exp_name" skip_submit=true; then
            log_success "${exp_id} completed successfully"
            ((++TOTAL_RUN))
            register_result "$exp_id" "$exp_name" "completed"
            return 0
        else
            log_error "${exp_id} predict failed"
            ((++TOTAL_FAIL))
            register_result "$exp_id" "$exp_name" "failed"
            return 1
        fi
    else
        log_error "${exp_id} model failed"
        ((++TOTAL_FAIL))
        register_result "$exp_id" "$exp_name" "failed"
        return 1
    fi
}

run_baseline() {
    log_info "=== Phase 1: Baseline (E00) ==="
    run_experiment "E00" "20260101120000_baseline"
}

run_single_steps() {
    log_info "=== Phase 2: Single-Step Experiments (E01-E20) ==="

    local templates=(
        "20260101120100_onehot"
        "20260101120200_target_enc"
        "20260101120300_custom_feat"
        "20260101120400_poly2"
        "20260101120500_standard_scale"
        "20260101120600_robust_scale"
        "20260101120700_log1p_target"
        "20260101120800_sqrt_target"
        "20260101120900_rare_001"
        "20260101121000_rare_0005"
        "20260101121100_var_thresh"
        "20260101121200_rfe_20"
        "20260101121300_rfe_15"
        "20260101121400_feat_interact"
        "20260101121500_datetime"
        "20260101121600_outlier_iqr"
        "20260101121700_target_smooth"
        "20260101121800_custom_only"
        "20260101121900_poly3"
        "20260101122000_catboost_enc"
    )

    local i=1
    for template in "${templates[@]}"; do
        local exp_id=$(printf "E%02d" $i)
        run_experiment "$exp_id" "$template" || log_warning "Continuing despite failure"
        ((i++))
    done
}

run_chains() {
    log_info "=== Phase 3: Chain Experiments (E21-E35) ==="

    # Chain A (E21-E23)
    log_info "Chain A: target_enc → custom_feat → log1p → rfe"
    run_experiment "E21" "20260101122100_chain_a1"
    run_experiment "E22" "20260101122200_chain_a2"
    run_experiment "E23" "20260101122300_chain_a3"

    # Chain B (E24-E26)
    log_info "Chain B: target_enc → poly2 → interact → rfe"
    run_experiment "E24" "20260101122400_chain_b1"
    run_experiment "E25" "20260101122500_chain_b2"
    run_experiment "E26" "20260101122600_chain_b3"

    # Chain C (E27-E29)
    log_info "Chain C: custom → target_enc → sqrt → rare"
    run_experiment "E27" "20260101122700_chain_c1"
    run_experiment "E28" "20260101122800_chain_c2"
    run_experiment "E29" "20260101122900_chain_c3"

    # Chain D (E30-E33)
    log_info "Chain D: log1p → target_enc → custom → poly/rfe"
    run_experiment "E30" "20260101123000_chain_d1"
    run_experiment "E31" "20260101123100_chain_d2"
    run_experiment "E32" "20260101123200_chain_d3"
    run_experiment "E33" "20260101123300_chain_d4"

    # Kitchen Sink (E34-E35)
    log_info "Kitchen Sink: All best features"
    run_experiment "E34" "20260101123400_ultra"
    run_experiment "E35" "20260101123500_ultra_v2"
}

run_all() {
    log_info "=== Running All 36 Experiments ==="

    run_baseline
    log_info "Baseline complete. Starting single-step experiments..."
    sleep 2

    run_single_steps
    log_info "Single-step experiments complete. Starting chains..."
    sleep 2

    run_chains

    log_success "All 36 experiments completed!"
}

# Main execution
case "$PHASE" in
    baseline)
        run_baseline
        ;;
    single)
        run_single_steps
        ;;
    chains)
        run_chains
        ;;
    all)
        run_all
        ;;
    *)
        log_error "Unknown phase: $PHASE"
        echo "Usage: $0 [baseline|single|chains|all]"
        exit 1
        ;;
esac

# Summary
echo ""
echo "================================================================================"
log_success "Experiment execution phase '$PHASE' completed!"
echo "================================================================================"
echo -e "  ${GREEN}✓ Completed:${NC} $TOTAL_RUN experiments"
echo -e "  ${BLUE}⊙ Skipped:${NC}   $TOTAL_SKIP experiments (already done)"
if [ $TOTAL_FAIL -gt 0 ]; then
    echo -e "  ${RED}✗ Failed:${NC}    $TOTAL_FAIL experiments"
fi
echo "================================================================================"
echo ""

if [ $TOTAL_FAIL -gt 0 ]; then
    log_warning "Some experiments failed. Review logs above for details."
    exit 1
fi

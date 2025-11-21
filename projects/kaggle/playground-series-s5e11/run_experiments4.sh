#!/bin/bash
# Run all 5 experiments sequentially
# Each experiment: ~1-1.5 hours
# Total time: ~6-8 hours

set -e  # Exit on error

PROJECT="playground-series-s5e11"
WAIT_TIME=45

echo "=========================================="
echo "SERIA 5 EKSPERYMENTÓW"
echo "Playground Series S5E11"
echo "=========================================="
echo ""
echo "Expected total time: 6-8 hours"
echo "Each experiment: ~1-1.5h"
echo ""

# Change to repo root
cd /mnt/ml/kaggle-fork1

# ============================================================
# EKSPERYMENT 2: Advanced Encoding
# ============================================================
echo ""
echo "=========================================="
echo "EKSPERYMENT 2/5: Advanced Encoding"
echo "Expected boost: +0.02-0.04 AUC"
echo "Time: ~1.5h"
echo "=========================================="
echo ""

uv run python scripts/ml_runner.py \
    --project ${PROJECT} \
    --template exp02-encoding \
    --auto-submit \
    --wait-seconds ${WAIT_TIME}

echo ""
echo "✓ Eksperyment 2 zakończony"
echo ""

# ============================================================
# EKSPERYMENT 3: LightGBM + Optuna
# ============================================================
echo ""
echo "=========================================="
echo "EKSPERYMENT 3/5: LightGBM + Optuna"
echo "Expected boost: +0.01-0.02 AUC"
echo "Time: ~1.5h"
echo "=========================================="
echo ""

uv run python scripts/ml_runner.py \
    --project ${PROJECT} \
    --template exp03-lgbm-optuna \
    --auto-submit \
    --wait-seconds ${WAIT_TIME}

echo ""
echo "✓ Eksperyment 3 zakończony"
echo ""

# ============================================================
# EKSPERYMENT 4: Stacking Ensemble (BEST BET)
# ============================================================
echo ""
echo "=========================================="
echo "EKSPERYMENT 4/5: Stacking Ensemble ⭐"
echo "Expected boost: +0.01-0.03 AUC"
echo "Time: ~1.5h"
echo "=========================================="
echo ""

uv run python scripts/ml_runner.py \
    --project ${PROJECT} \
    --template exp04-stacking \
    --auto-submit \
    --wait-seconds ${WAIT_TIME}

echo ""
echo "✓ Eksperyment 4 zakończony"
echo ""

# ============================================================
# EKSPERYMENT 5: Transfer Learning
# ============================================================
echo ""
echo "=========================================="
echo "EKSPERYMENT 5/5: Transfer Learning"
echo "Expected boost: +0.005-0.01 AUC"
echo "Time: ~1.5h"
echo "=========================================="
echo ""
echo "UWAGA: Eksperyment 5 wymaga oryginalnego datasetu!"
echo "Jeśli nie masz: kaggle datasets download -d nabihazahid/loan-prediction-dataset-2025"
echo ""

uv run python scripts/ml_runner.py \
    --project ${PROJECT} \
    --template exp05-transfer \
    --auto-submit \
    --wait-seconds ${WAIT_TIME}

echo ""
echo "✓ Eksperyment 5 zakończony"
echo ""

# ============================================================
# PODSUMOWANIE
# ============================================================
echo ""
echo "=========================================="
echo "WSZYSTKIE EKSPERYMENTY ZAKOŃCZONE!"
echo "=========================================="
echo ""
echo "Sprawdź wyniki:"
echo "  uv run python scripts/submissions_tracker.py --project ${PROJECT} list"
echo ""
echo "Baseline:"
echo "  Local CV: 0.93309"
echo "  Public:   0.92434"
echo ""
echo "Oczekiwane wyniki:"
echo "  Exp 1: Public ~0.940-0.945"
echo "  Exp 2: Public ~0.945-0.950"
echo "  Exp 3: Public ~0.943-0.948"
echo "  Exp 4: Public ~0.947-0.952 (BEST)"
echo "  Exp 5: Public ~0.941-0.946"
echo ""
echo "Następne kroki:"
echo "  1. Sprawdź który eksperyment osiągnął najlepszy wynik"
echo "  2. Uruchom zwycięzcę na dłużej (4-8h) modyfikując time_limit"
echo "  3. Przeanalizuj feature importance"
echo ""

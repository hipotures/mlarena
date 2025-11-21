# Quick Start - 5 Eksperymentów

## 🚀 Uruchomienie wszystkich eksperymentów (6-8h)

```bash
cd /mnt/ml/kaggle-fork1/projects/kaggle/playground-series-s5e11
./run_experiments.sh
```

## 📋 Pojedyncze eksperymenty

### Eksperyment 1: Tier 1 Features (~1.5h)
```bash
cd /mnt/ml/kaggle-fork1
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --template exp01-tier1 \
    --auto-submit \
    --wait-seconds 45
```

### Eksperyment 2: Advanced Encoding (~1.5h)
```bash
cd /mnt/ml/kaggle-fork1
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --template exp02-encoding \
    --auto-submit \
    --wait-seconds 45
```

### Eksperyment 3: LightGBM + Optuna (~1.5h)
```bash
cd /mnt/ml/kaggle-fork1
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --template exp03-lgbm-optuna \
    --auto-submit \
    --wait-seconds 45
```

### Eksperyment 4: Stacking Ensemble ⭐ (~1.5h)
```bash
cd /mnt/ml/kaggle-fork1
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --template exp04-stacking \
    --auto-submit \
    --wait-seconds 45
```

### Eksperyment 5: Transfer Learning (~1.5h)
```bash
# UWAGA: Wymaga oryginalnego datasetu!
cd /mnt/ml/kaggle-fork1/projects/kaggle/playground-series-s5e11/data
kaggle datasets download -d nabihazahid/loan-prediction-dataset-2025
unzip loan-prediction-dataset-2025.zip

cd /mnt/ml/kaggle-fork1
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --template exp05-transfer \
    --auto-submit \
    --wait-seconds 45
```

## 📊 Sprawdzanie wyników

```bash
# Lista wszystkich submissions
uv run python scripts/submissions_tracker.py --project playground-series-s5e11 list

# Ostatnie 10 submissions
uv run python scripts/submissions_tracker.py --project playground-series-s5e11 list | head -15
```

## 🎯 Oczekiwane wyniki

| Eksperyment | Expected Public AUC | Boost |
|-------------|--------------------:|------:|
| Baseline    | 0.92434             | -     |
| Exp 1       | 0.940-0.945         | +0.016-0.021 |
| Exp 2       | 0.945-0.950         | +0.021-0.026 |
| Exp 3       | 0.943-0.948         | +0.019-0.024 |
| **Exp 4**   | **0.947-0.952**     | **+0.023-0.028** ⭐ |
| Exp 5       | 0.941-0.946         | +0.017-0.022 |

**Target:** Public ≥ 0.945

## 📝 Struktura eksperymentów

### Exp 1: Enhanced FE (Tier 1)
- Log transforms (income, loan_amount)
- DTI features (monthly_debt, payment_capacity, remaining_income)
- Critical ratios (loan_to_income, payment_to_income)
- Interest cost analysis
- Risk flags
- **~25 nowych cech**

### Exp 2: Advanced Encoding
- **Buduje na Exp 1** (wszystkie Tier 1 featury)
- Target Encoding z CV (grade_subgrade, loan_purpose)
- Weight of Evidence encoding
- Polynomial features (degree 2)
- Cross-feature interactions
- **~40-45 total cech**

### Exp 3: LightGBM + Optuna
- Custom LightGBM (zamiast AutoGluon)
- Bayesian optimization (50 trials)
- class_weight='balanced'
- 5-fold CV
- **Feature importance analysis**

### Exp 4: Stacking Ensemble ⭐
- Level 1: LightGBM + XGBoost + CatBoost
- Level 2: Logistic Regression
- Out-of-fold predictions
- Isotonic calibration
- **Prawdopodobnie najlepszy**

### Exp 5: Transfer Learning
- Pre-train na oryginalnym datasecie (20k)
- Industry statistics jako featury
- Predictions jako meta-features
- Fine-tune na competition data

## 🔧 Troubleshooting

### Problem: Brak optuna
```bash
uv add optuna
uv sync
```

### Problem: Brak catboost
```bash
uv add catboost
uv sync
```

### Problem: Brak scipy
```bash
uv add scipy scikit-learn
uv sync
```

### Problem: Eksperyment trwa >2h
Zmniejsz w `configs/templates.yaml`:
- `time_limit: 5400` → `3600` (1h)
- `n_trials: 50` → `30` (dla Optuna)

### Problem: Memory error
Zmniejsz w `configs/templates.yaml`:
- `num_bag_folds: 5` → `3`
- Użyj `presets: medium_quality` zamiast `best_quality`

## 📖 Pełna dokumentacja

Szczegóły w `EXPERIMENTS_README.md`

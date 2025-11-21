# Seria 5 Eksperymentów - Feature Engineering & Modeling Optimization

## Przegląd

Seria 5 eksperymentów zaprojektowanych do systematycznej optymalizacji pipeline'u modelowania dla konkursu Playground Series S5E11 (Loan Payback Prediction). Każdy eksperyment trwa ~1-1.5h i wprowadza nowe techniki feature engineering lub modelowania.

**Cel:** Poprawa wyniku z obecnego **0.92434 public AUC** do **0.94+ AUC**

---

## 📋 Podsumowanie Eksperymentów

| # | Nazwa | Strategia | Expected Boost | Czas | Status |
|---|-------|-----------|----------------|------|--------|
| 1 | Enhanced FE (Tier 1) | Log transforms, DTI ratios, payment capacity | +0.03-0.05 AUC | 1.5h | ✅ Gotowy |
| 2 | Advanced Encoding | Target Encoding, WoE, polynomials, interactions | +0.02-0.04 AUC | 1.5h | ✅ Gotowy |
| 3 | LightGBM + Optuna | Custom LightGBM z Bayesian tuning (50 trials) | +0.01-0.02 AUC | 1.5h | ✅ Gotowy |
| 4 | Stacking Ensemble | LightGBM + XGBoost + CatBoost → LogReg meta | +0.01-0.03 AUC | 1.5h | ✅ Gotowy |
| 5 | Transfer Learning | Pre-train na oryginalnym datasecie (20k samples) | +0.005-0.01 AUC | 1.5h | ✅ Gotowy |

**Łączny oczekiwany boost (kumulatywny):** +0.055 - 0.15 AUC (przy założeniu addytywności)

---

## 🚀 Komendy Uruchomieniowe

### Eksperyment 1: Enhanced Feature Engineering (Tier 1)

**Opis:**
- Transformacje logarytmiczne dla skewed features (annual_income, loan_amount)
- Yeo-Johnson power transformations
- DTI-based features: monthly_debt, payment_capacity, remaining_income
- Critical ratios: loan_to_income, payment_to_income, combined_dti
- Interest cost analysis: total_interest_cost, interest_burden_ratio
- Risk flags: high_dti, low_remaining_income, high_loan_to_income

**Nowe featury:** ~25 dodatkowych cech

**Komenda:**
```bash
cd /mnt/ml/kaggle-fork1

# Uruchomienie (AutoGluon best_quality, 1.5h)
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --model exp01_tier1_features \
    --auto-submit \
    --wait-seconds 45
```

**Oczekiwany wynik:** Local CV: 0.945-0.950, Public: 0.940-0.945

---

### Eksperyment 2: Advanced Encoding + Interactions

**Opis:**
- **Buduje na Tier 1** (zawiera wszystkie featury z Exp 1)
- Target Encoding z CV dla grade_subgrade, loan_purpose (prevents leakage)
- Weight of Evidence (WoE) encoding dla kategorii
- Polynomial features (degree 2) dla kluczowych zmiennych
- Cross-feature interactions:
  - income_credit_power = income × credit_score
  - loan_cost_indicator = loan_amount × interest_rate
  - credit_risk_score = credit_score / (dti + 0.01)
  - risk_adjusted_return = interest_rate / credit_quality
- Grade decomposition: grade_subgrade → grade + subgrade_num

**Nowe featury:** +15-20 na top Tier 1 (~40-45 total)

**Komenda:**
```bash
cd /mnt/ml/kaggle-fork1

# Uruchomienie
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --model exp02_tier2_encoding \
    --auto-submit \
    --wait-seconds 45
```

**Oczekiwany wynik:** Local CV: 0.950-0.955, Public: 0.945-0.950

---

### Eksperyment 3: LightGBM z Optuna Hyperparameter Tuning

**Opis:**
- Custom LightGBM (zamiast AutoGluon)
- Optuna Bayesian optimization: 50 trials
- Hyperparameters tuned:
  - learning_rate: 0.01 - 0.1 (log scale)
  - num_leaves: 20 - 150
  - max_depth: 3 - 12
  - Regularization: min_child_samples, reg_alpha, reg_lambda
  - Sampling: subsample, colsample_bytree
- Stratified 5-Fold CV
- Early stopping (50 rounds)
- class_weight='balanced' (handles imbalance)
- Feature importance analysis

**Komenda:**
```bash
cd /mnt/ml/kaggle-fork1

# Uruchomienie (50 trials Optuna + final training)
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --model exp03_lgbm_optuna \
    --auto-submit \
    --wait-seconds 45
```

**Oczekiwany wynik:** Local CV: 0.948-0.953, Public: 0.943-0.948

**Uwaga:** Ten eksperyment może dać nieco niższy wynik niż Exp 1-2 jeśli AutoGluon ensemble jest silniejszy, ale będzie miał **najlepszą feature importance analysis**.

---

### Eksperyment 4: Stacking Ensemble

**Opis:**
- **Level 1 Base Models:**
  - LightGBM (class_weight='balanced')
  - XGBoost (scale_pos_weight=4)
  - CatBoost (auto_class_weights='Balanced')
- **Level 2 Meta-Model:**
  - Logistic Regression (class_weight='balanced')
  - Calibrated z Isotonic Regression
- Out-of-Fold predictions (5-fold CV) - prevents overfitting
- Używa Tier 2 features (Tier 1 + encodings)

**Architektura:**
```
┌─────────────────────────────────────┐
│  Input: Tier 2 Features (~45 cols) │
└─────────────────┬───────────────────┘
                  │
      ┌───────────┴───────────┐
      │   5-Fold CV Training  │
      └───────────┬───────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐    ┌───▼────┐   ┌───▼─────┐
│ LightGBM│    │ XGBoost│   │ CatBoost│
│ (500 it)│    │ (500 it)│   │ (500 it)│
└───┬───┘    └───┬────┘   └───┬─────┘
    │             │             │
    └─────────────┼─────────────┘
                  │
         ┌────────▼────────┐
         │ Out-of-Fold     │
         │ Predictions (3) │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ Logistic Reg    │
         │ + Calibration   │
         └────────┬────────┘
                  │
            Final Predictions
```

**Komenda:**
```bash
cd /mnt/ml/kaggle-fork1

# Uruchomienie (5 folds × 3 models = 15 base models total)
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --model exp04_stacking_ensemble \
    --auto-submit \
    --wait-seconds 45
```

**Oczekiwany wynik:** Local CV: 0.952-0.958, Public: 0.947-0.952

**Uwaga:** To prawdopodobnie **najlepszy eksperyment** - ensemble diversity + calibration często wygrywa konkursy.

---

### Eksperyment 5: Transfer Learning

**Opis:**
- **Pre-training** na oryginalnym datasecie (20k samples)
- Compute industry statistics:
  - Default rates by loan_purpose, grade, employment_status
  - Median income/loan by purpose
  - Median interest rate by grade
- Use pre-trained predictions as **meta-feature**
- Add **statistical augmentation** features:
  - industry_default_rate_purpose
  - industry_median_income_purpose
  - income_vs_industry (deviation from norm)
  - loan_vs_industry
  - rate_vs_industry
- **Fine-tune** na competition data
- Używa Tier 1 features

**Pipeline:**
```
1. Load original dataset (20k samples)
   ↓
2. Compute industry statistics
   ↓
3. Pre-train AutoGluon (30 min, medium_quality)
   ↓
4. Generate predictions on competition data
   ↓
5. Add industry stats + pretrain predictions as features
   ↓
6. Fine-tune AutoGluon (1h, best_quality)
```

**Wymagania:**
⚠️ **Musisz pobrać oryginalny dataset:**
```bash
cd /mnt/ml/kaggle-fork1/projects/kaggle/playground-series-s5e11/data

# Download original dataset
kaggle datasets download -d nabihazahid/loan-prediction-dataset-2025

# Unzip
unzip loan-prediction-dataset-2025.zip

# Rename if needed to: loan_dataset_20000.csv
```

**Komenda:**
```bash
cd /mnt/ml/kaggle-fork1

# Uruchomienie (30 min pretrain + 1h finetune)
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --model exp05_transfer_learning \
    --auto-submit \
    --wait-seconds 45
```

**Oczekiwany wynik:** Local CV: 0.945-0.950, Public: 0.941-0.946

**Uwaga:** Boost może być mniejszy jeśli oryginalny dataset ma inną dystrybucję niż syntetyczny.

---

## 📊 Strategia Uruchamiania

### Rekomendowana kolejność:

#### Faza 1: Quick Wins (Run All, ~6h total)
```bash
# Uruchom wszystkie 5 eksperymentów równolegle lub sekwencyjnie
# Każdy ~1.5h, więc jeśli masz możliwość równoległego uruchomienia:
# - Exp 1, 2, 3 równolegle (3x 1.5h = 1.5h wall time)
# - Exp 4, 5 po zakończeniu pierwszej partii

# Lub sekwencyjnie (bezpieczniejsze):
1. exp01_tier1_features
2. exp02_tier2_encoding
3. exp04_stacking_ensemble  # Ten prawdopodobnie najlepszy
4. exp03_lgbm_optuna
5. exp05_transfer_learning  # Jeśli masz oryginalny dataset
```

#### Faza 2: Identyfikacja zwycięzcy
Po zakończeniu wszystkich, sprawdź wyniki:
```bash
# Sprawdź submissions
uv run python scripts/submissions_tracker.py --project playground-series-s5e11 list

# Sprawdź local CV scores vs public scores
# Wybierz eksperyment z:
# 1. Najwyższym public score
# 2. Najmniejszym overfittingiem (CV - Public)
```

#### Faza 3: Long Training
Zwycięzcę uruchom na dłużej:
```bash
# Przykład: jeśli exp04_stacking_ensemble wygrał
# Zmodyfikuj config i uruchom na 4-8h:

# Dla AutoGluon-based (exp01, exp02, exp05):
# Zmień time_limit w kodzie na 14400 (4h) lub 28800 (8h)

# Dla ensemble (exp04):
# Zwiększ base_model_iterations z 500 do 1500
# Dodaj więcej folds (z 5 do 10)

# Dla Optuna (exp03):
# Zwiększ n_trials z 50 do 150-200
```

---

## 🔍 Analiza Wyników

### Po każdym eksperymencie sprawdź:

1. **Local CV Score** - z outputu treningu
2. **Public LB Score** - z Kaggle
3. **Overfitting** - różnica CV - Public (powinna być <0.01)
4. **Feature Importance** - które nowe featury pomagają?

### Przykładowa analiza:

```bash
# View latest submission
uv run python scripts/submissions_tracker.py --project playground-series-s5e11 list | head -5

# Porównaj z baseline:
# Baseline: autogluon_eda_features_fixed
#   Local CV: 0.93309
#   Public:   0.92434
#   Gap:      0.00875 (overfitting)

# Expected improvements:
# Exp 1: Public ~0.940-0.945 (gap should decrease)
# Exp 2: Public ~0.945-0.950
# Exp 4: Public ~0.947-0.952 (best bet)
```

---

## 🐛 Troubleshooting

### Problem: "Original dataset not found" (Exp 5)
**Solution:**
```bash
cd projects/kaggle/playground-series-s5e11/data
kaggle datasets download -d nabihazahid/loan-prediction-dataset-2025
unzip loan-prediction-dataset-2025.zip
```

### Problem: "ImportError: No module named 'optuna'" (Exp 3)
**Solution:**
```bash
uv add optuna
uv sync
```

### Problem: "Memory error" podczas treningu
**Solution:**
- Zmniejsz `num_bag_folds` z 5 do 3
- Zmniejsz `time_limit`
- Użyj `presets='medium_quality'` zamiast `best_quality`

### Problem: Eksperyment trwa >2h
**Solution:**
- Sprawdź czy dataset nie jest za duży
- Zmniejsz `time_limit` w get_default_config()
- Dla Optuna: zmniejsz `n_trials` z 50 do 30

### Problem: Public score gorszy niż baseline
**Przyczyny:**
1. **Overfitting** - za dużo cech, za mało regularyzacji
2. **Data leakage** - target encoding źle zaimplementowany
3. **Incompatible features** - test set ma inne rozkłady

**Debugging:**
```python
# Sprawdź feature statistics train vs test
import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Compare distributions
for col in train.columns:
    if col != 'loan_paid_back':
        print(f"{col}:")
        print(f"  Train: mean={train[col].mean():.3f}, std={train[col].std():.3f}")
        print(f"  Test:  mean={test[col].mean():.3f}, std={test[col].std():.3f}")
```

---

## 📈 Expected Performance Trajectory

```
Baseline (autogluon_eda_features_fixed)
└─ Local CV: 0.93309
└─ Public:   0.92434
   │
   ├─ Exp 1 (Tier 1 FE)
   │  └─ Local CV: 0.945-0.950 (+0.012-0.017)
   │  └─ Public:   0.940-0.945 (+0.016-0.021)
   │
   ├─ Exp 2 (Tier 2 Encoding)
   │  └─ Local CV: 0.950-0.955 (+0.017-0.022)
   │  └─ Public:   0.945-0.950 (+0.021-0.026)
   │
   ├─ Exp 3 (LightGBM Optuna)
   │  └─ Local CV: 0.948-0.953 (+0.015-0.020)
   │  └─ Public:   0.943-0.948 (+0.019-0.024)
   │
   ├─ Exp 4 (Stacking) ⭐ BEST BET
   │  └─ Local CV: 0.952-0.958 (+0.019-0.025)
   │  └─ Public:   0.947-0.952 (+0.023-0.028)
   │
   └─ Exp 5 (Transfer Learning)
      └─ Local CV: 0.945-0.950 (+0.012-0.017)
      └─ Public:   0.941-0.946 (+0.017-0.022)
```

**Target:** Public score **≥ 0.945** (currently at 0.92434)

**Stretch goal:** Public score **≥ 0.950** (top 10%)

---

## 📝 Notatki

### Kluczowe obserwacje z dokumentacji:

1. **Feature Engineering > Model Tuning**
   - Tier 1 ratios (loan_to_income, payment_capacity) dają +0.03-0.05 AUC
   - Target encoding daje +0.02-0.04 AUC
   - Polynomial features mogą nie działać na syntetycznych danych

2. **AutoGluon best practices:**
   - `presets='best_quality'` z `num_stack_levels=1-2` jest optymalne
   - `hyperparameters='zeroshot'` jest już bardzo dobry, ale stacking lepszy
   - Nie używać manual hyperparameter tuning dla AutoGluon

3. **Class Imbalance:**
   - Dataset: 80% paid back, 20% default (ratio 4:1)
   - `class_weight='balanced'` jest kluczowe
   - Unikać SMOTE (ryzyko leakage)

4. **Synthetic Data Characteristics:**
   - Prostsze modele z regularyzacją często lepsze niż głębokie sieci
   - Feature engineering bardziej krytyczny niż na prawdziwych danych
   - Transfer learning może nie działać jeśli dystrybucje się różnią

### Feature Priority (z dokumentacji):

**Tier 1 (Must-have):** +0.05-0.07 AUC
- ✅ loan_to_income_ratio
- ✅ payment_income_ratio
- ✅ interest_to_income
- ✅ income_after_payment
- ✅ log(annual_income), log(loan_amount)

**Tier 2 (High value):** +0.02-0.05 AUC
- ✅ Target encoding (grade_subgrade, loan_purpose)
- ✅ Polynomial features (degree 2)
- ✅ Transfer learning features

**Tier 3 (Optional):** +0.01-0.02 AUC
- Manual ensembling
- Pseudo-labeling
- GPU acceleration (dla szybszej iteracji)

---

## 🎯 Success Criteria

**Minimum Success:**
- Przynajmniej 1 eksperyment osiąga Public ≥ 0.940 (+1.6pp boost)

**Good Success:**
- Przynajmniej 2 eksperymenty osiągają Public ≥ 0.945 (+2.1pp boost)
- Overfitting <0.01 (CV - Public)

**Excellent Success:**
- Exp 4 (Stacking) osiąga Public ≥ 0.950 (+2.6pp boost)
- Feature importance analysis pokazuje, które featury naprawdę działają
- Final long-run model (8h) osiąga Public ≥ 0.955

---

**Powodzenia! 🚀**

Pytania? Sprawdź:
- `/docs/S5E11_claude_anaylys.md` - szczegółowa analiza feature engineering
- `/docs/S5E11_gemini_anaylys.md` - strategia transfer learning
- `/docs/S5E11_chatgpt_anaylys.md` - publiczne notebooki Kaggle

# Optymalizacja S5E11 - Podsumowanie i Plan Działania

## 📊 Analiza Aktualnych Wyników

### TOP 5 Najlepszych Modeli:
1. **fe17**: 0.92356 - `log_total_income` + `EMI_estimate` ✨
2. **fe20**: 0.92353 - `is_student_flag` + `purpose_is_debt_consolidation_flag` ✨
3. **fe13**: 0.92331 - `monthly_income` + `monthly_debt`
4. **fe15**: 0.92330 - `grade_numeric` + `subgrade_numeric`  
5. **fe14**: 0.92328 - `high_DTI_flag` + `credit_score_norm`

### Kluczowe Wnioski:
- **Logarytmiczne transformacje** (fe17) dają najlepszy wynik
- **Proste flagi binarne** (fe20) są zaskakująco skuteczne
- **EMI estimate** (rata kredytu) to najsilniejszy pojedynczy feature
- Local CV ~0.91 vs Public ~0.923 = mały overfitting (dobry znak!)

## 🚀 Plan Optymalizacji (3 Kroki)

### **Krok 1: fe21 - Ultimate Feature Combination**
**Plik:** `autogluon_features_21.py`

Łączy najlepsze featury z:
- ✅ fe17: `log_total_income`, `EMI_estimate`
- ✅ fe20: `is_student_flag`, `purpose_is_debt_consolidation_flag`
- ✅ fe13: `monthly_income`, `monthly_debt`
- ✅ fe15: `grade_numeric`, `subgrade_numeric`

**Dodatkowe featury:**
- `payment_risk_ratio` = EMI / (monthly_income * 0.3)
- `risk_score_v2` = kompleksowy wskaźnik ryzyka
- `loan_burden_index` = obciążenie kredytowe
- `income_credit_power` = interakcja dochód × kredyt
- `risk_flags_sum` = suma flag ryzyka

**Konfiguracja:**
- 3 godziny treningu
- 8-fold CV
- 2-level stacking
- Excluded: NN_TORCH

**Oczekiwany wynik:** 0.926+ public score

### **Krok 2: fe22 - Target Encoding z Cross-Validation**
**Plik:** `autogluon_features_22_target_encoding.py`

Dodaje do fe21:
- **Target Encoding** z 5-fold CV (zapobiega wyciekowi)
- **Weight of Evidence** dla kategorycznych
- Smoothing Bayesowski

**Enkodowane kolumny:**
- `grade_subgrade`
- `loan_purpose` 
- `employment_status`
- `home_ownership`

**Oczekiwany wynik:** 0.927+ public score

### **Krok 3: Final Ensemble**
**Plik:** `ensemble_best_models.py`

**Weighted Average Ensemble:**
```python
weights = {
    'fe17': 0.30,  # Najlepszy pojedynczy
    'fe20': 0.25,  # 2gi najlepszy
    'fe21': 0.20,  # Ultimate combo
    'fe22': 0.15,  # Target encoded
    'original': 0.10  # Baseline best_quality
}
```

**Opcje ensemble:**
- Weighted average (domyślnie)
- Rank average
- Power average
- Isotonic calibration

**Oczekiwany wynik:** 0.930+ public score

## 📝 Instrukcja Uruchomienia

### Opcja A: Pojedynczo
```bash
# Krok 1: fe21
uv run python scripts/experiment_manager.py model \
  --project playground-series-s5e11 \
  --template best-cpu-fe21 \
  --auto-submit

# Krok 2: fe22  
uv run python scripts/experiment_manager.py model \
  --project playground-series-s5e11 \
  --template best-cpu-fe22 \
  --auto-submit

# Krok 3: Ensemble
python ensemble_best_models.py \
  --project-root /mnt/ml/kaggle-fork1/projects/kaggle/playground-series-s5e11 \
  --test-data data/test.csv \
  --output submissions/ensemble_final.csv
```

### Opcja B: Wszystko razem
```bash
bash run_optimized_experiments.sh
```

## 📈 Oczekiwane Wyniki

| Eksperyment | Local CV | Public Score | Poprawa |
|------------|----------|--------------|---------|
| Obecny best (fe17) | 0.9121 | 0.92356 | baseline |
| fe21 (ultimate) | ~0.915 | **0.926+** | +0.3% |
| fe22 (target enc) | ~0.916 | **0.927+** | +0.4% |
| Final Ensemble | ~0.918 | **0.930+** | +0.7% |

## 🎯 Dodatkowe Rekomendacje

### Jeśli wyniki < oczekiwane:
1. **Zwiększ czas treningu** do 5h (18000s)
2. **Dodaj pseudo-labeling** na test set (threshold 0.95)
3. **Użyj GPU** dla szybszej iteracji
4. **Sprawdź data leakage** - może któryś feature jest zbyt silny

### Feature Engineering do testowania:
- **Clustering features**: K-means na numeric features
- **Isolation Forest anomaly scores**
- **PCA components** top 10
- **Frequency encoding** dla rzadkich kategorii

### Alternatywne modele:
- **CatBoost** z GPU (często lepszy na categorical)
- **TabNet** (attention mechanism)
- **Neural Oblivious Decision Trees**

## ⚠️ Uwagi

1. **fe20 był zaskoczeniem** - proste flagi binarne dały 2gi najlepszy wynik!
2. **EMI_estimate** to kluczowy feature - matematycznie poprawna rata kredytu
3. **Target encoding** musi być z CV, inaczej overfitting
4. **Ensemble diversity** jest kluczowa - modele muszą się różnić

## 📊 Tracking Postępu

Po każdym eksperymencie sprawdź:
```bash
uv run python scripts/submissions_tracker.py --project playground-series-s5e11 list
```

Monitoruj live:
```bash
watch -n 30 'kaggle competitions leaderboard playground-series-s5e11 --show | head -20'
```

---

**Powodzenia! 🚀** 

Bazując na analizie, masz realną szansę na **0.930+ public score** (top 10-20% konkursu).

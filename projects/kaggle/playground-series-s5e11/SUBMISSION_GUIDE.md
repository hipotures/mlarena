# Przewodnik Submission - Wytrenowane Modele

## 📋 Sprawdź wytrenowane eksperymenty

```bash
cd /mnt/ml/kaggle-fork1/projects/kaggle/playground-series-s5e11
./list_trained_experiments.sh
```

Przykładowy output:
```
┌────────────────────┬─────────────────┬──────────┬─────────────────────────┐
│ Experiment ID      │ Template        │ Status   │ Local CV                │
├────────────────────┼─────────────────┼──────────┼─────────────────────────┤
│ exp-20251121-024914│ exp01-tier1     │ completed│ 0.93456                 │
│ exp-20251121-031527│ exp02-encoding  │ completed│ 0.94123                 │
│ exp-20251121-044312│ exp03-lgbm-opt..│ completed│ 0.93872                 │
│ exp-20251121-061845│ exp04-stacking  │ completed│ 0.94567                 │
│ exp-20251121-075234│ exp05-transfer  │ completed│ 0.93654                 │
└────────────────────┴─────────────────┴──────────┴─────────────────────────┘
```

---

## 🚀 Metoda 1: Submit pojedynczy eksperyment

### Krok 1: Znajdź experiment_id
```bash
./list_trained_experiments.sh
```

### Krok 2: Submit wybrany eksperyment
```bash
./predict_and_submit.sh [experiment_id]
```

**Przykład:**
```bash
./predict_and_submit.sh exp-20251121-044312
```

Co się dzieje:
1. ✅ Ładuje wytrenowany model
2. ✅ Generuje predykcje na test set
3. ✅ Tworzy submission file
4. ✅ Uploaduje do Kaggle
5. ✅ Czeka 45s i fetchuje public score
6. ✅ Zapisuje wynik do submissions tracker

---

## 🔄 Metoda 2: Submit wszystkie naraz

```bash
./submit_all_experiments.sh
```

**Co robi:**
- Znajduje wszystkie wytrenowane eksperymenty
- Sprawdza status (tylko `completed`)
- Submituje każdy po kolei
- Czeka między submitami (5s delay)
- Pokazuje podsumowanie

**Output:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Experiment 1: exp-20251121-024914
   Template: exp01-tier1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generating predictions...
[exp01_tier1_features] Generating predictions on 254569 samples...
[exp01_tier1_features] Predictions generated. Mean: 0.8010

✓ Submitted exp-20251121-024914

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Experiment 2: exp-20251121-031527
   Template: exp02-encoding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
...

========================================
SUMMARY
========================================
Total experiments found: 5
Successfully submitted: 5
Skipped: 0
```

---

## 📊 Sprawdź wyniki

### Lista wszystkich submissions
```bash
uv run python scripts/submissions_tracker.py --project playground-series-s5e11 list
```

### Ostatnie 10 submissions
```bash
uv run python scripts/submissions_tracker.py --project playground-series-s5e11 list | head -15
```

### Porównanie z baseline
```bash
uv run python scripts/submissions_tracker.py --project playground-series-s5e11 list | grep -E "exp0[1-5]|baseline"
```

---

## 🔍 Troubleshooting

### Problem: "No trained model found"

**Przyczyna:** Model nie został zapisany lub ścieżka jest niepoprawna

**Rozwiązanie:**
```bash
# Sprawdź czy model istnieje
ls -la projects/kaggle/playground-series-s5e11/AutogluonModels/

# Jeśli brak, trzeba przetrenować:
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --template exp01-tier1 \
    --auto-submit \
    --wait-seconds 45
```

### Problem: "Model not completed"

**Przyczyna:** Trening się nie zakończył lub błąd

**Rozwiązanie:**
```bash
# Sprawdź status
./list_trained_experiments.sh

# Jeśli status = failed, sprawdź logi:
cat projects/kaggle/playground-series-s5e11/experiments/[experiment_id]/state.json
```

### Problem: Submission fails podczas upload

**Przyczyna:** Problem z Kaggle API lub submission format

**Rozwiązanie:**
```bash
# Sprawdź czy Kaggle credentials są OK
kaggle competitions list | head

# Sprawdź czy competition jest active
kaggle competitions list | grep playground-series-s5e11

# Manual submission test (bez score fetch)
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --template exp01-tier1 \
    --experiment-id [exp_id] \
    --skip-score-fetch
```

### Problem: Score fetch timeout

**Przyczyna:** Kaggle processing jest wolny lub Chrome CDP nie działa

**Rozwiązanie:**
```bash
# Zwiększ wait time
./predict_and_submit.sh [exp_id]
# Edytuj skrypt i zmień WAIT_TIME=45 na WAIT_TIME=120

# Lub submituj bez score fetch:
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --template exp01-tier1 \
    --experiment-id [exp_id] \
    --skip-score-fetch
```

---

## 🎯 Workflow Recommendation

### Jeśli masz 5 wytrenowanych eksperymentów:

1. **Sprawdź local CV scores:**
   ```bash
   ./list_trained_experiments.sh
   ```

2. **Identify top 3 by CV:**
   - Najwyższy CV = najpewniejszy kandydat
   - Ale overfitting może być problem!

3. **Submit top 3 najpierw:**
   ```bash
   ./predict_and_submit.sh [best_exp_id]
   ./predict_and_submit.sh [second_best_id]
   ./predict_and_submit.sh [third_best_id]
   ```

4. **Czekaj ~5 min i sprawdź public scores:**
   ```bash
   uv run python scripts/submissions_tracker.py --project playground-series-s5e11 list | head -10
   ```

5. **Jeśli top 3 są obiecujące, submit resztę:**
   ```bash
   ./submit_all_experiments.sh
   ```

6. **Wybierz zwycięzcę:**
   - Najwyższy public score
   - Najmniejszy gap (Local CV - Public)
   - Ten który nie overfittuje

7. **Uruchom zwycięzcę na dłużej:**
   - Zmodyfikuj `time_limit` w `templates/model.yaml`
   - Zmień 5400 (1.5h) na 14400 (4h) lub 28800 (8h)
   - Przetrenuj ponownie

---

## 📈 Expected Results

| Eksperyment | Expected Local CV | Expected Public | Notes |
|-------------|------------------:|----------------:|-------|
| exp01-tier1 | 0.935-0.940 | 0.940-0.945 | Tier 1 features |
| exp02-encoding | 0.940-0.945 | 0.945-0.950 | +Target encoding |
| exp03-lgbm-optuna | 0.938-0.943 | 0.943-0.948 | Tuned LightGBM |
| exp04-stacking ⭐ | 0.943-0.948 | 0.947-0.952 | **BEST BET** |
| exp05-transfer | 0.937-0.942 | 0.941-0.946 | +Original data |

**Baseline:** Local 0.93309, Public 0.92434

**Target:** Public ≥ 0.945

---

## 🔧 Manual Commands

Jeśli skrypty nie działają, użyj bezpośrednio:

```bash
cd /mnt/ml/kaggle-fork1

# Submit konkretny eksperyment
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --template exp04-stacking \
    --experiment-id exp-20251121-061845 \
    --auto-submit \
    --wait-seconds 45

# Tylko predykcja (bez submit)
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --template exp04-stacking \
    --experiment-id exp-20251121-061845 \
    --skip-submit

# Submit bez score fetch (manual check later)
uv run python scripts/ml_runner.py \
    --project playground-series-s5e11 \
    --template exp04-stacking \
    --experiment-id exp-20251121-061845 \
    --auto-submit \
    --skip-score-fetch
```

---

**Good luck! 🎯**

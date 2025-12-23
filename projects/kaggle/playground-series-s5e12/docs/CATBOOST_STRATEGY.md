# CatBoost Strategy - S5E12 Diabetes

## Strategia

Implementacja strategii z notebooka "S5E12 | Single CatBoost: Train-Bin, Tail Weighting & Seed Averaging"

### Komponenty

1. **Statistical & AI Binning** (`diabetes_binning.py`)
   - Kwantylowe binsy (10 kubełków) dla numeric features
   - Decision tree binsy (depth=3) dla numeric features
   - Wszystkie binsy jako kategorie (dla CatBoost)

2. **Original Dataset Statistics** (`diabetes_orig_stats.py`)
   - Mapowanie mean/count z `diabetes_dataset.csv`
   - Agregacje per wartość kolumny
   - Dodaje `orig_mean_{col}` i `orig_count_{col}` features

3. **Tail Weighting** (`diabetes_weights.py`)
   - 5× waga dla ostatnich 3% danych (distribution shift)
   - Automatyczne wykrycie cutoff (percentile-based)
   - Zapisane jako `sample_weights.csv`

4. **CatBoost Model** (`catboost-binned.yaml`)
   - AutoGluon z CatBoost-only
   - Tuned hyperparameters z notebooka
   - Użycie sample weights + opcjonalny merge orig_df (sterowane w template przez `merge_orig`)

## Uruchomienie

### 1. Pełny pipeline (preprocessing + model + predict + submit)

```bash
cd /home/xai/ml/kaggle

# Pełny run (10 min)
uv run python scripts/mla.py -p playground-series-s5e12 \
  preprocess_template=binned-full \
  model_template=catboost-binned
```

### 2. Smoke test (tylko preprocessing, szybki check)

```bash
# Test preprocessing chain (bez modelu)
uv run python scripts/mla.py -p playground-series-s5e12 \
  preprocess preprocess_template=binned-full
```

### 3. Tylko model (jeśli preprocessing już jest)

```bash
# Użyj istniejącego preprocessingu
uv run python scripts/mla.py -p playground-series-s5e12 \
  model model_template=catboost-binned \
  preprocess_template=binned-full
```

### 4. Force re-run (jeśli coś się zmieniło)

```bash
# Wymuś ponowne uruchomienie wszystkich kroków
uv run python scripts/mla.py -p playground-series-s5e12 \
  preprocess_template=binned-full \
  model_template=catboost-binned \
  --force
```

## Monitoring

### Preprocessing output

Preprocessing chain tworzy:
```
experiments/pre-binned-full/
├── 0-external_dataset/
│   ├── artifacts/
│   │   └── orig_processed.csv         # External dataset (aligned)
│   └── state.json
├── 1-diabetes_binning/
│   ├── artifacts/
│   │   ├── binning_report.json        # Binning statistics
│   │   └── train_processed.csv        # Train + binned columns
│   └── state.json
├── 2-diabetes_orig_stats/
│   ├── artifacts/
│   │   ├── orig_stats_report.json     # Stats mapping info
│   │   └── train_processed.csv        # Train + orig stats
│   └── state.json
└── 3-diabetes_weights/
    ├── artifacts/
    │   ├── sample_weights.csv         # Tail weights (IMPORTANT!)
    │   ├── weights_report.json        # Weight statistics
    │   └── train_processed.csv        # Final preprocessed train
    └── state.json
```

### Model output

Model experiment:
```
experiments/exp-YYYYMMDD-HHMMSS/
├── artifacts/
│   ├── AutogluonModels/              # CatBoost model files
│   ├── predictions.csv               # Test predictions
│   └── leaderboard.csv               # Model performance
└── state.json                        # Experiment metadata
```

### Submission

Submission automatycznie tworzy:
```
submissions/
├── submission_YYYYMMDD_HHMMSS.csv    # Kaggle submission file
└── submissions.json                   # Tracking all submissions
```

## Oczekiwane wyniki

### Preprocessing

```
✅ External dataset loaded and aligned
✅ Binning complete: 22 new columns created
   - 11 × bin_{col}_stat (statistical)
   - 11 × bin_{col}_ai (AI thresholds)
✅ Orig stats complete: 44 new columns created
   - 22 × orig_mean_{col}
   - 22 × orig_count_{col}
✅ Tail weighting complete: percentile strategy
   Tail samples: 21,000 (3.00%)
   Weight multiplier: 5.0x
   Cutoff ID: 679,000
```

### Model

> Uwaga: merge orig_df jest opcjonalny (flaga `merge_orig` w template).
> Jeśli `merge_orig: false`, logi o "Merging external dataset" nie pojawiaja sie.

```
[AutoGluon Sample Weights] Using sample weights from artifacts
[AutoGluon Sample Weights] weight_evaluation=True

Best model: CatBoost
Local CV score: ~0.72-0.73 (ROC-AUC)
```

### Submission

```
Public LB: ~0.702-0.703 (expected based on notebook)
```

## Struktura utworzonych plików

### Preprocessory (project-specific)

```
projects/kaggle/playground-series-s5e12/code/preprocessing/
├── diabetes_binning.py       # Statistical + AI binning
├── diabetes_orig_stats.py    # Orig dataset aggregations
└── diabetes_weights.py       # Tail weighting
```

### Templates

```
projects/kaggle/playground-series-s5e12/templates/
├── preprocess/
│   └── binned-full.yaml    # 4-step preprocessing chain
└── model/
    ├── catboost-binned.yaml           # CatBoost model config
    └── catboost-binned-hpo-noorig.yaml  # CatBoost HPO without orig merge
```

## Customizacja

### Zmiana tail weighting

Edytuj `templates/preprocess/binned-full.yaml`:

```yaml
- module: diabetes_weights
  config:
    weighting_strategy: percentile
    tail_percentile: 0.95        # Zmień na 5% zamiast 3%
    weight_multiplier: 10.0      # Zwiększ wagę do 10x
```

### Zmiana liczby binsów

Edytuj `templates/preprocess/binned-full.yaml`:

```yaml
- module: diabetes_binning
  config:
    stat_quantiles: 20           # Więcej kubełków (10 → 20)
    ai_max_depth: 5              # Głębsze drzewo (3 → 5)
```

### Dodanie innych kolumn do binningu

Edytuj `templates/preprocess/binned-full.yaml`:

```yaml
- module: diabetes_binning
  config:
    binning_columns:
      - age
      - bmi
      # ... dodaj więcej kolumn
```

### Zmiana hyperparametrów CatBoost

Edytuj `templates/model/catboost-binned.yaml`:

```yaml
CAT:
  learning_rate: 0.05           # Wolniejsze uczenie
  depth: 8                       # Głębsze drzewo
  iterations: 5000               # Więcej iteracji
```

### Wylaczenie merge orig_df (zgodnie z notebookiem)

W template modelu ustaw:

```yaml
merge_orig: false
```

Przykladowy template:
- `templates/model/catboost-binned-hpo-noorig.yaml`

## Troubleshooting

### Problem: "Column 'diagnosed_diabetes' not found in orig_df"

**Rozwiązanie**: Sprawdź, czy `diabetes_dataset.csv` ma kolumnę target.

```bash
head -1 projects/kaggle/playground-series-s5e12/data/diabetes_dataset.csv
```

### Problem: "Weights row count doesn't match train row count"

**Rozwiązanie**: Upewnij się, że preprocessing chain działa sekwencyjnie (nie był przerwany).
Wymuś ponowne uruchomienie z `--force`.

### Problem: Model nie używa sample weights

**Rozwiązanie**: Sprawdź logi:
- Powinno być: `[AutoGluon Sample Weights] Using sample weights from artifacts`
- Sprawdź, czy `sample_weights.csv` istnieje w artifacts preprocessing

### Problem: Niska CV score

**Możliwe przyczyny**:
1. Brak orig_df merge → sprawdz `merge_orig` w template (jesli `false`, to jest oczekiwane)
2. Brak sample weights → sprawdź logi "Sample Weights"
3. Słabe hyperparametry → dostosuj template model

## Porównanie z notebookiem

| Feature | Notebook | Nasza implementacja |
|---------|----------|---------------------|
| Statistical binning | ✅ | ✅ |
| AI binning | ✅ | ✅ |
| Orig stats | ✅ | ✅ |
| Tail weighting | ✅ | ✅ |
| CatBoost | ✅ | ✅ (via AutoGluon) |
| Seed averaging | ✅ (3 seeds) | ❌ (single run) |
| 10-fold CV | ✅ | ✅ (AutoGluon default) |

**Uwaga**: Seed averaging (3 seeds × 10 folds = 30 modeli) nie jest zaimplementowany.
AutoGluon używa single seed. Dla seed averaging trzeba by wielokrotnie uruchomić
model i uśrednić predykcje manualnie.

## Next Steps

1. **Baseline run**: Uruchom pełny pipeline i sprawdź CV score
2. **Hyperparameter tuning**: Dostosuj parametry CatBoost jeśli potrzeba
3. **Feature engineering**: Dodaj więcej kolumn do binningu
4. **Seed averaging**: Jeśli potrzeba, zaimplementuj custom model z seed averaging

## Credits

Strategia bazuje na:
- **Notebook**: S5E12 | Single CatBoost: Train-Bin, Tail Weighting & Seed Averaging (v3)
- **Author**: (Original Kaggle notebook author)
- **Competition**: Playground Series S5E12 - Diabetes Prediction

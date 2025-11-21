# 🚀 Quick Submit Guide

## Masz wytrenowane modele? Submituj je teraz!

### Krok 1: Zobacz co masz
```bash
cd /mnt/ml/kaggle-fork1/projects/kaggle/playground-series-s5e11
./list_trained_experiments.sh
```

### Krok 2: Submit wszystko
```bash
./submit_all_experiments.sh
```

### Krok 3: Sprawdź wyniki (~5 min później)
```bash
uv run python scripts/submissions_tracker.py --project playground-series-s5e11 list | head -10
```

---

## Lub submituj pojedynczo:

```bash
# 1. Sprawdź ID eksperymentu
./list_trained_experiments.sh

# 2. Submit wybranego
./predict_and_submit.sh [experiment_id]

# Przykład:
./predict_and_submit.sh exp-20251121-024914
```

---

## Co jeśli nie masz wytrenowanych modeli?

### Uruchom wszystkie 5 eksperymentów (~6-8h):
```bash
./run_experiments.sh
```

### Lub pojedynczo (~1.5h każdy):
```bash
cd /mnt/ml/kaggle-fork1

# Eksperyment 1
uv run python scripts/ml_runner.py --project playground-series-s5e11 --template exp01-tier1 --auto-submit --wait-seconds 45

# Eksperyment 2
uv run python scripts/ml_runner.py --project playground-series-s5e11 --template exp02-encoding --auto-submit --wait-seconds 45

# Eksperyment 3
uv run python scripts/ml_runner.py --project playground-series-s5e11 --template exp03-lgbm-optuna --auto-submit --wait-seconds 45

# Eksperyment 4 ⭐ (BEST)
uv run python scripts/ml_runner.py --project playground-series-s5e11 --template exp04-stacking --auto-submit --wait-seconds 45

# Eksperyment 5
uv run python scripts/ml_runner.py --project playground-series-s5e11 --template exp05-transfer --auto-submit --wait-seconds 45
```

---

**Więcej info:** Przeczytaj `SUBMISSION_GUIDE.md`

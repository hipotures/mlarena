# Competition Tools

Centralne, uniwersalne narzędzia dla wszystkich konkursów Kaggle.

## Zawartość

### `submissions_tracker.py`
System śledzenia submissions z local CV, public i private scores.

**Użycie z linii komend:**

```bash
# Dodaj nową submission
python tools/submissions_tracker.py add \
    --project playground-series-s5e11 \
    submission-20231116-model-v1.csv \
    autogluon-medium \
    --local-cv 0.85432 \
    --cv-std 0.00123 \
    --notes "Initial baseline"

# Zaktualizuj wyniki z leaderboard
python tools/submissions_tracker.py update \
    --project playground-series-s5e11 \
    1 \
    --public 0.85123 \
    --private 0.84987

# Wyświetl listę submissions
python tools/submissions_tracker.py list \
    --project playground-series-s5e11 \
    --sort-by public_score \
    --limit 10

# Eksportuj do CSV
python tools/submissions_tracker.py export \
    --project playground-series-s5e11
```

**Użycie w kodzie:**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

from submissions_tracker import SubmissionsTracker

# Inicjalizacja
project_root = Path(__file__).parent.parent.parent
tracker = SubmissionsTracker(project_root)

# Dodaj submission
tracker.add_submission(
    filename="submission-20231116.csv",
    model_name="autogluon-medium",
    local_cv_score=0.85432,
    cv_std=0.00123,
    notes="Baseline model with default params"
)

# Zaktualizuj scores
tracker.update_scores(
    submission_id=1,
    public_score=0.85123,
    private_score=0.84987
)

# Wyświetl top 5 submissions
tracker.display_submissions(limit=5, sort_by="public_score")

# Pobierz best submissions
best = tracker.get_best_submissions(metric="public_score", top_n=3)
```

## Integracja z projektem

Narzędzia automatycznie integrują się z funkcją `create_submission()` w każdym projekcie:

```python
from code.utils.submission import create_submission

# Submission zostanie automatycznie dodana do trackera
create_submission(
    predictions=y_pred,
    test_ids=test_df['id'],
    model_name="lgbm-v2",
    local_cv_score=0.85432,
    cv_std=0.00123,
    notes="LGBM with tuned hyperparameters",
    config={"learning_rate": 0.01, "n_estimators": 1000}
)
```

## Format danych

Tracker przechowuje dane w `[projekt]/submissions/submissions.json`:

```json
[
  {
    "id": 1,
    "timestamp": "2023-11-16 23:30:00",
    "filename": "submission-20231116233000.csv",
    "model_name": "autogluon-medium",
    "local_cv_score": 0.85432,
    "cv_std": 0.00123,
    "public_score": 0.85123,
    "private_score": null,
    "notes": "Baseline model",
    "config": {
      "preset": "medium_quality",
      "time_limit": 3600
    }
  }
]
```

## Przydatne funkcje

### Porównanie local CV vs public score
```python
tracker.display_submissions(sort_by="local_cv_score")
# Zobacz, czy local CV koreluje z public score
```

### Export do analizy
```python
tracker.export_to_csv()
# Otwórz submissions_tracking.csv w Excel/Pandas do analizy
```

### Znajdź best submission
```python
best_local = tracker.get_best_submissions("local_cv_score", top_n=1)[0]
best_public = tracker.get_best_submissions("public_score", top_n=1)[0]

if best_local['id'] != best_public['id']:
    print("⚠️  Local CV nie koreluje z public score!")
```

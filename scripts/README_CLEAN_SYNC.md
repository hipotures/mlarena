# Skrypty czyszczące i synchronizujące

Dwa narzędzia do zarządzania artefaktami projektów Kaggle.

## clean.py - Czyszczenie artefaktów

Usuwa duże artefakty AutoGluon zachowując dane potrzebne do reprodukcji eksperymentów.

### Co jest usuwane

- **Katalogi AutoGluon** (identyfikowane po `predictor.pkl`/`learner.pkl`):
  - `model/`, `AutogluonModels/`, `av_model/`, etc.
  - Zawartość: `models/`, `utils/`, `predictor.pkl`, `learner.pkl`
- **Pliki tymczasowe**: `*.lock`, `state.lock`
- **Cache Python**: `__pycache__/`
- **Opcjonalnie** (z `--remove-processed-csv`): `*_processed.csv`

### Co jest ZACHOWANE

- `state.json` - metadane eksperymentów
- `code_snapshot/` - snapshoty kodu
- `leaderboard.csv` - wyniki modeli (nie ma w state.json)
- `train_used.csv` - dane referencyjne
- `submission.csv` - predykcje
- `metadata.json`, `version.txt` - metadane AutoGluon
- `data/` - surowe dane

### Użycie

```bash
# Zobacz co zostanie usunięte (dry run)
python scripts/clean.py --project Titanic --dry-run

# Usuń artefakty AutoGluon (zachowaj processed CSV)
python scripts/clean.py --project Titanic

# Usuń artefakty + processed CSV
python scripts/clean.py --project Titanic --remove-processed-csv

# Pomoc
python scripts/clean.py --help
```

### Przykładowy output

```
🔍 DRY RUN - Pliki do usunięcia w projekcie: Titanic

┌────────────────────────────────────────────┬──────────┬──────────┐
│ Ścieżka                                    │ Typ      │ Rozmiar  │
├────────────────────────────────────────────┼──────────┼──────────┤
│ experiments/exp-20251214-110804/.../model/ │ model/   │ 193.4 MB │
│ experiments/exp-20251214-024605/.../model/ │ model/   │  31.1 MB │
│ ...                                        │ ...      │ ...      │
└────────────────────────────────────────────┴──────────┴──────────┘

Podsumowanie:
  Katalogów:            4
  Plików:               19
  Całkowity rozmiar:    232.3 MB
```

---

## sync.py - Synchronizacja projektów

Kopiuje projekty między lokalizacjami używając `rsync`, pomijając artefakty i cache.

### Co jest kopiowane

- `code/` - cały kod projektu
- `templates/` - wszystkie templaty
- `data/*.csv` - surowe dane
- `experiments/*/state.json` - metadane
- `experiments/*/code_snapshot/` - snapshoty kodu
- `experiments/*/artifacts/model/leaderboard.csv`
- `experiments/*/artifacts/predict/submission.csv`
- `experiments/pre-*/artifacts/preprocess/*.csv` - przetworzone dane
- `submissions/submissions.json` - tracking submisji
- `docs/`, `*.md` - dokumentacja

### Co jest POMIJANE

- `.git/` - repozytorium git
- `.venv/`, `venv/` - virtual env
- `**/AutogluonModels/`, `**/model/` (katalogi z predictor.pkl) - artefakty
- `**/__pycache__/`, `*.lock` - cache i temp
- `*.egg-info/`, `.pytest_cache/` - metadata

### Użycie

```bash
# Dry run - zobacz co zostanie skopiowane
python scripts/sync.py \
  projects/kaggle/Titanic \
  /mnt/backup/kaggle/Titanic \
  --dry-run

# Synchronizuj pojedynczy projekt
python scripts/sync.py \
  projects/kaggle/Titanic \
  /mnt/backup/kaggle/Titanic

# Synchronizuj wszystkie projekty
python scripts/sync.py \
  projects/kaggle \
  /mnt/backup/kaggle

# Pomoc
python scripts/sync.py --help
```

### Wymagania

- `rsync` musi być zainstalowany:
  ```bash
  # Ubuntu/Debian
  sudo apt install rsync

  # Arch/Manjaro
  sudo pacman -S rsync

  # macOS
  brew install rsync
  ```

### Przykładowy output

```
🔍 DRY RUN - Synchronizacja

Źródło:      /home/xai/ml/kaggle/projects/kaggle/Titanic
Cel:         /mnt/backup/kaggle/Titanic

Wykluczone (nie będą kopiowane):
  - .git/, .venv/
  - AutogluonModels/
  - __pycache__/, *.lock
  - *.egg-info/, cache/

[rsync output...]
```

---

## Przykładowy workflow

1. **Wyczyść projekt przed backupem:**
   ```bash
   python scripts/clean.py --project playground-series-s5e12 --dry-run
   python scripts/clean.py --project playground-series-s5e12
   ```

2. **Zsynchronizuj do innej lokalizacji:**
   ```bash
   python scripts/sync.py \
     projects/kaggle/playground-series-s5e12 \
     /mnt/backup/kaggle/playground-series-s5e12
   ```

3. **Wynik:** Backup z pełnym kodem + state.json, bez ~200GB artefaktów AutoGluon

---

## Bezpieczeństwo

- **clean.py**: Zawsze pyta o potwierdzenie przed usunięciem (chyba że `--dry-run`)
- **sync.py**: W trybie dry-run pokazuje co zostanie skopiowane
- Oba skrypty walidują ścieżki przed operacjami

## Oszczędność miejsca

Przykładowe oszczędności dla różnych projektów:

| Projekt              | Przed   | Po czyszczeniu | Oszczędność |
|----------------------|---------|----------------|-------------|
| Titanic              | 232 MB  | ~5 MB          | ~227 MB     |
| playground-s5e12     | 204 GB  | ~2-5 GB        | ~200 GB     |
| playground-s5e6      | 3.7 GB  | ~100 MB        | ~3.6 GB     |

**Uwaga:** Pełna reprodukcja eksperymentu wymaga przeliczenia modeli (może zająć od minut do godzin w zależności od `time_limit`).

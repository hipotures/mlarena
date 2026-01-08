# Przewodnik integracji: Categorical types w preprocessing chain

## Problem

Gdy używasz `--preprocess-template categorical_boost,av_weights`, typ `category` jest **tracony** przy zapisie do CSV:

```python
# categorical_boost:
df['alcohol_consumption'] = df['alcohol_consumption'].astype('category')
df.to_csv('train_processed.csv')  # ← traci dtype='category'

# av_weights czyta:
df = pd.read_csv('train_processed.csv')  # ← wraca do int/object
# AV trenuje z błędnymi typami!
```

## Rozwiązanie

Używaj `categorical_utils.py` aby przywrócić typy kategoryczne w kolejnych krokach chain.

---

## Integracja z istniejącymi modułami preprocessing

### Przykład: av_weights.py

**Przed:**
```python
def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict[str, Any]]:
    # ... existing code ...

    av_df = compute_adversarial_weights(
        train_df=train_df,  # ← Błędne typy (int zamiast category)
        test_df=test_df,
        # ...
    )
```

**Po:**
```python
# Dodaj import na górze pliku
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "config" / "code" / "preprocessing"))
from categorical_utils import restore_categorical_types_from_chain

def fit_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict[str, Any]]:
    # DODAJ NA POCZĄTKU FUNKCJI:
    # Przywróć typy kategoryczne z poprzedniego kroku (jeśli są)
    train_df, test_df = restore_categorical_types_from_chain(train_df, test_df, config)

    # ... reszta kodu bez zmian ...

    av_df = compute_adversarial_weights(
        train_df=train_df,  # ← Teraz ma poprawne typy!
        test_df=test_df,
        # ...
    )
```

---

## Co robi `restore_categorical_types_from_chain()`?

1. **Czyta state.json** z poprzedniego kroku (np. `experiments/pre-categorical_boost/state.json`)
2. **Wyciąga `categorical_columns`** z `custom_module_state`
3. **Konwertuje** te kolumny na `dtype='category'` w train i test
4. **Loguje** do konsoli ile kolumn skonwertowano

**Output:**
```
✓ Loaded 4 categorical columns from categorical_boost
  Columns: alcohol_consumption_per_week, family_history_diabetes, ... (+2 more)
✓ Converted 4 columns to category dtype in train
✓ Converted 4 columns to category dtype in test
```

---

## Fallback behavior

- Jeśli poprzedni krok NIE ma `categorical_columns` → nic się nie dzieje (backward compatible)
- Jeśli state.json nie istnieje → nic się nie dzieje
- Jeśli kolumna nie istnieje w DataFrame → skip z warningiem
- Jeśli kolumna już jest `category` → skip (nie konwertuj ponownie)

---

## Integracja z innymi modułami

### Przykład: feature_interactions.py

```python
from categorical_utils import restore_categorical_types_from_chain

def fit_transform(train_df, val_df, test_df, config):
    # Przywróć typy kategoryczne na początku
    train_df, test_df = restore_categorical_types_from_chain(train_df, test_df, config)

    # Teraz feature engineering z poprawnymi typami
    # ...

    return train_df, val_df, test_df, state
```

### Przykład: custom_preprocessing.py

```python
from categorical_utils import (
    read_categorical_columns_from_previous_step,
    apply_categorical_dtypes
)

def fit_transform(train_df, val_df, test_df, config):
    # Opcja 1: Użyj helper function
    train_df, test_df = restore_categorical_types_from_chain(train_df, test_df, config)

    # Opcja 2: Ręczna kontrola
    cat_cols = read_categorical_columns_from_previous_step(config)
    if cat_cols:
        train_df = apply_categorical_dtypes(train_df, cat_cols, "train")
        test_df = apply_categorical_dtypes(test_df, cat_cols, "test")
        if val_df is not None:
            val_df = apply_categorical_dtypes(val_df, cat_cols, "validation")

    # ... reszta kodu ...
```

---

## Wymagania

**config dict musi zawierać:**
```python
config = {
    "_system": {
        "project_root": Path("/path/to/project"),  # WYMAGANE
    },
    "input_source": "categorical_boost",  # OPCJONALNE (auto-detected w chain)
}
```

**MLArena automatycznie ustawia:**
- `_system.project_root` - zawsze dostępne
- `input_source` - automatycznie w chain (categorical_boost → av_weights)

---

## Testowanie

```bash
# Test chain bez categorical (jak dotychczas):
uv run python scripts/mla.py preprocess \
  --project playground-series-s5e12 \
  preprocess_template=av_weights_best_boost \
  --force

# Output: Moduł działa normalnie (brak categorical metadata)

# Test chain Z categorical:
uv run python scripts/mla.py preprocess \
  --project playground-series-s5e12 \
  preprocess_template=categorical_boost,av_weights_best_boost \
  --force

# Output:
# categorical_boost:
#   ✓ Found 4 numeric categorical columns
#   ✓ Converted 4 columns to category dtype
#
# av_weights_best_boost:
#   ✓ Loaded 4 categorical columns from categorical_boost  ← NOWE!
#   ✓ Converted 4 columns to category dtype in train       ← NOWE!
#   ✓ Converted 4 columns to category dtype in test        ← NOWE!
#   [AutoGluon output pokazuje poprawne typy]
```

---

## Weryfikacja poprawności

Po integracji, sprawdź output AutoGluon w av_weights:

**Przed (błędne typy):**
```
Types of features in processed data:
    ('int', [])    : 13 | ['alcohol_consumption_per_week', 'cardiovascular_history', ...]
```

**Po (poprawne typy):**
```
Types of features in processed data:
    ('int', [])       : 10 | ['age', 'alcohol_consumption_per_week', ...]
    ('int', ['bool']) :  3 | ['cardiovascular_history', 'family_history_diabetes', ...]
    ('category', [])  :  6 | ['gender', 'ethnicity', ...]
```

Albo jeszcze lepiej (jeśli AG rozpozna category):
```
Types of features in processed data:
    ('category', [])  : 10 | ['alcohol_consumption_per_week', 'cardiovascular_history', ...]
```

---

## Migration checklist

- [ ] Dodaj import `categorical_utils` do modułu preprocessing
- [ ] Dodaj `restore_categorical_types_from_chain()` na początku `fit_transform()`
- [ ] Przetestuj chain bez categorical (backward compatibility)
- [ ] Przetestuj chain z categorical (nowa funkcjonalność)
- [ ] Sprawdź output AutoGluon (czy typy się zgadzają)
- [ ] Porównaj wyniki AV weights przed/po (opcjonalne)

---

## FAQ

**Q: Czy to spowolni preprocessing?**
A: Nie, konwersja na category jest bardzo szybka (<1s dla 700k rows).

**Q: Czy muszę to dodać do KAŻDEGO preprocessing module?**
A: Tylko do tych, które używasz w chain PO categorical_boost (np. av_weights, feature_interactions).

**Q: Co jeśli zapominam dodać do jednego modułu?**
A: Moduł będzie działał jak dotychczas (backward compatible), ale z błędnymi typami.

**Q: Czy mogę użyć tego w modelach (nie preprocessing)?**
A: Tak! Funkcje działają wszędzie gdzie masz DataFrame + config dict.

---

## Przykładowy diff dla av_weights.py

```diff
+# Add at top of file
+import sys
+from pathlib import Path
+sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "config" / "code" / "preprocessing"))
+from categorical_utils import restore_categorical_types_from_chain

 def fit_transform(
     train_df: pd.DataFrame,
     val_df: pd.DataFrame | None,
     test_df: pd.DataFrame,
     config: Dict[str, Any],
 ) -> Tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, Dict[str, Any]]:
+    # Restore categorical dtypes from previous preprocessing step (if any)
+    train_df, test_df = restore_categorical_types_from_chain(train_df, test_df, config)
+
     cfg = config or {}
     artifact_base = Path(cfg.get("_artifact_dir") or PROJECT_ROOT)
     # ... rest unchanged ...
```

**Tylko 3 linijki kodu!**

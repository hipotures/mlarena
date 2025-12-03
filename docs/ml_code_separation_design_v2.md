# Architektura separacji kodu ML od infrastruktury v2

## 1. Problem i Cel

Obecny kod ML jest "zanieczyszczony" infrastrukturą (logging, tracking, snapshoting), co sprawia, że jest nieczytelny, trudny w modyfikacji i wymaga dużo boilerplate'u.

**Cel:** Umożliwić pisanie czystego kodu ML, który automatycznie integruje się z istniejącym systemem eksperymentów, bez konieczności martwienia się o infrastrukturę.

## 2. Proponowana Architektura v2

### 2.1. Struktura Plików

Struktura zostaje rozszerzona o dedykowany katalog na konfiguracje.

```
projects/kaggle/[competition]/
├── code/
│   ├── models/           # Czyste modele ML (jeden plik = jeden model)
│   │   ├── autogluon.py
│   │   └── xgboost_custom.py
│   └── preprocessing/    # Wspólne funkcje do preprocessingu
│       └── feature_engineering.py
├── configs/
│   └── templates.yaml    # Definicje template'ów i ich konfiguracji
└── experiments/          # Bez zmian
```
Katalog `code/preprocessing/` służy jako biblioteka funkcji pomocniczych, które mogą być importowane przez poszczególne modele.

### 2.2. Kontrakt i Konfiguracja Modelu

Aby zapewnić walidację i wsparcie dla narzędzi statycznych, odchodzimy od luźnego słownika `config` na rzecz bardziej ustrukturyzowanego podejścia.

#### 2.2.1. Hierarchia Konfiguracji

Parametry dla uruchomienia modelu będą ładowane i scalane w następującej kolejności (każdy kolejny krok nadpisuje poprzedni):
1.  **Wartości domyślne** zdefiniowane w modelu (`get_default_config()`).
2.  **Konfiguracja globalna projektu** (opcjonalny plik `configs/project.yaml`).
3.  **Konfiguracja z `templates.yaml`** dla wybranego szablonu.
4.  **Parametry z linii komend (CLI)**, np. `--config.time_limit=300`.

#### 2.2.2. Struktura Konfiguracji

Zaleca się stosowanie `Pydantic` lub `dataclasses` do definiowania struktury konfiguracji, co umożliwi walidację i autouzupełnianie. Runner będzie przekazywał do modelu obiekt konfiguracyjny, a nie słownik.

```python
# Przykład użycia Pydantic do zdefiniowania struktury
from pydantic import BaseModel, Field
from typing import List, Optional

class SystemConfig(BaseModel):
    model_path: str
    
class DatasetConfig(BaseModel):
    target: str
    id_column: str
    
class Hyperparameters(BaseModel):
    presets: str = 'best_quality'
    time_limit: int = 3600
    excluded_models: Optional[List[str]] = None

class ModelConfig(BaseModel):
    system: SystemConfig
    dataset: DatasetConfig
    hyperparameters: Hyperparameters
```

### 2.3. Interfejs Modelu

Każdy plik w `code/models/` powinien eksportować funkcje zgodne z poniższym, rozszerzonym interfejsem.

```python
# code/models/autogluon.py
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# --- Nowy, opcjonalny interfejs dla stanowego preprocessingu ---
def prepare_artifacts(
    train_df: pd.DataFrame, 
    config: ModelConfig
) -> Any:
    """
    (Opcjonalne) Przygotowuje i zwraca "artefakty" preprocessingu, 
    które wymagają nauczenia, np. scaler, encoder.
    """
    from sklearn.preprocessing import StandardScaler
    
    features_to_scale = [...]
    scaler = StandardScaler()
    scaler.fit(train_df[features_to_scale])
    return scaler

# --- Zaktualizowany interfejs ---
def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None
) -> Any:
    """
    Trenuje model.
    
    Args:
        ...
        artifacts: Obiekt zwrócony przez prepare_artifacts().
    
    Returns:
        Wytrenowany model.
    """
    if artifacts: # artifacts to np. nasz scaler
        train_df[features] = artifacts.transform(train_df[features])
        if val_df:
            val_df[features] = artifacts.transform(val_df[features])
            
    # Logika treningu...
    predictor = TabularPredictor(...)
    predictor.fit(train_df, tuning_data=val_df)
    return predictor

def predict(
    model: Any,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None
) -> pd.DataFrame:
    """
    Generuje predykcje.
    
    Returns:
        DataFrame z predykcjami. Jego struktura powinna być zgodna
        z sample_submission.csv.
    """
    if artifacts:
        test_df[features] = artifacts.transform(test_df[features])
        
    # Logika predykcji...
    predictions = model.predict_proba(test_df)
    
    # Tworzenie finalnego DataFrame'a
    submission_df = pd.DataFrame()
    submission_df[config.dataset.id_column] = test_df[config.dataset.id_column]
    
    # Obsługa predykcji jedno- i wielokolumnowych
    if isinstance(predictions, pd.Series):
        submission_df[config.dataset.target] = predictions
    else: # np. DataFrame dla multi-output
        submission_df[predictions.columns] = predictions
        
    return submission_df

# --- Bez zmian ---
def get_default_config() -> Dict[str, Any]:
    """Zwraca domyślne parametry dla tego modelu."""
    return {
        'hyperparameters': {
            'presets': 'best_quality',
            'time_limit': 3600
        }
    }
```

### 2.4. Uniwersalny Runner (`ml_runner.py`)

Runner zostanie zaktualizowany, aby obsłużyć nowy kontrakt:

1.  **Ładowanie Konfiguracji:** Wczyta i scali konfiguracje zgodnie z zdefiniowaną hierarchią. Przeprowadzi walidację (np. przy użyciu Pydantic).
2.  **Ładowanie Danych:** Domyślnie wczyta dane. Jeśli model eksportuje funkcję `load_data(config)`, runner użyje jej, pozwalając na niestandardowe źródła danych.
3.  **Artefakty Preprocessingu:** Przed treningiem sprawdzi, czy model posiada funkcję `prepare_artifacts()`. Jeśli tak, uruchomi ją i przekaże zwrócony obiekt do `train()` i `predict()`.
4.  **Trening i Predykcja:** Wywoła `train()` i `predict()`, przekazując obiekt konfiguracyjny i artefakty.
5.  **Obsługa Predykcji:** Zamiast zakładać stałą strukturę wyjściową, runner może (opcjonalnie) zweryfikować, czy kolumny w zwróconym przez `predict()` DataFrame pasują do `sample_submission.csv`.
6.  **Infrastruktura:** Bez zmian – nadal będzie odpowiedzialny za logowanie, snapshotowanie, śledzenie eksperymentów.
7.  **Override modelu z CLI:** Dodatkowa flaga `--model-name` pozwala podmienić moduł modelu wskazany w template (dotyczy train/all; stage=predict korzysta z modelu zapisanego w state). Dzięki temu można użyć tych samych parametrów template’u z innym plikiem modelu bez duplikowania wpisów w `templates/model.yaml`.

### 2.5. Migracja

Plan pozostaje stopniowy, ale zostaje wzbogacony o narzędzie automatyzujące:
1.  **Współistnienie:** Stary `autogluon_runner.py` i nowy `ml_runner.py` mogą działać równolegle. `experiment_manager` decyduje, którego użyć (np. na podstawie struktury projektu lub flagi).
2.  **Narzędzie do Migracji:** Stworzony zostanie skrypt, który potrafi "opakować" istniejący, prosty skrypt treningowy w nowy format `code/models/*.py`, generując podstawowe funkcje `train`/`predict`. To znacznie przyspieszy migrację.

## 3. Podsumowanie Zmian w v2

-   **Strukturalizacja Konfiguracji:** Wprowadzenie hierarchii ładowania i walidacji `config` (np. przez Pydantic), co zwiększa solidność.
-   **Obsługa Stanowego Preprocessingu:** Dodanie opcjonalnego kroku `prepare_artifacts`, co rozwiązuje problem przekazywania "nauczonych" obiektów (np. scalerów).
-   **Elastyczność Predykcji:** Model `predict` zwraca kompletny DataFrame, co pozwala na obsługę konkursów z wieloma kolumnami wyjściowymi.
-   **Automatyzacja Migracji:** Propozycja stworzenia narzędzia do szybszego przenoszenia istniejących modeli do nowej architektury.

Te zmiany adresują zidentyfikowane luki, czyniąc architekturę bardziej kompletną, elastyczną i gotową na bardziej złożone zadania, przy jednoczesnym zachowaniu pierwotnego celu – czystości kodu ML.

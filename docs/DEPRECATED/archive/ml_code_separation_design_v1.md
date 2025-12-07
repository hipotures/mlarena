# Architektura separacji kodu ML od infrastruktury

## Problem do rozwiązania

Obecny kod AutoGluon jest "zanieczyszczony" infrastrukturą (logging, tracking, snapshoting), co sprawia, że:
- Trudno zobaczyć faktyczną logikę ML
- Kod jest nieczytelny i trudny w modyfikacji
- Zbyt dużo boilerplate'u dla prostych eksperymentów

## Cel

Umożliwić pisanie czystego kodu ML, który automatycznie integruje się z istniejącym systemem eksperymentów, bez konieczności martwienia się o infrastrukturę.

## Proponowana architektura

### 1. Struktura plików

```
projects/kaggle/[competition]/
├── code/
│   ├── models/           # Czyste modele ML
│   │   ├── autogluon.py
│   │   ├── xgboost_custom.py
│   │   └── ensemble.py
│   └── preprocessing/    # Opcjonalne moduły pomocnicze
│       └── features.py
├── templates/
│   ├── model.yaml        # Szablony modeli (łączy z code/models/*.py)
│   └── preprocess.yaml   # Szablony preprocessing (łączy z code/preprocessing/*.py)
└── experiments/          # Bez zmian
```

### 2. Format czystego kodu ML

Każdy plik modelu eksportuje standardowy interfejs:

```python
# code/models/autogluon.py

from autogluon.tabular import TabularPredictor
import pandas as pd
from typing import Dict, Any, Optional

# Obowiązkowe funkcje
def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: Dict[str, Any]
) -> Any:
    """
    Args:
        train_df: DataFrame z danymi treningowymi
        val_df: DataFrame z danymi walidacyjnymi (może być None)
        config: Słownik z całą konfiguracją (target, features, hyperparameters, etc.)
    
    Returns:
        model: Wytrenowany model (dowolny typ)
    """
    predictor = TabularPredictor(
        label=config['target'],
        eval_metric=config.get('metric', 'roc_auc'),
        path=config['model_path']  # System zapewni unikalną ścieżkę
    )
    
    predictor.fit(
        train_df,
        val_df=val_df,
        presets=config.get('presets', 'best_quality'),
        time_limit=config.get('time_limit', 3600),
        excluded_model_types=config.get('excluded_models', [])
    )
    
    return predictor

def predict(
    model: Any,
    test_df: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Args:
        model: Model zwrócony przez train()
        test_df: DataFrame z danymi testowymi
        config: Ten sam config co w train()
    
    Returns:
        predictions: DataFrame z kolumnami wymaganymi przez competition
    """
    predictions = pd.DataFrame()
    predictions['id'] = test_df['id']
    
    if config.get('predict_proba', True):
        proba = model.predict_proba(test_df)
        predictions[config['target']] = proba.iloc[:, 1]
    else:
        predictions[config['target']] = model.predict(test_df)
    
    return predictions

# Opcjonalne funkcje
def preprocess(df: pd.DataFrame, config: Dict[str, Any], is_train: bool = True) -> pd.DataFrame:
    """Opcjonalny preprocessing specyficzny dla modelu"""
    # Tu możesz robić feature engineering
    df = df.copy()
    
    # Przykład: drop high cardinality columns
    if config.get('drop_high_cardinality', True):
        for col in df.select_dtypes(['object']).columns:
            if df[col].nunique() > 100:
                df = df.drop(col, axis=1)
    
    return df

def get_default_config() -> Dict[str, Any]:
    """Opcjonalne domyślne parametry"""
    return {
        'presets': 'best_quality',
        'time_limit': 3600,
        'excluded_models': ['NN_TORCH'],
        'drop_high_cardinality': True
    }
```

### 3. Konfiguracja template'ów

Template określa, który model uruchomić:

```yaml
# templates/model.yaml

templates:
  fast-cpu:
    model: "autogluon"  # nazwa pliku w code/models/
    config:
      presets: "good_quality" 
      time_limit: 600
      excluded_models: ["NN_TORCH"]
  
  time8-gpu:
    model: "autogluon"
    config:
      presets: "best_quality"
      time_limit: 28800  # 8 hours
      excluded_models: []
      
  xgboost-custom:
    model: "xgboost_custom"  # code/models/xgboost_custom.py
    config:
      n_estimators: 1000
      learning_rate: 0.01
      
  ensemble:
    model: "ensemble"
    config:
      models: ["autogluon", "xgboost_custom"]
      weights: [0.7, 0.3]
```

### 4. Uniwersalny runner

Runner zajmuje się całą infrastrukturą:

```python
# scripts/ml_runner.py

class MLRunner:
    def __init__(self, project: str, template: str):
        self.context = self._setup_context(project, template)
        self.model_module = self._load_model_module()
        
    def _load_model_module(self):
        """Dynamicznie ładuje moduł z code/models/"""
        model_name = self.context.template_config['model']
        model_path = f"projects/{self.context.project}/code/models/{model_name}.py"
        
        # Importuj moduł dynamicznie
        spec = importlib.util.spec_from_file_location(model_name, model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
        
    def run(self):
        """Główna pętla z całą infrastrukturą"""
        # Setup (logging, tracking, etc.)
        self._setup_experiment()
        
        try:
            # Wczytaj dane
            train_df, val_df, test_df = self._load_data()
            
            # Opcjonalny preprocessing
            if hasattr(self.model_module, 'preprocess'):
                train_df = self.model_module.preprocess(train_df, self.config, is_train=True)
                if val_df is not None:
                    val_df = self.model_module.preprocess(val_df, self.config, is_train=False)
                test_df = self.model_module.preprocess(test_df, self.config, is_train=False)
            
            # Trenuj model (czysty kod ML)
            model = self.model_module.train(train_df, val_df, self.config)
            
            # Predykcja
            predictions = self.model_module.predict(model, test_df, self.config)
            
            # Zapisz wyniki
            self._save_results(predictions)
            
            # Tracking, logging, etc.
            self._log_metrics()
            
        except Exception as e:
            self._handle_error(e)
        finally:
            self._cleanup()
```

### 5. Uruchamianie

Bez zmian w sposobie uruchamiania:

```bash
uv run python scripts/experiment_manager.py model \
    --template time8-gpu \
    --project playground-series-s5e11
```

Experiment manager:
1. Odczytuje template z `templates/model.yaml`
2. Uruchamia `ml_runner.py` z odpowiednimi parametrami
3. Runner dynamicznie ładuje kod z `code/models/{model_name}.py`
4. Snapshuje cały katalog `code/` do `experiments/exp-{timestamp}/code_snapshot/`

### 6. Zalety rozwiązania

**Dla użytkownika:**
- ✅ Czysty, czytelny kod ML bez infrastruktury
- ✅ Pełna kontrola nad preprocessingiem i feature engineering
- ✅ Łatwe dodawanie nowych modeli
- ✅ Ten sam sposób uruchamiania co teraz

**Dla systemu:**
- ✅ Pełna reprodukowalność (snapshot całego `code/`)
- ✅ Zachowane wszystkie funkcje (logging, tracking, etc.)
- ✅ Łatwe rozszerzanie o nowe features
- ✅ Backward compatibility z istniejącym kodem

### 7. Migracja istniejącego kodu

#### Krok 1: Wydziel czysty kod ML
Przekształć `autogluon_runner.py` w:
- `code/models/autogluon.py` (czysty kod ML)
- `scripts/ml_runner.py` (infrastruktura)

#### Krok 2: Zaktualizuj experiment_manager
Dodaj logikę wyboru między starym `autogluon_runner.py` a nowym `ml_runner.py`

#### Krok 3: Stopniowa migracja
- Nowe eksperymenty używają nowej architektury
- Stare eksperymenty dalej działają

### 8. Przykłady użycia

#### Prosty model XGBoost

```python
# code/models/xgboost_simple.py
import xgboost as xgb

def train(train_df, val_df, config):
    X_train = train_df.drop(columns=[config['target']])
    y_train = train_df[config['target']]
    
    model = xgb.XGBClassifier(
        n_estimators=config.get('n_estimators', 100),
        max_depth=config.get('max_depth', 6)
    )
    
    eval_set = [(X_train, y_train)]
    if val_df is not None:
        X_val = val_df.drop(columns=[config['target']])
        y_val = val_df[config['target']]
        eval_set.append((X_val, y_val))
    
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model

def predict(model, test_df, config):
    predictions = pd.DataFrame()
    predictions['id'] = test_df['id']
    predictions[config['target']] = model.predict_proba(test_df)[:, 1]
    return predictions
```

#### Model z custom preprocessingiem

```python
# code/models/neural_net.py
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def preprocess(df, config, is_train=True):
    # Custom feature engineering
    df['ratio_1'] = df['feature_1'] / (df['feature_2'] + 1)
    df['log_feature'] = np.log1p(df['feature_3'])
    
    # Drop columns z dużą ilością NaN
    nan_cols = df.columns[df.isna().sum() > len(df) * 0.5]
    df = df.drop(columns=nan_cols)
    
    return df

def train(train_df, val_df, config):
    # Standardowy kod trenowania sieci neuronowej
    ...
```

### 9. Rozszerzenia (opcjonalne na przyszłość)

1. **Callbacks**: Opcjonalna funkcja `on_epoch_end()` dla modeli iteracyjnych
2. **Custom metrics**: Funkcja `evaluate()` dla własnych metryk
3. **Ensemble**: Specjalny model który ładuje inne modele
4. **AutoML**: Template który automatycznie testuje różne modele
5. **Distributed training**: Wsparcie dla Ray/Dask

### 10. Podsumowanie

To rozwiązanie:
- **Minimalizuje zmiany** w istniejącym systemie
- **Maksymalizuje czytelność** kodu ML
- **Zachowuje pełną funkcjonalność** infrastruktury
- **Umożliwia stopniową migrację** bez breaking changes

Kluczowe decyzje:
1. Używamy konwencji (train/predict) zamiast dziedziczenia
2. Template definiuje który model uruchomić
3. Config przekazuje wszystkie parametry
4. Runner zajmuje się całą infrastrukturą
5. Snapshot całego katalogu `code/` zapewnia reprodukowalność

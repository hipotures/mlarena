"""
Helper functions for handling categorical columns in models.

Automatycznie wykrywa i konwertuje kolumny kategoryczne zakodowane jako int/float.
"""
import pandas as pd
from pathlib import Path
from typing import List, Optional
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import z tego samego katalogu
try:
    from categorical_detector import detect_categorical_int_columns
except ImportError:
    # Fallback: import absolutny
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "categorical_detector",
        Path(__file__).parent / "categorical_detector.py"
    )
    detector_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector_module)
    detect_categorical_int_columns = detector_module.detect_categorical_int_columns

try:
    from code.utils.config import TRAIN_PATH, TEST_PATH
except ImportError:
    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_PATH = DATA_DIR / "train.csv"
    TEST_PATH = DATA_DIR / "test.csv"


# Cache wykrytych kolumn (obliczane raz przy pierwszym wywołaniu)
_CACHED_CATEGORICAL_COLS = None


def get_categorical_int_columns(
    threshold: int = 25,
    force_refresh: bool = False
) -> List[str]:
    """
    Zwraca listę kolumn kategorycznych wykrytych automatycznie.

    Args:
        threshold: Maksymalna liczba unikalnych wartości (default: 25)
        force_refresh: Wymuś ponowne wykrywanie (default: False)

    Returns:
        Lista nazw kolumn kategorycznych
    """
    global _CACHED_CATEGORICAL_COLS

    if _CACHED_CATEGORICAL_COLS is None or force_refresh:
        results = detect_categorical_int_columns(
            TRAIN_PATH,
            TEST_PATH,
            max_unique_threshold=threshold
        )
        _CACHED_CATEGORICAL_COLS = list(results.keys())

    return _CACHED_CATEGORICAL_COLS


def convert_to_category(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    auto_detect: bool = True,
    threshold: int = 25,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Konwertuje kolumny kategoryczne na typ 'category'.

    Args:
        df: DataFrame do konwersji
        columns: Lista kolumn do konwersji (jeśli None, użyje auto_detect)
        auto_detect: Czy automatycznie wykryć kolumny kategoryczne (default: True)
        threshold: Próg dla auto-detect (default: 25)
        inplace: Czy modyfikować df in-place (default: False)

    Returns:
        DataFrame z przekonwertowanymi kolumnami
    """
    if not inplace:
        df = df.copy()

    if columns is None and auto_detect:
        columns = get_categorical_int_columns(threshold=threshold)

    if columns:
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype('category')

    return df


def prepare_for_autogluon(
    df: pd.DataFrame,
    auto_detect: bool = True,
    threshold: int = 25
) -> pd.DataFrame:
    """
    Przygotowuje DataFrame dla AutoGluon.

    AutoGluon automatycznie wykrywa kolumny binarne (0/1) jako bool,
    ale nie radzi sobie z innymi kolumnami kategorycznymi zakodowanymi jako int.

    Args:
        df: DataFrame do przygotowania
        auto_detect: Czy automatycznie wykryć kolumny kategoryczne
        threshold: Próg dla auto-detect (default: 25)

    Returns:
        DataFrame z przekonwertowanymi kolumnami kategorycznymi
    """
    return convert_to_category(df, auto_detect=auto_detect, threshold=threshold)


def get_lgb_categorical_features(
    df: pd.DataFrame,
    auto_detect: bool = True,
    threshold: int = 25
) -> List[str]:
    """
    Zwraca listę kolumn kategorycznych do użycia w LightGBM.

    Przykład użycia:
        cat_features = get_lgb_categorical_features(X_train)
        lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_features)

    Args:
        df: DataFrame z danymi
        auto_detect: Czy automatycznie wykryć kolumny kategoryczne
        threshold: Próg dla auto-detect (default: 25)

    Returns:
        Lista nazw kolumn kategorycznych obecnych w df
    """
    if auto_detect:
        all_categorical = get_categorical_int_columns(threshold=threshold)
        return [col for col in all_categorical if col in df.columns]
    return []


def get_xgb_feature_types(
    df: pd.DataFrame,
    auto_detect: bool = True,
    threshold: int = 25
) -> List[str]:
    """
    Zwraca listę typów cech dla XGBoost.

    Przykład użycia:
        feature_types = get_xgb_feature_types(X_train)
        dtrain = xgb.DMatrix(X_train, y_train, feature_types=feature_types)

    Args:
        df: DataFrame z danymi
        auto_detect: Czy automatycznie wykryć kolumny kategoryczne
        threshold: Próg dla auto-detect (default: 25)

    Returns:
        Lista typów: 'c' dla kategorycznych, 'q' dla numerycznych
    """
    categorical_cols = set()
    if auto_detect:
        categorical_cols = set(get_categorical_int_columns(threshold=threshold))

    return ['c' if col in categorical_cols else 'q' for col in df.columns]


def get_catboost_cat_features(
    df: pd.DataFrame,
    auto_detect: bool = True,
    threshold: int = 25
) -> List[int]:
    """
    Zwraca indeksy kolumn kategorycznych dla CatBoost.

    Przykład użycia:
        cat_features = get_catboost_cat_features(X_train)
        model = CatBoostClassifier(cat_features=cat_features)

    Args:
        df: DataFrame z danymi
        auto_detect: Czy automatycznie wykryć kolumny kategoryczne
        threshold: Próg dla auto-detect (default: 25)

    Returns:
        Lista indeksów kolumn kategorycznych
    """
    if auto_detect:
        categorical_cols = get_categorical_int_columns(threshold=threshold)
        return [df.columns.get_loc(col) for col in categorical_cols if col in df.columns]
    return []


# Wygodne aliasy
prepare_for_ag = prepare_for_autogluon
lgb_cat_features = get_lgb_categorical_features
xgb_feature_types = get_xgb_feature_types
catboost_cat_features = get_catboost_cat_features


if __name__ == "__main__":
    # Przykład użycia
    print("Wykryte kolumny kategoryczne:")
    cat_cols = get_categorical_int_columns()
    for col in cat_cols:
        print(f"  - {col}")

    print("\n" + "="*80)
    print("PRZYKŁADY UŻYCIA W KODZIE")
    print("="*80 + "\n")

    print("# AutoGluon:")
    print("from code.utils.categorical_helper import prepare_for_autogluon")
    print("train = prepare_for_autogluon(train)")
    print("test = prepare_for_autogluon(test)")
    print()

    print("# LightGBM:")
    print("from code.utils.categorical_helper import lgb_cat_features")
    print("cat_feats = lgb_cat_features(X_train)")
    print("lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=cat_feats)")
    print()

    print("# XGBoost:")
    print("from code.utils.categorical_helper import xgb_feature_types")
    print("feature_types = xgb_feature_types(X_train)")
    print("dtrain = xgb.DMatrix(X_train, y_train, feature_types=feature_types)")
    print()

    print("# CatBoost:")
    print("from code.utils.categorical_helper import catboost_cat_features")
    print("cat_features = catboost_cat_features(X_train)")
    print("model = CatBoostClassifier(cat_features=cat_features)")

"""
Automatyczne wykrywanie kolumn kategorycznych zakodowanych jako integery.

Analizuje train + test razem, aby wykryć wszystkie możliwe wartości kategorii.
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import paths from config
try:
    from code.utils.config import TRAIN_PATH, TEST_PATH
except ImportError:
    # Fallback: construct paths directly
    DATA_DIR = PROJECT_ROOT / "data"
    TRAIN_PATH = DATA_DIR / "train.csv"
    TEST_PATH = DATA_DIR / "test.csv"


def detect_categorical_int_columns(
    train_path: Path,
    test_path: Path,
    max_unique_threshold: int = 25,
    min_rows_ratio: float = 0.01,
    exclude_cols: List[str] = None
) -> Dict[str, Dict]:
    """
    Wykrywa kolumny kategoryczne zakodowane jako int/float.

    Args:
        train_path: Ścieżka do train.csv
        test_path: Ścieżka do test.csv
        max_unique_threshold: Maksymalna liczba unikalnych wartości dla kategorii (default: 25)
        min_rows_ratio: Minimalny stosunek unique/total_rows dla kategorii (default: 0.01 = 1%)
        exclude_cols: Lista kolumn do pominięcia (np. ['id', 'target'])

    Returns:
        Dict z informacjami o wykrytych kolumnach kategorycznych
    """
    if exclude_cols is None:
        exclude_cols = ['id']

    # Wczytaj dane
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    total_rows = len(df_train) + len(df_test)

    results = {}

    # Wspólne kolumny numeric (int/float)
    numeric_cols = [
        col for col in df_train.columns
        if col not in exclude_cols
        and df_train[col].dtype in ['int64', 'float64']
        and col in df_test.columns
    ]

    for col in numeric_cols:
        # Połącz wartości z train + test
        combined_values = pd.concat([
            df_train[col].dropna(),
            df_test[col].dropna()
        ])

        n_unique = combined_values.nunique()
        unique_ratio = n_unique / total_rows

        # Kryteria: max unique <= threshold ORAZ ratio < min_ratio
        is_categorical = (
            n_unique <= max_unique_threshold
            and unique_ratio < min_rows_ratio
        )

        if is_categorical:
            unique_vals = sorted(combined_values.unique())

            # Sprawdź czy binarna (0/1 lub podobne)
            is_binary = n_unique == 2

            # Sprawdź czy sekwencja całkowitoliczbowa (1,2,3... lub 0,1,2...)
            is_sequential = False
            if df_train[col].dtype == 'int64' and n_unique > 2:
                min_val = int(min(unique_vals))
                max_val = int(max(unique_vals))
                expected_range = list(range(min_val, max_val + 1))
                is_sequential = unique_vals == expected_range

            results[col] = {
                'n_unique': n_unique,
                'unique_ratio': round(unique_ratio, 6),
                'unique_values': unique_vals,
                'dtype': str(df_train[col].dtype),
                'is_binary': is_binary,
                'is_sequential': is_sequential,
                'train_unique': df_train[col].nunique(),
                'test_unique': df_test[col].nunique(),
                'train_only': sorted(set(df_train[col].unique()) - set(df_test[col].unique())),
                'test_only': sorted(set(df_test[col].unique()) - set(df_train[col].unique())),
            }

    return results


def print_categorical_report(results: Dict[str, Dict], verbose: bool = True):
    """
    Wyświetla raport z wykrytych kolumn kategorycznych.
    """
    if not results:
        print("✗ Nie wykryto kolumn kategorycznych zakodowanych jako int/float")
        return

    print(f"\n{'='*80}")
    print(f"WYKRYTE KOLUMNY KATEGORYCZNE (INT/FLOAT)")
    print(f"{'='*80}\n")
    print(f"Łącznie wykrytych: {len(results)} kolumn\n")

    # Sortuj: binarne najpierw, potem wg liczby unique
    sorted_cols = sorted(
        results.items(),
        key=lambda x: (not x[1]['is_binary'], x[1]['n_unique'])
    )

    for col, info in sorted_cols:
        category_type = "BINARY" if info['is_binary'] else "ORDINAL/NOMINAL"
        seq_marker = " [SEQUENTIAL]" if info['is_sequential'] else ""

        print(f"{col}")
        print(f"  Typ: {category_type}{seq_marker}")
        print(f"  Unique: {info['n_unique']} (ratio: {info['unique_ratio']:.2%})")
        print(f"  Dtype: {info['dtype']}")

        if verbose:
            print(f"  Wartości: {info['unique_values']}")
            print(f"  Train unique: {info['train_unique']} | Test unique: {info['test_unique']}")

            if info['train_only']:
                print(f"  ⚠ Tylko w train: {info['train_only']}")
            if info['test_only']:
                print(f"  ⚠ Tylko w test: {info['test_only']}")

        print()


def generate_conversion_code(results: Dict[str, Dict]) -> str:
    """
    Generuje kod Python do konwersji wykrytych kolumn na kategorie.
    """
    if not results:
        return "# Brak kolumn do konwersji"

    cols_list = list(results.keys())

    code = f"""# Konwersja kolumn kategorycznych wykrytych automatycznie
# Wykryto {len(cols_list)} kolumn

categorical_int_columns = {cols_list}

# Przed treningiem/predykcją:
for col in categorical_int_columns:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Lub dla XGBoost/LightGBM:
# lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_int_columns)
"""
    return code


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Wykryj kolumny kategoryczne zakodowane jako int/float"
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=25,
        help='Maksymalna liczba unikalnych wartości (default: 25)'
    )
    parser.add_argument(
        '--min-ratio',
        type=float,
        default=0.01,
        help='Minimalny stosunek unique/total_rows (default: 0.01 = 1%%)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Pokaż szczegółowe informacje'
    )
    parser.add_argument(
        '--generate-code',
        action='store_true',
        help='Wygeneruj kod do konwersji'
    )

    args = parser.parse_args()

    print(f"Analizowanie: {TRAIN_PATH.name} + {TEST_PATH.name}")
    print(f"Threshold: {args.threshold} unique values")
    print(f"Min ratio: {args.min_ratio:.2%}")

    results = detect_categorical_int_columns(
        TRAIN_PATH,
        TEST_PATH,
        max_unique_threshold=args.threshold,
        min_rows_ratio=args.min_ratio
    )

    print_categorical_report(results, verbose=args.verbose)

    if args.generate_code:
        print(f"\n{'='*80}")
        print("KOD DO UŻYCIA W MODELACH")
        print(f"{'='*80}\n")
        print(generate_conversion_code(results))

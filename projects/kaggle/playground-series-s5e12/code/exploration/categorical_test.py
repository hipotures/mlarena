"""
Test automatycznego wykrywania i konwersji kolumn kategorycznych.
Porównanie przed i po konwersji.
"""
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "code" / "utils"))

from categorical_helper import (
    get_categorical_int_columns,
    convert_to_category,
    prepare_for_autogluon
)

# Wczytaj dane
train = pd.read_csv(PROJECT_ROOT / "data" / "train.csv")
test = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")

print("="*80)
print("AUTOMATYCZNE WYKRYWANIE KOLUMN KATEGORYCZNYCH")
print("="*80 + "\n")

# Wykryj kolumny kategoryczne
cat_cols = get_categorical_int_columns(threshold=25)
print(f"Wykryto {len(cat_cols)} kolumn kategorycznych:\n")
for col in cat_cols:
    print(f"  ✓ {col}")

print("\n" + "="*80)
print("PORÓWNANIE: PRZED vs PO KONWERSJI")
print("="*80 + "\n")

# Przed konwersją
print("PRZED konwersją (train):")
for col in cat_cols:
    print(f"  {col:35s} | dtype: {train[col].dtype}")

print("\n" + "-"*80 + "\n")

# Konwersja
train_converted = convert_to_category(train)
test_converted = convert_to_category(test)

print("PO konwersji (train):")
for col in cat_cols:
    print(f"  {col:35s} | dtype: {train_converted[col].dtype}")

print("\n" + "="*80)
print("SZCZEGÓŁY KATEGORII")
print("="*80 + "\n")

for col in cat_cols:
    n_categories = train_converted[col].nunique()
    categories = sorted(train_converted[col].unique())

    print(f"{col}:")
    print(f"  Liczba kategorii: {n_categories}")
    print(f"  Wartości: {categories}")
    print(f"  Memory usage przed: {train[col].memory_usage(deep=True) / 1024:.1f} KB")
    print(f"  Memory usage po:    {train_converted[col].memory_usage(deep=True) / 1024:.1f} KB")
    print()

print("="*80)
print("ROZKŁAD WARTOŚCI (alcohol_consumption_per_week)")
print("="*80 + "\n")

print("Train:")
print(train_converted['alcohol_consumption_per_week'].value_counts().sort_index())

print("\nTest:")
print(test_converted['alcohol_consumption_per_week'].value_counts().sort_index())

print("\n" + "="*80)
print("VERYFIKACJA: CZY WARTOŚCI W TEST SĄ OBECNE W TRAIN?")
print("="*80 + "\n")

for col in cat_cols:
    train_vals = set(train[col].unique())
    test_vals = set(test[col].unique())

    only_in_test = test_vals - train_vals
    only_in_train = train_vals - test_vals

    status = "✓ OK" if not only_in_test else "⚠ WARNING"

    print(f"{col:35s} | {status}")
    if only_in_test:
        print(f"  Tylko w test: {sorted(only_in_test)}")
    if only_in_train:
        print(f"  Tylko w train: {sorted(only_in_train)}")

print("\n" + "="*80)
print("MEMORY SAVINGS")
print("="*80 + "\n")

memory_before = sum(train[col].memory_usage(deep=True) for col in cat_cols) / 1024**2
memory_after = sum(train_converted[col].memory_usage(deep=True) for col in cat_cols) / 1024**2
savings = memory_before - memory_after
savings_pct = (savings / memory_before) * 100

print(f"Memory usage przed:  {memory_before:.2f} MB")
print(f"Memory usage po:     {memory_after:.2f} MB")
print(f"Oszczędność:         {savings:.2f} MB ({savings_pct:.1f}%)")

print("\n✓ Test zakończony pomyślnie!")

"""
Test categorical_encoder with auto-detection on s5e12 data.
"""
import sys
from pathlib import Path
import pandas as pd

# Add paths
PROJECT_ROOT = Path(__file__).parent
PREPROCESS_PATH = PROJECT_ROOT.parent.parent.parent / "config" / "code" / "preprocessing"
sys.path.insert(0, str(PREPROCESS_PATH))

# Import module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "categorical_encoder",
    PREPROCESS_PATH / "categorical_encoder.py"
)
categorical_encoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(categorical_encoder)
fit_transform = categorical_encoder.fit_transform

# Load data
train_df = pd.read_csv(PROJECT_ROOT / "data" / "train.csv")
test_df = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")

print(f"Loaded train: {train_df.shape}")
print(f"Loaded test: {test_df.shape}")

# Config with auto-detect enabled
config = {
    "max_cardinality": 50,
    "exclude_text_type": False,
    "include_numeric_categories": True,
    "enable_auto_detect": True,
    "auto_detect_threshold": 25,
    "_system": {
        "project_root": PROJECT_ROOT,
    },
    "_dataset": {
        "target": "diagnosed_diabetes",
    },
}

# Run preprocessing
train_out, val_out, test_out, state = fit_transform(
    train_df, None, test_df, config
)

print("\n" + "="*80)
print("STATE SUMMARY")
print("="*80)
print(f"Categorical columns: {len(state['categorical_columns'])}")
print(f"  From EDA: {state['conversion_summary']['eda_count']}")
print(f"  From auto-detect: {state['conversion_summary']['auto_detect_count']}")
print(f"  Overlap: {state['conversion_summary']['overlap_count']}")

print("\n" + "="*80)
print("AUTO-DETECTED COLUMNS")
print("="*80)
auto_meta = state.get('auto_detect_metadata', {})
for col, meta in auto_meta.items():
    print(f"\n{col}:")
    print(f"  Type: {meta['type']}")
    print(f"  Distinct: {meta['n_distinct']}")
    print(f"  Values: {meta['unique_values'][:10]}...")  # First 10

print("\n✓ Test completed successfully!")

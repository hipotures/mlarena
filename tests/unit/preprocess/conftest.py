import pytest
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime, timedelta

REPO_ROOT = Path(__file__).resolve().parents[3]
SPACES_DIR = REPO_ROOT / "src" / "mlarena" / "search_spaces" / "preprocess"

@pytest.fixture
def dataset_no_null():
    """Dataset A: Bez wartosci null."""
    data = {
        "id": list(range(1, 13)),
        "target": [0, 1] * 6,
        "num_a": [float(i) for i in range(1, 13)],
        "num_b": [float(i)*2 for i in range(1, 13)],
        "cat_a": ["A", "B", "C"] * 4,
        "cat_b": ["X", "Y"] * 6,
        "dt_a": [datetime(2020, 1, 1) + timedelta(days=i) for i in range(12)],
        "bool_a": [True, False] * 6
    }
    return pd.DataFrame(data)

@pytest.fixture
def dataset_with_null(dataset_no_null):
    """Dataset B: Z wartosciami null."""
    df = dataset_no_null.copy()
    # Introduce NaNs
    df.loc[0:2, "num_a"] = np.nan
    df.loc[5, "num_a"] = np.nan
    df.loc[1:2, "cat_a"] = np.nan
    df.loc[8:9, "dt_a"] = np.nan
    
    # All NaN column
    df["num_c_all_nan"] = np.nan
    return df

@pytest.fixture
def all_search_spaces():
    """Zwraca slownik {module_name: yaml_content}."""
    spaces = {}
    if not SPACES_DIR.exists():
        return spaces
        
    for f in SPACES_DIR.glob("*.yaml"):
        with open(f, "r") as stream:
            spaces[f.stem] = yaml.safe_load(stream)
    return spaces

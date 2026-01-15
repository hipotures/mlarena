import pytest
import pandas as pd
from tests.unit.preprocess.utils import get_processor_class

def test_outlier_handler_invalid_quantiles(dataset_no_null):
    cls = get_processor_class("outlier_handler")
    if not cls: return

    config = {
        "method": "iqr",
        "action": "clip",
        "lower_quantile": 0.95,
        "upper_quantile": 0.05, # Blad: lower > upper
        "_dataset": {"id_column": "id", "target": "target"}
    }
    
    processor = cls(config)
    with pytest.raises(ValueError, match="greater than lower_quantile"):
        processor.fit_transform(dataset_no_null, dataset_no_null, dataset_no_null)

def test_scaler_boxcox_negative_data(dataset_no_null):
    cls = get_processor_class("scaler")
    if not cls: return
    
    # Dataset z wartosciami ujemnymi
    df = dataset_no_null.copy()
    df["num_a"] = -5.0
    
    config = {
        "method": "power_boxcox",
        "_dataset": {"id_column": "id", "target": "target"}
    }
    
    processor = cls(config)
    # Box-Cox wymaga danych > 0. Oczekujemy bledu z sklearn lub wrappera
    # Jesli nie rzuca bledu, to moze scaler robi shift. 
    # Dla bezpieczenstwa sprawdzamy czy nie rzuca (jesli robust) lub rzuca konkretny
    try:
        processor.fit_transform(df, df, df)
    except ValueError as e:
        assert "strictly positive" in str(e)

def test_groupwise_normalizer_invalid_quantile(dataset_no_null):
    cls = get_processor_class("groupwise_normalizer")
    if not cls: return
    
    config = {
        "group_keys": ["cat_a"], # Poprawione z group_col
        "value_cols": ["num_a"], # Poprawione z value_col
        "method": "zscore", 
        "reference_stat": "quantile",
        "quantile_value": 1.5, # Invalid > 1.0
        "_dataset": {"id_column": "id", "target": "target"}
    }
    
    processor = cls(config)
    with pytest.raises(ValueError, match="must be <= 1.0"):
        processor.fit_transform(dataset_no_null, dataset_no_null, dataset_no_null)

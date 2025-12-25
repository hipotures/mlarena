"""
Ratio-Based Physiological Features for Diabetes Prediction

Creates ratio features capturing relationships between activity, sleep, BMI, and diet.
These features represent lifestyle balance and efficiency metrics.

Source: Analysis of mariusborel notebook (57 votes)
Expected impact: +1-2% local CV
"""

import pandas as pd
import numpy as np


def fit_transform(train_df, val_df, test_df, config, orig_df=None):
    """
    Create ratio-based physiological features.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        config: Configuration dict
        orig_df: Original/external dataset (optional)

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)
    """
    epsilon = config.get('epsilon', 1e-9)

    # Apply features to all datasets
    for df in [train_df, val_df, test_df] + ([orig_df] if orig_df is not None else []):
        if df is None:
            continue

        # Activity/Sleep Balance
        df['activity_per_sleep_hour'] = (
            df['physical_activity_minutes_per_week'] / (df['sleep_hours_per_day'] * 60 + epsilon)
        )

        df['sleep_screen_ratio'] = (
            df['sleep_hours_per_day'] / (df['screen_time_hours_per_day'] + epsilon)
        )

        # BMI Efficiency/Interaction
        df['bmi_diet_ratio'] = df['bmi'] / (df['diet_score'] + epsilon)

        df['bmi_diastolic_product'] = df['bmi'] * df['diastolic_bp']

        # Blood Pressure Relationships
        df['diastolic_systolic_ratio'] = (
            df['diastolic_bp'] / (df['systolic_bp'] + epsilon)
        )

        df['pulse_pressure_bmi_ratio'] = (
            (df['systolic_bp'] - df['diastolic_bp']) / (df['bmi'] + epsilon)
        )

    state = {
        'epsilon': epsilon,
        'features_created': [
            'activity_per_sleep_hour',
            'sleep_screen_ratio',
            'bmi_diet_ratio',
            'bmi_diastolic_product',
            'diastolic_systolic_ratio',
            'pulse_pressure_bmi_ratio'
        ]
    }

    if orig_df is not None:
        return train_df, val_df, test_df, orig_df, state
    else:
        return train_df, val_df, test_df, state

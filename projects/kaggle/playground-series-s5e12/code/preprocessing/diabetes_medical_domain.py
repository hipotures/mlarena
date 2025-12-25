"""
Medical Domain Feature Engineering for Diabetes Prediction

Creates clinically-relevant features based on cardiovascular markers,
lipid profiles, and medical threshold categories.

Source: Analysis of 6/8 top Kaggle notebooks
Expected impact: +2-3% local CV
"""

import pandas as pd
import numpy as np


def fit_transform(train_df, val_df, test_df, config, orig_df=None):
    """
    Create medical domain features for diabetes prediction.

    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        config: Configuration dict with feature creation flags
        orig_df: Original/external dataset (optional)

    Returns:
        Tuple of (train_df, val_df, test_df, orig_df, state_dict)
    """
    epsilon = config.get('epsilon', 1e-9)

    # Apply features to all datasets
    for df in [train_df, val_df, test_df] + ([orig_df] if orig_df is not None else []):
        if df is None:
            continue

        # 1. Cardiovascular Features
        if config.get('create_cardiovascular', True):
            # Pulse pressure (systolic - diastolic)
            df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']

            # Mean Arterial Pressure
            df['mean_arterial_pressure'] = (df['systolic_bp'] + 2 * df['diastolic_bp']) / 3

            # Rate Pressure Product (heart rate × systolic BP)
            df['rate_pressure_product'] = df['heart_rate'] * df['systolic_bp']

        # 2. Lipid Profile Ratios
        if config.get('create_lipid_ratios', True):
            # LDL/HDL ratio (atherogenic index)
            df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + epsilon)

            # Triglycerides/HDL ratio
            df['tg_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + epsilon)

            # Total cholesterol/HDL ratio
            df['cholesterol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + epsilon)

            # Non-HDL cholesterol
            df['non_hdl_cholesterol'] = df['cholesterol_total'] - df['hdl_cholesterol']

            # Lipid burden composite score
            df['lipid_burden'] = (
                df['ldl_hdl_ratio'] +
                df['tg_hdl_ratio'] +
                df['cholesterol_hdl_ratio']
            )

        # 3. Clinical Threshold Categories
        if config.get('create_clinical_categories', True):
            # BMI categories (underweight/normal/overweight/obese)
            df['bmi_category'] = pd.cut(
                df['bmi'],
                bins=[0, 18.5, 25, 30, 100],
                labels=[0, 1, 2, 3]
            ).astype('int8')

            # Blood pressure categories (normal/elevated/high)
            df['bp_category'] = 0
            df.loc[
                (df['systolic_bp'] >= 130) | (df['diastolic_bp'] >= 80),
                'bp_category'
            ] = 1
            df.loc[
                (df['systolic_bp'] >= 140) | (df['diastolic_bp'] >= 90),
                'bp_category'
            ] = 2

        # 4. Medical Risk Composite Score
        if config.get('create_medical_risk', True):
            df['medical_risk'] = (
                df['family_history_diabetes'].astype(float) * 0.3 +
                df['hypertension_history'].astype(float) * 0.3 +
                df['cardiovascular_history'].astype(float) * 0.4
            )

    state = {
        'epsilon': epsilon,
        'features_created': {
            'cardiovascular': config.get('create_cardiovascular', True),
            'lipid_ratios': config.get('create_lipid_ratios', True),
            'clinical_categories': config.get('create_clinical_categories', True),
            'medical_risk': config.get('create_medical_risk', True)
        }
    }

    if orig_df is not None:
        return train_df, val_df, test_df, orig_df, state
    else:
        return train_df, val_df, test_df, state

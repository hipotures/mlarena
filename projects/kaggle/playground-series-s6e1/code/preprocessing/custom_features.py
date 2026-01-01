import pandas as pd
import numpy as np

def fit_transform(train_df, val_df, test_df, config, orig_df=None):
    """Advanced feature engineering for Exam Score prediction."""
    
    def add_features(df):
        if df is None:
            return None
        df = df.copy()
        
        # 1. Study & Attendance interaction (Real engagement)
        df['study_attendance_score'] = (df['study_hours'] / 8.0) * (df['class_attendance'] / 100.0)
        
        # 2. Sleep & Quality interaction
        # Mapping quality to numbers: poor=1, average=2, good=3
        quality_map = {'poor': 1, 'average': 2, 'good': 3}
        q_numeric = df['sleep_quality'].map(quality_map).fillna(2)
        df['effective_rest'] = df['sleep_hours'] * q_numeric
        
        # 3. Efficiency: study hours vs total hours
        df['study_sleep_ratio'] = df['study_hours'] / (df['sleep_hours'] + 1.0)
        
        # 4. Age-normalized features (Age as experience/maturity)
        df['study_per_age'] = df['study_hours'] / (df['age'] - 15)
        
        # 5. Difficulty vs Study Method (interaction)
        # Tree models handle this, but let's add a simple one
        df['hard_study'] = ((df['exam_difficulty'] == 'hard').astype(int) * df['study_hours'])
        
        return df

    train_df = add_features(train_df)
    if val_df is not None:
        val_df = add_features(val_df)
    test_df = add_features(test_df)
    if orig_df is not None:
        orig_df = add_features(orig_df)

    return train_df, val_df, test_df, orig_df, {}
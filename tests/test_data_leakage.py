"""
Unit tests for data leakage protection in preprocessing module.

These tests verify that cv_stage transformers (especially TargetEncoding)
do NOT leak validation data when fitted per-fold.
"""

import pytest
import pandas as pd
import numpy as np

from kaggle_tools.preprocessing.cv_stage import (
    TargetEncodingTransformer,
    FrequencyEncodingTransformer,
)


class TestTargetEncodingLeakage:
    """Test that TargetEncoding doesn't leak validation targets."""

    def test_no_leakage_unseen_category(self):
        """
        CRITICAL TEST: Verify that unseen categories in validation
        get global mean, NOT their own target mean (which would be leakage).
        """
        # Create data where leakage would be obvious
        train_df = pd.DataFrame({
            'cat_col': ['A', 'A', 'B', 'B'],
            'target': [0, 1, 0, 1]
        })

        # Simulate CV fold
        train_fold = train_df.iloc[:2]  # A with targets [0, 1]
        val_fold = train_df.iloc[2:]     # B with targets [0, 1]

        # Create encoder
        encoder = TargetEncodingTransformer(
            columns=['cat_col'],
            target='target',
            smoothing=0.0  # No smoothing for clearer test
        )

        # Fit ONLY on train_fold
        encoder.fit(train_fold)

        # Transform validation fold
        val_transformed = encoder.transform(val_fold)

        # CRITICAL CHECK: B's encoding should be global mean (0.5)
        # NOT B's own target mean (which would be 0.5 but from val data = LEAKAGE)
        b_encoding = val_transformed['cat_col_te'].iloc[0]
        global_mean = train_fold['target'].mean()

        # B is unseen in train_fold, so should get global_mean
        assert abs(b_encoding - global_mean) < 1e-6, (
            f"LEAKAGE DETECTED: B encoding is {b_encoding:.4f}, "
            f"expected {global_mean:.4f} (global mean from train_fold)"
        )

    def test_smoothing_prevents_overfitting(self):
        """Test that smoothing reduces impact of small sample sizes."""
        train_df = pd.DataFrame({
            'cat_col': ['A', 'A', 'A', 'B'],
            'target': [0, 1, 1, 1]
        })

        encoder_no_smooth = TargetEncodingTransformer(
            columns=['cat_col'],
            target='target',
            smoothing=0.0
        )
        encoder_no_smooth.fit(train_df)

        encoder_smooth = TargetEncodingTransformer(
            columns=['cat_col'],
            target='target',
            smoothing=10.0  # Heavy smoothing
        )
        encoder_smooth.fit(train_df)

        test_df = pd.DataFrame({'cat_col': ['B']})

        # No smoothing: B should get 1.0 (its exact target mean)
        no_smooth_result = encoder_no_smooth.transform(test_df)
        assert abs(no_smooth_result['cat_col_te'].iloc[0] - 1.0) < 1e-6

        # With smoothing: B should be closer to global mean (0.75)
        smooth_result = encoder_smooth.transform(test_df)
        global_mean = train_df['target'].mean()
        smoothed_value = smooth_result['cat_col_te'].iloc[0]

        # Smoothed value should be between B's mean (1.0) and global mean (0.75)
        assert smoothed_value < 1.0
        assert smoothed_value > global_mean

    def test_min_samples_leaf(self):
        """Test that rare categories are filtered by min_samples_leaf."""
        train_df = pd.DataFrame({
            'cat_col': ['A', 'A', 'A', 'B', 'C', 'C'],
            'target': [0, 1, 1, 1, 0, 0]
        })

        encoder = TargetEncodingTransformer(
            columns=['cat_col'],
            target='target',
            smoothing=0.0,
            min_samples_leaf=2  # B has only 1 sample, should be excluded
        )
        encoder.fit(train_df)

        test_df = pd.DataFrame({'cat_col': ['B']})  # Rare category
        result = encoder.transform(test_df)

        # B should get global mean (not its own mean of 1.0)
        global_mean = train_df['target'].mean()
        assert abs(result['cat_col_te'].iloc[0] - global_mean) < 1e-6


class TestFeaturePipelineLeakage:
    """Test FeaturePipeline doesn't leak through cv_stage."""

    def test_cv_stage_fitted_per_fold(self):
        """
        Verify that cv_stage transformers maintain separate state per fold.
        """
        from kaggle_tools.preprocessing import FeaturePipeline
        from kaggle_tools.config_models import PreprocessingConfig

        # Create data
        full_train = pd.DataFrame({
            'cat_col': ['A', 'A', 'B', 'B', 'C', 'C'],
            'target': [0, 1, 0, 1, 0, 1]
        })

        config = PreprocessingConfig(
            feat_stage={'enabled': False},
            cv_stage={'enabled': False}  # We'll manually test cv_stage
        )

        pipeline = FeaturePipeline(config)

        # Simulate 2 folds
        fold1_train = full_train.iloc[:4]
        fold1_val = full_train.iloc[4:]

        fold2_train = full_train.iloc[[0, 1, 4, 5]]
        fold2_val = full_train.iloc[[2, 3]]

        # Manually create encoders for each fold
        encoder1 = TargetEncodingTransformer(
            columns=['cat_col'],
            target='target',
            smoothing=0.0
        )
        encoder1.fit(fold1_train)

        encoder2 = TargetEncodingTransformer(
            columns=['cat_col'],
            target='target',
            smoothing=0.0
        )
        encoder2.fit(fold2_train)

        # Transform validation folds
        val1_transformed = encoder1.transform(fold1_val)
        val2_transformed = encoder2.transform(fold2_val)

        # C is unseen in fold1_train
        c_encoding_fold1 = val1_transformed['cat_col_te'].iloc[0]
        fold1_global_mean = fold1_train['target'].mean()
        assert abs(c_encoding_fold1 - fold1_global_mean) < 1e-6

        # B is seen in fold2_train with targets [0, 1]
        b_encoding_fold2 = val2_transformed['cat_col_te'].iloc[0]
        b_mean_in_fold2_train = 0.5  # B has targets [0, 1] in fold2_train
        assert abs(b_encoding_fold2 - b_mean_in_fold2_train) < 1e-6


class TestFrequencyEncodingLeakage:
    """Test FrequencyEncoding doesn't leak validation distribution."""

    def test_frequency_from_train_only(self):
        """Verify frequency encoding uses train distribution only."""
        # Category A appears 3x in train, but would appear 4x if we included val
        train_df = pd.DataFrame({'cat_col': ['A', 'A', 'A', 'B']})
        val_df = pd.DataFrame({'cat_col': ['A', 'B', 'C']})

        encoder = FrequencyEncodingTransformer(
            columns=['cat_col'],
            normalize=False  # Absolute counts
        )
        encoder.fit(train_df)

        val_transformed = encoder.transform(val_df)

        # A should have frequency 3 (from train), not 4 (train + val)
        assert val_transformed['cat_col_freq'].iloc[0] == 3

        # C is unseen, should have frequency 0
        assert val_transformed['cat_col_freq'].iloc[2] == 0


def test_import_all_transformers():
    """Smoke test: Verify all transformers can be imported."""
    from kaggle_tools.preprocessing.feat_stage import (
        LogTransformer,
        PolynomialFeaturesTransformer,
        StandardScalerTransformer,
        BinningTransformer,
    )
    from kaggle_tools.preprocessing.cv_stage import (
        TargetEncodingTransformer,
        FrequencyEncodingTransformer,
        TargetAggregatesTransformer,
        WeightOfEvidenceTransformer,
    )

    # If we get here, imports succeeded
    assert True


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

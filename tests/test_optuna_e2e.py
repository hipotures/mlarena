"""
End-to-end test for Optuna system.

Tests the complete pipeline:
1. Feature engineering (feat_stage + cv_stage)
2. Optuna hyperparameter tuning
3. Model training with best params
4. Prediction generation
5. Ensemble blending

Run with: pytest tests/test_optuna_e2e.py -v -s
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from kaggle_tools.preprocessing import FeaturePipeline
from kaggle_tools.preprocessing.feat_stage import LogTransformer, StandardScalerTransformer
from kaggle_tools.preprocessing.cv_stage import TargetEncodingTransformer
from kaggle_tools.config_models import PreprocessingConfig, OptunaConfig
from kaggle_tools.optuna import StudyManager, CVObjective, xgboost_param_space
from kaggle_tools.stacking import WeightedBlender, RankBlender


@pytest.fixture
def sample_data():
    """Generate sample dataset for testing."""
    np.random.seed(42)
    n_samples = 500

    data = {
        "id": range(n_samples),
        "numeric_1": np.random.randn(n_samples),
        "numeric_2": np.random.randn(n_samples) * 10 + 50,
        "category_1": np.random.choice(["A", "B", "C", "D"], n_samples),
        "category_2": np.random.choice(["X", "Y", "Z"], n_samples),
        "target": np.random.randint(0, 2, n_samples),
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestFeatureEngineering:
    """Test feature engineering pipeline."""

    def test_feat_stage_transformers(self, sample_data):
        """Test global feat_stage transformers."""
        config = PreprocessingConfig(
            feat_stage={
                "enabled": True,
                "transformers": [
                    {
                        "type": "LogTransformer",
                        "columns": ["numeric_2"],
                    },
                    {
                        "type": "StandardScalerTransformer",
                        "columns": ["numeric_1"],
                    },
                ],
            },
            cv_stage={"enabled": False},
        )

        pipeline = FeaturePipeline(config)

        # Fit on full train
        pipeline.fit_feat_stage(sample_data)

        # Transform
        transformed = pipeline.transform_feat_stage(sample_data)

        # Check that transformations were applied
        assert "numeric_2_log" in transformed.columns
        assert "numeric_1_scaled" in transformed.columns

        # Check log transformation (log1p should be less than original for large values)
        # numeric_2 is around 50, so log1p(50) ~ 3.9, which is < 50
        assert transformed["numeric_2_log"].mean() < sample_data["numeric_2"].mean()

        # Check standardization (mean~0, std~1)
        assert abs(transformed["numeric_1_scaled"].mean()) < 0.1
        assert abs(transformed["numeric_1_scaled"].std() - 1.0) < 0.1

    def test_cv_stage_transformers(self, sample_data):
        """Test per-fold cv_stage transformers (data leakage protection)."""
        config = PreprocessingConfig(
            feat_stage={"enabled": False},
            cv_stage={
                "enabled": True,
                "transformers": [
                    {
                        "type": "TargetEncodingTransformer",
                        "columns": ["category_1"],
                        "target": "target",
                        "smoothing": 5.0,
                    },
                ],
            },
        )

        pipeline = FeaturePipeline(config)

        # Simulate CV fold
        train_fold = sample_data.iloc[:400]
        val_fold = sample_data.iloc[400:]

        # Fit and transform per-fold
        train_transformed, val_transformed = pipeline.fit_transform_cv_stage(
            train_fold, val_fold, fold_id=0
        )

        # Check target encoding was applied
        assert "category_1_te" in train_transformed.columns
        assert "category_1_te" in val_transformed.columns

        # Check no leakage: unseen categories should get global mean
        global_mean = train_fold["target"].mean()
        # Any category that appears in val but not train should have encoding ~ global_mean
        # (Hard to test without guaranteeing unseen categories, so just check column exists)

    def test_save_load_pipeline(self, sample_data, temp_dir):
        """Test pipeline save/load functionality."""
        config = PreprocessingConfig(
            feat_stage={
                "enabled": True,
                "transformers": [
                    {"type": "LogTransformer", "columns": ["numeric_2"]},
                ],
            },
            storage_format="parquet",
            compression="snappy",
            save_transformed_data=True,
        )

        pipeline = FeaturePipeline(config)
        pipeline.fit_feat_stage(sample_data)

        # Transform data
        transformed = pipeline.transform_feat_stage(sample_data)

        # Save pipeline
        save_path = temp_dir / "pipeline"
        pipeline.save(save_path)

        # Check transformer pickles were created
        assert (save_path / "feat_stage.pkl").exists()

        # Save transformed data
        pipeline.save_transformed_data(save_path, transformed, transformed)

        # Check files were created
        assert (save_path / "feature_metadata.json").exists()
        assert (save_path / "train_feat.parquet").exists()

        # Load metadata
        with open(save_path / "feature_metadata.json") as f:
            metadata = json.load(f)

        assert "feature_names" in metadata
        assert metadata["storage_format"] == "parquet"


class TestOptunaSystem:
    """Test Optuna hyperparameter tuning system."""

    def test_study_creation(self, temp_dir):
        """Test Optuna study creation and persistence."""
        storage_path = temp_dir / "optuna.db"

        config = OptunaConfig(
            storage=f"sqlite:///{storage_path}",
            study_name="test_study",
            direction="maximize",
            n_trials=3,
            cv_folds=2,
        )

        manager = StudyManager(config)
        study = manager.create_or_load_study()

        assert study.study_name == "test_study"
        assert study.direction.name == "MAXIMIZE"
        assert storage_path.exists()

    def test_cv_objective_xgboost(self, sample_data):
        """Test CVObjective with XGBoost."""
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score

        # Keep only numeric columns for XGBoost
        train_df = sample_data.drop(columns=["category_1", "category_2"])

        # Simple param space for quick test
        def simple_param_space(trial):
            return {
                "max_depth": trial.suggest_int("max_depth", 3, 5),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            }

        objective = CVObjective(
            model_class=xgb.XGBClassifier,
            train_df=train_df,
            target_col="target",
            param_space_fn=simple_param_space,
            metric_fn=roc_auc_score,
            cv_folds=2,
            early_stopping_rounds=10,
            random_seed=42,
        )

        # Test single trial
        import optuna

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=2, show_progress_bar=False)

        assert len(study.trials) == 2
        assert study.best_value > 0.0  # Some reasonable score

    def test_param_space_generators(self):
        """Test parameter space generators for all models."""
        import optuna

        # XGBoost
        from kaggle_tools.optuna import xgboost_param_space

        xgb_config = {
            "learning_rate": [0.01, 0.3, "log"],
            "max_depth": [3, 10, "int"],
            "subsample": [0.5, 1.0, "float"],
        }

        trial_xgb = optuna.trial.FixedTrial(
            {
                "learning_rate": 0.1,
                "max_depth": 5,
                "subsample": 0.8,
            }
        )

        xgb_params = xgboost_param_space(trial_xgb, xgb_config)
        assert "learning_rate" in xgb_params
        assert "max_depth" in xgb_params
        assert xgb_params["learning_rate"] == 0.1
        assert xgb_params["max_depth"] == 5

        # LightGBM
        from kaggle_tools.optuna import lightgbm_param_space

        lgb_config = {
            "learning_rate": [0.01, 0.3, "log"],
            "num_leaves": [15, 255, "int"],
        }

        trial_lgb = optuna.trial.FixedTrial(
            {
                "learning_rate": 0.1,
                "num_leaves": 31,
            }
        )

        lgb_params = lightgbm_param_space(trial_lgb, lgb_config)
        assert "learning_rate" in lgb_params
        assert "num_leaves" in lgb_params

        # CatBoost
        from kaggle_tools.optuna import catboost_param_space

        cb_config = {
            "learning_rate": [0.01, 0.3, "log"],
            "depth": [4, 10, "int"],
        }

        trial_cb = optuna.trial.FixedTrial(
            {
                "learning_rate": 0.1,
                "depth": 6,
            }
        )

        cb_params = catboost_param_space(trial_cb, cb_config)
        assert "learning_rate" in cb_params
        assert "depth" in cb_params


class TestStacking:
    """Test model stacking and ensembling."""

    def test_weighted_blender(self):
        """Test weighted averaging of predictions."""
        pred1 = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        pred2 = pd.Series([0.2, 0.3, 0.4, 0.5, 0.6])
        pred3 = pd.Series([0.3, 0.4, 0.5, 0.6, 0.7])

        blender = WeightedBlender(weights=[0.5, 0.3, 0.2])
        result = blender.blend([pred1, pred2, pred3])

        # Check weighted average
        expected = 0.5 * pred1 + 0.3 * pred2 + 0.2 * pred3
        assert result.equals(expected)

    def test_rank_blender(self):
        """Test rank averaging."""
        pred1 = pd.Series([0.1, 0.9, 0.2, 0.8, 0.3])
        pred2 = pd.Series([0.2, 0.8, 0.3, 0.7, 0.4])

        blender = RankBlender()
        result = blender.blend([pred1, pred2])

        # Rank averaging should be array
        assert len(result) == len(pred1)

        # Should preserve relative ordering somewhat
        # Higher values should generally have higher ranks
        assert result.iloc[1] > result.iloc[0]  # 0.9, 0.8 > 0.1, 0.2

    def test_power_blender(self):
        """Test power averaging."""
        from kaggle_tools.stacking import PowerBlender

        pred1 = pd.Series([0.2, 0.4, 0.6, 0.8])
        pred2 = pd.Series([0.3, 0.5, 0.7, 0.9])

        blender = PowerBlender(power=2.0)

        # Power=2 (quadratic mean)
        result = blender.blend([pred1, pred2])

        # Result should be between simple average and max
        simple_avg = (pred1 + pred2) / 2
        assert len(result) == len(pred1)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self, sample_data, temp_dir):
        """Test complete pipeline: preprocessing + tuning + prediction."""
        # 1. Feature engineering
        preprocessing_config = PreprocessingConfig(
            feat_stage={
                "enabled": True,
                "transformers": [
                    {"type": "StandardScalerTransformer", "columns": ["numeric_1", "numeric_2"]},
                ],
            },
            cv_stage={"enabled": False},
        )

        pipeline = FeaturePipeline(preprocessing_config)
        pipeline.fit_feat_stage(sample_data)
        transformed = pipeline.transform_feat_stage(sample_data)

        # 2. Optuna tuning (very quick)
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score

        # Drop categorical columns for XGBoost
        train_for_optuna = transformed.drop(columns=["category_1", "category_2"])

        optuna_config = OptunaConfig(
            storage=f"sqlite:///{temp_dir}/optuna.db",
            study_name="integration_test",
            direction="maximize",
            n_trials=2,  # Just 2 trials for quick test
            cv_folds=2,
        )

        study_manager = StudyManager(optuna_config)
        study = study_manager.create_or_load_study()

        def simple_param_space(trial):
            return {
                "max_depth": trial.suggest_int("max_depth", 3, 4),
                "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.1),
            }

        objective = CVObjective(
            model_class=xgb.XGBClassifier,
            train_df=train_for_optuna,
            target_col="target",
            param_space_fn=simple_param_space,
            metric_fn=roc_auc_score,
            cv_folds=2,
            early_stopping_rounds=10,
            random_seed=42,
        )

        study.optimize(objective, n_trials=2, show_progress_bar=False)

        # 3. Train final model
        X = train_for_optuna.drop(columns=["target", "id"])
        y = train_for_optuna["target"]

        best_params = study.best_params
        model = xgb.XGBClassifier(**best_params, n_estimators=50, random_state=42)
        model.fit(X, y)

        # 4. Generate predictions
        predictions = model.predict_proba(X)[:, 1]

        # 5. Ensemble (blend with itself for testing)
        blender = WeightedBlender(weights=[0.6, 0.4])
        ensemble_pred = blender.blend(
            [pd.Series(predictions), pd.Series(predictions)]
        )

        # Verify predictions are reasonable
        assert len(predictions) == len(sample_data)
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()
        assert len(ensemble_pred) == len(predictions)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

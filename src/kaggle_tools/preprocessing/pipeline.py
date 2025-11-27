"""
Two-stage feature engineering pipeline with data leakage protection.

This module implements the core FeaturePipeline class that enforces
strict separation between safe (feat_stage) and risky (cv_stage) features.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pickle

import pandas as pd

from kaggle_tools.config_models import PreprocessingConfig


class FeaturePipeline:
    """
    Two-stage feature engineering with data leakage protection.

    feat_stage: Global transformers fitted on entire train set
                (e.g., log transforms, polynomial features, label encoding)

    cv_stage:   Per-fold transformers fitted on train folds only
                (e.g., target encoding, frequency encoding, feature interactions)

    Example:
        >>> config = PreprocessingConfig(
        ...     feat_stage={"enabled": True, "transformers": ["log_transform"]},
        ...     cv_stage={"enabled": True, "transformers": ["target_encoding"]},
        ... )
        >>> pipeline = FeaturePipeline(config)
        >>>
        >>> # Fit global features on full train
        >>> pipeline.fit_feat_stage(train_df)
        >>> train_transformed = pipeline.transform_feat_stage(train_df)
        >>> test_transformed = pipeline.transform_feat_stage(test_df)
        >>>
        >>> # Fit per-fold features (inside CV loop)
        >>> for fold_id, (train_idx, val_idx) in enumerate(cv.split(train_df)):
        ...     train_fold = train_df.iloc[train_idx]
        ...     val_fold = train_df.iloc[val_idx]
        ...     train_t, val_t = pipeline.fit_transform_cv_stage(
        ...         train_fold, val_fold, fold_id
        ...     )
        >>>
        >>> # Transform test using aggregated CV transformers
        >>> test_final = pipeline.transform_cv_stage_inference(test_transformed)
    """

    def __init__(self, config: PreprocessingConfig):
        """
        Initialize pipeline with configuration.

        Args:
            config: Preprocessing configuration with feat_stage and cv_stage settings
        """
        self.config = config
        self.feat_stage_transformers: List = []
        self.cv_stage_transformers: List[Dict] = []

    def fit_feat_stage(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        """
        Fit global transformers on entire train set.

        These transformers are SAFE because they don't depend on validation data.
        Examples: log transforms, polynomial features, simple label encoding.

        Args:
            train_df: Full training DataFrame

        Returns:
            self for method chaining
        """
        if not self.config.feat_stage.get("enabled", False):
            return self

        transformer_names = self.config.feat_stage.get("transformers", [])
        for transformer_name in transformer_names:
            transformer = self._create_transformer(transformer_name, stage="feat")
            transformer.fit(train_df)
            self.feat_stage_transformers.append(transformer)
            self.fitted = True

        return self

    def transform_feat_stage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply global transformers (works for both train and test).

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        result = df.copy()
        for transformer in self.feat_stage_transformers:
            result = transformer.transform(result)
        return result

    def fit_transform_cv_stage(
        self,
        train_fold: pd.DataFrame,
        val_fold: pd.DataFrame,
        fold_id: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit per-fold transformers on train_fold ONLY, transform both train/val.

        These transformers are RISKY because they can leak validation information
        if not handled carefully. They MUST be fitted per-fold.
        Examples: target encoding, frequency encoding based on target.

        Args:
            train_fold: Training fold (for fitting)
            val_fold: Validation fold (for transformation only)
            fold_id: Fold identifier for tracking

        Returns:
            Tuple of (transformed_train_fold, transformed_val_fold)
        """
        if not self.config.cv_stage.get("enabled", False):
            return train_fold, val_fold

        transformers = []
        transformer_names = self.config.cv_stage.get("transformers", [])

        for transformer_name in transformer_names:
            transformer = self._create_transformer(transformer_name, stage="cv")
            # CRITICAL: Fit ONLY on train_fold, NOT on val_fold
            transformer.fit(train_fold)
            transformers.append(transformer)

        # Transform both train and validation folds
        train_result = train_fold.copy()
        val_result = val_fold.copy()

        for transformer in transformers:
            train_result = transformer.transform(train_result)
            val_result = transformer.transform(val_result)

        # Save fold-specific transformers for test inference
        self.cv_stage_transformers.append(
            {"fold_id": fold_id, "transformers": transformers}
        )

        return train_result, val_result

    def transform_cv_stage_inference(
        self, test_df: pd.DataFrame, aggregation: str = "mean"
    ) -> pd.DataFrame:
        """
        Apply CV-stage transformers to test set using aggregation.

        Since CV transformers are fitted per-fold, we need to aggregate
        their predictions to avoid bias.

        Args:
            test_df: Test DataFrame
            aggregation: "mean", "median", "first", or "all"

        Returns:
            Aggregated test predictions or list of DataFrames if aggregation="all"
        """
        if not self.cv_stage_transformers:
            return test_df

        transformed_versions = []

        for fold_dict in self.cv_stage_transformers:
            result = test_df.copy()
            for transformer in fold_dict["transformers"]:
                result = transformer.transform(result)
            transformed_versions.append(result)

        if aggregation == "first":
            return transformed_versions[0]
        elif aggregation == "all":
            return transformed_versions
        elif aggregation == "mean":
            # Average numeric features across folds
            return pd.concat(transformed_versions).groupby(level=0).mean()
        elif aggregation == "median":
            return pd.concat(transformed_versions).groupby(level=0).median()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    def save(self, path: Path):
        """
        Serialize pipeline to disk (pickle + parquet + metadata).

        Saves:
        - feat_stage.pkl: Global transformers
        - cv_stage_fold{i}.pkl: Per-fold transformers
        - train_feat.parquet: Transformed training data (if enabled)
        - test_feat.parquet: Transformed test data (if enabled)
        - feature_metadata.json: Column types and transformer info

        Args:
            path: Directory to save artifacts
        """
        path.mkdir(parents=True, exist_ok=True)

        # Save feat_stage transformers
        with open(path / "feat_stage.pkl", "wb") as f:
            pickle.dump(self.feat_stage_transformers, f)

        # Save cv_stage transformers (per-fold)
        for fold_dict in self.cv_stage_transformers:
            fold_id = fold_dict["fold_id"]
            with open(path / f"cv_stage_fold{fold_id}.pkl", "wb") as f:
                pickle.dump(fold_dict, f)

    def save_transformed_data(
        self,
        path: Path,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        """
        Save transformed data to parquet with metadata.

        Args:
            path: Directory to save data
            train_df: Transformed training DataFrame
            test_df: Transformed test DataFrame
        """
        if not self.config.save_transformed_data:
            return

        path.mkdir(parents=True, exist_ok=True)

        # Save as parquet (USER DECISION)
        compression = self.config.compression
        storage_format = self.config.storage_format

        if storage_format == "parquet":
            train_df.to_parquet(path / "train_feat.parquet", compression=compression, index=False)
            test_df.to_parquet(path / "test_feat.parquet", compression=compression, index=False)
        else:  # csv fallback
            train_df.to_csv(path / "train_feat.csv", index=False)
            test_df.to_csv(path / "test_feat.csv", index=False)

        # Save feature metadata (USER DECISION)
        metadata = {
            "feature_names": list(train_df.columns),
            "dtypes": {col: str(dtype) for col, dtype in train_df.dtypes.items()},
            "n_features": len(train_df.columns),
            "storage_format": storage_format,
            "compression": compression,
            "feat_stage_transformers": [
                t.__class__.__name__ for t in self.feat_stage_transformers
            ],
            "cv_stage_transformers": [
                {
                    "fold_id": fd["fold_id"],
                    "transformers": [t.__class__.__name__ for t in fd["transformers"]],
                }
                for fd in self.cv_stage_transformers
            ],
        }

        with open(path / "feature_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: Path, config: PreprocessingConfig) -> "FeaturePipeline":
        """
        Load pipeline from disk.

        Args:
            path: Directory with saved artifacts
            config: Preprocessing configuration

        Returns:
            Loaded FeaturePipeline
        """
        pipeline = cls(config)

        # Load feat_stage transformers
        feat_path = path / "feat_stage.pkl"
        if feat_path.exists():
            with open(feat_path, "rb") as f:
                pipeline.feat_stage_transformers = pickle.load(f)

        # Load cv_stage transformers
        cv_files = sorted(path.glob("cv_stage_fold*.pkl"))
        for cv_file in cv_files:
            with open(cv_file, "rb") as f:
                pipeline.cv_stage_transformers.append(pickle.load(f))

        return pipeline

    def _create_transformer(self, transformer_name: str, stage: str):
        """
        Create transformer instance by name.

        Args:
            transformer_name: Name of transformer class
            stage: "feat" or "cv"

        Returns:
            Transformer instance

        Raises:
            ImportError: If transformer not found
        """
        # Import from appropriate module based on stage
        try:
            if stage == "feat":
                from kaggle_tools.preprocessing import feat_stage

                transformer_class = getattr(feat_stage, transformer_name)
            else:  # cv stage
                from kaggle_tools.preprocessing import cv_stage

                transformer_class = getattr(cv_stage, transformer_name)

            return transformer_class()

        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Transformer '{transformer_name}' not found in {stage}_stage module. "
                f"Error: {e}"
            )

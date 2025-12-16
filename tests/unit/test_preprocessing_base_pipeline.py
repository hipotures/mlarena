from pathlib import Path
import pickle

import pandas as pd
import pytest

from kaggle_tools.config_models import PreprocessingConfig
from kaggle_tools.preprocessing.base import BaseTransformer
from kaggle_tools.preprocessing.pipeline import FeaturePipeline


class AddConstant(BaseTransformer):
    """Simple transformer used in tests."""

    def __init__(self, value=1, **kwargs):
        super().__init__({"value": value})

    def fit(self, df: pd.DataFrame) -> "AddConstant":
        self.fitted = True
        self.value = self.config.get("value", 1)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        result = df.copy()
        result["const"] = self.value
        return result


def test_base_transformer_requires_fit():
    t = AddConstant()
    with pytest.raises(RuntimeError):
        t.transform(pd.DataFrame({"a": [1]}))

    out = t.fit_transform(pd.DataFrame({"a": [1]}))
    assert "const" in out.columns


def test_feature_pipeline_feat_and_cv(monkeypatch, tmp_path):
    # Register dummy transformer in both stages
    from kaggle_tools.preprocessing import feat_stage, cv_stage

    monkeypatch.setattr(feat_stage, "AddConstant", AddConstant, raising=False)
    monkeypatch.setattr(cv_stage, "AddConstant", AddConstant, raising=False)

    cfg = PreprocessingConfig(
        feat_stage={"enabled": True, "transformers": [{"type": "AddConstant", "value": 2}]},
        cv_stage={"enabled": True, "transformers": ["AddConstant"]},
    )
    pipeline = FeaturePipeline(cfg)

    train = pd.DataFrame({"a": [1, 2]})
    test = pd.DataFrame({"a": [3]})
    pipeline.fit_feat_stage(train)
    train_feat = pipeline.transform_feat_stage(train)
    test_feat = pipeline.transform_feat_stage(test)
    assert (train_feat["const"] == 2).all()
    assert (test_feat["const"] == 2).all()

    # CV stage per-fold transformers
    t_train, t_val = pipeline.fit_transform_cv_stage(train, train.iloc[[0]], fold_id=0)
    assert "const" in t_train.columns
    assert len(pipeline.cv_stage_transformers) == 1

    test_cv = pipeline.transform_cv_stage_inference(test, aggregation="mean")
    assert "const" in test_cv.columns

    # Save/load roundtrip
    save_dir = tmp_path / "pipe"
    pipeline.save(save_dir)
    loaded = FeaturePipeline.load(save_dir, cfg)
    assert len(loaded.cv_stage_transformers) == 1
    assert isinstance(loaded.feat_stage_transformers[0], AddConstant)


def test_feature_pipeline_handles_empty_and_all_aggregation(monkeypatch):
    from kaggle_tools.preprocessing import feat_stage, cv_stage

    monkeypatch.setattr(feat_stage, "AddConstant", AddConstant, raising=False)
    monkeypatch.setattr(cv_stage, "AddConstant", AddConstant, raising=False)

    cfg = PreprocessingConfig(
        feat_stage={"enabled": True, "transformers": ["AddConstant"]},
        cv_stage={"enabled": True, "transformers": ["AddConstant"]},
    )
    pipeline = FeaturePipeline(cfg)

    empty = pd.DataFrame(columns=["a"])
    pipeline.fit_feat_stage(empty)
    out = pipeline.transform_feat_stage(empty)
    assert "const" in out.columns

    # Multiple folds to exercise aggregation="all"
    pipeline.fit_transform_cv_stage(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}), fold_id=0)
    pipeline.fit_transform_cv_stage(pd.DataFrame({"a": [2]}), pd.DataFrame({"a": [2]}), fold_id=1)
    all_versions = pipeline.transform_cv_stage_inference(pd.DataFrame({"a": [3]}), aggregation="all")
    assert isinstance(all_versions, list)
    assert len(all_versions) == 2

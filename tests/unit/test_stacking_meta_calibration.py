import numpy as np
import pandas as pd
import pytest

from kaggle_tools.stacking.meta_learner import MetaLearner
from kaggle_tools.stacking.calibration import CalibratedPredictor


class DummyModel:
    def __init__(self, **kwargs):
        self.fitted = False
        self.kwargs = kwargs

    def fit(self, X, y):
        self.fitted = True
        self.X_shape = X.shape
        self.y = y
        return self

    def predict(self, X):
        # Simple mean-based prediction
        return np.full(len(X), np.mean(self.y))

    def predict_proba(self, X):
        probs = np.full((len(X), 2), 0.0)
        probs[:, 1] = 0.7
        probs[:, 0] = 0.3
        return probs


def test_meta_learner_fit_predict(monkeypatch):
    # Stub cross_val_score to avoid sklearn CV constraints
    calls = {}

    def fake_cv(model, X, y, cv, scoring):
        calls["cv"] = cv
        calls["shape"] = X.shape
        return np.array([0.8, 0.85])

    monkeypatch.setattr("kaggle_tools.stacking.meta_learner.cross_val_score", fake_cv)

    oof1 = pd.DataFrame({"m1": [0.1, 0.9, 0.4, 0.6]})
    oof2 = pd.DataFrame({"m2": [0.2, 0.8, 0.3, 0.7]})
    y = pd.Series([0, 1, 0, 1])

    meta = MetaLearner(meta_model_class=DummyModel, meta_model_params={"alpha": 1})
    meta.fit([oof1, oof2], y)
    assert calls["cv"] == 5
    assert calls["shape"] == (4, 2)

    test_preds = meta.predict([pd.DataFrame({"m1": [0.5]}), pd.DataFrame({"m2": [0.5]})])
    assert len(test_preds) == 1
    assert isinstance(test_preds.iloc[0], float)


def test_meta_learner_predict_requires_fit():
    meta = MetaLearner(meta_model_class=DummyModel)
    with pytest.raises(RuntimeError):
        meta.predict([pd.DataFrame({"m": [0.5]})])


def test_calibrated_predictor_isotonic_and_sigmoid():
    preds = pd.Series([0.1, 0.4, 0.6, 0.9])
    y = pd.Series([0, 0, 1, 1])

    # Isotonic
    calibrator = CalibratedPredictor(method="isotonic")
    calibrator.fit(preds, y)
    calibrated = calibrator.transform(preds)
    assert len(calibrated) == len(preds)

    # Sigmoid (Platt)
    sigmoid = CalibratedPredictor(method="sigmoid")
    sigmoid.fit(preds, y)
    calibrated_sig = sigmoid.transform(preds)
    assert len(calibrated_sig) == len(preds)


def test_calibrated_predictor_errors():
    with pytest.raises(ValueError):
        CalibratedPredictor(method="bad")

    cal = CalibratedPredictor()
    with pytest.raises(RuntimeError):
        cal.transform(pd.Series([0.1, 0.2]))

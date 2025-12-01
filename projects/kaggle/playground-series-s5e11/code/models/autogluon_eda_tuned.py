"""
Feature-engineered AutoGluon model with lightweight hyperparameter tuning.

Features come from EDA summary exp-20251118-004408. Hyperparameters are tuned
via AutoGluon's built-in search (local scheduler, auto searcher).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from kaggle_tools.config_models import ModelConfig


GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
GRADE_SCORE = {grade: len(GRADE_ORDER) - idx for idx, grade in enumerate(GRADE_ORDER)}
DTI_BUCKETS = {
    "bins": [0, 0.1, 0.2, 0.35, 0.5, 0.75, np.inf],
    "labels": ["<=10%", "10-20%", "20-35%", "35-50%", "50-75%", ">75%"],
}
CS_BUCKETS = {
    "bins": [0, 500, 580, 640, 700, 760, np.inf],
    "labels": ["<500", "500-580", "580-640", "640-700", "700-760", "760+"],
}


def get_default_config() -> Dict[str, Any]:
    return {
        "hyperparameters": {
            "presets": "best_quality",
            "time_limit": 6 * 3600,
            "use_gpu": False,
            "excluded_models": ["NN_TORCH"],
        },
        "model": {
            "hpo": {
                "hyperparameter_tune_kwargs": {
                    "num_trials": 20,
                    "scheduler": "local",
                    "searcher": "auto",
                },
                "hyperparameters": {
                    "GBM": [
                        {"extra_trees": True, "learning_rate": 0.08, "num_boost_round": 150},
                        {"learning_rate": 0.02, "num_boost_round": 400, "feature_fraction": 0.8},
                    ],
                    "XGB": [
                        {"eta": 0.05, "max_depth": 6, "subsample": 0.8},
                    ],
                    "CAT": [
                        {"depth": 8, "learning_rate": 0.04, "l2_leaf_reg": 5.0},
                    ],
                },
            }
        },
    }


def _drop_ignored(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    drop_cols = set(config.dataset.ignored_columns + [config.dataset.id_column])
    drop_cols.discard(config.dataset.target)
    return df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    safe_den = den.replace(0, pd.NA)
    ratio = num / safe_den
    return ratio.replace([np.inf, -np.inf], pd.NA).fillna(0)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    enriched["log_annual_income"] = np.log1p(enriched["annual_income"])
    enriched["log_loan_amount"] = np.log1p(enriched["loan_amount"])

    enriched["loan_to_income_ratio"] = _safe_ratio(enriched["loan_amount"], enriched["annual_income"])
    enriched["debt_burden_amount"] = enriched["debt_to_income_ratio"] * enriched["loan_amount"]
    enriched["interest_burden_amount"] = enriched["interest_rate"] * enriched["loan_amount"]
    enriched["income_per_credit_point"] = _safe_ratio(enriched["annual_income"], enriched["credit_score"])
    enriched["debt_service_ratio"] = _safe_ratio(enriched["debt_burden_amount"], enriched["annual_income"])

    enriched["credit_score_bucket"] = pd.cut(
        enriched["credit_score"],
        bins=CS_BUCKETS["bins"],
        labels=CS_BUCKETS["labels"],
        include_lowest=True,
    ).astype(str)
    enriched["dti_bucket"] = pd.cut(
        enriched["debt_to_income_ratio"],
        bins=DTI_BUCKETS["bins"],
        labels=DTI_BUCKETS["labels"],
        include_lowest=True,
    ).astype(str)

    grade_letter = enriched["grade_subgrade"].astype(str).str[0]
    subgrade_num = pd.to_numeric(enriched["grade_subgrade"].astype(str).str[1:], errors="coerce")
    enriched["grade_letter"] = grade_letter
    enriched["subgrade_num"] = subgrade_num.fillna(0)
    enriched["grade_letter_score"] = grade_letter.map(GRADE_SCORE).fillna(0)
    enriched["grade_rank"] = enriched["grade_letter_score"] * 10 + enriched["subgrade_num"]
    enriched["grade_high_quality"] = grade_letter.isin(["A", "B"]).astype(int)

    enriched["education_employment"] = (
        enriched["education_level"].astype(str) + "_" + enriched["employment_status"].astype(str)
    )
    enriched["purpose_grade"] = enriched["loan_purpose"].astype(str) + "_" + grade_letter.astype(str)

    return enriched


def preprocess(df: pd.DataFrame, config: ModelConfig, is_train: bool = True) -> pd.DataFrame:
    return _engineer_features(df)


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> Tuple[TabularPredictor, Dict[str, Any]]:
    features = _drop_ignored(train_df, config)
    train_data = features.copy()
    train_data[config.dataset.target] = train_df[config.dataset.target]
    tuning_data = None
    if val_df is not None:
        val_features = _drop_ignored(val_df, config)
        tuning_data = val_features.copy()
        tuning_data[config.dataset.target] = val_df[config.dataset.target]

    predictor = TabularPredictor(
        label=config.dataset.target,
        path=str(config.system.model_path),
        problem_type=config.dataset.problem_type,
        eval_metric=config.dataset.metric,
        verbosity=2,
    )

    hpo_cfg = config.model.get("hpo", {})
    tune_kwargs = hpo_cfg.get("hyperparameter_tune_kwargs")
    custom_hparams = hpo_cfg.get("hyperparameters")

    fit_kwargs = {
        "presets": config.hyperparameters.presets,
        "time_limit": config.hyperparameters.time_limit,
        "num_gpus": 1 if config.hyperparameters.use_gpu else 0,
        "hyperparameters": custom_hparams,
    }
    if config.hyperparameters.excluded_models:
        fit_kwargs["excluded_model_types"] = config.hyperparameters.excluded_models

    predictor.fit(
        train_data,
        tuning_data=tuning_data,
        hyperparameter_tune_kwargs=tune_kwargs,
        **fit_kwargs,
    )

    leaderboard = predictor.leaderboard(train_data, silent=True)
    local_cv = float(leaderboard.iloc[0]["score_val"]) if not leaderboard.empty else None
    summary = {"local_cv": local_cv}
    return predictor, summary


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    features = _drop_ignored(test_df, config)
    submission = pd.DataFrame()
    submission[config.dataset.id_column] = test_df[config.dataset.id_column]

    if config.dataset.submission_probas:
        preds = model.predict_proba(features, as_multiclass=False)
        if isinstance(preds, pd.DataFrame):
            submission[config.dataset.target] = preds.iloc[:, 1]
        else:
            submission[config.dataset.target] = preds
    else:
        submission[config.dataset.target] = model.predict(features)
    return submission

"""
AutoGluon baseline model with adversarial validation weights.
Uses the same config as autogluon_baseline but applies sample_weight from data/train_av_weights.csv (unless overridden).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor

from kaggle_tools.config_models import ModelConfig

WEIGHT_COL = "__sample_weight__"


def get_default_config() -> Dict[str, Any]:
    return {
        "hyperparameters": {
            "presets": "best",
            "time_limit": 36000,
            "use_gpu": False,
        },
        "model": {
            "leaderboard_rows": 10,
            "av_weights_path": "data/train_av_weights.csv",
        },
    }


def _drop_ignored(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    drop_cols = set(config.dataset.ignored_columns + [config.dataset.id_column])
    drop_cols.discard(config.dataset.target)
    return df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")


def _load_av_weights(train_df: pd.DataFrame, config: ModelConfig) -> Optional[pd.Series]:
    weights_path = config.model.get("av_weights_path") or "data/train_av_weights.csv"
    weights_path = Path(weights_path)
    if not weights_path.is_absolute():
        weights_path = config.system.project_root / weights_path
    if not weights_path.exists():
        return None
    try:
        av_df = pd.read_csv(weights_path)
    except Exception:
        return None
    id_col = config.dataset.id_column
    if id_col not in av_df.columns or "av_prob" not in av_df.columns:
        return None
    weight_map = dict(zip(av_df[id_col], av_df["av_prob"]))
    weights = train_df[id_col].map(weight_map)
    if weights.isnull().all():
        return None
    return weights.fillna(weights.mean())


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

    # Categorical fix: Convert categorical columns to category dtype
    # HARDCODED for s5e12 - these columns should be category dtype
    CATEGORICAL_COLUMNS = [
        'alcohol_consumption_per_week',
        'family_history_diabetes',
        'hypertension_history',
        'cardiovascular_history',
    ]

    print(f"[Categorical Fix] Converting {len(CATEGORICAL_COLUMNS)} columns to category dtype")
    for col in CATEGORICAL_COLUMNS:
        if col in train_data.columns:
            train_data[col] = train_data[col].astype('category')
            print(f"  ✓ train_data[{col}] → category")
        if tuning_data is not None and col in tuning_data.columns:
            tuning_data[col] = tuning_data[col].astype('category')
            print(f"  ✓ tuning_data[{col}] → category")

    av_weights = _load_av_weights(train_df, config)
    if av_weights is not None:
        train_data[WEIGHT_COL] = av_weights
        if tuning_data is not None:
            tuning_data[WEIGHT_COL] = av_weights.reindex(val_df.index).fillna(av_weights.mean()) if not av_weights.empty else av_weights

    predictor = TabularPredictor(
        label=config.dataset.target,
        path=str(config.system.model_path),
        problem_type=config.dataset.problem_type,
        eval_metric=config.dataset.metric,
        sample_weight=WEIGHT_COL if av_weights is not None else None,
        verbosity=2,
    )

    # Configure default hyperparameters for native categorical handling in boost models
    default_hyperparams = {

#        'CAT': {},  # CatBoost uses category dtype natively
        'CAT': {
    	    "ag_args_fit": {"num_gpus": 0},
        },
        'GBM': {"ag_args_fit": {"num_gpus": 0}},  # LightGBM uses category dtype natively        
        'XGB': {
            'enable_categorical': True,  # Enable XGBoost native categorical
            'tree_method': 'hist',       # Required for categorical support
            'max_cat_to_onehot': 1,      # Disable one-hot fallback
            "ag_args_fit": {"num_gpus": 0}
        },
    }

    fit_kwargs = {
        "presets": config.hyperparameters.presets,
        "time_limit": config.hyperparameters.time_limit,
        "num_gpus": 1 if config.hyperparameters.use_gpu else 0,
        #"num_gpus": 1,

    }

    fit_kwargs["hyperparameter_tune_kwargs"] = {
	'num_trials': 50,
	'scheduler': 'local',
	'searcher': 'auto',
    }


    if config.hyperparameters.excluded_models:
        fit_kwargs["excluded_model_types"] = config.hyperparameters.excluded_models
    included_models = getattr(config.hyperparameters, "included_model_types", None)
    if included_models:
        fit_kwargs["included_model_types"] = included_models
    hyper_dict = config.hyperparameters.model_dump(exclude_none=True)
    known_keys = {
        "presets",
        "time_limit",
        "use_gpu",
        "excluded_models",
        "included_model_types",
        "preset",
    }
    model_hparams = {k: v for k, v in hyper_dict.items() if k not in known_keys}

    # Merge user hyperparameters with default categorical hyperparameters
    if model_hparams:
        # Deep merge: update default_hyperparams with user's model_hparams
        for model_type, params in default_hyperparams.items():
            if model_type in model_hparams:
                # Merge user params into default params
                default_hyperparams[model_type].update(model_hparams[model_type])
        # Add any user model types not in defaults
        for model_type, params in model_hparams.items():
            if model_type not in default_hyperparams:
                default_hyperparams[model_type] = params
        fit_kwargs["hyperparameters"] = default_hyperparams
    else:
        fit_kwargs["hyperparameters"] = default_hyperparams

    predictor.fit(
        train_data,
        tuning_data=tuning_data,
        **fit_kwargs,
    )

    leaderboard = predictor.leaderboard(train_data, silent=True)
    local_cv_score = None
    if not leaderboard.empty and "score_val" in leaderboard:
        scores = leaderboard["score_val"].dropna()
        if not scores.empty:
            local_cv_score = float(scores.max())
    summary = {"local_cv_score": local_cv_score}
    return predictor, summary


def predict(
    model: TabularPredictor,
    test_df: pd.DataFrame,
    config: ModelConfig,
    artifacts: Optional[Any] = None,
) -> pd.DataFrame:
    features = _drop_ignored(test_df, config)

    # Categorical fix: Convert categorical columns to category dtype (same as training)
    # HARDCODED for s5e12 - these columns should be category dtype
    CATEGORICAL_COLUMNS = [
        'alcohol_consumption_per_week',
        'family_history_diabetes',
        'hypertension_history',
        'cardiovascular_history',
    ]

    for col in CATEGORICAL_COLUMNS:
        if col in features.columns:
            features[col] = features[col].astype('category')

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

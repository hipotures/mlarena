#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""subset_compat.py

Compatibility (subset vs complement): PSI + adversarial AUC.
Optional model-selection stability: rank correlation / hit@k / regret for boosting models.

Requires: numpy, pandas, scikit-learn, scipy
Optional (stability): lightgbm, xgboost, catboost
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from importlib.util import spec_from_file_location, module_from_spec
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Optional UI
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _HAVE_RICH = True
except Exception:
    _HAVE_RICH = False

# Optional boosters
try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except Exception:
    CatBoostRegressor = None
    CatBoostClassifier = None


# ----------------------------
# Config / EDA loading
# ----------------------------

def load_py_config(config_path: str) -> Any:
    config_path = os.path.abspath(config_path)
    spec = spec_from_file_location("project_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import config from: {config_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def load_eda_state(eda_json_path: str) -> Dict[str, Any]:
    with open(eda_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_train_profile_vars(eda_state: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return eda_state["modules"]["eda"]["payload"]["train_profile"]["summary"]["variables"]
    except KeyError as e:
        raise KeyError(
            "EDA JSON has unexpected structure. Expected: modules.eda.payload.train_profile.summary.variables"
        ) from e


def infer_feature_types_from_eda(train_vars: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    numeric, categorical = [], []
    for col, meta in train_vars.items():
        t = meta.get("type")
        if t == "Numeric":
            numeric.append(col)
        elif t == "Categorical":
            categorical.append(col)
    return numeric, categorical


def detect_time_like_columns(columns: List[str]) -> List[str]:
    patterns = ("date", "time", "timestamp", "datetime", "ts", "created", "updated")
    cols = []
    for c in columns:
        cl = c.lower()
        if any(p in cl for p in patterns):
            cols.append(c)
    return cols


# ----------------------------
# PSI
# ----------------------------

def _psi_from_proportions(p_ref: np.ndarray, p_sub: np.ndarray, eps: float = 1e-6) -> float:
    p_ref = np.clip(p_ref, eps, 1.0)
    p_sub = np.clip(p_sub, eps, 1.0)
    return float(np.sum((p_sub - p_ref) * np.log(p_sub / p_ref)))


def psi_numeric(ref: pd.Series, sub: pd.Series, bins: int = 10, eps: float = 1e-6) -> float:
    ref = ref.dropna()
    sub = sub.dropna()
    if ref.empty or sub.empty:
        return 0.0

    q = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref.values, q))
    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_bins = pd.cut(ref, bins=edges, include_lowest=True)
    sub_bins = pd.cut(sub, bins=edges, include_lowest=True)

    ref_counts = ref_bins.value_counts(sort=False).values.astype(float)
    sub_counts = sub_bins.value_counts(sort=False).values.astype(float)

    p_ref = ref_counts / max(ref_counts.sum(), 1.0)
    p_sub = sub_counts / max(sub_counts.sum(), 1.0)
    return _psi_from_proportions(p_ref, p_sub, eps=eps)


def psi_categorical(ref: pd.Series, sub: pd.Series, eps: float = 1e-6) -> float:
    ref = ref.astype("object")
    sub = sub.astype("object")

    categories = ref.value_counts(dropna=False).index.tolist()
    ref_mapped = ref.where(ref.isin(categories), other="__OTHER__")
    sub_mapped = sub.where(sub.isin(categories), other="__OTHER__")

    all_cats = list(dict.fromkeys(categories + ["__OTHER__"]))
    ref_dist = ref_mapped.value_counts(dropna=False).reindex(all_cats, fill_value=0).values.astype(float)
    sub_dist = sub_mapped.value_counts(dropna=False).reindex(all_cats, fill_value=0).values.astype(float)

    p_ref = ref_dist / max(ref_dist.sum(), 1.0)
    p_sub = sub_dist / max(sub_dist.sum(), 1.0)
    return _psi_from_proportions(p_ref, p_sub, eps=eps)


# ----------------------------
# Adversarial validation
# ----------------------------

def build_adv_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=200,
        n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def adversarial_auc(
    X_sub: pd.DataFrame,
    X_ref: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    seed: int = 42,
    n_splits: int = 5,
    max_rows_per_side: int = 200_000,
) -> float:
    n_sub, n_ref = len(X_sub), len(X_ref)
    if n_sub < 2 or n_ref < 2:
        return float("nan")

    rng = np.random.default_rng(seed)
    m = min(n_sub, n_ref, max_rows_per_side)
    if m < 200:
        return float("nan")

    sub_idx = rng.choice(n_sub, size=m, replace=False)
    ref_idx = rng.choice(n_ref, size=m, replace=False)

    X = pd.concat([X_sub.iloc[sub_idx], X_ref.iloc[ref_idx]], axis=0, ignore_index=True)
    y = np.concatenate([np.ones(m, dtype=int), np.zeros(m, dtype=int)])

    used_num = [c for c in numeric_cols if c in X.columns]
    used_cat = [c for c in categorical_cols if c in X.columns]

    pipe = build_adv_pipeline(used_num, used_cat)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    aucs: List[float] = []
    for tr, te in cv.split(X, y):
        pipe.fit(X.iloc[tr], y[tr])
        proba = pipe.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], proba))

    return float(np.mean(aucs))

# ----------------------------
# Subset sampling / stratify
# ----------------------------

def make_stratify_labels(y: pd.Series, problem_type: str, n_bins: int = 10) -> Optional[np.ndarray]:
    if y.isna().any():
        return None
    pt = (problem_type or "").lower()
    if pt in ("binary", "multiclass", "classification"):
        return y.values
    # regression: quantile bins
    nunique = y.nunique(dropna=True)
    if nunique <= min(20, max(2, n_bins)):
        return y.values
    try:
        bins = pd.qcut(y, q=min(n_bins, nunique), duplicates="drop")
        return bins.astype(str).values
    except Exception:
        return None


def stratified_sample_indices(total_n: int, sample_n: int, stratify_labels: Optional[np.ndarray], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(total_n)
    if stratify_labels is None:
        return rng.choice(idx, size=sample_n, replace=False)

    labels = np.asarray(stratify_labels)
    if len(labels) != total_n:
        raise ValueError(f"stratify_labels length {len(labels)} != total_n {total_n}")

    uniq, counts = np.unique(labels, return_counts=True)
    desired = np.floor(counts / counts.sum() * sample_n).astype(int)

    diff = sample_n - desired.sum()
    if diff > 0:
        order = np.argsort(-counts)
        for k in range(diff):
            desired[order[k % len(order)]] += 1
    elif diff < 0:
        order = np.argsort(counts)
        for k in range(-diff):
            j = order[k % len(order)]
            if desired[j] > 0:
                desired[j] -= 1

    chosen: List[int] = []
    for u, d in zip(uniq, desired):
        if d <= 0:
            continue
        stratum_idx = idx[labels == u]
        if len(stratum_idx) <= d:
            chosen.extend(stratum_idx.tolist())
        else:
            chosen.extend(rng.choice(stratum_idx, size=d, replace=False).tolist())

    chosen = np.asarray(chosen, dtype=int)
    if len(chosen) < sample_n:
        remaining = np.setdiff1d(idx, chosen, assume_unique=False)
        extra = rng.choice(remaining, size=sample_n - len(chosen), replace=False)
        chosen = np.concatenate([chosen, extra])
    elif len(chosen) > sample_n:
        chosen = rng.choice(chosen, size=sample_n, replace=False)

    return chosen


# ----------------------------
# Compatibility evaluation
# ----------------------------

@dataclass
class CompatThresholds:
    adv_auc_max: float = 0.55
    psi_max: float = 0.10
    psi_bad_frac_max: float = 0.05
    target_psi_max: float = 0.05


@dataclass
class CompatResult:
    fraction: float
    n_subset: int
    n_rest: int
    status: str
    adv_auc: Optional[float]
    max_psi: Optional[float]
    bad_psi_frac: Optional[float]
    target_psi: Optional[float]


def compute_psi_map(df_sub: pd.DataFrame, df_rest: pd.DataFrame, num_cols: List[str], cat_cols: List[str], bins: int) -> Dict[str, float]:
    psi_map: Dict[str, float] = {}
    for c in num_cols:
        if c in df_sub.columns and c in df_rest.columns:
            psi_map[c] = psi_numeric(df_rest[c], df_sub[c], bins=bins)
    for c in cat_cols:
        if c in df_sub.columns and c in df_rest.columns:
            psi_map[c] = psi_categorical(df_rest[c], df_sub[c])
    return psi_map


def evaluate_compatibility(
    df: pd.DataFrame,
    target_col: str,
    ignored_cols: List[str],
    problem_type: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    thresholds: CompatThresholds,
    seed: int,
    fractions: List[float],
    psi_bins: int,
    adv_cv_splits: int,
    adv_max_rows_per_side: int,
    min_rest_rows: int,
) -> List[CompatResult]:
    n_total = len(df)
    ignored = set(ignored_cols + [target_col])

    used_num = [c for c in numeric_cols if c not in ignored and c in df.columns]
    used_cat = [c for c in categorical_cols if c not in ignored and c in df.columns]

    # fallback typing for columns not present in EDA typing
    for c in df.columns:
        if c in ignored:
            continue
        if c not in used_num and c not in used_cat:
            if pd.api.types.is_numeric_dtype(df[c]):
                used_num.append(c)
            else:
                used_cat.append(c)

    y = df[target_col]
    strat = make_stratify_labels(y, problem_type=problem_type, n_bins=10)

    results: List[CompatResult] = []

    for frac in fractions:
        n_sub = int(round(n_total * frac))
        n_sub = max(1, min(n_sub, n_total))
        n_rest = n_total - n_sub

        if frac == 1.0:
            results.append(CompatResult(frac, n_sub, n_rest, "PASS", None, 0.0, 0.0, 0.0))
            continue

        if n_rest < min_rest_rows:
            # proceed, but metrics can be noisy
            pass

        sub_idx = stratified_sample_indices(total_n=n_total, sample_n=n_sub, stratify_labels=strat, seed=seed + int(frac * 10_000))
        mask = np.zeros(n_total, dtype=bool)
        mask[sub_idx] = True

        df_sub = df.loc[mask].reset_index(drop=True)
        df_rest = df.loc[~mask].reset_index(drop=True)

        psi_map = compute_psi_map(df_sub, df_rest, used_num, used_cat, bins=psi_bins)
        psi_vals = np.array(list(psi_map.values()), dtype=float) if psi_map else np.array([], dtype=float)

        max_psi = float(np.nanmax(psi_vals)) if psi_vals.size else float("nan")
        bad_frac = float(np.mean(psi_vals > thresholds.psi_max)) if psi_vals.size else float("nan")

        # target PSI
        if pd.api.types.is_numeric_dtype(df[target_col]):
            target_psi = psi_numeric(df_rest[target_col], df_sub[target_col], bins=psi_bins)
        else:
            target_psi = psi_categorical(df_rest[target_col], df_sub[target_col])

        # adversarial AUC
        X_sub = df_sub[used_num + used_cat]
        X_rest = df_rest[used_num + used_cat]
        adv_auc = adversarial_auc(
            X_sub,
            X_rest,
            numeric_cols=used_num,
            categorical_cols=used_cat,
            seed=seed + int(frac * 10_000),
            n_splits=adv_cv_splits,
            max_rows_per_side=adv_max_rows_per_side,
        )

        checks = []
        checks.append((not math.isnan(adv_auc)) and (adv_auc <= thresholds.adv_auc_max))
        checks.append((not math.isnan(max_psi)) and (max_psi <= thresholds.psi_max))
        checks.append((not math.isnan(bad_frac)) and (bad_frac <= thresholds.psi_bad_frac_max))
        checks.append(target_psi <= thresholds.target_psi_max)

        status = "PASS" if all(checks) else "FAIL"

        results.append(CompatResult(
            fraction=frac,
            n_subset=n_sub,
            n_rest=n_rest,
            status=status,
            adv_auc=float(adv_auc) if not math.isnan(adv_auc) else None,
            max_psi=float(max_psi) if not math.isnan(max_psi) else None,
            bad_psi_frac=float(bad_frac) if not math.isnan(bad_frac) else None,
            target_psi=float(target_psi),
        ))

    return results


def render_compat_table(results: List[CompatResult]) -> None:
    if _HAVE_RICH:
        console = Console()
        t = Table(title="Subset Compatibility Analysis", box=box.ROUNDED)
        t.add_column("Fraction", justify="right")
        t.add_column("N Subset", justify="right")
        t.add_column("N Rest", justify="right")
        t.add_column("Status", justify="center")
        t.add_column("Adv AUC", justify="right")
        t.add_column("Max PSI", justify="right")
        t.add_column("Bad PSI %", justify="right")
        t.add_column("Target PSI", justify="right")
        for r in results:
            t.add_row(
                f"{r.fraction:.4f}",
                str(r.n_subset),
                str(r.n_rest),
                r.status,
                "-" if r.adv_auc is None else f"{r.adv_auc:.8f}",
                "-" if r.max_psi is None else f"{r.max_psi:.8f}",
                "-" if r.bad_psi_frac is None else f"{r.bad_psi_frac:.8f}",
                "-" if r.target_psi is None else f"{r.target_psi:.8f}",
            )
        console.print(t)
    else:
        print("fraction\tn_subset\tn_rest\tstatus\tadv_auc\tmax_psi\tbad_psi_frac\ttarget_psi")
        for r in results:
            print(
                f"{r.fraction:.4f}\t{r.n_subset}\t{r.n_rest}\t{r.status}\t"
                f"{'' if r.adv_auc is None else f'{r.adv_auc:.6f}'}\t"
                f"{'' if r.max_psi is None else f'{r.max_psi:.6f}'}\t"
                f"{'' if r.bad_psi_frac is None else f'{r.bad_psi_frac:.6f}'}\t"
                f"{'' if r.target_psi is None else f'{r.target_psi:.6f}'}"
            )

# ----------------------------
# Stability evaluation
# ----------------------------

@dataclass
class StabilityThresholds:
    spearman_p10_min: float = 0.90
    kendall_p10_min: float = 0.80
    hitk_mean_min: float = 0.80
    regret_p90_max_abs: Optional[float] = None
    regret_p90_max_rel: float = 0.01  # 1% of |best_full| by default


@dataclass
class StabilityResult:
    fraction: float
    model: str
    n_subset: int
    spearman_p10: float
    kendall_p10: float
    hitk_mean: float
    regret_p90: float
    status: str


def _metric_is_higher_better(metric_name: str, problem_type: str) -> bool:
    m = (metric_name or "").lower()
    # common loss metrics
    if m in (
        "root_mean_squared_error",
        "rmse",
        "mean_squared_error",
        "mse",
        "mean_absolute_error",
        "mae",
        "log_loss",
        "cross_entropy",
    ):
        return False
    # r2 / auc / accuracy / f1 etc
    if m in ("r2", "roc_auc", "auc", "accuracy", "acc", "f1", "f1_score"):
        return True
    # default by problem type
    pt = (problem_type or "").lower()
    if pt in ("binary", "multiclass", "classification"):
        return True
    return False


def _compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    problem_type: str,
    proba: bool = False,
) -> float:
    m = (metric_name or "").lower()
    pt = (problem_type or "").lower()

    # local imports to keep header minimal
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        log_loss,
        roc_auc_score as _roc_auc,
        accuracy_score,
        f1_score,
    )

    if m in ("root_mean_squared_error", "rmse"):
        try:
            return float(mean_squared_error(y_true, y_pred, squared=False))
        except TypeError:
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if m in ("mean_squared_error", "mse"):
        try:
            return float(mean_squared_error(y_true, y_pred, squared=True))
        except TypeError:
            return float(mean_squared_error(y_true, y_pred))
    if m in ("mean_absolute_error", "mae"):
        return float(mean_absolute_error(y_true, y_pred))
    if m == "r2":
        return float(r2_score(y_true, y_pred))

    if pt in ("binary", "classification"):
        if m in ("roc_auc", "auc"):
            return float(_roc_auc(y_true, y_pred))  # y_pred = proba
        if m in ("log_loss", "cross_entropy"):
            # y_pred = proba
            return float(log_loss(y_true, np.clip(y_pred, 1e-6, 1 - 1e-6)))
        if m in ("accuracy", "acc"):
            return float(accuracy_score(y_true, (y_pred >= 0.5).astype(int)))
        if m in ("f1", "f1_score"):
            return float(f1_score(y_true, (y_pred >= 0.5).astype(int)))

    if pt in ("multiclass",):
        if m in ("log_loss", "cross_entropy"):
            return float(log_loss(y_true, y_pred))  # y_pred = proba matrix
        if m in ("accuracy", "acc"):
            return float(accuracy_score(y_true, np.argmax(y_pred, axis=1)))
        if m in ("f1", "f1_score"):
            return float(f1_score(y_true, np.argmax(y_pred, axis=1), average="macro"))

    # fallback
    return float(mean_squared_error(y_true, y_pred, squared=False))


def _effective_score(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    return values if higher_is_better else -values


def _sample_params_lgbm(rng: np.random.Generator, problem_type: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "learning_rate": float(rng.uniform(0.01, 0.2)),
        "n_estimators": int(rng.integers(200, 2000)),
        "num_leaves": int(rng.integers(16, 256)),
        "max_depth": int(rng.integers(-1, 16)),
        "min_child_samples": int(rng.integers(5, 100)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_lambda": float(rng.uniform(0.0, 5.0)),
    }
    return params


def _sample_params_xgb(rng: np.random.Generator, problem_type: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "learning_rate": float(rng.uniform(0.01, 0.2)),
        "n_estimators": int(rng.integers(200, 2000)),
        "max_depth": int(rng.integers(3, 12)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "min_child_weight": float(rng.uniform(0.5, 20.0)),
        "reg_lambda": float(rng.uniform(0.0, 5.0)),
    }
    return params


def _sample_params_cat(rng: np.random.Generator, problem_type: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "learning_rate": float(rng.uniform(0.01, 0.2)),
        "iterations": int(rng.integers(300, 3000)),
        "depth": int(rng.integers(4, 10)),
        "l2_leaf_reg": float(rng.uniform(1.0, 10.0)),
        "bagging_temperature": float(rng.uniform(0.0, 1.0)),
        "random_strength": float(rng.uniform(0.0, 2.0)),
    }
    return params


def _fit_preprocessor(X: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_tf = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_tf = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    pre = ColumnTransformer(
        transformers=[("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)],
        remainder="drop",
        sparse_threshold=0.3,
    )
    pre.fit(X)
    return pre


def _train_eval_lgbm(
    X_tr,
    y_tr,
    X_ev,
    y_ev,
    params: Dict[str, Any],
    problem_type: str,
    metric_name: str,
    threads: int,
    seed: int,
) -> float:
    if lgb is None:
        raise RuntimeError("lightgbm not installed")

    pt = (problem_type or "").lower()
    if pt in ("binary", "classification"):
        model = lgb.LGBMClassifier(
            random_state=seed,
            n_jobs=threads,
            verbosity=-1,
            **params,
        )
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_ev)[:, 1]
        return _compute_metric(y_ev, proba, metric_name, problem_type, proba=True)

    if pt in ("multiclass",):
        model = lgb.LGBMClassifier(
            random_state=seed,
            n_jobs=threads,
            verbosity=-1,
            **params,
        )
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_ev)
        return _compute_metric(y_ev, proba, metric_name, problem_type, proba=True)

    model = lgb.LGBMRegressor(
        random_state=seed,
        n_jobs=threads,
        verbosity=-1,
        **params,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_ev)
    return _compute_metric(y_ev, pred, metric_name, problem_type)


def _train_eval_xgb(
    X_tr,
    y_tr,
    X_ev,
    y_ev,
    params: Dict[str, Any],
    problem_type: str,
    metric_name: str,
    threads: int,
    seed: int,
) -> float:
    if xgb is None:
        raise RuntimeError("xgboost not installed")

    pt = (problem_type or "").lower()

    common = dict(
        random_state=seed,
        n_jobs=threads,
        tree_method="hist",
        verbosity=0,
        **params,
    )

    if pt in ("binary", "classification"):
        model = xgb.XGBClassifier(**common)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_ev)[:, 1]
        return _compute_metric(y_ev, proba, metric_name, problem_type, proba=True)

    if pt in ("multiclass",):
        model = xgb.XGBClassifier(**common)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_ev)
        return _compute_metric(y_ev, proba, metric_name, problem_type, proba=True)

    model = xgb.XGBRegressor(**common)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_ev)
    return _compute_metric(y_ev, pred, metric_name, problem_type)


def _train_eval_cat(
    X_tr_df: pd.DataFrame,
    y_tr: np.ndarray,
    X_ev_df: pd.DataFrame,
    y_ev: np.ndarray,
    params: Dict[str, Any],
    problem_type: str,
    metric_name: str,
    threads: int,
    seed: int,
    cat_cols: List[str],
) -> float:
    if (CatBoostRegressor is None) or (CatBoostClassifier is None):
        raise RuntimeError("catboost not installed")

    # Simple handling: fill NaNs, keep cat cols as strings
    X_tr = X_tr_df.copy()
    X_ev = X_ev_df.copy()

    for c in cat_cols:
        if c in X_tr.columns:
            X_tr[c] = X_tr[c].astype("object").fillna("__MISSING__").astype(str)
        if c in X_ev.columns:
            X_ev[c] = X_ev[c].astype("object").fillna("__MISSING__").astype(str)

    cat_features = [X_tr.columns.get_loc(c) for c in cat_cols if c in X_tr.columns]

    pt = (problem_type or "").lower()
    if pt in ("binary", "classification"):
        model = CatBoostClassifier(
            random_seed=seed,
            thread_count=threads,
            verbose=False,
            loss_function="Logloss",
            **params,
        )
        model.fit(X_tr, y_tr, cat_features=cat_features)
        proba = model.predict_proba(X_ev)[:, 1]
        return _compute_metric(y_ev, proba, metric_name, problem_type, proba=True)

    if pt in ("multiclass",):
        model = CatBoostClassifier(
            random_seed=seed,
            thread_count=threads,
            verbose=False,
            loss_function="MultiClass",
            **params,
        )
        model.fit(X_tr, y_tr, cat_features=cat_features)
        proba = model.predict_proba(X_ev)
        return _compute_metric(y_ev, proba, metric_name, problem_type, proba=True)

    model = CatBoostRegressor(
        random_seed=seed,
        thread_count=threads,
        verbose=False,
        loss_function="RMSE",
        **params,
    )
    model.fit(X_tr, y_tr, cat_features=cat_features)
    pred = model.predict(X_ev)
    return _compute_metric(y_ev, pred, metric_name, problem_type)


def evaluate_stability(
    df: pd.DataFrame,
    target_col: str,
    ignored_cols: List[str],
    problem_type: str,
    metric_name: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    models: List[str],
    configs_per_model: int,
    repeats: int,
    topk: int,
    holdout_frac: float,
    max_train_rows: Optional[int],
    max_holdout_rows: Optional[int],
    preprocess_fit: str,
    seed: int,
    threads: int,
    thresholds: StabilityThresholds,
    fractions: List[float],
) -> Tuple[List[StabilityResult], Dict[str, Any]]:

    ignored = set(ignored_cols + [target_col])
    used_num = [c for c in numeric_cols if c not in ignored and c in df.columns]
    used_cat = [c for c in categorical_cols if c not in ignored and c in df.columns]
    for c in df.columns:
        if c in ignored:
            continue
        if c not in used_num and c not in used_cat:
            if pd.api.types.is_numeric_dtype(df[c]):
                used_num.append(c)
            else:
                used_cat.append(c)

    X_all = df[used_num + used_cat]
    y_all = df[target_col]

    strat = make_stratify_labels(y_all, problem_type=problem_type, n_bins=10)

    X_pool, X_eval, y_pool, y_eval = train_test_split(
        X_all,
        y_all,
        test_size=holdout_frac,
        random_state=seed,
        stratify=strat if strat is not None else None,
    )

    rng = np.random.default_rng(seed)

    # cap rows
    if max_train_rows is not None and len(X_pool) > max_train_rows:
        idx = rng.choice(len(X_pool), size=max_train_rows, replace=False)
        X_pool = X_pool.iloc[idx].reset_index(drop=True)
        y_pool = y_pool.iloc[idx].reset_index(drop=True)

    if max_holdout_rows is not None and len(X_eval) > max_holdout_rows:
        idx = rng.choice(len(X_eval), size=max_holdout_rows, replace=False)
        X_eval = X_eval.iloc[idx].reset_index(drop=True)
        y_eval = y_eval.iloc[idx].reset_index(drop=True)

    higher_is_better = _metric_is_higher_better(metric_name, problem_type)

    # Preprocess for lgbm/xgb if we fit on full pool
    pre_full: Optional[ColumnTransformer] = None
    if preprocess_fit.lower() == "full":
        pre_full = _fit_preprocessor(X_pool, used_num, used_cat)
        X_pool_tr = pre_full.transform(X_pool)
        X_eval_tr = pre_full.transform(X_eval)
    else:
        X_pool_tr, X_eval_tr = None, None

    meta: Dict[str, Any] = {
        "metric": metric_name,
        "higher_is_better": higher_is_better,
        "n_pool": int(len(X_pool)),
        "n_eval": int(len(X_eval)),
        "models": models,
        "configs_per_model": int(configs_per_model),
        "repeats": int(repeats),
        "topk": int(topk),
        "preprocess_fit": preprocess_fit,
    }

    results: List[StabilityResult] = []

    # Generate configs per model
    model_configs: Dict[str, List[Dict[str, Any]]] = {}
    for m in models:
        mrng = np.random.default_rng(seed + hash(m) % 10_000)
        cfgs: List[Dict[str, Any]] = []
        for _ in range(configs_per_model):
            if m == "lgbm":
                cfgs.append(_sample_params_lgbm(mrng, problem_type))
            elif m == "xgb":
                cfgs.append(_sample_params_xgb(mrng, problem_type))
            elif m == "cat":
                cfgs.append(_sample_params_cat(mrng, problem_type))
            else:
                raise ValueError(f"Unknown model: {m}")
        model_configs[m] = cfgs

    # Compute full-reference vector F per model
    full_scores: Dict[str, np.ndarray] = {}
    for m in models:
        cfgs = model_configs[m]
        scores: List[float] = []
        for j, p in enumerate(cfgs):
            local_seed = seed + j
            if m == "lgbm":
                if preprocess_fit.lower() == "full":
                    sc = _train_eval_lgbm(X_pool_tr, y_pool.values, X_eval_tr, y_eval.values, p, problem_type, metric_name, threads, local_seed)
                else:
                    pre = _fit_preprocessor(X_pool, used_num, used_cat)
                    sc = _train_eval_lgbm(pre.transform(X_pool), y_pool.values, pre.transform(X_eval), y_eval.values, p, problem_type, metric_name, threads, local_seed)
            elif m == "xgb":
                if preprocess_fit.lower() == "full":
                    sc = _train_eval_xgb(X_pool_tr, y_pool.values, X_eval_tr, y_eval.values, p, problem_type, metric_name, threads, local_seed)
                else:
                    pre = _fit_preprocessor(X_pool, used_num, used_cat)
                    sc = _train_eval_xgb(pre.transform(X_pool), y_pool.values, pre.transform(X_eval), y_eval.values, p, problem_type, metric_name, threads, local_seed)
            else:  # cat
                sc = _train_eval_cat(X_pool, y_pool.values, X_eval, y_eval.values, p, problem_type, metric_name, threads, local_seed, used_cat)
            scores.append(sc)
        full_scores[m] = np.asarray(scores, dtype=float)

    # Threshold for regret
    # If absolute threshold not provided, use relative to |best_full| (on original metric scale)
    best_full_value_global = None

    for frac in fractions:
        n_sub = int(round(len(X_pool) * frac))
        n_sub = max(1, min(n_sub, len(X_pool)))

        # deterministic seeds per frac
        frac_seed_base = seed + int(frac * 10_000)

        for m in models:
            F = full_scores[m]
            eff_F = _effective_score(F, higher_is_better)
            best_full_idx = int(np.argmax(eff_F))
            best_full_value = float(F[best_full_idx])

            if best_full_value_global is None:
                best_full_value_global = best_full_value

            regret_thr = thresholds.regret_p90_max_abs
            if regret_thr is None:
                regret_thr = thresholds.regret_p90_max_rel * max(1e-12, abs(best_full_value))

            spearmans: List[float] = []
            kendalls: List[float] = []
            hits: List[float] = []
            regrets: List[float] = []

            cfgs = model_configs[m]

            for r in range(repeats):
                rseed = frac_seed_base + r
                sub_idx = stratified_sample_indices(total_n=len(X_pool), sample_n=n_sub, stratify_labels=make_stratify_labels(y_pool, problem_type, 10), seed=rseed)

                if m in ("lgbm", "xgb"):
                    if preprocess_fit.lower() == "full":
                        X_tr_sub = X_pool_tr[sub_idx]
                        X_ev = X_eval_tr
                    else:
                        pre = _fit_preprocessor(X_pool.iloc[sub_idx], used_num, used_cat)
                        X_tr_sub = pre.transform(X_pool.iloc[sub_idx])
                        X_ev = pre.transform(X_eval)

                # compute Y vector
                Ys: List[float] = []
                for j, p in enumerate(cfgs):
                    local_seed = rseed + j
                    if m == "lgbm":
                        if preprocess_fit.lower() == "full":
                            sc = _train_eval_lgbm(X_tr_sub, y_pool.iloc[sub_idx].values, X_ev, y_eval.values, p, problem_type, metric_name, threads, local_seed)
                        else:
                            sc = _train_eval_lgbm(X_tr_sub, y_pool.iloc[sub_idx].values, X_ev, y_eval.values, p, problem_type, metric_name, threads, local_seed)
                    elif m == "xgb":
                        sc = _train_eval_xgb(X_tr_sub, y_pool.iloc[sub_idx].values, X_ev, y_eval.values, p, problem_type, metric_name, threads, local_seed)
                    else:
                        sc = _train_eval_cat(X_pool.iloc[sub_idx], y_pool.iloc[sub_idx].values, X_eval, y_eval.values, p, problem_type, metric_name, threads, local_seed, used_cat)
                    Ys.append(sc)

                Y = np.asarray(Ys, dtype=float)
                eff_Y = _effective_score(Y, higher_is_better)

                # rank correlations
                rho = spearmanr(eff_F, eff_Y).correlation
                tau = kendalltau(eff_F, eff_Y).correlation
                spearmans.append(float(rho) if rho is not None and not math.isnan(rho) else 0.0)
                kendalls.append(float(tau) if tau is not None and not math.isnan(tau) else 0.0)

                # hit@k
                topk_idx = np.argsort(-eff_Y)[: max(1, topk)]
                hits.append(1.0 if best_full_idx in set(topk_idx.tolist()) else 0.0)

                # regret on original metric scale
                best_sub_idx = int(np.argmax(eff_Y))
                chosen = float(Y[best_sub_idx])
                if higher_is_better:
                    regret = best_full_value - chosen
                else:
                    regret = chosen - best_full_value
                regrets.append(float(regret))

            spearman_p10 = float(np.quantile(spearmans, 0.10))
            kendall_p10 = float(np.quantile(kendalls, 0.10))
            hitk_mean = float(np.mean(hits))
            regret_p90 = float(np.quantile(regrets, 0.90))

            checks = [
                spearman_p10 >= thresholds.spearman_p10_min,
                kendall_p10 >= thresholds.kendall_p10_min,
                hitk_mean >= thresholds.hitk_mean_min,
                regret_p90 <= float(regret_thr),
            ]
            status = "PASS" if all(checks) else "FAIL"

            results.append(StabilityResult(
                fraction=frac,
                model=m,
                n_subset=n_sub,
                spearman_p10=spearman_p10,
                kendall_p10=kendall_p10,
                hitk_mean=hitk_mean,
                regret_p90=regret_p90,
                status=status,
            ))

    meta["regret_threshold_mode"] = "abs" if thresholds.regret_p90_max_abs is not None else "rel"
    meta["regret_threshold_rel"] = thresholds.regret_p90_max_rel

    return results, meta


def render_stability_table(stability_results: List[StabilityResult], metric_name: str) -> None:
    if not stability_results:
        return

    if _HAVE_RICH:
        console = Console()
        t = Table(title=f"Model-Selection Stability (metric={metric_name})", box=box.ROUNDED)
        t.add_column("Fraction", justify="right")
        t.add_column("Model", justify="left")
        t.add_column("N Subset", justify="right")
        t.add_column("Spearman p10", justify="right")
        t.add_column("Kendall p10", justify="right")
        t.add_column("Hit@k mean", justify="right")
        t.add_column("Regret p90", justify="right")
        t.add_column("Status", justify="center")
        for r in stability_results:
            t.add_row(
                f"{r.fraction:.4f}",
                r.model,
                str(r.n_subset),
                f"{r.spearman_p10:.4f}",
                f"{r.kendall_p10:.4f}",
                f"{r.hitk_mean:.4f}",
                f"{r.regret_p90:.6f}",
                r.status,
            )
        console.print(t)
    else:
        print("fraction\tmodel\tn_subset\tspearman_p10\tkendall_p10\thitk_mean\tregret_p90\tstatus")
        for r in stability_results:
            print(f"{r.fraction:.4f}\t{r.model}\t{r.n_subset}\t{r.spearman_p10:.4f}\t{r.kendall_p10:.4f}\t{r.hitk_mean:.4f}\t{r.regret_p90:.6f}\t{r.status}")

# ----------------------------
# CLI / main
# ----------------------------

def parse_fractions(step: float, min_fraction: float, extra: Optional[str] = None) -> List[float]:
    if extra:
        fr = []
        for p in extra.split(','):
            p = p.strip()
            if not p:
                continue
            fr.append(float(p))
        fr = sorted(set(fr), reverse=True)
        return fr
    fracs = []
    x = 1.0
    while x >= min_fraction - 1e-12:
        fracs.append(round(x, 4))
        x -= step
    fracs = [float(f) for f in fracs if f > 0]
    if fracs[-1] != round(min_fraction, 4):
        fracs.append(round(min_fraction, 4))
    return fracs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-path", required=True)
    ap.add_argument("--eda-json", required=True)
    ap.add_argument("--config-py", required=True)
    ap.add_argument("--out-json", default=None)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=10)

    # compatibility params
    ap.add_argument("--step", type=float, default=0.10)
    ap.add_argument("--min-fraction", type=float, default=0.10)
    ap.add_argument("--fractions", default=None, help="Comma-separated fractions; overrides --step/--min-fraction")

    ap.add_argument("--psi-bins", type=int, default=10)
    ap.add_argument("--adv-cv-splits", type=int, default=5)
    ap.add_argument("--adv-max-rows-per-side", type=int, default=200000)
    ap.add_argument("--min-rest-rows", type=int, default=5000)

    ap.add_argument("--thr-adv-auc", type=float, default=0.55)
    ap.add_argument("--thr-psi-max", type=float, default=0.10)
    ap.add_argument("--thr-psi-bad-frac", type=float, default=0.05)
    ap.add_argument("--thr-target-psi", type=float, default=0.05)

    # stability params
    ap.add_argument("--enable-stability", action="store_true")
    ap.add_argument("--stability-models", default="lgbm,xgb")
    ap.add_argument("--stability-configs-per-model", type=int, default=30)
    ap.add_argument("--stability-repeats", type=int, default=5)
    ap.add_argument("--stability-topk", type=int, default=3)
    ap.add_argument("--stability-holdout-frac", type=float, default=0.20)
    ap.add_argument("--stability-max-train-rows", type=int, default=200000)
    ap.add_argument("--stability-max-holdout-rows", type=int, default=50000)
    ap.add_argument("--stability-preprocess-fit", choices=["full", "subset"], default="full")
    ap.add_argument("--stability-fractions", default=None, help="Comma-separated fractions for stability; default uses --fractions")

    ap.add_argument("--thr-spearman-p10", type=float, default=0.90)
    ap.add_argument("--thr-kendall-p10", type=float, default=0.80)
    ap.add_argument("--thr-hitk-mean", type=float, default=0.80)
    ap.add_argument("--thr-regret-rel-p90", type=float, default=0.01)
    ap.add_argument("--thr-regret-abs-p90", type=float, default=None)

    args = ap.parse_args()

    cfg = load_py_config(args.config_py)
    target_col = getattr(cfg, "TARGET_COLUMN", None)
    if not target_col:
        raise RuntimeError("TARGET_COLUMN missing in config.py")

    ignored_cols = list(getattr(cfg, "IGNORED_COLUMNS", []))
    problem_type = getattr(cfg, "AUTOGLUON_PROBLEM_TYPE", getattr(cfg, "PROBLEM_TYPE", "regression"))
    metric_name = getattr(cfg, "AUTOGLUON_EVAL_METRIC", getattr(cfg, "EVAL_METRIC", "root_mean_squared_error"))

    df = pd.read_csv(args.train_path)
    if target_col not in df.columns:
        raise RuntimeError(f"Target column '{target_col}' not found in train")

    eda_state = load_eda_state(args.eda_json)
    train_vars = extract_train_profile_vars(eda_state)
    num_cols, cat_cols = infer_feature_types_from_eda(train_vars)

    time_like = detect_time_like_columns(list(df.columns))
    if time_like and not _HAVE_RICH:
        print(f"[info] time-like columns detected: {time_like}", file=sys.stderr)

    fracs = parse_fractions(args.step, args.min_fraction, args.fractions)

    compat_thr = CompatThresholds(
        adv_auc_max=args.thr_adv_auc,
        psi_max=args.thr_psi_max,
        psi_bad_frac_max=args.thr_psi_bad_frac,
        target_psi_max=args.thr_target_psi,
    )

    compat_results = evaluate_compatibility(
        df=df,
        target_col=target_col,
        ignored_cols=ignored_cols,
        problem_type=problem_type,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        thresholds=compat_thr,
        seed=args.seed,
        fractions=fracs,
        psi_bins=args.psi_bins,
        adv_cv_splits=args.adv_cv_splits,
        adv_max_rows_per_side=args.adv_max_rows_per_side,
        min_rest_rows=args.min_rest_rows,
    )

    render_compat_table(compat_results)

    stability_results: List[StabilityResult] = []
    stability_meta: Dict[str, Any] = {}

    if args.enable_stability:
        st_fracs = parse_fractions(args.step, args.min_fraction, args.stability_fractions or args.fractions)
        models = [m.strip() for m in args.stability_models.split(',') if m.strip()]

        st_thr = StabilityThresholds(
            spearman_p10_min=args.thr_spearman_p10,
            kendall_p10_min=args.thr_kendall_p10,
            hitk_mean_min=args.thr_hitk_mean,
            regret_p90_max_abs=args.thr_regret_abs_p90,
            regret_p90_max_rel=args.thr_regret_rel_p90,
        )

        stability_results, stability_meta = evaluate_stability(
            df=df,
            target_col=target_col,
            ignored_cols=ignored_cols,
            problem_type=problem_type,
            metric_name=metric_name,
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            models=models,
            configs_per_model=args.stability_configs_per_model,
            repeats=args.stability_repeats,
            topk=args.stability_topk,
            holdout_frac=args.stability_holdout_frac,
            max_train_rows=args.stability_max_train_rows,
            max_holdout_rows=args.stability_max_holdout_rows,
            preprocess_fit=args.stability_preprocess_fit,
            seed=args.seed,
            threads=args.threads,
            thresholds=st_thr,
            fractions=st_fracs,
        )

        render_stability_table(stability_results, metric_name)

    if args.out_json:
        payload = {
            "meta": {
                "target": target_col,
                "ignored_cols": ignored_cols,
                "problem_type": problem_type,
                "metric": metric_name,
                "seed": args.seed,
            },
            "compatibility": {
                "thresholds": asdict(compat_thr),
                "results": [asdict(r) for r in compat_results],
            },
            "stability": {
                "enabled": bool(args.enable_stability),
                "meta": stability_meta,
                "thresholds": {
                    "spearman_p10_min": args.thr_spearman_p10,
                    "kendall_p10_min": args.thr_kendall_p10,
                    "hitk_mean_min": args.thr_hitk_mean,
                    "regret_p90_max_abs": args.thr_regret_abs_p90,
                    "regret_p90_max_rel": args.thr_regret_rel_p90,
                },
                "results": [asdict(r) for r in stability_results],
            },
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
subset_compat.py

Minimalna walidacja "kompatybilności" subsetu z pełnym train:
- PSI (per feature + target) subset vs reszta
- Adversarial validation AUC (subset vs reszta)

Wejście:
- --train-path: train.csv lub train.csv.gz
- --eda-json: np. state.json z EDA
- --config-py: config.py z projektu

Wyjście:
- tabela PASS/FAIL dla 100%, 90%, ..., 10%
- opcjonalnie JSON report z metrykami i top drifting features
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import concurrent.futures
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
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    log_loss,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table
from rich import box

# Suppress convergence warnings from LogisticRegression
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ----------------------------
# Optional boosting backends (used only when --enable-stability)
# ----------------------------

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostRegressor = None
    CatBoostClassifier = None


# ----------------------------
# Config / EDA loading
# ----------------------------

def load_py_config(config_path: str) -> Any:
    """Load config.py as a module-like object."""
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
    """
    Expected structure (per provided example):
    modules -> eda -> payload -> train_profile -> summary -> variables
    """
    try:
        return eda_state["modules"]["eda"]["payload"]["train_profile"]["summary"]["variables"]
    except KeyError as e:
        raise KeyError(
            "EDA JSON has unexpected structure. "
            "Expected modules.eda.payload.train_profile.summary.variables"
        ) from e


def infer_feature_types_from_eda(train_vars: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    numeric, categorical = [], []
    for col, meta in train_vars.items():
        t = meta.get("type")
        if t == "Numeric":
            numeric.append(col)
        elif t == "Categorical":
            categorical.append(col)
        # inne typy ignorujemy (ew. rozbudowa)
    return numeric, categorical


def detect_time_like_columns(columns: List[str]) -> List[str]:
    """
    Heurystyka: nazwy sugerujące czas.
    (Zostawione pod rozbudowę; domyślnie nic nie zmienia w kryteriach.)
    """
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
    """
    PSI for numeric feature using quantile bins defined on reference.
    """
    ref = ref.dropna()
    sub = sub.dropna()

    if ref.empty or sub.empty:
        return 0.0

    # quantile edges on reference
    q = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref.values, q))
    if len(edges) < 3:
        # feature almost constant
        return 0.0

    # include -inf/inf to catch extremes safely
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
    """
    PSI for categorical feature over categories observed in reference (+ OTHER bucket).
    """
    ref = ref.astype("object")
    sub = sub.astype("object")

    ref_counts = ref.value_counts(dropna=False)
    categories = ref_counts.index.tolist()

    # map unseen in reference -> OTHER
    sub_mapped = sub.where(sub.isin(categories), other="__OTHER__")
    ref_mapped = ref.where(ref.isin(categories), other="__OTHER__")

    # align distributions
    all_cats = list(dict.fromkeys(categories + ["__OTHER__"]))

    ref_dist = ref_mapped.value_counts(dropna=False).reindex(all_cats, fill_value=0).values.astype(float)
    sub_dist = sub_mapped.value_counts(dropna=False).reindex(all_cats, fill_value=0).values.astype(float)

    p_ref = ref_dist / max(ref_dist.sum(), 1.0)
    p_sub = sub_dist / max(sub_dist.sum(), 1.0)
    return _psi_from_proportions(p_ref, p_sub, eps=eps)


# ----------------------------
# Adversarial validation (AUC)
# ----------------------------

def build_adv_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    """
    Pipeline:
    - numeric: impute median
    - categorical: impute most_frequent + onehot
    - classifier: logistic regression
    """
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", categorical_tf, categorical_cols),
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

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def adversarial_auc(
    X_sub: pd.DataFrame,
    X_ref: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    seed: int = 42,
    n_splits: int = 5,
    max_rows_per_side: int = 200_000,
) -> float:
    """
    Compute cross-validated ROC AUC to separate subset vs reference (complement).
    Balances sizes by downsampling the larger side.
    """
    n_sub = len(X_sub)
    n_ref = len(X_ref)
    if n_sub < 2 or n_ref < 2:
        return float("nan")

    rng = np.random.default_rng(seed)

    # balance and cap
    m = min(n_sub, n_ref, max_rows_per_side)
    if m < 200:
        return float("nan")

    sub_idx = rng.choice(n_sub, size=m, replace=False)
    ref_idx = rng.choice(n_ref, size=m, replace=False)

    X = pd.concat([X_sub.iloc[sub_idx], X_ref.iloc[ref_idx]], axis=0, ignore_index=True)
    y = np.concatenate([np.ones(m, dtype=int), np.zeros(m, dtype=int)])

    # ensure columns exist
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
# Subset selection (stratified)
# ----------------------------

def make_stratify_bins_for_target(y: pd.Series, problem_type: str, n_bins: int = 10) -> Optional[np.ndarray]:
    """
    - classification: stratify by y directly
    - regression: stratify by quantile bins
    """
    if y.isna().any():
        return None

    if problem_type.lower() in ("binary", "multiclass", "classification"):
        return y.values

    # regression
    # if low unique -> treat as classification-like
    nunique = y.nunique(dropna=True)
    if nunique <= min(20, max(2, n_bins)):
        return y.values

    try:
        bins = pd.qcut(y, q=min(n_bins, nunique), duplicates="drop")
        return bins.astype(str).values
    except Exception:
        return None


def stratified_sample_indices(
    total_n: int,
    sample_n: int,
    stratify_labels: Optional[np.ndarray],
    seed: int,
) -> np.ndarray:
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

    chosen = np.array(chosen, dtype=int)
    if len(chosen) < sample_n:
        remaining = np.setdiff1d(idx, chosen, assume_unique=False)
        extra = rng.choice(remaining, size=sample_n - len(chosen), replace=False)
        chosen = np.concatenate([chosen, extra])
    elif len(chosen) > sample_n:
        chosen = rng.choice(chosen, size=sample_n, replace=False)

    return chosen


# ----------------------------
# Orchestration / Reporting
# ----------------------------

@dataclass
class Thresholds:
    adv_auc_max: float = 0.55
    psi_max: float = 0.10
    psi_bad_frac_max: float = 0.05
    target_psi_max: float = 0.05


@dataclass
class FractionResult:
    fraction: float
    n_subset: int
    n_rest: int
    pass_fail: str
    adv_auc: Optional[float]
    max_psi: Optional[float]
    bad_psi_frac: Optional[float]
    target_psi: Optional[float]
    top_drifting_features: List[Tuple[str, float]]
    notes: List[str]


def compute_psi_bundle(
    df_sub: pd.DataFrame,
    df_rest: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    bins: int = 10,
) -> Dict[str, float]:
    psi_map: Dict[str, float] = {}

    for c in numeric_cols:
        if c in df_sub.columns and c in df_rest.columns:
            psi_map[c] = psi_numeric(df_rest[c], df_sub[c], bins=bins)

    for c in categorical_cols:
        if c in df_sub.columns and c in df_rest.columns:
            psi_map[c] = psi_categorical(df_rest[c], df_sub[c])

    return psi_map


def evaluate_single_fraction(
    frac: float,
    df: pd.DataFrame,
    n_total: int,
    target_col: str,
    strat_labels: Optional[np.ndarray],
    seed: int,
    thresholds: Thresholds,
    numeric_cols: List[str],
    categorical_cols: List[str],
    min_rest_rows: int,
    psi_bins: int,
    adv_cv_splits: int,
    adv_max_rows_per_side: int,
) -> FractionResult:
    n_sub = int(round(n_total * frac))
    n_sub = max(1, min(n_sub, n_total))
    n_rest = n_total - n_sub

    notes: List[str] = []

    if frac == 1.0:
        return FractionResult(
            fraction=frac,
            n_subset=n_sub,
            n_rest=n_rest,
            pass_fail="PASS",
            adv_auc=None,
            max_psi=0.0,
            bad_psi_frac=0.0,
            target_psi=0.0,
            top_drifting_features=[],
            notes=["fraction=1.0: subset == full; metrics skipped"],
        )

    if n_rest < min_rest_rows:
        notes.append(f"small_rest(n_rest={n_rest}) may be noisy")

    sub_idx = stratified_sample_indices(
        total_n=n_total,
        sample_n=n_sub,
        stratify_labels=strat_labels,
        seed=seed + int(frac * 1000),
    )
    mask = np.zeros(n_total, dtype=bool)
    mask[sub_idx] = True

    df_sub = df.loc[mask].reset_index(drop=True)
    df_rest = df.loc[~mask].reset_index(drop=True)

    # PSI features (subset vs rest)
    psi_map = compute_psi_bundle(df_sub, df_rest, numeric_cols, categorical_cols, bins=psi_bins)
    psi_vals = np.array(list(psi_map.values()), dtype=float) if psi_map else np.array([], dtype=float)

    max_psi = float(np.nanmax(psi_vals)) if psi_vals.size else float("nan")
    bad_frac = float(np.mean(psi_vals > thresholds.psi_max)) if psi_vals.size else float("nan")

    # target PSI
    if pd.api.types.is_numeric_dtype(df[target_col]):
        target_psi = psi_numeric(df_rest[target_col], df_sub[target_col], bins=psi_bins)
    else:
        target_psi = psi_categorical(df_rest[target_col], df_sub[target_col])

    # Adversarial AUC
    X_sub = df_sub[numeric_cols + categorical_cols].copy()
    X_rest = df_rest[numeric_cols + categorical_cols].copy()
    adv_auc = adversarial_auc(
        X_sub, X_rest,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        seed=seed + int(frac * 1000),
        n_splits=adv_cv_splits,
        max_rows_per_side=adv_max_rows_per_side,
    )

    # top drifting features
    top = sorted(psi_map.items(), key=lambda kv: kv[1], reverse=True)[:10]

    # PASS/FAIL
    checks = []
    if not math.isnan(adv_auc):
        checks.append(adv_auc <= thresholds.adv_auc_max)
    else:
        notes.append("adv_auc=nan (insufficient rows?)")
        checks.append(False)

    if not math.isnan(max_psi):
        checks.append(max_psi <= thresholds.psi_max)
    else:
        notes.append("max_psi=nan")
        checks.append(False)

    if not math.isnan(bad_frac):
        checks.append(bad_frac <= thresholds.psi_bad_frac_max)
    else:
        notes.append("bad_psi_frac=nan")
        checks.append(False)

    checks.append(target_psi <= thresholds.target_psi_max)

    pass_fail = "PASS" if all(checks) else "FAIL"

    return FractionResult(
        fraction=frac,
        n_subset=n_sub,
        n_rest=n_rest,
        pass_fail=pass_fail,
        adv_auc=float(adv_auc) if not math.isnan(adv_auc) else None,
        max_psi=float(max_psi) if not math.isnan(max_psi) else None,
        bad_psi_frac=float(bad_frac) if not math.isnan(bad_frac) else None,
        target_psi=float(target_psi),
        top_drifting_features=[(k, float(v)) for k, v in top],
        notes=notes,
    )


def evaluate_fractions(
    df: pd.DataFrame,
    target_col: str,
    ignored_cols: List[str],
    problem_type: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    thresholds: Thresholds,
    seed: int = 42,
    step: float = 0.10,
    min_rest_rows: int = 5000,
    psi_bins: int = 10,
    adv_cv_splits: int = 5,
    adv_max_rows_per_side: int = 200_000,
    n_threads: int = 4,
) -> List[FractionResult]:
    n_total = len(df)
    if n_total < 10:
        raise ValueError("Dataset too small for this procedure.")

    # prepare feature lists: remove ignored + target
    ignored = set(ignored_cols + [target_col])
    used_num = [c for c in numeric_cols if c not in ignored]
    used_cat = [c for c in categorical_cols if c not in ignored]

    # fallback typing if EDA didn't include some columns
    for c in df.columns:
        if c in ignored:
            continue
        if c not in used_num and c not in used_cat:
            if pd.api.types.is_numeric_dtype(df[c]):
                used_num.append(c)
            else:
                used_cat.append(c)

    y = df[target_col]
    strat_labels = make_stratify_bins_for_target(y, problem_type=problem_type, n_bins=10)

    results: List[FractionResult] = []

    if math.isclose(step, 0.10):
        # extended default set
        fractions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 
                     0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    else:
        # strict arithmetic
        fractions = [round(x, 4) for x in np.arange(1.0, 0.0, -step)]
        if fractions[-1] != round(step, 4):
            fractions.append(round(step, 4))
    
    # filter out duplicates and ensure valid range
    fractions = sorted(list(set(f for f in fractions if 0.0 < f <= 1.0)), reverse=True)

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task("subset-compat", total=len(fractions))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_frac = {
                executor.submit(
                    evaluate_single_fraction,
                    frac,
                    df,
                    n_total,
                    target_col,
                    strat_labels,
                    seed,
                    thresholds,
                    used_num,
                    used_cat,
                    min_rest_rows,
                    psi_bins,
                    adv_cv_splits,
                    adv_max_rows_per_side
                ): frac
                for frac in fractions
            }

            for future in concurrent.futures.as_completed(future_to_frac):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    print(f"Fraction generated an exception: {exc}")
                finally:
                    progress.advance(task)

    # Sort results because async execution returns them in random order
    results.sort(key=lambda x: x.fraction, reverse=True)
    return results


def print_results_table(results: List[FractionResult]) -> None:
    console = Console()
    table = Table(title="Subset Compatibility Analysis", box=box.ROUNDED)

    table.add_column("Fraction", justify="right", style="cyan", no_wrap=True)
    table.add_column("N Subset", justify="right", style="magenta")
    table.add_column("N Rest", justify="right", style="magenta")
    table.add_column("Status", justify="center")
    table.add_column("Adv AUC", justify="right")
    table.add_column("Max PSI", justify="right")
    table.add_column("Bad PSI %", justify="right")
    table.add_column("Target PSI", justify="right")

    for r in results:
        status_style = "green bold" if r.pass_fail == "PASS" else "red bold"
        
        adv_auc_str = f"{r.adv_auc:.8f}" if r.adv_auc is not None else "-"
        max_psi_str = f"{r.max_psi:.8f}" if r.max_psi is not None else "-"
        bad_psi_frac_str = f"{r.bad_psi_frac:.8f}" if r.bad_psi_frac is not None else "-"
        target_psi_str = f"{r.target_psi:.8f}" if r.target_psi is not None else "-"

        table.add_row(
            f"{r.fraction:.4f}",
            str(r.n_subset),
            str(r.n_rest),
            f"[{status_style}]{r.pass_fail}[/{status_style}]",
            adv_auc_str,
            max_psi_str,
            bad_psi_frac_str,
            target_psi_str,
        )

    console.print(table)


# ----------------------------
# Stability of model selection (ranking) on subset
# ----------------------------


@dataclass
class StabilityThresholds:
    spearman_p10_min: float = 0.90
    hitk_mean_min: float = 0.80
    regret_p90_max_rel: float = 0.01
    regret_p90_max_abs: Optional[float] = None


@dataclass
class StabilityFractionResult:
    fraction: float
    model: str
    n_subset: int
    n_train_pool: int
    n_holdout: int
    n_configs: int
    repeats: int
    spearman_p10: float
    kendall_p10: float
    hitk_mean: float
    regret_p90: float
    pass_fail: str
    notes: List[str]


def _normalize_metric_name(metric: str) -> str:
    return metric.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class MetricSpec:
    name: str
    greater_is_better: bool
    needs_proba: bool


def get_metric_spec(metric_name: str, problem_type: str) -> MetricSpec:
    m = _normalize_metric_name(metric_name)
    p = _normalize_metric_name(problem_type)

    # Regression metrics
    if m in ("root_mean_squared_error", "rmse"):
        return MetricSpec(name="rmse", greater_is_better=False, needs_proba=False)
    if m in ("mean_squared_error", "mse"):
        return MetricSpec(name="mse", greater_is_better=False, needs_proba=False)
    if m in ("mean_absolute_error", "mae"):
        return MetricSpec(name="mae", greater_is_better=False, needs_proba=False)
    if m in ("r2", "r2_score"):
        return MetricSpec(name="r2", greater_is_better=True, needs_proba=False)

    # Classification metrics
    if m in ("roc_auc", "auc"):
        return MetricSpec(name="roc_auc", greater_is_better=True, needs_proba=True)
    if m in ("log_loss", "cross_entropy"):
        return MetricSpec(name="log_loss", greater_is_better=False, needs_proba=True)
    if m in ("accuracy", "acc"):
        return MetricSpec(name="accuracy", greater_is_better=True, needs_proba=False)
    if m in ("f1", "f1_score"):
        return MetricSpec(name="f1", greater_is_better=True, needs_proba=False)

    # Fallbacks by problem type
    if p in ("binary", "multiclass", "classification"):
        return MetricSpec(name="roc_auc", greater_is_better=True, needs_proba=True)
    return MetricSpec(name="rmse", greater_is_better=False, needs_proba=False)


def score_predictions(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray],
    y_proba: Optional[np.ndarray],
    spec: MetricSpec,
) -> float:
    if spec.name == "rmse":
        if y_pred is None:
            raise ValueError("rmse requires y_pred")
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))
    if spec.name == "mse":
        if y_pred is None:
            raise ValueError("mse requires y_pred")
        return float(mean_squared_error(y_true, y_pred))
    if spec.name == "mae":
        if y_pred is None:
            raise ValueError("mae requires y_pred")
        return float(mean_absolute_error(y_true, y_pred))
    if spec.name == "r2":
        if y_pred is None:
            raise ValueError("r2 requires y_pred")
        return float(r2_score(y_true, y_pred))
    if spec.name == "roc_auc":
        if y_proba is None:
            raise ValueError("roc_auc requires y_proba")
        # binary: y_proba is P(class=1)
        return float(roc_auc_score(y_true, y_proba))
    if spec.name == "log_loss":
        if y_proba is None:
            raise ValueError("log_loss requires y_proba")
        # allow both (n,) and (n,2)
        if y_proba.ndim == 1:
            p = np.vstack([1 - y_proba, y_proba]).T
        else:
            p = y_proba
        return float(log_loss(y_true, p))
    if spec.name == "accuracy":
        if y_pred is None:
            raise ValueError("accuracy requires y_pred")
        return float(accuracy_score(y_true, y_pred))
    if spec.name == "f1":
        if y_pred is None:
            raise ValueError("f1 requires y_pred")
        return float(f1_score(y_true, y_pred))

    raise ValueError(f"Unsupported metric spec: {spec}")


def build_training_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_tf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric_cols),
            ("cat", categorical_tf, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def _downsample_indices(idx: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if max_n <= 0 or len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    return rng.choice(idx, size=max_n, replace=False)


def prepare_stability_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    problem_type: str,
    holdout_frac: float,
    seed: int,
    max_train_rows: int,
    max_holdout_rows: int,
    numeric_cols: List[str],
    categorical_cols: List[str],
):
    """Split into train_pool / holdout, optionally downsample, fit preprocessor on train_pool."""

    y = df[target_col]
    strat = make_stratify_bins_for_target(y, problem_type=problem_type, n_bins=10)
    if strat is not None:
        X_train, X_hold, y_train, y_hold = train_test_split(
            df[feature_cols],
            y,
            test_size=holdout_frac,
            random_state=seed,
            shuffle=True,
            stratify=strat,
        )
    else:
        X_train, X_hold, y_train, y_hold = train_test_split(
            df[feature_cols],
            y,
            test_size=holdout_frac,
            random_state=seed,
            shuffle=True,
        )

    # downsample
    train_idx = np.arange(len(X_train))
    hold_idx = np.arange(len(X_hold))
    train_idx = _downsample_indices(train_idx, max_train_rows, seed=seed + 11)
    hold_idx = _downsample_indices(hold_idx, max_holdout_rows, seed=seed + 17)

    X_train = X_train.iloc[train_idx].reset_index(drop=True)
    y_train = y_train.iloc[train_idx].reset_index(drop=True)
    X_hold = X_hold.iloc[hold_idx].reset_index(drop=True)
    y_hold = y_hold.iloc[hold_idx].reset_index(drop=True)

    pre = build_training_preprocessor(numeric_cols, categorical_cols)
    X_train_mat = pre.fit_transform(X_train)
    X_hold_mat = pre.transform(X_hold)

    return X_train_mat, y_train.to_numpy(), X_hold_mat, y_hold.to_numpy(), pre


def sample_model_configs(model: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Random hyperparam grid for boosting models (fast, extendable)."""
    rng = np.random.default_rng(seed)

    def logu(lo: float, hi: float) -> float:
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    configs: List[Dict[str, Any]] = []
    m = model.lower()

    for _ in range(n):
        if m == "lgbm":
            cfg = {
                "n_estimators": int(rng.integers(200, 700)),
                "learning_rate": logu(0.01, 0.2),
                "num_leaves": int(rng.integers(16, 256)),
                "max_depth": int(rng.integers(-1, 12)),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.6, 1.0)),
                "min_child_samples": int(rng.integers(10, 200)),
                "reg_alpha": logu(1e-3, 10.0),
                "reg_lambda": logu(1e-3, 10.0),
            }
        elif m == "xgb":
            cfg = {
                "n_estimators": int(rng.integers(200, 900)),
                "learning_rate": logu(0.01, 0.2),
                "max_depth": int(rng.integers(2, 10)),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "colsample_bytree": float(rng.uniform(0.6, 1.0)),
                "min_child_weight": logu(0.5, 10.0),
                "reg_alpha": logu(1e-3, 10.0),
                "reg_lambda": logu(1e-3, 10.0),
            }
        elif m == "cat":
            cfg = {
                "iterations": int(rng.integers(300, 1200)),
                "learning_rate": logu(0.01, 0.2),
                "depth": int(rng.integers(4, 10)),
                "l2_leaf_reg": logu(1.0, 30.0),
                "subsample": float(rng.uniform(0.6, 1.0)),
                "rsm": float(rng.uniform(0.6, 1.0)),
            }
        else:
            raise ValueError(f"Unknown model for stability: {model}")

        configs.append(cfg)

    return configs


def fit_score_boost(
    model: str,
    params: Dict[str, Any],
    X_train,
    y_train: np.ndarray,
    X_eval,
    y_eval: np.ndarray,
    spec: MetricSpec,
    problem_type: str,
    seed: int,
    n_threads: int,
) -> float:
    m = model.lower()
    ptype = _normalize_metric_name(problem_type)

    if m == "lgbm":
        if lgb is None:
            raise RuntimeError("lightgbm is not installed")
        if ptype in ("binary", "multiclass", "classification"):
            est = lgb.LGBMClassifier(
                **params,
                random_state=seed,
                n_jobs=n_threads,
                verbose=-1,
            )
        else:
            est = lgb.LGBMRegressor(
                **params,
                random_state=seed,
                n_jobs=n_threads,
                verbose=-1,
            )
        est.fit(X_train, y_train)

        if spec.needs_proba:
            proba = est.predict_proba(X_eval)
            y_proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba
            return score_predictions(y_eval, None, y_proba, spec)
        else:
            pred = est.predict(X_eval)
            return score_predictions(y_eval, pred, None, spec)

    if m == "xgb":
        if xgb is None:
            raise RuntimeError("xgboost is not installed")
        common = dict(
            **params,
            random_state=seed,
            n_jobs=n_threads,
            tree_method="hist",
            verbosity=0,
        )
        if ptype in ("binary", "multiclass", "classification"):
            est = xgb.XGBClassifier(**common)
        else:
            est = xgb.XGBRegressor(**common)
        est.fit(X_train, y_train)

        if spec.needs_proba:
            proba = est.predict_proba(X_eval)
            y_proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba
            return score_predictions(y_eval, None, y_proba, spec)
        else:
            pred = est.predict(X_eval)
            return score_predictions(y_eval, pred, None, spec)

    if m == "cat":
        if CatBoostRegressor is None:
            raise RuntimeError("catboost is not installed")
        # Use sparse matrix features as pure numeric; no categorical handling needed here.
        if ptype in ("binary", "multiclass", "classification"):
            est = CatBoostClassifier(
                **params,
                random_seed=seed,
                thread_count=n_threads,
                verbose=False,
                allow_writing_files=False,
            )
        else:
            est = CatBoostRegressor(
                **params,
                random_seed=seed,
                thread_count=n_threads,
                verbose=False,
                allow_writing_files=False,
            )
        est.fit(X_train, y_train)

        if spec.needs_proba:
            proba = est.predict_proba(X_eval)
            y_proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba
            return score_predictions(y_eval, None, y_proba, spec)
        else:
            pred = est.predict(X_eval)
            return score_predictions(y_eval, pred, None, spec)

    raise ValueError(f"Unknown model: {model}")


def run_stability_analysis(
    df: pd.DataFrame,
    fractions: List[float],
    feature_cols: List[str],
    target_col: str,
    problem_type: str,
    metric_name: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
    models: List[str],
    n_configs_per_model: int,
    repeats: int,
    topk: int,
    holdout_frac: float,
    max_train_rows: int,
    max_holdout_rows: int,
    thresholds: StabilityThresholds,
    seed: int,
    model_threads: int,
) -> List[StabilityFractionResult]:
    spec = get_metric_spec(metric_name, problem_type)

    X_pool, y_pool, X_hold, y_hold, _ = prepare_stability_data(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        problem_type=problem_type,
        holdout_frac=holdout_frac,
        seed=seed,
        max_train_rows=max_train_rows,
        max_holdout_rows=max_holdout_rows,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    n_train_pool = len(y_pool)
    n_holdout = len(y_hold)
    if n_train_pool < 5000:
        raise ValueError(f"Train pool too small for stability analysis: {n_train_pool}")

    # Make strat labels on y_pool for subset sampling
    strat_pool = make_stratify_bins_for_target(pd.Series(y_pool), problem_type=problem_type, n_bins=10)

    out: List[StabilityFractionResult] = []

    for model in models:
        cfgs = sample_model_configs(model, n=n_configs_per_model, seed=seed + hash(model) % 10_000)

        # Full (train_pool) scores for each config
        full_scores: List[float] = []
        for i, hp in enumerate(cfgs):
            s = fit_score_boost(
                model=model,
                params=hp,
                X_train=X_pool,
                y_train=y_pool,
                X_eval=X_hold,
                y_eval=y_hold,
                spec=spec,
                problem_type=problem_type,
                seed=seed + 1000 + i,
                n_threads=model_threads,
            )
            full_scores.append(s)
        F = np.array(full_scores, dtype=float)

        # Identify best config on full
        if spec.greater_is_better:
            best_full_idx = int(np.nanargmax(F))
            best_full_score = float(np.nanmax(F))
        else:
            best_full_idx = int(np.nanargmin(F))
            best_full_score = float(np.nanmin(F))

        for frac in fractions:
            notes: List[str] = []
            if frac <= 0.0 or frac > 1.0:
                continue

            n_sub = int(round(n_train_pool * frac))
            n_sub = max(50, min(n_sub, n_train_pool))

            # repeats
            rhos: List[float] = []
            taus: List[float] = []
            hits: List[float] = []
            regrets: List[float] = []

            for r in range(repeats):
                sub_idx = stratified_sample_indices(
                    total_n=n_train_pool,
                    sample_n=n_sub,
                    stratify_labels=strat_pool,
                    seed=seed + int(frac * 10_000) + 10 * r,
                )
                X_sub = X_pool[sub_idx]
                y_sub = y_pool[sub_idx]

                sub_scores: List[float] = []
                for i, hp in enumerate(cfgs):
                    s = fit_score_boost(
                        model=model,
                        params=hp,
                        X_train=X_sub,
                        y_train=y_sub,
                        X_eval=X_hold,
                        y_eval=y_hold,
                        spec=spec,
                        problem_type=problem_type,
                        seed=seed + 2000 + i + 10 * r,
                        n_threads=model_threads,
                    )
                    sub_scores.append(s)
                Y = np.array(sub_scores, dtype=float)

                # Spearman / Kendall
                rho = float(spearmanr(Y, F).correlation)
                tau = float(kendalltau(Y, F).correlation)
                if math.isnan(rho):
                    rho = 0.0
                if math.isnan(tau):
                    tau = 0.0
                rhos.append(rho)
                taus.append(tau)

                # Top-k hit: best-on-full is inside top-k on subset
                if spec.greater_is_better:
                    topk_idx = np.argsort(-Y)[:topk]
                    chosen_idx = int(np.argmax(Y))
                    regret = best_full_score - float(F[chosen_idx])
                else:
                    topk_idx = np.argsort(Y)[:topk]
                    chosen_idx = int(np.argmin(Y))
                    regret = float(F[chosen_idx]) - best_full_score

                hits.append(1.0 if best_full_idx in set(topk_idx.tolist()) else 0.0)
                regrets.append(float(max(0.0, regret)))

            spearman_p10 = float(np.quantile(rhos, 0.10))
            kendall_p10 = float(np.quantile(taus, 0.10))
            hitk_mean = float(np.mean(hits))
            regret_p90 = float(np.quantile(regrets, 0.90))

            # Regret threshold
            regret_thr = None
            if thresholds.regret_p90_max_abs is not None:
                regret_thr = thresholds.regret_p90_max_abs
            else:
                denom = abs(best_full_score) if abs(best_full_score) > 1e-12 else 1.0
                regret_thr = thresholds.regret_p90_max_rel * denom

            ok = (
                spearman_p10 >= thresholds.spearman_p10_min
                and hitk_mean >= thresholds.hitk_mean_min
                and regret_p90 <= regret_thr
            )
            pf = "PASS" if ok else "FAIL"

            if n_sub < 500:
                notes.append("very_small_subset: ranking metrics may be noisy")

            out.append(
                StabilityFractionResult(
                    fraction=float(frac),
                    model=model,
                    n_subset=n_sub,
                    n_train_pool=n_train_pool,
                    n_holdout=n_holdout,
                    n_configs=n_configs_per_model,
                    repeats=repeats,
                    spearman_p10=spearman_p10,
                    kendall_p10=kendall_p10,
                    hitk_mean=hitk_mean,
                    regret_p90=regret_p90,
                    pass_fail=pf,
                    notes=notes,
                )
            )

    out.sort(key=lambda r: (r.fraction, r.model), reverse=True)
    return out


def print_stability_table(results: List[StabilityFractionResult], metric_name: str) -> None:
    console = Console()
    table = Table(title=f"Model-Selection Stability (metric={metric_name})", box=box.ROUNDED)

    table.add_column("Fraction", justify="right", style="cyan", no_wrap=True)
    table.add_column("Model", justify="left", style="magenta")
    table.add_column("N Subset", justify="right")
    table.add_column("Spearman p10", justify="right")
    table.add_column("Kendall p10", justify="right")
    table.add_column("Hit@k mean", justify="right")
    table.add_column("Regret p90", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        status_style = "green bold" if r.pass_fail == "PASS" else "red bold"
        table.add_row(
            f"{r.fraction:.4f}",
            r.model,
            str(r.n_subset),
            f"{r.spearman_p10:.4f}",
            f"{r.kendall_p10:.4f}",
            f"{r.hitk_mean:.4f}",
            f"{r.regret_p90:.6f}",
            f"[{status_style}]{r.pass_fail}[/{status_style}]",
        )

    console.print(table)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-path", required=True, help="Path to train.csv or train.csv.gz")
    ap.add_argument("--eda-json", required=True, help="Path to EDA JSON (e.g., state.json)")
    ap.add_argument("--config-py", required=True, help="Path to config.py")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--step", type=float, default=0.10, help="Fraction step (default 0.10)")
    ap.add_argument("--min-rest-rows", type=int, default=5000)
    ap.add_argument("--psi-bins", type=int, default=10)
    ap.add_argument("--adv-cv-splits", type=int, default=5)
    ap.add_argument("--adv-max-rows-per-side", type=int, default=200_000)
    ap.add_argument("--threads", type=int, default=4, help="Number of parallel threads (default 4)")
    ap.add_argument("--out-json", default=None, help="Optional path to save detailed JSON report")
    # thresholds
    ap.add_argument("--thr-adv-auc", type=float, default=0.55)
    ap.add_argument("--thr-psi-max", type=float, default=0.10)
    ap.add_argument("--thr-psi-bad-frac", type=float, default=0.05)
    ap.add_argument("--thr-target-psi", type=float, default=0.05)

    # --- optional: stability of model selection (ranking) ---
    ap.add_argument("--enable-stability", action="store_true", help="Run model-selection stability checks")
    ap.add_argument(
        "--stability-models",
        default="lgbm,xgb",
        help="Comma-separated: lgbm,xgb,cat (default: lgbm,xgb)",
    )
    ap.add_argument("--stability-configs-per-model", type=int, default=20)
    ap.add_argument("--stability-repeats", type=int, default=5)
    ap.add_argument("--stability-topk", type=int, default=3)
    ap.add_argument("--stability-holdout-frac", type=float, default=0.2)
    ap.add_argument("--stability-max-train-rows", type=int, default=200_000)
    ap.add_argument("--stability-max-holdout-rows", type=int, default=50_000)
    ap.add_argument(
        "--stability-fractions",
        default=None,
        help="Optional comma-separated fractions (e.g. 1,0.5,0.2). Default: reuse fractions from compatibility run.",
    )
    ap.add_argument("--thr-spearman-p10", type=float, default=0.90)
    ap.add_argument("--thr-hitk-mean", type=float, default=0.80)
    ap.add_argument("--thr-regret-rel-p90", type=float, default=0.01)
    ap.add_argument("--thr-regret-abs-p90", type=float, default=None)
    ap.add_argument(
        "--stability-model-threads",
        type=int,
        default=None,
        help="Threads per boosting model (default: --threads)",
    )

    args = ap.parse_args()

    cfg = load_py_config(args.config_py)
    eda_state = load_eda_state(args.eda_json)
    train_vars = extract_train_profile_vars(eda_state)
    num_cols, cat_cols = infer_feature_types_from_eda(train_vars)

    target_col = getattr(cfg, "TARGET_COLUMN", None)
    if not target_col:
        raise RuntimeError("TARGET_COLUMN missing in config.py")

    ignored_cols = list(getattr(cfg, "IGNORED_COLUMNS", []))
    problem_type = getattr(cfg, "AUTOGLUON_PROBLEM_TYPE", "regression")
    metric_name = getattr(cfg, "AUTOGLUON_EVAL_METRIC", None) or getattr(cfg, "METRIC", None) or "rmse"

    # read train
    df = pd.read_csv(args.train_path)

    if target_col not in df.columns:
        raise RuntimeError(f"Target column '{target_col}' not found in train.")

    # basic informational hooks (extendable)
    time_like = detect_time_like_columns(list(df.columns))
    if time_like:
        print(f"[info] detected time-like columns (heuristic): {time_like}", file=sys.stderr)

    thr = Thresholds(
        adv_auc_max=args.thr_adv_auc,
        psi_max=args.thr_psi_max,
        psi_bad_frac_max=args.thr_psi_bad_frac,
        target_psi_max=args.thr_target_psi,
    )

    results = evaluate_fractions(
        df=df,
        target_col=target_col,
        ignored_cols=ignored_cols,
        problem_type=problem_type,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        thresholds=thr,
        seed=args.seed,
        step=args.step,
        min_rest_rows=args.min_rest_rows,
        psi_bins=args.psi_bins,
        adv_cv_splits=args.adv_cv_splits,
        adv_max_rows_per_side=args.adv_max_rows_per_side,
        n_threads=args.threads,
    )

    print_results_table(results)

    stability_results: List[StabilityFractionResult] = []
    if args.enable_stability:
        # features: remove ignored + target
        ignored = set(ignored_cols + [target_col])
        st_num = [c for c in num_cols if c in df.columns and c not in ignored]
        st_cat = [c for c in cat_cols if c in df.columns and c not in ignored]

        # fallback typing (if EDA missed any columns)
        for c in df.columns:
            if c in ignored:
                continue
            if c not in st_num and c not in st_cat:
                if pd.api.types.is_numeric_dtype(df[c]):
                    st_num.append(c)
                else:
                    st_cat.append(c)

        feature_cols = list(dict.fromkeys(st_num + st_cat))

        if args.stability_fractions:
            fractions = []
            for part in args.stability_fractions.split(","):
                part = part.strip()
                if not part:
                    continue
                fractions.append(float(part))
            fractions = sorted(list(set(fractions)), reverse=True)
        else:
            fractions = sorted(list(set(r.fraction for r in results)), reverse=True)

        st_thr = StabilityThresholds(
            spearman_p10_min=args.thr_spearman_p10,
            hitk_mean_min=args.thr_hitk_mean,
            regret_p90_max_rel=args.thr_regret_rel_p90,
            regret_p90_max_abs=args.thr_regret_abs_p90,
        )

        models = [m.strip().lower() for m in args.stability_models.split(",") if m.strip()]
        model_threads = int(args.stability_model_threads) if args.stability_model_threads is not None else int(args.threads)

        stability_results = run_stability_analysis(
            df=df,
            fractions=fractions,
            feature_cols=feature_cols,
            target_col=target_col,
            problem_type=problem_type,
            metric_name=metric_name,
            numeric_cols=st_num,
            categorical_cols=st_cat,
            models=models,
            n_configs_per_model=args.stability_configs_per_model,
            repeats=args.stability_repeats,
            topk=args.stability_topk,
            holdout_frac=args.stability_holdout_frac,
            max_train_rows=args.stability_max_train_rows,
            max_holdout_rows=args.stability_max_holdout_rows,
            thresholds=st_thr,
            seed=args.seed,
            model_threads=model_threads,
        )

        print_stability_table(stability_results, metric_name=metric_name)

    if args.out_json:
        payload: Dict[str, Any] = {
            "compatibility": {
                "thresholds": asdict(thr),
                "results": [asdict(r) for r in results],
                "psi_bins": args.psi_bins,
                "adv_cv_splits": args.adv_cv_splits,
                "adv_max_rows_per_side": args.adv_max_rows_per_side,
            },
            "meta": {
                "target": target_col,
                "ignored_cols": ignored_cols,
                "problem_type": problem_type,
                "metric": metric_name,
            },
        }
        if args.enable_stability:
            payload["stability"] = {
                "thresholds": {
                    "spearman_p10_min": args.thr_spearman_p10,
                    "hitk_mean_min": args.thr_hitk_mean,
                    "regret_p90_max_rel": args.thr_regret_rel_p90,
                    "regret_p90_max_abs": args.thr_regret_abs_p90,
                },
                "settings": {
                    "models": args.stability_models,
                    "configs_per_model": args.stability_configs_per_model,
                    "repeats": args.stability_repeats,
                    "topk": args.stability_topk,
                    "holdout_frac": args.stability_holdout_frac,
                    "max_train_rows": args.stability_max_train_rows,
                    "max_holdout_rows": args.stability_max_holdout_rows,
                },
                "results": [asdict(r) for r in stability_results],
            }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

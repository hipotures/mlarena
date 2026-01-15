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
from dataclasses import dataclass, asdict
from importlib.util import spec_from_file_location, module_from_spec
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn


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
    n_total: int,
    n_sample: int,
    stratify_labels: Optional[np.ndarray],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)

    if stratify_labels is None:
        return rng.choice(idx, size=n_sample, replace=False)

    # sample proportionally per stratum
    labels = np.asarray(stratify_labels)
    if len(labels) != n_total:
        raise ValueError(
            f"stratify_labels length ({len(labels)}) does not match n_total ({n_total})"
        )
    uniq, counts = np.unique(labels, return_counts=True)
    # desired counts per stratum (rounded, with adjustment)
    desired = np.floor(counts / counts.sum() * n_sample).astype(int)

    # fix rounding so total == n
    diff = n_sample - desired.sum()
    if diff > 0:
        # distribute remaining to largest strata
        order = np.argsort(-counts)
        for k in range(diff):
            desired[order[k % len(order)]] += 1
    elif diff < 0:
        order = np.argsort(counts)  # remove from smallest
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
    # if due to edge cases we have less/more, fix by random fill/trim
    if len(chosen) < n_sample:
        remaining = np.setdiff1d(idx, chosen, assume_unique=False)
        extra = rng.choice(remaining, size=n_sample - len(chosen), replace=False)
        chosen = np.concatenate([chosen, extra])
    elif len(chosen) > n_sample:
        chosen = rng.choice(chosen, size=n_sample, replace=False)

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

    fractions = [round(x, 2) for x in np.arange(1.0, 0.0, -step)]
    if fractions[-1] != round(step, 2):
        fractions.append(round(step, 2))

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with progress:
        task = progress.add_task("subset-compat", total=len(fractions))
        for frac in fractions:
            n_sub = int(round(n_total * frac))
            n_sub = max(1, min(n_sub, n_total))
            n_rest = n_total - n_sub

            notes: List[str] = []

            if frac == 1.0:
                results.append(FractionResult(
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
                ))
                progress.advance(task)
                continue

            if n_rest < min_rest_rows:
                # still compute (might be noisy), but flag
                notes.append(f"small_rest(n_rest={n_rest}) may be noisy")

            sub_idx = stratified_sample_indices(
                n_total,
                n_sub,
                stratify_labels=strat_labels,
                seed=seed + int(frac * 1000),
            )
            mask = np.zeros(n_total, dtype=bool)
            mask[sub_idx] = True

            df_sub = df.loc[mask].reset_index(drop=True)
            df_rest = df.loc[~mask].reset_index(drop=True)

            # PSI features (subset vs rest)
            psi_map = compute_psi_bundle(df_sub, df_rest, used_num, used_cat, bins=psi_bins)
            psi_vals = np.array(list(psi_map.values()), dtype=float) if psi_map else np.array([], dtype=float)

            max_psi = float(np.nanmax(psi_vals)) if psi_vals.size else float("nan")
            bad_frac = float(np.mean(psi_vals > thresholds.psi_max)) if psi_vals.size else float("nan")

            # target PSI
            if pd.api.types.is_numeric_dtype(df[target_col]):
                target_psi = psi_numeric(df_rest[target_col], df_sub[target_col], bins=psi_bins)
            else:
                target_psi = psi_categorical(df_rest[target_col], df_sub[target_col])

            # Adversarial AUC
            X_sub = df_sub[used_num + used_cat].copy()
            X_rest = df_rest[used_num + used_cat].copy()
            adv_auc = adversarial_auc(
                X_sub, X_rest,
                numeric_cols=used_num,
                categorical_cols=used_cat,
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

            results.append(FractionResult(
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
            ))
            progress.advance(task)

    return results


def print_results_table(results: List[FractionResult]) -> None:
    cols = ["fraction", "n_subset", "n_rest", "PASS/FAIL", "adv_auc", "max_psi", "bad_psi_frac", "target_psi"]
    print("\t".join(cols))
    for r in results:
        print(
            f"{r.fraction:.2f}\t{r.n_subset}\t{r.n_rest}\t{r.pass_fail}\t"
            f"{'' if r.adv_auc is None else f'{r.adv_auc:.4f}'}\t"
            f"{'' if r.max_psi is None else f'{r.max_psi:.4f}'}\t"
            f"{'' if r.bad_psi_frac is None else f'{r.bad_psi_frac:.4f}'}\t"
            f"{'' if r.target_psi is None else f'{r.target_psi:.4f}'}"
        )


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
    ap.add_argument("--out-json", default=None, help="Optional path to save detailed JSON report")
    # thresholds
    ap.add_argument("--thr-adv-auc", type=float, default=0.55)
    ap.add_argument("--thr-psi-max", type=float, default=0.10)
    ap.add_argument("--thr-psi-bad-frac", type=float, default=0.05)
    ap.add_argument("--thr-target-psi", type=float, default=0.05)

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
    )

    print_results_table(results)

    if args.out_json:
        payload = {
            "thresholds": asdict(thr),
            "results": [asdict(r) for r in results],
            "target": target_col,
            "ignored_cols": ignored_cols,
            "problem_type": problem_type,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

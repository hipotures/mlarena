#!/usr/bin/env python
import sqlite3
import pandas as pd
import json
import argparse
import time
import shutil
import warnings
from pathlib import Path
import yaml
from autogluon.tabular import TabularPredictor

# Console output (rich)
from rich.console import Console
from rich.table import Table
from rich import box

# Optional (used for ablation / significance checks)
import numpy as np

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import train_test_split
    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False

console = Console()
warnings.filterwarnings("ignore")


def info(message: str) -> None:
    console.print(message)


def warn(message: str) -> None:
    console.print(f"[yellow]Warning:[/yellow] {message}")


def err(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")


def _format_cell(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_kv_table(title: str, rows: list[tuple[str, str]]) -> None:
    table = Table(title=title, show_header=False, box=box.ASCII)
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in rows:
        table.add_row(str(key), str(value))
    console.print(table)


def _print_df_table(title: str, df: pd.DataFrame) -> None:
    table = Table(title=title, box=box.ASCII)
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.iterrows():
        table.add_row(*[_format_cell(v) for v in row.tolist()])
    console.print(table)


def _extract_final_value(details_json):
    """Extract numeric final_value from details_json.

    Expected structure: {"final_value": <float>}
    Returns None if not present or not parseable.
    """
    if details_json is None or details_json == "" or pd.isna(details_json):
        return None
    try:
        data = json.loads(details_json)
    except Exception:
        return None

    value = None
    if isinstance(data, dict):
        value = data.get("final_value")

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def _load_default_mcts_settings(project_slug=None):
    """Load default study_name and oracle eps from mla_super_chain.yaml if available."""
    if project_slug:
        # Use project-specific config
        config_path = Path("projects/kaggle") / project_slug / "conf" / "preprocess" / "mla_super_chain.yaml"
    else:
        # Fallback to global config
        config_path = Path(__file__).resolve().parents[1] / "conf" / "preprocess" / "mla_super_chain.yaml"

    if not config_path.exists():
        warn(f"Default config not found at {config_path}")
        return {}
    try:
        data = yaml.safe_load(config_path.read_text()) or {}
        mcts = data.get("mcts") or {}
        oracle = mcts.get("oracle") or {}
        return {
            "study_name": mcts.get("study_name"),
            "oracle_eps": oracle.get("eps"),
        }
    except Exception as e:
        warn(f"Failed to read default settings from {config_path}: {e}")
        return {}


def parse_action(action_json_str, prefix=""):
    if not action_json_str or pd.isna(action_json_str):
        return {}
    try:
        data = json.loads(action_json_str)

        flat = {}
        group = data.get("group_name", "unknown")
        variant = data.get("variant", "unknown")

        flat[f"{prefix}action_group"] = group
        flat[f"{prefix}action_variant"] = variant

        config = data.get("config", {})
        for k, v in config.items():
            key = f"{prefix}{group}_{k}"
            if isinstance(v, (list, dict)):
                flat[key] = json.dumps(v, sort_keys=True)
            else:
                flat[key] = v

        return flat
    except Exception:
        return {}


def extract_data(db_path, study_id=None, study_name=None):
    info(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path, timeout=30.0)

    # 1. Get Action History (Map trial_id -> action_json)
    history_query = "SELECT child_trial_id, action_json FROM mcts_edges"
    history_df = pd.read_sql_query(history_query, conn)
    node_action_map = dict(zip(history_df["child_trial_id"], history_df["action_json"]))

    # 2. Extract transitions with best evaluation per trial
    where_clause = ""
    params = []
    if study_id is not None:
        where_clause = "WHERE parent_node.study_id = ?"
        params.append(int(study_id))
    elif study_name:
        where_clause = "WHERE s.study_name = ?"
        params.append(study_name)

    query = f"""
    WITH eval_ranked AS (
        SELECT
            trial_id,
            fidelity,
            status,
            value,
            metric_name,
            duration_sec,
            details_json,
            ROW_NUMBER() OVER (
                PARTITION BY trial_id
                ORDER BY
                    (status = 'COMPLETE') DESC,
                    (value IS NOT NULL) DESC,
                    CASE
                        WHEN fidelity GLOB 'F[0-9]*' THEN CAST(substr(fidelity, 2) AS INTEGER)
                        WHEN fidelity GLOB '[0-9]*' THEN CAST(fidelity AS INTEGER)
                        ELSE -1
                    END DESC,
                    fidelity DESC
            ) AS rn
        FROM mcts_evaluations
    )
    SELECT
        parent.trial_id as parent_id,
        child.trial_id as child_id,
        edge.action_json as curr_action_json,
        eval_parent.value as parent_score_raw,
        eval_child.value as child_score_raw,
        eval_parent.details_json as parent_details_json,
        eval_child.details_json as child_details_json,
        child_node.depth,
        eval_parent.duration_sec as prev_duration,
        sd.direction as direction
    FROM mcts_edges edge
    JOIN mcts_nodes parent_node ON edge.parent_trial_id = parent_node.trial_id
    JOIN mcts_nodes child_node ON edge.child_trial_id = child_node.trial_id
    JOIN trials parent ON parent_node.trial_id = parent.trial_id
    JOIN trials child ON child_node.trial_id = child.trial_id
    LEFT JOIN studies s ON parent_node.study_id = s.study_id
    LEFT JOIN study_directions sd ON parent_node.study_id = sd.study_id AND sd.objective = 0
    LEFT JOIN eval_ranked eval_parent ON parent.trial_id = eval_parent.trial_id AND eval_parent.rn = 1
    LEFT JOIN eval_ranked eval_child ON child.trial_id = eval_child.trial_id AND eval_child.rn = 1
    {where_clause}
    """

    info("Executing main query...")
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    raw_n = len(df)
    info(f"Raw transitions found: {raw_n}")

    # Deduplicate
    df = df.drop_duplicates(subset=["parent_id", "child_id"])

    # Prefer final_value from details_json, fallback to raw value
    df["parent_final"] = df["parent_details_json"].apply(_extract_final_value)
    df["child_final"] = df["child_details_json"].apply(_extract_final_value)
    parent_final_cov = float(df["parent_final"].notna().mean()) if len(df) else 0.0
    child_final_cov = float(df["child_final"].notna().mean()) if len(df) else 0.0

    df["parent_score"] = df["parent_final"].where(df["parent_final"].notna(), df["parent_score_raw"])
    df["child_score"] = df["child_final"].where(df["child_final"].notna(), df["child_score_raw"])

    df = df.dropna(subset=["child_score", "parent_score"])
    valid_n = len(df)
    info(f"Valid transitions (with scores): {valid_n}")

    if df.empty:
        return pd.DataFrame()

    # Calculate signed delta (direction-aware)
    # Convention in this DB: direction == 1 -> minimize, direction == 2 -> maximize
    df["direction"] = df["direction"].fillna(2)
    df["delta_score"] = df["child_score"] - df["parent_score"]
    df.loc[df["direction"] == 1, "delta_score"] *= -1

    # Fill defaults
    if "prev_duration" in df.columns:
        df["prev_duration"] = df["prev_duration"].fillna(0.0)

    # Add previous action json context
    df["prev_action_json"] = df["parent_id"].map(node_action_map)

    # Parse action JSONs
    info("Parsing action JSONs...")
    curr_actions = df["curr_action_json"].apply(lambda x: parse_action(x, prefix=""))
    curr_actions_df = pd.DataFrame(curr_actions.tolist())

    prev_actions = df["prev_action_json"].apply(lambda x: parse_action(x, prefix="prev_"))
    prev_actions_df = pd.DataFrame(prev_actions.tolist())

    # Combine
    meta_df = pd.concat(
        [
            df[["parent_score", "depth", "delta_score", "prev_duration"]],
            curr_actions_df,
            prev_actions_df,
        ],
        axis=1,
    )

    meta_df.attrs["diagnostics"] = {
        "raw_transitions": raw_n,
        "valid_transitions": valid_n,
        "parent_final_coverage": parent_final_cov,
        "child_final_coverage": child_final_cov,
        "direction_counts": df["direction"].value_counts(dropna=False).to_dict(),
        "study_filter": {"study_id": study_id, "study_name": study_name},
    }

    return meta_df


def clean_output_dir(path):
    p = Path(path)
    if not p.exists():
        return

    # Safety check
    safe_names = {"oracle", "meta_model", "experiments", "model", "model-new", "model-old"}
    if len(p.parts) < 2 or p.name not in safe_names:
        # Allow generic if it ends with 'oracle'
        if not str(p).endswith(("oracle", "model", "model-new", "model-old")):
            warn(f"Safety warning: refusing to clean generic path {p}")
            return

    info(f"Cleaning output directory: {p}")
    for item in p.iterdir():
        if item.name.endswith(".csv"):
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            warn(f"Failed to delete {item}: {e}")


def _prepare_training_frame(df: pd.DataFrame, *, eps: float):
    """Prepare a training dataframe with label included.

    Returns:
        train_df: dataframe containing features + label column `is_improvement`
        feature_cols: list of feature column names (in train_df)
    """
    work = df.copy()
    work["is_improvement"] = (work["delta_score"] > eps).astype(int)

    # Drop leakage/unused columns from FEATURES (do NOT drop the label itself)
    drop_feature_cols = [
        # raw JSON blobs / identifiers (high-cardinality)
        "curr_action_json",
        "prev_action_json",
        "parent_id",
        "child_id",
        # target construction / post-decision values
        "delta_score",
        "child_score",
    ]
    work = work.drop(columns=[c for c in drop_feature_cols if c in work.columns], errors="ignore")

    feature_cols = [c for c in work.columns if c != "is_improvement"]
    return work, feature_cols


def train_model(df, output_dir, num_gpus=0, eps=0.0):
    # Prepare target. Keep it in the training frame; AutoGluon expects the label column to be present.
    # Also attach it to the original df so downstream reports (e.g., pruning recall loss) work.
    df["is_improvement"] = (df["delta_score"] > eps).astype(int)
    train_df, feature_cols = _prepare_training_frame(df, eps=eps)

    info(
        f"Training classification on {len(train_df)} rows. Target: is_improvement (delta_score > eps, eps={eps})"
    )

    clean_output_dir(output_dir)

    predictor = (
        TabularPredictor(
            label="is_improvement",
            path=str(output_dir),
            eval_metric="average_precision",
            problem_type="binary",
            sample_weight="balance_weight",
        )
        .fit(
            train_df,
            time_limit=600,  # 10 minutes
            presets="best_quality",
            num_gpus=num_gpus,
            calibrate_decision_threshold=True,
            hyperparameters={
                "GBM": {},
                "CAT": {},
                "XGB": {},
            },
        )
    )

    leaderboard = predictor.leaderboard(display=False)
    _print_df_table("Leaderboard", leaderboard)
    leaderboard_path = Path(output_dir) / "leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    info(f"Leaderboard saved: {leaderboard_path}")
    # Return predictor and the exact feature set used (for consistent inference / reports)
    return predictor, feature_cols


def _prepare_staging_dir(target_dir: Path) -> Path:
    staging_dir = target_dir.with_name(f"{target_dir.name}-new")
    if staging_dir.exists():
        info(f"Removing stale staging dir: {staging_dir}")
        shutil.rmtree(staging_dir)
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    return staging_dir


def _atomic_swap_dirs(target_dir: Path, staging_dir: Path) -> None:
    old_dir = target_dir.with_name(f"{target_dir.name}-old")
    if old_dir.exists():
        info(f"Removing stale old dir: {old_dir}")
        shutil.rmtree(old_dir)

    if target_dir.exists():
        info(f"Renaming {target_dir} -> {old_dir}")
        target_dir.rename(old_dir)

    info(f"Renaming {staging_dir} -> {target_dir}")
    staging_dir.rename(target_dir)

    if old_dir.exists():
        info(f"Removing old model dir: {old_dir}")
        shutil.rmtree(old_dir)


def generate_pruning_report(predictor, df, feature_cols, threshold=0.2):
    # Predict using the same feature columns as training (avoids surprises with extra columns)
    X = df.copy()
    if "is_improvement" not in X.columns:
        # Fallback, but normally set in train_model
        X["is_improvement"] = 0
    X = X.reindex(columns=feature_cols, fill_value=pd.NA)

    try:
        probs = predictor.predict_proba(X)
        pos_col = 1 if 1 in probs.columns else probs.columns[-1]
        p_vals = probs[pos_col]

        # Stats
        n_total = len(p_vals)
        n_pruned = (p_vals < threshold).sum()
        pct_pruned = (n_pruned / n_total) * 100 if n_total else 0.0

        rows = [
            ("Total actions", n_total),
            ("Pruned", f"{n_pruned} ({pct_pruned:.2f}%)"),
            ("Kept", f"{n_total - n_pruned} ({100 - pct_pruned:.2f}%)"),
        ]

        # Recall check
        if "is_improvement" in df.columns:
            actual_pos = int(df["is_improvement"].sum())
            pruned_mask = p_vals < threshold
            lost_pos = int((pruned_mask & (df["is_improvement"] == 1)).sum())
            recall_loss = (lost_pos / actual_pos) * 100 if actual_pos > 0 else 0.0
            rows.extend(
                [
                    ("Real improvements", actual_pos),
                    ("Lost opportunities", f"{lost_pos} (Recall Loss: {recall_loss:.2f}%)"),
                ]
            )

        _print_kv_table(f"Pruning Simulation (threshold={threshold})", rows)

    except Exception as e:
        err(f"Pruning simulation failed: {e}")


def _compute_pruning_stats(y_true: np.ndarray, p_pos: np.ndarray, *, threshold: float):
    n_total = int(len(p_pos))
    pruned_mask = (p_pos < threshold)
    n_pruned = int(pruned_mask.sum())
    pct_pruned = (n_pruned / n_total) * 100 if n_total else 0.0
    actual_pos = int(y_true.sum())
    lost_pos = int(((y_true == 1) & pruned_mask).sum())
    recall_loss = (lost_pos / actual_pos) * 100 if actual_pos > 0 else 0.0
    return {
        "total": n_total,
        "pruned": n_pruned,
        "pct_pruned": pct_pruned,
        "kept": n_total - n_pruned,
        "actual_pos": actual_pos,
        "lost_pos": lost_pos,
        "recall_loss": recall_loss,
    }


def _bootstrap_auc_diff(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, *, n_boot: int, seed: int):
    """Paired bootstrap of AUC difference (A - B) on the same samples."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        # Skip degenerate resamples
        if np.unique(yb).size < 2:
            continue
        try:
            auc_a = roc_auc_score(yb, p_a[idx])
            auc_b = roc_auc_score(yb, p_b[idx])
        except Exception:
            continue
        diffs.append(float(auc_a - auc_b))

    if not diffs:
        return {"n": 0, "diff_mean": None, "ci95": (None, None), "p_value": None}

    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    diff_mean = float(diffs.mean())
    # Two-sided p-value for H0: diff == 0
    p_le = float((diffs <= 0).mean())
    p_ge = float((diffs >= 0).mean())
    p_two = 2.0 * min(p_le, p_ge)
    p_two = float(min(1.0, p_two))
    return {"n": int(len(diffs)), "diff_mean": diff_mean, "ci95": (float(lo), float(hi)), "p_value": p_two}


def _fit_fast_logreg(train_df: pd.DataFrame, *, label: str):
    """Fast baseline used for ablation checks (not production oracle)."""
    X_train = train_df.drop(columns=[label])
    y_train = train_df[label].astype(int).values

    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object" or str(X_train[c].dtype).startswith("category")]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    transformers = []
    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))
    if num_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("num", num_pipe, num_cols))

    if not transformers:
        raise ValueError("No features available for ablation model")

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LogisticRegression(
        max_iter=2000,
        solver="saga",
        n_jobs=1,
        class_weight="balanced",
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def run_ablation_checks(df: pd.DataFrame, *, eps: float, threshold: float, test_size: float, seed: int, n_boot: int):
    """Ablation + significance checks focused on prev_* signal.

    Uses a fast logistic-regression baseline trained on a held-out split.
    """
    if not _SKLEARN_OK:
        info("Ablation checks skipped (scikit-learn not available).")
        return

    base_df, _ = _prepare_training_frame(df, eps=eps)
    if "is_improvement" not in base_df.columns:
        info("Ablation checks skipped (label missing).")
        return

    # Split once, reuse for all variants (paired comparison)
    train_idx, test_idx = train_test_split(
        np.arange(len(base_df)),
        test_size=float(test_size),
        random_state=int(seed),
        stratify=base_df["is_improvement"].astype(int) if base_df["is_improvement"].nunique() > 1 else None,
    )
    base_train = base_df.iloc[train_idx].reset_index(drop=True)
    base_test = base_df.iloc[test_idx].reset_index(drop=True)

    # Build variants
    def drop_prev_all(d: pd.DataFrame):
        cols = [c for c in d.columns if c.startswith("prev_")]
        return d.drop(columns=cols, errors="ignore")

    def keep_prev_min(d: pd.DataFrame):
        keep = {"prev_action_group", "prev_action_variant"}
        cols = [c for c in d.columns if c.startswith("prev_") and c not in keep]
        return d.drop(columns=cols, errors="ignore")

    variants = {
        "FULL": (base_train, base_test),
        "NO_PREV": (drop_prev_all(base_train), drop_prev_all(base_test)),
        "PREV_MIN": (keep_prev_min(base_train), keep_prev_min(base_test)),
    }

    results = {}
    y_test = base_test["is_improvement"].astype(int).values

    _print_kv_table(
        "Ablation Settings",
        [
            ("test_size", test_size),
            ("seed", seed),
            ("bootstrap", n_boot),
        ],
    )

    table = Table(title="Ablation Results", box=box.ASCII)
    table.add_column("Variant")
    table.add_column("AUC")
    table.add_column("AP")
    table.add_column("Pruned %")
    table.add_column("Recall loss %")

    for name, (tr, te) in variants.items():
        label = "is_improvement"
        feature_cols = [c for c in tr.columns if c != label and tr[c].notna().any()]
        if not feature_cols:
            table.add_row(name, "SKIPPED", "SKIPPED", "-", "-")
            continue

        tr = tr[[label] + feature_cols]
        te = te[[label] + feature_cols]

        model = _fit_fast_logreg(tr, label=label)
        X_te = te.drop(columns=[label])
        p_pos = model.predict_proba(X_te)[:, 1]

        auc = None
        ap = None
        if np.unique(y_test).size >= 2:
            auc = float(roc_auc_score(y_test, p_pos))
            ap = float(average_precision_score(y_test, p_pos))

        pruning = _compute_pruning_stats(y_test, p_pos, threshold=threshold)
        results[name] = {"auc": auc, "ap": ap, "p": p_pos, "pruning": pruning}

        table.add_row(
            name,
            f"{auc:.4f}" if auc is not None else "NA",
            f"{ap:.4f}" if ap is not None else "NA",
            f"{pruning['pct_pruned']:.1f}%",
            f"{pruning['recall_loss']:.2f}%",
        )

    console.print(table)

    diff_rows = []
    if results.get("FULL", {}).get("auc") is not None:
        p_full = results["FULL"]["p"]
        for other in ("NO_PREV", "PREV_MIN"):
            if results.get(other, {}).get("auc") is None:
                continue
            boot = _bootstrap_auc_diff(y_test, p_full, results[other]["p"], n_boot=n_boot, seed=seed)
            if boot["n"]:
                lo, hi = boot["ci95"]
                diff_rows.append(
                    (
                        f"FULL - {other}",
                        f"{boot['diff_mean']:+.4f}",
                        f"[{lo:+.4f}, {hi:+.4f}]",
                        f"{boot['p_value']:.3f}",
                        str(boot["n"]),
                    )
                )
            else:
                diff_rows.append((f"FULL - {other}", "NA", "NA", "NA", "0"))

    if diff_rows:
        diff_table = Table(title="AUC Diff (bootstrap)", box=box.ASCII)
        diff_table.add_column("Comparison")
        diff_table.add_column("Mean")
        diff_table.add_column("CI95")
        diff_table.add_column("p")
        diff_table.add_column("n")
        for row in diff_rows:
            diff_table.add_row(*row)
        console.print(diff_table)


def _log_post_training_summary(*, out_dir, link_path, csv_name, df, eps, threshold, diagnostics):
    info("=" * 60)
    info("POST-TRAINING SUMMARY")

    # Data/links
    info(f"Data link updated: {link_path.name} -> {csv_name}")

    # Dataset stats
    n = len(df)
    pos_rate = float((df["delta_score"] > eps).mean()) if n else 0.0
    info(f"Rows: {n} | eps: {eps} | label positive-rate: {pos_rate:.4f}")

    if diagnostics:
        raw_n = diagnostics.get("raw_transitions")
        valid_n = diagnostics.get("valid_transitions")
        info(f"Transitions: raw={raw_n} | valid={valid_n}")

        p_cov = diagnostics.get("parent_final_coverage")
        c_cov = diagnostics.get("child_final_coverage")
        if p_cov is not None and c_cov is not None:
            info(
                f"details_json.final_value coverage: parent={100*p_cov:.1f}% | child={100*c_cov:.1f}% (fallback to raw value otherwise)"
            )

        dcounts = diagnostics.get("direction_counts")
        if dcounts:
            info(f"direction counts (DB): {dcounts} (direction==1 treated as minimize; delta_score is signed)" )

        sf = diagnostics.get("study_filter") or {}
        info(f"study filter: study_id={sf.get('study_id')} | study_name={sf.get('study_name')}")

    info("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Project slug under projects/kaggle/")
    parser.add_argument("--db", help="Path to mcts.db (overrides --project default)")
    parser.add_argument("--out-dir", help="Output directory for model and data (overrides --project default)")
    parser.add_argument("--study-id", help="Filter by study_id (numeric) or study_name")
    parser.add_argument("--study-name", help="Filter by study_name")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.20, help="Pruning threshold for report (default: 0.20)")
    parser.add_argument("--eps", type=float, default=None, help="Margin for improvement label (default: config or 0.0)")

    # Optional diagnostics
    parser.add_argument("--no-ablation", action="store_true", help="Disable ablation/significance checks")
    parser.add_argument("--ablation-test-size", type=float, default=0.25, help="Holdout fraction for ablation (default: 0.25)")
    parser.add_argument("--ablation-seed", type=int, default=7, help="Seed for ablation split/bootstrap (default: 7)")
    parser.add_argument("--ablation-bootstrap", type=int, default=500, help="Bootstrap reps for AUC diff (default: 500)")
    args = parser.parse_args()

    if args.project:
        base_dir = Path("projects/kaggle") / args.project / "experiments"
        db_path = Path(args.db) if args.db else base_dir / "db" / "mcts.db"
        out_dir = Path(args.out_dir) if args.out_dir else base_dir / "oracle" / "model"
    else:
        if not args.db or not args.out_dir:
            parser.error("--db and --out-dir are required unless --project is provided")
        db_path = Path(args.db)
        out_dir = Path(args.out_dir)

    default_mcts = {}
    study_id = None
    study_name = args.study_name
    if args.study_id:
        if str(args.study_id).isdigit():
            study_id = int(args.study_id)
        else:
            if study_name and study_name != args.study_id:
                warn("Both --study-id (non-numeric) and --study-name provided. Using --study-name.")
            else:
                study_name = args.study_id
                info(f"Interpreting --study-id as study_name: {study_name}")

    if study_id is None and not study_name:
        default_mcts = _load_default_mcts_settings(args.project)
        if default_mcts.get("study_name"):
            study_name = default_mcts["study_name"]
            info(f"Using default study_name from mla_super_chain.yaml: {study_name}")

    if args.eps is None:
        if not default_mcts:
            default_mcts = _load_default_mcts_settings(args.project)
        if default_mcts.get("oracle_eps") is not None:
            args.eps = float(default_mcts["oracle_eps"])
            info(f"Using default oracle eps from mla_super_chain.yaml: {args.eps}")
        else:
            args.eps = 0.0

    target_dir = out_dir
    staging_dir = _prepare_staging_dir(target_dir)

    # 1. Extract
    df = extract_data(db_path, study_id=study_id, study_name=study_name)
    if df.empty:
        err("No valid transitions found. Exiting.")
        return

    diagnostics = getattr(df, "attrs", {}).get("diagnostics", {})

    # 2. Train
    try:
        predictor, feature_cols = train_model(df, staging_dir, num_gpus=args.num_gpus, eps=args.eps)

        # 2b. Report
        generate_pruning_report(predictor, df, feature_cols, threshold=args.threshold)

    except Exception as e:
        err(f"Training failed: {e}")
        return

    # 3. Save Data & Link (Atomic data link)
    timestamp = int(time.time())
    csv_name = f"mcts_oracle_{timestamp}.csv"
    csv_path = staging_dir / csv_name

    info(f"Saving training data to {csv_path}...")
    df.to_csv(csv_path, index=False)

    link_path = staging_dir / "mcts_oracle.csv"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()

    # Relative symlink is safer if folder moves
    link_path.symlink_to(csv_name)
    info(f"Updated production data link in staging: {link_path.name} -> {csv_name}")

    # 4. Swap staging into place
    try:
        _atomic_swap_dirs(target_dir, staging_dir)
    except Exception as e:
        err(f"Atomic model swap failed: {e}")
        return

    out_dir = target_dir
    link_path = out_dir / "mcts_oracle.csv"
    info(f"Activated new model directory: {out_dir}")

    # 5. Summary (after relinking)
    _log_post_training_summary(
        out_dir=out_dir,
        link_path=link_path,
        csv_name=csv_name,
        df=df,
        eps=args.eps,
        threshold=args.threshold,
        diagnostics=diagnostics,
    )

    # 6. Ablation + significance checks (optional)
    if not args.no_ablation:
        try:
            run_ablation_checks(
                df,
                eps=args.eps,
                threshold=args.threshold,
                test_size=args.ablation_test_size,
                seed=args.ablation_seed,
                n_boot=args.ablation_bootstrap,
            )
        except Exception as e:
            warn(f"Ablation checks failed: {e}")

    info("MCTS Oracle update complete successfully.")


if __name__ == "__main__":
    main()

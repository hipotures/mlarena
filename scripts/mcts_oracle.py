#!/usr/bin/env python
import sqlite3
import pandas as pd
import json
import argparse
import time
import shutil
import logging
from pathlib import Path
import yaml
from autogluon.tabular import TabularPredictor

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def _extract_final_value(details_json):
    if not details_json or pd.isna(details_json):
        return None
    try:
        data = json.loads(details_json)
    except Exception:
        return None

def _load_default_mcts_settings():
    config_path = Path(__file__).resolve().parents[1] / "conf" / "preprocess" / "mla_super_chain.yaml"
    if not config_path.exists():
        logger.warning(f"Default config not found at {config_path}")
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
        logger.warning(f"Failed to read default study_name from {config_path}: {e}")
        return {}
    value = data.get("final_value")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None

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
    except Exception as e:
        return {}

def extract_data(db_path, study_id=None, study_name=None):
    logger.info(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # 1. Get Action History (Map trial_id -> action_json)
    history_query = "SELECT child_trial_id, action_json FROM mcts_edges"
    history_df = pd.read_sql_query(history_query, conn)
    node_action_map = dict(zip(history_df['child_trial_id'], history_df['action_json']))
    
    # 2. Extract Transitions with Best Evaluation per Trial
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
    
    logger.info("Executing main query...")
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    logger.info(f"Raw transitions found: {len(df)}")
    
    # Deduplicate (just in case SQL missed something, though rn=1 is robust)
    df = df.drop_duplicates(subset=['parent_id', 'child_id'])
    
    # Filter valid targets
    df["parent_final"] = df["parent_details_json"].apply(_extract_final_value)
    df["child_final"] = df["child_details_json"].apply(_extract_final_value)
    df["parent_score"] = df["parent_final"].where(df["parent_final"].notna(), df["parent_score_raw"])
    df["child_score"] = df["child_final"].where(df["child_final"].notna(), df["child_score_raw"])

    df = df.dropna(subset=['child_score', 'parent_score'])
    logger.info(f"Valid transitions (with scores): {len(df)}")
    
    if df.empty:
        return pd.DataFrame()

    # Calculate Delta
    df["direction"] = df["direction"].fillna(2)
    df['delta_score'] = df['child_score'] - df['parent_score']
    df.loc[df["direction"] == 1, "delta_score"] *= -1
    
    # Fill defaults for context
    if 'prev_duration' in df.columns:
        df['prev_duration'] = df['prev_duration'].fillna(0.0)

    # Add Previous Action JSON context
    df['prev_action_json'] = df['parent_id'].map(node_action_map)
    
    # Parse JSONs
    logger.info("Parsing action JSONs...")
    curr_actions = df['curr_action_json'].apply(lambda x: parse_action(x, prefix=""))
    curr_actions_df = pd.DataFrame(curr_actions.tolist())
    
    prev_actions = df['prev_action_json'].apply(lambda x: parse_action(x, prefix="prev_"))
    prev_actions_df = pd.DataFrame(prev_actions.tolist())
    
    # Combine
    meta_df = pd.concat([
        df[['parent_score', 'depth', 'delta_score', 'prev_duration']],
        curr_actions_df,
        prev_actions_df
    ], axis=1)
    
    return meta_df

def clean_output_dir(path):
    p = Path(path)
    if not p.exists(): return
    
    # Safety check
    if len(p.parts) < 2 or p.name not in ("oracle", "meta_model", "experiments"):
        # Allow generic if it ends with 'oracle'
        if not str(p).endswith("oracle"):
            logger.warning(f"Safety warning: refusing to clean generic path {p}")
            return

    logger.info(f"Cleaning output directory: {p}")
    for item in p.iterdir():
        if item.name.endswith(".csv"):
            continue 
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete {item}: {e}")

def train_model(df, output_dir, num_gpus=0, eps=0.0):
    # Prepare Target
    df['is_improvement'] = (df['delta_score'] > eps).astype(int)
    
    # Drop Leakage
    drop_cols = ['child_score', 'delta_score']
    train_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    logger.info(f"Training classification on {len(train_df)} rows. Target: is_improvement")
    
    clean_output_dir(output_dir)
    
    predictor = TabularPredictor(
        label='is_improvement', 
        path=str(output_dir),
        eval_metric='roc_auc',
        problem_type='binary'
    ).fit(
        train_df, 
        time_limit=600,  # 10 minutes
        presets='best_quality', 
        num_gpus=num_gpus,
        hyperparameters={
            'GBM': {},
            'CAT': {},
            'XGB': {}
        }
    )
    
    logger.info("Leaderboard:")
    print(predictor.leaderboard(display=True))
    return predictor

def generate_pruning_report(predictor, df, threshold=0.2):
    logger.info("-" * 40)
    logger.info(f"PRUNING SIMULATION (Threshold={threshold})")
    
    # Predict
    # Drop leakage columns (df here has is_improvement and delta_score)
    X = df.drop(columns=['child_score', 'delta_score', 'is_improvement'], errors='ignore')
    
    try:
        probs = predictor.predict_proba(X)
        pos_col = 1 if 1 in probs.columns else probs.columns[-1]
        p_vals = probs[pos_col]
        
        # Stats
        n_total = len(p_vals)
        n_pruned = (p_vals < threshold).sum()
        pct_pruned = (n_pruned / n_total) * 100
        
        logger.info(f"Total Actions: {n_total}")
        logger.info(f"Pruned: {n_pruned} ({pct_pruned:.2f}%)")
        logger.info(f"Kept:   {n_total - n_pruned} ({100 - pct_pruned:.2f}%)")
        
        # Recall Check
        if 'is_improvement' in df.columns:
            actual_pos = df['is_improvement'].sum()
            pruned_mask = (p_vals < threshold)
            lost_pos = (pruned_mask & df['is_improvement']).sum()
            
            recall_loss = (lost_pos / actual_pos) * 100 if actual_pos > 0 else 0.0
            logger.info(f"Real Improvements: {actual_pos}")
            logger.info(f"Lost Opportunities: {lost_pos} (Recall Loss: {recall_loss:.2f}%)")
            
    except Exception as e:
        logger.error(f"Pruning simulation failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Project slug under projects/kaggle/")
    parser.add_argument("--db", help="Path to mcts.db (overrides --project default)")
    parser.add_argument("--out-dir", help="Output directory for model and data (overrides --project default)")
    parser.add_argument("--study-id", type=int, help="Filter by study_id")
    parser.add_argument("--study-name", help="Filter by study_name")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.20, help="Pruning threshold for report (default: 0.20)")
    parser.add_argument("--eps", type=float, default=None, help="Margin for improvement label (default: config or 0.0)")
    args = parser.parse_args()

    if args.project:
        base_dir = Path("projects/kaggle") / args.project / "experiments"
        db_path = Path(args.db) if args.db else base_dir / "db" / "mcts.db"
        out_dir = Path(args.out_dir) if args.out_dir else base_dir / "oracle"
    else:
        if not args.db or not args.out_dir:
            parser.error("--db and --out-dir are required unless --project is provided")
        db_path = Path(args.db)
        out_dir = Path(args.out_dir)

    default_mcts = {}
    if args.study_id is None and not args.study_name:
        default_mcts = _load_default_mcts_settings()
        if default_mcts.get("study_name"):
            args.study_name = default_mcts["study_name"]
            logger.info(f"Using default study_name from mla_super_chain.yaml: {args.study_name}")
    if args.eps is None:
        if not default_mcts:
            default_mcts = _load_default_mcts_settings()
        if default_mcts.get("oracle_eps") is not None:
            args.eps = float(default_mcts["oracle_eps"])
            logger.info(f"Using default oracle eps from mla_super_chain.yaml: {args.eps}")
        else:
            args.eps = 0.0

    # Ensure output dir exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract
    if args.study_id is not None and args.study_name:
        logger.warning("Both --study-id and --study-name provided. Using --study-id.")
    df = extract_data(db_path, study_id=args.study_id, study_name=args.study_name)
    if df.empty:
        logger.error("No valid transitions found. Exiting.")
        return

    # 2. Train
    try:
        predictor = train_model(df, out_dir, num_gpus=args.num_gpus, eps=args.eps)
        
        # 2b. Report
        generate_pruning_report(predictor, df, threshold=args.threshold)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    # 3. Save Data & Link (Atomic Switch logic)
    timestamp = int(time.time())
    csv_name = f"mcts_oracle_{timestamp}.csv"
    csv_path = out_dir / csv_name
    
    logger.info(f"Saving training data to {csv_path}...")
    df.to_csv(csv_path, index=False)
    
    link_path = out_dir / "mcts_oracle.csv"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
        
    # Relative symlink is safer if folder moves
    link_path.symlink_to(csv_name)
    logger.info(f"Updated production data link: {link_path.name} -> {csv_name}")
    
    logger.info("MCTS Oracle update complete successfully.")

if __name__ == "__main__":
    main()

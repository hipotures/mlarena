#!/usr/bin/env python
import sqlite3
import pandas as pd
import json
import argparse
import time
import shutil
import logging
from pathlib import Path
from autogluon.tabular import TabularPredictor

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

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

def extract_data(db_path):
    logger.info(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    
    # 1. Extract Transitions with Best Evaluation per Trial
    query = """
    WITH eval_ranked AS (
        SELECT
            trial_id,
            fidelity,
            status,
            value,
            metric_name,
            duration_sec,
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
        edge.action_json as curr_action_json,
        eval_parent.value as parent_score,
        eval_child.value as child_score,
        child_node.depth
    FROM mcts_edges edge
    JOIN mcts_nodes parent_node ON edge.parent_trial_id = parent_node.trial_id
    JOIN mcts_nodes child_node ON edge.child_trial_id = child_node.trial_id
    JOIN trials parent ON parent_node.trial_id = parent.trial_id
    JOIN trials child ON child_node.trial_id = child.trial_id
    LEFT JOIN eval_ranked eval_parent ON parent.trial_id = eval_parent.trial_id AND eval_parent.rn = 1
    LEFT JOIN eval_ranked eval_child ON child.trial_id = eval_child.trial_id AND eval_child.rn = 1
    """
    
    logger.info("Executing main query...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Raw transitions found: {len(df)}")
    
    # Filter valid targets
    df = df.dropna(subset=['child_score', 'parent_score'])
    logger.info(f"Valid transitions (with scores): {len(df)}")
    
    if df.empty:
        return pd.DataFrame()

    # Calculate Delta
    df['delta_score'] = df['child_score'] - df['parent_score']
    
    # Parse JSONs
    logger.info("Parsing current action JSONs...")
    curr_actions = df['curr_action_json'].apply(lambda x: parse_action(x, prefix=""))
    curr_actions_df = pd.DataFrame(curr_actions.tolist())
    
    # Combine (NO PREVIOUS ACTIONS)
    meta_df = pd.concat([
        df[['parent_score', 'depth', 'delta_score']], 
        curr_actions_df
    ], axis=1)
    
    return meta_df

def clean_output_dir(path):
    p = Path(path)
    if not p.exists(): return
    
    # Safety check
    if len(p.parts) < 2 or p.name not in ("oracle", "oracle_simple", "experiments"):
        if not str(p).endswith("oracle") and not str(p).endswith("oracle_simple"):
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

def train_model(df, output_dir, num_gpus=0):
    # Prepare Target
    df['is_improvement'] = (df['delta_score'] > 0).astype(int)
    
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
        time_limit=600, 
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
    
    X = df.drop(columns=['child_score', 'delta_score', 'is_improvement'], errors='ignore')
    
    try:
        probs = predictor.predict_proba(X)
        pos_col = 1 if 1 in probs.columns else probs.columns[-1]
        p_vals = probs[pos_col]
        
        n_total = len(p_vals)
        n_pruned = (p_vals < threshold).sum()
        pct_pruned = (n_pruned / n_total) * 100
        
        logger.info(f"Total Actions: {n_total}")
        logger.info(f"Pruned: {n_pruned} ({pct_pruned:.2f}%)")
        logger.info(f"Kept:   {n_total - n_pruned} ({100 - pct_pruned:.2f}%)")
        
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
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.20, help="Pruning threshold for report (default: 0.20)")
    args = parser.parse_args()

    if args.project:
        base_dir = Path("projects/kaggle") / args.project / "experiments"
        db_path = Path(args.db) if args.db else base_dir / "db" / "mcts.db"
        out_dir = Path(args.out_dir) if args.out_dir else base_dir / "oracle_simple"
    else:
        if not args.db or not args.out_dir:
            parser.error("--db and --out-dir are required unless --project is provided")
        db_path = Path(args.db)
        out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract
    df = extract_data(db_path)
    if df.empty:
        logger.error("No valid transitions found. Exiting.")
        return

    # 2. Train
    try:
        predictor = train_model(df, out_dir, num_gpus=args.num_gpus)
        generate_pruning_report(predictor, df, threshold=args.threshold)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return

    # 3. Save Data & Link
    timestamp = int(time.time())
    csv_name = f"mcts_oracle_simple_{timestamp}.csv"
    csv_path = out_dir / csv_name
    
    logger.info(f"Saving training data to {csv_path}...")
    df.to_csv(csv_path, index=False)
    
    link_path = out_dir / "mcts_oracle_simple.csv"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
        
    link_path.symlink_to(csv_name)
    logger.info(f"Updated production data link: {link_path.name} -> {csv_name}")
    
    logger.info("MCTS Oracle Simple update complete successfully.")

if __name__ == "__main__":
    main()

import sqlite3
import pandas as pd
import json
import argparse
import time
from pathlib import Path

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
            # Create prefixed key: prev_imputer_numeric_strategy or imputer_numeric_strategy
            # Note: We want group prefix ALWAYS (imputer_...) and context prefix OPTIONALLY (prev_...)
            key = f"{prefix}{group}_{k}"
            
            if isinstance(v, (list, dict)):
                flat[key] = json.dumps(v, sort_keys=True)
            else:
                flat[key] = v
                
        return flat
    except Exception as e:
        return {}

def main(db_path, output_path):
    conn = sqlite3.connect(db_path)
    
    # 1. Get all edges to build a map: child_id -> action_json (History)
    print("Loading action history...")
    history_query = "SELECT child_trial_id, action_json FROM mcts_edges"
    history_df = pd.read_sql_query(history_query, conn)
    # Map trial_id -> action_json_str
    node_action_map = dict(zip(history_df['child_trial_id'], history_df['action_json']))
    
    # 2. Get main transitions with scores
    print("Loading transitions...")
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
        parent.trial_id as parent_id,
        child.trial_id as child_id,
        edge.action_json as curr_action_json,
        eval_parent.value as parent_score,
        eval_child.value as child_score,
        child_node.depth,
        parent_node.n_visits as parent_visits,
        eval_parent.duration_sec as prev_duration
    FROM mcts_edges edge
    JOIN mcts_nodes parent_node ON edge.parent_trial_id = parent_node.trial_id
    JOIN mcts_nodes child_node ON edge.child_trial_id = child_node.trial_id
    JOIN trials parent ON parent_node.trial_id = parent.trial_id
    JOIN trials child ON child_node.trial_id = child.trial_id
    LEFT JOIN eval_ranked eval_parent ON parent.trial_id = eval_parent.trial_id AND eval_parent.rn = 1
    LEFT JOIN eval_ranked eval_child ON child.trial_id = eval_child.trial_id AND eval_child.rn = 1
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Raw transitions: {len(df)}")
    
    # Filter valid targets (must have both scores to calculate delta)
    # Note: Root node (depth 0) usually has no parent_score if it's the absolute start, 
    # but in our query logic, we require an edge. 
    # Edges start from Root -> Child. 
    # If Root has no evaluation, we lose the first step.
    # Let's check if we can assume parent_score for root if missing.
    # But for now, strict filtering ensures high quality data.
    df = df.dropna(subset=['child_score', 'parent_score'])
    print(f"Valid transitions (with scores): {len(df)}")
    
    if len(df) == 0:
        print("No valid transitions found. Exiting.")
        return

    if 'prev_duration' in df.columns:
        df['prev_duration'] = df['prev_duration'].fillna(0.0)

    df['delta_score'] = df['child_score'] - df['parent_score']
    
    # 3. Add Previous Action JSON
    # parent_id in current row was a child_id in a previous edge
    df['prev_action_json'] = df['parent_id'].map(node_action_map)
    
    # 4. Parse JSONs
    print("Parsing current actions...")
    curr_actions = df['curr_action_json'].apply(lambda x: parse_action(x, prefix=""))
    curr_actions_df = pd.DataFrame(curr_actions.tolist())
    
    print("Parsing previous actions...")
    prev_actions = df['prev_action_json'].apply(lambda x: parse_action(x, prefix="prev_"))
    prev_actions_df = pd.DataFrame(prev_actions.tolist())
    
    # 5. Combine
    meta_df = pd.concat([
        df[['parent_score', 'depth', 'delta_score', 'parent_visits', 'prev_duration']], 
        curr_actions_df,
        prev_actions_df
    ], axis=1)
    
    # Versioning based on timestamp (robust for automated runs)
    timestamp = int(time.time())
    
    # Target file: experiments/oracle/mcts_oracle_1700000000.csv
    base_out = Path(output_path)
    versioned_name = f"{base_out.stem}_{timestamp}{base_out.suffix}"
    versioned_path = base_out.parent / versioned_name
    
    # Ensure directory exists
    versioned_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Feature columns: {len(meta_df.columns)}")
    print(f"Saving versioned dataset to {versioned_path}...")
    meta_df.to_csv(versioned_path, index=False)
    
    print(f"Dataset created: {versioned_path}")
    # Note: Symlink update moved to training script to ensure atomicity with model success.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Project slug under projects/kaggle/")
    parser.add_argument("--db", help="Path to mcts.db (overrides --project default)")
    parser.add_argument("--out", help="Output CSV path (overrides --project default)")
    args = parser.parse_args()

    if args.project:
        base_dir = Path("projects/kaggle") / args.project / "experiments"
        db_path = Path(args.db) if args.db else base_dir / "db" / "mcts.db"
        out_path = Path(args.out) if args.out else base_dir / "oracle" / "mcts_oracle.csv"
    else:
        if not args.db or not args.out:
            parser.error("--db and --out are required unless --project is provided")
        db_path = Path(args.db)
        out_path = Path(args.out)

    main(str(db_path), str(out_path))

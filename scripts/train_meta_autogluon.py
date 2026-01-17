import pandas as pd
import argparse
import os
import shutil
from pathlib import Path

def clean_output_dir(path):
    p = Path(path)
    if not p.exists(): return
    
    # Safety check: avoid deleting root/project/experiments by mistake
    # Require last component to be specific
    if len(p.parts) < 2 or p.name not in ("oracle", "meta_model", "meta_autogluon", "meta_autogluon_clf"):
        # If user named it "my_model", allow it too? 
        # Let's rely on it being a subdirectory.
        pass

    print(f"Cleaning output directory: {p}")
    for item in p.iterdir():
        # Preserve training data if it's in the same folder
        if item.name.endswith(".csv"):
            continue 
        try:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        except Exception as e:
            print(f"Failed to delete {item}: {e}")

def train(data_path, output_dir):
    clean_output_dir(output_dir)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Clean target
    df = df.dropna(subset=['delta_score'])
    
    # Create classification target
    df['is_improvement'] = (df['delta_score'] > 0).astype(int)
    
    # Drop leakage columns
    # child_score and delta_score contain the answer
    drop_cols = ['child_score', 'delta_score']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
    print(f"Training classification on {len(df)} rows. Target: is_improvement")
    
    predictor = TabularPredictor(
        label='is_improvement', 
        path=output_dir,
        eval_metric='roc_auc',
        problem_type='binary'
    ).fit(
        df, 
        time_limit=600,  # 10 minutes
        presets='best_quality', # Enables stacking/bagging - crucial for small data
        num_gpus=0, # Force CPU
        hyperparameters={
            'GBM': {},
            'CAT': {},
            'XGB': {}
        }
    )
    
    print("\nLeaderboard:")
    print(predictor.leaderboard(display=True))
    
    # Detailed feature importance - DISABLED for speed
    # print("\nFeature Importance:")
    # try:
    #     fi = predictor.feature_importance(df)
    #     print(fi.head(20))
    #     fi.to_csv(os.path.join(output_dir, "feature_importance.csv"))
    # except Exception as e:
    #     print(f"Could not calculate feature importance: {e}")
        
    # Update Data Symlink (Atomic Switch)
    # We point mcts_oracle.csv to the file we just trained on.
    try:
        p_data = Path(data_path).resolve()
        p_out = Path(output_dir).resolve()
        
        # Only update link if data is in the model folder (standard workflow)
        if p_data.parent == p_out:
            link_path = p_out / "mcts_oracle.csv"
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(p_data.name)
            print(f"Updated production symlink: {link_path.name} -> {p_data.name}")
    except Exception as e:
        print(f"Warning: Failed to update data symlink: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", help="Project slug under projects/kaggle/")
    parser.add_argument("--data", help="Path to input CSV (overrides --project default)")
    parser.add_argument("--out-dir", help="Output dir for AutoGluon artifacts (overrides --project default)")
    args = parser.parse_args()

    if args.project:
        base_dir = Path("projects/kaggle") / args.project / "experiments"
        data_path = Path(args.data) if args.data else base_dir / "oracle" / "mcts_oracle.csv"
        out_dir = Path(args.out_dir) if args.out_dir else base_dir / "oracle"
    else:
        if not args.data:
            parser.error("--data is required unless --project is provided")
        data_path = Path(args.data)
        out_dir = Path(args.out_dir) if args.out_dir else Path("/tmp") / "oracle"

    train(str(data_path), str(out_dir))

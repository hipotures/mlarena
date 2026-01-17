from autogluon.tabular import TabularPredictor
import pandas as pd
import argparse
import sys

def check_pruning(model_path, data_path, threshold=0.20):
    print(f"Loading model from {model_path}...")
    try:
        predictor = TabularPredictor.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Prepare features
    drop_cols = ['child_score', 'delta_score', 'relative_delta']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    print("Predicting...")
    probs = predictor.predict_proba(X)
    
    # Get positive class prob
    if 1 in probs.columns:
        p_vals = probs[1]
    else:
        # Fallback to last column (usually positive in binary)
        p_vals = probs.iloc[:, -1]
        
    # Analysis
    n_total = len(p_vals)
    n_pruned = (p_vals < threshold).sum()
    percent_pruned = (n_pruned / n_total) * 100
    
    print("-" * 40)
    print(f"THRESHOLD: {threshold}")
    print(f"Total Sample: {n_total}")
    print(f"PRUNED: {n_pruned} ({percent_pruned:.2f}%)")
    print(f"KEPT:   {n_total - n_pruned} ({100 - percent_pruned:.2f}%)")
    
    # Validation against Ground Truth
    if 'delta_score' in df.columns:
        # Ground truth: Did it improve?
        is_improvement = (df['delta_score'] > 0)
        n_improvements = is_improvement.sum()
        
        # Pruned actions that were actually good
        pruned_mask = (p_vals < threshold)
        lost_opportunities = (pruned_mask & is_improvement).sum()
        
        print("-" * 40)
        print(f"Real Improvements in data: {n_improvements}")
        print(f"Lost Opportunities (FN):   {lost_opportunities}")
        print(f"Recall: {(1 - lost_opportunities/n_improvements):.2%} (We kept {n_improvements - lost_opportunities} good actions)")
        print(f"Precision of Pruning: {(1 - lost_opportunities/n_pruned):.2%} of pruned actions were indeed bad (True Negatives / Pruned)")

if __name__ == "__main__":
    check_pruning(
        "/home/xai/ml/kaggle/projects/kaggle/playground-series-s6e1/experiments/oracle",
        "projects/kaggle/playground-series-s6e1/data/meta_mcts_train_full.csv"
    )

import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import argparse
import pickle
from pathlib import Path
import numpy as np

def train(data_path, output_dir):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Target definition
    target_col = 'delta_score'
    
    # Drop rows where target is NaN
    df = df.dropna(subset=[target_col])
    print(f"Samples after target clean: {len(df)}")
    
    # Drop non-feature columns explicitly
    # We KEEP parent_score and depth because they provide context (difficulty of improvement)
    drop_cols = ['delta_score', 'child_score'] 
    if 'is_improvement' in df.columns:
        drop_cols.append('is_improvement')
        
    features = [c for c in df.columns if c not in drop_cols]
    
    print(f"Training with {len(features)} features on {len(df)} samples.")
    
    X = df[features].copy() # Make a copy to avoid SettingWithCopy warnings
    y = df[target_col]
    
    # Handle categorical columns strictly as strings for CatBoost
    cat_features = []
    
    # Heuristic for detecting categorical columns in sparse flattened data
    # 1. Object dtype
    # 2. Columns ending in _group or _variant
    # 3. Columns that look like params but are clearly categorical strings
    
    for col in X.columns:
        is_cat = False
        if X[col].dtype == 'object':
            is_cat = True
        elif col.endswith('action_group') or col.endswith('action_variant'):
             is_cat = True
        
        if is_cat:
            X[col] = X[col].astype(str).fillna("NaN")
            cat_features.append(col)
        else:
            # Numeric columns with NaNs are fine for CatBoost
            pass
            
    print(f"Identified {len(cat_features)} categorical features.")
            
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train CatBoost
    model = cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        cat_features=cat_features,
        verbose=100,
        loss_function='RMSE',
        eval_metric='RMSE'
    )
    
    print("Starting training...")
    model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=100)
    
    # Eval
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Test RMSE: {rmse:.5f}")
    
    # Classification Proxy Eval
    # If we predicted > 0 (improvement), was it actually > 0?
    actual_pos = (y_test > 0)
    pred_pos = (preds > 0)
    acc = accuracy_score(actual_pos, pred_pos)
    print(f"Direction Accuracy (Predict > 0): {acc:.2%}")
    
    # Top 10 Accuracy (Ranking Proxy)
    # If we pick the top 10 predicted improvements, how many were real improvements?
    test_res = pd.DataFrame({'actual': y_test, 'pred': preds})
    test_res = test_res.sort_values(by='pred', ascending=False)
    top_10 = test_res.head(10)
    top_10_hits = (top_10['actual'] > 0).sum()
    print(f"Top 10 Precision: {top_10_hits}/10 positive")

    # Feature Importance
    print("\nFeature Importance:")
    fi = model.get_feature_importance()
    fi_df = pd.DataFrame({'feature': X.columns, 'importance': fi}).sort_values(by='importance', ascending=False)
    print(fi_df.head(15))
    
    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model_path = out_path / "meta_model_v1.cbm"
    model.save_model(str(model_path))
    print(f"Model saved to {model_path}")
    
    # Save Feature Importance CSV
    fi_df.to_csv(out_path / "feature_importance.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", default="artifacts/meta_model")
    args = parser.parse_args()
    train(args.data, args.out_dir)

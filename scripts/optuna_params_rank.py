#!/usr/bin/env python3
import argparse
import pandas as pd
import optuna
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    storage = f"sqlite:///{Path(args.db).resolve()}"
    study = optuna.load_study(study_name=None, storage=storage)
    df = study.trials_dataframe()

    # Drop failed/running trials
    df = df[df.state == "COMPLETE"]
    
    param_cols = [c for c in df.columns if c.startswith('params_')]
    
    # Calculate correlation with objective value
    correlations = []
    for col in param_cols:
        # Try to convert to numeric if possible (for correlation)
        try:
            series = pd.to_numeric(df[col], errors='coerce')
            if series.nunique() > 1:
                corr = series.corr(df['value'])
                correlations.append({'param': col.replace('params_', ''), 'corr': corr, 'abs_corr': abs(corr)})
        except:
            continue

    corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)
    
    print(f"\nTop 20 Parameters by Correlation with Objective Value (n={len(df)} trials):")
    print("-" * 80)
    for _, row in corr_df.head(20).iterrows():
        direction = "+" if row['corr'] > 0 else "-"
        print(f"  {row['param']:50} | {row['corr']:+.4f} ({direction})")

if __name__ == "__main__":
    main()


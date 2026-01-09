#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import optuna
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator

def show_image(path):
    # Try kitty icat first
    try:
        subprocess.run(["kitty", "+kitten", "icat", str(path)], check=True)
    except:
        try:
            # Fallback to general icat if available
            subprocess.run(["icat", str(path)], check=True)
        except:
            print(f"Could not display image. File saved at: {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--type", choices=["history", "importance", "parallel", "slice", "box", "rank", "tpe", "contour", "timeline", "edf"], default="importance")
    parser.add_argument("--params", help="Comma-separated list of params to plot (e.g. 'param1,param2')")
    args = parser.parse_args()

    storage = f"sqlite:///{Path(args.db).resolve()}"
    study = optuna.load_study(study_name=None, storage=storage)
    
    # Pre-select top parameters
    df = study.trials_dataframe()
    df = df[df.state == "COMPLETE"].copy()

    if args.type == "timeline":
        from optuna.visualization.matplotlib import plot_timeline
        print("Generating timeline plot...")
        plt.figure(figsize=(16, 9))
        plot_timeline(study)
        plt.tight_layout()
        p_path = Path("/tmp/optuna_timeline.png")
        plt.savefig(p_path)
        plt.close()
        show_image(p_path)
        return

    if args.type == "edf":
        from optuna.visualization.matplotlib import plot_edf
        print("Generating EDF plot...")
        plt.figure(figsize=(16, 9))
        plot_edf(study)
        plt.tight_layout()
        p_path = Path("/tmp/optuna_edf.png")
        plt.savefig(p_path)
        plt.close()
        show_image(p_path)
        return
    
    if args.type == "contour":
        from optuna.visualization.matplotlib import plot_contour
        
        if args.params:
            top_2 = args.params.split(",")
        else:
            # Smart selection: find top correlated params that actually APPEAR together
            param_cols = [c for c in df.columns if c.startswith('params_')]
            candidates = []
            for col in param_cols:
                try:
                    c = pd.to_numeric(df[col], errors='coerce').corr(df['value'])
                    if not pd.isna(c) and df[col].nunique() > 1:
                        candidates.append((col, abs(c)))
                except: continue
            
            candidates = [name for name, _ in sorted(candidates, key=lambda x: x[1], reverse=True)]
            
            top_2 = []
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    p1, p2 = candidates[i], candidates[j]
                    common_count = df[df[p1].notna() & df[p2].notna()].shape[0]
                    if common_count >= 5: # Need at least some points to draw
                        top_2 = [p1.replace('params_', ''), p2.replace('params_', '')]
                        break
                if top_2: break
        
        if not top_2:
            print("Could not find any pair of parameters with enough common trials.")
            return

        print(f"Generating contour plot for: {top_2}")
        plt.figure(figsize=(16, 12))
        plot_contour(study, params=top_2)
        plt.title(f"Interactive Landscape: {top_2[0]} vs {top_2[1]}", fontsize=16)
        plt.tight_layout()
        
        p_path = Path("/tmp/optuna_contour.png")
        plt.savefig(p_path)
        plt.close()
        show_image(p_path)
        return
        import seaborn as sns
        # TPE splits trials into "good" and "bad" based on a quantile (default 0.1 or 0.25)
        threshold = df['value'].quantile(0.75) # Assuming maximization
        df['tpe_class'] = df['value'].apply(lambda x: 'Good (Top 25%)' if x >= threshold else 'Bad (Rest)')
        
        # Select top 3 most influential numeric params from correlation
        param_cols = [c for c in df.columns if c.startswith('params_')]
        corrs = []
        for col in param_cols:
            try:
                c = pd.to_numeric(df[col], errors='coerce').corr(df['value'])
                if not pd.isna(c) and df[col].nunique() > 2:
                    corrs.append((col, abs(c)))
            except: continue
        top_tpe_params = [name for name, _ in sorted(corrs, key=lambda x: x[1], reverse=True)[:4]]
        
        print(f"Visualizing TPE density for: {top_tpe_params}")
        
        for col in top_tpe_params:
            plt.figure(figsize=(16, 6))
            sns.kdeplot(data=df, x=col, hue='tpe_class', fill=True, common_norm=False, palette='viridis', alpha=.5, linewidth=2)
            plt.title(f"TPE Knowledge: Distribution of {col.replace('params_', '')}", fontsize=16)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            
            p_path = Path(f"/tmp/optuna_tpe_{col.replace('.', '_')}.png")
            plt.savefig(p_path)
            plt.close()
            show_image(p_path)
        return
        print("Generating Best Score Progression plot...")
        df['best_so_far'] = df['value'].cummax() # Assuming maximization
        
        plt.figure(figsize=(16, 9))
        plt.step(df['number'], df['best_so_far'], where='post', color='red', linewidth=3, label='Best Score')
        plt.scatter(df['number'], df['value'], alpha=0.3, color='blue', label='Trial Result')
        
        # Annotate top 3 improvements
        improvements = df[df['value'] == df['best_so_far']].drop_duplicates('best_so_far')
        for _, row in improvements.tail(3).iterrows():
            plt.annotate(f"{row['value']:.5f} (# {row['number']})", 
                         xy=(row['number'], row['value']),
                         xytext=(10, 10), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', color='green'))

        plt.title("Best Score Progression (Optimization Trail)", fontsize=16)
        plt.xlabel("Trial Number")
        plt.ylabel("Objective Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        p_path = Path("/tmp/optuna_rank_progression.png")
        plt.savefig(p_path)
        plt.close()
        show_image(p_path)
        return
        # Select categorical parameters with most trials
        cat_cols = [c for c in df.columns if c.startswith('params_') and df[c].nunique() < 15 and df[c].nunique() > 1]
        # Sort by those that are present in most trials
        cat_cols = sorted(cat_cols, key=lambda c: df[c].notna().sum(), reverse=True)[:5]
        print(f"Generating box plots for: {cat_cols}")
        
        for col in cat_cols:
            plt.figure(figsize=(16, 9))
            import seaborn as sns
            sns.boxplot(data=df, x=col, y='value')
            plt.title(f"Score Distribution by {col.replace('params_', '')}", fontsize=16)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            p_path = Path(f"/tmp/optuna_box_{col.replace('.', '_')}.png")
            plt.savefig(p_path)
            plt.close()
            show_image(p_path)
        return
    
    tmp_path = Path("/tmp/optuna_plot.png")
    
    # Double the size as requested (24x14 inches)
    plt.figure(figsize=(24, 14))
    
    if args.type == "history":
        plot_optimization_history(study)
    elif args.type == "parallel":
        from optuna.visualization.matplotlib import plot_parallel_coordinate
        plot_parallel_coordinate(study, params=top_params)
        fig = plt.gcf()
        fig.set_size_inches(24, 14)
    elif args.type == "slice":
        from optuna.visualization.matplotlib import plot_slice
        for param in top_params[:5]: # Focus on top 5 to avoid clutter
            print(f"Generating slice plot for: {param}")
            plt.figure(figsize=(16, 9))
            plot_slice(study, params=[param])
            plt.title(f"Slice Plot: {param}", fontsize=16)
            plt.tight_layout()
            
            p_path = Path(f"/tmp/optuna_slice_{param.replace('.', '_')}.png")
            plt.savefig(p_path)
            plt.close()
            show_image(p_path)
        return # Exit since we handled display inside loop
    else:
        from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
        plot_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator())
    
    plt.tight_layout()
    plt.savefig(tmp_path)
    plt.close()
    
    show_image(tmp_path)

if __name__ == "__main__":
    main()

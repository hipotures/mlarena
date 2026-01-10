import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os
import subprocess
from io import StringIO
from pathlib import Path
import json
import glob

def scan_local_experiments(project_name):
    """
    Scans local and NFS experiment directories for state.json files
    to extract model templates and local CV scores.
    """
    print(f"Scanning local and NFS experiments for project: {project_name}...")
    
    # Define paths to scan
    paths = [
        Path(f"projects/kaggle/{project_name}/experiments"),
        Path(f"/mnt/mlarena/projects/kaggle/{project_name}/experiments")
    ]
    
    experiments = {} # template -> best_score
    
    for base_path in paths:
        if not base_path.exists():
            continue
            
        # Use glob to find all state.json files recursively
        # Optimizing: directly look for state.json
        state_files = base_path.rglob("state.json")
        
        for state_file in state_files:
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                
                # Extract template
                template = None
                
                # Try different locations for template
                modules = data.get("modules", {})
                model_mod = modules.get("model", {})
                
                # 1. From model payload
                if not template:
                    template = model_mod.get("payload", {}).get("template")
                
                # 2. From model invocation
                if not template:
                    template = model_mod.get("invocation", {}).get("model_template")
                
                # 3. From top level (sometimes saved directly)
                if not template:
                    template = data.get("template") or data.get("model_template")

                if not template:
                    continue
                    
                # Extract Score
                score = None
                
                # 1. From payload
                score = model_mod.get("payload", {}).get("local_cv_score") or model_mod.get("payload", {}).get("local_cv")
                
                # 2. From module root
                if score is None:
                    score = model_mod.get("local_cv_score") or model_mod.get("local_cv")
                
                if score is None:
                    continue
                    
                try:
                    score = float(score)
                    # For RMSE/LogLoss (negative), take absolute value if needed? 
                    # Assuming consistency, usually stored as negative.
                    # But analyze_correlation expects absolute usually?
                    # Let's keep raw and take abs later if needed or consistent.
                    
                    # Store only the best score found for this template (if duplicates exist)
                    # Usually we want the latest or best. Let's assume best (min/max depending on metric?)
                    # For RMSE (negative), max is better (closer to 0).
                    if template not in experiments:
                        experiments[template] = score
                    else:
                        # Update if "better"? Or just overwrite?
                        # Let's keep the one closer to 0 (assuming negative metric like RMSE/LogLoss)
                        # If positive metric (AUC), larger is better.
                        # It's safer to just overwrite or keep list. Let's keep list.
                        if isinstance(experiments[template], list):
                            experiments[template].append(score)
                        else:
                            experiments[template] = [experiments[template], score]
                            
                except (ValueError, TypeError):
                    continue
                    
            except Exception:
                continue
                
    # Aggregate lists to single value (mean or max/min?)
    # For correlation, we want the representative score.
    # Let's take the one with max absolute value? No, min absolute value (best error).
    final_experiments = {}
    for tpl, val in experiments.items():
        if isinstance(val, list):
            # Heuristic: Take the one closest to 0 (assuming error metric)
            # But wait, if metric is AUC?
            # Let's just take the last one encountered (simplification) or average?
            # Let's take the mean to smooth out noise.
            final_experiments[tpl] = sum(val) / len(val)
        else:
            final_experiments[tpl] = val
            
    return final_experiments

def analyze_fast_vs_full(project_name, prefix=None):
    experiments = scan_local_experiments(project_name)
    
    # Find pairs: X and X_full
    pairs = []
    for tpl, score in experiments.items():
        # Apply prefix filter if provided
        if prefix and not tpl.startswith(prefix):
            continue

        if tpl.endswith("_full"):
            base_tpl = tpl[:-5] # remove _full
            if base_tpl in experiments:
                pairs.append({
                    'template': base_tpl,
                    'fast_cv': experiments[base_tpl],
                    'full_cv': score,
                    'diff': abs(score) - abs(experiments[base_tpl]) # Improvement?
                })
    
    if not pairs:
        print(f"No Fast/Full pairs found matching prefix '{prefix}'." if prefix else "No Fast/Full pairs found.")
        return

    df = pd.DataFrame(pairs)
    # Convert to abs for correlation (assuming error metrics)
    df['fast_cv_abs'] = df['fast_cv'].abs()
    df['full_cv_abs'] = df['full_cv'].abs()
    
    print(f"\nAnalysis Fast vs Full (Count: {len(df)})")
    if prefix:
        print(f"Filter prefix: {prefix}")
    print("-" * 75)
    print(f"{'Template':<30} | {'Fast CV':<10} | {'Full CV':<10} | {'Diff':<10}")
    print("-" * 75)
    for _, row in df.iterrows():
        print(f"{row['template']:<30} | {row['fast_cv_abs']:<10.5f} | {row['full_cv_abs']:<10.5f} | {row['diff']:<10.5f}")
    print("-" * 75)
    
    if len(df) > 1:
        corr = df['fast_cv_abs'].corr(df['full_cv_abs'])
        print(f"Pearson Correlation (Fast vs Full): {corr:.6f}")
        
        # Test_c only filter (if not already filtered by prefix)
        if not prefix or not prefix.startswith('test_c'):
            df_test_c = df[df['template'].str.startswith('test_c')]
            if len(df_test_c) > 1:
                corr_c = df_test_c['fast_cv_abs'].corr(df_test_c['full_cv_abs'])
                print(f"Pearson Correlation (Fast vs Full, test_c_ only): {corr_c:.6f}")


def analyze_correlation(project_name, prefix, show_plot=False):
    print(f"Fetching submissions for competition: {project_name}...")
    
    # Run the kaggle CLI command directly to get the CSV output
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "submissions", project_name, "--csv"],
            capture_output=True,
            text=True,
            check=True
        )
        csv_data = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running kaggle command: {e}")
        print(f"Stderr: {e.stderr}")
        return

    # Load the CSV data into pandas
    try:
        df = pd.read_csv(StringIO(csv_data))
    except Exception as e:
        print(f"Error parsing CSV data: {e}")
        return
    
    data = []
    for _, row in df.iterrows():
        # Check status if column exists
        if 'status' in row and 'complete' not in str(row['status']).lower():
            continue
            
        desc = str(row['description'])
        parts = [p.strip() for p in desc.split('|')]
        
        # Format: CV | feat: X | exp-ID | ModelTemplate | PreprocTemplate | Filename
        if len(parts) < 4:
            continue
            
        try:
            cv_score = abs(float(parts[0]))
            model_template = parts[3]
            
            # Use publicScore from CSV
            if 'publicScore' not in row or pd.isna(row['publicScore']):
                continue
                
            public_score = float(row['publicScore'])
            
            if model_template.startswith(prefix):
                data.append({
                    'template': model_template,
                    'cv': cv_score,
                    'public': public_score
                })
        except (ValueError, IndexError):
            continue

    if not data:
        print(f"No submissions found for prefix: {prefix}")
    else:
        analysis_df = pd.DataFrame(data)
        
        # Sort by CV for better readability
        analysis_df = analysis_df.sort_values('cv')

        print(f"\nAnalysis for prefix: {prefix}")
        print("-" * 60)
        print(f"{ 'Template':<30} | { 'CV (abs)':<10} | { 'Public':<10}")
        print("-" * 60)
        for _, row in analysis_df.iterrows():
            print(f"{row['template']:<30} | {row['cv']:<10.5f} | {row['public']:<10.5f}")
        
        print("-" * 60)
        
        x = analysis_df['cv'].values
        y = analysis_df['public'].values
        
        if len(x) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            correlation = analysis_df['cv'].corr(analysis_df['public'])
            
            print(f"Number of points: {len(x)}")
            print(f"Pearson Correlation: {correlation:.6f}")
            print(f"R-squared: {r_value**2:.6f}")
            print(f"Regression line: Public = {slope:.6f} * CV + {intercept:.6f}")
            
            if r_value**2 > 0.9:
                print("Status: EXCELLENT correlation - CV is a very reliable proxy.")
            elif r_value**2 > 0.7:
                print("Status: GOOD correlation - CV is generally reliable.")
            else:
                print("Status: WEAK correlation - Be careful, LB might be drifting from CV.")

            # Plotting
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.scatter(x, y, alpha=0.7, label='Data points')
                plt.plot(x, slope*x + intercept, color='red', label=f'Regression (R²={r_value**2:.3f})')
                
                for i, txt in enumerate(analysis_df['template']):
                    # Remove prefix and '_full' to get a cleaner label
                    label = txt
                    if label.startswith(prefix):
                        label = label[len(prefix):]
                    if label.endswith('_full'):
                        label = label[:-5]
                    plt.annotate(label, (x[i], y[i]), fontsize=8, alpha=0.8, xytext=(5, 5), textcoords='offset points')

                plt.xlabel('CV Score (abs)')
                plt.ylabel('Public Leaderboard Score')
                plt.title(f'Correlation Analysis: {prefix}')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                
                plot_name = f"/tmp/correlation_{prefix.strip('_')}.png"
                plt.savefig(plot_name)
                print(f"Plot saved as: {plot_name}")

                if show_plot:
                    try:
                        subprocess.run(["kitten", "icat", plot_name])
                    except Exception as e:
                        print(f"Could not display plot using kitty: {e}")
            except Exception as e:
                print(f"Could not generate plot: {e}")
        else:
            print("Not enough points to calculate regression.")

    # Interactive Prompt for Fast vs Full
    print("\n")
    user_input = input("Check correlation Fast vs Full? [y/N]: ")
    if user_input.lower() == 'y':
        analyze_fast_vs_full(project_name, prefix)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlation between Local CV and Public Leaderboard scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples of usage:
  # Analyze test_c_ models and show plot in Kitty terminal
  python3 scripts/analyze_correlation.py -p playground-series-s6e1 --prefix test_c_ --show

  # Analyze test_02_ models without showing the plot
  python3 scripts/analyze_correlation.py -p playground-series-s6e1 --prefix test_02_
        """
    )
    parser.add_argument("-p", "--project", required=True, help="Competition slug (e.g. playground-series-s6e1)")
    parser.add_argument("--prefix", required=True, help="Template prefix to filter (e.g., 'test_c_')")
    parser.add_argument("--show", action="store_true", help="Show plot in terminal using kitty graphics protocol")
    args = parser.parse_args()
    
    # args.project is treated as the competition slug
    analyze_correlation(args.project, args.prefix, show_plot=args.show)

if __name__ == "__main__":
    main()
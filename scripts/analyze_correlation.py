import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os
from pathlib import Path

def analyze_correlation(project_path, prefix, show_plot=False):
    csv_path = Path(project_path) / "my_submissions.csv"
    if not csv_path.exists():
        print(f"Error: File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    data = []
    for _, row in df.iterrows():
        desc = str(row['description'])
        parts = [p.strip() for p in desc.split('|')]
        
        # Format: CV | feat: X | exp-ID | ModelTemplate | PreprocTemplate | Filename
        if len(parts) < 4:
            continue
            
        try:
            cv_score = abs(float(parts[0]))
            model_template = parts[3]
            public_score = float(row['publicScore'])
            
            if pd.isna(public_score):
                continue

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
        return

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
            
            plot_name = f"correlation_{prefix.strip('_')}.png"
            plt.savefig(plot_name)
            print(f"Plot saved as: {plot_name}")
            
            if show_plot:
                import subprocess
                try:
                    subprocess.run(["kitten", "icat", plot_name])
                except Exception as e:
                    print(f"Could not display plot using kitty: {e}")
        except Exception as e:
            print(f"Could not generate plot: {e}")
    else:
        print("Not enough points to calculate regression.")

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

  # Use a different project folder
  python3 scripts/analyze_correlation.py -p titanic --prefix baseline_
        """
    )
    parser.add_argument("-p", "--project", required=True, help="Project name (folder in projects/kaggle/)")
    parser.add_argument("--prefix", required=True, help="Template prefix to filter (e.g., 'test_c_')")
    parser.add_argument("--show", action="store_true", help="Show plot in terminal using kitty graphics protocol")
    args = parser.parse_args()
    
    project_dir = Path("projects/kaggle") / args.project
    analyze_correlation(project_dir, args.prefix, show_plot=args.show)

if __name__ == "__main__":
    main()
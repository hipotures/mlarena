import os
import gzip
import pandas as pd
from collections import Counter
import glob

def analyze():
    base_path = "/mnt/mlarena/projects/kaggle/playground-series-s6e1/experiments/optuna_smoke_s6e1_heavy_v2/"
    pattern = os.path.join(base_path, "trial_*/optuna_model/artifacts/model/leaderboard.csv.gz")
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} leaderboard files.")
    
    top_models = []
    model_times = {} # model_name -> list of fit_times
    
    for f in files:
        try:
            with gzip.open(f, 'rt') as f_in:
                df = pd.read_csv(f_in)
                if not df.empty:
                    # Top 1
                    top_model = df.iloc[0]['model']
                    top_models.append(top_model)
                    
                    # Times for all models in this leaderboard
                    for _, row in df.iterrows():
                        m_name = row['model']
                        m_time = row['fit_time']
                        if m_name not in model_times:
                            model_times[m_name] = []
                        model_times[m_name].append(m_time)
        except Exception as e:
            pass
            
    top_counts = Counter(top_models)
    
    # Calculate averages
    stats = []
    for m_name, times in model_times.items():
        avg_time = sum(times) / len(times)
        stats.append({
            'model': m_name,
            'top1_count': top_counts.get(m_name, 0),
            'avg_fit_time': avg_time
        })
    
    # Sort by top1_count descending
    stats.sort(key=lambda x: x['top1_count'], reverse=True)
    
    print(f"\n{'Algorithm':30} | {'Top 1':>6} | {'Avg Fit Time (s)':>16}")
    print("-" * 60)
    for s in stats:
        print(f"{s['model']:30} | {s['top1_count']:6} | {s['avg_fit_time']:16.2f}")

if __name__ == "__main__":
    analyze()

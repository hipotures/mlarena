
import optuna
import sys

db_url = "sqlite:////mnt/mlarena/projects/kaggle/playground-series-s6e1/experiments/db/mla.db"
study_name = "smoke_s6e1_heavy_v2"

id_a = 35948 # Good (supposedly correlated?)
id_b = 5784 # Bad (good local, bad kaggle)

print(f"Comparing {id_a} vs {id_b}...")
try:
    study = optuna.load_study(study_name=study_name, storage=db_url)
    t_a = study.trials[id_a]
    t_b = study.trials[id_b]

    params_a = t_a.params
    params_b = t_b.params
    
    all_keys = sorted(set(params_a.keys()) | set(params_b.keys()))
    
    print(f"{'PARAM':<50} | {'GOOD (35948)':<20} | {'BAD (5784)':<20}")
    print("-" * 100)
    
    for k in all_keys:
        val_a = params_a.get(k, "N/A")
        val_b = params_b.get(k, "N/A")
        
        # Highlight differences
        prefix = "  "
        if val_a != val_b:
            prefix = ">>"
            
        print(f"{prefix} {k:<47} | {str(val_a):<20} | {str(val_b):<20}")

except Exception as e:
    print(e)

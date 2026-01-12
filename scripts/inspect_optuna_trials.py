import optuna
import sys
import os

db_url = "sqlite:////mnt/mlarena/projects/kaggle/playground-series-s6e1/experiments/db/mla.db"
study_name = "smoke_s6e1_heavy_v2"

# Good Trials (Top score, consistent)
good_trials = [36559, 36588, 35948]

# Bad Trials (High CV, Low LB - suspected leakage)
bad_trials = [5784, 5912]

print(f"Loading study '{study_name}' from {db_url}...")

try:
    study = optuna.load_study(study_name=study_name, storage=db_url)
    
    print("\n--- GOOD TRIALS ---")
    for trial_id in good_trials:
        try:
            trial = study.trials[trial_id]
            print(f"\nTrial {trial_id} (Value: {trial.value}):")
            print(f"  target_transformer.enabled: {trial.params.get('target_transformer.enabled', 'N/A')}")
            print(f"  target_transformer.variant: {trial.params.get('target_transformer.variant', 'N/A')}")
            print(f"  encoder.enabled: {trial.params.get('encoder.enabled', 'N/A')}")
            print(f"  encoder.variant: {trial.params.get('encoder.variant', 'N/A')}")
        except Exception as e:
            print(f"Trial {trial_id}: Error {e}")

    print("\n--- BAD TRIALS ---")
    for trial_id in bad_trials:
        try:
            trial = study.trials[trial_id]
            print(f"\nTrial {trial_id} (Value: {trial.value}):")
            print(f"  target_transformer.enabled: {trial.params.get('target_transformer.enabled', 'N/A')}")
            print(f"  target_transformer.variant: {trial.params.get('target_transformer.variant', 'N/A')}")
            print(f"  encoder.enabled: {trial.params.get('encoder.enabled', 'N/A')}")
            print(f"  encoder.variant: {trial.params.get('encoder.variant', 'N/A')}")
        except Exception as e:
            print(f"Trial {trial_id}: Error {e}")

except Exception as e:
    print(f"Failed to load study: {e}")

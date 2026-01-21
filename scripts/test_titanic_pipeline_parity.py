import subprocess
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time
import shutil
import re
import os

PROJECT = "titanic"
# Templates that are definitely chains
TEMPLATES = ["scaler_test_chain", "test_imputer_chain"]
PROJECT_DIR = Path(f"projects/kaggle/{PROJECT}")

def run_mla(template, classic):
    cmd = [
        "uv", "run", "python", "scripts/mla.py", "preprocess",
        f"project={PROJECT}",
        f"preprocess_template={template}",
        f"classic={str(classic).lower()}",
        "force=true",
        "quiet_preprocess_panel=true"
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout

def find_output_classic(template):
    # Classic mode: Output is in experiments/pre-<template>/<hash>/<step_name>/artifacts/preprocess/
    # We want the LAST step.
    base = PROJECT_DIR / "experiments" / f"pre-{template}"
    if not base.exists(): return None
    
    # Get latest hash
    hashes = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
    if not hashes: return None
    latest_hash_dir = hashes[0]
    
    # Get all steps in this hash dir
    steps = sorted([d for d in latest_hash_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    # Sort by index prefix (0-..., 1-...)
    # Filter out 'artifacts' or other dirs that might be there if pipeline mode ran partially
    valid_steps = []
    for s in steps:
        if s.name == "artifacts": continue
        parts = s.name.split('-')
        if parts[0].isdigit():
            valid_steps.append(s)
            
    valid_steps.sort(key=lambda x: int(x.name.split('-')[0]))
    
    if not valid_steps: return None
    last_step = valid_steps[-1]
    
    return last_step / "artifacts/preprocess/train_processed.csv.gz"

def find_output_pipeline(template):
    # Pipeline mode: Output is ALSO in the last step directory (for compatibility/structure)
    # So we use the same directory finding logic as classic.
    
    base = PROJECT_DIR / "experiments" / f"pre-{template}"
    if not base.exists(): return None
    
    # Get latest hash
    hashes = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
    if not hashes: return None
    latest_hash_dir = hashes[0]
    
    # Get all steps in this hash dir
    steps = sorted([d for d in latest_hash_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
    
    valid_steps = []
    for s in steps:
        if s.name == "artifacts": continue
        parts = s.name.split('-')
        if parts[0].isdigit():
            valid_steps.append(s)
            
    valid_steps.sort(key=lambda x: int(x.name.split('-')[0]))
    
    if not valid_steps: return None
    last_step = valid_steps[-1]
    
    # Check for parquet
    p_path = last_step / "artifacts/preprocess/train_processed.parquet"
    if p_path.exists(): return p_path
    
    # Check for csv.gz (fallback)
    c_path = last_step / "artifacts/preprocess/train_processed.csv.gz"
    if c_path.exists(): return c_path
    
    return None

def compare_files(csv_path, parquet_path):
    print(f"Comparing:\n  Classic: {csv_path}\n  Pipeline: {parquet_path}")
    
    df_c = pd.read_csv(csv_path)
    if "Unnamed: 0" in df_c.columns:
        df_c = df_c.drop(columns=["Unnamed: 0"])
        
    df_p = pd.read_parquet(parquet_path)
    
    # Align columns
    cols = sorted(list(set(df_c.columns) & set(df_p.columns)))
    df_c = df_c[cols]
    df_p = df_p[cols]
    
    # Convert to compatible types
    for c in cols:
        try:
            df_c[c] = pd.to_numeric(df_c[c])
            df_p[c] = pd.to_numeric(df_p[c])
        except:
            pass
            
    try:
        pd.testing.assert_frame_equal(df_c, df_p, check_dtype=False, atol=1e-4, check_index_type=False)
        return True, "Match"
    except AssertionError as e:
        return False, str(e)

def main():
    results = {}
    
    for t in TEMPLATES:
        print(f"\n\n{'='*20} Testing Template: {t} {'='*20}")
        try:
            # 1. Run Classic
            print(f"--> Running Classic Mode")
            run_mla(t, classic=True)
            out_classic = find_output_classic(t)
            
            if not out_classic or not out_classic.exists():
                results[t] = f"FAIL: Classic output missing: {out_classic}"
                continue
                
            # Copy to temp
            tmp_classic = Path(f"/tmp/{t}_classic.csv.gz")
            shutil.copy(out_classic, tmp_classic)
            print(f"Backed up classic output to {tmp_classic}")
            
            # 2. Run Pipeline
            print(f"--> Running Pipeline Mode")
            run_mla(t, classic=False)
            out_pipeline = find_output_pipeline(t)
            
            if not out_pipeline or not out_pipeline.exists():
                results[t] = f"FAIL: Pipeline output missing: {out_pipeline}"
                continue
                
            # 3. Compare
            match, msg = compare_files(tmp_classic, out_pipeline)
            results[t] = "PASS" if match else f"FAIL: {msg[:200]}..."
            
        except Exception as e:
            results[t] = f"ERROR: {e}"
            import traceback
            traceback.print_exc()

    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    for t, res in results.items():
        print(f"{t:<20}: {res}")

if __name__ == "__main__":
    main()
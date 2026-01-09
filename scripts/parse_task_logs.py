import re
import os
from pathlib import Path

def parse_logs():
    log_dir = Path("projects/kaggle/playground-series-s6e1/queue/logs")
    results = []
    
    for i in range(66, 80):
        log_file = log_dir / f"task-{i}.log"
        if not log_file.exists():
            continue
            
        content = log_file.read_text()
        
        # Extract variant/template
        v_match = re.search(r"test_c_01_0306_v\d{2}", content)
        template = v_match.group(0) if v_match else "unknown"
        
        # Extract Method and N Features
        method_match = re.search(r"Selection Method:\s+([^\s]+)", content)
        method = method_match.group(1) if method_match else "?"
        
        n_feat_match = re.search(r"N Features:\s+([^\s]+)", content)
        if not n_feat_match:
            n_feat_match = re.search(r"Importance Cumulative:\s+([^\s]+)", content)
        n_feat = n_feat_match.group(1) if n_feat_match else "-"
        
        # Robust extraction from Preprocess Completed blocks
        # We need the one BEFORE selector (engineer) and the selector one.
        
        before = "?"
        after = "?"
        
        # Find all shape summaries
        # 📄 train_processed.csv.gz (378,000 × 53)
        all_shapes = re.findall(r"train_processed\.csv\.gz \(\d+,000 × (\d+)\)", content)
        
        if len(all_shapes) >= 2:
            # The selector is usually the last step (index -1)
            # The one before it is feature_engineer (index -2)
            after = all_shapes[-1]
            before = all_shapes[-2]

        # Duration
        dur_match = re.search(r"Template: .*?feature_selector.*?Duration: (.*?)\n", content, re.DOTALL)
        sel_duration = dur_match.group(1).strip() if dur_match else "unknown"
        
        # Score
        cv_match = re.search(r"Local CV:\s+([-]?\d+\.\d+)", content)
        score = cv_match.group(1) if cv_match else "N/A"
        
        results.append({
            "task": i,
            "template": template,
            "method": method,
            "n_feat": n_feat,
            "before": before,
            "after": after,
            "duration": sel_duration,
            "score": score
        })

    # Print Table
    header = f"{ 'Task':<5} | { 'Template':<20} | { 'Method':<18} | { 'N':<5} | { 'Bef':<4} | { 'Aft':<4} | { 'Dur':<8} | { 'Score'}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['task']:<5} | {r['template']:<20} | {r['method']:<18} | {r['n_feat']:<5} | {r['before']:<4} | {r['after']:<4} | {r['duration']:<8} | {r['score']}")

if __name__ == "__main__":
    parse_logs()

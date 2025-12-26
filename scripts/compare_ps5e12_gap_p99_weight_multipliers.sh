#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import json, glob, os
root = "projects/kaggle/playground-series-s5e12/experiments"
rows=[]
for path in glob.glob(os.path.join(root, "exp-*/state.json")):
    with open(path) as f:
        data=json.load(f)
    model = data.get("modules", {}).get("model", {})
    inv = model.get("invocation", {})
    model_template = inv.get("model_template")
    preprocess_template = inv.get("preprocess_template")
    if model_template != "catboost-binned-hpo-gap":
        continue
    if not (preprocess_template or "").startswith("catboost-full-clean-rfe-gap-p99-wm"):
        continue
    local = model.get("payload", {}).get("local_cv_score")
    if local is None:
        continue
    rows.append((preprocess_template, float(local), data.get("experiment_id")))
rows.sort(key=lambda r: r[0])
print("preprocess_chain\tlocal_cv\texp_id")
for tpl, local, exp_id in rows:
    print(f"{tpl}\t{local:.5f}\t{exp_id}")
PY

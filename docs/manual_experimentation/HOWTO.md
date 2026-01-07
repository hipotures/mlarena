# Manual Experimentation HOWTO

This guide describes how to manually clone, modify, and queue experiments in the `mlarena` framework, specifically tailored for the local `~/ml/kaggle` environment while respecting the NFS `/mnt/mlarena` execution context.

## 1. Directory Structure & Philosophy

*   **`~/ml/kaggle` (Local)**: This is your **workspace**. You create templates, edit code, and manage the queue here.
*   **`/mnt/mlarena` (NFS)**: This is the **execution environment**. Heavy computations often run here. Experiment results (`state.json`, artifacts) are physically stored here.

**Key Rule**: You EDIT in Local, but you CHECK RESULTS in NFS.

## 2. Cloning an Experiment (The `test_c_02_0012` Pattern)

When you have a successful experiment (e.g., `test_c_02_0047`) and want to test variations, follow this "Variant Pattern":

### Step 1: Identify Source Templates
Find the base templates for the experiment you want to clone:
*   Model: `projects/<proj>/templates/model/BASE_NAME.yaml`
*   Preprocess: `projects/<proj>/templates/preprocess/BASE_NAME.yaml` (Chain)
*   Sub-modules: `projects/<proj>/templates/preprocess/BASE_NAME-*.yaml`

### Step 2: Create Variants (01, 02, ...)
Create new template files for each variant using a suffix (e.g., `_01`, `_02`).

**CRITICAL**: You must isolate **every** file for the variant to avoid side effects.

1.  **Duplicate Sub-modules**: Copy the specific preprocessing step you want to change (e.g., `feature_engineer`) AND all other steps in the chain, renaming them with the suffix.
    *   `BASE_NAME_01-feature_engineer.yaml` (Modified)
    *   `BASE_NAME_01-scaler.yaml` (Copy of base, or modified)
    *   `...`

2.  **Create Chain File**: Create `templates/preprocess/BASE_NAME_01.yaml`.
    *   Update the `chain:` list to point to your **newly created suffix files** (`..._01-scaler`, etc.).

3.  **Create Model File**: Create `templates/model/BASE_NAME_01.yaml`.
    *   Set `preprocess_template: BASE_NAME_01`.
    *   Keep or modify model params.

### Example Naming Convention
If Base is `test_c_02_0047`:
*   **Variant 01**:
    *   Model: `templates/model/test_c_02_0047_01.yaml`
    *   Preprocess Chain: `templates/preprocess/test_c_02_0047_01.yaml`
    *   Step: `templates/preprocess/test_c_02_0047_01-feature_engineer.yaml`

## 3. Modifying Parameters

Open your new suffix files (e.g., `..._01-feature_engineer.yaml`) and change specific parameters in the `config:` section.

**Common Tuning Targets:**
*   **Feature Engineer**: `max_generated_features`, `interaction_types` (add, mul, sub, div), `poly_degree`.
*   **Outlier Handler**: `outlier_method` (isolation_forest vs zscore), `contamination`.
*   **Scaler**: `scaling_method` (standard, quantile_normal, quantile_uniform).
*   **Rare Category**: `min_freq`, `top_k`.

## 4. Adding to Queue

Use `scripts/task_queue.py` to schedule your manual experiments. Use the explicit `--command` mode for full control.

**Command Syntax:**
```bash
python scripts/task_queue.py --project <PROJECT_NAME> add --command "model --model-template <VARIANT_NAME> --exp-id <VARIANT_NAME> skip_submit=true skip_git=true model.mla_retention=true"
```

*   `--model-template`: Your new variant model file (e.g., `test_c_02_0047_01`).
*   `--exp-id`: Explicitly name the experiment ID same as the template for clarity.
*   `skip_submit=true`: Don't submit to Kaggle yet.
*   `skip_git=true`: Don't auto-commit (useful for quick iteration).
*   `model.mla_retention=true`: Keeps artifacts for analysis (important for manual checks).

**Batch Example:**
```bash
for i in 01 02 03; do
  python scripts/task_queue.py -p playground-series-s6e1 add --command "model --model-template test_c_02_0047_${i} --exp-id test_c_02_0047_${i} skip_submit=true skip_git=true model.mla_retention=true"
done
```

# MetaML: Predictive Modeling for MCTS Actions

## 1. Goal
The primary objective is to build a Machine Learning model that predicts the probability of an MCTS Action (preprocessing step) improving the target metric. Specifically, we want to predict whether transition $A \to B$ (Action $A_2$) results in a score gain or loss.

## 2. Data Source
*   **Database**: `projects/kaggle/playground-series-s6e1/experiments/db/mcts.db`
*   **Key Tables**:
    *   `mcts_edges`: Contains the `action_json` defining the transition.
    *   `mcts_nodes`: Defines the state (Parent/Child) and topology (Depth).
    *   `mcts_evaluations`: Provides the ground truth scores ($Score_{parent}$, $Score_{child}$). 

## 3. Feature Engineering Strategy

### 3.1. Core Features (Context)
These features describe "where we are" and "what we are doing".

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `parent_score` | Float | The evaluation score of the parent node. |
| `parent_depth` | Int | Current depth in the MCTS tree. |
| `prev_action_group` | Categorical | The group of the action that led to the parent node (Context). |
| `prev_action_variant`| Categorical | The variant of the previous action. |
| `action_group` | Categorical | The group of the candidate action (e.g., `scaler`, `imputer`). |
| `action_variant` | Categorical | The specific variant of the candidate action (e.g., `rank_gauss`, `mean`). |

### 3.2. Parameter Handling: Sparse Flattening
The biggest challenge is the heterogeneity of parameters (e.g., `imputer` has `strategy`, while `binner` has `n_bins`).

**Solution**: **Sparse Flattening with Group Prefixing**
We will create a global "super-space" of all possible parameters. Columns that are irrelevant for a specific action group will be filled with `NaN`.

*   **Naming Convention**: `{group_name}_{param_name}`
*   **Why**: This prevents name collisions (e.g., `strategy` exists in multiple groups) and allows Tree-based models (CatBoost/XGBoost) to naturally handle the conditional logic (Split: Is `action_group` == `binner`? If yes, check `binner_n_bins`).

**Example Feature Vector:**
```json
{
  "action_group": "numeric_binner",
  "action_variant": "quantile_onehot",
  "imputer_numeric_strategy": NaN,   // Inactive
  "binner_n_bins": 30,               // Active
  "binner_strategy": "quantile",     // Active
  "outlier_threshold": NaN           // Inactive
}
```

### 3.3. Target Variable
*   **Regression Target**: `delta_score = child_score - parent_score`
*   **Classification Target**: `is_positive = 1 if delta_score > 0 else 0`

## 4. Modeling Approach
1.  **Framework**: Use `mlarena` infrastructure.
2.  **Algorithm**: **CatBoost** or **XGBoost**.
    *   *Rationale*: Best-in-class handling of Categorical features and Missing Values (`NaN`). They can effectively learn the conditional structure of the Sparse Flattened parameters.
3.  **Validation**: Time-based or ID-based split (train on older trials, test on newer) to simulate the MCTS process.

## 5. Implementation Steps
1.  **Data Extraction**: Create a script to query `mcts.db`, join tables, and flatten `action_json`.
2.  **Dataset Construction**: Generate a training dataset (`train.csv` / `train.parquet`).
3.  **Project Setup**: Initialize a new experiment in `mlarena` (or use `playground-series-s6e1`) to train the Meta-Model.
4.  **Analysis**: Analyze feature importance to understand which parameters drive performance improvements.

# Specification: MCTS Preprocessing Search

## Overview
Implement **Monte Carlo Tree Search (MCTS)** as an alternative search strategy to Optuna for the `mla pre tune` command. The goal is to intelligently navigate the preprocessing search space using a tree-based approach with progressive widening, while maintaining full compatibility with the existing MLA pipeline and enabling persistent, resumable execution via SQLite.

## Functional Requirements

### 1. CLI Integration
- Add a `--mcts` flag to `mla pre tune`.
- Ensure that when `--mcts` is present, Optuna is NOT initialized.
- Instead, instantiate and run `MCTSRunner`.

### 2. MCTS Engine & Logic
- **Algorithm:** Implement MCTS with UCT/PUCT selection policies.
- **Progressive Widening:** Implement dynamic branching control ($m(n) = k \cdot N(n)^\alpha$) to handle large search spaces.
- **Action Space:** 
    - Reuse logic from `mla_super_chain.yaml` and search spaces to generate valid "next steps".
    - Enforce EDA gating, "heavy" step filtering, and problem type compatibility.
- **State Representation:** 
    - Canonical `pipeline_signature` for deduplication.
    - Path in the tree represents the sequence of preprocessing steps.

### 3. Execution Model
- **MLA-Native:** MCTS does not execute code itself. It:
    1. Selects a pipeline config.
    2. Materializes it to standard MLA YAML templates.
    3. Invokes `mla.py model ...` as a subprocess (or TaskQueue task).
- **Baseline (Model Zero):** Always evaluate the empty/baseline pipeline (depth 0) before starting expansion.
- **Multi-fidelity:** Support F0 (fast), F1 (medium), and F2 (full) fidelity levels with promotion strategies (ASHA/Successive Halving).

### 4. Persistence & Storage (SQLite)
- **Single Source of Truth:** Use a local SQLite database (`mcts.db`) for all state (tree, trials, params, results).
- **Optuna-like Schema:** Use a schema compatible with Optuna tools (studies, trials, trial_params, trial_values) to allow reuse of visualization tools.
- **Resumability:** 
    - Allow resuming a study if the configuration fingerprint matches.
    - Prevent resuming if critical config (super-chain, objective) has changed.
- **Queryability:** Support "live" queries for the current best score during execution.

### 5. Template Management
- **Materialization:** Convert MCTS node states into physical YAML files for execution.
- **Retention Policy:** Automatically clean up templates for non-best trials to save disk space (configurable: keep best, keep top-k).
- **Rehydration:** Ability to reconstruct a YAML template from the database state if needed for replay.

### 6. Logging & Traceability
- **Structured Logs:** Emit INFO/DEBUG logs with correlation IDs (`trial_id`, `node_id`, `experiment_id`).
- **JSON Transport:** Parse MLA execution results from stdout (`--json-output`) to populate the database.

## Non-Functional Requirements
- **Performance:** MCTS overhead should be negligible compared to model training time.
- **Modularity:** The implementation must be broken down into small, testable components (Config, Loader, Node, Tree, Runner).
- **Stability:** The system must handle subprocess failures gracefully (mark trial as FAIL, do not crash the study).

## Out of Scope
- Rewriting the core MLA pipeline (init/eda/model).
- Implementing a generic MCTS library (focus is strictly on preprocessing tuning).

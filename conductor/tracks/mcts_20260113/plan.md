# Plan: MCTS Preprocessing Search Implementation

## Phase 1: Foundation & Configuration [checkpoint: 96a0ca4]
- [x] Task: Create `MCTSRunner` Stub & CLI Routing [bbb7c35]
    - [x] Sub-task: Create `src/mlarena/modules/mcts/runner.py` with a basic `MCTSRunner` class.
    - [x] Sub-task: Update `src/mlarena/modules/preprocess_tune.py` to handle the `--mcts` flag and instantiate `MCTSRunner`.
    - [x] Sub-task: Write unit test to verify CLI routing.
- [x] Task: Implement MCTS Configuration System [d2e9025]
    - [x] Sub-task: Define `MCTSConfig` Pydantic model in `src/mlarena/modules/mcts/config.py`.
    - [x] Sub-task: Implement loading logic from `mla_super_chain.yaml` (parsing the `mcts:` section).
    - [x] Sub-task: Write unit tests for config validation and defaults.
- [x] Task: Implement Super-Chain & Search Space Loaders [45e5a0f]
    - [x] Sub-task: Create `src/mlarena/modules/mcts/space.py`.
    - [x] Sub-task: Implement logic to load and parse `mla_super_chain.yaml` (preserving order).
    - [x] Sub-task: Refactor/reuse `_load_search_spaces` from `preprocess_tune.py` to be shared.
    - [x] Sub-task: Write unit tests verifying correct loading of chains and spaces.

## Phase 2: Core MCTS Logic (No Execution) [checkpoint: ca699eb]
- [x] Task: Define State Representation & Action Generation [3e668bb]
    - [x] Sub-task: Create `src/mlarena/modules/mcts/node.py` (PipelineState definition).
    - [x] Sub-task: Implement `pipeline_signature` generation (canonical hash).
    - [x] Sub-task: Implement `next_actions(state)` logic in `space.py` (respecting gating and groups).
    - [x] Sub-task: Write unit tests for action generation and signature stability.
- [x] Task: Implement Parameter Sampler [1e1d298]
    - [x] Sub-task: Implement RNG-based parameter sampler in `src/mlarena/modules/mcts/sampler.py` (supporting choice, int/float ranges).
    - [x] Sub-task: Write unit tests verifying sampled values are within bounds.
- [x] Task: Implement Tree Search & Selection Policies [ca906e0]
    - [x] Sub-task: Implement UCT and PUCT selection logic.
    - [x] Sub-task: Implement **Progressive Widening** logic in the expansion phase.
    - [x] Sub-task: Implement Backpropagation logic.
    - [x] Sub-task: Write unit tests using a `FakeExecutor` to verify tree growth and stats updates.

## Phase 3: Persistence & Execution Infrastructure [checkpoint: e13be14]
- [x] Task: Implement SQLite Storage Layer [5582c25]
    - [x] Sub-task: Create `src/mlarena/modules/mcts/storage.py`.
    - [x] Sub-task: Implement DDL for tables (`studies`, `trials`, `trial_params`, `mcts_nodes`, etc.).
    - [x] Sub-task: Implement methods for `create_study`, `add_trial`, `update_trial`.
    - [x] Sub-task: Write integration tests for database operations.
- [x] Task: Implement Template Materialization [e95f464]
    - [x] Sub-task: Create `src/mlarena/modules/mcts/materializer.py`.
    - [x] Sub-task: Implement logic to convert `PipelineState` to YAML templates (chain + steps).
    - [x] Sub-task: Implement logic for "Ephemeral Templates" (deletion policy).
    - [x] Sub-task: Write unit tests verifying YAML content against state.
- [x] Task: Implement `ExperimentExecutor` & CLI Wrapper [cbf10c2]
    - [x] Sub-task: Create `src/mlarena/modules/mcts/executor.py` and `MlaCliExecutor`.
    - [x] Sub-task: Implement subprocess command builder (`mla.py model ...`).
    - [x] Sub-task: Implement stdout JSON parser to capture results.
    - [x] Sub-task: Write unit tests for command generation and result parsing.

## Phase 4: Integration & Refinement [checkpoint: 3327643]
- [x] Task: End-to-End Integration [9c47227]
    - [x] Sub-task: Connect `MCTSRunner` to `Storage`, `Materializer`, and `Executor`.
    - [x] Sub-task: Implement the main optimization loop (Selection -> Expansion -> Execution -> Backprop).
    - [x] Sub-task: Implement Baseline (Model Zero) evaluation logic at start.
- [x] Task: Implement Multi-Fidelity & Pruning [790d265]
    - [x] Sub-task: Add logic for F0/F1/F2 levels in `MCTSRunner`.
    - [x] Sub-task: Implement ASHA/Successive Halving promotion logic.
    - [x] Sub-task: Write tests verifying that poor performers are pruned.
- [x] Task: Final Polish & Reporting [ba08b99]
    - [x] Sub-task: Implement "New Best Score" reporting/logging.
    - [x] Sub-task: Add final top-K export and cleanup logic.
    - [x] Sub-task: Verify `mla pre tune --mcts` works on a small subset of the Titanic dataset.
    - [x] Task: Conductor - User Manual Verification 'Phase 4: Integration & Refinement' (Protocol in workflow.md)
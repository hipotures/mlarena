# Plan: MCTS Preprocessing Search Implementation

## Phase 1: Foundation & Configuration [checkpoint: 96a0ca4]
- [x] Task: Create `MCTSRunner` Stub & CLI Routing [bbb7c35]
    - [ ] Sub-task: Create `src/mlarena/modules/mcts/runner.py` with a basic `MCTSRunner` class.
    - [ ] Sub-task: Update `src/mlarena/modules/preprocess_tune.py` to handle the `--mcts` flag and instantiate `MCTSRunner`.
    - [ ] Sub-task: Write unit test to verify CLI routing.
- [x] Task: Implement MCTS Configuration System [d2e9025]
    - [ ] Sub-task: Define `MCTSConfig` Pydantic model in `src/mlarena/modules/mcts/config.py`.
    - [ ] Sub-task: Implement loading logic from `mla_super_chain.yaml` (parsing the `mcts:` section).
    - [ ] Sub-task: Write unit tests for config validation and defaults.
- [x] Task: Implement Super-Chain & Search Space Loaders [45e5a0f]
    - [ ] Sub-task: Create `src/mlarena/modules/mcts/space.py`.
    - [ ] Sub-task: Implement logic to load and parse `mla_super_chain.yaml` (preserving order).
    - [ ] Sub-task: Refactor/reuse `_load_search_spaces` from `preprocess_tune.py` to be shared.
    - [ ] Sub-task: Write unit tests verifying correct loading of chains and spaces.

## Phase 2: Core MCTS Logic (No Execution)
- [x] Task: Define State Representation & Action Generation [3e668bb]
    - [ ] Sub-task: Create `src/mlarena/modules/mcts/node.py` (PipelineState definition).
    - [ ] Sub-task: Implement `pipeline_signature` generation (canonical hash).
    - [ ] Sub-task: Implement `next_actions(state)` logic in `space.py` (respecting gating and groups).
    - [ ] Sub-task: Write unit tests for action generation and signature stability.
- [ ] Task: Implement Parameter Sampler
    - [ ] Sub-task: Implement RNG-based parameter sampler in `src/mlarena/modules/mcts/sampler.py` (supporting choice, int/float ranges).
    - [ ] Sub-task: Write unit tests verifying sampled values are within bounds.
- [ ] Task: Implement Tree Search & Selection Policies
    - [ ] Sub-task: Implement UCT and PUCT selection logic.
    - [ ] Sub-task: Implement **Progressive Widening** logic in the expansion phase.
    - [ ] Sub-task: Implement Backpropagation logic.
    - [ ] Sub-task: Write unit tests using a `FakeExecutor` to verify tree growth and stats updates.

## Phase 3: Persistence & Execution Infrastructure
- [ ] Task: Implement SQLite Storage Layer
    - [ ] Sub-task: Create `src/mlarena/modules/mcts/storage.py`.
    - [ ] Sub-task: Implement DDL for tables (`studies`, `trials`, `trial_params`, `mcts_nodes`, etc.).
    - [ ] Sub-task: Implement methods for `create_study`, `add_trial`, `update_trial`.
    - [ ] Sub-task: Write integration tests for database operations.
- [ ] Task: Implement Template Materialization
    - [ ] Sub-task: Create `src/mlarena/modules/mcts/materializer.py`.
    - [ ] Sub-task: Implement logic to convert `PipelineState` to YAML templates (chain + steps).
    - [ ] Sub-task: Implement logic for "Ephemeral Templates" (deletion policy).
    - [ ] Sub-task: Write unit tests verifying YAML content against state.
- [ ] Task: Implement `ExperimentExecutor` & CLI Wrapper
    - [ ] Sub-task: Create `src/mlarena/modules/mcts/executor.py` and `MlaCliExecutor`.
    - [ ] Sub-task: Implement subprocess command builder (`mla.py model ...`).
    - [ ] Sub-task: Implement stdout JSON parser to capture results.
    - [ ] Sub-task: Write unit tests for command generation and result parsing.

## Phase 4: Integration & Refinement
- [ ] Task: End-to-End Integration
    - [ ] Sub-task: Connect `MCTSRunner` to `Storage`, `Materializer`, and `Executor`.
    - [ ] Sub-task: Implement the main optimization loop (Selection -> Expansion -> Execution -> Backprop).
    - [ ] Sub-task: Implement Baseline (Model Zero) evaluation logic at start.
- [ ] Task: Implement Multi-Fidelity & Pruning
    - [ ] Sub-task: Add logic for F0/F1/F2 levels in `MCTSRunner`.
    - [ ] Sub-task: Implement ASHA/Successive Halving promotion logic.
    - [ ] Sub-task: Write tests verifying that poor performers are pruned.
- [ ] Task: Final Polish & Reporting
    - [ ] Sub-task: Implement "New Best Score" reporting/logging.
    - [ ] Sub-task: Add final top-K export and cleanup logic.
    - [ ] Sub-task: Verify `mla pre tune --mcts` works on a small subset of the Titanic dataset.
    - [ ] Task: Conductor - User Manual Verification 'Phase 4: Integration & Refinement' (Protocol in workflow.md)

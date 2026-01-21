# TODO

## Refactoring & Cleanup
- [ ] **Stack Module**: Refactor `src/mlarena/modules/stack.py` to be more user-friendly via CLI. currently requires manual passing of file lists.
- [ ] **Tune Module**: Complete implementation of `src/mlarena/modules/tune.py`. Ensure integration with full AutoGluon HPO capabilities and clarify usage vs native HPO.
- [ ] **Docs**: Unify "Task Queue" vs "Submission Queue" terminology across all guides if any remnants remain.

## MCTS & Oracle (Active Learning)
- [ ] **Stochastic Top-K Strategy**: Add "4 Best + 1 Random from Tail" strategy to Oracle to challenge its internal model and avoid local optima.
- [ ] **Value Function (Future Potential)**: Train Oracle on `value_best` (the best score ever reached from a branch) instead of immediate `delta_score`. This helps identify steps that are good long-term but neutral short-term.
- [ ] **Automated Retraining Loop**: Integrate `scripts/mcts_oracle.py` into the MCTS lifecycle (e.g., auto-trigger retraining every 500 trials).
- [ ] **N-step History for Oracle**: Expand Oracle features to include more than 1 previous step (e.g., full path encoding or sliding window of N actions) to distinguish identical sub-sequences in different contexts.
- [ ] **Oracle Confidence Scoring**: Implement uncertainty estimation for Oracle predictions to prioritize exploration where the model is least certain.

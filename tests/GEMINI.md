# Testing Memory & Infinite Loop Safety Guide

## ⚠️ The "MagicMock Infinite Chain" Problem

During the development of MCTS (Monte Carlo Tree Search), we encountered a critical issue where test execution would suddenly consume massive amounts of RAM (40GB+) and hang the system.

### Root Cause
The issue occurs when testing code that traverses a parent-child relationship (like a tree) using `MagicMock` objects.

```python
# Production Code
curr = node
while curr is not None:
    # do something
    curr = curr.parent
```

In a test environment:
1. `node` is a `MagicMock()`.
2. By default, `MagicMock` returns another `MagicMock` for any attribute access.
3. Therefore, `curr.parent` is a new `MagicMock`, which is **never `None`**.
4. The `while` loop runs forever, and the `MagicMock` accumulates call history in memory, leading to exponential RAM growth.

---

## 🛠️ How to Prevent (Test Design)

1. **Close the Chain**: Always explicitly set the end of your mock chain to `None`.
   ```python
   root_node = MagicMock()
   root_node.parent = None # CRITICAL
   
   child_node = MagicMock()
   child_node.parent = root_node
   ```

2. **Return Primitives for IDs**: Ensure mocks return `int` or `str` for ID fields, as mocks are always "truthy".
   ```python
   mock_storage.create_trial.return_value = 123 # Not a mock
   ```

3. **Use Specs**: Use `spec` or `spec_set` to prevent mocks from having attributes they shouldn't.
   ```python
   node = MagicMock(spec=MCTSNode)
   ```

4. **Patch Recursion**: If the recursion isn't the focus of your test, patch the traversal method entirely.
   ```python
   with patch.object(runner, "_persist_node_stats_path", return_value=None):
       runner.run()
   ```

---

## 🛡️ Defensive Coding (Production)

To protect the system from infinite loops (whether from bugs or tests), recursive traversals should include safety guards:

1. **Cycle Detection**:
   ```python
   seen = set()
   curr = node
   while curr is not None:
       if id(curr) in seen:
           logger.error("Cycle detected!")
           break
       seen.add(id(curr))
       curr = curr.parent
   ```

3.  **Limit Logging Side-Effects**: Avoid `json.dumps()` on objects that might be mocks in f-strings.
   ```python
   if logger.isEnabledFor(logging.DEBUG):
       try:
           logger.debug(f"State: {json.dumps(node.state)}")
       except TypeError:
           logger.debug(f"State (non-serializable): {node.state}")
   ```

---

## 🚀 MCTS Full-Test Verification

To verify the complete integration (CLI -> Runner -> Storage -> Materializer -> Executor) on a real project, use the Titanic dataset:

```bash
uv run python scripts/mla.py preprocess-tune project=titanic mcts.enabled=true mcts.study_name=verify_manual_mcts mcts.budget=2
```

**What to check:**
- Tree visualization appears in console (or log).
- Trials are successfully created in `projects/kaggle/titanic/experiments/db/mcts.db`.
- No "MagicMock" or "Infinite Loop" errors occur in logs.

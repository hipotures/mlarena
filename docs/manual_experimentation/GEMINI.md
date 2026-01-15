# Manual Experimentation Memory

## Workflow Summary
1.  **Context**: Work in `~/ml/kaggle`, results in `/mnt/mlarena`.
2.  **Pattern**: Use `BASE_NAME_XX` (01, 02...) suffix for ALL files.
3.  **Isolation**: Duplicate ALL chain steps (even unchanged ones) to `_XX` files to ensure variant isolation.
4.  **Queue**: Use `scripts/task_queue.py add --command "..."`.

## Queue Command Template
```bash
python scripts/task_queue.py -p <PROJECT> add --command "model model_template=<NAME_XX> experiment_id=<NAME_XX> skip_submit=true skip_git=true model.mla_retention=true"
```

## Critical Checks
*   [ ] Did you create the specific `_XX` sub-module file?
*   [ ] Did you update the `chain` list in the `_XX` preprocess file to point to `_XX` sub-modules?
*   [ ] Did you point the model template to the `_XX` preprocess file?

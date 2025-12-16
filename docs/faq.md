# FAQ

- **“Module already completed” appears — what now?** Use `--force` on that module (or the auto-flow) to re-run and overwrite the cached result.
- **Kaggle API errors about credentials.** Ensure `~/.kaggle/kaggle.json` exists with mode `600` and that the Kaggle CLI is installed in the current environment.
- **No such preprocess/model template.** List available options with `uv run python scripts/mla.py modules` (for modules) or `uv run python scripts/mla.py model --model-template list --project <slug>` (for model templates); verify the filename (without `.yaml`) matches the template name.
- **Ambiguous preprocessing/model (project vs global).** Only keep one file per name. If both exist locally and in `src/mlarena/defaults`, rename or remove one to avoid ambiguity errors.
- **Fetch-score fails or hangs.** Start Chrome with remote debugging (`google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug`), log into Kaggle there, and pass `--cdp-url` if using a non-default port/host.
- **How do I rerun predict/submit on an existing experiment?** Pass `--experiment-id <id>` to `predict`, `submit`, or `fetch-score`; the pipeline will reuse artifacts and skip dependencies automatically.
- **State shows “running” after a crash.** Re-run the same module; stale “running” entries are marked failed automatically before execution starts.
- **Where are processed files saved?** Under `projects/kaggle/<slug>/experiments/<id>/artifacts/`, with preprocessing artifacts nested under `preprocess/`.
- **Can I run just preprocessing chains?** Yes: `uv run python scripts/mla.py preprocess --project <slug> --preprocess-template full-pipeline` (or a comma-separated chain). Completed steps are skipped unless `--force` is used.
- **How do I add a custom preprocessing step?** Create `projects/kaggle/<slug>/code/preprocessing/<name>.py` implementing `fit_transform`, then reference it from a project template in `templates/preprocess/<name>.yaml`.
- **Git commit fails at the end of auto-flow.** Check for staged changes limited to the project directory; you can commit manually or rerun with `--skip-git`.
- **Why is my submission rejected for bad format?** Ensure the submission columns match `sample_submission.csv` exactly; the `submit` module validates column names, row counts, and missing values before upload.

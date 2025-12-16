init Module
===========

Initialize a Kaggle project directory, download competition data, and seed config/state under ``experiments/init/``.

- **Depends on:** none
- **Key flags:** ``--competition``, ``--skip-download``, ``--target-column``, ``--problem-type``, ``--metric``, ``--id-column``, ``--ignore-columns``, ``--submit-probas`` / ``--submit-labels``, ``--cdp-url``
- **Outputs:** project scaffold, downloaded datasets, state at ``experiments/init/state.json``

Source: ``src/mlarena/modules/init.py``

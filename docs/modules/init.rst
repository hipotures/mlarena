init Module
===========

Initialize a Kaggle project directory, download competition data, and seed config/state under ``experiments/init/``.

- **Depends on:** none
- **Key overrides:** ``init.competition=<slug>``, ``init.skip_download=true``, ``init.target_column=col``, ``init.problem_type=binary``, ``init.metric=auc``, ``init.id_column=id``, ``init.ignore_columns=[c1,c2]``, ``init.submit_probas=true``
- **Outputs:** project scaffold, downloaded datasets, state at ``experiments/init/state.json``

Source: ``src/mlarena/modules/init.py``

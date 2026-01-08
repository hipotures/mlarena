preprocess Module
================

Execute a preprocessing step or chain element defined by a template; writes processed train/test (and optional orig) datasets plus shape metadata.

- **Depends on:** none
- **Key overrides:** ``preprocess_template=<name>`` (required), ``preprocess.cache=true``, ``preprocess.quiet_preprocess_panel=true``
- **Outputs:** ``train_processed.csv``, ``test_processed.csv``, optional ``orig_processed.csv`` under ``experiments/<id>/artifacts/preprocess/``; shapes and template info in ``state.json``

Source: ``src/mlarena/modules/preprocess.py``  
See also: ``docs/modules/preprocessing.rst`` for submodule parameter reference.

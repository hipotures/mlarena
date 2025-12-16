stack Module
============

Average multiple prediction files to produce a stacked submission.

- **Depends on:** predict
- **Key flags:** ``--prediction-files``, ``--id-column``, ``--target-column``
- **Outputs:** ``stacked_submission.csv`` under ``experiments/<exp>/artifacts/stack/`` plus payload listing input files

Source: ``src/mlarena/modules/stack.py``

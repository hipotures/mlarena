predict Module
==============

Load the trained model artifact and generate a submission-ready prediction file, reusing preprocessing context when available.

- **Depends on:** model
- **Key flags:** ``--predict-suffix``
- **Outputs:** submission CSV under ``experiments/<exp>/artifacts/predict/``, payload with ``submission_file`` and feature count

Source: ``src/mlarena/modules/predict.py``

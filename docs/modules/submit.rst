submit Module
=============

Validate and optionally upload a submission CSV to Kaggle, recording payload and artifacts.

- **Depends on:** predict
- **Key flags:** ``--skip-submit``, ``--message``, ``--auto-submit``
- **Outputs:** submission status markers, payload with submission path/local CV/public score when available; updates experiment state

Source: ``src/mlarena/modules/submit.py``

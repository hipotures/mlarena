submit Module
=============

Validate and optionally upload a submission CSV to Kaggle, recording payload and artifacts.

- **Depends on:** predict
- **Key overrides:** ``skip_submit=true``, ``submit.message="text"``, ``submit.auto_submit=true``
- **Outputs:** submission status markers, payload with submission path/local CV/public score when available; updates experiment state

Source: ``src/mlarena/modules/submit.py``

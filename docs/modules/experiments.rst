Experiments Module
==================

The ``experiments`` module allows you to browse and summarize the results of all pipeline runs stored in the ``experiments/`` directory. It provides a high-level view of experiment status, model templates used, and achieved scores.

Overview
--------

- **Module name**: ``experiments``
- **Dependencies**: None
- **Source**: ``src/mlarena/modules/experiments.py``

Usage
-----

Listing Experiments
~~~~~~~~~~~~~~~~~~

To list all timestamped experiments (``exp-YYYYMMDD-HHMMSS``) for the current project:

.. code-block:: bash

   uv run python scripts/mla.py experiments project=<competition-slug> list

Display Modes
~~~~~~~~~~~~~

The module supports different table views:

- **Standard Table** (default): Shows experiment ID, status, last module, template, scores, and git hash.
- **Compact Table**: Hides columns like Preset, GPU, and TimeLimit for a cleaner view on small terminals.

.. code-block:: bash

   uv run python scripts/mla.py experiments project=<competition-slug> list --show-table-compact

Information Displayed
--------------------

For each experiment, the following information is extracted from its ``state.json``:

- **Experiment ID**: The unique timestamped identifier.
- **Last State**: The status of the most recently executed module (e.g., ``completed``, ``failed``, ``running``).
- **Module**: The name of the last module that was active.
- **Template**: The model or preprocessing template used.
- **Local CV**: The best cross-validation score reported by the model.
- **Public**: The public leaderboard score (if fetched or submitted).
- **Started**: The start timestamp of the experiment.
- **Elapsed**: Duration of the run.
- **Git**: Short git hash of the codebase at the time of execution.

CLI Arguments
-------------

- ``list``: Command to list experiments.
- ``--show-table``: Explicitly request the full table view.
- ``--show-table-compact``: Request the compact table view.
Project is selected via the core override: ``project=<slug>``.

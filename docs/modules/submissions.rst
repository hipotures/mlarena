Submissions Module
==================

The ``submissions`` module provides tools for listing, adding, and updating tracked submissions for a project. It interfaces with the ``submissions/submissions.json`` file to manage competition entries and their scores.

Overview
--------

- **Module name**: ``submissions``
- **Dependencies**: None
- **Source**: ``src/mlarena/modules/submissions.py``

Usage
-----

Listing Submissions
~~~~~~~~~~~~~~~~~~

To list all tracked submissions for the current project:

.. code-block:: bash

   uv run python scripts/mla.py submissions --project <competition-slug> list

You can limit the output or sort by different criteria:

.. code-block:: bash

   uv run python scripts/mla.py submissions --project <competition-slug> list --limit 10 --sort-by public_score

**Available sort keys**: ``id``, ``local_cv_score``, ``public_score``, ``private_score``, ``timestamp``.

Adding a Submission
~~~~~~~~~~~~~~~~~~

Manual addition of a submission entry (if not created via the ``submit`` module):

.. code-block:: bash

   uv run python scripts/mla.py submissions --project <competition-slug> add <filename> <model_name> --local-cv 0.85 --public 0.82

Updating Scores
~~~~~~~~~~~~~~~

Update public or private scores for an existing submission by its ID:

.. code-block:: bash

   uv run python scripts/mla.py submissions --project <competition-slug> update <id> --public 0.835

Exporting
~~~~~~~~~

Export all submissions to a CSV file:

.. code-block:: bash

   uv run python scripts/mla.py submissions --project <competition-slug> export

Commands Summary
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``list``
     - Display a table of submissions from ``submissions.json``.
   * - ``add``
     - Manually add a new submission record.
   * - ``update``
     - Update scores for a specific submission ID.
   * - ``export``
     - Save the submissions list to ``submissions_export.csv``.

CLI Arguments
-------------

- ``--limit <int>``: Max rows to show in list.
- ``--sort-by <choice>``: Column to sort by.
- ``--local-cv <float>``: Local CV score.
- ``--public <float>``: Public leaderboard score.
- ``--private <float>``: Private leaderboard score.
- ``--notes <str>``: Custom notes for the entry.

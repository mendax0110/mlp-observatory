Development
===========

Backend
-------

.. code-block:: bash

   uv sync --extra dev
   uv run mlp-observatory

Frontend
--------

.. code-block:: bash

   cd frontend
   npm install
   npm run dev

Tests
-----

.. code-block:: bash

   uv run pytest

Lint and type check
-------------------

.. code-block:: bash

   uv run ruff check src tests
   uv run mypy

Docs
----

.. code-block:: bash

   uv run sphinx-build -b html docs/source docs/build

Quickstart
==========

Backend
-------

.. code-block:: bash

   uv sync
   uv run mlp-observatory

Open ``http://127.0.0.1:8000/docs`` for the OpenAPI UI.

Frontend
--------

.. code-block:: bash

   cd frontend
   npm install
   npm run dev

Open ``http://localhost:5173``.

Documentation
-------------

Sphinx is installed with the dev extra:

.. code-block:: bash

   uv sync --extra dev
   uv run sphinx-build -b html docs/source docs/build

Frontend build
--------------

.. code-block:: bash

   cd frontend
   npm run build

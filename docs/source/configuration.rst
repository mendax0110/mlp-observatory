Configuration
=============

Backend settings
----------------

Settings use the ``MLPO_`` prefix and read from ``.env`` when present.

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``MLPO_HOST``
     - ``127.0.0.1``
     - Bind host for the FastAPI server.
   * - ``MLPO_PORT``
     - ``8000``
     - Bind port for the FastAPI server.
   * - ``MLPO_ARTIFACTS_DIR``
     - ``runs``
     - Directory for run artifacts (config, history, summary, system stats).
   * - ``MLPO_UPDATE_EVERY_STEPS``
     - ``2``
     - Emit ``batch_update`` events every N train steps.
   * - ``MLPO_CORS_ORIGINS``
     - ``["http://localhost:5173", "http://127.0.0.1:5173"]``
     - Allowed origins for the browser frontend.

Frontend settings
-----------------

The Vite frontend uses these environment variables:

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``VITE_API_BASE``
     - ``http://<host>:8000``
     - Base URL for HTTP requests. Defaults to the current host on port 8000.
   * - ``VITE_WS_BASE``
     - ``ws://<host>:8000``
     - Base URL for WebSocket connections. Defaults to ``VITE_API_BASE`` with ``ws`` scheme.

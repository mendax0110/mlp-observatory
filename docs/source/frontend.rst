Frontend
========

Overview
--------

The frontend is a Vite + React app in ``frontend/``. It connects to the backend via REST and WebSocket.

Key capabilities
----------------

- Start/stop runs and attach to existing runs.
- Live network visualization with edge weights and activation heatmaps.
- Metrics panels for loss, accuracy/R2, gradients, activations, and histograms.
- System resource monitor (CPU/RAM/GPU).
- Replay mode and forward-trace playback.
- Run comparison and diff panels.

Endpoints used
--------------

- ``POST /runs`` (start)
- ``POST /runs/{run_id}/stop`` (stop)
- ``GET /runs`` (list)
- ``GET /runs/{run_id}`` (summary)
- ``GET /runs/{run_id}/history`` (history)
- ``GET /runs/{run_id}/system-stats`` (system stats)
- ``GET /ws/runs/{run_id}`` (live stream)

Frontend configuration
----------------------

See :doc:`configuration` for ``VITE_API_BASE`` and ``VITE_WS_BASE``.

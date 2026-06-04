Architecture
============

Modules
-------

- ``app``: FastAPI app factory and dependency container.
- ``api``: HTTP and WebSocket routes plus request/response schemas.
- ``services``: ``RunService`` orchestrates runs, events, and summaries.
- ``training``: dataset creation, model construction, training loop.
- ``infra``: run repository, WebSocket hub, system monitor.
- ``visualization``: network graph serialization for the frontend.
- ``domain``: config and event dataclasses.

Data flow
---------

1. ``POST /runs`` validates input and calls ``RunService.start_run``.
2. ``RunService`` creates the run directory and saves ``config.json``.
3. Dataset creation and model initialization happen in ``training``.
4. ``Trainer`` emits ``batch_update`` and ``epoch_end`` events.
5. ``SystemMonitor`` emits ``system_stats`` once per second.
6. Events are queued and broadcast via ``WebSocketHub``.
7. On completion, ``RunService`` writes ``history.json``, ``summary.json``, and ``system_stats.json``.

Concurrency
-----------

- Each run executes in an ``asyncio`` task.
- WebSocket publishing uses a bounded queue (max size 256) and drops batch updates when full.
- The service caches recent events (last 1000) and system stats (last 2000).

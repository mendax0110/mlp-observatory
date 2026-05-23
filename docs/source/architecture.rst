Architecture
============

Layers
------

- ``api``: HTTP and WebSocket transport.
- ``services``: run orchestration.
- ``training``: model and optimization loop.
- ``infra``: repository and WebSocket hub.
- ``visualization``: graph serialization.

Design Patterns
---------------

- Strategy pattern for model strategies.
- Factory pattern for model construction.
- Repository pattern for run artifacts.
- Observer/publish pattern for live events.

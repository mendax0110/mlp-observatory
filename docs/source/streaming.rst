WebSocket streaming
====================

Endpoint
--------

``GET /ws/runs/{run_id}``
  WebSocket stream for a run. The server sends buffered events on connect and then streams live updates.

Event types
-----------

All messages are JSON with:

- ``event``: string
- ``run_id``: string
- ``payload``: object

``run_started``
  Payload fields:

  - ``task``: ``binary_classification`` | ``regression``
  - ``model_family``: ``mlp`` | ``linear``
  - ``device``: actual device (``cuda``/``mps``/``cpu``)
  - ``requested_device``: requested device value
  - ``device_warning``: optional warning if fallback occurred
  - ``graph``: ``{nodes: [...], edges: [...]}``
  - ``train``: ``{epochs: int}``
  - ``parameter_count``: int

``batch_update``
  Emitted every ``MLPO_UPDATE_EVERY_STEPS`` steps.

  - ``epoch``
  - ``step``
  - ``train_loss``
  - ``weights`` (sample of parameters)
  - ``activations`` (mean per layer)
  - ``activation_stds`` (std per layer)
  - ``activation_histograms``
  - ``grad_norms``
  - ``dead_neuron_ratio``
  - ``weight_update_norms``
  - ``weight_norms``
  - ``update_ratios``
  - ``saturation`` (``near_zero``/``high_mag`` per layer)
  - ``forward_trace`` (list of layer/value snapshots)

``epoch_end``
  Payload is ``EpochMetrics`` (same shape as ``/runs/{run_id}/history`` entry).

``early_stopped``
  - ``epoch``
  - ``patience``

``run_finished``
  Payload is ``RunSummaryResponse`` (same shape as ``/runs/{run_id}``).

``run_failed``
  - ``message``

``system_stats``
  - ``cpu_percent``
  - ``ram_percent``
  - ``ram_used_gb``
  - ``ram_total_gb``
  - ``gpu`` object or ``null``
  - ``ts``

Retention
---------

- The server keeps the last 1000 events in memory per run.
- The server keeps the last 2000 system stats samples per run.

HTTP API
========

Health
------

``GET /health``
  Returns ``{"status": "ok"}``.

Runs
----

``POST /runs``
  Start a new run.

  Request body: ``StartRunRequest``

  - ``task``: ``binary_classification`` | ``regression`` (default ``binary_classification``)
  - ``data``: ``DataConfigRequest``

    - ``source``: ``synthetic`` | ``csv`` (default ``synthetic``)
    - ``samples``: int, ``>=128`` (default ``8192``)
    - ``features``: int, ``>=1`` (default ``16``)
    - ``train_ratio``: float, ``>0.5`` and ``<0.95`` (default ``0.8``)
    - ``noise``: float, ``0..5`` (default ``0.2``)
    - ``seed``: int (default ``7``)
    - ``dataset_path``: string | null (required when ``source="csv"``)
    - ``target_column``: string | null (required when ``source="csv"``)

  - ``model``: ``ModelConfigRequest``

    - ``family``: ``mlp`` | ``linear`` (default ``mlp``)
    - ``layers``: list of ``LayerConfigRequest`` (default: 96/96/64 units)
      - ``units``: int, ``1..4096`` (default ``64``)
      - ``activation``: ``relu`` | ``gelu`` | ``tanh`` | ``silu`` | ``leaky_relu`` (default ``gelu``)
      - ``dropout``: float, ``0..0.95`` (default ``0.1``)
      - ``norm``: ``none`` | ``batchnorm`` | ``layernorm`` (default ``none``)
    - ``residual_every_2``: bool (default ``false``)
    - ``initialization``: ``xavier`` | ``kaiming`` | ``orthogonal`` (default ``xavier``)
    - ``preset``: ``custom`` | ``fast_baseline`` | ``stable_deep`` | ``sparse_regularized`` | ``high_capacity`` (default ``custom``)

  - ``train``: ``TrainConfigRequest``

    - ``epochs``: int, ``1..1000`` (default ``30``)
    - ``batch_size``: int, ``>=8`` (default ``256``)
    - ``learning_rate``: float, ``>0`` (default ``1e-3``)
    - ``weight_decay``: float, ``>=0`` (default ``1e-4``)
    - ``weight_decay_decoupled``: bool (default ``true``)
    - ``optimizer``: ``adamw`` | ``sgd`` | ``rmsprop`` (default ``adamw``)
    - ``scheduler``: ``none`` | ``cosine`` | ``one_cycle`` | ``step`` (default ``none``)
    - ``scheduler_step_size``: int, ``>=1`` (default ``10``)
    - ``scheduler_gamma``: float, ``>0`` and ``<1`` (default ``0.5``)
    - ``one_cycle_max_lr``: float | null (optional)
    - ``momentum``: float, ``0..1`` (default ``0.9``)
    - ``gradient_clip_norm``: float | null (optional)
    - ``early_stopping_patience``: int | null (optional)
    - ``early_stopping_min_delta``: float, ``>=0`` (default ``0.0``)
    - ``label_smoothing``: float, ``0..0.49`` (default ``0.0``)
    - ``mixed_precision``: bool (default ``false``)
    - ``l1_lambda``: float, ``>=0`` (default ``0.0``)
    - ``l2_lambda``: float, ``>=0`` (default ``0.0``)
    - ``input_normalization``: bool (default ``true``)
    - ``seed``: int (default ``7``)
    - ``device``: ``auto`` | ``cuda`` | ``mps`` | ``cpu`` (default ``auto``)
    - ``trace_sample_index``: int, ``>=0`` (default ``0``)

  Constraints:

  - ``source="csv"`` requires ``dataset_path`` and ``target_column``.
  - MLP models require at least one layer and are limited to ~20M parameters.
  - For synthetic data, ``batch_size`` must be ``<= samples``.
  - ``label_smoothing`` is only valid for classification.

  Response: ``{"run_id": "run-..."}``

``POST /runs/{run_id}/stop``
  Stop a run. Returns a message or ``404`` if not running.

``GET /runs``
  List runs.

  Response: ``RunListResponse``

  - ``runs[]`` with:
    - ``run_id``
    - ``created_at`` (ISO timestamp)
    - ``status``: ``running`` | ``finished`` | ``failed`` | ``stopped`` | ``unknown``
    - ``has_summary``

``GET /runs/{run_id}``
  Fetch summary.

  Response: ``RunSummaryResponse``

  - ``run_id``
  - ``best_val_loss``
  - ``best_val_accuracy`` (accuracy or R2 for regression)
  - ``epochs``
  - ``diagnostics`` (optional)
  - ``recommendations`` (optional)
  - ``feature_importance`` (optional)

``GET /runs/{run_id}/history``
  Fetch epoch history.

  Response: ``{"run_id": ..., "history": [...]}`` where each entry is:

  - ``epoch``
  - ``train_loss``
  - ``val_loss``
  - ``val_accuracy`` (accuracy or R2)
  - ``extras`` with:

    - ``grad_norm_mean``
    - ``weight_norm_mean``
    - ``update_ratio_mean``
    - ``dead_ratio_mean``
    - ``sharpness_proxy``
    - ``saturation`` (``near_zero``/``high_mag`` per layer)
    - ``weight_histograms`` / ``grad_histograms``
    - ``train_activation_histograms`` / ``val_activation_histograms``
    - ``confidence_calibration`` (classification)
    - ``activation_snapshot`` (weights/activations/trace)
    - ``lr``

``GET /runs/{run_id}/trace``
  Latest forward trace captured from ``batch_update``.

  Response: ``{"run_id": ..., "trace": [{"layer": ..., "values": [...]}, ...]}``.

``GET /runs/{run_id}/system-stats``
  System monitor history.

  Response: ``{"run_id": ..., "system_stats": [...]}`` with entries:

  - ``cpu_percent``
  - ``ram_percent``
  - ``ram_used_gb``
  - ``ram_total_gb``
  - ``gpu`` (object or ``null``): ``util_percent``, ``mem_used_mb``, ``mem_total_mb``, ``temp_c``, ``power_w``, ``name``
  - ``ts`` (unix seconds)

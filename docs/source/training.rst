Training workflow
=================

Overview
--------

1. ``POST /runs`` creates a run directory under ``MLPO_ARTIFACTS_DIR``.
2. A dataset is created (synthetic or CSV).
3. A model is built (MLP or linear) and initialized.
4. The trainer runs epochs, emitting WebSocket events.
5. Artifacts are written to disk on completion or failure.

Dataset sources
---------------

Synthetic data
  - Binary classification: Gaussian features, linear boundary, optional noise.
  - Regression: Gaussian features with linear target plus noise.

CSV data
  - Requires ``dataset_path`` and ``target_column``.
  - Non-numeric values cause validation errors.
  - For classification, targets are coerced to 0/1 if needed.

Training loop
-------------

- Train/validation split uses ``train_ratio`` and a fixed seed.
- Optional input normalization is applied on the full dataset.
- Loss:
  - Classification: ``BCEWithLogitsLoss``
  - Regression: ``MSELoss``
- Optimizers: ``adamw``, ``sgd``, ``rmsprop``.
- Schedulers: ``none``, ``cosine``, ``step``, ``one_cycle``.
- Optional gradient clipping, label smoothing, L1/L2 regularization, mixed precision.
- Early stopping uses ``early_stopping_patience`` and ``early_stopping_min_delta``.

Metrics and diagnostics
-----------------------

Per-epoch metrics (``epoch_end``/``history``):

- ``train_loss``
- ``val_loss``
- ``val_accuracy``
  - Classification: accuracy
  - Regression: R2 score

Extras collected:

- ``grad_norm_mean``, ``weight_norm_mean``, ``update_ratio_mean``
- ``dead_ratio_mean`` per layer
- ``sharpness_proxy``
- ``saturation`` (near-zero / high-magnitude fractions)
- ``weight_histograms`` / ``grad_histograms``
- ``train_activation_histograms`` / ``val_activation_histograms``
- ``confidence_calibration`` (classification)
- ``activation_snapshot`` (weights/activations/trace)
- ``lr``

Run summary
-----------

At completion, the service selects the best epoch by ``val_loss`` and writes:

- ``best_val_loss``
- ``best_val_accuracy``
- ``epochs``
- ``diagnostics`` (averaged dead ratios, grad norms, weight update norms)
- ``recommendations`` (simple heuristics)
- ``feature_importance`` (first-layer weight magnitudes)

Artifacts
---------

Each run directory contains:

- ``meta.json``: run id, created_at, status (and optional error)
- ``config.json``: full project config
- ``started.json``: run_started payload
- ``history.json``: epoch metrics
- ``summary.json``: run summary
- ``system_stats.json``: system monitor samples

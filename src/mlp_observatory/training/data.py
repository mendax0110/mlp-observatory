from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from mlp_observatory.domain.models import DataConfig, DataSourceType, ProjectConfig, TaskType


def _make_features_and_weights(config: DataConfig,) -> tuple[np.random.Generator, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    x = rng.normal(0.0, 1.0, size=(config.samples, config.features)).astype(np.float32)
    w = rng.normal(0.0, 1.0, size=(config.features, 1)).astype(np.float32)
    return rng, x, w

def _make_linear_signal(config: DataConfig,) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    x = rng.normal(0.0, 1.0, size=(config.samples, config.features)).astype(np.float32)
    w = rng.normal(0.0, 1.0, size=(config.features, 1)).astype(np.float32)
    signal = x @ w
    noise = (config.noise * rng.normal(0.0, 1.0, size=(config.samples, 1)).astype(np.float32))
    return x, signal + noise


def make_synthetic_binary_dataset(config: DataConfig) -> TensorDataset:
    #rng, x, w = _make_features_and_weights(config)
    #logits = x @ w + config.noise * rng.normal(0.0, 1.0, size=(config.samples, 1)).astype(np.float32)
    x, logits = _make_linear_signal(config)
    y = (logits > 0.0).astype(np.float32)
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


def _make_synthetic_regression_dataset(config: DataConfig) -> TensorDataset:
    #rng, x, w = _make_features_and_weights(config)
    #y = x @ w + config.noise * rng.normal(0.0, 1.0, size=(config.samples, 1)).astype(np.float32)
    x, y = _make_linear_signal(config)
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


def _make_csv_dataset(config: DataConfig, task: TaskType) -> TensorDataset:
    if not config.dataset_path:
        raise ValueError("dataset_path is required for csv data source")
    if not config.target_column:
        raise ValueError("target_column is required for csv data source")

    path = Path(config.dataset_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        if config.target_column not in reader.fieldnames:
            raise ValueError(f"target_column '{config.target_column}' not found in CSV header")

        feature_cols = [name for name in reader.fieldnames if name != config.target_column]
        if not feature_cols:
            raise ValueError("CSV must contain at least one feature column")

        x_rows: list[list[float]] = []
        y_rows: list[float] = []

        for row_index, row in enumerate(reader, start=2):
            try:
                x_rows.append([float(row[col]) for col in feature_cols])
                y_rows.append(float(row[config.target_column]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric value at CSV row {row_index}: {exc}") from exc

    if not x_rows:
        raise ValueError("CSV dataset is empty")

    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.float32).reshape(-1, 1)

    if task == TaskType.binary_classification:
        unique = set(np.unique(y).tolist())
        if not unique.issubset({0.0, 1.0}):
            #y = (y > 0.5).astype(np.float32)
            raise ValueError(
                "Binary classification target must contain only 0 and 1 values"
            )
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


def make_dataset(config: ProjectConfig) -> tuple[TensorDataset, int]:
    source = str(config.data.source)
    if source == DataSourceType.csv.value:
        dataset = _make_csv_dataset(config.data, config.task)
    elif config.task == TaskType.regression:
        dataset = _make_synthetic_regression_dataset(config.data)
    else:
        dataset = make_synthetic_binary_dataset(config.data)

    x, _ = dataset[0]
    input_dim = int(x.shape[0])
    return dataset, input_dim

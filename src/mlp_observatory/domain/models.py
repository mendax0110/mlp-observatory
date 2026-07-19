from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class TaskType(str, Enum):
    binary_classification = "binary_classification"
    regression = "regression"


class DataSourceType(str, Enum):
    synthetic = "synthetic"
    csv = "csv"


ActivationName = Literal["relu", "gelu", "tanh", "silu", "leaky_relu"]
NormName = Literal["none", "batchnorm", "layernorm"]
ModelFamily = Literal["mlp", "linear"]
InitStrategy = Literal["kaiming", "orthogonal", "xavier"]
OptimizerName = Literal["adamw", "adam", "sgd", "rmsprop", "adagrad"]
SchedulerName = Literal["none", "cosine", "step", "one_cycle"]


@dataclass(slots=True)
class LayerConfig:
    units: int = 64
    activation: ActivationName = "gelu"
    dropout: float = 0.1
    norm: NormName = "none"


@dataclass(slots=True)
class DataConfig:
    source: DataSourceType = DataSourceType.synthetic
    samples: int = 8192
    features: int = 16
    train_ratio: float = 0.8
    noise: float = 0.2
    seed: int = 7
    dataset_path: str | None = None
    target_column: str | None = None


@dataclass(slots=True)
class ModelConfig:
    family: ModelFamily = "mlp"
    layers: list[LayerConfig] = field(default_factory=lambda: [LayerConfig(units=96), LayerConfig(units=96), LayerConfig(units=64)])
    residual_every_2: bool = False
    initialization: InitStrategy = "xavier"
    preset: str = "stable_deep"


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    weight_decay_decoupled: bool = True
    optimizer: OptimizerName = "adamw"
    scheduler: SchedulerName = "none"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    one_cycle_max_lr: float | None = None
    momentum: float = 0.9
    gradient_clip_norm: float | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    label_smoothing: float = 0.0
    mixed_precision: bool = False
    l1_lambda: float = 0.0
    l2_lambda: float = 0.0
    input_normalization: bool = True
    seed: int = 7  # seeds model init / training-time randomness (dropout, shuffling, etc.)
    device: str = "auto"
    trace_sample_index: int = 0


@dataclass(slots=True)
class ProjectConfig:
    task: TaskType = TaskType.binary_classification
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass(slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    extras: dict[str, object] = field(default_factory=dict)
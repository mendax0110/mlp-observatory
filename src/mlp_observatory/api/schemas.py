from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DataConfigRequest(BaseModel):
    source: Literal["synthetic", "csv"] = "synthetic"
    samples: int = Field(default=8192, ge=128)
    features: int = Field(default=16, ge=1)
    train_ratio: float = Field(default=0.8, gt=0.5, lt=0.95)
    noise: float = Field(default=0.2, ge=0.0, le=5.0)
    seed: int = 7
    dataset_path: str | None = None
    target_column: str | None = None


class LayerConfigRequest(BaseModel):
    units: int = Field(default=64, ge=1, le=4096)
    activation: Literal["relu", "gelu", "tanh", "silu", "leaky_relu"] = "gelu"
    dropout: float = Field(default=0.1, ge=0.0, lt=0.95)
    norm: Literal["none", "batchnorm", "layernorm"] = "none"


class ModelConfigRequest(BaseModel):
    family: Literal["mlp", "linear"] = "mlp"
    layers: list[LayerConfigRequest] = Field(default_factory=lambda: [LayerConfigRequest(units=96), LayerConfigRequest(units=96), LayerConfigRequest(units=64)])
    residual_every_2: bool = False
    initialization: Literal["xavier", "kaiming", "orthogonal"] = "xavier"
    preset: Literal["custom", "fast_baseline", "stable_deep", "sparse_regularized", "high_capacity"] = "custom"


class TrainConfigRequest(BaseModel):
    epochs: int = Field(default=30, ge=1, le=1000)
    batch_size: int = Field(default=256, ge=8)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    weight_decay_decoupled: bool = True
    optimizer: Literal["adamw", "sgd", "rmsprop"] = "adamw"
    scheduler: Literal["none", "cosine", "one_cycle", "step"] = "none"
    scheduler_step_size: int = Field(default=10, ge=1)
    scheduler_gamma: float = Field(default=0.5, gt=0.0, lt=1.0)
    one_cycle_max_lr: float | None = Field(default=None, gt=0.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    gradient_clip_norm: float | None = Field(default=None, gt=0.0)
    early_stopping_patience: int | None = Field(default=None, ge=1, le=1000)
    early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.49)
    mixed_precision: bool = False
    l1_lambda: float = Field(default=0.0, ge=0.0)
    l2_lambda: float = Field(default=0.0, ge=0.0)
    input_normalization: bool = True
    seed: int = 7
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    trace_sample_index: int = Field(default=0, ge=0)


class StartRunRequest(BaseModel):
    task: Literal["binary_classification", "regression"] = "binary_classification"
    data: DataConfigRequest = Field(default_factory=DataConfigRequest)
    model: ModelConfigRequest = Field(default_factory=ModelConfigRequest)
    train: TrainConfigRequest = Field(default_factory=TrainConfigRequest)

    @model_validator(mode="after")
    def validate_complexity(self) -> StartRunRequest:
        if self.data.source == "csv":
            if not self.data.dataset_path:
                raise ValueError("dataset_path is required when data.source='csv'.")
            if not self.data.target_column:
                raise ValueError("target_column is required when data.source='csv'.")

        if self.model.family == "mlp":
            if len(self.model.layers) == 0:
                raise ValueError("MLP family requires at least one hidden layer.")
            prev = self.data.features
            approx_params = 0
            for layer in self.model.layers:
                approx_params += prev * layer.units + layer.units
                prev = layer.units
            approx_params += prev + 1
            if approx_params > 20_000_000:
                raise ValueError("Model too large for interactive visualization. Reduce layer sizes.")

        if self.data.source == "synthetic" and self.train.batch_size > self.data.samples:
            raise ValueError("Batch size must be <= samples for synthetic data.")

        if self.task == "regression" and self.train.label_smoothing > 0:
            raise ValueError("Label smoothing is only valid for classification.")

        return self


class StartRunResponse(BaseModel):
    run_id: str


class RunSummaryResponse(BaseModel):
    run_id: str
    best_val_loss: float
    best_val_accuracy: float
    epochs: int
    diagnostics: dict[str, object] | None = None
    recommendations: list[str] | None = None
    feature_importance: list[float] | None = None


class RunListItem(BaseModel):
    run_id: str
    created_at: str
    status: Literal["running", "finished", "failed", "stopped", "unknown"]
    has_summary: bool = False


class RunListResponse(BaseModel):
    runs: list[RunListItem]

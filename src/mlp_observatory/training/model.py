from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import logging
from torch import nn

from mlp_observatory.domain.models import LayerConfig, ModelConfig

logger = logging.getLogger(__name__)

_TRACE_VALUE_LIMIT = 24

_ACTIVATIONS = {
    "relu": lambda: nn.ReLU(),
    "tanh": lambda: nn.Tanh(),
    "silu": lambda: nn.SiLU(),
    "gelu": lambda: nn.GELU(),
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1)
}

_NORMS = {
    "batchnorm": lambda dim: nn.BatchNorm1d(dim),
    "layernorm": lambda dim: nn.LayerNorm(dim),
    "none": lambda dim: nn.Identity()
}

_INIT_STRATEGIES = {
    "kaiming": lambda w: nn.init.kaiming_uniform_(w, nonlinearity="relu"),
    "orthogonal": lambda w: nn.init.orthogonal_(w),
    "xavier": lambda w: nn.init.xavier_uniform_(w),
}


class ModelStrategy(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward_with_diagnostics(
        self,
        x: torch.Tensor,
        trace_sample_index: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[dict[str, object]]]:
        raise NotImplementedError


def _build_activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation type: {name}, supported types are: {list(_ACTIVATIONS.keys())}"
        )
    return _ACTIVATIONS[name]()


def _build_norm(name: str, dim: int) -> nn.Module:
    if name not in _NORMS:
        raise ValueError(
            f"Unsupported normalization type: {name}, supported types are: {list(_NORMS.keys())}"
        )
    return _NORMS[name](dim)


def _clamp_sample_index(trace_sample_index: int, batch_size: int) -> int:
    return int(max(0, min(trace_sample_index, batch_size - 1)))


def _trace_entry(
    layer_name: str,
    tensor: torch.Tensor,
    sample_idx: int,
    limit: int | None = _TRACE_VALUE_LIMIT,
) -> dict[str, object]:
    values = tensor[sample_idx].detach().flatten()
    if limit is not None:
        values = values[:limit]
    return {"layer": layer_name, "values": values.cpu().tolist()}


class MlpPredictor(ModelStrategy):
    def __init__(self, config: ModelConfig, input_dim: int) -> None:
        super().__init__()
        self.residual_every_2 = bool(config.residual_every_2)

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_dim = input_dim
        layers: list[LayerConfig] = config.layers
        for layer in layers:
            self.linears.append(nn.Linear(in_dim, layer.units))
            self.norms.append(_build_norm(layer.norm, layer.units))
            self.activations.append(_build_activation(layer.activation))
            self.dropouts.append(nn.Dropout(layer.dropout))
            in_dim = layer.units

        self.output = nn.Linear(in_dim, 1)

    def _apply_residual(self, h: torch.Tensor, hidden_states: list[torch.Tensor], layer_index: int) -> torch.Tensor:
        if not self.residual_every_2:
            return h
        if layer_index < 2:
            return h
        skip = hidden_states[layer_index - 2]
        if skip.shape[-1] == h.shape[-1]:
            return h + skip
        logger.debug(
            "Skipping residual connection at layer %s due to shape mismatch: skip %s, h %s",
            layer_index,
            skip.shape,
            h.shape,
        )
        return h

    def _run_layers(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        out = x
        hidden_states: list[torch.Tensor] = []

        for i, (linear, norm, activation, dropout) in enumerate(zip(self.linears, self.norms, self.activations, self.dropouts)):
            h = linear(out)
            h = norm(h)
            h = activation(h)
            h = self._apply_residual(h, hidden_states, i)
            hidden_states.append(h)
            out = dropout(h)

        return out, hidden_states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self._run_layers(x)
        return logits

    def forward_with_diagnostics(
        self,
        x: torch.Tensor,
        trace_sample_index: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[dict[str, object]]]:
        sample_idx = _clamp_sample_index(trace_sample_index, x.shape[0])
        out, hidden_states = self._run_layers(x)

        logits = self.output(out)

        trace: list[dict[str, object]] = [_trace_entry("input", x, sample_idx)]
        trace.extend(
            _trace_entry(f"hidden_{i}", h, sample_idx)
            for i, h in enumerate(hidden_states, start=1)
        )
        trace.append(_trace_entry("output", logits, sample_idx, limit=None))

        return logits, hidden_states, trace


class LinearPredictor(ModelStrategy):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.output = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(x)

    def forward_with_diagnostics(
        self,
        x: torch.Tensor,
        trace_sample_index: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[dict[str, object]]]:
        sample_idx = _clamp_sample_index(trace_sample_index, x.shape[0])
        logits = self.output(x)
        trace = [
            _trace_entry("input", x, sample_idx),
            _trace_entry("output", logits, sample_idx, limit=None),
        ]
        return logits, [], trace


def _initialize(model: nn.Module, strategy: str) -> None:
    init_fn = _INIT_STRATEGIES.get(strategy, _INIT_STRATEGIES["xavier"])
    for module in model.modules():
        if isinstance(module, nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class ModelFactory:
    @staticmethod
    def build(config: ModelConfig, input_dim: int) -> ModelStrategy:
        if config.family == "linear":
            model: ModelStrategy = LinearPredictor(input_dim=input_dim)
        else:
            model = MlpPredictor(config, input_dim)
        _initialize(model, config.initialization)
        return model
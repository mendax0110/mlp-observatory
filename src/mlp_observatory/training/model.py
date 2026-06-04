from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from mlp_observatory.domain.models import LayerConfig, ModelConfig

_ACTIVATIONS = {
    "relu": lambda: nn.ReLU(),
    "tanh": lambda: nn.Tanh(),
    "silu": lambda: nn.SiLU(),
    "gelu": lambda: nn.GELU(),
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1)
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
    activations = _ACTIVATIONS.get(name, _ACTIVATIONS["gelu"])
    return activations()

def _build_norm(name: str, dim: int) -> nn.Module:
    if name == "batchnorm":
        return nn.BatchNorm1d(dim)
    if name == "layernorm":
        return nn.LayerNorm(dim)
    return nn.Identity()


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
            h = h + skip
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        hidden_states: list[torch.Tensor] = []

        for i, (linear, norm, activation, dropout) in enumerate(zip(self.linears, self.norms, self.activations, self.dropouts)):
            h = linear(out)
            h = norm(h)
            h = activation(h)
            h = self._apply_residual(h, hidden_states, i)
            hidden_states.append(h)
            out = dropout(h)

        return self.output(out)

    def forward_with_diagnostics(
        self,
        x: torch.Tensor,
        trace_sample_index: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[dict[str, object]]]:
        out = x
        hidden_activations: list[torch.Tensor] = []
        trace: list[dict[str, object]] = []

        sample_idx = int(max(0, min(trace_sample_index, x.shape[0] - 1)))
        trace.append({"layer": "input", "values": x[sample_idx].detach().flatten()[:24].cpu().tolist()})

        for i, (linear, norm, activation, dropout) in enumerate(zip(self.linears, self.norms, self.activations, self.dropouts), start=1):
            h = linear(out)
            h = norm(h)
            h = activation(h)
            h = self._apply_residual(h, hidden_activations, i - 1)
            hidden_activations.append(h)
            trace.append({"layer": f"hidden_{i}", "values": h[sample_idx].detach().flatten()[:24].cpu().tolist()})
            out = dropout(h)

        logits = self.output(out)
        trace.append({"layer": "output", "values": logits[sample_idx].detach().flatten().cpu().tolist()})
        return logits, hidden_activations, trace


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
        sample_idx = int(max(0, min(trace_sample_index, x.shape[0] - 1)))
        logits = self.output(x)
        trace = [
            {"layer": "input", "values": x[sample_idx].detach().flatten()[:24].cpu().tolist()},
            {"layer": "output", "values": logits[sample_idx].detach().flatten().cpu().tolist()},
        ]
        return logits, [], trace


def _initialize(model: nn.Module, strategy: str) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if strategy == "kaiming":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif strategy == "orthogonal":
                nn.init.orthogonal_(module.weight)
            else:
                nn.init.xavier_uniform_(module.weight)
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

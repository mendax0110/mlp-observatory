from __future__ import annotations

from dataclasses import dataclass

from mlp_observatory.domain.models import DataConfig, ModelConfig


@dataclass(slots=True)
class NetworkGraph:
    nodes: list[dict[str, object]]
    edges: list[dict[str, object]]


class NetworkGraphBuilder:
    @staticmethod
    def build(data: DataConfig, model: ModelConfig) -> NetworkGraph:
        hidden_sizes = [layer.units for layer in model.layers] if model.family == "mlp" else []
        layer_sizes = [data.features, *hidden_sizes, 1]

        nodes: list[dict[str, object]] = []
        edges: list[dict[str, object]] = []

        for layer_idx, layer_size in enumerate(layer_sizes):
            cap = min(layer_size, 24)
            for n in range(cap):
                node_id = f"L{layer_idx}N{n}"
                nodes.append({"id": node_id, "layer": layer_idx, "index": n})

        for layer_idx in range(len(layer_sizes) - 1):
            src_cap = min(layer_sizes[layer_idx], 24)
            dst_cap = min(layer_sizes[layer_idx + 1], 24)
            for i in range(src_cap):
                for j in range(dst_cap):
                    edges.append(
                        {
                            "id": f"E{layer_idx}_{i}_{j}",
                            "source": f"L{layer_idx}N{i}",
                            "target": f"L{layer_idx + 1}N{j}",
                        }
                    )

        return NetworkGraph(nodes=nodes, edges=edges)

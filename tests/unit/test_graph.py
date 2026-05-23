from mlp_observatory.domain.models import DataConfig, ModelConfig
from mlp_observatory.visualization.network_graph import NetworkGraphBuilder


def test_graph_builder() -> None:
    graph = NetworkGraphBuilder.build(DataConfig(features=8), ModelConfig(hidden_dim=16, hidden_layers=2))
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0

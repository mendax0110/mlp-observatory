from mlp_observatory.domain.models import DataConfig, ModelConfig, LayerConfig
from mlp_observatory.visualization.network_graph import NetworkGraphBuilder


def test_graph_builder() -> None:
    graph = NetworkGraphBuilder.build(
        DataConfig(features=8),
        ModelConfig(layers=[LayerConfig(units=16), LayerConfig(units=16)])
    )
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0

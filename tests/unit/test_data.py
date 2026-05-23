from mlp_observatory.domain.models import DataConfig
from mlp_observatory.training.data import make_synthetic_binary_dataset


def test_dataset_shape() -> None:
    dataset = make_synthetic_binary_dataset(DataConfig(samples=1024, features=12))
    x, y = dataset[0]
    assert x.shape[0] == 12
    assert y.shape[0] == 1

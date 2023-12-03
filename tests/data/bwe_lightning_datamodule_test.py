import torch
import hydra
import pytest
from src.data.bwe_lightning_datamodule import BWELightningDataModule


@pytest.fixture
def dataset_name() -> str:
    return "Cnam-LMSSC/vibravox"


@pytest.fixture(params=["BWE_In-ear_Comply_Foam_microphone"])
def config_name(request) -> str:
    return request.param


@pytest.fixture(params=[True])
def streaming(request) -> bool:
    return request.param


@pytest.fixture(params=[16000])
def sample_rate(request) -> int:
    """Sample frequency."""
    return request.param


@pytest.fixture(params=[1, 4])
def batch_size(request) -> int:
    """Number of files."""
    return request.param


@pytest.fixture(params=[4])
def num_workers(request) -> int:
    """Number of files."""
    return request.param


@pytest.fixture
def bwe_lightning_datamodule_instance(
    dataset_name, config_name, streaming, sample_rate, batch_size, num_workers
) -> BWELightningDataModule:
    """BWELightningDataModule instance."""

    datamodule = BWELightningDataModule(
        dataset_name, config_name, streaming, sample_rate, batch_size, num_workers
    )

    return datamodule


class TestBWELightningDataModule:
    def test_dataset_returns_torch_tensor(self, bwe_lightning_datamodule_instance):
        bwe_lightning_datamodule_instance.setup()
        train_dataset = bwe_lightning_datamodule_instance.train_dataset
        sample = next(iter(train_dataset))

        assert isinstance(sample["audio"], torch.Tensor)



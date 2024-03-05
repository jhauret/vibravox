import os
import torch
import lightning
import hydra
import pytest
from vibravox.lightning_datamodules.bwe import BWELightningDataModule


@pytest.fixture(params=os.listdir("./configs/lightning_datamodules"))
def datamodule_config_name(request) -> str:
    return request.param

@pytest.fixture
def bwe_lightning_datamodule_instance_from_hydra(datamodule_config_name) -> lightning.LightningModule:
    with hydra.initialize(version_base="1.3", config_path="../../configs/lightning_datamodule"):
        cfg = hydra.compose(config_name=datamodule_config_name)
        return hydra.utils.instantiate(cfg)


@pytest.fixture(
    params=["bwe_in-ear_rigid_earpiece_microphone"]
)  # , "bwe_throat_piezoelectric_sensor"
def config_name(request) -> str:
    return request.param


@pytest.fixture(params=[True, False])
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
    config_name, streaming, sample_rate, batch_size, num_workers
) -> BWELightningDataModule:
    """BWELightningDataModule instance."""

    datamodule = BWELightningDataModule(
        config_name, streaming, sample_rate, batch_size, num_workers
    )

    return datamodule


class TestBWELightningDataModule:
    def test_dataset_returns_torch_tensor(self, bwe_lightning_datamodule_instance):
        bwe_lightning_datamodule_instance.setup()
        train_dataset = bwe_lightning_datamodule_instance.train_dataset
        sample = next(iter(train_dataset))

        assert isinstance(sample["audio"]["array"], torch.Tensor)

    def test_dataloader_returns_torch_tensor(self, bwe_lightning_datamodule_instance):
        bwe_lightning_datamodule_instance.setup()
        train_dataloder = bwe_lightning_datamodule_instance.train_dataloader()
        sample = next(iter(train_dataloder))

        assert isinstance(sample[0], torch.Tensor), "Expected two tensors in the batch."
        assert isinstance(sample[1], torch.Tensor), "Expected two tensors in the batch."

    def test_hydra_instantiation(self, bwe_lightning_datamodule_instance_from_hydra):
        bwe_lightning_datamodule_instance_from_hydra.setup()

        assert isinstance(bwe_lightning_datamodule_instance_from_hydra, BWELightningDataModule)

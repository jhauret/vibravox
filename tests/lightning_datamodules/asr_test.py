import numpy as np
import torch

from vibravox.lightning_datamodules.stp import STPLightningDataModule


class TestSTPLightningDataModule:
    def test_dataset_return_type(self, stp_lightning_datamodule_instance):
        stp_lightning_datamodule_instance.setup()
        train_dataset = stp_lightning_datamodule_instance.train_dataset
        dataset_sample = next(iter(train_dataset))

        assert isinstance(dataset_sample["audio"]["array"], np.ndarray)
        assert isinstance(dataset_sample["phonemes"], str)

    def test_dataloader_returns_format(self, stp_lightning_datamodule_instance):
        stp_lightning_datamodule_instance.setup()
        train_dataloder = stp_lightning_datamodule_instance.train_dataloader()
        dataloader_sample = next(iter(train_dataloder))

        assert isinstance(dataloader_sample, list), "Expected a list."
        assert all(
            [isinstance(dataloader_sample[0], torch.Tensor), isinstance(dataloader_sample[1], torch.Tensor)]
        ), "Expected all elements in the tuple to be torch.Tensor."

    def test_hydra_instantiation(self, stp_lightning_datamodule_instance_from_hydra):

        assert isinstance(
            stp_lightning_datamodule_instance_from_hydra, STPLightningDataModule
        )

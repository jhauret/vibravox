import torch

from vibravox.lightning_datamodules.asr import ASRLightningDataModule


class TestASRLightningDataModule:
    def test_dataset_returns_torch_tensor(self, asr_lightning_datamodule_instance):
        asr_lightning_datamodule_instance.setup()
        train_dataset = asr_lightning_datamodule_instance.train_dataset
        dataset_sample = next(iter(train_dataset))

        assert isinstance(dataset_sample["audio"]["array"], torch.Tensor)

    def test_dataloader_returns_format(self, asr_lightning_datamodule_instance):
        asr_lightning_datamodule_instance.setup()
        train_dataloder = asr_lightning_datamodule_instance.train_dataloader()
        dataloader_sample = next(iter(train_dataloder))

        assert isinstance(dataloader_sample, list), "Expected a list."
        assert all(
            [isinstance(dataloader_sample[0], torch.Tensor), isinstance(dataloader_sample[1], torch.Tensor)]
        ), "Expected all elements in the tuple to be torch.Tensor."

    def test_hydra_instantiation(self, asr_lightning_datamodule_instance_from_hydra):

        assert isinstance(
            asr_lightning_datamodule_instance_from_hydra, ASRLightningDataModule
        )

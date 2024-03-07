import torch
from vibravox.lightning_datamodules.bwe import BWELightningDataModule


class TestBWELightningDataModule:
    def test_dataset_returns_torch_tensor(self, bwe_lightning_datamodule_instance):
        bwe_lightning_datamodule_instance.setup()
        train_dataset = bwe_lightning_datamodule_instance.train_dataset
        sample = next(iter(train_dataset))

        assert isinstance(sample["audio"]["array"], torch.Tensor)

    def test_dataloader_returns_format(self, bwe_lightning_datamodule_instance):
        bwe_lightning_datamodule_instance.setup()
        train_dataloder = bwe_lightning_datamodule_instance.train_dataloader()
        sample = next(iter(train_dataloder))

        assert isinstance(sample, tuple), "Expected a tuple."
        assert all([isinstance(sample[0], torch.Tensor), isinstance(sample[1], torch.Tensor)]), "Expected all elements in the tuple to be torch.Tensor."
        assert sample[0].shape == sample[1].shape, "Expected the same number of samples in both tensors."
        assert sample[0].dim() == 3, "Expected 3 dimensions in the tensor."

    def test_hydra_instantiation(self, bwe_lightning_datamodule_instance_from_hydra):
        bwe_lightning_datamodule_instance_from_hydra.setup()

        assert isinstance(bwe_lightning_datamodule_instance_from_hydra, BWELightningDataModule)

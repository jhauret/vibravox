import torch

from vibravox.lightning_datamodules.bwe import BWELightningDataModule


class TestBWELightningDataModule:
    def test_dataset_returns_torch_tensor(self, bwe_lightning_datamodule_instance):
        bwe_lightning_datamodule_instance.setup()
        train_dataset = bwe_lightning_datamodule_instance.train_dataset
        dataset_sample = next(iter(train_dataset))

        assert isinstance(dataset_sample["audio"]["array"], torch.Tensor)

    def test_dataloader_returns_format(self, bwe_lightning_datamodule_instance):
        bwe_lightning_datamodule_instance.setup()
        train_dataloder = bwe_lightning_datamodule_instance.train_dataloader()
        dataloader_sample = next(iter(train_dataloder))

        assert isinstance(dataloader_sample, list), "Expected a list."
        assert all(
            [isinstance(dataloader_sample[0], torch.Tensor), isinstance(dataloader_sample[1], torch.Tensor)]
        ), "Expected all elements in the tuple to be torch.Tensor."
        assert (
            dataloader_sample[0].shape == dataloader_sample[1].shape
        ), "Expected the same number of samples in both tensors."
        assert dataloader_sample[0].dim() == 3, "Expected 3 dimensions in the tensor."

    def test_no_offset_between_audio_samples(self, bwe_lightning_datamodule_instance):
        bwe_lightning_datamodule_instance.setup()
        train_dataloder = bwe_lightning_datamodule_instance.train_dataloader()
        dataloader_sample = next(iter(train_dataloder))

        corrupted_audio = dataloader_sample[0][0:1, 0:1, :]
        reference_audio = dataloader_sample[1][0:1, 0:1, :]

        # Note: applying remove_hf on reference_audio and remove_bf on corrupted audio is not necessary

        correlation = torch.nn.functional.conv1d(corrupted_audio, reference_audio, padding=corrupted_audio.shape[-1])

        shift = torch.argmax(correlation) - corrupted_audio.shape[-1] + 1

        assert shift in range(-12, 12), "Expected minimal offset between audio samples. [-12,12] corresponds to a spacing of 21cm between the microphones."

    def test_hydra_instantiation(self, bwe_lightning_datamodule_instance_from_hydra):

        assert isinstance(
            bwe_lightning_datamodule_instance_from_hydra, BWELightningDataModule
        )

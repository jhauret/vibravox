import torch
import pytest
from vibravox.lightning_datamodules.bwe import BWELightningDataModule


class TestBWELightningDataModule:
    def test_dataset_returns_torch_tensor(self, bwe_lightning_datamodule_instance):
        bwe_lightning_datamodule_instance.setup(stage="test")
        test_dataset = bwe_lightning_datamodule_instance.test_dataset
        dataset_sample = next(iter(test_dataset))

        assert isinstance(dataset_sample["audio_body_conducted"]["array"], torch.Tensor)
        assert isinstance(dataset_sample["audio_airborne"]["array"], torch.Tensor)

    def test_dataloader_returns_format(self, bwe_lightning_datamodule_instance):
        bwe_lightning_datamodule_instance.setup(stage="test")
        test_dataloader = bwe_lightning_datamodule_instance.test_dataloader()
        dataloader_sample = next(iter(test_dataloader))

        assert isinstance(dataloader_sample, dict), "Expected a dict."
        assert all(
            [
                isinstance(dataloader_sample["audio_body_conducted"], torch.Tensor),
                isinstance(dataloader_sample["audio_airborne"], torch.Tensor),
            ]
        ), "Expected all elements in the list to be torch.Tensor."

        assert dataloader_sample["audio_body_conducted"].dim() == 3, "Expected 3 dimensions in the tensor."

    def test_dataloader_returns_tensors_with_same_shape(self, bwe_lightning_datamodule_instance):
        bwe_lightning_datamodule_instance.setup(stage="test")
        test_dataloader = bwe_lightning_datamodule_instance.test_dataloader()
        dataloader_sample = next(iter(test_dataloader))

        if bwe_lightning_datamodule_instance.dataset_name == "Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp":
            pytest.skip("Skipping for vibravox_enhanced_by_EBEN_tmp because the audio_body_conducted are shorter.")
        assert (
            dataloader_sample["audio_body_conducted"].shape == dataloader_sample["audio_airborne"].shape
        ), "Expected the same number of samples in both tensors."

    def test_no_offset_between_audio_samples(self, bwe_lightning_datamodule_instance):
        bwe_lightning_datamodule_instance.setup(stage="test")
        if bwe_lightning_datamodule_instance.dataset_name == "Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp":
            pytest.skip("Skipping for Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp")
        if bwe_lightning_datamodule_instance.subset == "speech_noisy":
            pytest.skip("Skipping for noisy speech.")

        test_dataloder = bwe_lightning_datamodule_instance.test_dataloader()
        dataloader_sample = next(iter(test_dataloder))

        corrupted_audio = dataloader_sample["audio_body_conducted"][0:1, 0:1, :]
        reference_audio = dataloader_sample["audio_airborne"][0:1, 0:1, :]

        # Note: applying remove_hf on reference_audio and remove_bf on corrupted audio is not necessary

        correlation = torch.nn.functional.conv1d(
            corrupted_audio, reference_audio, padding=corrupted_audio.shape[-1]
        )

        shift = torch.argmax(correlation) - corrupted_audio.shape[-1] + 1

        assert shift in range(
            -24, 24
        ), "Expected minimal offset between audio samples. [-24,24] corresponds to a spacing of 42cm between the microphones."

    def test_hydra_instantiation(self, bwe_lightning_datamodule_instance_from_hydra):

        assert isinstance(
            bwe_lightning_datamodule_instance_from_hydra, BWELightningDataModule
        )

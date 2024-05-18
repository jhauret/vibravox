import numpy as np
import torch

from vibravox.lightning_datamodules.spkv import SPKVLightningDataModule


class TestSPKVLightningDataModule:
    def test_dataset_test_return_type(self, spkv_lightning_datamodule_same_sensors_instance):
        spkv_lightning_datamodule_same_sensors_instance.setup("test")
        test_dataset_a = spkv_lightning_datamodule_same_sensors_instance.test_dataset_a
        test_dataset_b = spkv_lightning_datamodule_same_sensors_instance.test_dataset_b

        dataset_sample_a = next(iter(test_dataset_a))
        dataset_sample_b = next(iter(test_dataset_b))

        assert isinstance(dataset_sample_a["audio"]["array"], torch.Tensor)
        assert isinstance(dataset_sample_b["audio"]["array"], torch.Tensor)

        assert isinstance(dataset_sample_a["speaker_id"], str)
        assert isinstance(dataset_sample_b["speaker_id"], str)

        assert isinstance(dataset_sample_a["sentence_id"], torch.Tensor)
        assert isinstance(dataset_sample_b["sentence_id"], torch.Tensor)

        assert isinstance(dataset_sample_a["gender"], str)
        assert isinstance(dataset_sample_b["gender"], str)

        assert isinstance(dataset_sample_a["sensor"], str)
        assert isinstance(dataset_sample_b["sensor"], str)

    def test_dataloader_test_returns_format(self, spkv_lightning_datamodule_same_sensors_instance):
        spkv_lightning_datamodule_same_sensors_instance.setup("test")
        test_dataloder = spkv_lightning_datamodule_same_sensors_instance.test_dataloader()
        dataloader_sample = next(iter(test_dataloder))

        assert isinstance(dataloader_sample, tuple), "Expected dataloader_sample to be a dictionary."
        assert dataloader_sample.keys() == {"sensor_a", "sensor_b"}
        assert isinstance(dataloader_sample["sensor_a"], dict), "Expected dataloader_sample['sensor_a'] to be a dictionary."
        assert isinstance(dataloader_sample["sensor_b"], dict), "Expected dataloader_sample['sensor_b'] to be a dictionary."
        assert dataloader_sample["sensor_a"].keys() == {"audio", "speaker_id", "sentence_id", "gender", "sensor"}
        assert dataloader_sample["sensor_b"].keys() == {"audio", "speaker_id", "sentence_id", "gender", "sensor"}


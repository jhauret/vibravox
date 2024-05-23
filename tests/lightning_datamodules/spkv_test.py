import torch
import pytest
from vibravox.lightning_datamodules.spkv import SPKVLightningDataModule


class TestSPKVLightningDataModule:
    def test_dataset_test_stage_returns_type(self, spkv_lightning_datamodule_same_sensors_instance):
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

    def test_dataloader_test_stage_returns_format(self, spkv_lightning_datamodule_same_sensors_instance):
        spkv_lightning_datamodule_same_sensors_instance.setup("test")
        test_dataloder = spkv_lightning_datamodule_same_sensors_instance.test_dataloader()
        dataloader_sample = next(iter(test_dataloder))
        dataloader_sample_0 = dataloader_sample[0]

        # assert isinstance(dataloader_sample, dict), "Expected dataloader_sample to be a dictionary."
        # assert dataloader_sample.keys() == {"sensor_a", "sensor_b"}
        assert isinstance(dataloader_sample_0["sensor_a"], dict), "Expected dataloader_sample['sensor_a'] to be a dictionary."
        assert isinstance(dataloader_sample_0["sensor_b"], dict), "Expected dataloader_sample['sensor_b'] to be a dictionary."
        assert dataloader_sample_0["sensor_a"].keys() == {"audio", "speaker_id", "sentence_id", "gender", "sensor"}
        assert dataloader_sample_0["sensor_b"].keys() == {"audio", "speaker_id", "sentence_id", "gender", "sensor"}

        assert isinstance(dataloader_sample_0["sensor_a"]["audio"], torch.Tensor)
        assert isinstance(dataloader_sample_0["sensor_b"]["audio"], torch.Tensor)
        assert dataloader_sample_0["sensor_a"]["audio"].ndim == 3
        assert dataloader_sample_0["sensor_b"]["audio"].ndim == 3
        assert dataloader_sample_0["sensor_a"]["audio"].shape[1] == 1
        assert dataloader_sample_0["sensor_b"]["audio"].shape[1] == 1

        assert isinstance(dataloader_sample_0["sensor_a"]["speaker_id"], list)
        assert isinstance(dataloader_sample_0["sensor_b"]["speaker_id"], list)
        assert isinstance(dataloader_sample_0["sensor_a"]["speaker_id"][0], str)
        assert isinstance(dataloader_sample_0["sensor_b"]["speaker_id"][0], str)

        assert isinstance(dataloader_sample_0["sensor_a"]["sentence_id"], list)
        assert isinstance(dataloader_sample_0["sensor_b"]["sentence_id"], list)
        assert isinstance(dataloader_sample_0["sensor_a"]["sentence_id"][0], int)
        assert isinstance(dataloader_sample_0["sensor_b"]["sentence_id"][0], int)

        assert isinstance(dataloader_sample_0["sensor_a"]["gender"], list)
        assert isinstance(dataloader_sample_0["sensor_b"]["gender"], list)
        assert isinstance(dataloader_sample_0["sensor_a"]["gender"][0], str)
        assert isinstance(dataloader_sample_0["sensor_b"]["gender"][0], str)
        assert dataloader_sample_0["sensor_a"]["gender"][0] in ["male", "female"]
        assert dataloader_sample_0["sensor_b"]["gender"][0] in ["male", "female"]

        assert isinstance(dataloader_sample_0["sensor_a"]["sensor"], list)
        assert isinstance(dataloader_sample_0["sensor_b"]["sensor"], list)
        assert isinstance(dataloader_sample_0["sensor_a"]["sensor"][0], str)
        assert isinstance(dataloader_sample_0["sensor_b"]["sensor"][0], str)

        assert dataloader_sample_0["sensor_a"]["sensor"][0] in ["body_conducted.temple.contact_microphone",
                                                        "body_conducted.throat.piezoelectric_sensor",
                                                        "body_conducted.in_ear.rigid_earpiece_microphone",
                                                        "body_conducted.in_ear.comply_foam_microphone",
                                                        "body_conducted.forehead.miniature_accelerometer",
                                                        "airborne.mouth_headworn.reference_microphone"]

        assert dataloader_sample_0["sensor_b"]["sensor"][0] in ["body_conducted.temple.contact_microphone",
                                                      "body_conducted.throat.piezoelectric_sensor",
                                                      "body_conducted.in_ear.rigid_earpiece_microphone",
                                                      "body_conducted.in_ear.comply_foam_microphone",
                                                      "body_conducted.forehead.miniature_accelerometer",
                                                      "airborne.mouth_headworn.reference_microphone"]

    def test_dataset_fit_stage_returns_type(self, spkv_lightning_datamodule_same_sensors_instance):
        if spkv_lightning_datamodule_same_sensors_instance.dataset_name=="Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp":
            pytest.skip("Skipping for 'Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp' because there is no train/validation sets")
        spkv_lightning_datamodule_same_sensors_instance.setup("fit")
        val_dataset = spkv_lightning_datamodule_same_sensors_instance.val_dataset
        train_dataset = spkv_lightning_datamodule_same_sensors_instance.train_dataset
        dataset_sample_val = next(iter(val_dataset))
        dataset_sample_train = next(iter(train_dataset))

        assert isinstance(dataset_sample_val["audio"]["array"], torch.Tensor)
        assert isinstance(dataset_sample_train["audio"]["array"], torch.Tensor)

        assert isinstance(dataset_sample_val["speaker_id"], str)
        assert isinstance(dataset_sample_train["speaker_id"], str)

        assert isinstance(dataset_sample_val["sentence_id"], torch.Tensor)
        assert isinstance(dataset_sample_train["sentence_id"], torch.Tensor)

        assert isinstance(dataset_sample_val["gender"], str)
        assert isinstance(dataset_sample_train["gender"], str)

        assert isinstance(dataset_sample_val["sensor"], str)
        assert isinstance(dataset_sample_train["sensor"], str)

    def test_dataloader_fit_stage_returns_format(self, spkv_lightning_datamodule_same_sensors_instance):
        if spkv_lightning_datamodule_same_sensors_instance.dataset_name=="Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp":
            pytest.skip("Skipping for 'Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp' because there is no train/validation sets")

        spkv_lightning_datamodule_same_sensors_instance.setup("fit")
        train_dataloder = spkv_lightning_datamodule_same_sensors_instance.train_dataloader()
        val_dataloder = spkv_lightning_datamodule_same_sensors_instance.val_dataloder()
        train_dataloader_sample = next(iter(train_dataloder))
        val_dataloader_sample = next(iter(val_dataloder))

        assert isinstance(train_dataloader_sample, dict), "Expected dataloader_sample to be a dictionary."
        assert isinstance(val_dataloader_sample, dict), "Expected dataloader_sample to be a dictionary."
        assert train_dataloader_sample.keys() == {"audio", "speaker_id", "sentence_id", "gender", "sensor"}
        assert val_dataloader_sample.keys() == {"audio", "speaker_id", "sentence_id", "gender", "sensor"}

        assert isinstance(train_dataloader_sample["audio"], torch.Tensor)
        assert isinstance(val_dataloader_sample["audio"], torch.Tensor)
        assert train_dataloader_sample["audio"].ndim == 3
        assert val_dataloader_sample["audio"].ndim == 3
        assert train_dataloader_sample["audio"].shape[1] == 1
        assert val_dataloader_sample["audio"].shape[1] == 1

        assert isinstance(train_dataloader_sample["speaker_id"], list)
        assert isinstance(val_dataloader_sample["speaker_id"], list)
        assert isinstance(train_dataloader_sample["speaker_id"][0], str)
        assert isinstance(val_dataloader_sample["speaker_id"][0], str)

        assert isinstance(train_dataloader_sample["sentence_id"], list)
        assert isinstance(val_dataloader_sample["sentence_id"], list)
        assert isinstance(train_dataloader_sample["sentence_id"][0], int)
        assert isinstance(val_dataloader_sample["sentence_id"][0], int)

        assert isinstance(train_dataloader_sample["gender"], list)
        assert isinstance(val_dataloader_sample["gender"], list)
        assert isinstance(train_dataloader_sample["gender"][0], str)
        assert isinstance(val_dataloader_sample["gender"][0], str)
        assert train_dataloader_sample["gender"][0] in ["male", "female"]
        assert val_dataloader_sample["gender"][0] in ["male", "female"]

        assert isinstance(train_dataloader_sample["sensor"], list)
        assert isinstance(val_dataloader_sample["sensor"], list)
        assert isinstance(train_dataloader_sample["sensor"][0], str)
        assert isinstance(val_dataloader_sample["sensor"][0], str)

        assert train_dataloader_sample["sensor"][0] in ["body_conducted.temple.contact_microphone",
                                                        "body_conducted.throat.piezoelectric_sensor",
                                                        "body_conducted.in_ear.rigid_earpiece_microphone",
                                                        "body_conducted.in_ear.comply_foam_microphone",
                                                        "body_conducted.forehead.miniature_accelerometer",
                                                        "airborne.mouth_headworn.reference_microphone"]

        assert val_dataloader_sample["sensor"][0] in ["body_conducted.temple.contact_microphone",
                                                        "body_conducted.throat.piezoelectric_sensor",
                                                        "body_conducted.in_ear.rigid_earpiece_microphone",
                                                        "body_conducted.in_ear.comply_foam_microphone",
                                                        "body_conducted.forehead.miniature_accelerometer",
                                                        "airborne.mouth_headworn.reference_microphone"]

    def test_hydra_instantiation(self, spkv_lightning_datamodule_instance_from_hydra):
        pytest.skip("Skipping for now.")
        assert isinstance(
            spkv_lightning_datamodule_instance_from_hydra, SPKVLightningDataModule
        )


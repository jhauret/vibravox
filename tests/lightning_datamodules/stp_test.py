import numpy as np
import torch

from vibravox.lightning_datamodules.stp import STPLightningDataModule


class TestSTPLightningDataModule:
    def test_dataset_return_type(self, stp_lightning_datamodule_instance):
        train_dataset = stp_lightning_datamodule_instance.train_dataset
        dataset_sample = next(iter(train_dataset))

        assert isinstance(dataset_sample["audio"]["array"], np.ndarray)
        assert isinstance(dataset_sample["phonemes"], str)

    def test_dataloader_returns_format(self, stp_lightning_datamodule_instance):
        train_dataloder = stp_lightning_datamodule_instance.train_dataloader()
        dataloader_sample = next(iter(train_dataloder))

        assert isinstance(dataloader_sample, list), "Expected a list."
        assert all(
            [isinstance(dataloader_sample[0], torch.Tensor), isinstance(dataloader_sample[1], torch.Tensor)]
        ), "Expected all elements in the tuple to be torch.Tensor."

    def test_tokenize_detokenize_is_bijection(self, stp_lightning_datamodule_instance):
        train_dataset = stp_lightning_datamodule_instance.train_dataset
        dataset_sample = next(iter(train_dataset))
        phonemes = dataset_sample["phonemes"]
        tokenized_phonemes = stp_lightning_datamodule_instance.tokenizer(phonemes)
        detokenized_phonemes = stp_lightning_datamodule_instance.tokenizer.decode(token_ids=tokenized_phonemes.input_ids)

        assert phonemes == detokenized_phonemes

    def test_hydra_instantiation(self, stp_lightning_datamodule_instance_from_hydra):

        assert isinstance(
            stp_lightning_datamodule_instance_from_hydra, STPLightningDataModule
        )

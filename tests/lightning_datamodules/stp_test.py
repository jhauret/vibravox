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

        assert isinstance(dataloader_sample, dict), "Expected a list."
        assert all(
            [isinstance(dataloader_sample['audio'], torch.Tensor), isinstance(dataloader_sample['phonemes_ids'], torch.Tensor)]
        ), "Expected all elements in the tuple to be torch.Tensor."

    def test_tokenize_detokenize_is_bijection_from_dataset(self, stp_lightning_datamodule_instance):
        train_dataset = stp_lightning_datamodule_instance.train_dataset
        dataset_sample = next(iter(train_dataset))
        phonemes = dataset_sample["phonemes"]
        tokenized_phonemes = stp_lightning_datamodule_instance.tokenizer(phonemes, add_special_tokens=True)
        detokenized_phonemes = stp_lightning_datamodule_instance.tokenizer.decode(token_ids=tokenized_phonemes.input_ids)

        assert phonemes == detokenized_phonemes

    def test_tokenize_detokenize_is_bijection_from_dataloader(self, stp_lightning_datamodule_instance):
        train_dataloder = stp_lightning_datamodule_instance.train_dataloader()
        dataloader_sample = next(iter(train_dataloder))
        phonemes_ids = dataloader_sample["phonemes_ids"][0, :]  # First sample in the batch
        phonemes = stp_lightning_datamodule_instance.tokenizer.decode(token_ids=phonemes_ids)
        phonemes_ids_bis = stp_lightning_datamodule_instance.tokenizer(phonemes,  add_special_tokens=True)
        phonemes_ids_bis = torch.Tensor(phonemes_ids_bis.input_ids)
        phonemes_bis = stp_lightning_datamodule_instance.tokenizer.decode(token_ids=phonemes_ids_bis)

        assert phonemes == phonemes_bis

    def test_hydra_instantiation(self, stp_lightning_datamodule_instance_from_hydra):

        assert isinstance(
            stp_lightning_datamodule_instance_from_hydra, STPLightningDataModule
        )

import numpy as np
import torch
import os
from huggingface_hub import login
from vibravox.lightning_datamodules.stp import STPLightningDataModule

if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])


class TestSTPLightningDataModule:
    def test_dataset_return_type(self, stp_lightning_datamodule_instance):
        stp_lightning_datamodule_instance.setup(stage="test")
        test_dataset = stp_lightning_datamodule_instance.test_dataset_principal
        dataset_sample = next(iter(test_dataset))

        assert isinstance(dataset_sample["audio"]["array"], np.ndarray)
        assert isinstance(dataset_sample["phonemized_text"], str)

    def test_dataloader_returns_format(self, stp_lightning_datamodule_instance):
        stp_lightning_datamodule_instance.setup(stage="test")
        test_dataloder = stp_lightning_datamodule_instance.test_dataloader()
        dataloader_sample = next(iter(test_dataloder))

        assert isinstance(dataloader_sample, dict), "Expected a list."
        assert all(
            [
                isinstance(dataloader_sample["audio"], torch.Tensor),
                isinstance(dataloader_sample["phonemes_ids"], torch.Tensor),
            ]
        ), "Expected all elements in the tuple to be torch.Tensor."

    def test_tokenize_detokenize_is_bijection_from_dataset(self, stp_lightning_datamodule_instance):
        stp_lightning_datamodule_instance.setup(stage="test")
        test_dataset = stp_lightning_datamodule_instance.test_dataset_principal
        dataset_sample = next(iter(test_dataset))
        phonemes = dataset_sample["phonemized_text"]
        tokenized_phonemes = stp_lightning_datamodule_instance.tokenizer(phonemes, add_special_tokens=True)
        detokenized_phonemes = stp_lightning_datamodule_instance.tokenizer.decode(
            token_ids=tokenized_phonemes.input_ids
        )

        assert phonemes == detokenized_phonemes

    def test_tokenize_detokenize_is_bijection_from_dataloader(self, stp_lightning_datamodule_instance):
        stp_lightning_datamodule_instance.setup(stage="test")
        test_dataloader = stp_lightning_datamodule_instance.test_dataloader()
        dataloader_sample = next(iter(test_dataloader))
        phonemes_ids = dataloader_sample["phonemes_ids"][0, :]  # First sample in the batch
        phonemes = stp_lightning_datamodule_instance.tokenizer.decode(token_ids=phonemes_ids)
        phonemes_ids_bis = stp_lightning_datamodule_instance.tokenizer(phonemes, add_special_tokens=True)
        phonemes_ids_bis = torch.Tensor(phonemes_ids_bis.input_ids)
        phonemes_bis = stp_lightning_datamodule_instance.tokenizer.decode(token_ids=phonemes_ids_bis)

        assert phonemes == phonemes_bis

    def test_hydra_instantiation(self, stp_lightning_datamodule_instance_from_hydra):

        assert isinstance(stp_lightning_datamodule_instance_from_hydra, STPLightningDataModule)

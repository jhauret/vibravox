from typing import Dict, Union, List

import torch
from datasets import load_dataset, Audio, DatasetDict
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from vibravox.torch_modules.dsp.data_augmentation import WaveformDataAugmentation


class STPLightningDataModule(LightningDataModule):

    LIST_OF_VIBRAVOX = [
        "Cnam-LMSSC/vibravox",
        "Cnam-LMSSC/vibravox2",
        "Cnam-LMSSC/vibravox-test",
        "Cnam-LMSSC/non_curated_vibravox",
        "Cnam-LMSSC/vibravox_enhanced_by_EBEN",
    ]

    def __init__(
        self,
        sample_rate: int = 16000,
        dataset_name_principal: str = "Cnam-LMSSC/vibravox",
        dataset_name_secondary: str = None,
        subset: str = "speech_clean",
        sensor: str = "headset_microphone",
        streaming: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        feature_extractor: Wav2Vec2FeatureExtractor = None,
        tokenizer: Wav2Vec2CTCTokenizer = None,
        data_augmentation: torch.nn.Module = None,
        **kwargs,
    ):
        """
        LightningDataModule for Speech-to-Phoneme (STP).

        Args:
            sample_rate (int, optional): Sample rate at which the dataset is output. Defaults to 16000.
            dataset_name_principal (str, optional): Principal dataset name that is going to be used for train/validation and testing.
                Default to "Cnam-LMSSC/vibravox".
            dataset_name_secondary (str, optional): Secondary dataset name that is going to be used for validation and testing.
                Default to None
            subset (str, optional): Subset. Defaults to "speech_clean"
            sensor (str, optional): Sensor. Defaults to "headset_microphone"
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
            feature_extractor (Wav2Vec2FeatureExtractor): Feature extractor. Defaults to None.
            tokenizer (Wav2Vec2CTCTokenizer): Tokenizer. Defaults to None.
            data_augmentation (nn.Module, optional): Data augmentation module. Defaults to None.
        """

        super().__init__()

        self.sample_rate = sample_rate
        self.dataset_name_principal = dataset_name_principal
        assert (
            dataset_name_principal in self.LIST_OF_VIBRAVOX
        ), f"dataset_name_principal {dataset_name_principal} not supported."

        self.dataset_name_secondary = dataset_name_secondary
        assert (
            dataset_name_secondary is None or dataset_name_secondary in self.LIST_OF_VIBRAVOX
        ), f"dataset_name_secondary {dataset_name_secondary} not supported."
        self.subset = subset
        self.sensor = sensor
        self.streaming = streaming
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        
        if data_augmentation is None:
            data_augmentation = WaveformDataAugmentation(sample_rate)
        assert isinstance(
            data_augmentation, WaveformDataAugmentation
        ), "data_augmentation must be a WaveformDataAugmentation"

        self.data_augmentation = data_augmentation

    def setup(self, stage=None):
        """
        Set up the datasets.

        Args:
            stage (str): Pipeline stage among ['fit', 'validate', 'test', 'predict']. Defaults to None.

        Notes:
            This function runs on every accelerator in distributed mode.
            That is why it is necessary to define attributes here rather than in __init__.
        """

        dataset_dict_principal = load_dataset(self.dataset_name_principal, self.subset, streaming=self.streaming)
        dataset_dict_principal = self.prepare_dataset_dict(dataset_dict_principal)

        if self.dataset_name_secondary is not None:
            dataset_dict_secondary = load_dataset(self.dataset_name_secondary, self.subset, streaming=self.streaming)
            dataset_dict_secondary = self.prepare_dataset_dict(dataset_dict_secondary)

        if stage == "fit" or stage is None:
            self.train_dataset_principal = dataset_dict_principal["train"]
            self.val_dataset_principal = dataset_dict_principal["validation"]
            if self.dataset_name_secondary is not None:
                self.val_dataset_secondary = dataset_dict_secondary["validation"]
        elif stage == "test" or stage is None:
            self.test_dataset_principal = dataset_dict_principal["test"]
            if self.dataset_name_secondary is not None:
                self.test_dataset_secondary = dataset_dict_secondary["test"]

    def prepare_dataset_dict(self, dataset_dict: DatasetDict) -> DatasetDict:
        """
        Prepare the dataset dictionary.

        Args:
            dataset_dict (DatasetDict): Dataset dictionary.

        Returns:
            DatasetDict: Prepared dataset dictionary.
        """

        dataset_dict = dataset_dict.rename_column(f"audio.{self.sensor}", "audio")

        dataset_dict = dataset_dict.select_columns(["audio", "phonemized_text"])

        # Resample the audio to the right sample rate
        dataset_dict = dataset_dict.cast_column("audio", Audio(sampling_rate=self.sample_rate, mono=False))

        return dataset_dict

    def train_dataloader(self) -> DataLoader:
        """
        Train dataloader.

        Returns:
            DataLoader
        """

        return DataLoader(
            self.train_dataset_principal,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(
                    batch, deterministic=False
                ),
        )

    def val_dataloader(self) -> Union[DataLoader, Dict[str, DataLoader]]:
        """
        Validation dataloader.

        Returns:
             Union[DataLoader, Dict[str, DataLoader]]
        """

        dataloader_principal = DataLoader(
            self.val_dataset_principal,
            batch_size=min(1, self.batch_size // 4),
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(
                    batch, deterministic=True
                ),
        )

        if self.dataset_name_secondary is not None:
            dataloader_secondary = DataLoader(
                self.val_dataset_secondary,
                batch_size=min(1, self.batch_size // 4),
                num_workers=self.num_workers,
                collate_fn=lambda batch: self.data_collator(
                    batch, deterministic=True
                ),
            )
            return {"principal": dataloader_principal, "secondary": dataloader_secondary}
        else:
            return dataloader_principal

    def test_dataloader(self) -> Union[DataLoader, Dict[str, DataLoader]]:
        """
        Test dataloader.

        Returns:
             Union[DataLoader, Dict[str, DataLoader]]
        """

        dataloader_principal = DataLoader(
            self.test_dataset_principal,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(
                    batch, deterministic=True
                ),
        )

        if self.dataset_name_secondary is not None:
            dataloader_secondary = DataLoader(
                self.test_dataset_secondary,
                batch_size=1,
                num_workers=self.num_workers,
                collate_fn=lambda batch: self.data_collator(
                    batch, deterministic=True
                ),
            )
            return {"principal": dataloader_principal, "secondary": dataloader_secondary}
        else:
            return dataloader_principal

    def data_collator(
        self, batch: Dict[str, Union[torch.Tensor, List[str]]], deterministic: bool
    ) -> Dict[str, Union[torch.Tensor, List[int], List[str]]]:
        """
        Custom data collator function to dynamically pad the data.

        Args:
            batch (Dict[str, Union[torch.Tensor, List[str]]]) : Dict from the dataset with the keys 'audio' and 'phonemes':
                - 'audio' (torch.Tensor of dimension (sample_rate * duration))
                - 'phonemes' (str)
            deterministic (bool): If True, always select the same part of the signal.

        Returns:
            Dict[str, Union[torch.Tensor, List[int], List[str]]]: A dictionary containing collated data with keys:
            - 'audio' (torch.Tensor of dimension (batch_size, sample_rate * duration)),
            - 'phonemes_ids' (torch.Tensor of dimension (batch_size, multiples of 128),
            - 'phonemes_str' (List[str]),
        """

        audios = [sample["audio"]["array"] for sample in batch]
        phonemes = [sample["phonemized_text"] for sample in batch]
        
        audio_processed = self.feature_extractor(
            raw_speech=audios,
            padding="longest",
            return_tensors="pt",
            sampling_rate=self.sample_rate,  # Do not resample anything, simple verification
            pad_to_multiple_of=128,
            # Because NVIDIA GeForce RTX 2080 Ti have 128 Concurrent Kernel Execution
        )

        labels_processed = self.tokenizer(
            text=phonemes,
            padding="longest",
            return_tensors="pt",
            pad_to_multiple_of=128,
            return_attention_mask=True,
            # Because NVIDIA GeForce RTX 2080 Ti have 128 Concurrent Kernel Execution
        )

        labels = labels_processed.input_ids.masked_fill(labels_processed.attention_mask.ne(1), -100)
        audio_processed = audio_processed.input_values
        
        # Apply data augmentation
        if deterministic is False:    
            with torch.no_grad():
                audio_processed, _ = self.data_augmentation(audio_processed)

        return {
            "audio": audio_processed,
            "phonemes_ids": labels,
            "phonemes_str": phonemes,
        }

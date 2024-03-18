from typing import Dict, Any

from datasets import load_dataset, Audio
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer


class STPLightningDataModule(LightningDataModule):
    DATASET_NAME = "Cnam-LMSSC/vibravox"
    LANGUAGE = "fr"

    def __init__(
        self,
        sample_rate: int = 16000,
        subset_name: str = "bwe_in-ear_rigid_earpiece_microphone",
        streaming: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        feature_extractor: Wav2Vec2FeatureExtractor = None,
        tokenizer: Wav2Vec2CTCTokenizer = None,
    ):
        """
        LightningDataModule for Speech-to-Phoneme (STP).

        Args:
            sample_rate (int, optional): Sample rate of the audio files. Defaults to 16000.
            subset_name (str, optional): Name of the configuration. Defaults to "bwe_in-ear_rigid_earpiece_microphone".
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
            feature_extractor (Wav2Vec2FeatureExtractor): Feature extractor. Defaults to None.
            tokenizer (Wav2Vec2CTCTokenizer): Tokenizer. Defaults to None.
        """

        super().__init__()

        self.sample_rate = sample_rate
        self.subset_name = subset_name
        self.streaming = streaming
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def setup(self, stage=None):
        """
        Set up the datasets.

        Args:
            stage (str): Pipeline stage among ['fit', 'validate', 'test', 'predict']. Defaults to None.

        Notes:
            This function runs on every accelerator in distributed mode.
            That is why it is necessary to define attributes here rather than in __init__.
        """

        datasets = load_dataset(
            self.DATASET_NAME, self.subset_name, streaming=self.streaming
        )

        datasets = datasets.select_columns(["audio", "phonemes"])

        # Resample the audio to the right sample rate
        datasets = datasets.cast_column(
            "audio", Audio(sampling_rate=self.sample_rate, mono=False)
        )

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["validation"]
        self.test_dataset = datasets["test"]

    def train_dataloader(self):
        """
        Train dataloader.

        Returns:
            DataLoader
        """

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        """
        Validation dataloader.

        Returns:
            DataLoader
        """

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        """
        Test dataloader.

        Returns:
            DataLoader
        """

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def data_collator(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Custom data collator function to dynamically pad the data.

        Args:
            batch: Dict from the dataset with the keys "audio" and "phonemes"
        Returns:
            dict
        """

        audios = [sample["audio"]["array"] for sample in batch]
        phonemes = [sample["phonemes"] for sample in batch]

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

        labels = labels_processed.input_ids.masked_fill(
            labels_processed.attention_mask.ne(1), -100
        )

        return {
            "audio": audio_processed.input_values,
            "phonemes_ids": labels,
            "phonemes_str": phonemes,
        }

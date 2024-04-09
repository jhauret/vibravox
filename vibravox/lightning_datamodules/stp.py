from typing import Dict, Any, Tuple

from datasets import load_dataset, Audio
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer


class STPLightningDataModule(LightningDataModule):
    DATASET_NAME = "Cnam-LMSSC/vibravox"

    def __init__(
        self,
        sample_rate: int = 16000,
        sensor: Tuple[str] = ("bwe_in-ear_rigid_earpiece_microphone",),
        subset: Tuple[str] = ("speech_clean",),
        streaming: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        feature_extractor: Wav2Vec2FeatureExtractor = None,
        tokenizer: Wav2Vec2CTCTokenizer = None,
    ):
        """
        LightningDataModule for Speech-to-Phoneme (STP).

        Args:
            sample_rate (int, optional): Sample rate at which the dataset is output. Defaults to 16000.
            sensor (Tuple[str], optional): Sensor. Defaults to ("bwe_in-ear_rigid_earpiece_microphone",).
            subset (Tuple[str], optional): Subset. Defaults to ("speech_clean",).
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
            feature_extractor (Wav2Vec2FeatureExtractor): Feature extractor. Defaults to None.
            tokenizer (Wav2Vec2CTCTokenizer): Tokenizer. Defaults to None.
        """

        super().__init__()

        if len(sensor) != 1:
            raise NotImplementedError("Only one sensor is supported for now.")
        if len(subset) != 1:
            raise NotImplementedError("Only one subset is supported for now.")

        self.sample_rate = sample_rate
        self.sensor = sensor
        self.subset = subset
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

        dataset_dict = load_dataset(
            self.DATASET_NAME, self.subset[0], streaming=self.streaming
        )

        dataset_dict = dataset_dict.rename_column(f"audio.{self.sensor[0]}", "audio")

        dataset_dict = dataset_dict.select_columns(["audio", "phonemized_text"])

        # Resample the audio to the right sample rate
        dataset_dict = dataset_dict.cast_column(
            "audio", Audio(sampling_rate=self.sample_rate, mono=False)
        )

        self.train_dataset = dataset_dict["train"]
        self.val_dataset = dataset_dict["validation"]
        self.test_dataset = dataset_dict["test"]

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

        labels = labels_processed.input_ids.masked_fill(
            labels_processed.attention_mask.ne(1), -100
        )

        return {
            "audio": audio_processed.input_values,
            "phonemes_ids": labels,
            "phonemes_str": phonemes,
        }

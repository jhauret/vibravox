from typing import Dict, Any, Tuple

from datasets import load_dataset, Audio
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer


class STPLightningDataModule(LightningDataModule):

    def __init__(
        self,
        sample_rate: int = 16000,
        dataset_name: str = "Cnam-LMSSC/vibravox",
        subset: str = "speech_clean",
        sensor: str = "airborne.mouth_headworn.reference_microphone",
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
            dataset_name (str, optional): Dataset name. Defaults to "Cnam-LMSSC/vibravox"
            subset (str, optional): Subset. Defaults to "speech_clean"
            sensor (str, optional): Sensor. Defaults to "bwe_in-ear_rigid_earpiece_microphone"
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
            feature_extractor (Wav2Vec2FeatureExtractor): Feature extractor. Defaults to None.
            tokenizer (Wav2Vec2CTCTokenizer): Tokenizer. Defaults to None.
        """

        super().__init__()

        self.sample_rate = sample_rate
        assert dataset_name in ["Cnam-LMSSC/vibravox", "Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp"], \
            "dataset_name must be 'Cnam-LMSSC/vibravox' or 'Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp'"
        self.dataset_name = dataset_name
        self.subset = subset
        self.sensor = sensor
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
            self.dataset_name, self.subset, streaming=self.streaming
        )

        dataset_dict = dataset_dict.rename_column(f"audio.{self.sensor}", "audio")

        dataset_dict = dataset_dict.select_columns(["audio", "phonemized_text"])

        # Resample the audio to the right sample rate
        dataset_dict = dataset_dict.cast_column(
            "audio", Audio(sampling_rate=self.sample_rate, mono=False)
        )

        if stage == "fit" or stage is None:
            self.train_dataset = dataset_dict["train"]
            self.val_dataset = dataset_dict["validation"]
        elif stage == "test" or stage is None:
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

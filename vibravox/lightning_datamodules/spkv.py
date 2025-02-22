import torch
from typing import Dict, List, Any, Union, Tuple

import pickle

from pathlib import Path

from lightning import LightningDataModule
from datasets import Audio, load_dataset, interleave_datasets
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from lightning.pytorch.utilities import CombinedLoader


class SPKVLightningDataModule(LightningDataModule):

    def __init__(
        self,
        sample_rate: int = 16000,
        dataset_name: str = "Cnam-LMSSC/vibravox",
        subset: str = "speech_clean",
        sensor_a: str = "headset_microphone",
        sensor_b: str = "headset_microphone",
        pairs: str = "mixed_gender",
        streaming: bool = False,
        batch_size: int = 1,
        num_workers: int = 4,
        **kwargs,
    ):
        """
        LightningDataModule for end-to-end Speaker Verification (SPKV)

        Args:
            sample_rate (int, optional): Sample rate at which the dataset is output. Defaults to 16000.
            dataset_name (str, optional): Dataset name.
                Must be one of "Cnam-LMSSC/vibravox" or "Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp".
                Defaults to "Cnam-LMSSC/vibravox".
            subset (str, optional): Subset. Defaults to ("speech_clean").
            sensor_a (str, optional): Sensor. Defaults to ("headset_microphone").
            sensor_b (str, optional): Sensor. Defaults to ("headset_microphone").
            pairs (str, optional): Pairs configuration. Must be one of "mixed_gender" or "same_gender".
                Default to "mixed_gender"
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 1 for testing since ECAPA2 pretrained model only supports this Batchsize
            num_workers (int, optional): Number of workers. Defaults to 4.
        """
        super().__init__()

        self.sample_rate = sample_rate
        assert dataset_name in [
            "Cnam-LMSSC/vibravox",
            "Cnam-LMSSC/vibravox-test",
            "Cnam-LMSSC/vibravox_mixed_for_spkv" "Cnam-LMSSC/vibravox_enhanced_by_EBEN",
        ], f"dataset_name {dataset_name} not supported."
        self.dataset_name = dataset_name
        if self.dataset_name != "Cnam-LMSSC/vibravox-test":
            assert subset in [
                "speech_clean",
                "speech_noisy_mixed",
            ], "speech_noisy is not supported for SPKV (too few samples to have relevant results)"
        self.subset = subset
        self.sensorA = sensor_a
        self.sensorB = sensor_b
        assert pairs in ["mixed_gender", "same_gender"], "pairs must be 'mixed_gender' or 'same_gender'"
        self.pairs = pairs
        self.streaming = streaming
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Set up the datasets.

        Args:
            stage (str): Pipeline stage among ['fit', 'validate', 'test', 'predict']. Defaults to None.

        Notes:
            This function runs on every accelerator in distributed mode.
            That is why it is necessary to define attributes here rather than in __init__.
        """

        dataset_dict = load_dataset(self.dataset_name, self.subset, streaming=self.streaming)

        if stage == "fit" or stage is None:
            # Generating dataset for training and validation
            train_dataset = dataset_dict.get("train", None)
            val_dataset = dataset_dict.get("validation", None)

            if train_dataset is None or val_dataset is None:
                pass

            if self.sensorA == self.sensorB and train_dataset is not None and val_dataset is not None:
                # When self.sensorA and self.sensorB are the same, only generate the dataset using one column

                # Only keep the relevant columns for this task :
                train_dataset = train_dataset.select_columns(
                    [f"audio.{self.sensorA}", "speaker_id", "sentence_id", "gender"]
                )
                val_dataset = val_dataset.select_columns(
                    [f"audio.{self.sensorA}", "speaker_id", "sentence_id", "gender"]
                )

                # Resample the audios to the right sample rate
                train_dataset = train_dataset.cast_column(
                    f"audio.{self.sensorA}", Audio(sampling_rate=self.sample_rate, mono=False)
                )
                val_dataset = val_dataset.cast_column(
                    f"audio.{self.sensorA}", Audio(sampling_rate=self.sample_rate, mono=False)
                )

                # Tag a column with the sensor name :
                train_dataset = train_dataset.add_column("sensor", [self.sensorA] * len(train_dataset))
                val_dataset = val_dataset.add_column("sensor", [self.sensorA] * len(val_dataset))

                # Renaming columns to match the format expected by the model :
                train_dataset = train_dataset.rename_column(f"audio.{self.sensorA}", "audio")
                val_dataset = val_dataset.rename_column(f"audio.{self.sensorA}", "audio")

            elif self.sensorA != self.sensorB and train_dataset is not None and val_dataset is not None:
                # When self.sensorA and self.sensorB are different, generate the dataset by interleaving both sensors
                # in the same dataset for train/validation, which results in a two times larger dataset for training,
                # but allows to learn embeddings for both sensors

                # Only keep the relevant columns for this task :
                train_dataset_a = train_dataset.select_columns(
                    [f"audio.{self.sensorA}", "speaker_id", "sentence_id", "gender"]
                )
                train_dataset_b = train_dataset.select_columns(
                    [f"audio.{self.sensorB}", "speaker_id", "sentence_id", "gender"]
                )

                val_dataset_a = val_dataset.select_columns(
                    [f"audio.{self.sensorA}", "speaker_id", "sentence_id", "gender"]
                )
                val_dataset_b = val_dataset.select_columns(
                    [f"audio.{self.sensorB}", "speaker_id", "sentence_id", "gender"]
                )

                # Resample the audios to the right sample rate
                train_dataset_a = train_dataset_a.cast_column(
                    f"audio.{self.sensorA}", Audio(sampling_rate=self.sample_rate, mono=False)
                )

                train_dataset_b = train_dataset_b.cast_column(
                    f"audio.{self.sensorB}", Audio(sampling_rate=self.sample_rate, mono=False)
                )

                val_dataset_a = val_dataset_a.cast_column(
                    f"audio.{self.sensorA}", Audio(sampling_rate=self.sample_rate, mono=False)
                )

                val_dataset_b = val_dataset_b.cast_column(
                    f"audio.{self.sensorB}", Audio(sampling_rate=self.sample_rate, mono=False)
                )

                # Renaming columns to match the format expected by the model :
                train_dataset_a = train_dataset_a.rename_column(f"audio.{self.sensorA}", "audio")
                train_dataset_b = train_dataset_b.rename_column(f"audio.{self.sensorB}", "audio")

                val_dataset_a = val_dataset_a.rename_column(f"audio.{self.sensorA}", "audio")
                val_dataset_b = val_dataset_b.rename_column(f"audio.{self.sensorB}", "audio")

                # Tag a column with the sensor name :
                train_dataset_a = train_dataset_a.add_column("sensor", [self.sensorA] * len(train_dataset_a))
                train_dataset_b = train_dataset_b.add_column("sensor", [self.sensorB] * len(train_dataset_b))

                val_dataset_a = val_dataset_a.add_column("sensor", [self.sensorA] * len(val_dataset_a))
                val_dataset_b = val_dataset_b.add_column("sensor", [self.sensorB] * len(val_dataset_b))

                # Interleave datasets of two sensors for training/validation :
                train_dataset = interleave_datasets(
                    datasets=[train_dataset_a, train_dataset_b],
                    probabilities=[0.5, 0.5],
                    stopping_strategy="all_exhausted",
                )

                val_dataset = interleave_datasets(
                    datasets=[val_dataset_a, val_dataset_b], probabilities=[0.5, 0.5], stopping_strategy="all_exhausted"
                )

            if train_dataset is not None and val_dataset is not None:
                # Setting format to torch
                self.train_dataset = train_dataset.with_format("torch")
                self.val_dataset = val_dataset.with_format("torch")

        if stage == "test":
            # Generating dataset for testing for Speaker Verification (only for the test set) : pairs are needed
            # Pairs are only formed for the test set for Speaker Verification. Training and validation in end-to-end
            # fashion do not need to be paired.
            # The strategy to form pairs is the same as in the paper from Brydinskyi et al.,
            # "Comparison of Modern Deep Learning Models for Speaker Verification." Applied Sciences 14.4 (2024): 1329.

            test_dataset = dataset_dict["test"]

            if self.streaming:
                raise AttributeError("Streaming is not supported for testing SPKVLightningDataModule")
                # because IterableDataset does not support the sort method nor the select method

            # Order by speaker_id for easier pairing of audios :
            test_dataset = test_dataset.sort("speaker_id")

            # Only keep the relevant columns for this task :
            test_dataset_a = test_dataset.select_columns(
                [f"audio.{self.sensorA}", "speaker_id", "sentence_id", "gender"]
            )
            test_dataset_b = test_dataset.select_columns(
                [f"audio.{self.sensorB}", "speaker_id", "sentence_id", "gender"]
            )

            # Tag a column with the sensor name :
            test_dataset_a = test_dataset_a.add_column("sensor", [self.sensorA] * len(test_dataset_a))
            test_dataset_b = test_dataset_b.add_column("sensor", [self.sensorB] * len(test_dataset_b))

            # Resample the audios to the right sample rate
            test_dataset_a = test_dataset_a.cast_column(
                f"audio.{self.sensorA}", Audio(sampling_rate=self.sample_rate, mono=False)
            )

            test_dataset_b = test_dataset_b.cast_column(
                f"audio.{self.sensorB}", Audio(sampling_rate=self.sample_rate, mono=False)
            )

            # Load the pickle file located in pkfile_path generated by scripts/gen_pairs_for_spkv.py :
            if self.dataset_name == "Cnam-LMSSC/vibravox-test":
                pickle_file = f"configs/lightning_datamodule/spkv_pairs/vibravox-test/{self.subset}/{self.pairs}.pkl"
            else:
                pickle_file = f"configs/lightning_datamodule/spkv_pairs/{self.pairs}.pkl"

            pickle_path = Path(__file__).parent.parent.parent / pickle_file

            if not pickle_path.exists():
                raise ValueError(
                    f"File {pickle_path} does not exist, please generate one for your dataset using scripts/gen_pairs_for_spkv.py"
                )

            with open(pickle_path, "rb") as file:
                pairs = pickle.load(file)

            test_dataset_a = test_dataset_a.select([pair[0] for pair in pairs])
            test_dataset_b = test_dataset_b.select([pair[1] for pair in pairs])

            # Renaming columns to match the format expected by the model :

            test_dataset_a = test_dataset_a.rename_column(f"audio.{self.sensorA}", "audio")
            test_dataset_b = test_dataset_b.rename_column(f"audio.{self.sensorB}", "audio")

            # Setting format to torch
            self.test_dataset_a = test_dataset_a.with_format("torch")
            self.test_dataset_b = test_dataset_b.with_format("torch")

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
            Collection of two DataLoaders corresponding to dataset_A and dataset_B
        """

        dataloader_a = DataLoader(
            self.test_dataset_a,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            shuffle=False,
        )  # We do not shuffle the dataset to keep the order of the pairs

        dataloader_b = DataLoader(
            self.test_dataset_b,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            shuffle=False,
        )

        return CombinedLoader(iterables={"sensor_a": dataloader_a, "sensor_b": dataloader_b}, mode="min_size")

    def data_collator(
        self, batch: List[Dict[str, Union[torch.Tensor, str, int]]]
    ) -> Dict[str, Union[torch.Tensor, List[str], List[int]]]:
        """
        Collates data samples into a single batch

        Note : since SPKV uses a CombinedLoader, this data_collator is used for both DataLoader

        Parameters:
            batch (List[Dict[str, Union[torch.Tensor, str, int]]]): List of dictionaries with keys 'audio', 'speaker_id', 'sentence_id', 'gender', and 'sensor'
            Each Dict has the following keys :
                - 'audio' (torch.Tensor of dimension (sample_rate * duration)),
                - 'speaker_id' (str),
                - 'sentence_id' (torch.Tensor of int),
                - 'gender' (str),
                - 'sensor' (str)

        Returns:
            Dict : A dictionary containing collated data with keys:
            - 'audio' (torch.Tensor of dimension (batch_size, 1, sample_rate * duration)),
            - 'speaker_id' (List[str]),
            - 'sentence_id' (List[torch.Tensor of int]),
            - 'gender' (List[str]),
            - 'sensor' (List[str])
        """

        audio_batch = [sample["audio"]["array"] for sample in batch]
        audio_batch = pad_sequence(audio_batch, batch_first=True, padding_value=0.0).unsqueeze(1)
        speaker_id_batch = [sample["speaker_id"] for sample in batch]
        sentence_id_batch = [int(sample["sentence_id"]) for sample in batch]
        gender_batch = [sample["gender"] for sample in batch]
        sensor_batch = [sample["sensor"] for sample in batch]

        return {
            "audio": audio_batch,
            "speaker_id": speaker_id_batch,
            "sentence_id": sentence_id_batch,
            "gender": gender_batch,
            "sensor": sensor_batch,
        }

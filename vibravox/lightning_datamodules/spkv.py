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

    DATASET_NAME = "Cnam-LMSSC/vibravox"

    def __init__(
        self,
        pklfile_path: str,
        sample_rate: int = 16000,
        sensor_a: str = "airborne.mouth_headworn.reference_microphone",
        sensor_b: str = "airborne.mouth_headworn.reference_microphone",
        subset: str = "speech_clean",
        streaming: bool = False,
        batch_size: int = 1,
        num_workers: int = 4,
    ):
        """
        LightningDataModule for end-to-end Speaker Verification (SPKV)

        Args:
            sample_rate (int, optional): Sample rate at which the dataset is output. Defaults to 16000.
            sensor_a (str, optional): Sensor. Defaults to ("airborne.mouth_headworn.reference_microphone").
            sensor_b (str, optional): Sensor. Defaults to ("airborne.mouth_headworn.reference_microphone").
            subset (str, optional): Subset. Defaults to ("speech_clean").
            pklfile_path (str, optional): Pickle file path. Defaults to "configs/lightning_datamodule/spkv_pairs/vibravox/speech_clean/pairs.pkl".
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 1 for testing since ECAPA2 pretrained model only supports this Batchsize
            num_workers (int, optional): Number of workers. Defaults to 4.
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.sensorA = sensor_a
        self.sensorB = sensor_b
        self.subset = subset
        self.pklfile_path = pklfile_path

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

        dataset_dict = load_dataset(
            self.DATASET_NAME, self.subset, streaming=self.streaming
        )

        train_dataset_dict = dataset_dict["train"]
        val_dataset_dict = dataset_dict["validation"]
        test_dataset_dict = dataset_dict["test"]

        if stage == "fit" or stage is None:
            print("Generating dataset for training and validation ...")

            if self.sensorA == self.sensorB:
                print("Sensor A and Sensor B are the same")

                # Only keep the relevant columns for this task :
                train_dataset_dict = train_dataset_dict.select_columns(
                    [f"audio.{self.sensorA}", "speaker_id", "sentence_id", "gender"])
                val_dataset_dict = val_dataset_dict.select_columns(
                    [f"audio.{self.sensorA}", "speaker_id", "sentence_id", "gender"])

                # Resample the audios to the right sample rate
                train_dataset_dict = train_dataset_dict.cast_column(
                f"audio.{self.sensorA}", Audio(sampling_rate=self.sample_rate, mono=False)
                )
                val_dataset_dict = val_dataset_dict.cast_column(
                f"audio.{self.sensorA}", Audio(sampling_rate=self.sample_rate, mono=False)
                )

                # Tag a column with the sensor name :

                train_dataset_dict = train_dataset_dict.add_column("sensor", [self.sensorA] * len(train_dataset_dict))
                val_dataset_dict = val_dataset_dict.add_column("sensor", [self.sensorA] * len(val_dataset_dict))

                # Renaming columns to match the format expected by the model :
                train_dataset_dict = train_dataset_dict.rename_column(f"audio.{self.sensorA}", "audio")
                val_dataset_dict = val_dataset_dict.rename_column(f"audio.{self.sensorA}", "audio")

            else:
                print("Sensor A and Sensor B are not the same")

                # Only keep the relevant columns for this task :
                train_dataset_dict_a = train_dataset_dict.select_columns(
                    [f"audio.{self.sensorA}", "speaker_id", "sentence_id", "gender"])
                train_dataset_dict_b = train_dataset_dict.select_columns(
                    [f"audio.{self.sensorB}", "speaker_id", "sentence_id", "gender"])

                val_dataset_dict_a = val_dataset_dict.select_columns(
                    [f"audio.{self.sensorA}", "speaker_id", "sentence_id", "gender"])
                val_dataset_dict_b = val_dataset_dict.select_columns(
                    [f"audio.{self.sensorB}", "speaker_id", "sentence_id", "gender"])

                # Resample the audios to the right sample rate
                train_dataset_dict_a = train_dataset_dict_a.cast_column(
                    f"audio.{self.sensorA}", Audio(sampling_rate=self.sample_rate, mono=False)
                )

                train_dataset_dict_b = train_dataset_dict_b.cast_column(
                    f"audio.{self.sensorB}", Audio(sampling_rate=self.sample_rate, mono=False)
                )

                val_dataset_dict_a = val_dataset_dict_a.cast_column(
                    f"audio.{self.sensorA}", Audio(sampling_rate=self.sample_rate, mono=False)
                )

                val_dataset_dict_b = val_dataset_dict_b.cast_column(
                    f"audio.{self.sensorB}", Audio(sampling_rate=self.sample_rate, mono=False)
                )

                # Renaming columns to match the format expected by the model :
                train_dataset_dict_a = train_dataset_dict_a.rename_column(f"audio.{self.sensorA}", "audio")
                train_dataset_dict_b = train_dataset_dict_b.rename_column(f"audio.{self.sensorB}", "audio")

                val_dataset_dict_a = val_dataset_dict_a.rename_column(f"audio.{self.sensorA}", "audio")
                val_dataset_dict_b = val_dataset_dict_b.rename_column(f"audio.{self.sensorB}", "audio")

                # Tag a column with the sensor name :

                train_dataset_dict_a = train_dataset_dict_a.add_column("sensor", [self.sensorA] * len(train_dataset_dict_a))
                train_dataset_dict_b = train_dataset_dict_b.add_column("sensor", [self.sensorB] * len(train_dataset_dict_b))

                val_dataset_dict_a = val_dataset_dict_a.add_column("sensor", [self.sensorA] * len(val_dataset_dict_a))
                val_dataset_dict_b = val_dataset_dict_b.add_column("sensor", [self.sensorB] * len(val_dataset_dict_b))


                # Interleave datasets of two sensors for training/validation :
                train_dataset_dict = interleave_datasets(datasets=[train_dataset_dict_a, train_dataset_dict_b],
                                                             probabilities=[0.5,0.5],
                                                             stopping_strategy='all_exhausted'
                                                             )

                val_dataset_dict = interleave_datasets(datasets=[val_dataset_dict_a, val_dataset_dict_b],
                                                             probabilities=[0.5,0.5],
                                                             stopping_strategy='all_exhausted'
                                                             )
            # Setting format to torch :

            train_dataset_dict = train_dataset_dict.with_format("torch")
            val_dataset_dict = val_dataset_dict.with_format("torch")

            print("Size of the train dataset : ", len(train_dataset_dict))
            print("Size of the val dataset : ", len(val_dataset_dict))

            self.train_dataset = train_dataset_dict
            self.val_dataset = val_dataset_dict


        if stage == "test":
            if self.streaming:
                raise AttributeError("Streaming is not supported for testing SPKVLightningDataModule")
                # because IterableDataset does not support the sort method nor the select method

            # Order by speaker_id for easier pairing of audios :
            test_dataset_dict = test_dataset_dict.sort("speaker_id")

            # Pairs are only formed for the test set for Speaker Verification. Training and validation in end-to-end
            # fashion do not need to be paired.

            # Only keep the relevant columns for this task :
            dataset_dict_a = test_dataset_dict.select_columns(
                [f"audio.{self.sensorA}", "speaker_id", "sentence_id", "gender"])
            dataset_dict_b = test_dataset_dict.select_columns(
                [f"audio.{self.sensorB}", "speaker_id", "sentence_id", "gender"])

            # Tag a column with the sensor name :

            dataset_dict_a = dataset_dict_a.add_column("sensor", [self.sensorA] * len(dataset_dict_a))
            dataset_dict_b = dataset_dict_b.add_column("sensor", [self.sensorB] * len(dataset_dict_b))

            # Resample the audios to the right sample rate
            dataset_dict_a = dataset_dict_a.cast_column(
                f"audio.{self.sensorA}", Audio(sampling_rate=self.sample_rate, mono=False)
            )

            dataset_dict_b = dataset_dict_b.cast_column(
                f"audio.{self.sensorB}", Audio(sampling_rate=self.sample_rate, mono=False)
            )

            # Load the pickle file located in pkfile_path generated by scripts/gen_pairs_for_spkv.py :

            with open(Path(__file__).parent.parent.parent / self.pklfile_path, 'rb') as file:
                pairs = pickle.load(file)

            dataset_dict_a = dataset_dict_a.select([pair[0] for pair in pairs])
            dataset_dict_b = dataset_dict_b.select([pair[1] for pair in pairs])

            # Renaming columns to match the format expected by the model :

            dataset_dict_a = dataset_dict_a.rename_column(f"audio.{self.sensorA}", "audio")
            dataset_dict_b = dataset_dict_b.rename_column(f"audio.{self.sensorB}", "audio")

            print("Size of the test dataset : ", len(dataset_dict_a) , len(dataset_dict_b))


            # Setting format to torch :

            dataset_dict_a = dataset_dict_a.with_format("torch")
            dataset_dict_b = dataset_dict_b.with_format("torch")

            self.test_dataset_a = dataset_dict_a
            self.test_dataset_b = dataset_dict_b



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

        dataloader_a = DataLoader(self.test_dataset_a,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  collate_fn=self.data_collator,
                                  shuffle=False)  # We do not shuffle the dataset to keep the order of the pairs

        dataloader_b = DataLoader(self.test_dataset_b,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  collate_fn=self.data_collator,
                                  shuffle=False)

        return CombinedLoader(iterables={"sensor_a": dataloader_a, "sensor_b": dataloader_b}, mode='min_size')

    def data_collator(self, batch: List[Dict[str, Any]]) -> Dict[str, Union[ torch.Tensor, List[str], List[int],List[str]]]:
        """
            Collates data samples into a single batch

            Note : since SPKV uses a CombinedLoader, this data_collator is used for both DataLoader

            Parameters:
                batch (Dict[str, Any]): List of dictionaries with keys "audio", "speaker_id", "sentence_id", "gender", and "sensor"

            Returns:
                Dict : A dictionary containing collated data with keys:
                "audio" (torch.Tensor of dimension (batch_size, 1, sample_rate * duration)),
                "speaker_id" (List[str]),
                "sentence_id" (List[int]),
                "gender" (List[str]),
                "sensor" (List[str])
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

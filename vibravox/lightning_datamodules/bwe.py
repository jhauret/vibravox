import torch
import re
from typing import Dict, List

from lightning import LightningDataModule
from datasets import Audio, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import vibravox.torch_modules.dsp.data_augmentation
from vibravox.torch_modules.dsp.data_augmentation import WaveformDataAugmentation
from vibravox.utils import set_audio_duration


class BWELightningDataModule(LightningDataModule):

    def __init__(
        self,
        sample_rate: int = 16000,
        dataset_name: str = "Cnam-LMSSC/vibravox",
        subset: str = "speech_clean",
        sensor: str = "airborne.mouth_headworn.reference_microphone",
        collate_strategy: str = "constant_length-2500-ms",
        data_augmentation: torch.nn.Module = None,
        streaming: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        LightningDataModule for Bandwidth Extension (BWE)

        Args:
            sample_rate (int, optional): Sample rate at which the dataset is output. Defaults to 16000.
            dataset_name (str, optional): Dataset name. Defaults to "Cnam-LMSSC/vibravox"
            subset (str, optional): Subset. Defaults to "speech_clean"
            sensor (str, optional): Sensor. Defaults to "bwe_in-ear_rigid_earpiece_microphone"
            collate_strategy (str, optional): What strategy to use to collate the data. One of:
                - "pad": Pad the audio signals to the length of the longest signal in the batch.
                - "constant_length-XXX-ms": Cut or pad the audio signals to XXXms.
            Defaults to "constant_length-3000-ms".
            data_augmentation (nn.Module, optional): Data augmentation module. Defaults to None.
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
        """
        super().__init__()

        self.sample_rate = sample_rate
        assert dataset_name in ["Cnam-LMSSC/vibravox", "Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp"], \
            "dataset_name must be 'Cnam-LMSSC/vibravox' or 'Cnam-LMSSC/vibravox_enhanced_by_EBEN_tmp'"
        self.dataset_name = dataset_name
        self.subset = subset
        self.sensor = sensor

        assert collate_strategy == "pad" or re.match(r"constant_length-\d+-ms", collate_strategy), \
            "collate_strategy must be 'pad' or match the pattern 'constant_length-XXX-ms'"

        self.collate_strategy = collate_strategy

        if data_augmentation is None:
            data_augmentation = WaveformDataAugmentation(sample_rate)
        assert isinstance(data_augmentation, WaveformDataAugmentation), "data_augmentation must be a WaveformDataAugmentation"

        self.data_augmentation = data_augmentation

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
            self.dataset_name, self.subset, streaming=self.streaming
        )

        dataset_dict = dataset_dict.rename_column(f"audio.airborne.mouth_headworn.reference_microphone", "audio_airborne")
        dataset_dict = dataset_dict.rename_column(f"audio.{self.sensor}", "audio_body_conducted")

        dataset_dict = dataset_dict.select_columns(["audio_airborne", "audio_body_conducted"])

        # Resample the audio to the right sample rate
        dataset_dict = dataset_dict.cast_column(
            "audio_airborne", Audio(sampling_rate=self.sample_rate, mono=False)
        )
        dataset_dict = dataset_dict.cast_column(
            "audio_body_conducted", Audio(sampling_rate=self.sample_rate, mono=False)
        )
        dataset_dict = dataset_dict.with_format("torch")

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
            collate_fn=lambda batch: self.data_collator(batch, deterministic=False, collate_strategy=self.collate_strategy)
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
            collate_fn=lambda batch: self.data_collator(batch, deterministic=True, collate_strategy=self.collate_strategy),
        )

    def test_dataloader(self):
        """
        Test dataloader.

        Returns:
            DataLoader
        """

        return DataLoader(
            self.test_dataset,
            batch_size=4,
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(batch, deterministic=True, collate_strategy="pad"),
        )

    def data_collator(self, batch: List[Dict[str, Audio]], deterministic: bool, collate_strategy: str) -> Dict[str, torch.Tensor]:
        """
        Custom data collator function to dynamically pad the data.

        Args:
            batch (List[Dict[str, Audio]]): Dict from the dataset with the keys "audio_body_conducted" and "audio_airborne"
            deterministic (bool): If True, always select the same part of the signal
            collate_strategy (str, optional): What strategy to use to collate the data. One of:
                - "pad": Pad the audio signals to the length of the longest signal in the batch.
                - "constant_length-XXX-ms": Cut or pad the audio signals to XXXms.
        Returns:
            Dict[str, torch.Tensor]
        """

        body_conducted_batch = [item["audio_body_conducted"]["array"] for item in batch]
        air_conducted_batch = [item["audio_airborne"]["array"] for item in batch]

        if collate_strategy == "pad":

            body_conducted_padded_batch = pad_sequence(
                body_conducted_batch, batch_first=True, padding_value=0.0
            ).unsqueeze(1)
            air_conducted_padded_batch = pad_sequence(
                air_conducted_batch, batch_first=True, padding_value=0.0
            ).unsqueeze(1)

        else:
            ms_length = int(self.collate_strategy.split('-')[1])
            samples = int(self.sample_rate * ms_length / 1000)

            body_conducted_padded_batch = []
            air_conducted_padded_batch = []
            for body_conducted, air_conducted in zip(body_conducted_batch, air_conducted_batch):
                body_conducted_padded, air_conducted_padded = set_audio_duration(
                    audio=body_conducted,
                    desired_samples=samples,
                    audio_bis=air_conducted,
                    deterministic=deterministic,
                )
                body_conducted_padded_batch.append(body_conducted_padded.unsqueeze(0))
                air_conducted_padded_batch.append(air_conducted_padded.unsqueeze(0))
            body_conducted_padded_batch = torch.stack(body_conducted_padded_batch, dim=0)
            air_conducted_padded_batch = torch.stack(air_conducted_padded_batch, dim=0)

        # Apply data augmentation
        if deterministic is False:
            with torch.no_grad():
                body_conducted_padded_batch, air_conducted_padded_batch = self.data_augmentation(body_conducted_padded_batch, air_conducted_padded_batch)

        return {
            "audio_body_conducted": body_conducted_padded_batch,
            "audio_airborne":  air_conducted_padded_batch,
        }



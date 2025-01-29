import torch
import re
from typing import Dict, List, Union

from lightning import LightningDataModule
from datasets import Audio, load_dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from vibravox.torch_modules.dsp.data_augmentation import WaveformDataAugmentation
from vibravox.utils import set_audio_duration


class BWELightningDataModule(LightningDataModule):

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
        collate_strategy: str = "constant_length-2500-ms",
        data_augmentation: torch.nn.Module = None,
        streaming: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs,
    ):
        """
        LightningDataModule for Bandwidth Extension (BWE)

        Args:
            sample_rate (int, optional): Sample rate at which the dataset is output. Defaults to 16000.
            dataset_name_principal (str, optional): Principal dataset name that is going to be used for train/validation and testing.
                Default to "Cnam-LMSSC/vibravox".
            dataset_name_secondary (str, optional): Secondary dataset name that is going to be used for validation and testing.
                Default to None
            subset (str, optional): Subset. Defaults to "speech_clean"
            sensor (str, optional): Sensor. Defaults to "headset_microphone"
            collate_strategy (str, optional): What strategy to use to collate the data. One of:
                - "pad": Pad the audio signals to the length of the longest signal in the batch.
                - "constant_length-XXX-ms": Cut or pad the audio signals to XXXms.
            Defaults to "constant_length-3000-ms".
            data_augmentation (nn.Module, optional): Data augmentation module. Defaults to None.
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
            pin_memory (bool, optional): If True, the data loader will copy Tensors into CUDA pinned memory before returning them. Defaults to True.
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

        assert collate_strategy == "pad" or re.match(
            r"constant_length-\d+-ms", collate_strategy
        ), "collate_strategy must be 'pad' or match the pattern 'constant_length-XXX-ms'"

        self.collate_strategy = collate_strategy

        if data_augmentation is None:
            data_augmentation = WaveformDataAugmentation(sample_rate)
        assert isinstance(
            data_augmentation, WaveformDataAugmentation
        ), "data_augmentation must be a WaveformDataAugmentation"

        self.data_augmentation = data_augmentation

        self.streaming = streaming
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

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

        dataset_dict = dataset_dict.rename_column(f"audio.headset_microphone", "audio_airborne")
        dataset_dict = dataset_dict.rename_column(f"audio.{self.sensor}", "audio_body_conducted")
        dataset_dict = dataset_dict.select_columns(["audio_airborne", "audio_body_conducted"])

        # Resample the audio to the right sample rate
        dataset_dict = dataset_dict.cast_column("audio_airborne", Audio(sampling_rate=self.sample_rate, mono=False))
        dataset_dict = dataset_dict.cast_column(
            "audio_body_conducted", Audio(sampling_rate=self.sample_rate, mono=False)
        )
        dataset_dict = dataset_dict.with_format("torch")

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
                batch, deterministic=False, collate_strategy=self.collate_strategy
            ),
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, Dict[str, DataLoader]]:
        """
        Validation dataloader(s).

        Returns:
            Union[DataLoader, Dict[str, DataLoader]]
        """

        dataloader_principal = DataLoader(
            self.val_dataset_principal,
            batch_size=min(1, self.batch_size // 4),
            num_workers=self.num_workers,
            collate_fn=lambda batch: self.data_collator(
                batch, deterministic=True, collate_strategy=self.collate_strategy
            ),
            pin_memory=self.pin_memory,
        )
        if self.dataset_name_secondary is not None:
            dataloader_secondary = DataLoader(
                self.val_dataset_secondary,
                batch_size=min(1, self.batch_size // 4),
                num_workers=self.num_workers,
                collate_fn=lambda batch: self.data_collator(
                    batch, deterministic=True, collate_strategy=self.collate_strategy
                ),
                pin_memory=self.pin_memory,
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
                batch, deterministic=True, collate_strategy=self.collate_strategy
            ),
            pin_memory=self.pin_memory,
        )

        if self.dataset_name_secondary is not None:
            dataloader_secondary = DataLoader(
                self.test_dataset_secondary,
                batch_size=1,
                num_workers=self.num_workers,
                collate_fn=lambda batch: self.data_collator(
                    batch, deterministic=True, collate_strategy=self.collate_strategy
                ),
                pin_memory=self.pin_memory,
            )

            return {"principal": dataloader_principal, "secondary": dataloader_secondary}
        else:
            return dataloader_principal

    def data_collator(
        self, batch: List[Dict[str, Audio]], deterministic: bool, collate_strategy: str
    ) -> Dict[str, torch.Tensor]:
        """
        Custom data collator function to dynamically pad the data.

        Args:
            - batch (List[Dict[str, Audio]]): Dict from the dataset with the keys 'audio_body_conducted' and 'audio_airborne':
                - 'audio_body_conducted': (torch.Tensor of dimension (sample_rate * duration)),
                - 'audio_airborne': (torch.Tensor of dimension (sample_rate * duration))
            - deterministic (bool): If True, always select the same part of the signal.
            - collate_strategy (str, optional): What strategy to use to collate the data. One of:
                - 'pad': Pad the audio signals to the length of the longest signal in the batch,
                - 'constant_length-XXX-ms': Cut or pad the audio signals to XXXms

        Returns:
            Dict[str, torch.Tensor] A dictionary containing collated data with keys:
                - 'audio_body_conducted': (torch.Tensor of dimension (batch_size, 1, sample_rate * duration)),
                - 'audio_airborne': (torch.Tensor of dimension (batch_size, 1, sample_rate * duration))
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
            ms_length = int(self.collate_strategy.split("-")[1])
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
                body_conducted_padded_batch, air_conducted_padded_batch = self.data_augmentation(
                    body_conducted_padded_batch, air_conducted_padded_batch
                )

        return {
            "audio_body_conducted": body_conducted_padded_batch,
            "audio_airborne": air_conducted_padded_batch,
        }

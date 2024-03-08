import torch
from lightning import LightningDataModule
from datasets import Audio, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class BWELightningDataModule(LightningDataModule):

    DATASET_NAME = "Cnam-LMSSC/vibravox"

    def __init__(
        self,
        config_name="bwe_in-ear_rigid_earpiece_microphone",
        streaming=False,
        sample_rate=16000,
        batch_size=32,
        num_workers=4,
    ):
        """
        LightningDataModule for Bandwidth Extension (BWE)

        Args:
            config_name (str, optional): Name of the configuration. Defaults to "BWE_In-ear_Comply_Foam_microphone".
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            sample_rate (int, optional): Sample rate of the audio files. Defaults to 16000.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
        """
        super().__init__()

        self.config_name = config_name
        self.sample_rate = sample_rate
        self.streaming = streaming
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        datasets = load_dataset(
            self.DATASET_NAME, self.config_name, streaming=self.streaming
        )

        datasets = datasets.select_columns(["audio"])
        datasets = datasets.cast_column(
            "audio", Audio(sampling_rate=self.sample_rate, mono=False)
        )
        datasets = datasets.with_format("torch")

        def foo(sample):
            return {
                "body_conducted": self.set_audio_duration(
                    audio=sample["audio"]["array"][0, :],
                    desired_duration=3,
                    deterministic=False,
                ),
                "air_conducted": self.set_audio_duration(
                    audio=sample["audio"]["array"][1, :],
                    desired_duration=3,
                    deterministic=False,
                ),
            }

        datasets = datasets.map(foo)

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["validation"]
        self.test_dataset = datasets["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def data_collator(self, batch):
        body_conducted_batch = [item["body_conducted"] for item in batch]
        air_conducted_batch = [item["air_conducted"] for item in batch]

        body_conducted_padded_batch = pad_sequence(
            body_conducted_batch, batch_first=True, padding_value=0.0
        )
        air_conducted_padded_batch = pad_sequence(
            air_conducted_batch, batch_first=True, padding_value=0.0
        )

        return [
            body_conducted_padded_batch.unsqueeze(1),
            air_conducted_padded_batch.unsqueeze(1),
        ]

    def set_audio_duration(
        self, audio: torch.Tensor, desired_duration: float, deterministic: bool = False
    ):
        """
        Make the audio signal have the desired duration.

        Args:
            audio (torch.Tensor): input signal. Shape: (time_len,)
            desired_duration (float): duration of selected signal in seconds
            deterministic (bool): if True, always select the same part of the signal

        """
        original_time_len = audio.numel()
        desired_time_len = int(desired_duration * self.sample_rate)

        # If the signal is longer than the desired duration, select a random part of the signal
        if original_time_len >= desired_time_len:
            if deterministic:
                offset_time_len = original_time_len - desired_time_len // 2
            else:
                offset_time_len = torch.randint(
                    low=0, high=original_time_len - desired_time_len + 1, size=(1,)
                )
            audio = audio[offset_time_len : offset_time_len + desired_time_len]

        # If the signal is shorter than the desired duration, pad the signal with zeros
        else:
            num_zeros_left = desired_time_len - original_time_len // 2
            audio = torch.nn.functional.pad(
                audio,
                pad=(
                    num_zeros_left,
                    desired_time_len - original_time_len - num_zeros_left,
                ),
                mode="constant",
                value=0,
            )

        return audio

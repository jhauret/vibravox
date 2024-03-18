from io import BytesIO

from lightning import LightningDataModule
from datasets import Audio, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from vibravox.utils import set_audio_duration


class BWELightningDataModule(LightningDataModule):

    DATASET_NAME = "Cnam-LMSSC/vibravox"

    def __init__(
        self,
        sample_rate: int = 16000,
        subset_name: str = "bwe_in-ear_rigid_earpiece_microphone",
        streaming: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """
        LightningDataModule for Bandwidth Extension (BWE)

        Args:
            sample_rate (int, optional): Sample rate of the audio files. Defaults to 16000.
            subset_name (str, optional): Name of the configuration. Defaults to "BWE_In-ear_Comply_Foam_microphone".
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.subset_name = subset_name
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

        datasets = load_dataset(
            self.DATASET_NAME, self.subset_name, streaming=self.streaming
        )

        datasets = datasets.select_columns(["audio"])

        # Resample the audio to the right sample rate
        datasets = datasets.cast_column(
            "audio", Audio(sampling_rate=self.sample_rate, mono=False)
        )
        datasets = datasets.with_format("torch")

        # The hash of self is not deterministic, so we need to externalize the compute
        # So the map function is serializable. This makes possible to re-use the huggingface cache
        desired_time_len = int(3 * self.sample_rate)

        def process_sample(sample):
            """
            Extract and process the sample to have the desired duration.

            Args:
                sample (dict): sample from the dataset
            """
            waveform = sample["audio"]["array"]
            waveform = set_audio_duration(audio=waveform, desired_time_len=desired_time_len, deterministic=False)
            return {"body_conducted": waveform[0, :], "air_conducted": waveform[1, :]}

        datasets = datasets.map(process_sample)

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

    @staticmethod
    def data_collator(batch):

        # Note: not really necessary as set_audio_duration is applied in the setup

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

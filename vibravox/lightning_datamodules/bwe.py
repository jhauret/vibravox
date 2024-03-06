import pytorch_lightning as pl
from datasets import Audio, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


class BWELightningDataModule(pl.LightningDataModule):

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
        datasets = load_dataset(self.DATASET_NAME, self.config_name, streaming=self.streaming)

        datasets = datasets.select_columns(["audio"])
        datasets = datasets.cast_column("audio", Audio(sampling_rate=self.sample_rate, mono=False))
        datasets = datasets.with_format("torch")
        datasets = datasets.map(
            lambda sample: {
                "body_conducted": sample["audio"]["array"][0, :],
                "air_conducted": sample["audio"]["array"][1, :],
            }
        )

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

        body_conducted_padded_batch = pad_sequence(body_conducted_batch, batch_first=True, padding_value=0.0)
        air_conducted_padded_batch = pad_sequence(air_conducted_batch, batch_first=True, padding_value=0.0)

        return [body_conducted_padded_batch, air_conducted_padded_batch]

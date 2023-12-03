from datasets import load_dataset, Audio
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BWELightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "Cnam-LMSSC/vibravox",
        config_name="BWE_In-ear_Comply_Foam_microphone",
        streaming=False,
        sample_rate=16000,
        batch_size=32,
        num_workers=4,
    ):
        """
        LightningDataModule for Bandwidth Extension (BWE)

        Args:
            dataset_name (str, optional): Name of the dataset. Defaults to "Cnam-LMSSC/vibravox".
            config_name (str, optional): Name of the configuration. Defaults to "BWE_In-ear_Comply_Foam_microphone".
            streaming (bool, optional): If True, the audio files are dynamically downloaded. Defaults to False.
            sample_rate (int, optional): Sample rate of the audio files. Defaults to 16000.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers. Defaults to 4.
        """
        super().__init__()

        self.dataset_name = dataset_name
        self.config_name = config_name
        self.sample_rate = sample_rate
        self.streaming = streaming
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        datasets = load_dataset(
            self.dataset_name, self.config_name, streaming=self.streaming
        )
        datasets = datasets.remove_columns(
            [
                "audio_length",
                "gender",
                "num_channels",
                "type",
                "sensor_id",
                "split",
                "sentence_id",
                "transcription",
                "speaker_id",
            ]
        )
        datasets = datasets.cast_column("audio", Audio(sampling_rate=self.sample_rate))
        datasets = datasets.with_format("torch")

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["validation"]
        self.test_dataset = datasets["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.data_collator,
        )

    def data_collator(self, batch):
        padded_batch = pad_sequence(batch["audio"], batch_first=True, padding_value=0.0)
        return padded_batch

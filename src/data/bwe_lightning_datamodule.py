from datasets import load_dataset, Audio
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BWELightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "Cnam-LMSSC/vibravox",
        config_name="BWE_In-ear_Comply_Foam_microphone",
        sample_rate=16000,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.config_name = config_name
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):

        datasets = load_dataset(self.dataset_name, self.config_name)
        datasets = datasets.remove_columns(["audio_length", "gender", "num_channels", "type", "sensor_id", "split", "sentence_id", "transcription", "speaker_id"])
        datasets = datasets.cast_column("audio", Audio(sampling_rate=16000))
        datasets = datasets.with_format("torch")

        self.train_dataset = datasets["train"]
        self.val_dataset = datasets["validation"]
        self.test_dataset = datasets["test"]


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=pad_sequence)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=pad_sequence)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=pad_sequence)

    def data_collator(self, batch):
        padded_batch = pad_sequence(batch["audio"], batch_first=True, padding_value=0.0)
        return padded_batch

